// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hdmitsuba/motion.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/interval.h>
#include <pxr/base/gf/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformOp.h>
#include <pxr/usd/usdGeom/xformable.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(HdMitsubaMotionTokens, HDMITSUBA_MOTION_TOKENS);

namespace {

constexpr int kDefaultMotionSamples = 32;

// Tolerances of the keyframe refinement in `SampleTransform`.
constexpr double kMaxRotationErrorDegrees = 1.0;
// Keeps segments well below 180 degrees, where the shortest arc is ambiguous.
constexpr double kMaxSegmentRotationDegrees = 90.0;
// Relative to the scale, resp. the extent of the translation over a segment.
constexpr double kMaxRelativeError = 1e-2;

// Evaluates a world-transform matrix at a frame-relative shutter offset.
using MatrixEvalFn = std::function<GfMatrix4d(float)>;

// A transform sample at a given shutter offset, paired with its TRS
// decomposition (matching the keyframes of Mitsuba's AnimatedTransform) so
// recursive subdivision does not re-decompose segment endpoints.
struct EvaluatedSample {
  float time;
  GfMatrix4d matrix;
  GfVec3d scale;
  GfQuatd rotation;
  GfVec3d translation;

  EvaluatedSample(const MatrixEvalFn& eval_matrix, float t)
      : time(t),
        matrix(eval_matrix(t)),
        scale(matrix.GetRow3(0).GetLength(), matrix.GetRow3(1).GetLength(),
              matrix.GetRow3(2).GetLength()),
        rotation(matrix.RemoveScaleShear().ExtractRotationQuat()),
        translation(matrix.ExtractTranslation()) {}
};

// Returns whether the angle between the rotations `a` and `b` is at most
// `degrees`. The angle between unit quaternions is 2 * acos(|a . b|).
bool RotationAngleAtMost(const GfQuatd& a, const GfQuatd& b, double degrees) {
  return std::abs(GfDot(a, b)) >= std::cos(0.5 * GfDegreesToRadians(degrees));
}

// Returns whether interpolating between `a` and `b` at `alpha` (linearly for
// scale and translation, along the shortest arc for rotation, like Mitsuba's
// AnimatedTransform) approximates `actual`.
bool IsInterpolatedAccurately(const EvaluatedSample& a,
                              const EvaluatedSample& b, double alpha,
                              const EvaluatedSample& actual) {
  const double extent = (b.translation - a.translation).GetLength() +
                        (actual.translation - a.translation).GetLength();
  return (GfLerp(alpha, a.translation, b.translation) - actual.translation)
                 .GetLength() <= kMaxRelativeError * extent + 1e-6 &&
         (GfLerp(alpha, a.scale, b.scale) - actual.scale).GetLength() <=
             kMaxRelativeError * actual.scale.GetLength() + 1e-6 &&
         RotationAngleAtMost(GfSlerp(alpha, a.rotation, b.rotation),
                             actual.rotation, kMaxRotationErrorDegrees);
}

// Returns whether the segment (s0, s1) must be subdivided: it spans a large
// rotation, or TRS interpolation deviates from the actual trajectory at its
// midpoint `s_mid` or at its quarter point (which catches trajectories that
// happen to agree at the midpoint, e.g. symmetric wobbles).
bool NeedsRefinement(const MatrixEvalFn& eval_matrix, const EvaluatedSample& s0,
                     const EvaluatedSample& s1, const EvaluatedSample& s_mid) {
  return !RotationAngleAtMost(s0.rotation, s1.rotation,
                              kMaxSegmentRotationDegrees) ||
         !IsInterpolatedAccurately(s0, s1, 0.5, s_mid) ||
         !IsInterpolatedAccurately(
             s0, s1, 0.25,
             EvaluatedSample(eval_matrix, GfLerp(0.25, s0.time, s1.time)));
}

// Returns the time of the `i`-th of `n + 1` uniformly spaced grid points over
// `interval`.
float GridTime(const GfVec2f& interval, int i, int n) {
  return static_cast<float>(
      GfLerp(static_cast<double>(i) / n, interval[0], interval[1]));
}

// Returns whether `times` (ending with the interval end) lie on the uniform
// grid with `times.size()` segments over `interval`.
bool IsUniformlySpaced(const GfVec2f& interval,
                       const std::vector<float>& times) {
  const int n = static_cast<int>(times.size());
  for (int i = 0; i < n; ++i) {
    if (std::abs(times[i] - GridTime(interval, i + 1, n)) >
        1e-4 * (interval[1] - interval[0])) {
      return false;
    }
  }
  return true;
}

// Appends at most `max_segments` samples of `eval_matrix` in (s0.time, s1.time]
// to `samples`. Because USD interpolates xformOps before composing matrices
// while Mitsuba's AnimatedTransform decomposes keyframe matrices into (scale,
// quat, translation) and slerps along the shortest arc, segments with large
// (>= 90 deg) rotations or curved/orbiting trajectories are adaptively
// subdivided.
void AppendRefinedSamples(
    const MatrixEvalFn& eval_matrix, const EvaluatedSample& s0,
    const EvaluatedSample& s1, int max_segments,
    std::vector<std::pair<float, GfMatrix4d>>& samples) {
  if (max_segments > 1) {
    const EvaluatedSample s_mid(eval_matrix, GfLerp(0.5, s0.time, s1.time));
    if (NeedsRefinement(eval_matrix, s0, s1, s_mid)) {
      const int left = max_segments / 2;
      AppendRefinedSamples(eval_matrix, s0, s_mid, left, samples);
      AppendRefinedSamples(eval_matrix, s_mid, s1, max_segments - left,
                           samples);
      return;
    }
  }
  samples.emplace_back(s1.time, s1.matrix);
}

// Evaluates `eval_matrix` at the `n + 1` uniformly spaced grid points over
// `interval`.
std::vector<EvaluatedSample> EvaluateGrid(const MatrixEvalFn& eval_matrix,
                                          const GfVec2f& interval, int n) {
  std::vector<EvaluatedSample> grid;
  grid.reserve(n + 1);
  for (int i = 0; i <= n; ++i) {
    grid.emplace_back(eval_matrix, GridTime(interval, i, n));
  }
  return grid;
}

// Samples `eval_matrix` on a uniform grid over `interval` with at most
// `max_samples` points, as required by Mitsuba's `instance` plugin. The grid
// is seeded with `seed_segments` segments and doubled in lockstep until all
// segments interpolate accurately, falling back to the finest grid with
// `max_samples` points if the budget is exhausted before that. Grid times are
// always snapped to the exact uniform positions: Mitsuba verifies uniformity
// with a 1e-5 relative tolerance.
std::vector<EvaluatedSample> SampleUniformGrid(const MatrixEvalFn& eval_matrix,
                                               const GfVec2f& interval,
                                               int seed_segments,
                                               int max_samples) {
  std::vector<EvaluatedSample> grid =
      EvaluateGrid(eval_matrix, interval, seed_segments);
  while (grid.size() < static_cast<size_t>(max_samples)) {
    const int m = static_cast<int>(grid.size()) - 1;
    std::vector<EvaluatedSample> next_grid;
    next_grid.reserve(2 * m + 1);
    bool accurate = true;
    for (int i = 0; i < m; ++i) {
      EvaluatedSample mid(eval_matrix, GridTime(interval, 2 * i + 1, 2 * m));
      accurate =
          accurate && !NeedsRefinement(eval_matrix, grid[i], grid[i + 1], mid);
      next_grid.push_back(grid[i]);
      next_grid.push_back(std::move(mid));
    }
    if (accurate) break;
    if (2 * m + 1 > max_samples) {
      return EvaluateGrid(eval_matrix, interval, max_samples - 1);
    }
    next_grid.push_back(grid.back());
    grid = std::move(next_grid);
  }
  return grid;
}

// Samples `eval_matrix` over `interval` (including any `times` strictly inside
// `interval`) with at most `max_samples >= 2` samples, refining segments as
// needed and collapsing to a static sample if all resulting matrices match.
// When `uniform` is true, or there are too many `times` to refine adaptively,
// samples a uniform grid that `times` only seed if they lie on it; otherwise
// the grid starts from the interval endpoints and refinement discovers the
// kinks at the authored samples.
MotionSamples<GfMatrix4d> SampleTransformOverInterval(
    const MatrixEvalFn& eval_matrix, const GfVec2f& interval,
    std::vector<float> times, bool uniform, int max_samples) {
  std::sort(times.begin(), times.end());
  times.erase(
      std::remove_if(
          times.begin(), times.end(),
          [&](float t) { return t <= interval[0] || t >= interval[1]; }),
      times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  times.push_back(interval[1]);

  MotionSamples<GfMatrix4d> result;
  const int init_segments = static_cast<int>(times.size());
  if (!uniform && init_segments < max_samples) {
    EvaluatedSample prev(eval_matrix, interval[0]);
    result.samples.emplace_back(prev.time, prev.matrix);
    const int total_segments = max_samples - 1;
    for (int i = 0; i < init_segments; ++i) {
      EvaluatedSample next(eval_matrix, times[i]);
      // Splits the segment budget as evenly as possible over `times`.
      const int budget = (total_segments * (i + 1)) / init_segments -
                         (total_segments * i) / init_segments;
      AppendRefinedSamples(eval_matrix, prev, next, budget, result.samples);
      prev = std::move(next);
    }
  } else {
    const bool seed_with_times =
        init_segments < max_samples && IsUniformlySpaced(interval, times);
    for (auto& s : SampleUniformGrid(eval_matrix, interval,
                                     seed_with_times ? init_segments : 1,
                                     max_samples)) {
      result.samples.emplace_back(s.time, std::move(s.matrix));
    }
  }
  return std::move(result).Collapsed();
}

// Returns `primvars:mitsuba:motion_samples` on `prim` or its closest ancestor,
// or `kDefaultMotionSamples` if unauthored.
int GetUsdMotionSamples(const UsdPrim& prim) {
  int samples = kDefaultMotionSamples;
  if (UsdGeomPrimvar pv = UsdGeomPrimvarsAPI(prim).FindPrimvarWithInheritance(
          HdMitsubaMotionTokens->motion_samples);
      pv && !pv.Get(&samples)) {
    TF_WARN("Ignoring %s on %s: expected an int.",
            pv.GetName().GetText(), prim.GetPath().GetText());
  }
  return samples;
}

// Returns the frame-relative shutter offsets in `interval` around `time` of all
// time-varying ancestor xformOps of `prim` and any `extra_attrs`, or `nullopt`
// if none of them might vary over the shutter.
std::optional<std::vector<float>> GetSampleOffsetsInInterval(
    const UsdPrim& prim, UsdTimeCode time, const GfVec2f& interval,
    std::initializer_list<UsdAttribute> extra_attrs) {
  if (interval[1] <= interval[0] || !time.IsNumeric()) return std::nullopt;
  std::vector<UsdAttribute> attrs;
  for (UsdPrim p = prim; p; p = p.GetParent()) {
    if (UsdGeomXformable xformable{p}) {
      bool resets = false;
      for (const UsdGeomXformOp& op : xformable.GetOrderedXformOps(&resets)) {
        if (op.MightBeTimeVarying()) attrs.push_back(op.GetAttr());
      }
      if (resets) break;
    }
  }
  for (const UsdAttribute& attr : extra_attrs) {
    if (attr.ValueMightBeTimeVarying()) attrs.push_back(attr);
  }
  if (attrs.empty()) return std::nullopt;
  const double t0 = time.GetValue();
  std::vector<double> usd_samples;
  UsdAttribute::GetUnionedTimeSamplesInInterval(
      attrs, GfInterval(t0 + interval[0], t0 + interval[1], false, false),
      &usd_samples);
  std::vector<float> offsets;
  offsets.reserve(usd_samples.size());
  for (double s : usd_samples) offsets.push_back(static_cast<float>(s - t0));
  return offsets;
}

}  // namespace

GfVec2f GetMotionInterval(HdSceneDelegate* scene_delegate) {
  return GetParam<GfVec2f>(
      GetPrimDataSource(scene_delegate, SdfPath::AbsoluteRootPath()),
      HdMitsubaMotionTokens->motion_interval, GfVec2f(0.0f));
}

int GetMotionSamples(const HdContainerDataSourceHandle& prim_source) {
  return GetParam<int>(
      HdPrimvarsSchema::GetFromParent(prim_source).GetContainer(),
      HdDataSourceLocator(HdMitsubaMotionTokens->motion_samples,
                          HdPrimvarSchemaTokens->primvarValue),
      kDefaultMotionSamples);
}

MotionSamples<GfMatrix4d> SampleTransform(
    const HdContainerDataSourceHandle& prim_source, const GfVec2f& interval,
    bool uniform) {
  const HdMatrixDataSourceHandle matrix =
      HdXformSchema::GetFromParent(prim_source).GetMatrix();
  if (!matrix) {
    return MotionSamples<GfMatrix4d>::Static(GfMatrix4d(1.0));
  }
  const int max_samples = GetMotionSamples(prim_source);
  std::vector<float> times;
  if (max_samples <= 1 || interval[1] <= interval[0] ||
      !matrix->GetContributingSampleTimesForInterval(interval[0], interval[1],
                                                     &times)) {
    return MotionSamples<GfMatrix4d>::Static(matrix->GetTypedValue(0.0f));
  }
  return SampleTransformOverInterval(
      [&](float t) { return matrix->GetTypedValue(t); }, interval,
      std::move(times), uniform, max_samples);
}

std::vector<MotionSamples<GfMatrix4d>> SampleInstanceTransforms(
    const std::function<VtMatrix4dArray(float)>& eval_matrices,
    const GfVec2f& interval, const std::optional<std::vector<float>>& times,
    int max_samples) {
  VtMatrix4dArray t0_xforms = eval_matrices(0.0f);
  std::vector<MotionSamples<GfMatrix4d>> result;
  result.reserve(t0_xforms.size());
  if (!times.has_value() || max_samples <= 1) {
    for (const GfMatrix4d& m : t0_xforms) {
      result.push_back(MotionSamples<GfMatrix4d>::Static(m));
    }
    return result;
  }
  // Each instance is refined independently, which costs up to
  // O(instances * max_samples) TRS decompositions and caches one full matrix
  // array per distinct sample time. This is fine for moderate instance counts;
  // very large animated point clouds would benefit from deciding the grid once
  // for the whole array instead.
  absl::flat_hash_map<float, VtMatrix4dArray> cache = {
      {0.0f, std::move(t0_xforms)}};
  for (size_t i = 0, n = cache[0.0f].size(); i < n; ++i) {
    result.push_back(SampleTransformOverInterval(
        [&](float t) {
          auto [it, inserted] = cache.try_emplace(t);
          if (inserted) it->second = eval_matrices(t);
          return it->second.AsConst()[i];
        },
        interval, *times, /*uniform=*/true, max_samples));
  }
  return result;
}

MotionSamples<GfMatrix4d> SampleTransform(const UsdPrim& prim, UsdTimeCode time,
                                          const GfVec2f& interval,
                                          bool uniform) {
  UsdGeomImageable imageable(prim);
  if (!imageable) {
    return MotionSamples<GfMatrix4d>::Static(GfMatrix4d(1.0));
  }
  const int max_samples = GetUsdMotionSamples(prim);
  std::optional<std::vector<float>> offsets =
      GetSampleOffsetsInInterval(prim, time, interval, {});
  if (max_samples <= 1 || !offsets) {
    return MotionSamples<GfMatrix4d>::Static(
        imageable.ComputeLocalToWorldTransform(time));
  }
  return SampleTransformOverInterval(
      [&](float offset) {
        return imageable.ComputeLocalToWorldTransform(time.GetValue() + offset);
      },
      interval, std::move(*offsets), uniform, max_samples);
}

std::vector<MotionSamples<GfMatrix4d>> SamplePointInstancerTransforms(
    const UsdPrim& prim, UsdTimeCode time, const GfVec2f& interval) {
  UsdGeomPointInstancer instancer(prim);
  // Like in the Hydra path, where the prototype's own xform is baked into the
  // prototype geometry at frame time, only the instancer's attributes and
  // ancestors contribute to the motion: the prototype xforms are evaluated at
  // `time` and re-applied to the (masked) instances below.
  VtIntArray proto_indices;
  instancer.GetProtoIndicesAttr().Get(&proto_indices, time);
  SdfPathVector proto_paths;
  instancer.GetPrototypesRel().GetTargets(&proto_paths);
  std::vector<GfMatrix4d> proto_xforms(proto_paths.size(), GfMatrix4d(1.0));
  for (size_t i = 0; i < proto_paths.size(); ++i) {
    bool resets = false;
    if (UsdGeomXformable proto{prim.GetStage()->GetPrimAtPath(proto_paths[i])}) {
      proto.GetLocalTransformation(&proto_xforms[i], &resets, time);
    }
  }
  // The prototype xform of each unmasked instance, in the order of
  // `ComputeInstanceTransformsAtTime`.
  const std::vector<bool> mask = instancer.ComputeMaskAtTime(time);
  std::vector<GfMatrix4d> instance_proto_xforms;
  for (size_t i = 0; i < proto_indices.size(); ++i) {
    if (!mask.empty() && !mask[i]) continue;
    const int proto = proto_indices[i];
    instance_proto_xforms.push_back(
        proto >= 0 && proto < static_cast<int>(proto_xforms.size())
            ? proto_xforms[proto]
            : GfMatrix4d(1.0));
  }
  return SampleInstanceTransforms(
      [&](float offset) {
        const UsdTimeCode t =
            offset == 0.0f ? time : UsdTimeCode(time.GetValue() + offset);
        VtMatrix4dArray xforms;
        instancer.ComputeInstanceTransformsAtTime(
            &xforms, t, time, UsdGeomPointInstancer::ExcludeProtoXform);
        const GfMatrix4d to_world = instancer.ComputeLocalToWorldTransform(t);
        for (size_t j = 0;
             j < std::min(xforms.size(), instance_proto_xforms.size()); ++j) {
          xforms[j] = instance_proto_xforms[j] * xforms[j] * to_world;
        }
        return xforms;
      },
      interval,
      GetSampleOffsetsInInterval(
          prim, time, interval,
          {instancer.GetPositionsAttr(), instancer.GetOrientationsAttr(),
           instancer.GetOrientationsfAttr(), instancer.GetScalesAttr()}),
      GetUsdMotionSamples(prim));
}

PXR_NAMESPACE_CLOSE_SCOPE
