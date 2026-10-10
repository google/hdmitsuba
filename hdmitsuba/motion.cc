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
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/xformOp.h>
#include <pxr/usd/usdGeom/xformable.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(HdMitsubaMotionTokens, HDMITSUBA_MOTION_TOKENS);

namespace {

// Tolerances of the keyframe refinement in `SampleTransform`.
constexpr int kMaxRefinementDepth = 5;
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

// Appends samples of `eval_matrix` in (s0.time, s1.time] to `samples`.
// Because USD interpolates xformOps before composing matrices while Mitsuba's
// AnimatedTransform decomposes keyframe matrices into (scale, quat,
// translation) and slerps along the shortest arc, segments with large (>= 90
// deg) rotations or curved/orbiting trajectories are adaptively subdivided.
void AppendRefinedSamples(
    const MatrixEvalFn& eval_matrix, const EvaluatedSample& s0,
    const EvaluatedSample& s1, int depth,
    std::vector<std::pair<float, GfMatrix4d>>& samples) {
  if (depth < kMaxRefinementDepth) {
    const auto eval_at = [&](double alpha) {
      return EvaluatedSample(eval_matrix, GfLerp(alpha, s0.time, s1.time));
    };
    const EvaluatedSample s_mid = eval_at(0.5);
    if (!RotationAngleAtMost(s0.rotation, s1.rotation,
                             kMaxSegmentRotationDegrees) ||
        !IsInterpolatedAccurately(s0, s1, 0.5, s_mid) ||
        !IsInterpolatedAccurately(s0, s1, 0.25, eval_at(0.25))) {
      AppendRefinedSamples(eval_matrix, s0, s_mid, depth + 1, samples);
      AppendRefinedSamples(eval_matrix, s_mid, s1, depth + 1, samples);
      return;
    }
  }
  samples.emplace_back(s1.time, s1.matrix);
}

}  // namespace

MotionSamples<GfMatrix4d> SampleTransformOverInterval(
    const MatrixEvalFn& eval_matrix, const GfVec2f& interval,
    std::vector<float> times, bool uniform) {
  std::sort(times.begin(), times.end());
  times.push_back(interval[1]);

  MotionSamples<GfMatrix4d> result;
  EvaluatedSample prev(eval_matrix, interval[0]);
  result.samples.emplace_back(prev.time, prev.matrix);
  for (float t : times) {
    if (t > prev.time && t <= interval[1]) {
      EvaluatedSample next(eval_matrix, t);
      AppendRefinedSamples(eval_matrix, prev, next, 0, result.samples);
      prev = std::move(next);
    }
  }
  if (std::all_of(result.samples.begin() + 1, result.samples.end(),
                  [&](const auto& s) { return s.second == result.First(); })) {
    return MotionSamples<GfMatrix4d>::Static(result.First());
  }
  if (uniform && result.samples.size() > 2) {
    const float start = interval[0];
    const float end = interval[1];
    const float span = end - start;
    constexpr int kMaxUniformSegments = 1 << kMaxRefinementDepth;
    int num_segments = kMaxUniformSegments;
    for (int n = static_cast<int>(result.samples.size()) - 1;
         n <= kMaxUniformSegments; ++n) {
      if (std::all_of(result.samples.begin(), result.samples.end(),
                      [&](const auto& s) {
                        const double pos = (s.first - start) / span * n;
                        return std::abs(pos - std::round(pos)) <= 1e-4;
                      })) {
        num_segments = n;
        break;
      }
    }
    const float step = span / num_segments;
    if (static_cast<int>(result.samples.size()) != num_segments + 1 ||
        !std::all_of(result.samples.begin(), result.samples.end(),
                     [&](const auto& s) {
                       const size_t i = &s - result.samples.data();
                       return std::abs(s.first - (start + i * step)) <=
                              1e-5f * span;
                     })) {
      std::vector<std::pair<float, GfMatrix4d>> uniform_samples;
      uniform_samples.reserve(num_segments + 1);
      for (int i = 0; i <= num_segments; ++i) {
        const float t = static_cast<float>(
            GfLerp(static_cast<double>(i) / num_segments, start, end));
        uniform_samples.emplace_back(t, eval_matrix(t));
      }
      result.samples = std::move(uniform_samples);
    }
  }
  return result;
}

MotionSamples<GfMatrix4d> SampleTransform(
    const HdContainerDataSourceHandle& prim_source, const GfVec2f& interval,
    bool uniform) {
  const HdMatrixDataSourceHandle matrix =
      HdXformSchema::GetFromParent(prim_source).GetMatrix();
  if (!matrix) {
    return MotionSamples<GfMatrix4d>::Static(GfMatrix4d(1.0));
  }
  std::vector<float> times;
  if (interval[1] <= interval[0] ||
      !matrix->GetContributingSampleTimesForInterval(interval[0], interval[1],
                                                     &times)) {
    return MotionSamples<GfMatrix4d>::Static(matrix->GetTypedValue(0.0f));
  }
  return SampleTransformOverInterval(
      [&](float t) { return matrix->GetTypedValue(t); }, interval,
      std::move(times), uniform);
}

MotionSamples<GfMatrix4d> SampleTransform(const UsdPrim& prim, UsdTimeCode time,
                                          const GfVec2f& interval,
                                          bool uniform) {
  UsdGeomImageable imageable(prim);
  if (!imageable) {
    return MotionSamples<GfMatrix4d>::Static(GfMatrix4d(1.0));
  }
  std::vector<UsdGeomXformOp> ops;
  if (interval[1] > interval[0] && time.IsNumeric()) {
    for (UsdPrim p = prim; p; p = p.GetParent()) {
      if (UsdGeomXformable xformable{p}) {
        bool resets_xform_stack = false;
        for (const UsdGeomXformOp& op :
             xformable.GetOrderedXformOps(&resets_xform_stack)) {
          if (op.MightBeTimeVarying()) ops.push_back(op);
        }
        if (resets_xform_stack) break;
      }
    }
  }
  if (ops.empty()) {
    return MotionSamples<GfMatrix4d>::Static(
        imageable.ComputeLocalToWorldTransform(time));
  }
  const double t0 = time.GetValue();
  std::vector<double> usd_samples;
  UsdGeomXformable::GetTimeSamplesInInterval(
      ops, GfInterval(t0 + interval[0], t0 + interval[1], false, false),
      &usd_samples);
  std::vector<float> offsets;
  offsets.reserve(usd_samples.size());
  for (double s : usd_samples) offsets.push_back(static_cast<float>(s - t0));
  return SampleTransformOverInterval(
      [&](float offset) {
        return imageable.ComputeLocalToWorldTransform(t0 + offset);
      },
      interval, std::move(offsets), uniform);
}

std::vector<MotionSamples<GfMatrix4d>> SamplePointInstancerTransforms(
    const UsdPrim& prim, UsdTimeCode time, const GfVec2f& interval) {
  UsdGeomPointInstancer instancer(prim);
  if (!instancer) return {};

  auto eval_at = [&](UsdTimeCode t) {
    VtMatrix4dArray xforms;
    instancer.ComputeInstanceTransformsAtTime(&xforms, t, t);
    const GfMatrix4d instancer_xform =
        instancer.ComputeLocalToWorldTransform(t);
    for (GfMatrix4d& m : xforms) {
      m *= instancer_xform;
    }
    return xforms;
  };

  std::vector<UsdGeomXformOp> ops;
  std::vector<UsdAttribute> inst_attrs;
  bool has_velocities = false;
  if (interval[1] > interval[0] && time.IsNumeric()) {
    for (UsdPrim p = prim; p; p = p.GetParent()) {
      if (UsdGeomXformable xformable{p}) {
        bool resets_xform_stack = false;
        for (const UsdGeomXformOp& op :
             xformable.GetOrderedXformOps(&resets_xform_stack)) {
          if (op.MightBeTimeVarying()) ops.push_back(op);
        }
        if (resets_xform_stack) break;
      }
    }
    for (const UsdAttribute& attr :
         {instancer.GetPositionsAttr(), instancer.GetOrientationsAttr(),
          instancer.GetOrientationsfAttr(), instancer.GetScalesAttr(),
          instancer.GetVelocitiesAttr(),
          instancer.GetAngularVelocitiesAttr()}) {
      if (attr.ValueMightBeTimeVarying()) inst_attrs.push_back(attr);
    }
    has_velocities = instancer.GetVelocitiesAttr().HasAuthoredValue() ||
                     instancer.GetAngularVelocitiesAttr().HasAuthoredValue();
  }

  const VtMatrix4dArray t0_xforms = eval_at(time);
  std::vector<MotionSamples<GfMatrix4d>> result;
  result.reserve(t0_xforms.size());
  if (ops.empty() && inst_attrs.empty() && !has_velocities) {
    for (const GfMatrix4d& m : t0_xforms) {
      result.push_back(MotionSamples<GfMatrix4d>::Static(m));
    }
    return result;
  }

  const double t0 = time.GetValue();
  const GfInterval usd_interval(t0 + interval[0], t0 + interval[1], false,
                                false);
  std::vector<double> usd_samples;
  if (!ops.empty()) {
    UsdGeomXformable::GetTimeSamplesInInterval(ops, usd_interval, &usd_samples);
  }
  for (const UsdAttribute& attr : inst_attrs) {
    std::vector<double> attr_samples;
    if (attr.GetTimeSamplesInInterval(usd_interval, &attr_samples)) {
      usd_samples.insert(usd_samples.end(), attr_samples.begin(),
                         attr_samples.end());
    }
  }
  std::vector<float> offsets;
  offsets.reserve(usd_samples.size());
  for (double s : usd_samples) offsets.push_back(static_cast<float>(s - t0));

  absl::flat_hash_map<float, VtMatrix4dArray> cache;
  cache.emplace(0.0f, t0_xforms);
  for (size_t i = 0; i < t0_xforms.size(); ++i) {
    result.push_back(SampleTransformOverInterval(
        [&](float offset) {
          auto it = cache.find(offset);
          if (it == cache.end()) {
            it = cache.emplace(offset, eval_at(t0 + offset)).first;
          }
          return it->second.AsConst()[i];
        },
        interval, offsets, /*uniform=*/true));
  }
  return result;
}

GfVec2f GetMotionInterval(HdSceneDelegate* scene_delegate) {
  return GetParam<GfVec2f>(
      GetPrimDataSource(scene_delegate, SdfPath::AbsoluteRootPath()),
      HdMitsubaMotionTokens->motion_interval, GfVec2f(0.0f));
}

PXR_NAMESPACE_CLOSE_SCOPE
