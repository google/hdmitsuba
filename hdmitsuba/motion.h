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

// Utilities for sampling time-varying transforms over camera shutter intervals.

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

// Root-prim data-source key publishing the union of all camera shutter
// intervals from HdMitsuba_MotionIntervalSceneIndex, and per-prim primvar
// controlling the maximum number of transform motion samples.
#define HDMITSUBA_MOTION_TOKENS                  \
  ((motion_interval, "mitsuba:motion_interval")) \
  ((motion_samples, "mitsuba:motion_samples"))

TF_DECLARE_PUBLIC_TOKENS(HdMitsubaMotionTokens, HDMITSUBA_MOTION_TOKENS);

// A value sampled at a sorted list of frame-relative shutter offsets.
template <typename T>
struct MotionSamples {
  std::vector<std::pair<float, T>> samples;

  static MotionSamples Static(T value) { return {{{0.0f, std::move(value)}}}; }

  bool IsAnimated() const { return samples.size() > 1; }
  const T& First() const {
    TF_DEV_AXIOM(!samples.empty());
    return samples.front().second;
  }

  // Returns these samples, collapsed to Static if all values match.
  MotionSamples Collapsed() && {
    if (std::all_of(samples.begin() + 1, samples.end(),
                    [&](const auto& s) { return s.second == First(); })) {
      return Static(First());
    }
    return std::move(*this);
  }

  // Maps every sample through `f`, collapsing to Static if all results match.
  template <typename F>
  auto Map(F&& f) const {
    MotionSamples<decltype(f(First()))> result;
    result.samples.reserve(samples.size());
    for (const auto& [time, value] : samples) {
      result.samples.emplace_back(time, f(value));
    }
    return std::move(result).Collapsed();
  }
};

using MotionTransform = MotionSamples<ScalarAffineTransform4f>;

// --- Hydra render delegate ---

// Returns the union of the shutter intervals of all cameras in the scene.
GfVec2f GetMotionInterval(HdSceneDelegate* scene_delegate);

// Returns `primvars:mitsuba:motion_samples` on `prim_source`, or the default
// of 32 if unauthored.
int GetMotionSamples(const HdContainerDataSourceHandle& prim_source);

// Samples the world transform of `prim_source` over `interval` with at most
// `primvars:mitsuba:motion_samples` samples, adaptively subdividing segments
// where TRS interpolation deviates from the actual trajectory (e.g. rotations
// >= 90 deg or orbits). When `uniform` is true, produces uniformly spaced
// keyframes over `interval` as required by Mitsuba's `instance` plugin.
MotionSamples<GfMatrix4d> SampleTransform(
    const HdContainerDataSourceHandle& prim_source, const GfVec2f& interval,
    bool uniform = false);

// Samples an array of instance world-transform matrices over `interval` with
// uniform keyframe spacing and at most `max_samples` samples. `times` holds the
// authored frame-relative sample offsets of the varying inputs, or `nullopt`
// if nothing varies over `interval` (static sampling at `0.0f`).
std::vector<MotionSamples<GfMatrix4d>> SampleInstanceTransforms(
    const std::function<VtMatrix4dArray(float)>& eval_matrices,
    const GfVec2f& interval, const std::optional<std::vector<float>>& times,
    int max_samples);

// --- Offline translation (usd_mitsuba, via geometry_ext) ---

// Like the Hydra `SampleTransform` above, for `prim` at `time`. The motion
// sample primvar is inherited from ancestors.
MotionSamples<GfMatrix4d> SampleTransform(const UsdPrim& prim, UsdTimeCode time,
                                          const GfVec2f& interval,
                                          bool uniform = false);

// Samples the instance-to-world transforms of a `UsdGeomPointInstancer` `prim`
// at `time` over `interval` with uniform keyframe spacing.
std::vector<MotionSamples<GfMatrix4d>> SamplePointInstancerTransforms(
    const UsdPrim& prim, UsdTimeCode time, const GfVec2f& interval);

PXR_NAMESPACE_CLOSE_SCOPE
