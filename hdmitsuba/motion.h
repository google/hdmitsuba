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
#include <utility>
#include <vector>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

// Root-prim data-source key publishing the union of all camera shutter
// intervals from HdMitsuba_MotionIntervalSceneIndex to HdMitsubaLight::Sync.
#define HDMITSUBA_MOTION_TOKENS ((motion_interval, "mitsuba:motion_interval"))

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

  // Maps every sample through `f`, collapsing to Static if all results match.
  template <typename F>
  auto Map(F&& f) const {
    using U = decltype(f(First()));
    MotionSamples<U> result;
    result.samples.reserve(samples.size());
    for (const auto& [time, value] : samples) {
      result.samples.emplace_back(time, f(value));
    }
    if (std::all_of(result.samples.begin() + 1, result.samples.end(),
                    [&](const auto& s) { return s.second == result.First(); })) {
      return MotionSamples<U>::Static(result.First());
    }
    return result;
  }
};

using MotionTransform = MotionSamples<ScalarAffineTransform4f>;

// Samples the world transform of `prim_source` (or `prim` at `time`) over
// `interval`, adaptively subdividing segments where TRS interpolation deviates
// from the actual trajectory (e.g. rotations >= 90 deg or orbits).
MotionSamples<GfMatrix4d> SampleTransform(
    const HdContainerDataSourceHandle& prim_source, const GfVec2f& interval);
MotionSamples<GfMatrix4d> SampleTransform(const UsdPrim& prim, UsdTimeCode time,
                                          const GfVec2f& interval);

// Returns the union of the shutter intervals of all cameras in the scene.
GfVec2f GetMotionInterval(HdSceneDelegate* scene_delegate);

PXR_NAMESPACE_CLOSE_SCOPE
