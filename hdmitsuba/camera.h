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

#pragma once

#include <optional>
#include <string_view>
#include <tuple>

#include <pxr/base/gf/vec2f.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/motion.h"
#include "hdmitsuba/plugin_builder.h"

PXR_NAMESPACE_OPEN_SCOPE

inline constexpr std::string_view kMitsubaSensorNamespace = "mitsuba:sensor:";

class HdMitsubaCamera final : public HdCamera {
 public:
  explicit HdMitsubaCamera(const SdfPath& id);
  ~HdMitsubaCamera() override = default;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

 private:
  float GetHorizontalFieldOfView() const;
  float GetHorizontalPrincipalPointOffset() const;
  float GetVerticalPrincipalPointOffset() const;

  PluginSpecNode sensor_spec_{"perspective"};
  std::optional<SdfPath> target_shape_id_;
  std::tuple<float, float, float, float, float> usd_params_{
      90.0f, 0.0f, 0.0f, 0.01f, 1000.0f};
  bool is_instantiated_ = false;
  MotionTransform transform_;
  GfVec2f shutter_{0.0f};
};

PXR_NAMESPACE_CLOSE_SCOPE
