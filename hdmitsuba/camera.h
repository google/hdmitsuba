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

#include <string>

#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

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

  std::string sensor_type_ = "";
  std::string film_pixel_filter_type_ = "";
  bool is_instantiated_ = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
