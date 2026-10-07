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
#include <string>
#include <tuple>

#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

class SceneManager;

class HdMitsubaLight final : public HdLight {
 public:
  HdMitsubaLight(const SdfPath& id, const TfToken& typeId);
  ~HdMitsubaLight() override = default;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

  void Finalize(HdRenderParam* renderParam) override;

 private:
  void RemoveFromScene(SceneManager* scene);

  // Properties that determine the Mitsuba plugin type, whose changes require
  // rebuilding the light object rather than updating it in place.
  struct RebuildKey {
    bool treat_as_point = false;
    bool has_positive_angle = false;
    bool is_spot = false;
    std::string texture_file_path;

    bool operator==(const RebuildKey& o) const {
      return std::tie(treat_as_point, has_positive_angle, is_spot,
                      texture_file_path) ==
             std::tie(o.treat_as_point, o.has_positive_angle, o.is_spot,
                      o.texture_file_path);
    }
    bool operator!=(const RebuildKey& o) const { return !(*this == o); }
  };

  TfToken type_id_;
  std::optional<RebuildKey> rebuild_key_;
};

PXR_NAMESPACE_CLOSE_SCOPE
