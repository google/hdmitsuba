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
#include <unordered_map>

#include <absl/base/const_init.h>
#include <absl/synchronization/mutex.h>
#include <mitsuba/core/object.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/aov.h>
#include <optional>
#include <pxr/base/gf/rect2i.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh.h"
#include "hdmitsuba/spec_types.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdRenderPass;

class SceneManager {
 public:
  using PrimvarMap = std::unordered_map<TfToken, HdMitsubaMesh::PrimvarState,
                                        TfToken::HashFunctor>;

  SceneManager();
  virtual ~SceneManager();

  // Prim creation and update methods
  virtual void SyncMesh(MeshSpec spec) = 0;
  virtual void SyncCurves(CurveSpec spec) = 0;
  virtual void SyncCamera(CameraSpec spec) = 0;
  virtual void SyncLight(LightSpec spec) = 0;
  virtual void SyncMaterial(MaterialSpec spec) = 0;

  virtual void RemoveShape(const SdfPath& id) = 0;
  virtual void RemoveLight(const SdfPath& id) = 0;
  virtual void RemoveMaterial(const SdfPath& id) = 0;

  virtual void UpdateNamespacedSettings(
      const VtDictionary& namespaced_settings) = 0;

  virtual void SetAovBindings(
      const HdRenderPass* render_pass,
      const HdRenderPassAovBindingVector& aov_bindings) = 0;

  virtual void Render(
      const HdRenderPass* render_pass,
      const HdCamera* camera,
      const std::optional<GfRect2i>& crop_window = std::nullopt) = 0;

  virtual bool IsConverged() const = 0;

  virtual void CommitResources() = 0;

  virtual mitsuba::Object* GetScene() = 0;

  static SceneManager* CreateSceneManager(const std::string& variant);

 private:
  inline static absl::Mutex lifecycle_mutex_{absl::kConstInit};
  inline static bool owns_static_initialization_ = false;
  inline static int active_instances_ = 0;
};

PXR_NAMESPACE_CLOSE_SCOPE
