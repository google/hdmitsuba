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

#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

class SceneManager;

// A sphere or cylinder rendered as a native Mitsuba shape. Other implicit
// surfaces are tessellated by the scene index (see scene_index_plugin.cc),
// which also folds the spine axis of cylinders into their transform.
//
// Mitsuba has no tapered cylinder, so a cylinder with differing top and bottom
// radii is rendered with its bottom radius.
class HdMitsubaImplicitSurface final : public HdRprim {
 public:
  HdMitsubaImplicitSurface(const SdfPath& id, const TfToken& type);
  ~HdMitsubaImplicitSurface() override = default;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits, const TfToken& reprToken) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;
  const TfTokenVector& GetBuiltinPrimvarNames() const override;

 protected:
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;
  void _InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits) override;

 private:
  void RemoveFromScene(SceneManager* scene);

  TfToken type_;
  bool in_scene_ = false;
  bool warned_tapered_ = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
