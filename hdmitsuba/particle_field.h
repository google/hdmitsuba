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

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

class SceneManager;

class HdMitsubaParticleField final : public HdRprim {
 public:
  explicit HdMitsubaParticleField(const SdfPath& id);
  ~HdMitsubaParticleField() override = default;

  HdDirtyBits GetInitialDirtyBits() const;

  TfTokenVector const& GetBuiltinPrimvarNames() const override;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits, const TfToken& reprToken) override;

  void Finalize(HdRenderParam* renderParam) override;

 protected:
  HdDirtyBits GetInitialDirtyBitsMask() const override;
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;
  void _InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits) override;

 private:
  void RemoveFromScene(SceneManager* scene);

  VtVec3fArray points_;
  VtVec3fArray scales_;
  VtQuatfArray orientations_;
  VtFloatArray opacities_;
  VtVec3fArray sh_coeffs_;
  int sh_degree_ = 0;
  GfMatrix4d transform_ = GfMatrix4d(1.0);
  // Whether the shape currently exists in the Mitsuba scene. Cleared when the
  // prim is removed for being invisible, so that re-showing it rebuilds.
  bool in_scene_ = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
