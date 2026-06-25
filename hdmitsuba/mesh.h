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

#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/mesh/subdivision.h"

PXR_NAMESPACE_OPEN_SCOPE

#define HDMITSUBA_MESH_TOKENS ((subdivision_level, "mitsuba:subdivision_level"))

TF_DECLARE_PUBLIC_TOKENS(HdMitsubaMeshTokens, HDMITSUBA_MESH_TOKENS);

class SceneManager;

class HdMitsubaMesh final : public HdMesh {
 public:
  using PrimvarState = PXR_NS::PrimvarState;
  using PrimvarMap = PXR_NS::PrimvarMap;

  explicit HdMitsubaMesh(const SdfPath& id,
                         const SdfPath& instancerId = SdfPath());
  ~HdMitsubaMesh() override = default;

  HdDirtyBits GetInitialDirtyBits() const;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits, const TfToken& reprToken) override;

  void Finalize(HdRenderParam* renderParam) override;

 protected:
  HdDirtyBits GetInitialDirtyBitsMask() const override;
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;
  void _InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits) override;

 private:
  SubdivisionEvaluator subdiv_evaluator_;

  // For Multi-Material Meshes: Maps coarse face index to a material index.
  VtIntArray face_material_indices_;
  // Stores the SdfPath for each material index.
  std::vector<SdfPath> material_ids_;

  HdMeshTopology topology_;
  PrimvarMap primvars_;

  void SyncTopology(HdSceneDelegate* sceneDelegate);
  PrimvarMap SyncPrimvars(HdSceneDelegate* sceneDelegate,
                          HdDirtyBits* dirtyBits);
  void UpdateScene(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
                   const PrimvarMap& final_primvars, HdDirtyBits* dirtyBits);

  void RemoveFromScene(SceneManager* scene);

  absl::flat_hash_map<TfToken, HdPrimvarDescriptor, TfToken::HashFunctor>
  GetAllPrimvarDescriptors(HdSceneDelegate* sceneDelegate);

  size_t instance_count_ = 0;
  bool in_scene_ = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
