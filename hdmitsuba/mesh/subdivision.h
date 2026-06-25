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

#include <memory>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/topologyRefiner.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class SubdivisionEvaluator {
 public:
  SubdivisionEvaluator() = default;
  ~SubdivisionEvaluator() = default;

  // Initializes the OpenSubdiv refiner and stencil tables for the given mesh
  // topology.
  void Initialize(const PxOsdMeshTopology& topology, int refine_level,
                  const TfToken& scheme, const PxOsdSubdivTags& subdiv_tags,
                  const std::vector<VtIntArray>& fvar_topologies,
                  const absl::flat_hash_map<TfToken, int, TfToken::HashFunctor>&
                      fvar_primvar_to_channel,
                  const TfToken& mesh_name);

  // Clears all OpenSubdiv data structures.
  void Clear();

  // Returns true if the mesh is currently configured for subdivision.
  bool IsSubdivided() const { return refiner_ != nullptr; }

  // Refines a primvar using the evaluated subdivision stencils.
  VtValue RefinePrimvar(const VtValue& value, HdInterpolation interpolation,
                        const TfToken& token) const;

  const VtIntArray& GetRefinedFaceVertexCounts() const {
    return refined_face_vertex_counts_;
  }

  const VtIntArray& GetRefinedFaceVertexIndices() const {
    return refined_face_vertex_indices_;
  }

  const std::vector<int>& GetRefinedToCoarseMap() const {
    return refined_to_coarse_map_;
  }

  const OpenSubdiv::Far::PatchTable* GetPatchTable() const {
    return patch_table_.get();
  }

 private:
  template <typename T>
  VtValue RefinePrimvarImpl(const VtValue& value,
                            const OpenSubdiv::Far::StencilTable* stencil,
                            int fvar_channel) const;

  std::shared_ptr<OpenSubdiv::Far::TopologyRefiner> refiner_;
  std::unique_ptr<OpenSubdiv::Far::StencilTable const> vertex_stencils_;
  std::unique_ptr<OpenSubdiv::Far::StencilTable const> varying_stencils_;
  std::vector<std::unique_ptr<const OpenSubdiv::Far::StencilTable>>
      face_varying_stencils_;
  std::unique_ptr<OpenSubdiv::Far::PatchTable const> patch_table_;

  // Maps each refined face index to its parent coarse face index.
  std::vector<int> refined_to_coarse_map_;

  // Maps face-varying primvar tokens to their corresponding channels in the
  // OpenSubdiv topology.
  absl::flat_hash_map<TfToken, int, TfToken::HashFunctor>
      fvar_primvar_to_channel_;

  VtIntArray refined_face_vertex_counts_;
  VtIntArray refined_face_vertex_indices_;
};

PXR_NAMESPACE_CLOSE_SCOPE
