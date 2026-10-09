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

#include "hdmitsuba/mesh/subdivision.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/types.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/refinerFactory.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// Evaluates subdivision stencils for a primvar array.
template <typename T>
VtArray<T> EvalSubdivisionStencil(
    const VtArray<T>& coarsePrimvar,
    const OpenSubdiv::Far::StencilTable* stencil,
    bool include_coarse = true) {
  if (!TF_VERIFY(stencil, "OSD stencil must not be null")) {
    return {};
  }
  const int num_coarse =
      include_coarse ? stencil->GetNumControlVertices() : 0;
  const int num_refined = stencil->GetNumStencils();
  VtArray<T> refined_primvar(num_coarse + num_refined);
  T* dst_ptr = refined_primvar.data();
  const size_t src_size = coarsePrimvar.size();

  // 1. Copy coarse primvars.
  if (include_coarse) {
    const size_t num_src_to_copy =
        std::min(static_cast<size_t>(num_coarse), src_size);
    for (size_t i = 0; i < num_src_to_copy; ++i) {
      dst_ptr[i] = coarsePrimvar[i];
    }
    for (size_t i = num_src_to_copy; i < static_cast<size_t>(num_coarse); ++i) {
      dst_ptr[i] = T(0);
    }
  }

  if (num_refined == 0) {
    return refined_primvar;
  }

  // 2. Compute refined primvars.
  const int* sizes = stencil->GetSizes().data();
  const int* offsets = stencil->GetOffsets().data();
  const int* indices = stencil->GetControlIndices().data();
  const float* weights = stencil->GetWeights().data();

  for (int i = 0; i < num_refined; ++i) {
    T refined_value(0);
    const int stencil_size = sizes[i];
    const int stencil_offset = offsets[i];
    for (int j = 0; j < stencil_size; ++j) {
      const int index = indices[stencil_offset + j];
      const float weight = weights[stencil_offset + j];
      if (index >= 0 && static_cast<size_t>(index) < src_size) {
        refined_value += coarsePrimvar[index] * weight;
      }
    }
    dst_ptr[num_coarse + i] = refined_value;
  }
  return refined_primvar;
}

}  // namespace

void SubdivisionEvaluator::Initialize(
    const PxOsdMeshTopology& topology, int refine_level,
    const std::vector<VtIntArray>& fvar_topologies,
    const absl::flat_hash_map<TfToken, int, TfToken::HashFunctor>&
        fvar_primvar_to_channel,
    const TfToken& mesh_name) {
  Clear();

  const TfToken scheme = topology.GetScheme();
  if ((scheme != PxOsdOpenSubdivTokens->catmullClark &&
       scheme != PxOsdOpenSubdivTokens->loop) ||
      refine_level <= 0) {
    return;
  }

  fvar_primvar_to_channel_ = fvar_primvar_to_channel;

  auto refiner =
      PxOsdRefinerFactory::Create(topology, fvar_topologies, mesh_name);
  refiner->RefineUniform(
      OpenSubdiv::Far::TopologyRefiner::UniformOptions(refine_level));
  refiner_ = std::move(refiner);

  OpenSubdiv::Far::StencilTableFactory::Options stencil_options;
  stencil_options.generateOffsets = true;
  stencil_options.generateIntermediateLevels = false;
  stencil_options.interpolationMode =
      OpenSubdiv::Far::StencilTableFactory::INTERPOLATE_VERTEX;
  vertex_stencils_ = std::unique_ptr<OpenSubdiv::Far::StencilTable const>(
      OpenSubdiv::Far::StencilTableFactory::Create(*refiner_, stencil_options));

  stencil_options.interpolationMode =
      OpenSubdiv::Far::StencilTableFactory::INTERPOLATE_VARYING;
  varying_stencils_ = std::unique_ptr<OpenSubdiv::Far::StencilTable const>(
      OpenSubdiv::Far::StencilTableFactory::Create(*refiner_, stencil_options));

  const int num_fvar_channels = refiner_->GetNumFVarChannels();
  stencil_options.interpolationMode =
      OpenSubdiv::Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
  for (int i = 0; i < num_fvar_channels; ++i) {
    stencil_options.fvarChannel = i;
    face_varying_stencils_.push_back(
        std::unique_ptr<const OpenSubdiv::Far::StencilTable>(
            OpenSubdiv::Far::StencilTableFactory::Create(*refiner_,
                                                         stencil_options)));
  }

  OpenSubdiv::Far::PatchTableFactory::Options patch_options(refine_level);
  if (num_fvar_channels > 0) {
    patch_options.generateFVarTables = true;
  }
  patch_table_ = std::unique_ptr<OpenSubdiv::Far::PatchTable const>(
      OpenSubdiv::Far::PatchTableFactory::Create(*refiner_, patch_options));

  const int max_level = refiner_->GetMaxLevel();
  const int num_refined_faces = patch_table_->GetNumPatchesTotal();
  refined_to_coarse_map_.resize(num_refined_faces);
  for (int i = 0; i < num_refined_faces; ++i) {
    int parent_face_index = i;
    for (int level = max_level; level > 0; --level) {
      parent_face_index =
          refiner_->GetLevel(level).GetFaceParentFace(parent_face_index);
    }
    refined_to_coarse_map_[i] = parent_face_index;
  }

  const OpenSubdiv::Far::Index* refined_indices_ptr =
      patch_table_->GetPatchControlVerticesTable().data();
  const size_t num_refined_indices =
      patch_table_->GetPatchControlVerticesTable().size();
  if (num_refined_indices > 0) {
    refined_face_vertex_indices_.assign(
        refined_indices_ptr, refined_indices_ptr + num_refined_indices);
  }

  if (scheme == PxOsdOpenSubdivTokens->loop) {
    refined_face_vertex_counts_.assign(num_refined_indices / 3, 3);
  } else {
    refined_face_vertex_counts_.assign(num_refined_indices / 4, 4);
  }
}

void SubdivisionEvaluator::Clear() {
  refiner_.reset();
  vertex_stencils_.reset();
  varying_stencils_.reset();
  face_varying_stencils_.clear();
  patch_table_.reset();
  refined_to_coarse_map_.clear();
  fvar_primvar_to_channel_.clear();
  refined_face_vertex_counts_.clear();
  refined_face_vertex_indices_.clear();
}

VtIntArray SubdivisionEvaluator::MapRefinedMaterialIndices(
    const VtIntArray& coarse_material_indices) const {
  if (!IsSubdivided()) return coarse_material_indices;
  const size_t num_refined = refined_to_coarse_map_.size();
  VtIntArray mapped(num_refined);
  int* dst_ptr = mapped.data();
  for (size_t i = 0; i < num_refined; ++i) {
    dst_ptr[i] = coarse_material_indices[refined_to_coarse_map_[i]];
  }
  return mapped;
}

template <typename T>
VtValue SubdivisionEvaluator::RefinePrimvarImpl(
    const VtValue& value, const OpenSubdiv::Far::StencilTable* stencil,
    int fvar_channel) const {
  if (!value.IsHolding<T>()) {
    return value;
  }
  const T& initial_values = value.UncheckedGet<T>();
  if (initial_values.empty()) {
    return value;
  }

  if (fvar_channel < 0) {
    T refined_values = EvalSubdivisionStencil(initial_values, stencil,
                                              /*include_coarse=*/true);
    return VtValue(std::move(refined_values));
  }

  // If the primvar is face-varying, we only need the refined values and flatten
  // them via the patch table's face-varying value indices.
  const T refined_values = EvalSubdivisionStencil(initial_values, stencil,
                                                  /*include_coarse=*/false);
  const auto& fvar_indices = patch_table_->GetFVarValues(fvar_channel);
  const size_t num_fvar = fvar_indices.size();
  T flattened_values(num_fvar);
  auto* dst_ptr = flattened_values.data();
  for (size_t i = 0; i < num_fvar; ++i) {
    dst_ptr[i] = refined_values[fvar_indices[i]];
  }
  return VtValue(std::move(flattened_values));
}

VtValue SubdivisionEvaluator::RefinePrimvar(const VtValue& value,
                                            HdInterpolation interpolation,
                                            const TfToken& token) const {
  if (!IsSubdivided()) {
    return value;
  }

  int fvar_channel = -1;
  const OpenSubdiv::Far::StencilTable* stencil = nullptr;
  if (interpolation == HdInterpolationFaceVarying) {
    auto it = fvar_primvar_to_channel_.find(token);
    if (it != fvar_primvar_to_channel_.end()) {
      fvar_channel = it->second;
      if (fvar_channel >= 0 &&
          static_cast<size_t>(fvar_channel) < face_varying_stencils_.size()) {
        stencil = face_varying_stencils_[fvar_channel].get();
      }
    }
  } else if (interpolation == HdInterpolationVertex) {
    stencil = vertex_stencils_.get();
  } else if (interpolation == HdInterpolationVarying) {
    stencil = varying_stencils_.get();
  }

  if (!stencil) {
    return value;
  }

  if (value.IsHolding<VtVec3fArray>()) {
    return RefinePrimvarImpl<VtVec3fArray>(value, stencil, fvar_channel);
  }
  if (value.IsHolding<VtVec2fArray>()) {
    return RefinePrimvarImpl<VtVec2fArray>(value, stencil, fvar_channel);
  }
  if (value.IsHolding<VtFloatArray>()) {
    return RefinePrimvarImpl<VtFloatArray>(value, stencil, fvar_channel);
  }
  return value;
}

PXR_NAMESPACE_CLOSE_SCOPE
