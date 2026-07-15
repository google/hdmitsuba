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

#include "hdmitsuba/mesh.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/extComputationUtils.h>
#include <pxr/imaging/hd/geomSubset.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/pxr.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/instancer.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/mesh/subdivision.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

bool ValidatePrimvarSize(const VtValue& value, HdInterpolation interpolation,
                         size_t vertex_count, size_t face_count,
                         size_t corner_count) {
  if (value.IsEmpty()) return false;

  size_t size = 0;
  if (value.IsHolding<VtVec3fArray>()) {
    size = value.Get<VtVec3fArray>().size();
  } else if (value.IsHolding<VtVec2fArray>()) {
    size = value.Get<VtVec2fArray>().size();
  } else if (value.IsHolding<GfVec3f>() || value.IsHolding<GfVec2f>()) {
    size = 1;
  } else {
    return true;
  }

  switch (interpolation) {
    case HdInterpolationConstant:
      return size == 1;
    case HdInterpolationUniform:
      return size == face_count;
    case HdInterpolationVertex:
      return size == vertex_count;
    case HdInterpolationFaceVarying:
      return size == corner_count;
    default:
      return false;
  }
}

std::optional<SdfPath> GetAttachedSensorId(HdSceneDelegate* sceneDelegate,
                                           const SdfPath& id) {
  VtValue attached_sensor = sceneDelegate->Get(id, TfToken("mitsuba:sensor"));
  if (attached_sensor.IsHolding<SdfPath>()) {
    return attached_sensor.Get<SdfPath>();
  } else if (attached_sensor.IsHolding<std::string>()) {
    return SdfPath(attached_sensor.Get<std::string>());
  }
  return std::nullopt;
}

std::optional<LightSpec> GetMeshEmitterSpec(HdSceneDelegate* sceneDelegate,
                                            const SdfPath& id) {
  VtValue light_intensity =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity);
  if (light_intensity.IsEmpty() || !light_intensity.IsHolding<float>()) {
    return std::nullopt;
  }
  LightSpec spec;
  spec.id = id;
  float intensity = light_intensity.Get<float>();
  GfVec3f color(1.0f, 1.0f, 1.0f);
  VtValue light_color =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
  if (light_color.IsHolding<GfVec3f>()) {
    color = light_color.Get<GfVec3f>();
  }
  float exposure = 0.0f;
  VtValue light_exposure =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->exposure);
  if (light_exposure.IsHolding<float>()) {
    exposure = light_exposure.Get<float>();
  }
  spec.emission = color * intensity * std::exp2(exposure);
  return spec;
}

}  // namespace

TF_DEFINE_PUBLIC_TOKENS(HdMitsubaMeshTokens, HDMITSUBA_MESH_TOKENS);

HdMitsubaMesh::HdMitsubaMesh(const SdfPath& id, const SdfPath& /*instancerId*/)
    : HdMesh(id) {}

HdDirtyBits HdMitsubaMesh::GetInitialDirtyBits() const {
  return HdChangeTracker::AllDirty;
}

HdDirtyBits HdMitsubaMesh::GetInitialDirtyBitsMask() const {
  return HdChangeTracker::AllDirty;
}

HdDirtyBits HdMitsubaMesh::_PropagateDirtyBits(HdDirtyBits bits) const {
  return bits;
}

void HdMitsubaMesh::Sync(HdSceneDelegate* sceneDelegate,
                         HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                         const TfToken& /*reprToken*/) {
  TRACE_FUNCTION();
  if (*dirtyBits == HdChangeTracker::Clean) {
    return;
  }
  _UpdateInstancer(sceneDelegate, dirtyBits);
  HdInstancer::_SyncInstancerAndParents(sceneDelegate->GetRenderIndex(),
                                        GetInstancerId());
  TF_DEBUG(HDMITSUBA_SYNC).Msg("HdMitsubaMesh::Sync: %s\n", GetId().GetText());

  auto* mitsuba_render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  auto* scene = mitsuba_render_param->GetScene();

  // Visibility changes can make the mesh visible or invisible.
  bool visible = sceneDelegate->GetVisible(GetId());
  if ((*dirtyBits & HdChangeTracker::DirtyVisibility) && visible) {
    *dirtyBits |= HdChangeTracker::AllDirty;
  }
  if (!visible) {
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // Update topology and/or primvars if the corresponding bits are set.
  bool topology_dirty = *dirtyBits & (HdChangeTracker::DirtyTopology |
                                      HdChangeTracker::DirtyMaterialId |
                                      HdChangeTracker::DirtyDisplayStyle);
  topology_dirty =
      topology_dirty ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, GetId(),
                                      HdMitsubaMeshTokens->subdivision_level);
  bool primvars_dirty =
      *dirtyBits &
      (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyPoints |
       HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyTransform);
  bool instancer_dirty = *dirtyBits & (HdChangeTracker::DirtyInstancer |
                                       HdChangeTracker::DirtyInstanceIndex);

  if (topology_dirty) {
    SyncTopology(sceneDelegate);
  }

  if (topology_dirty || primvars_dirty || instancer_dirty) {
    UpdateScene(sceneDelegate, renderParam,
                SyncPrimvars(sceneDelegate, dirtyBits), dirtyBits);
  }

  *dirtyBits = HdChangeTracker::Clean;
}

void HdMitsubaMesh::Finalize(HdRenderParam* renderParam) {
  auto* mitsuba_render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  auto* scene = mitsuba_render_param->GetScene();
  RemoveFromScene(scene);
}

void HdMitsubaMesh::_InitRepr(const TfToken& reprToken,
                              HdDirtyBits* dirtyBits) {
  if (reprToken == HdReprTokens->hull) {
    *dirtyBits |=
        HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
        HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyMaterialId;
  }
}

void HdMitsubaMesh::SyncTopology(HdSceneDelegate* sceneDelegate) {
  TRACE_FUNCTION();
  const HdDisplayStyle display_style = GetDisplayStyle(sceneDelegate);

  int refineLevel = display_style.refineLevel;
  VtValue subdivLevelValue =
      sceneDelegate->Get(GetId(), HdMitsubaMeshTokens->subdivision_level);
  if (!subdivLevelValue.IsEmpty() && subdivLevelValue.IsHolding<int>()) {
    refineLevel = subdivLevelValue.Get<int>();
  }
  HdMeshTopology topology =
      HdMeshTopology(GetMeshTopology(sceneDelegate), refineLevel);

  const HdGeomSubsets& geom_subsets = topology.GetGeomSubsets();
  const int num_coarse_faces = topology.GetNumFaces();
  const SdfPath material_id = sceneDelegate->GetMaterialId(GetId());
  material_ids_.clear();
  material_ids_.push_back(material_id);
  face_material_indices_.assign(num_coarse_faces, 0);
  for (const auto& subset : geom_subsets) {
    auto it = std::find(material_ids_.begin(), material_ids_.end(),
                        subset.materialId);
    int material_index = 0;
    if (it == material_ids_.end()) {
      material_index = material_ids_.size();
      material_ids_.push_back(subset.materialId);
    } else {
      material_index = std::distance(material_ids_.begin(), it);
    }
    for (const int face_index : subset.indices) {
      if (face_index >= 0 && face_index < num_coarse_faces) {
        face_material_indices_[face_index] = material_index;
      }
    }
  }

  const TfToken scheme = topology.GetScheme();
  absl::flat_hash_map<TfToken, int, TfToken::HashFunctor>
      fvar_primvar_to_channel;
  std::vector<VtIntArray> fvar_topologies;
  const TfToken fvar_interp_rule =
      topology.GetSubdivTags().GetFaceVaryingInterpolationRule();

  if (fvar_interp_rule != PxOsdOpenSubdivTokens->all && refineLevel > 0 &&
      (scheme == PxOsdOpenSubdivTokens->catmullClark ||
       scheme == PxOsdOpenSubdivTokens->loop)) {
    auto fv_primvars =
        GetPrimvarDescriptors(sceneDelegate, HdInterpolationFaceVarying);
    for (const auto& primvar : fv_primvars) {
      VtIntArray indices;
      if (primvar.indexed) {
        GetIndexedPrimvar(sceneDelegate, primvar.name, &indices);
      } else {
        VtValue value = GetPrimvar(sceneDelegate, primvar.name);
        if (!value.IsEmpty()) {
          const int num_face_varyings = topology.GetNumFaceVaryings();
          indices.resize(num_face_varyings);
          std::iota(indices.begin(), indices.end(), 0);
        }
      }
      if (!indices.empty()) {
        auto it =
            std::find(fvar_topologies.begin(), fvar_topologies.end(), indices);
        int channel;
        if (it == fvar_topologies.end()) {
          channel = fvar_topologies.size();
          fvar_topologies.push_back(indices);
        } else {
          channel = std::distance(fvar_topologies.begin(), it);
        }
        fvar_primvar_to_channel[primvar.name] = channel;
      }
    }
  }

  topology.SetSubdivTags(GetSubdivTags(sceneDelegate));

  subdiv_evaluator_.Initialize(topology.GetPxOsdMeshTopology(), refineLevel,
                               scheme, topology.GetSubdivTags(),
                               fvar_topologies, fvar_primvar_to_channel,
                               TfToken(GetId().GetText()));

  if (subdiv_evaluator_.IsSubdivided()) {
    topology_ = HdMeshTopology(scheme, topology.GetOrientation(),
                               subdiv_evaluator_.GetRefinedFaceVertexCounts(),
                               subdiv_evaluator_.GetRefinedFaceVertexIndices());
  } else {
    topology_ = std::move(topology);
  }
}

void HdMitsubaMesh::UpdateScene(HdSceneDelegate* sceneDelegate,
                                HdRenderParam* renderParam,
                                const PrimvarMap& primvars,
                                HdDirtyBits* dirtyBits) {
  TRACE_FUNCTION();
  auto* mitsuba_render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  auto* scene = mitsuba_render_param->GetScene();
  auto id = GetId();
  auto points_it = primvars.find(HdTokens->points);
  if (points_it == primvars.end() || points_it->second.value.IsEmpty() ||
      points_it->second.value.Get<VtVec3fArray>().empty()) {
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Mesh %s has no points. Removing from scene.\n", id.GetText());
    RemoveFromScene(scene);
    return;
  }

  std::optional<SdfPath> attached_sensor_id =
      GetAttachedSensorId(sceneDelegate, id);
  std::optional<LightSpec> emitter_spec = GetMeshEmitterSpec(sceneDelegate, id);

  VtMatrix4dArray instance_transforms;
  bool transforms_dirty = false;
  const SdfPath& instancer_id = GetInstancerId();
  if (!instancer_id.IsEmpty()) {
    HdMitsubaInstancer* instancer = static_cast<HdMitsubaInstancer*>(
        sceneDelegate->GetRenderIndex().GetInstancer(instancer_id));
    if (instancer) {
      instance_transforms = instancer->ComputeInstanceTransforms(id);
      transforms_dirty =
          (dirtyBits && (*dirtyBits & HdChangeTracker::DirtyInstancer)) ||
          (dirtyBits && (*dirtyBits & HdChangeTracker::DirtyInstanceIndex));
    }
  }

  bool instance_count_changed = instance_transforms.size() != instance_count_;
  instance_count_ = instance_transforms.size();

  bool topology_dirty =
      dirtyBits && (*dirtyBits & HdChangeTracker::DirtyTopology);
  bool needs_rebuild = topology_dirty || instance_count_changed;

  MeshSpec spec;
  spec.id = id;
  spec.material_ids = material_ids_;
  spec.primvars = primvars;
  spec.transform = sceneDelegate->GetTransform(id);
  spec.attached_sensor_id = attached_sensor_id;
  spec.emitter_spec = emitter_spec;
  spec.instance_transforms = instance_transforms;
  spec.transforms_dirty = transforms_dirty;
  spec.needs_rebuild = needs_rebuild;
  spec.is_subdivided = subdiv_evaluator_.IsSubdivided();

  if (subdiv_evaluator_.IsSubdivided()) {
    const auto& refined_to_coarse = subdiv_evaluator_.GetRefinedToCoarseMap();
    spec.face_material_indices.resize(refined_to_coarse.size());
    for (size_t i = 0; i < refined_to_coarse.size(); ++i) {
      spec.face_material_indices[i] =
          face_material_indices_[refined_to_coarse[i]];
    }
    spec.face_vertex_counts = subdiv_evaluator_.GetRefinedFaceVertexCounts();
    spec.face_vertex_indices = subdiv_evaluator_.GetRefinedFaceVertexIndices();
  } else {
    spec.face_material_indices = face_material_indices_;
    spec.face_vertex_counts = topology_.GetFaceVertexCounts();
    spec.face_vertex_indices = topology_.GetFaceVertexIndices();
  }

  scene->SyncMesh(std::move(spec));
  in_scene_ = true;
}

void HdMitsubaMesh::RemoveFromScene(SceneManager* scene) {
  if (in_scene_) {
    scene->RemoveShape(GetId());
    in_scene_ = false;
  }
}

absl::flat_hash_map<TfToken, HdPrimvarDescriptor, TfToken::HashFunctor>
HdMitsubaMesh::GetAllPrimvarDescriptors(HdSceneDelegate* sceneDelegate) {
  static const HdInterpolation interpolations[] = {
      HdInterpolationConstant, HdInterpolationUniform, HdInterpolationVertex,
      HdInterpolationFaceVarying, HdInterpolationVarying};
  absl::flat_hash_map<TfToken, HdPrimvarDescriptor, TfToken::HashFunctor>
      primvar_descriptors;
  for (const auto& interpolation : interpolations) {
    auto descriptors = GetPrimvarDescriptors(sceneDelegate, interpolation);
    for (const auto& descriptor : descriptors) {
      primvar_descriptors[descriptor.name] = descriptor;
    }
  }
  return primvar_descriptors;
}

HdMitsubaMesh::PrimvarMap HdMitsubaMesh::SyncPrimvars(
    HdSceneDelegate* sceneDelegate, HdDirtyBits* dirtyBits) {
  TRACE_FUNCTION();
  TF_DEBUG(HDMITSUBA_SYNC)
      .Msg("SyncPrimvars for %s dirtyBits: %d subdivided: %d\n",
           GetId().GetText(), *dirtyBits, subdiv_evaluator_.IsSubdivided());
  auto primvar_descriptors = GetAllPrimvarDescriptors(sceneDelegate);

  // Remove primvars that no longer exist from the primvar map.
  // Protect built-in geometric attributes (points, normals) from removal.
  if (*dirtyBits & HdChangeTracker::DirtyPrimvar) {
    for (auto it = primvars_.begin(); it != primvars_.end();) {
      if (it->first != HdTokens->points && it->first != HdTokens->normals &&
          !primvar_descriptors.contains(it->first)) {
        it = primvars_.erase(it);
      } else {
        ++it;
      }
    }
  }

  const SdfPath& id = GetId();
  if (subdiv_evaluator_.IsSubdivided()) {
    primvars_.erase(HdTokens->normals);
  }

  bool transform_dirty = *dirtyBits & HdChangeTracker::DirtyTransform;

  // Query and evaluate computed primvars.
  HdExtComputationPrimvarDescriptorVector computed_primvar_descs;
  for (size_t i = 0; i < HdInterpolationCount; ++i) {
    HdInterpolation interp = static_cast<HdInterpolation>(i);
    auto descs = sceneDelegate->GetExtComputationPrimvarDescriptors(id, interp);
    for (const auto& desc : descs) {
      if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, desc.name) ||
          transform_dirty) {
        computed_primvar_descs.push_back(desc);
      }
    }
  }

  HdExtComputationUtils::ValueStore computed_values;
  if (!computed_primvar_descs.empty()) {
    computed_values = HdExtComputationUtils::GetComputedPrimvarValues(
        computed_primvar_descs, sceneDelegate);
  }

  // Helper to resolve the value (computed vs standard)
  auto resolve_primvar_value = [&](const TfToken& name) -> VtValue {
    auto it = computed_values.find(name);
    if (it != computed_values.end()) {
      return it->second;
    }
    return GetPrimvar(sceneDelegate, name);
  };

  // Helper to sync and optionally refine a primvar
  auto sync_primvar = [&](const TfToken& name, const HdPrimvarDescriptor& desc,
                          bool refine = true) {
    VtValue value = resolve_primvar_value(name);
    if (value.IsEmpty()) return;
    if (refine && subdiv_evaluator_.IsSubdivided()) {
      value =
          subdiv_evaluator_.RefinePrimvar(value, desc.interpolation, desc.name);
    }
    primvars_[name] = {std::move(value), desc};
  };

  // 1. Explicitly sync points (built-in geometric attribute)
  if ((*dirtyBits & HdChangeTracker::DirtyPoints) || transform_dirty) {
    HdPrimvarDescriptor desc{HdTokens->points, HdInterpolationVertex,
                             HdPrimvarRoleTokens->point};
    sync_primvar(HdTokens->points, desc);
  }

  // 2. Explicitly sync normals (only if no subdivision is used).
  if (!subdiv_evaluator_.IsSubdivided() &&
      ((*dirtyBits & HdChangeTracker::DirtyNormals) || transform_dirty)) {
    VtValue value = resolve_primvar_value(HdTokens->normals);
    if (!value.IsEmpty()) {
      HdPrimvarDescriptor desc;
      if (primvar_descriptors.contains(HdTokens->normals)) {
        desc = primvar_descriptors.at(HdTokens->normals);
      } else {
        desc.name = HdTokens->normals;
        desc.interpolation = HdInterpolationVertex;
        desc.role = HdPrimvarRoleTokens->normal;
      }
      size_t vertex_count = 0;
      auto points_it = primvars_.find(HdTokens->points);
      if (points_it != primvars_.end() &&
          points_it->second.value.IsHolding<VtVec3fArray>()) {
        vertex_count = points_it->second.value.Get<VtVec3fArray>().size();
      }
      size_t face_count = topology_.GetNumFaces();
      size_t corner_count = topology_.GetNumFaceVaryings();
      if (ValidatePrimvarSize(value, desc.interpolation, vertex_count,
                              face_count, corner_count)) {
        primvars_[HdTokens->normals] = {std::move(value), desc};
      } else {
        TF_WARN(
            "Ignored invalid normals for %s (size mismatch). "
            "Interpolation: %d, Actual size: %zu, Expected (Vertex: %zu, Face: "
            "%zu, Corner: %zu)",
            id.GetText(), (int)desc.interpolation,
            value.IsHolding<VtVec3fArray>() ? value.Get<VtVec3fArray>().size()
                                            : 0,
            vertex_count, face_count, corner_count);
      }
    }
  }

  // 3. Sync user primvars
  for (auto const& [token, descriptor] : primvar_descriptors) {
    // Skip built-in geometric attributes that are handled explicitly
    if (token == HdTokens->points || token == HdTokens->normals) {
      continue;
    }
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncPrimvar: %s\n", token.GetText());
    if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, token)) {
      sync_primvar(token, descriptor);
    }
  }
  return primvars_;
}

PXR_NAMESPACE_CLOSE_SCOPE
