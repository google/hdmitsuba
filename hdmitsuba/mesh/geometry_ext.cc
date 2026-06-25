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

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/utils.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/mesh/subdivision.h"
#include "nanobind/usd.h"

NANOBIND_BOOST_CASTER(pxr::VtIntArray, "Vt.IntArray");
NANOBIND_BOOST_CASTER(pxr::VtVec3fArray, "Vt.Vec3fArray");
NANOBIND_BOOST_CASTER(pxr::VtVec2fArray, "Vt.Vec2fArray");
NANOBIND_BOOST_CASTER(pxr::GfMatrix4d, "Gf.Matrix4d");

namespace nb = nanobind;
PXR_NAMESPACE_USING_DIRECTIVE

struct MeshExtractionData {
  SdfPath id;
  VtIntArray face_vertex_counts;
  VtIntArray face_vertex_indices;
  TfToken scheme;
  TfToken orientation;
  TfToken interp_boundary = PxOsdOpenSubdivTokens->edgeAndCorner;
  TfToken face_varying_interp = PxOsdOpenSubdivTokens->cornersPlus1;
  PrimvarMap primvars;
  std::vector<SdfPath> material_ids;
  VtIntArray face_material_indices;

  HdMeshTopology GetTopology() const {
    PxOsdSubdivTags tags(interp_boundary, face_varying_interp, TfToken(),
                         TfToken(),
                         /*creaseIndices=*/VtIntArray(),
                         /*creaseLengths=*/VtIntArray(),
                         /*creaseWeights=*/VtFloatArray(),
                         /*cornerIndices=*/VtIntArray(),
                         /*cornerWeights=*/VtFloatArray());
    PxOsdMeshTopology px_osd_topo(scheme, orientation, face_vertex_counts,
                                  face_vertex_indices, VtIntArray(), tags);
    return HdMeshTopology(px_osd_topo, 0);
  }
};

HdInterpolation UsdInterpolationToHdInterpolation(const TfToken& interp) {
  if (interp == UsdGeomTokens->constant) return HdInterpolationConstant;
  if (interp == UsdGeomTokens->uniform) return HdInterpolationUniform;
  if (interp == UsdGeomTokens->vertex) return HdInterpolationVertex;
  if (interp == UsdGeomTokens->varying) return HdInterpolationVarying;
  if (interp == UsdGeomTokens->faceVarying) return HdInterpolationFaceVarying;
  return HdInterpolationConstant;
}

namespace {

// Fetches the mesh's vertex positions at a given time.
std::pair<VtVec3fArray, VtVec3fArray> GetMeshPointsAndNormals(
    const UsdPrim& prim, const UsdGeomMesh& mesh, UsdTimeCode time) {
  VtVec3fArray points;
  VtVec3fArray normals;
  if (!mesh.GetPointsAttr().Get(&points, time)) {
    return {points, normals};
  }
  bool has_normals = mesh.GetNormalsAttr().Get(&normals, time);

  UsdSkelRoot skel_root = UsdSkelRoot::Find(prim);
  if (!skel_root) return {points, normals};
  UsdSkelCache skel_cache;
  skel_cache.Populate(skel_root, UsdPrimDefaultPredicate);
  UsdSkelSkinningQuery skinning_query = skel_cache.GetSkinningQuery(prim);
  if (!skinning_query) return {points, normals};

  std::vector<UsdSkelBinding> bindings;
  skel_cache.ComputeSkelBindings(skel_root, &bindings, UsdPrimDefaultPredicate);
  UsdSkelSkeletonQuery skel_query;
  for (const auto& binding : bindings) {
    for (const auto& target : binding.GetSkinningTargets()) {
      if (target.GetPrim() == prim) {
        skel_query = skel_cache.GetSkelQuery(binding.GetSkeleton());
        break;
      }
    }
    if (skel_query) break;
  }
  if (!skel_query) return {points, normals};

  VtMatrix4dArray skinning_xforms;
  if (!skel_query.ComputeSkinningTransforms(&skinning_xforms, time)) {
    return {points, normals};
  }

  bool skinned_points = false;
  if (!points.empty()) {
    skinned_points =
        skinning_query.ComputeSkinnedPoints(skinning_xforms, &points, time);
  }

  bool skinned_normals = false;
  if (has_normals && !normals.empty()) {
    skinned_normals =
        skinning_query.ComputeSkinnedNormals(skinning_xforms, &normals, time);
  }

  if (skinned_points || skinned_normals) {
    UsdGeomXformCache xform_cache(time);
    GfMatrix4d skel_local_to_world =
        xform_cache.GetLocalToWorldTransform(skel_query.GetPrim());
    GfMatrix4d prim_local_to_world = xform_cache.GetLocalToWorldTransform(prim);
    GfMatrix4d skel_to_prim_local =
        skel_local_to_world * prim_local_to_world.GetInverse();

    if (skinned_points) {
      for (auto& point : points) {
        point = GfVec3f(skel_to_prim_local.Transform(point));
      }
    }
    if (skinned_normals) {
      GfMatrix3d skel_to_prim_local_inv_transpose =
          skel_to_prim_local.ExtractRotationMatrix()
              .GetInverse()
              .GetTranspose();
      for (auto& normal : normals) {
        normal = GfVec3f(normal * skel_to_prim_local_inv_transpose);
      }
    }
  }
  return {points, normals};
}

}  // namespace

MeshExtractionData ExtractMeshData(UsdStagePtr stage, const SdfPath& path,
                                   UsdTimeCode time) {
  MeshExtractionData data;
  data.id = path;

  if (!stage) return data;
  UsdPrim prim = stage->GetPrimAtPath(path);
  if (!prim || !prim.IsA<UsdGeomMesh>()) return data;

  UsdGeomMesh mesh(prim);
  mesh.GetFaceVertexCountsAttr().Get(&data.face_vertex_counts, time);
  mesh.GetFaceVertexIndicesAttr().Get(&data.face_vertex_indices, time);
  mesh.GetSubdivisionSchemeAttr().Get(&data.scheme);
  mesh.GetOrientationAttr().Get(&data.orientation);
  if (UsdAttribute attr = mesh.GetInterpolateBoundaryAttr()) {
    attr.Get(&data.interp_boundary);
  }
  if (UsdAttribute attr = mesh.GetFaceVaryingLinearInterpolationAttr()) {
    attr.Get(&data.face_varying_interp);
  }

  UsdGeomPrimvarsAPI primvars_api(mesh);
  for (const UsdGeomPrimvar& primvar : primvars_api.GetPrimvars()) {
    HdPrimvarDescriptor desc;
    desc.name = primvar.GetPrimvarName();
    desc.interpolation =
        UsdInterpolationToHdInterpolation(primvar.GetInterpolation());
    VtValue value;
    if (primvar.ComputeFlattened(&value, time) && !value.IsEmpty()) {
      data.primvars[desc.name] = {std::move(value), desc};
    }
  }

  auto [points, normals] = GetMeshPointsAndNormals(prim, mesh, time);

  if (!points.empty()) {
    HdPrimvarDescriptor desc;
    desc.name = HdTokens->points;
    desc.interpolation = HdInterpolationVertex;
    desc.role = HdPrimvarRoleTokens->point;
    data.primvars[desc.name] = {VtValue(points), desc};
  }

  if (!normals.empty()) {
    HdPrimvarDescriptor desc;
    desc.name = HdTokens->normals;
    desc.interpolation =
        UsdInterpolationToHdInterpolation(mesh.GetNormalsInterpolation());
    desc.role = HdPrimvarRoleTokens->normal;
    data.primvars[desc.name] = {VtValue(normals), desc};
  }

  int num_faces = data.face_vertex_counts.size();
  data.face_material_indices.assign(num_faces, 0);
  data.material_ids.push_back(SdfPath());

  if (UsdAttribute subset_family_attr =
          prim.GetAttribute(TfToken("subsetFamily:materialBind:familyType"))) {
    TfToken family_type;
    subset_family_attr.Get(&family_type);
    if (family_type == TfToken("nonOverlapping")) {
      for (const UsdPrim& child : prim.GetChildren()) {
        if (child.IsA<UsdGeomSubset>()) {
          UsdGeomSubset subset(child);
          TfToken family_name;
          subset.GetFamilyNameAttr().Get(&family_name);
          TfToken elem_type;
          subset.GetElementTypeAttr().Get(&elem_type);
          if (family_name == TfToken("materialBind") &&
              elem_type == TfToken("face")) {
            VtIntArray indices;
            subset.GetIndicesAttr().Get(&indices, time);
            int material_index = data.material_ids.size();
            data.material_ids.push_back(child.GetPath());
            for (int face_idx : indices) {
              if (face_idx >= 0 && face_idx < num_faces) {
                data.face_material_indices[face_idx] = material_index;
              }
            }
          }
        }
      }
    }
  }

  return data;
}

NB_MODULE(geometry_ext, m) {
  m.doc() = "Python bindings for hdmitsuba geometry processing utilities.";

  nb::class_<PrimvarState>(m, "PrimvarState")
      .def_rw("value", &PrimvarState::value);

  nb::class_<SubMeshOutput>(m, "SubMeshOutput")
      .def_ro("material_id", &SubMeshOutput::material_id)
      .def_ro("triangles", &SubMeshOutput::triangles)
      .def_rw("primvars", &SubMeshOutput::primvars);

  nb::class_<HdMeshTopology>(m, "HdMeshTopology")
      .def(nb::init<const TfToken&, const TfToken&, const VtIntArray&,
                    const VtIntArray&>());

  nb::class_<MeshExtractionData>(m, "MeshExtractionData")
      .def_ro("scheme", &MeshExtractionData::scheme)
      .def_ro("orientation", &MeshExtractionData::orientation)
      .def_ro("material_ids", &MeshExtractionData::material_ids);

  m.def("extract_and_process_meshes", [](UsdStagePtr stage, const SdfPath& path,
                                         UsdTimeCode time, int refine_level,
                                         bool has_displacement) {
    MeshExtractionData mesh_data = ExtractMeshData(stage, path, time);
    SubdivisionEvaluator subdiv;
    HdMeshTopology topology = mesh_data.GetTopology();

    std::vector<VtIntArray> fvar_topologies;
    absl::flat_hash_map<TfToken, int, TfToken::HashFunctor> fvar_map;
    const TfToken fvar_interp_rule =
        topology.GetSubdivTags().GetFaceVaryingInterpolationRule();

    if (fvar_interp_rule != PxOsdOpenSubdivTokens->all && refine_level > 0 &&
        (mesh_data.scheme == PxOsdOpenSubdivTokens->catmullClark ||
         mesh_data.scheme == PxOsdOpenSubdivTokens->loop)) {
      for (const auto& [token, state] : mesh_data.primvars) {
        if (state.descriptor.interpolation == HdInterpolationFaceVarying) {
          VtIntArray indices;
          const int num_face_varyings = topology.GetNumFaceVaryings();
          indices.resize(num_face_varyings);
          std::iota(indices.begin(), indices.end(), 0);
          auto it = std::find(fvar_topologies.begin(), fvar_topologies.end(),
                              indices);
          int channel = 0;
          if (it == fvar_topologies.end()) {
            channel = fvar_topologies.size();
            fvar_topologies.push_back(indices);
          } else {
            channel = std::distance(fvar_topologies.begin(), it);
          }
          fvar_map[token] = channel;
        }
      }
    }

    subdiv.Initialize(topology.GetPxOsdMeshTopology(), refine_level,
                      mesh_data.scheme, topology.GetSubdivTags(),
                      fvar_topologies, fvar_map, mesh_data.id.GetNameToken());

    HdMeshTopology refined_topology = topology;
    PrimvarMap primvars = mesh_data.primvars;
    bool is_subdivided = subdiv.IsSubdivided();
    std::vector<int> refined_to_coarse_map;

    bool had_normals = primvars.find(HdTokens->normals) != primvars.end();
    if (is_subdivided) {
      // Normals will be recomputed after subdivision, skip refining them here.
      if (had_normals) {
        primvars.erase(HdTokens->normals);
      }
      for (auto& [token, state] : primvars) {
        state.value = subdiv.RefinePrimvar(
            state.value, state.descriptor.interpolation, token);
      }

      refined_topology = HdMeshTopology(mesh_data.scheme, mesh_data.orientation,
                                        subdiv.GetRefinedFaceVertexCounts(),
                                        subdiv.GetRefinedFaceVertexIndices());
      refined_to_coarse_map = subdiv.GetRefinedToCoarseMap();
    }
    bool needs_normals = has_displacement || (is_subdivided && had_normals);
    if (primvars.find(HdTokens->normals) == primvars.end() && needs_normals) {
      GeometryProcessor::ComputeNormals(primvars, refined_topology);
    }

    auto [face_indices, expanded] =
        GeometryProcessor::ExpandPrimData(refined_topology, primvars);
    auto [triangles, prim_params] =
        GeometryProcessor::TriangulateWithFaceMapping(
            refined_topology.GetFaceVertexCounts(), face_indices);

    VtIntArray mapped_material_indices;
    if (is_subdivided) {
      mapped_material_indices.resize(refined_to_coarse_map.size());
      for (size_t i = 0; i < refined_to_coarse_map.size(); ++i) {
        mapped_material_indices[i] =
            mesh_data.face_material_indices[refined_to_coarse_map[i]];
      }
    } else {
      mapped_material_indices = mesh_data.face_material_indices;
    }

    auto sub_meshes = GeometryProcessor::SplitAndCompactMeshes(
        mesh_data.id, triangles, prim_params, expanded, mesh_data.material_ids,
        mapped_material_indices);

    return std::make_pair(mesh_data, sub_meshes);
  });

  m.def("compute_normals",
        [](PrimvarMap& primvars, const HdMeshTopology& topology) {
          GeometryProcessor::ComputeNormals(primvars, topology);
          return primvars;
        });
  m.def("transform_primvars",
        [](PrimvarMap& primvars, const GfMatrix4d& transform) {
          GeometryProcessor::TransformPrimvars(primvars, transform);
          return primvars;
        });
}
