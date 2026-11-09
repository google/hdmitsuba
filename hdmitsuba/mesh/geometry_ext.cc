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

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>
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

#include "hdmitsuba/ies_emitter.h"
#include "hdmitsuba/mesh/displacement.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/mesh/subdivision.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/utils.h"
#include "nanobind/usd.h"

NANOBIND_BOOST_CASTER(pxr::VtIntArray, "Vt.IntArray");
NANOBIND_BOOST_CASTER(pxr::VtVec3fArray, "Vt.Vec3fArray");
NANOBIND_BOOST_CASTER(pxr::VtVec2fArray, "Vt.Vec2fArray");
NANOBIND_BOOST_CASTER(pxr::GfMatrix4d, "Gf.Matrix4d");

namespace nb = nanobind;
PXR_NAMESPACE_USING_DIRECTIVE

// The coarse mesh geometry of a UsdGeomMesh with its subdivision settings.
// The material ids are the paths of the GeomSubsets, and the empty path for
// the faces not covered by any subset.
struct MeshExtractionData : public MeshGeometry {
  SdfPath id;
  TfToken scheme;
  TfToken interp_boundary = PxOsdOpenSubdivTokens->edgeAndCorner;
  TfToken face_varying_interp = PxOsdOpenSubdivTokens->cornersPlus1;
  // Unflattened values and indices (empty if not indexed) of faceVarying
  // primvars, which are subdivided using their own face-varying topology.
  absl::flat_hash_map<TfToken, VtValue, TfToken::HashFunctor>
      unflattened_fvar_values;
  absl::flat_hash_map<TfToken, VtIntArray, TfToken::HashFunctor> fvar_indices;

  PxOsdMeshTopology GetTopology() const {
    PxOsdSubdivTags tags(interp_boundary, face_varying_interp, TfToken(),
                         TfToken(),
                         /*creaseIndices=*/VtIntArray(),
                         /*creaseLengths=*/VtIntArray(),
                         /*creaseWeights=*/VtFloatArray(),
                         /*cornerIndices=*/VtIntArray(),
                         /*cornerWeights=*/VtFloatArray());
    return PxOsdMeshTopology(scheme,
                             left_handed ? PxOsdOpenSubdivTokens->leftHanded
                                         : PxOsdOpenSubdivTokens->rightHanded,
                             face_vertex_counts, face_vertex_indices,
                             VtIntArray(), tags);
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
      GfVec3f* pts_ptr = points.data();
      const size_t n = points.size();
      for (size_t i = 0; i < n; ++i) {
        pts_ptr[i] = GfVec3f(skel_to_prim_local.Transform(pts_ptr[i]));
      }
    }
    if (skinned_normals) {
      GfMatrix3d skel_to_prim_local_inv_transpose =
          skel_to_prim_local.ExtractRotationMatrix()
              .GetInverse()
              .GetTranspose();
      GfVec3f* nrm_ptr = normals.data();
      const size_t n = normals.size();
      for (size_t i = 0; i < n; ++i) {
        nrm_ptr[i] = GfVec3f(nrm_ptr[i] * skel_to_prim_local_inv_transpose);
      }
    }
  }
  return {std::move(points), std::move(normals)};
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
  TfToken orientation;
  mesh.GetOrientationAttr().Get(&orientation);
  data.left_handed = orientation == UsdGeomTokens->leftHanded;
  if (UsdAttribute attr = mesh.GetInterpolateBoundaryAttr()) {
    attr.Get(&data.interp_boundary);
  }
  if (UsdAttribute attr = mesh.GetFaceVaryingLinearInterpolationAttr()) {
    attr.Get(&data.face_varying_interp);
  }

  UsdGeomPrimvarsAPI primvars_api(mesh);
  for (const UsdGeomPrimvar& primvar : primvars_api.GetPrimvars()) {
    const TfToken name = primvar.GetPrimvarName();
    const HdInterpolation interpolation =
        UsdInterpolationToHdInterpolation(primvar.GetInterpolation());
    VtValue value;
    if (primvar.ComputeFlattened(&value, time) && !value.IsEmpty()) {
      data.primvars[name] = {std::move(value), interpolation};
      if (interpolation == HdInterpolationFaceVarying) {
        primvar.Get(&data.unflattened_fvar_values[name], time);
        primvar.GetIndices(&data.fvar_indices[name], time);
      }
    }
  }

  auto [points, normals] = GetMeshPointsAndNormals(prim, mesh, time);

  if (!points.empty()) {
    data.primvars[HdTokens->points] = {VtValue(std::move(points)),
                                       HdInterpolationVertex};
  }

  if (!normals.empty()) {
    data.primvars[HdTokens->normals] = {
        VtValue(std::move(normals)),
        UsdInterpolationToHdInterpolation(mesh.GetNormalsInterpolation())};
  }

  std::vector<std::pair<SdfPath, VtIntArray>> subsets;
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
            subsets.emplace_back(child.GetPath(), std::move(indices));
          }
        }
      }
    }
  }
  GeometryProcessor::AssignFaceMaterials(subsets, SdfPath(), data);
  return data;
}

// Returns the name of the active Mitsuba variant.
std::string ActiveVariant() {
  return nb::cast<std::string>(
      nb::module_::import_("mitsuba").attr("variant")());
}

// Subdivides `mesh_data` and runs `RunGeometryPipeline` on it, see
// `process_mesh` below.
std::vector<SubMeshOutput> ProcessMesh(
    const MeshExtractionData& mesh_data, int refine_level,
    const GfMatrix4d& transform,
    const std::vector<std::tuple<nb::object, float, bool, bool>>& displacements,
    const std::set<std::string>& requested_attributes) {
  SubdivisionEvaluator subdiv;
  subdiv.Initialize(mesh_data.GetTopology(), refine_level,
                    mesh_data.fvar_indices, mesh_data.id.GetNameToken());
  MeshGeometry geometry = mesh_data;
  if (subdiv.IsSubdivided()) {
    // Normals will be recomputed after subdivision, skip refining them here.
    geometry.primvars.erase(HdTokens->normals);
    for (auto& [token, state] : geometry.primvars) {
      // FaceVarying primvars are refined on their unflattened values.
      if (subdiv.HasFVarChannel(token)) {
        state.value = mesh_data.unflattened_fvar_values.at(token);
      }
      state.value =
          subdiv.RefinePrimvar(state.value, state.interpolation, token);
    }
    subdiv.RefineTopology(geometry);
  }

  std::vector<MaterialDisplacement> material_displacements;
  material_displacements.reserve(displacements.size());
  for (const auto& [texture, bias, requires_uv, is_vector] : displacements) {
    MaterialDisplacement& displacement = material_displacements.emplace_back();
    if (!texture.is_none()) {
      displacement.texture = reinterpret_cast<mitsuba::Object*>(
          nb::cast<uintptr_t>(texture.attr("ptr")));
    }
    displacement.bias = bias;
    displacement.requires_uv = requires_uv;
    displacement.is_vector = is_vector;
  }
  const absl::flat_hash_set<std::string> requested(requested_attributes.begin(),
                                                   requested_attributes.end());

  // Displacement textures are evaluated in the active Mitsuba variant.
  using mitsuba::ClassName;
  using mitsuba::Color;
  using mitsuba::Error;
  using mitsuba::Spectrum;
  namespace dr = drjit;
  return MI_INVOKE_VARIANT(ActiveVariant(), RunGeometryPipeline, mesh_data.id,
                           geometry, transform, material_displacements,
                           requested);
}

template <typename Float, typename Spectrum>
void AttachMeshAttributes(mitsuba::Object* mesh,
                          const VertexPrimvarMap& primvars,
                          const MeshAttributeRequests& mesh_attributes) {
  PrimTranslator<Float, Spectrum>::AttachMeshAttributes(mesh, primvars,
                                                        mesh_attributes);
}

// Adds the requested primvars of `sub` as vertex attributes to the Mitsuba
// mesh `mesh` of the active variant, see `add_mesh_attributes` below.
void AddMeshAttributes(const nb::object& mesh, const SubMeshOutput& sub,
                       const MeshAttributeRequests& mesh_attributes) {
  using mitsuba::ClassName;
  using mitsuba::Color;
  using mitsuba::Error;
  using mitsuba::Spectrum;
  namespace dr = drjit;
  MI_INVOKE_VARIANT(
      ActiveVariant(), AttachMeshAttributes,
      reinterpret_cast<mitsuba::Object*>(nb::cast<uintptr_t>(mesh.attr("ptr"))),
      sub.primvars, mesh_attributes);
}

NB_MODULE(geometry_ext, m) {
  PXR_NS::RegisterIesEmitter();
  m.doc() = "Python bindings for hdmitsuba geometry processing utilities.";

  nb::class_<SubMeshOutput>(m, "SubMeshOutput")
      .def_ro("material_id", &SubMeshOutput::material_id)
      .def_ro("triangles", &SubMeshOutput::triangles)
      .def_ro("primvars", &SubMeshOutput::primvars);

  nb::class_<MeshExtractionData>(m, "MeshExtractionData")
      .def_ro("material_ids", &MeshExtractionData::material_ids);

  m.def("add_mesh_attributes", &AddMeshAttributes, nb::arg("mesh"),
        nb::arg("sub"), nb::arg("mesh_attributes"),
        "Adds the `(primvar name, channel count)` pairs `mesh_attributes` of "
        "the sub-mesh `sub` that exist to the Mitsuba mesh `mesh` of the "
        "active variant as vertex attributes, like the Hydra delegate.");
  m.def("extract_mesh_data", &ExtractMeshData, nb::arg("stage"),
        nb::arg("path"), nb::arg("time"),
        "Extracts the topology, primvars and GeomSubset material assignment "
        "of the UsdGeomMesh at `path`.");
  m.def("process_mesh", &ProcessMesh, nb::arg("mesh_data"),
        nb::arg("refine_level"), nb::arg("transform"), nb::arg("displacements"),
        nb::arg("requested_attributes"),
        "Subdivides, displaces, transforms to world space, triangulates and "
        "splits `mesh_data` into one sub-mesh per material.\n\n"
        "`displacements` holds one `(texture, bias, requires_uv, is_vector)` "
        "tuple per entry of `mesh_data.material_ids`, where `texture` is a "
        "Mitsuba texture of the active variant or None. "
        "`requested_attributes` are the primvar names read by the materials, "
        "which may include the intrinsic `__p_object`, `__n_object`, "
        "`__n_disp_object` and `__n_disp_world` primvars.");
  m.def(
      "sample_world_transform",
      [](const UsdPrim& prim, UsdTimeCode time,
         std::pair<float, float> interval, bool uniform) {
        return SampleTransform(
                   prim, time, GfVec2f(interval.first, interval.second), uniform)
            .samples;
      },
      nb::arg("prim"), nb::arg("time"), nb::arg("interval"),
      nb::arg("uniform") = false);
  m.def("sample_point_instancer_transforms",
        [](const UsdPrim& prim, UsdTimeCode time,
           std::pair<float, float> interval) {
          std::vector<std::vector<std::pair<float, GfMatrix4d>>> out;
          for (auto& s : SamplePointInstancerTransforms(
                   prim, time, GfVec2f(interval.first, interval.second))) {
            out.push_back(std::move(s.samples));
          }
          return out;
        });
}
