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
#include <absl/strings/match.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/ar/resolverContextBinder.h>
#include <pxr/usd/ar/resolverScopedCache.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/utils.h>
#include <pxr/usdImaging/usdImaging/materialParamUtils.h>

#include "hdmitsuba/ies_emitter.h"
#include "hdmitsuba/materialx/translator.h"
#include "hdmitsuba/materialx/vm.h"
#include "hdmitsuba/mesh/displacement.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/mesh/subdivision.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/texture_cache.h"
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
  TfToken triangle_subdiv = PxOsdOpenSubdivTokens->catmullClark;
  VtIntArray crease_indices, crease_lengths, corner_indices;
  VtFloatArray crease_sharpnesses, corner_sharpnesses;
  // Unflattened values and indices (empty if not indexed) of faceVarying
  // primvars, which are subdivided using their own face-varying topology.
  absl::flat_hash_map<TfToken, VtValue, TfToken::HashFunctor>
      unflattened_fvar_values;
  absl::flat_hash_map<TfToken, VtIntArray, TfToken::HashFunctor> fvar_indices;

  PxOsdMeshTopology GetTopology() const {
    PxOsdSubdivTags tags(interp_boundary, face_varying_interp, TfToken(),
                         triangle_subdiv, crease_indices, crease_lengths,
                         crease_sharpnesses, corner_indices,
                         corner_sharpnesses);
    return PxOsdMeshTopology(scheme,
                             left_handed ? PxOsdOpenSubdivTokens->leftHanded
                                         : PxOsdOpenSubdivTokens->rightHanded,
                             face_vertex_counts, face_vertex_indices,
                             hole_indices, tags);
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
  data.is_smooth_surface = IsSmoothScheme(data.scheme);
  TfToken orientation;
  mesh.GetOrientationAttr().Get(&orientation);
  data.left_handed = orientation == UsdGeomTokens->leftHanded;
  mesh.GetInterpolateBoundaryAttr().Get(&data.interp_boundary, time);
  mesh.GetFaceVaryingLinearInterpolationAttr().Get(&data.face_varying_interp,
                                                   time);
  mesh.GetTriangleSubdivisionRuleAttr().Get(&data.triangle_subdiv, time);
  mesh.GetHoleIndicesAttr().Get(&data.hole_indices, time);
  mesh.GetCreaseIndicesAttr().Get(&data.crease_indices, time);
  mesh.GetCreaseLengthsAttr().Get(&data.crease_lengths, time);
  mesh.GetCreaseSharpnessesAttr().Get(&data.crease_sharpnesses, time);
  mesh.GetCornerIndicesAttr().Get(&data.corner_indices, time);
  mesh.GetCornerSharpnessesAttr().Get(&data.corner_sharpnesses, time);

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
    const std::vector<std::tuple<nb::object, bool, bool>>& displacements,
    const std::set<std::string>& requested_attributes) {
  const auto points_it = mesh_data.primvars.find(HdTokens->points);
  const std::optional<size_t> num_referenced_points =
      GeometryProcessor::ValidateTopology(mesh_data.id,
                                          mesh_data.face_vertex_counts,
                                          mesh_data.face_vertex_indices);
  if (!num_referenced_points ||
      !GeometryProcessor::ValidatePointCount(
          mesh_data.id, *num_referenced_points,
          points_it == mesh_data.primvars.end()
              ? 0
              : points_it->second.value.GetArraySize())) {
    return {};
  }
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
  for (const auto& [texture, requires_uv, is_vector] : displacements) {
    MaterialDisplacement& displacement = material_displacements.emplace_back();
    if (!texture.is_none()) {
      displacement.texture = reinterpret_cast<mitsuba::Object*>(
          nb::cast<uintptr_t>(texture.attr("ptr")));
    }
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

namespace {

nb::object CastMitsubaObject(mitsuba::Object* obj) {
  if (!obj) return nb::none();
  using Caster = nb::object (*)(mitsuba::Object*);
  return reinterpret_cast<Caster>(
      nb::borrow<nb::capsule>(
          nb::module_::import_("mitsuba.mitsuba_ext").attr("cast_object"))
          .data())(obj);
}

HdMaterialNetwork2 BuildMaterialNetwork(const UsdPrim& prim, UsdTimeCode time) {
  UsdShadeMaterial material(prim);
  if (!material) return {};
  static const TfTokenVector kContexts = {TfToken("mitsuba"), TfToken("mtlx"),
                                          TfToken("")};
  static const TfTokenVector kSourceTypes = {TfToken("mtlx"), TfToken("")};
  HdMaterialNetwork2 net;
  auto add_terminal = [&](const UsdShadeShader& shader,
                          const TfToken& terminal) {
    if (!shader) return;
    TfToken id;
    if (shader.GetShaderId(&id) &&
        absl::StartsWith(id.GetString(), "mitsuba_")) {
      return;
    }
    // Build each terminal in isolation so an SdrRegistry miss on one terminal
    // does not discard valid nodes from the other.
    HdMaterialNetworkMap network_map;
    UsdImagingBuildHdMaterialNetworkFromTerminal(shader.GetPrim(), terminal,
                                                 kSourceTypes, kContexts,
                                                 &network_map, time);
    bool is_volume = false;
    HdMaterialNetwork2 term_net =
        HdConvertToHdMaterialNetwork2(network_map, &is_volume);
    net.nodes.merge(std::move(term_net.nodes));
    net.terminals.merge(std::move(term_net.terminals));
  };
  add_terminal(material.ComputeSurfaceSource(kContexts),
               HdMaterialTerminalTokens->surface);
  return net;
}

template <typename Float, typename Spectrum>
nb::object TranslateMaterialXImpl(const SdfPath& id,
                                  const HdMaterialNetwork2& net,
                                  bool double_sided) {
  absl::flat_hash_set<TextureKey> keys;
  DiscoverMaterialXTextures(net, [&](const TextureKey& k) { keys.insert(k); });
  TextureCache<Float, Spectrum> texture_cache;
  texture_cache.Preload(keys);
  TranslatedMaterial trans = TranslateMaterialXNetwork(
      id, net, [&](const TextureKey& k) { return texture_cache.Find(k); },
      mitsuba::BSDF<Float, Spectrum>::Variant);
  mitsuba::ref<mitsuba::Object> emitter;
  if (trans.shape_emitter_props) {
    if (double_sided) {
      trans.shape_emitter_props->set("twosided", true, false);
    }
    emitter = mitsuba::PluginManager::instance()->create_object(
        *trans.shape_emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
        mitsuba::Emitter<Float, Spectrum>::Type);
  }
  return nb::make_tuple(CastMitsubaObject(trans.bsdf.get()),
                        CastMitsubaObject(emitter.get()),
                        trans.mesh_attributes);
}

}  // namespace

NB_MODULE(geometry_ext, m) {
  PXR_NS::RegisterIesEmitter();
  mitsuba::mtlx::RegisterMaterialXExprTexture();
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
  m.def("mesh_attribute_name", &MeshAttributeName, nb::arg("primvar"),
        nb::arg("channels"),
        "Returns the name of the Mitsuba vertex attribute holding `primvar` "
        "with `channels` channels, as read by a `mesh_attribute` texture.");
  m.def("extract_mesh_data", &ExtractMeshData, nb::arg("stage"),
        nb::arg("path"), nb::arg("time"),
        "Extracts the topology, primvars and GeomSubset material assignment "
        "of the UsdGeomMesh at `path`.");
  m.def("process_mesh", &ProcessMesh, nb::arg("mesh_data"),
        nb::arg("refine_level"), nb::arg("transform"), nb::arg("displacements"),
        nb::arg("requested_attributes"),
        "Subdivides, displaces, transforms to world space, triangulates and "
        "splits `mesh_data` into one sub-mesh per material.\n\n"
        "`displacements` holds one `(texture, requires_uv, is_vector)` "
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
  m.def(
      "translate_materialx",
      [](const UsdPrim& prim, UsdTimeCode time,
         bool double_sided) -> nb::object {
        if (!prim || !prim.GetStage()) return nb::none();
        ArResolverContextBinder binder(
            prim.GetStage()->GetPathResolverContext());
        ArResolverScopedCache resolver_cache;
        HdMaterialNetwork2 net = BuildMaterialNetwork(prim, time);
        if (!IsMaterialXNetwork(net)) return nb::none();
        using mitsuba::ClassName;
        using mitsuba::Color;
        using mitsuba::Error;
        using mitsuba::Spectrum;
        namespace dr = drjit;
        return MI_INVOKE_VARIANT(ActiveVariant(), TranslateMaterialXImpl,
                                 prim.GetPath(), net, double_sided);
      },
      nb::arg("prim"), nb::arg("time"), nb::arg("double_sided") = false,
      "Translates a MaterialX UsdShadeMaterial prim into a "
      "`(bsdf, emitter, mesh_attributes)` tuple, or None if the material has "
      "no MaterialX nodes.");
}
