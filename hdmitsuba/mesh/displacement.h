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

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <absl/types/span.h>
#include <drjit-core/jit.h>
#include <drjit/array.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/texture.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/prim_translator.h"

PXR_NAMESPACE_OPEN_SCOPE

// Evaluates the displacement texture of the faces of `mesh` with material
// `material_index` at their points that are not yet set in `displaced_points`,
// and offsets these object-space `points` (which supersede the "points"
// primvar of `mesh`) in place and sets their entries.
// Scalar displacement offsets the points along `normals` (one per point):
//   p' = p + (texture.eval_1(si) - displacement.bias) * n
// Vector displacement (`displacement.is_vector`) offsets the points by the 3D
// vector `texture.eval_3(si)` expressed in the tangent frame computed by
// `mitsuba::Mesh::tangents()`, or in an arbitrary orthonormal frame around the
// normal if the mesh has no UVs.
//
// `uv_mesh` is the mesh expanded by its "st" primvar (if any), so that each of
// its vertices has a single UV. A point on a UV seam is evaluated at the UV of
// its first vertex, and its tangent is averaged over all of its vertices, so
// that both sides of the seam are displaced identically. The texture is
// evaluated with the world-space position, normal and shading frame (using
// the object-to-world `transform`), so procedural displacement that does not
// require UVs (`displacement.requires_uv == false`) also works on meshes
// without "st". On JIT variants, all points are evaluated in a single
// vectorized Dr.Jit pass and migrated back to host memory.
template <typename Float, typename Spectrum>
void ApplyDisplacement(const SdfPath& mesh_id, const MeshGeometry& mesh,
                       const std::vector<bool>& is_hole,
                       const MaterialDisplacement& displacement,
                       int material_index, const ExpandedMesh& uv_mesh,
                       const VtVec3fArray& normals, const GfMatrix4d& transform,
                       std::vector<bool>& displaced_points,
                       VtVec3fArray& points) {
  TRACE_FUNCTION();
  namespace dr = drjit;
  using Texture = mitsuba::Texture<Float, Spectrum>;
  using Mesh = mitsuba::Mesh<Float, Spectrum>;
  using FloatBuffer = typename Mesh::FloatBuffer;
  using IndexBuffer = typename Mesh::IndexBuffer;
  using TensorXf32 = typename Mesh::TensorXf32;
  using TensorXu32 = typename Mesh::TensorXu32;
  using Vector2f = mitsuba::Vector<Float, 2>;
  using Vector3f = mitsuba::Vector<Float, 3>;
  using Frame = mitsuba::Frame<Float>;

  const auto* displacement_texture =
      dynamic_cast<const Texture*>(displacement.texture.get());
  if (!displacement_texture) return;
  auto uv_it = uv_mesh.primvars.find(TfToken("st"));
  const bool has_uv = uv_it != uv_mesh.primvars.end() &&
                      uv_it->second.IsHolding<VtVec2fArray>();
  if (displacement.requires_uv && !has_uv) {
    TF_RUNTIME_ERROR("Missing UVs for displacement on %s", mesh_id.GetText());
    return;
  }
  const VtVec2fArray& st =
      has_uv ? uv_it->second.UncheckedGet<VtVec2fArray>() : VtVec2fArray();
  // Vector displacement is expressed in the UV tangent frame if possible.
  const bool build_tangent_mesh = displacement.is_vector && has_uv;

  // 1) Collect the points to displace with the UV of their first vertex, and
  // the faces of the material in the vertices of the tangent mesh.
  std::vector<int> point_targets(points.size(), -1);
  std::vector<int> target_points;
  std::vector<GfVec2f> target_uvs;
  std::vector<int> mesh_vertices, vertex_to_mesh;
  VtIntArray mesh_face_counts, mesh_face_vertices;
  if (build_tangent_mesh) {
    vertex_to_mesh.resize(uv_mesh.vertex_points.size(), -1);
  }
  int corner = 0;
  for (size_t face = 0; face < mesh.face_vertex_counts.size(); ++face) {
    const int count = mesh.face_vertex_counts[face];
    if (is_hole[face] || mesh.face_material_indices[face] != material_index) {
      corner += count;
      continue;
    }
    for (int i = 0; i < count; ++i, ++corner) {
      const int v = uv_mesh.face_vertices[corner];
      const int p = uv_mesh.vertex_points[v];
      if (point_targets[p] < 0 && !displaced_points[p]) {
        point_targets[p] = static_cast<int>(target_points.size());
        target_points.push_back(p);
        target_uvs.push_back(has_uv ? st[v] : GfVec2f(0.f));
      }
      if (build_tangent_mesh) {
        if (vertex_to_mesh[v] < 0) {
          vertex_to_mesh[v] = static_cast<int>(mesh_vertices.size());
          mesh_vertices.push_back(v);
        }
        mesh_face_vertices.push_back(vertex_to_mesh[v]);
      }
    }
    if (build_tangent_mesh) mesh_face_counts.push_back(count);
  }
  const size_t n_targets = target_points.size();
  if (n_targets == 0) return;

  // 2) Object- and world-space positions, normals and tangents of the points.
  const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
  std::vector<GfVec3f> target_normals(n_targets), world_positions(n_targets),
      world_normals(n_targets);
  for (size_t i = 0; i < n_targets; ++i) {
    target_normals[i] = normals[target_points[i]].GetNormalized();
    world_positions[i] =
        GfVec3f(transform.TransformAffine(points[target_points[i]]));
    world_normals[i] = GfVec3f(
        normal_transform.TransformDir(target_normals[i]).GetNormalized());
  }
  FloatBuffer pos_buf =
                  dr::load<FloatBuffer>(world_positions.data(), n_targets * 3),
              norm_buf =
                  dr::load<FloatBuffer>(target_normals.data(), n_targets * 3),
              world_norm_buf =
                  dr::load<FloatBuffer>(world_normals.data(), n_targets * 3),
              uv_buf = dr::load<FloatBuffer>(target_uvs.data(), n_targets * 2);
  FloatBuffer tan_buf, world_tan_buf;
  if (build_tangent_mesh) {
    auto [triangles, _] = GeometryProcessor::TriangulateWithFaceMapping(
        mesh_face_counts, mesh_face_vertices, mesh.left_handed);
    if (triangles.empty()) return;
    const size_t n_mesh = mesh_vertices.size();
    std::vector<GfVec3f> mesh_positions(n_mesh), mesh_normals(n_mesh);
    std::vector<GfVec2f> mesh_uvs(n_mesh);
    for (size_t i = 0; i < n_mesh; ++i) {
      const int p = uv_mesh.vertex_points[mesh_vertices[i]];
      mesh_positions[i] = points[p];
      mesh_normals[i] = normals[p].GetNormalized();
      mesh_uvs[i] = st[mesh_vertices[i]];
    }
    mitsuba::ref<Mesh> mesh = new Mesh(
        mesh_id.GetString(),
        TensorXu32(dr::load<IndexBuffer>(triangles.data(), triangles.size()),
                   {triangles.size() / 3, 3}),
        TensorXf32(dr::load<FloatBuffer>(mesh_positions.data(), n_mesh * 3),
                   {n_mesh, 3}),
        TensorXf32(dr::load<FloatBuffer>(mesh_normals.data(), n_mesh * 3),
                   {n_mesh, 3}),
        TensorXf32(dr::load<FloatBuffer>(mesh_uvs.data(), n_mesh * 2),
                   {n_mesh, 2}));
    const FloatBuffer& host_tan =
        dr::migrate(mesh->tangents().array(), JitBackend::None);
    if constexpr (dr::is_jit_v<Float>) dr::sync_thread();

    // Average the tangents of the vertices of each point and re-orthogonalize
    // them against the normal.
    const float* tan_ptr = host_tan.data();
    std::vector<GfVec3f> obj_tan(n_targets, GfVec3f(0.f)), world_tan(n_targets);
    for (size_t i = 0; i < n_mesh; ++i) {
      if (const int target =
              point_targets[uv_mesh.vertex_points[mesh_vertices[i]]];
          target >= 0) {
        obj_tan[target] +=
            GfVec3f(tan_ptr[3 * i + 0], tan_ptr[3 * i + 1], tan_ptr[3 * i + 2]);
      }
    }
    for (size_t i = 0; i < n_targets; ++i) {
      const GfVec3f& n = target_normals[i];
      GfVec3f t = obj_tan[i] - n * GfDot(n, obj_tan[i]);
      if (const float length = t.GetLength(); length > 1e-10f) {
        t /= length;
      } else {
        const auto [s, _] = mitsuba::coordinate_system(
            mitsuba::Vector<float, 3>(n[0], n[1], n[2]));
        t = GfVec3f(s[0], s[1], s[2]);
      }
      obj_tan[i] = t;
      world_tan[i] = GfVec3f(transform.TransformDir(t));
    }
    tan_buf = dr::load<FloatBuffer>(obj_tan.data(), n_targets * 3);
    world_tan_buf = dr::load<FloatBuffer>(world_tan.data(), n_targets * 3);
  }

  // 3) Evaluate the displacement offsets and add them to the target points.
  // The offsets are computed in object space, while the texture is evaluated
  // in world space.
  using mitsuba::detail::deinterleave;
  using mitsuba::detail::interleaved;
  const float bias = displacement.bias;
  FloatBuffer offsets =
      interleaved<3, FloatBuffer>(n_targets, [&](const auto& i) -> Vector3f {
        Vector2f uv(deinterleave<2>(uv_buf, i));
        Vector3f n_obj(deinterleave<3>(norm_buf, i));
        Vector3f n_world(deinterleave<3>(world_norm_buf, i));
        auto si = dr::zeros<mitsuba::SurfaceInteraction<Float, Spectrum>>();
        si.uv = {uv[0], 1.0f - uv[1]};
        si.p = Vector3f(deinterleave<3>(pos_buf, i));
        si.n = n_world;
        si.prim_index = i;
        si.sh_frame = Frame(n_world);
        if (!displacement.is_vector) {
          return (displacement_texture->eval_1(si) - bias) * n_obj;
        }
        si.dp_du = si.sh_frame.s;
        si.dp_dv = si.sh_frame.t;
        Frame obj_frame(n_obj);
        if (build_tangent_mesh) {
          Vector3f t_obj(deinterleave<3>(tan_buf, i));
          obj_frame = Frame(t_obj, dr::cross(n_obj, t_obj), n_obj);
          // A non-uniform scale can shear the tangent away from the normal.
          Vector3f raw_t(deinterleave<3>(world_tan_buf, i));
          Vector3f t_world = raw_t - n_world * dr::dot(n_world, raw_t);
          si.dp_du = dr::select(dr::squared_norm(t_world) > 1e-20f,
                                dr::normalize(t_world), si.sh_frame.s);
          si.dp_dv = dr::cross(n_world, si.dp_du);
          si.sh_frame = Frame(si.dp_du, si.dp_dv, n_world);
        }
        return obj_frame.to_world(Vector3f(displacement_texture->eval_3(si)));
      });

  const FloatBuffer& host_offset = dr::migrate(offsets, JitBackend::None);
  if constexpr (dr::is_jit_v<Float>) dr::sync_thread();
  const float* offset_ptr = host_offset.data();
  for (size_t i = 0; i < n_targets; ++i) {
    points[target_points[i]] += GfVec3f(
        offset_ptr[3 * i + 0], offset_ptr[3 * i + 1], offset_ptr[3 * i + 2]);
    displaced_points[target_points[i]] = true;
  }
}

// Runs the full post-subdivision mesh preparation pipeline to produce triangle
// sub-meshes ready for `PrimTranslator::BuildMesh` / `UpdateMeshInPlace` or
// the `usd_mitsuba` Python translator. All steps but the last are performed in
// object space:
//   1. Applies per-material displacements (`displacements[i]` corresponds to
//      `mesh.material_ids[i]`) via `ApplyDisplacement`, along the authored
//      vertex normals if present, and along smooth geometric normals otherwise
//      (including for authored face-varying normals, which would displace the
//      vertices of hard edges along an arbitrary face normal). Points shared
//      by several displaced materials are displaced by the first one.
//   2. Computes smooth normals if there are no authored normals and the mesh
//      is subdivided or displaced. Authored vertex and face-varying normals of
//      displaced meshes are rotated along with the geometric normals, while
//      authored uniform or constant normals are replaced by the smooth
//      geometric normals.
//   3. Expands face-varying/uniform primvars into vertex-indexed buffers,
//      transforms them to world space by `transform`, triangulates polygons,
//      and splits/compacts the mesh into one `SubMeshOutput` per assigned
//      material.
//
// `requested_attributes` are the primvars read by the materials of the mesh.
// The following intrinsic primvars are generated if requested:
//   - `__p_object`: Object-space positions before displacement.
//   - `__n_object`: Object-space shading normals (after displacement), or the
//     face normals of meshes without normals.
//   - `__n_disp_object`: Object-space normals along which the mesh was
//     displaced (only on displaced meshes).
//   - `__n_disp_world`: World-space normals along which the mesh was displaced,
//     scaled by the inverse of the normal-to-world length scale, i.e. `u /
//     |u|^2` for the transformed normal `u` (only on displaced meshes).
template <typename Float, typename Spectrum>
std::vector<SubMeshOutput> RunGeometryPipeline(
    const SdfPath& id, const MeshGeometry& mesh, const GfMatrix4d& transform,
    absl::Span<const MaterialDisplacement> displacements,
    const absl::flat_hash_set<std::string>& requested_attributes) {
  TRACE_FUNCTION();
  static const TfToken kPObject("__p_object");
  static const TfToken kNObject("__n_object");
  static const TfToken kNDispObject("__n_disp_object");
  static const TfToken kNDispWorld("__n_disp_world");
  static const TfToken kSt("st");
  const TfToken& kPoints = HdTokens->points;
  const TfToken& kNormals = HdTokens->normals;
  const VtIntArray& counts = mesh.face_vertex_counts;
  const VtIntArray& indices = mesh.face_vertex_indices;

  PrimvarMap primvars = mesh.primvars;
  auto points_it = primvars.find(kPoints);
  if (points_it == primvars.end() ||
      !points_it->second.value.IsHolding<VtVec3fArray>()) {
    return {};
  }
  VtVec3fArray points = points_it->second.value.UncheckedGet<VtVec3fArray>();
  // Adds the intrinsic primvar `name` if it is requested.
  auto add_intrinsic = [&](const TfToken& name, VtValue value,
                           HdInterpolation interpolation) {
    if (requested_attributes.contains(name.GetString())) {
      primvars[name] = {std::move(value), interpolation};
    }
  };
  auto compute_normals = [&] {
    return GeometryProcessor::ComputeNormals(
        points, counts, indices, mesh.left_handed, mesh.hole_indices);
  };
  auto set_smooth_normals = [&] {
    primvars[kNormals] = {VtValue(compute_normals()), HdInterpolationVertex};
  };
  add_intrinsic(kPObject, VtValue(points), HdInterpolationVertex);

  std::vector<bool> is_hole(counts.size());
  for (const int face : mesh.hole_indices) {
    if (face >= 0 && static_cast<size_t>(face) < is_hole.size()) {
      is_hole[face] = true;
    }
  }
  bool has_displacement = false;
  for (size_t face = 0; face < counts.size(); ++face) {
    if (!is_hole[face] &&
        displacements[mesh.face_material_indices[face]].texture != nullptr) {
      has_displacement = true;
      break;
    }
  }

  auto normals_it = primvars.find(kNormals);
  PrimvarState* authored_normals =
      normals_it != primvars.end() &&
              normals_it->second.value.IsHolding<VtVec3fArray>()
          ? &normals_it->second
          : nullptr;
  if (has_displacement) {
    const VtVec3fArray geo_normals = compute_normals();
    const HdInterpolation interp = authored_normals
                                       ? authored_normals->interpolation
                                       : HdInterpolationConstant;
    const bool authored_per_point =
        (interp == HdInterpolationVertex || interp == HdInterpolationVarying) &&
        authored_normals->value.GetArraySize() == points.size();
    const bool authored_per_corner =
        interp == HdInterpolationFaceVarying &&
        authored_normals->value.GetArraySize() == indices.size();
    const VtVec3fArray disp_normals =
        authored_per_point
            ? authored_normals->value.UncheckedGet<VtVec3fArray>()
            : geo_normals;
    add_intrinsic(kNDispObject, VtValue(disp_normals), HdInterpolationVertex);
    if (requested_attributes.contains(kNDispWorld.GetString())) {
      const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
      VtVec3fArray world(disp_normals.size());
      for (size_t i = 0; i < world.size(); ++i) {
        const GfVec3d u = normal_transform.TransformDir(
            GfVec3d(disp_normals[i].GetNormalized()));
        world[i] = GfVec3f(u / u.GetLengthSq());
      }
      add_intrinsic(kNDispWorld, VtValue(std::move(world)),
                    HdInterpolationVertex);
    }

    PrimvarMap uv_primvars;
    if (auto st_it = primvars.find(kSt); st_it != primvars.end()) {
      uv_primvars.insert(*st_it);
    }
    const ExpandedMesh uv_mesh = GeometryProcessor::ExpandPrimData(
        counts, indices, points.size(), uv_primvars);
    std::vector<bool> displaced_points(points.size());
    for (size_t i = 0; i < mesh.material_ids.size(); ++i) {
      if (displacements[i].texture) {
        ApplyDisplacement<Float, Spectrum>(
            id, mesh, is_hole, displacements[i], static_cast<int>(i), uv_mesh,
            disp_normals, transform, displaced_points, points);
      }
    }
    primvars[kPoints].value = VtValue(points);

    if (authored_per_point || authored_per_corner) {
      // Rotate the authored normals by the minimal rotation from the geometric
      // normal before to the one after displacement, clamped to avoid the
      // singularity of (nearly) opposite normals.
      const VtVec3fArray geo_normals_after = compute_normals();
      VtVec3fArray normals =
          authored_normals->value.UncheckedGet<VtVec3fArray>();
      GfVec3f* normals_ptr = normals.data();
      for (size_t i = 0; i < normals.size(); ++i) {
        const int p = authored_per_corner ? indices[i] : static_cast<int>(i);
        const GfVec3f& n0 = geo_normals[p];
        const GfVec3f& n1 = geo_normals_after[p];
        const float c = GfDot(n0, n1);
        const GfVec3f k = GfCross(n0, n1);
        const GfVec3f& n = normals_ptr[i];
        normals_ptr[i] = (n * c + GfCross(k, n) +
                          k * (GfDot(k, n) / std::max(1.f + c, 1e-3f)))
                             .GetNormalized();
      }
      authored_normals->value = VtValue(std::move(normals));
    } else {
      set_smooth_normals();
    }
  } else if (!authored_normals && mesh.is_subdivided) {
    set_smooth_normals();
  }

  if (auto it = primvars.find(kNormals); it != primvars.end()) {
    add_intrinsic(kNObject, it->second.value, it->second.interpolation);
  } else if (requested_attributes.contains(kNObject.GetString())) {
    // Flat-shaded mesh: Provide the face normals as a uniform primvar, but
    // keep the mesh itself without normals.
    add_intrinsic(kNObject,
                  VtValue(GeometryProcessor::ComputeFaceNormals(
                      points, counts, indices, mesh.left_handed)),
                  HdInterpolationUniform);
  }

  // Expand primvar data, transform to world space, triangulate and split into
  // submeshes. A mirroring transform flips the winding of the polygons.
  ExpandedMesh expanded = GeometryProcessor::ExpandPrimData(
      counts, indices, points.size(), primvars);
  GeometryProcessor::TransformPrimvars(expanded.primvars, transform);
  auto [triangles, primitive_params] =
      GeometryProcessor::TriangulateWithFaceMapping(
          counts, expanded.face_vertices,
          mesh.left_handed ^ (transform.GetDeterminant() < 0.0),
          mesh.hole_indices);
  return GeometryProcessor::SplitAndCompactMeshes(
      id, triangles, primitive_params, expanded.primvars, mesh.material_ids,
      mesh.face_material_indices);
}

PXR_NAMESPACE_CLOSE_SCOPE
