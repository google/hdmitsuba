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
#include <map>
#include <string>
#include <tuple>
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

// Evaluates a material's displacement texture at each unique vertex of a face
// subset and offsets the object-space `primvars[HdTokens->points]` in place.
// Scalar displacement offsets the points along the interpolated vertex normal:
//   p' = p + (texture.eval_1(si) - displacement.bias) * n
// Vector displacement (`displacement.is_vector`) offsets the points by the 3D
// vector `texture.eval_3(si)` expressed in the vertex tangent frame computed
// by `mitsuba::Mesh::tangents()`, or in an arbitrary orthonormal frame around
// the normal if the mesh has no UVs.
//
// The texture is evaluated with the vertex UV (if any) and the world-space
// position, normal and shading frame (using the object-to-world `transform`),
// so procedural displacement that does not require UVs
// (`displacement.requires_uv == false`) also works on meshes without "st".
//
// `vertex_indices` and `face_counts` describe the subset of faces bound to this
// material, while `global_face_indices` and `global_corner_indices` map each
// subset face/corner back to the full mesh topology so uniform and face-varying
// `st` / `normals` primvars are sampled at the original mesh indices. On JIT
// variants, all unique vertices in the subset are evaluated in a single
// vectorized Dr.Jit pass and migrated back to host memory. Vertices whose
// `displaced_points` entry is set (as they are shared with the faces of a
// previously displaced material) are not displaced again, and the entries of
// the displaced vertices are set.
template <typename Float, typename Spectrum>
void ApplyDisplacement(const SdfPath& mesh_id,
                       const MaterialDisplacement& displacement,
                       const VtIntArray& vertex_indices,
                       const VtIntArray& face_counts,
                       const std::vector<int>& global_face_indices,
                       const std::vector<int>& global_corner_indices,
                       const GfMatrix4d& transform, bool left_handed,
                       std::vector<bool>& displaced_points,
                       PrimvarMap& primvars) {
  TRACE_FUNCTION();
  using Texture = mitsuba::Texture<Float, Spectrum>;
  const auto* displacement_texture =
      dynamic_cast<const Texture*>(displacement.texture.get());
  if (!displacement_texture) return;

  namespace dr = drjit;
  using Mesh = mitsuba::Mesh<Float, Spectrum>;
  using FloatBuffer = typename Mesh::FloatBuffer;
  using IndexBuffer = typename Mesh::IndexBuffer;
  using TensorXf32 = typename Mesh::TensorXf32;
  using TensorXu32 = typename Mesh::TensorXu32;
  using Vector2f = mitsuba::Vector<Float, 2>;
  using Vector3f = mitsuba::Vector<Float, 3>;
  using Frame = mitsuba::Frame<Float>;

  // 1) Retrieve the UV coordinates, positions and normals of the unique
  // vertices of the face subset.
  auto uv_it = primvars.find(TfToken("st"));
  auto normal_it = primvars.find(HdTokens->normals);
  auto points_it = primvars.find(HdTokens->points);
  const bool has_uv = uv_it != primvars.end();
  if ((displacement.requires_uv && !has_uv) || normal_it == primvars.end() ||
      points_it == primvars.end()) {
    TF_RUNTIME_ERROR("Missing required primvars for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  if ((has_uv && !uv_it->second.value.IsHolding<VtVec2fArray>()) ||
      !normal_it->second.value.IsHolding<VtVec3fArray>() ||
      !points_it->second.value.IsHolding<VtVec3fArray>()) {
    TF_RUNTIME_ERROR("Invalid primvar types for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  // Vector displacement is expressed in the UV tangent frame if possible.
  const bool build_tangent_mesh = displacement.is_vector && has_uv;
  const auto& normal_primvar = normal_it->second;
  VtVec3fArray points = points_it->second.value.Get<VtVec3fArray>();

  auto normal_interpolator = GeometryProcessor::GetInterpolator(
      normal_primvar.value.Get<VtVec3fArray>(),
      normal_primvar.descriptor.interpolation);
  // Without UVs, the interpolator of an empty array returns zero UVs.
  const VtVec2fArray no_uvs;
  auto uv_interpolator = GeometryProcessor::GetInterpolator(
      has_uv ? uv_it->second.value.UncheckedGet<VtVec2fArray>() : no_uvs,
      has_uv ? uv_it->second.descriptor.interpolation
             : HdInterpolationVertex);

  const size_t max_unique_vertices =
      std::min(points.size(), vertex_indices.size());

  std::vector<GfVec2f> uv_coords;
  std::vector<GfVec3f> positions, normals;
  std::vector<int> target_vertex_indices;
  uv_coords.reserve(max_unique_vertices);
  positions.reserve(max_unique_vertices);
  normals.reserve(max_unique_vertices);
  target_vertex_indices.reserve(max_unique_vertices);

  // Vertex indices of the tangent mesh. `mitsuba::Mesh::tangents()` requires a
  // single UV per vertex, so corners whose face-varying UV differs from the UV
  // of the (first visited) unique vertex reference a seam duplicate of it,
  // encoded as `-1 - seam_index`.
  VtIntArray local_vertex_indices;
  int* local_vi_ptr = nullptr;
  std::vector<int> seam_parent;
  std::vector<GfVec2f> seam_uvs;
  std::map<std::tuple<int, float, float>, int> seam_lookup;
  if (build_tangent_mesh) {
    local_vertex_indices.resize(vertex_indices.size());
    local_vi_ptr = local_vertex_indices.data();
  }

  std::vector<int> vertex_remap(points.size(), -1);
  int corner = 0;
  for (size_t face = 0; face < face_counts.size(); ++face) {
    const int face_count = face_counts[face];
    const int global_face = global_face_indices[face];
    for (int v = 0; v < face_count; ++v, ++corner) {
      const int vertex_index = vertex_indices[corner];
      const int global_corner = global_corner_indices[corner];
      const int remapped = vertex_remap[vertex_index];
      // Each vertex should only be displaced once, even if it is part of
      // multiple faces.
      if (remapped < 0) {
        vertex_remap[vertex_index] =
            static_cast<int>(target_vertex_indices.size());
        uv_coords.push_back(uv_interpolator(global_face, corner, global_corner,
                                            vertex_indices));
        positions.push_back(points.AsConst()[vertex_index]);
        normals.push_back(normal_interpolator(global_face, corner,
                                              global_corner, vertex_indices)
                              .GetNormalized());
        target_vertex_indices.push_back(vertex_index);
      } else if (build_tangent_mesh) {
        const GfVec2f corner_uv =
            uv_interpolator(global_face, corner, global_corner, vertex_indices);
        if (corner_uv != uv_coords[remapped]) {
          auto [it, inserted] = seam_lookup.try_emplace(
              std::make_tuple(remapped, corner_uv[0], corner_uv[1]),
              static_cast<int>(seam_parent.size()));
          if (inserted) {
            seam_parent.push_back(remapped);
            seam_uvs.push_back(corner_uv);
          }
          local_vi_ptr[corner] = -1 - it->second;
          continue;
        }
      }
      if (local_vi_ptr) {
        local_vi_ptr[corner] = vertex_remap[vertex_index];
      }
    }
  }

  const size_t n_vertices = target_vertex_indices.size();
  if (n_vertices == 0) return;
  FloatBuffer pos_buf = dr::load<FloatBuffer>(positions.data(), n_vertices * 3);
  FloatBuffer norm_buf = dr::load<FloatBuffer>(normals.data(), n_vertices * 3);
  FloatBuffer uv_buf = dr::load<FloatBuffer>(uv_coords.data(), n_vertices * 2);

  // World-space positions and normals at which the texture is evaluated.
  const bool has_transform = transform != GfMatrix4d(1.0);
  FloatBuffer world_pos_buf = pos_buf, world_norm_buf = norm_buf;
  if (has_transform) {
    const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
    std::vector<GfVec3f> world_positions(n_vertices), world_normals(n_vertices);
    for (size_t i = 0; i < n_vertices; ++i) {
      world_positions[i] = GfVec3f(transform.TransformAffine(positions[i]));
      world_normals[i] =
          GfVec3f(normal_transform.TransformDir(normals[i]).GetNormalized());
    }
    world_pos_buf =
        dr::load<FloatBuffer>(world_positions.data(), n_vertices * 3);
    world_norm_buf =
        dr::load<FloatBuffer>(world_normals.data(), n_vertices * 3);
  }

  FloatBuffer tan_buf, world_tan_buf;
  if (build_tangent_mesh) {
    const size_t n_seams = seam_parent.size();
    for (size_t i = 0; i < local_vertex_indices.size(); ++i) {
      if (local_vi_ptr[i] < 0) {
        local_vi_ptr[i] = static_cast<int>(n_vertices) + (-1 - local_vi_ptr[i]);
      }
    }
    // Append the seam duplicates (only the first `n_vertices` entries are used
    // beyond this point).
    for (size_t k = 0; k < n_seams; ++k) {
      positions.push_back(positions[seam_parent[k]]);
      normals.push_back(normals[seam_parent[k]]);
      uv_coords.push_back(seam_uvs[k]);
    }
    auto [triangles, _] = GeometryProcessor::TriangulateWithFaceMapping(
        face_counts, local_vertex_indices, left_handed);
    if (triangles.empty()) return;
    const size_t n_mesh_vertices = n_vertices + n_seams;
    mitsuba::ref<Mesh> mesh = new Mesh(
        mesh_id.GetString(),
        TensorXu32(dr::load<IndexBuffer>(triangles.data(), triangles.size()),
                   {triangles.size() / 3, 3}),
        TensorXf32(dr::load<FloatBuffer>(positions.data(), n_mesh_vertices * 3),
                   {n_mesh_vertices, 3}),
        TensorXf32(dr::load<FloatBuffer>(normals.data(), n_mesh_vertices * 3),
                   {n_mesh_vertices, 3}),
        TensorXf32(dr::load<FloatBuffer>(uv_coords.data(), n_mesh_vertices * 2),
                   {n_mesh_vertices, 2}));
    const FloatBuffer& host_tan =
        dr::migrate(mesh->tangents().array(), JitBackend::None);
    if constexpr (dr::is_jit_v<Float>) dr::sync_thread();

    // Average the tangents of the seam duplicates back onto their unique
    // vertex, so that both sides of a seam are displaced identically, and
    // re-orthogonalize them against the normal.
    const float* tan_ptr = host_tan.data();
    std::vector<GfVec3f> obj_tan(n_vertices, GfVec3f(0.f)),
        world_tan(n_vertices);
    for (size_t i = 0; i < n_mesh_vertices; ++i) {
      obj_tan[i < n_vertices ? i : seam_parent[i - n_vertices]] +=
          GfVec3f(tan_ptr[3 * i + 0], tan_ptr[3 * i + 1], tan_ptr[3 * i + 2]);
    }
    for (size_t i = 0; i < n_vertices; ++i) {
      const GfVec3f& n = normals[i];
      GfVec3f t = obj_tan[i] - n * GfDot(n, obj_tan[i]);
      if (const float length = t.GetLength(); length > 1e-10f) {
        t /= length;
      } else {
        const auto [s, _] = mitsuba::coordinate_system(
            mitsuba::Vector<float, 3>(n[0], n[1], n[2]));
        t = GfVec3f(s[0], s[1], s[2]);
      }
      obj_tan[i] = t;
      world_tan[i] = has_transform ? GfVec3f(transform.TransformDir(t)) : t;
    }
    tan_buf = dr::load<FloatBuffer>(obj_tan.data(), n_vertices * 3);
    world_tan_buf = dr::load<FloatBuffer>(world_tan.data(), n_vertices * 3);
  }

  // 2) Evaluate the displacement offsets and add them to the target vertices.
  // The offsets are computed in object space, while the texture is evaluated
  // in world space.
  using mitsuba::detail::deinterleave;
  using mitsuba::detail::interleaved;
  const float bias = displacement.bias;
  FloatBuffer offsets = interleaved<3, FloatBuffer>(
      n_vertices, [&](const auto& i) -> Vector3f {
        Vector2f uv(deinterleave<2>(uv_buf, i));
        Vector3f n_obj(deinterleave<3>(norm_buf, i));
        Vector3f n_world(deinterleave<3>(world_norm_buf, i));
        auto si = dr::zeros<mitsuba::SurfaceInteraction<Float, Spectrum>>();
        si.uv = {uv[0], 1.0f - uv[1]};
        si.p = Vector3f(deinterleave<3>(world_pos_buf, i));
        si.n = n_world;
        si.prim_index = i;
        if (displacement.is_vector) {
          Frame obj_frame(n_obj);
          if (build_tangent_mesh) {
            Vector3f t_obj(deinterleave<3>(tan_buf, i));
            obj_frame = Frame(t_obj, dr::cross(n_obj, t_obj), n_obj);
            // A non-uniform scale can shear the tangent away from the normal.
            Vector3f raw_t(deinterleave<3>(world_tan_buf, i));
            Vector3f t_world = raw_t - n_world * dr::dot(n_world, raw_t);
            t_world = dr::select(dr::squared_norm(t_world) > 1e-20f,
                                 dr::normalize(t_world),
                                 mitsuba::coordinate_system(n_world).first);
            Vector3f b_world = dr::cross(n_world, t_world);
            si.dp_du = t_world;
            si.dp_dv = b_world;
            si.sh_frame = Frame(t_world, b_world, n_world);
          } else {
            si.sh_frame = Frame(n_world);
            si.dp_du = si.sh_frame.s;
            si.dp_dv = si.sh_frame.t;
          }
          return obj_frame.to_world(Vector3f(displacement_texture->eval_3(si)));
        }
        si.sh_frame = Frame(n_world);
        return (displacement_texture->eval_1(si) - bias) * n_obj;
      });

  const FloatBuffer& host_offset = dr::migrate(offsets, JitBackend::None);
  if constexpr (dr::is_jit_v<Float>) dr::sync_thread();
  const float* offset_ptr = host_offset.data();
  GfVec3f* points_ptr = points.data();
  for (size_t i = 0; i < n_vertices; ++i) {
    const int vertex_index = target_vertex_indices[i];
    if (displaced_points[vertex_index]) continue;
    displaced_points[vertex_index] = true;
    points_ptr[vertex_index] += GfVec3f(
        offset_ptr[3 * i + 0], offset_ptr[3 * i + 1], offset_ptr[3 * i + 2]);
  }
  primvars[HdTokens->points].value = VtValue(std::move(points));
}

// Runs the full post-subdivision mesh preparation pipeline to produce triangle
// sub-meshes ready for `PrimTranslator::BuildMesh` / `UpdateMeshInPlace` or
// the `usd_mitsuba` Python translator:
//   1. Computes smooth vertex normals if missing and needed by subdivision or
//      displacement.
//   2. Applies per-material displacements (`displacements[i]` corresponds to
//      `material_ids[i]`) in object space via `ApplyDisplacement`. Meshes
//      are displaced along their vertex normals, i.e. along authored vertex
//      normals if present, and along smooth geometric normals otherwise
//      (including for authored face-varying normals, which would displace the
//      vertices of hard edges along an arbitrary face normal). Authored
//      vertex and face-varying normals of unsubdivided meshes are then
//      rotated along with the geometric normals, while authored uniform or
//      constant normals are replaced by the smooth geometric normals.
//   3. If any displacement occurred, transforms points and normals by
//      `transform` and recomputes smooth vertex normals (unless authored
//      normals were preserved).
//   4. Expands face-varying/uniform primvars into vertex-indexed buffers (and,
//      for non-displaced meshes, transforms the deduplicated primvars),
//      triangulates polygons, and splits/compacts the mesh into one
//      `SubMeshOutput` per assigned material.
//
// `requested_attributes` are the primvars read by the materials of the mesh.
// The following intrinsic primvars are generated if requested:
//   - `__p_object`: Object-space positions before displacement.
//   - `__n_object`: Object-space shading normals (after displacement).
//   - `__n_disp_object`: Object-space normals along which the mesh was
//     displaced (only on displaced meshes).
//   - `__n_disp_world`: World-space normals along which the mesh was displaced,
//     scaled by the inverse of the normal-to-world length scale (only on
//     displaced meshes, see `GeometryProcessor::TransformPrimvars`).
template <typename Float, typename Spectrum>
std::vector<SubMeshOutput> RunGeometryPipeline(
    const SdfPath& id, const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices, bool left_handed, bool is_subdivided,
    const GfMatrix4d& transform, absl::Span<const SdfPath> material_ids,
    const VtIntArray& face_material_indices, const PrimvarMap& input_primvars,
    absl::Span<const MaterialDisplacement> displacements,
    const absl::flat_hash_set<std::string>& requested_attributes) {
  TRACE_FUNCTION();
  static const TfToken kPObject("__p_object");
  static const TfToken kNObject("__n_object");
  static const TfToken kNDispObject("__n_disp_object");
  static const TfToken kNDispWorld("__n_disp_world");
  const TfToken& kNormals = HdTokens->normals;
  const TfToken& kNoRole = HdPrimvarRoleTokens->none;

  // A mirroring transform flips the winding of the transformed polygons.
  const bool effective_left_handed =
      left_handed ^ (transform.GetDeterminant() < 0.0);
  const bool want_n_obj = requested_attributes.contains(kNObject.GetString());
  const bool has_displacement =
      std::any_of(displacements.begin(), displacements.end(),
                  [](const auto& d) { return d.texture != nullptr; });

  PrimvarMap final_primvars = input_primvars;
  // Copies `final_primvars[src]` (if present) to `final_primvars[name]`.
  auto copy_primvar = [&](const TfToken& src, const TfToken& name,
                          const TfToken& role) {
    auto it = final_primvars.find(src);
    if (it == final_primvars.end()) return;
    PrimvarState state = it->second;
    state.descriptor.name = name;
    state.descriptor.role = role;
    final_primvars[name] = std::move(state);
  };
  if (requested_attributes.contains(kPObject.GetString())) {
    copy_primvar(HdTokens->points, kPObject, kNoRole);
  }
  auto normals_it = final_primvars.find(kNormals);
  const bool preserve_authored_normals =
      has_displacement && !is_subdivided &&
      normals_it != final_primvars.end() &&
      normals_it->second.value.IsHolding<VtVec3fArray>() &&
      final_primvars.count(HdTokens->points) > 0;
  PrimvarState authored_normals;
  bool authored_per_vertex = false, authored_per_corner = false;
  if (preserve_authored_normals) {
    authored_normals = std::move(normals_it->second);
    authored_normals.descriptor.role = HdPrimvarRoleTokens->normal;
    final_primvars.erase(normals_it);
    const HdInterpolation interp = authored_normals.descriptor.interpolation;
    const size_t n_normals = authored_normals.value.GetArraySize();
    authored_per_vertex =
        (interp == HdInterpolationVertex || interp == HdInterpolationVarying) &&
        n_normals == final_primvars.at(HdTokens->points).value.GetArraySize();
    authored_per_corner = interp == HdInterpolationFaceVarying &&
                          n_normals == face_vertex_indices.size();
  }
  if (final_primvars.count(kNormals) == 0 &&
      (has_displacement || is_subdivided)) {
    GeometryProcessor::ComputeNormals(final_primvars, face_vertex_indices,
                                      face_vertex_counts, left_handed);
  }
  VtVec3fArray geo_normals_before;
  if (preserve_authored_normals) {
    geo_normals_before =
        final_primvars.at(kNormals).value.UncheckedGet<VtVec3fArray>();
  }
  if (authored_per_vertex) {
    final_primvars[kNormals] = authored_normals;
  }
  if (has_displacement) {
    if (requested_attributes.contains(kNDispObject.GetString())) {
      copy_primvar(kNormals, kNDispObject, kNoRole);
    }
    if (requested_attributes.contains(kNDispWorld.GetString())) {
      copy_primvar(kNormals, kNDispWorld, kNoRole);
    }
  }

  bool displaced = false;
  // Points shared by several displaced materials are displaced only once, by
  // the first material.
  std::vector<bool> displaced_points;
  if (auto it = final_primvars.find(HdTokens->points);
      has_displacement && it != final_primvars.end()) {
    displaced_points.resize(it->second.value.GetArraySize());
  }
  for (size_t i = 0; i < material_ids.size(); ++i) {
    if (!displacements[i].texture) continue;
    VtIntArray material_vertex_indices;
    VtIntArray material_face_counts;
    std::vector<int> global_face_indices;
    std::vector<int> global_corner_indices;
    int corner_index = 0;
    for (size_t face_index = 0; face_index < face_vertex_counts.size();
         ++face_index) {
      const int material_index = face_material_indices[face_index];
      const int vertex_count = face_vertex_counts[face_index];
      if (static_cast<size_t>(material_index) == i) {
        global_face_indices.push_back(face_index);
        for (int j = 0; j < vertex_count; ++j) {
          global_corner_indices.push_back(corner_index + j);
          material_vertex_indices.push_back(
              face_vertex_indices[corner_index + j]);
        }
        material_face_counts.push_back(vertex_count);
      }
      corner_index += vertex_count;
    }
    if (!material_vertex_indices.empty()) {
      ApplyDisplacement<Float, Spectrum>(
          id, displacements[i], material_vertex_indices, material_face_counts,
          global_face_indices, global_corner_indices, transform, left_handed,
          displaced_points, final_primvars);
      displaced = true;
    }
  }

  bool restored_authored_normals = false;
  if (displaced) {
    GeometryProcessor::ComputeNormals(final_primvars, face_vertex_indices,
                                      face_vertex_counts, left_handed);
    if (authored_per_vertex || authored_per_corner) {
      const VtVec3fArray& geo_normals_after =
          final_primvars.at(kNormals).value.UncheckedGet<VtVec3fArray>();
      VtVec3fArray norms = authored_normals.value.UncheckedGet<VtVec3fArray>();
      GfVec3f* norms_ptr = norms.data();
      for (size_t i = 0; i < norms.size(); ++i) {
        const int v =
            authored_per_corner ? face_vertex_indices[i] : static_cast<int>(i);
        // Rotate by the minimal rotation from the geometric normal before
        // to the one after displacement, clamped to avoid the singularity
        // of (nearly) opposite normals.
        const GfVec3f& n0 = geo_normals_before[v];
        const GfVec3f& n1 = geo_normals_after[v];
        const float c = GfDot(n0, n1);
        const GfVec3f k = GfCross(n0, n1);
        const GfVec3f& n = norms_ptr[i];
        norms_ptr[i] = (n * c + GfCross(k, n) +
                        k * (GfDot(k, n) / std::max(1.f + c, 1e-3f)))
                           .GetNormalized();
      }
      authored_normals.value = VtValue(std::move(norms));
      final_primvars[kNormals] = std::move(authored_normals);
      restored_authored_normals = true;
    }
  } else if (preserve_authored_normals) {
    final_primvars[kNormals] = std::move(authored_normals);
  }

  if (want_n_obj && final_primvars.count(kNormals) > 0) {
    copy_primvar(kNormals, kNObject, kNoRole);
  } else if (want_n_obj && final_primvars.count(HdTokens->points) > 0) {
    // Flat-shaded mesh: Provide the face normals as a uniform primvar, but
    // keep the mesh itself without normals.
    const VtVec3fArray& points =
        final_primvars.at(HdTokens->points).value.UncheckedGet<VtVec3fArray>();
    VtVec3fArray face_normals(face_vertex_counts.size(), GfVec3f(0.f));
    int index = 0;
    for (size_t face = 0; face < face_vertex_counts.size(); ++face) {
      const int count = face_vertex_counts[face];
      GfVec3f n(0.f);
      for (int j = 1; j + 1 < count; ++j) {
        const GfVec3f& p0 = points[face_vertex_indices[index]];
        n += GfCross(points[face_vertex_indices[index + j]] - p0,
                     points[face_vertex_indices[index + j + 1]] - p0);
      }
      n.Normalize();
      face_normals[face] = left_handed ? -n : n;
      index += count;
    }
    final_primvars[kNObject] = {
        VtValue(std::move(face_normals)),
        HdPrimvarDescriptor(kNObject, HdInterpolationUniform, kNoRole)};
  }

  if (displaced) {
    GeometryProcessor::TransformPrimvars(final_primvars, transform);
    if (transform != GfMatrix4d(1.0) && !restored_authored_normals) {
      GeometryProcessor::ComputeNormals(final_primvars, face_vertex_indices,
                                        face_vertex_counts,
                                        effective_left_handed);
    }
  }

  // Expand primvar data, triangulate and split into submeshes.
  auto [face_indices, expanded_primvars] = GeometryProcessor::ExpandPrimData(
      face_vertex_indices, face_vertex_counts, final_primvars);
  if (!displaced) {
    GeometryProcessor::TransformPrimvars(expanded_primvars, transform);
  }
  auto [triangles, primitive_params] =
      GeometryProcessor::TriangulateWithFaceMapping(
          face_vertex_counts, face_indices, effective_left_handed);
  return GeometryProcessor::SplitAndCompactMeshes(
      id, triangles, primitive_params, expanded_primvars, material_ids,
      face_material_indices);
}

PXR_NAMESPACE_CLOSE_SCOPE
