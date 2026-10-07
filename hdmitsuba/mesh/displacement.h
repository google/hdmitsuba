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

#include <cstddef>
#include <utility>
#include <vector>

#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/texture.h>
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
// subset and offsets `primvars[HdTokens->points]` in place. Scalar
// displacement offsets the points along the interpolated vertex normal:
//   p' = p + (texture.eval_1(si) - displacement.bias) * n
// Vector displacement (`displacement.is_vector`) offsets the points by the 3D
// vector `texture.eval_3(si)` expressed in the vertex tangent frame computed
// by `mitsuba::Mesh::tangents()`.
//
// The texture is evaluated with the vertex UV (if any), object-space position
// and normal, so procedural displacement that does not require UVs
// (`displacement.requires_uv == false`) also works on meshes without "st".
//
// `vertex_indices` and `face_counts` describe the subset of faces bound to this
// material, while `global_face_indices` and `global_corner_indices` map each
// subset face/corner back to the full mesh topology so uniform and face-varying
// `st` / `normals` primvars are sampled at the original mesh indices. On JIT
// variants, all unique vertices in the subset are evaluated in a single
// vectorized Dr.Jit pass and migrated back to host memory.
template <typename Float, typename Spectrum>
void ApplyDisplacement(const SdfPath& mesh_id,
                       const MaterialDisplacement& displacement,
                       const VtIntArray& vertex_indices,
                       const VtIntArray& face_counts,
                       const std::vector<int>& global_face_indices,
                       const std::vector<int>& global_corner_indices,
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

  // 1) Retrieve the UV coordinates, positions and normals of the unique
  // vertices of the face subset.
  auto uv_it = primvars.find(TfToken("st"));
  auto normal_it = primvars.find(HdTokens->normals);
  auto points_it = primvars.find(HdTokens->points);
  const bool has_uv = uv_it != primvars.end();
  if (((displacement.requires_uv || displacement.is_vector) && !has_uv) ||
      normal_it == primvars.end() || points_it == primvars.end()) {
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

  VtVec2fArray uv_coords;
  VtVec3fArray positions, normals;
  VtIntArray target_vertex_indices, local_vertex_indices;
  std::vector<int> vertex_remap(points.size(), -1);
  size_t corner = 0;
  for (size_t face = 0; face < face_counts.size(); ++face) {
    for (int v = 0; v < face_counts[face]; ++v, ++corner) {
      const int vertex_index = vertex_indices[corner];
      // Each vertex should only be displaced once, even if it is part of
      // multiple faces.
      if (vertex_remap[vertex_index] < 0) {
        vertex_remap[vertex_index] = target_vertex_indices.size();
        uv_coords.push_back(uv_interpolator(global_face_indices[face], corner,
                                            global_corner_indices[corner],
                                            vertex_indices));
        positions.push_back(points[vertex_index]);
        normals.push_back(
            normal_interpolator(global_face_indices[face], corner,
                                global_corner_indices[corner], vertex_indices)
                .GetNormalized());
        target_vertex_indices.push_back(vertex_index);
      }
      local_vertex_indices.push_back(vertex_remap[vertex_index]);
    }
  }

  const size_t n_vertices = target_vertex_indices.size();
  FloatBuffer pos_buf = dr::load<FloatBuffer>(positions.data(), n_vertices * 3);
  FloatBuffer norm_buf = dr::load<FloatBuffer>(normals.data(), n_vertices * 3);
  FloatBuffer uv_buf = dr::load<FloatBuffer>(uv_coords.data(), n_vertices * 2);
  FloatBuffer tan_buf;
  if (displacement.is_vector) {
    auto [triangles, _] = GeometryProcessor::TriangulateWithFaceMapping(
        face_counts, local_vertex_indices);
    if (triangles.empty()) return;
    mitsuba::ref<Mesh> mesh = new Mesh(
        mesh_id.GetString(),
        TensorXu32(dr::load<IndexBuffer>(triangles.data(), triangles.size()),
                   {triangles.size() / 3, 3}),
        TensorXf32(pos_buf, {n_vertices, 3}),
        TensorXf32(norm_buf, {n_vertices, 3}),
        TensorXf32(uv_buf, {n_vertices, 2}));
    tan_buf = mesh->tangents().array();
  }

  // 2) Evaluate the displacement offsets and add them to the target vertices.
  using mitsuba::detail::deinterleave;
  using mitsuba::detail::interleaved;
  const float bias = displacement.bias;
  FloatBuffer offsets = interleaved<3, FloatBuffer>(
      n_vertices, [&](const auto& i) -> Vector3f {
        Vector2f uv(deinterleave<2>(uv_buf, i));
        Vector3f p(deinterleave<3>(pos_buf, i));
        Vector3f n(deinterleave<3>(norm_buf, i));
        auto si = dr::zeros<mitsuba::SurfaceInteraction<Float, Spectrum>>();
        si.uv = {uv[0], 1.0f - uv[1]};
        si.p = p;
        si.n = n;
        if (displacement.is_vector) {
          Vector3f t(deinterleave<3>(tan_buf, i));
          Vector3f b = dr::cross(n, t);
          si.dp_du = t;
          si.dp_dv = b;
          si.sh_frame = mitsuba::Frame<Float>(t, b, n);
          return si.sh_frame.to_world(
              Vector3f(displacement_texture->eval_3(si)));
        }
        si.sh_frame = mitsuba::Frame<Float>(n);
        return (displacement_texture->eval_1(si) - bias) * n;
      });

  const FloatBuffer& host_offset = dr::migrate(offsets, JitBackend::None);
  if constexpr (dr::is_jit_v<Float>) dr::sync_thread();
  for (size_t i = 0; i < n_vertices; ++i) {
    points[target_vertex_indices[i]] +=
        GfVec3f(host_offset[3 * i + 0], host_offset[3 * i + 1],
                host_offset[3 * i + 2]);
  }
  primvars[HdTokens->points].value = VtValue(std::move(points));
}

PXR_NAMESPACE_CLOSE_SCOPE
