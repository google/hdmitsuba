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

#include <absl/container/flat_hash_set.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <mitsuba/render/interaction.h>
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

PXR_NAMESPACE_OPEN_SCOPE

// Evaluates a material's scalar displacement texture at each unique vertex of a
// face subset and offsets `primvars[HdTokens->points]` in place along the
// interpolated vertex normal:
//   p' = p + (displacement_texture.eval_1(uv) - 0.5) * n
//
// `vertex_indices` and `face_counts` describe the subset of faces bound to this
// material, while `global_face_indices` and `global_corner_indices` map each
// subset face/corner back to the full mesh topology so uniform and face-varying
// `st` / `normals` primvars are sampled at the original mesh indices. On JIT
// variants, all unique vertices in the subset are evaluated in a single
// vectorized Dr.Jit pass and migrated back to host memory.
template <typename Float, typename Spectrum>
void ApplyDisplacement(
    const SdfPath& mesh_id,
    const mitsuba::Texture<Float, Spectrum>* displacement_texture,
    const VtIntArray& vertex_indices, const VtIntArray& face_counts,
    const std::vector<int>& global_face_indices,
    const std::vector<int>& global_corner_indices, PrimvarMap& primvars) {
  TRACE_FUNCTION();
  if (!displacement_texture) return;

  namespace dr = drjit;
  using Vector2f = mitsuba::Vector<Float, 2>;
  using Vector3f = mitsuba::Vector<Float, 3>;
  using UInt32 = dr::uint32_array_t<Float>;

  // 1) Retrieve vector of UV coordinates Vec2f, normals and target vertex
  // index.
  VtVec2fArray uv_coords;
  VtVec3fArray normals;
  VtIntArray target_vertex_indices;
  absl::flat_hash_set<int> target_vertex_indices_set;
  auto uv_it = primvars.find(TfToken("st"));
  auto normal_it = primvars.find(HdTokens->normals);
  auto points_it = primvars.find(HdTokens->points);
  if (uv_it == primvars.end() || normal_it == primvars.end() ||
      points_it == primvars.end()) {
    TF_RUNTIME_ERROR("Missing required primvars for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  if (!uv_it->second.value.IsHolding<VtVec2fArray>() ||
      !normal_it->second.value.IsHolding<VtVec3fArray>() ||
      !points_it->second.value.IsHolding<VtVec3fArray>()) {
    TF_RUNTIME_ERROR("Invalid primvar types for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  const auto& uv_primvar = uv_it->second;
  const auto& normal_primvar = normal_it->second;
  VtVec3fArray points = points_it->second.value.Get<VtVec3fArray>();

  auto uv_interpolator =
      GeometryProcessor::GetInterpolator(uv_primvar.value.Get<VtVec2fArray>(),
                                         uv_primvar.descriptor.interpolation);
  auto normal_interpolator = GeometryProcessor::GetInterpolator(
      normal_primvar.value.Get<VtVec3fArray>(),
      normal_primvar.descriptor.interpolation);

  size_t corner = 0;
  const float bias = 0.5f;
  for (size_t face = 0; face < face_counts.size(); ++face) {
    for (int v = 0; v < face_counts[face]; ++v) {
      int vertex_index = vertex_indices[corner];
      // Each vertex should only be displaced once, even if it is part of
      // multiple faces.
      if (target_vertex_indices_set.contains(vertex_index)) {
        corner++;
        continue;
      }
      uv_coords.push_back(uv_interpolator(global_face_indices[face], corner,
                                          global_corner_indices[corner],
                                          vertex_indices));
      normals.push_back(normal_interpolator(global_face_indices[face], corner,
                                            global_corner_indices[corner],
                                            vertex_indices));
      target_vertex_indices.push_back(vertex_index);
      target_vertex_indices_set.insert(vertex_index);
      corner++;
    }
  }

  // 2) Query displacement and scatter add on vertices at target index.
  using FloatStorage = mitsuba::DynamicBuffer<dr::float32_array_t<Float>>;
  if constexpr (!dr::is_dynamic_v<Float>) {
    for (size_t i = 0; i < target_vertex_indices.size(); ++i) {
      int vertex_index = target_vertex_indices[i];
      GfVec2f uv = uv_coords[i];
      GfVec3f normal = normals[i];
      mitsuba::SurfaceInteraction<Float, Spectrum> si;
      si.uv = {uv[0], 1.0f - uv[1]};
      GfVec3f displacement = (displacement_texture->eval_1(si) - bias) * normal;
      points[vertex_index] += displacement;
    }
  } else {
    size_t n_vertices = target_vertex_indices.size();
    FloatStorage uv = dr::load<FloatStorage>(uv_coords.data(), n_vertices * 2);
    FloatStorage normal =
        dr::load<FloatStorage>(normals.data(), n_vertices * 3);
    UInt32 indices = dr::arange<UInt32>(n_vertices);
    Vector2f uv_vec = dr::gather<Vector2f>(uv, indices);
    mitsuba::SurfaceInteraction<Float, Spectrum> si;
    si.uv = {uv_vec[0], 1.0f - uv_vec[1]};
    Vector3f displacement = (displacement_texture->eval_1(si) - bias) *
                            dr::gather<Vector3f>(normal, indices);
    Float displacement_flat = dr::ravel(displacement);
    dr::eval(displacement_flat);
    auto&& host_displacement = dr::migrate(displacement_flat, JitBackend::None);
    dr::sync_thread();
    for (size_t i = 0; i < n_vertices; ++i) {
      GfVec3f offset(host_displacement[3 * i + 0], host_displacement[3 * i + 1],
                     host_displacement[3 * i + 2]);
      points[target_vertex_indices[i]] += offset;
    }
  }
  primvars[HdTokens->points].value = VtValue(std::move(points));
}

PXR_NAMESPACE_CLOSE_SCOPE
