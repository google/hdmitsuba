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

#include "hdmitsuba/mesh/geometry_processor.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_replace.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/types/span.h>
#include <drjit/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/debug_codes.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using PrimvarArrayVariant = std::variant<VtFloatArray, VtIntArray, VtVec2fArray,
                                         VtVec3fArray, VtVec4fArray>;

template <typename T>
constexpr size_t PrimvarDimension() {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
    return 1;
  } else {
    return T::dimension;
  }
}

struct ExpandChannel {
  TfToken name;
  HdInterpolation interpolation;
  PrimvarArrayVariant src;
};

// Flat view of a FaceVarying or Uniform channel used to build dedup keys.
struct VaryingChannelView {
  // Raw float bits of a FaceVarying channel, or nullptr for Uniform channels
  // (keyed by face index).
  const int* data = nullptr;
  size_t dim = 0;
};

std::optional<PrimvarArrayVariant> ExtractPrimvarArray(
    const VtValue& value, bool allow_scalar = false) {
  if (value.IsHolding<VtVec3fArray>()) {
    return value.UncheckedGet<VtVec3fArray>();
  }
  if (value.IsHolding<VtVec2fArray>()) {
    return value.UncheckedGet<VtVec2fArray>();
  }
  if (value.IsHolding<VtFloatArray>()) {
    return value.UncheckedGet<VtFloatArray>();
  }
  if (value.IsHolding<VtVec4fArray>()) {
    return value.UncheckedGet<VtVec4fArray>();
  }
  if (value.IsHolding<VtIntArray>()) {
    return value.UncheckedGet<VtIntArray>();
  }
  if (value.IsHolding<VtBoolArray>()) {
    const VtBoolArray& src = value.UncheckedGet<VtBoolArray>();
    return VtIntArray(src.begin(), src.end());
  }
  if (!allow_scalar) return std::nullopt;
  if (value.IsHolding<GfVec3f>()) {
    return VtVec3fArray{value.UncheckedGet<GfVec3f>()};
  }
  if (value.IsHolding<GfVec2f>()) {
    return VtVec2fArray{value.UncheckedGet<GfVec2f>()};
  }
  if (value.IsHolding<float>()) {
    return VtFloatArray{value.UncheckedGet<float>()};
  }
  if (value.IsHolding<GfVec4f>()) {
    return VtVec4fArray{value.UncheckedGet<GfVec4f>()};
  }
  if (value.IsHolding<int>()) {
    return VtIntArray{value.UncheckedGet<int>()};
  }
  if (value.IsHolding<bool>()) {
    return VtIntArray{static_cast<int>(value.UncheckedGet<bool>())};
  }
  return std::nullopt;
}

template <typename T>
VtValue GatherVertices(const VtArray<T>& src,
                       absl::Span<const int> unique_old_vertices) {
  VtArray<T> out(unique_old_vertices.size());
  T* dst_ptr = out.data();
  for (size_t i = 0; i < unique_old_vertices.size(); ++i) {
    dst_ptr[i] = src[unique_old_vertices[i]];
  }
  return VtValue(std::move(out));
}

std::vector<bool> HoleMask(int num_faces, absl::Span<const int> hole_indices) {
  std::vector<bool> is_hole(num_faces);
  for (const int face : hole_indices) {
    if (face >= 0 && face < num_faces) is_hole[face] = true;
  }
  return is_hole;
}

}  // namespace

std::pair<VtIntArray, VtIntArray> GeometryProcessor::TriangulateWithFaceMapping(
    const VtIntArray& face_vertex_counts, const VtIntArray& face_vertex_indices,
    bool left_handed, absl::Span<const int> hole_indices) {
  TRACE_FUNCTION();
  const int num_faces = static_cast<int>(face_vertex_counts.size());
  const std::vector<bool> is_hole = HoleMask(num_faces, hole_indices);
  auto num_triangles = [&](int face) {
    return is_hole[face] ? 0 : std::max(face_vertex_counts[face] - 2, 0);
  };

  int total_triangles = 0;
  for (int face = 0; face < num_faces; ++face) {
    total_triangles += num_triangles(face);
  }
  VtIntArray triangles(total_triangles * 3);
  VtIntArray primitive_params(total_triangles);
  int* tri_ptr = triangles.data();
  int* param_ptr = primitive_params.data();

  int index = 0;
  int tri_idx = 0;
  for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
    const int count = face_vertex_counts[face_idx];
    if (num_triangles(face_idx) > 0) {
      const int v0 = face_vertex_indices[index];
      for (int i = 0; i < count - 2; ++i) {
        tri_ptr[tri_idx * 3 + 0] = v0;
        tri_ptr[tri_idx * 3 + 1] =
            face_vertex_indices[index + i + (left_handed ? 2 : 1)];
        tri_ptr[tri_idx * 3 + 2] =
            face_vertex_indices[index + i + (left_handed ? 1 : 2)];
        param_ptr[tri_idx] = face_idx;
        ++tri_idx;
      }
    }
    index += count;
  }
  return {std::move(triangles), std::move(primitive_params)};
}

VtVec3fArray GeometryProcessor::ComputeNormals(
    const VtVec3fArray& points, const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices, bool left_handed,
    absl::Span<const int> hole_indices) {
  TRACE_FUNCTION();
  const int num_faces = static_cast<int>(face_vertex_counts.size());
  const std::vector<bool> is_hole = HoleMask(num_faces, hole_indices);
  VtVec3fArray normals(points.size(), GfVec3f(0.0f, 0.0f, 0.0f));
  GfVec3f* normals_ptr = normals.data();
  int index = 0;
  for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
    const int count = face_vertex_counts[face_idx];
    if (!is_hole[face_idx]) {
      for (int i = 0; i + 2 < count; ++i) {
        GfVec3i face(face_vertex_indices[index],
                     face_vertex_indices[index + i + 1],
                     face_vertex_indices[index + i + 2]);
        GfVec3f p[3] = {points[face[0]], points[face[1]], points[face[2]]};
        GfVec3f face_normal = GfCross(p[1] - p[0], p[2] - p[0]).GetNormalized();
        if (left_handed) face_normal = -face_normal;
        for (int j = 0; j < 3; ++j) {
          GfVec3f d0 = (p[(j + 1) % 3] - p[j]).GetNormalized();
          GfVec3f d1 = (p[(j + 2) % 3] - p[j]).GetNormalized();
          float face_angle = drjit::safe_acos(GfDot(d0, d1));
          normals_ptr[face[j]] += face_normal * face_angle;
        }
      }
    }
    index += count;
  }
  for (GfVec3f& n : normals) {
    n.Normalize();
  }
  return normals;
}

VtVec3fArray GeometryProcessor::ComputeFaceNormals(
    const VtVec3fArray& points, const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices, bool left_handed) {
  VtVec3fArray face_normals(face_vertex_counts.size());
  int index = 0;
  for (size_t face = 0; face < face_vertex_counts.size(); ++face) {
    const int count = face_vertex_counts[face];
    const GfVec3f& p0 = points[face_vertex_indices[index]];
    GfVec3f n(0.f);
    for (int j = 1; j + 1 < count; ++j) {
      n += GfCross(points[face_vertex_indices[index + j]] - p0,
                   points[face_vertex_indices[index + j + 1]] - p0);
    }
    n.Normalize();
    face_normals[face] = left_handed ? -n : n;
    index += count;
  }
  return face_normals;
}

void GeometryProcessor::TransformPrimvars(VertexPrimvarMap& primvars,
                                          const GfMatrix4d& transform) {
  // Replaces each value of the vec3 primvar `name` (if any) by `fn(value)`.
  auto transform_primvar = [&](const TfToken& name, auto fn) {
    auto it = primvars.find(name);
    if (it == primvars.end() || !it->second.IsHolding<VtVec3fArray>()) {
      return;
    }
    VtVec3fArray values = it->second.UncheckedGet<VtVec3fArray>();
    for (GfVec3f& value : values) value = fn(value);
    it->second = VtValue(std::move(values));
  };
  if (transform != GfMatrix4d(1.0)) {
    transform_primvar(HdTokens->points, [&](const GfVec3f& p) {
      return GfVec3f(transform.TransformAffine(p));
    });
  }
  const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
  transform_primvar(HdTokens->normals, [&](const GfVec3f& n) {
    return GfVec3f(normal_transform.TransformDir(n).GetNormalized());
  });
}

ExpandedMesh GeometryProcessor::ExpandPrimData(
    const VtIntArray& face_vertex_counts, const VtIntArray& face_vertex_indices,
    size_t num_points, const PrimvarMap& primvars) {
  TRACE_FUNCTION();
  absl::Time start = absl::Now();

  std::vector<ExpandChannel> channels;
  channels.reserve(primvars.size());
  for (const auto& [token, state] : primvars) {
    auto src_opt = ExtractPrimvarArray(
        state.value, state.interpolation == HdInterpolationConstant);
    if (!src_opt) continue;
    channels.push_back({token, state.interpolation, std::move(*src_opt)});
  }

  // The vertex of a corner is determined by its point and the values of the
  // FaceVarying and Uniform channels, which form the key of the vertex.
  std::vector<VaryingChannelView> varying_channels;
  size_t key_stride = 0;
  for (const auto& channel : channels) {
    const HdInterpolation interp = channel.interpolation;
    if (interp == HdInterpolationUniform) {
      varying_channels.push_back({nullptr, 1});
      key_stride += 1;
    } else if (interp == HdInterpolationFaceVarying) {
      std::visit(
          [&](const auto& arr) {
            // Empty channels are dropped from the output and cannot
            // distinguish corners, so they do not contribute to the key.
            if (arr.empty()) return;
            constexpr size_t dim = PrimvarDimension<
                typename std::decay_t<decltype(arr)>::value_type>();
            varying_channels.push_back(
                {reinterpret_cast<const int*>(arr.cdata()), dim});
            key_stride += dim;
          },
          channel.src);
    }
  }

  const size_t num_corners = face_vertex_indices.size();
  ExpandedMesh mesh;
  mesh.face_vertices.resize(num_corners);
  mesh.vertex_points.reserve(num_corners);
  int* face_vertices_ptr = mesh.face_vertices.data();
  // Face and corner of the first corner of each vertex.
  struct SourceLoc {
    int face;
    int corner;
  };
  std::vector<SourceLoc> unique_locs;
  unique_locs.reserve(num_corners);
  // The vertices of each point form a linked list, which is short in practice
  // (typically 1-2 entries).
  std::vector<int> first_vertex_for_point(num_points, -1);
  std::vector<int> next_vertex;
  if (key_stride > 0) next_vertex.reserve(num_corners);
  // Candidate keys are written in place at the slot of the next unique vertex,
  // so accepting a new vertex requires no copy.
  std::vector<int> vertex_keys(num_corners * key_stride);
  int corner = 0;
  for (int face = 0; face < static_cast<int>(face_vertex_counts.size());
       ++face) {
    for (int i = 0; i < face_vertex_counts[face]; ++i, ++corner) {
      const int point = face_vertex_indices[corner];
      int v = first_vertex_for_point[point];
      if (key_stride > 0) {
        int* candidate_key =
            vertex_keys.data() + unique_locs.size() * key_stride;
        int* key_ptr = candidate_key;
        for (const VaryingChannelView& ch : varying_channels) {
          if (ch.data != nullptr) {
            std::memcpy(key_ptr, ch.data + corner * ch.dim,
                        ch.dim * sizeof(int));
            key_ptr += ch.dim;
          } else {
            *key_ptr++ = face;
          }
        }
        while (v >= 0 &&
               std::memcmp(
                   vertex_keys.data() + static_cast<size_t>(v) * key_stride,
                   candidate_key, key_stride * sizeof(int)) != 0) {
          v = next_vertex[v];
        }
      }
      if (v < 0) {
        v = static_cast<int>(unique_locs.size());
        unique_locs.push_back({face, corner});
        mesh.vertex_points.push_back(point);
        if (key_stride > 0)
          next_vertex.push_back(first_vertex_for_point[point]);
        first_vertex_for_point[point] = v;
      }
      face_vertices_ptr[corner] = v;
    }
  }
  TF_DEBUG(HDMITSUBA_GEOMETRY)
      .Msg("Compressed to %zu / %zu vertices. Time: %f ms\n",
           unique_locs.size(), num_corners,
           absl::ToDoubleMilliseconds(absl::Now() - start));

  const size_t num_unique = unique_locs.size();
  for (auto& channel : channels) {
    const HdInterpolation interp = channel.interpolation;
    std::visit(
        [&](const auto& src) {
          using ArrayType = std::decay_t<decltype(src)>;
          if (src.empty() || num_unique == 0) return;
          ArrayType dst_array(num_unique);
          auto* dst_ptr = dst_array.data();
          switch (interp) {
            case HdInterpolationConstant:
              std::fill_n(dst_ptr, num_unique, src[0]);
              break;
            case HdInterpolationUniform:
              for (size_t i = 0; i < num_unique; ++i) {
                dst_ptr[i] = src[unique_locs[i].face];
              }
              break;
            case HdInterpolationVertex:
            case HdInterpolationVarying:
              for (size_t i = 0; i < num_unique; ++i) {
                dst_ptr[i] = src[mesh.vertex_points[i]];
              }
              break;
            case HdInterpolationFaceVarying:
              for (size_t i = 0; i < num_unique; ++i) {
                dst_ptr[i] = src[unique_locs[i].corner];
              }
              break;
            default:
              break;
          }
          mesh.primvars[channel.name] = VtValue(std::move(dst_array));
        },
        channel.src);
  }
  return mesh;
}

std::vector<SubMeshOutput> GeometryProcessor::SplitAndCompactMeshes(
    const SdfPath& id, const VtIntArray& triangles,
    const VtIntArray& primitive_params, const VertexPrimvarMap& final_primvars,
    absl::Span<const SdfPath> material_ids,
    const VtIntArray& face_material_indices) {
  TRACE_FUNCTION();
  std::vector<SubMeshOutput> sub_meshes;
  if (triangles.empty()) return sub_meshes;

  // The single material case is trivial.
  if (material_ids.size() == 1) {
    SubMeshOutput out;
    out.id = id.GetAsString();
    out.material_id = material_ids[0];
    out.triangles = triangles;
    out.primvars = final_primvars;
    sub_meshes.push_back(std::move(out));
    return sub_meshes;
  }

  // Split the triangles into sub-meshes by material index.
  std::vector<std::vector<int>> material_indices(material_ids.size());
  for (size_t i = 0; i < triangles.size() / 3; ++i) {
    const int face_index = primitive_params[i];
    const int material_index = face_material_indices[face_index];
    auto& mat_tris = material_indices[material_index];
    mat_tris.push_back(triangles[i * 3]);
    mat_tris.push_back(triangles[i * 3 + 1]);
    mat_tris.push_back(triangles[i * 3 + 2]);
  }

  auto points_it = final_primvars.find(HdTokens->points);
  const size_t total_source_vertices =
      points_it != final_primvars.end()
          ? points_it->second.Get<VtVec3fArray>().size()
          : 0;

  std::vector<int> old_to_new_vertex_map(total_source_vertices, -1);
  std::vector<int> unique_old_vertices;

  for (size_t i = 0; i < material_ids.size(); ++i) {
    if (material_indices[i].empty()) continue;
    unique_old_vertices.clear();

    size_t max_vertices =
        std::min(material_indices[i].size(), total_source_vertices > 0
                                                 ? total_source_vertices
                                                 : material_indices[i].size());
    unique_old_vertices.reserve(max_vertices);

    const auto& src_indices = material_indices[i];
    VtIntArray submesh_triangles(src_indices.size());
    int* sub_tri_ptr = submesh_triangles.data();

    // Compact the triangles and record unique source vertex indices.
    for (size_t k = 0; k < src_indices.size(); ++k) {
      const int vertex_index = src_indices[k];
      int mapped = old_to_new_vertex_map[vertex_index];
      if (mapped < 0) {
        mapped = static_cast<int>(unique_old_vertices.size());
        old_to_new_vertex_map[vertex_index] = mapped;
        unique_old_vertices.push_back(vertex_index);
      }
      sub_tri_ptr[k] = mapped;
    }

    for (int old_v : unique_old_vertices) {
      old_to_new_vertex_map[old_v] = -1;
    }

    VertexPrimvarMap primvars;
    for (const auto& [token, value] : final_primvars) {
      if (auto src_opt = ExtractPrimvarArray(value)) {
        primvars[token] = std::visit(
            [&](const auto& src) {
              return GatherVertices(src, unique_old_vertices);
            },
            *src_opt);
      }
    }

    SubMeshOutput out;
    out.id = MakeSubMeshId(id, material_ids[i], i);
    out.material_id = material_ids[i];
    out.triangles = std::move(submesh_triangles);
    out.primvars = std::move(primvars);
    sub_meshes.push_back(std::move(out));
  }

  return sub_meshes;
}

void GeometryProcessor::AssignFaceMaterials(
    absl::Span<const std::pair<SdfPath, VtIntArray>> subsets,
    const SdfPath& mesh_material, MeshGeometry& mesh) {
  auto find_or_add_material = [&](const SdfPath& material_id) -> int {
    auto it = std::find(mesh.material_ids.begin(), mesh.material_ids.end(),
                        material_id);
    if (it != mesh.material_ids.end()) {
      return static_cast<int>(std::distance(mesh.material_ids.begin(), it));
    }
    mesh.material_ids.push_back(material_id);
    return static_cast<int>(mesh.material_ids.size()) - 1;
  };
  const int num_faces = static_cast<int>(mesh.face_vertex_counts.size());
  mesh.material_ids.clear();
  mesh.face_material_indices.assign(num_faces, -1);
  for (const auto& [material_id, face_indices] : subsets) {
    const int material_index = find_or_add_material(material_id);
    for (const int face : face_indices) {
      if (face >= 0 && face < num_faces) {
        mesh.face_material_indices[face] = material_index;
      }
    }
  }
  if (mesh.material_ids.empty() ||
      std::find(mesh.face_material_indices.cbegin(),
                mesh.face_material_indices.cend(),
                -1) != mesh.face_material_indices.cend()) {
    std::replace(mesh.face_material_indices.begin(),
                 mesh.face_material_indices.end(), -1,
                 find_or_add_material(mesh_material));
  }
}

std::string GeometryProcessor::MakeSubMeshId(const SdfPath& mesh_id,
                                             const SdfPath& material_id,
                                             size_t material_index) {
  std::string mat_name =
      material_id.IsEmpty()
          ? "mat_" + std::to_string(material_index)
          : absl::StrReplaceAll(material_id.GetAsString(),
                                {{"/", "_"}, {":", "_"}});
  return absl::StrCat(mesh_id.GetAsString(), "#", mat_name);
}

PXR_NAMESPACE_CLOSE_SCOPE
