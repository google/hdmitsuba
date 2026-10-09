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
#include <absl/container/inlined_vector.h>
#include <absl/strings/str_replace.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/types/span.h>
#include <drjit/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/debug_codes.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

constexpr bool kUseMeshCompression = true;

using VertexDataKey = absl::InlinedVector<int, 8>;

inline int AsInt(float f) {
  int i;
  std::memcpy(&i, &f, sizeof(f));
  return i;
}

using PrimvarArrayVariant =
    std::variant<VtFloatArray, VtVec2fArray, VtVec3fArray>;

template <typename T>
constexpr size_t PrimvarDimension() {
  if constexpr (std::is_same_v<T, float>) {
    return 1;
  } else {
    return T::dimension;
  }
}

inline void AppendValueToKey(VertexDataKey& key, float val) {
  key.push_back(AsInt(val));
}

template <typename Vec>
inline void AppendValueToKey(VertexDataKey& key, const Vec& val) {
  for (size_t i = 0; i < Vec::dimension; ++i) {
    key.push_back(AsInt(val[i]));
  }
}

template <typename T>
T SamplePrimvar(const VtArray<T>& data, HdInterpolation interpolation, int face,
                int corner, const VtIntArray& face_indices) {
  if (data.empty()) return T(0.f);
  switch (interpolation) {
    case HdInterpolationConstant:
      return data[0];
    case HdInterpolationUniform:
      return data[face];
    case HdInterpolationVertex:
    case HdInterpolationVarying:
      return data[face_indices[corner]];
    case HdInterpolationFaceVarying:
      return data[corner];
    default:
      return T(0.f);
  }
}

struct ExpandChannel {
  TfToken name;
  HdPrimvarDescriptor descriptor;
  PrimvarArrayVariant src;
};

void AppendChannelToKey(const ExpandChannel& channel, VertexDataKey& key,
                        int face, int corner, const VtIntArray& face_indices) {
  switch (channel.descriptor.interpolation) {
    case HdInterpolationVertex:
    case HdInterpolationVarying:
      key.push_back(face_indices[corner]);
      break;
    case HdInterpolationFaceVarying:
      std::visit(
          [&](const auto& src) {
            AppendValueToKey(key, SamplePrimvar(src, HdInterpolationFaceVarying,
                                                face, corner, face_indices));
          },
          channel.src);
      break;
    case HdInterpolationUniform:
      key.push_back(face);
      break;
    case HdInterpolationConstant:
      key.push_back(0);
      break;
    default:
      break;
  }
}

std::optional<PrimvarArrayVariant> ExtractPrimvarArray(const VtValue& value) {
  if (value.IsHolding<VtVec3fArray>()) {
    return value.UncheckedGet<VtVec3fArray>();
  }
  if (value.IsHolding<VtVec2fArray>()) {
    return value.UncheckedGet<VtVec2fArray>();
  }
  if (value.IsHolding<VtFloatArray>()) {
    return value.UncheckedGet<VtFloatArray>();
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

template <typename T>
VtValue TransformArray(const VtArray<T>& initial_values,
                       const HdPrimvarDescriptor& descriptor,
                       const GfMatrix4d& transform);

template <>
VtValue TransformArray<GfVec3f>(const VtVec3fArray& initial_values,
                                const HdPrimvarDescriptor& descriptor,
                                const GfMatrix4d& transform) {
  const size_t n = initial_values.size();
  if (descriptor.role == HdPrimvarRoleTokens->point) {
    VtVec3fArray values(n);
    GfVec3f* dst = values.data();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = GfVec3f(transform.TransformAffine(initial_values[i]));
    }
    return VtValue(std::move(values));
  } else if (descriptor.role == HdPrimvarRoleTokens->normal) {
    VtVec3fArray values(n);
    GfVec3f* dst = values.data();
    const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = GfVec3f(
          normal_transform.TransformDir(initial_values[i]).GetNormalized());
    }
    return VtValue(std::move(values));
  } else {
    return VtValue(initial_values);
  }
}

template <>
VtValue TransformArray<GfVec2f>(const VtVec2fArray& initial_values,
                                const HdPrimvarDescriptor& descriptor,
                                const GfMatrix4d& /*transform*/) {
  if (descriptor.role == HdPrimvarRoleTokens->textureCoordinate) {
    const size_t n = initial_values.size();
    VtVec2fArray values(n);
    GfVec2f* dst = values.data();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = GfVec2f(initial_values[i][0], 1.0f - initial_values[i][1]);
    }
    return VtValue(std::move(values));
  } else {
    return VtValue(initial_values);
  }
}

}  // namespace

std::pair<VtIntArray, VtIntArray> GeometryProcessor::TriangulateWithFaceMapping(
    const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices) {
  TRACE_FUNCTION();

  int total_triangles = 0;
  for (int count : face_vertex_counts) {
    if (count >= 3) {
      total_triangles += count - 2;
    }
  }
  VtIntArray triangles(total_triangles * 3);
  VtIntArray primitive_params(total_triangles);
  int* tri_ptr = triangles.data();
  int* param_ptr = primitive_params.data();

  int index = 0;
  int tri_idx = 0;
  const int num_faces = static_cast<int>(face_vertex_counts.size());
  for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
    const int count = face_vertex_counts[face_idx];
    if (count >= 3) {
      const int v0 = face_vertex_indices[index];
      for (int i = 0; i < count - 2; ++i) {
        tri_ptr[tri_idx * 3 + 0] = v0;
        tri_ptr[tri_idx * 3 + 1] = face_vertex_indices[index + i + 1];
        tri_ptr[tri_idx * 3 + 2] = face_vertex_indices[index + i + 2];
        param_ptr[tri_idx] = face_idx;
        ++tri_idx;
      }
    }
    index += count;
  }
  return {std::move(triangles), std::move(primitive_params)};
}

void GeometryProcessor::ComputeNormals(PrimvarMap& primvars,
                                       const HdMeshTopology& topology) {
  ComputeNormals(primvars, topology.GetFaceVertexIndices(),
                 topology.GetFaceVertexCounts());
}

void GeometryProcessor::ComputeNormals(PrimvarMap& primvars,
                                       const VtIntArray& face_vertex_indices,
                                       const VtIntArray& face_vertex_counts) {
  if (primvars.find(HdTokens->points) == primvars.end()) {
    return;
  }
  TF_DEBUG(HDMITSUBA_GEOMETRY).Msg("ComputeNormals\n");
  HdPrimvarDescriptor descriptor;
  descriptor.interpolation = HdInterpolationVertex;
  descriptor.indexed = false;
  descriptor.role = HdPrimvarRoleTokens->normal;

  const VtVec3fArray& points =
      primvars[HdTokens->points].value.Get<VtVec3fArray>();

  VtVec3fArray normals(points.size(), GfVec3f(0.0f, 0.0f, 0.0f));
  GfVec3f* normals_ptr = normals.data();

  int index = 0;
  for (int count : face_vertex_counts) {
    if (count >= 3) {
      for (int i = 0; i < count - 2; ++i) {
        GfVec3i face(face_vertex_indices[index],
                     face_vertex_indices[index + i + 1],
                     face_vertex_indices[index + i + 2]);
        GfVec3f p[3] = {points[face[0]], points[face[1]], points[face[2]]};
        GfVec3f face_normal = GfCross(p[1] - p[0], p[2] - p[0]).GetNormalized();
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
  primvars[HdTokens->normals] = {VtValue(std::move(normals)), descriptor};
}

void GeometryProcessor::TransformPrimvars(PrimvarMap& primvars,
                                          const GfMatrix4d& transform) {
  for (auto& [token, state] : primvars) {
    const auto& value = state.value;
    if (value.IsHolding<VtVec3fArray>() && !value.Get<VtVec3fArray>().empty()) {
      state.value = TransformArray<GfVec3f>(value.Get<VtVec3fArray>(),
                                            state.descriptor, transform);
    } else if (value.IsHolding<VtVec2fArray>() &&
               !value.Get<VtVec2fArray>().empty()) {
      state.value = TransformArray<GfVec2f>(value.Get<VtVec2fArray>(),
                                            state.descriptor, transform);
    }
  }
}

std::pair<VtIntArray, PrimvarMap> GeometryProcessor::ExpandPrimData(
    const HdMeshTopology& topology, const PrimvarMap& primvars) {
  return ExpandPrimData(topology.GetFaceVertexIndices(),
                        topology.GetFaceVertexCounts(), primvars);
}

std::pair<VtIntArray, PrimvarMap> GeometryProcessor::ExpandPrimData(
    const VtIntArray& face_vertex_indices, const VtIntArray& face_vertex_counts,
    const PrimvarMap& primvars) {
  TRACE_FUNCTION();
  absl::Time start = absl::Now();

  std::vector<ExpandChannel> channels;
  channels.reserve(primvars.size());
  for (const auto& [token, state] : primvars) {
    auto src_opt = ExtractPrimvarArray(state.value);
    if (!src_opt) continue;
    channels.push_back({token, state.descriptor, std::move(*src_opt)});
  }

  const int num_faces = static_cast<int>(face_vertex_counts.size());

  size_t num_face_varyings = 0;
  for (int count : face_vertex_counts) {
    num_face_varyings += count;
  }

  struct SourceLoc {
    int face;
    int corner;
  };
  std::vector<SourceLoc> unique_locs;
  unique_locs.reserve(num_face_varyings);

  VtIntArray final_face_indices(num_face_varyings);
  int* final_face_indices_ptr = final_face_indices.data();

  if constexpr (kUseMeshCompression) {
    // Channels with Vertex/Varying/Constant interpolation are determined solely
    // by the source point index `face_vertex_indices[corner]`. Grouping corners
    // by source point index means we only need to compare FaceVarying and
    // Uniform channels within each point's valence list (typically 1-2 entries).
    bool has_vertex_channel = false;
    size_t num_points = 0;
    std::vector<const ExpandChannel*> varying_channels;
    size_t key_stride = 0;
    for (const auto& channel : channels) {
      const HdInterpolation interp = channel.descriptor.interpolation;
      if (interp == HdInterpolationVertex || interp == HdInterpolationVarying) {
        has_vertex_channel = true;
        std::visit(
            [&](const auto& arr) {
              num_points = std::max(num_points, arr.size());
            },
            channel.src);
      } else if (interp == HdInterpolationFaceVarying ||
                 interp == HdInterpolationUniform) {
        varying_channels.push_back(&channel);
        if (interp == HdInterpolationUniform) {
          key_stride += 1;
        } else {
          std::visit(
              [&](const auto& arr) {
                using ArrayType = std::decay_t<decltype(arr)>;
                key_stride +=
                    PrimvarDimension<typename ArrayType::value_type>();
              },
              channel.src);
        }
      }
    }
    if (!has_vertex_channel) {
      num_points = 1;
    }

    std::vector<int> first_vertex_for_point(std::max<size_t>(num_points, 1),
                                            -1);
    if (varying_channels.empty()) {
      int corner_index = 0;
      for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
        const int count = face_vertex_counts[face_idx];
        for (int i = 0; i < count; ++i) {
          const int p =
              has_vertex_channel ? face_vertex_indices[corner_index] : 0;
          int v = first_vertex_for_point[p];
          if (v < 0) {
            v = static_cast<int>(unique_locs.size());
            first_vertex_for_point[p] = v;
            unique_locs.push_back({face_idx, corner_index});
          }
          final_face_indices_ptr[corner_index] = v;
          corner_index++;
        }
      }
    } else {
      std::vector<int> next_vertex;
      next_vertex.reserve(num_face_varyings);
      std::vector<int> vertex_keys;
      vertex_keys.reserve(num_face_varyings * key_stride);
      VertexDataKey key;
      key.reserve(key_stride);

      int corner_index = 0;
      for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
        const int count = face_vertex_counts[face_idx];
        for (int i = 0; i < count; ++i) {
          const int p =
              has_vertex_channel ? face_vertex_indices[corner_index] : 0;
          key.clear();
          for (const ExpandChannel* channel : varying_channels) {
            AppendChannelToKey(*channel, key, face_idx, corner_index,
                               face_vertex_indices);
          }
          int v = first_vertex_for_point[p];
          while (v >= 0) {
            if (std::memcmp(
                    vertex_keys.data() + static_cast<size_t>(v) * key_stride,
                    key.data(), key_stride * sizeof(int)) == 0) {
              break;
            }
            v = next_vertex[v];
          }
          if (v < 0) {
            v = static_cast<int>(unique_locs.size());
            unique_locs.push_back({face_idx, corner_index});
            next_vertex.push_back(first_vertex_for_point[p]);
            first_vertex_for_point[p] = v;
            vertex_keys.insert(vertex_keys.end(), key.begin(), key.end());
          }
          final_face_indices_ptr[corner_index] = v;
          corner_index++;
        }
      }
    }
  } else {
    int corner_index = 0;
    for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
      const int count = face_vertex_counts[face_idx];
      for (int i = 0; i < count; ++i) {
        final_face_indices_ptr[corner_index] = corner_index;
        unique_locs.push_back({face_idx, corner_index});
        corner_index++;
      }
    }
  }
  absl::Time end = absl::Now();
  absl::Duration duration = end - start;
  if constexpr (kUseMeshCompression) {
    TF_DEBUG(HDMITSUBA_GEOMETRY)
        .Msg("Compressed to %zu / %zu vertices. Time: %f ms\n",
             unique_locs.size(), num_face_varyings,
             absl::ToDoubleMilliseconds(duration));
  }

  PrimvarMap final_primvars;
  const size_t num_unique = unique_locs.size();
  for (auto& channel : channels) {
    const HdInterpolation interp = channel.descriptor.interpolation;
    std::visit(
        [&](const auto& src) {
          using ArrayType = std::decay_t<decltype(src)>;
          if (src.empty() || num_unique == 0) return;
          ArrayType dst_array(num_unique);
          auto* dst_ptr = dst_array.data();
          for (size_t i = 0; i < num_unique; ++i) {
            dst_ptr[i] = SamplePrimvar(src, interp, unique_locs[i].face,
                                       unique_locs[i].corner,
                                       face_vertex_indices);
          }
          final_primvars[channel.name] = {VtValue(std::move(dst_array)),
                                          channel.descriptor};
        },
        channel.src);
  }

  return {std::move(final_face_indices), std::move(final_primvars)};
}

std::vector<SubMeshOutput> GeometryProcessor::SplitAndCompactMeshes(
    const SdfPath& id, const VtIntArray& triangles,
    const VtIntArray& primitive_params, const PrimvarMap& final_primvars,
    absl::Span<const SdfPath> material_ids,
    const VtIntArray& face_material_indices) {
  TRACE_FUNCTION();
  std::vector<SubMeshOutput> sub_meshes;

  // The single material case is trivial.
  if (material_ids.size() == 1) {
    SubMeshOutput out;
    out.id = id;
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
          ? points_it->second.value.Get<VtVec3fArray>().size()
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

    PrimvarMap primvars;
    for (const auto& [token, state] : final_primvars) {
      if (auto src_opt = ExtractPrimvarArray(state.value)) {
        VtValue compacted = std::visit(
            [&](const auto& src) {
              return GatherVertices(src, unique_old_vertices);
            },
            *src_opt);
        primvars[token] = {std::move(compacted), state.descriptor};
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

SdfPath GeometryProcessor::MakeSubMeshId(const SdfPath& mesh_id,
                                         const SdfPath& material_id,
                                         size_t material_index) {
  std::string mat_name =
      material_id.IsEmpty()
          ? "mat_" + std::to_string(material_index)
          : absl::StrReplaceAll(material_id.GetAsString(),
                                {{"/", "_"}, {":", "_"}});
  return mesh_id.AppendChild(TfToken(mat_name));
}

PXR_NAMESPACE_CLOSE_SCOPE
