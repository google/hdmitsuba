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
  PrimvarArrayVariant dst;
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

void AppendChannelValue(ExpandChannel& channel, int face, int corner,
                        const VtIntArray& face_indices) {
  const HdInterpolation interp = channel.descriptor.interpolation;
  std::visit(
      [&](const auto& src) {
        using ArrayType = std::decay_t<decltype(src)>;
        std::get<ArrayType>(channel.dst)
            .push_back(SamplePrimvar(src, interp, face, corner, face_indices));
      },
      channel.src);
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
  VtArray<T> out;
  out.reserve(unique_old_vertices.size());
  for (int old_v : unique_old_vertices) {
    out.push_back(src[old_v]);
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
  VtVec3fArray values;
  if (descriptor.role == HdPrimvarRoleTokens->point) {
    values.reserve(initial_values.size());
    for (const auto& v : initial_values) {
      values.push_back(GfVec3f(transform.TransformAffine(v)));
    }
  } else if (descriptor.role == HdPrimvarRoleTokens->normal) {
    values.reserve(initial_values.size());
    const GfMatrix4d normal_transform = transform.GetInverse().GetTranspose();
    for (const auto& v : initial_values) {
      values.push_back(
          GfVec3f(normal_transform.TransformDir(v).GetNormalized()));
    }
  } else {
    return VtValue(initial_values);
  }
  return VtValue(values);
}

template <>
VtValue TransformArray<GfVec2f>(const VtVec2fArray& initial_values,
                                const HdPrimvarDescriptor& descriptor,
                                const GfMatrix4d& /*transform*/) {
  VtVec2fArray values;
  if (descriptor.role == HdPrimvarRoleTokens->textureCoordinate) {
    values.reserve(initial_values.size());
    for (const auto& v : initial_values) {
      values.push_back(GfVec2f(v[0], 1.0f - v[1]));
    }
  } else {
    return VtValue(initial_values);
  }
  return VtValue(values);
}

}  // namespace

std::pair<VtIntArray, VtIntArray> GeometryProcessor::TriangulateWithFaceMapping(
    const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices) {
  TRACE_FUNCTION();
  VtIntArray triangles;
  VtIntArray primitive_params;

  int total_triangles = 0;
  for (int count : face_vertex_counts) {
    if (count >= 3) {
      total_triangles += count - 2;
    }
  }
  triangles.reserve(total_triangles * 3);
  primitive_params.reserve(total_triangles);

  int index = 0;
  for (int face_idx = 0; face_idx < static_cast<int>(face_vertex_counts.size());
       ++face_idx) {
    const int count = face_vertex_counts[face_idx];
    if (count >= 3) {
      for (int i = 0; i < count - 2; ++i) {
        triangles.push_back(face_vertex_indices[index]);
        triangles.push_back(face_vertex_indices[index + i + 1]);
        triangles.push_back(face_vertex_indices[index + i + 2]);
        primitive_params.push_back(face_idx);
      }
    }
    index += count;
  }
  return {triangles, primitive_params};
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

  int index = 0;
  for (int face_idx = 0; face_idx < static_cast<int>(face_vertex_counts.size());
       ++face_idx) {
    int count = face_vertex_counts[face_idx];
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
          normals[face[j]] += face_normal * face_angle;
        }
      }
    }
    index += count;
  }
  for (size_t i = 0; i < normals.size(); ++i) {
    normals[i].Normalize();
  }
  primvars[HdTokens->normals] = {VtValue(normals), descriptor};
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
  size_t total_primvar_dim = 0;
  for (const auto& [token, state] : primvars) {
    auto src_opt = ExtractPrimvarArray(state.value);
    if (!src_opt) continue;
    PrimvarArrayVariant dst = std::visit(
        [&](const auto& arr) -> PrimvarArrayVariant {
          using ArrayType = std::decay_t<decltype(arr)>;
          total_primvar_dim +=
              PrimvarDimension<typename ArrayType::value_type>();
          return ArrayType();
        },
        *src_opt);
    channels.push_back(
        {token, state.descriptor, std::move(*src_opt), std::move(dst)});
  }

  size_t num_face_varyings = 0;
  for (int count : face_vertex_counts) {
    num_face_varyings += count;
  }
  absl::flat_hash_map<VertexDataKey, int> unique_vertex_map;
  VtIntArray final_face_indices(num_face_varyings);
  int corner_index = 0;
  for (int face_idx = 0; face_idx < static_cast<int>(face_vertex_counts.size());
       ++face_idx) {
    for (int i = 0; i < face_vertex_counts[face_idx]; ++i) {
      size_t vertex_index;
      bool is_new_vertex = true;
      if constexpr (kUseMeshCompression) {
        VertexDataKey key;
        key.reserve(total_primvar_dim);
        for (const auto& channel : channels) {
          AppendChannelToKey(channel, key, face_idx, corner_index,
                             face_vertex_indices);
        }
        vertex_index = unique_vertex_map.size();
        auto [it, inserted] = unique_vertex_map.try_emplace(key, vertex_index);
        vertex_index = it->second;
        is_new_vertex = inserted;
      } else {
        vertex_index = corner_index;
      }
      final_face_indices[corner_index] = vertex_index;
      if (is_new_vertex) {
        for (auto& channel : channels) {
          AppendChannelValue(channel, face_idx, corner_index,
                             face_vertex_indices);
        }
      }
      corner_index++;
    }
  }
  absl::Time end = absl::Now();
  absl::Duration duration = end - start;
  if constexpr (kUseMeshCompression) {
    TF_DEBUG(HDMITSUBA_GEOMETRY)
        .Msg("Compressed to %zu / %zu vertices. Time: %f ms\n",
             unique_vertex_map.size(), num_face_varyings,
             absl::ToDoubleMilliseconds(duration));
  }

  PrimvarMap final_primvars;
  for (auto& channel : channels) {
    std::visit(
        [&](auto& specific_array) {
          if (!specific_array.empty()) {
            final_primvars[channel.name] = {VtValue(std::move(specific_array)),
                                            channel.descriptor};
          }
        },
        channel.dst);
  }

  return {final_face_indices, final_primvars};
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
    material_indices[material_index].push_back(triangles[i * 3]);
    material_indices[material_index].push_back(triangles[i * 3 + 1]);
    material_indices[material_index].push_back(triangles[i * 3 + 2]);
  }

  size_t total_source_vertices =
      (final_primvars.find(HdTokens->points) != final_primvars.end())
          ? final_primvars.at(HdTokens->points).value.Get<VtVec3fArray>().size()
          : 0;

  absl::flat_hash_map<int, int> old_to_new_vertex_map;
  old_to_new_vertex_map.reserve(total_source_vertices);
  std::vector<int> unique_old_vertices;

  for (size_t i = 0; i < material_ids.size(); ++i) {
    if (material_indices[i].empty()) continue;
    old_to_new_vertex_map.clear();
    unique_old_vertices.clear();

    size_t max_vertices =
        std::min(material_indices[i].size(), total_source_vertices > 0
                                                 ? total_source_vertices
                                                 : material_indices[i].size());
    unique_old_vertices.reserve(max_vertices);

    VtIntArray submesh_triangles;
    submesh_triangles.reserve(material_indices[i].size());

    // Compact the triangles and record unique source vertex indices.
    for (const int vertex_index : material_indices[i]) {
      auto [it, inserted] = old_to_new_vertex_map.try_emplace(
          vertex_index, static_cast<int>(unique_old_vertices.size()));
      submesh_triangles.push_back(it->second);
      if (inserted) {
        unique_old_vertices.push_back(vertex_index);
      }
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

    std::string mat_name =
        material_ids[i].IsEmpty()
            ? "mat_" + std::to_string(i)
            : absl::StrReplaceAll(material_ids[i].GetAsString(),
                                  {{"/", "_"}, {":", "_"}});
    SubMeshOutput out;
    out.id = id.AppendChild(TfToken(mat_name));
    out.material_id = material_ids[i];
    out.triangles = std::move(submesh_triangles);
    out.primvars = std::move(primvars);
    sub_meshes.push_back(std::move(out));
  }

  return sub_meshes;
}

PXR_NAMESPACE_CLOSE_SCOPE
