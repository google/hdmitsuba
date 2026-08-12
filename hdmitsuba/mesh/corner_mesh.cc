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

#include "hdmitsuba/mesh/corner_mesh.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <absl/strings/str_replace.h>
#include <absl/types/span.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// A primvar that survived validation, ready to be turned into per-sub-mesh
// records. The VtValue is a refcounted handle, so this copies nothing.
struct AttributeSource {
  TfToken name;
  VtValue values;
  size_t dim;
  HdInterpolation interpolation;
};

// Channel count of a primvar array, or 0 if it is not a type we can forward.
size_t PrimvarDim(const VtValue& value) {
  if (value.IsHolding<VtVec3fArray>()) return 3;
  if (value.IsHolding<VtVec2fArray>()) return 2;
  return 0;
}

size_t PrimvarSize(const VtValue& value) {
  if (value.IsHolding<VtVec3fArray>()) return value.Get<VtVec3fArray>().size();
  if (value.IsHolding<VtVec2fArray>()) return value.Get<VtVec2fArray>().size();
  return 0;
}

}  // namespace

NormalPolicy DecideNormals(bool had_authored_normals, bool is_subdivided,
                           bool has_displacement, bool multi_material) {
  NormalPolicy policy;
  policy.smooth_normals =
      had_authored_normals || is_subdivided || has_displacement;
  policy.compute_by_hand =
      !had_authored_normals && (has_displacement || multi_material);
  return policy;
}

void RefreshNormalsAfterDisplacement(PrimvarMap& primvars,
                                     const VtIntArray& face_vertex_counts,
                                     const VtIntArray& face_vertex_indices,
                                     bool multi_material) {
  if (multi_material) {
    GeometryProcessor::ComputeNormals(primvars, face_vertex_indices,
                                      face_vertex_counts);
  } else {
    primvars.erase(HdTokens->normals);
  }
}

size_t CornerMeshBuilder::ExpectedValueCount(HdInterpolation interpolation,
                                            size_t point_count,
                                            size_t face_count,
                                            size_t corner_count) {
  switch (interpolation) {
    case HdInterpolationConstant:
      // Every record reads row 0, so anything non-empty works.
      return std::numeric_limits<size_t>::max();
    case HdInterpolationUniform:
      return face_count;
    case HdInterpolationVertex:
    case HdInterpolationVarying:
      // Varying primvars have already been refined to the vertex count by the
      // subdivision evaluator, so the two modes address the same array.
      return point_count;
    case HdInterpolationFaceVarying:
      return corner_count;
    default:
      return 0;
  }
}

CornerMeshSpec CornerMeshBuilder::Build(
    const SdfPath& id, const VtIntArray& face_vertex_counts,
    const VtIntArray& face_vertex_indices, const PrimvarMap& primvars,
    absl::Span<const SdfPath> material_ids,
    const VtIntArray& face_material_indices, bool smooth_normals,
    absl::Span<const TfToken> attribute_names) {
  TRACE_FUNCTION();

  CornerMeshSpec spec;
  spec.smooth_normals = smooth_normals;
  spec.primvars = primvars;

  auto points_it = primvars.find(HdTokens->points);
  if (points_it != primvars.end() &&
      points_it->second.value.IsHolding<VtVec3fArray>()) {
    spec.positions = points_it->second.value.Get<VtVec3fArray>();
  }
  if (spec.positions.empty()) return spec;

  const size_t num_faces = face_vertex_counts.size();
  const size_t num_points = spec.positions.size();

  // Corner offset of each face into the global corner arrays.
  std::vector<int> face_corner_base(num_faces + 1, 0);
  for (size_t f = 0; f < num_faces; ++f) {
    face_corner_base[f + 1] =
        face_corner_base[f] + std::max(0, face_vertex_counts[f]);
  }
  const size_t num_corners = static_cast<size_t>(face_corner_base[num_faces]);
  if (num_corners > face_vertex_indices.size()) {
    TF_RUNTIME_ERROR("Mesh %s: faceVertexCounts sum (%zu) exceeds "
                     "faceVertexIndices size (%zu).",
                     id.GetText(), num_corners, face_vertex_indices.size());
    return spec;
  }

  // Validate every requested primvar once, up front. `from_corners()` treats an
  // out-of-range index as fatal, whereas the old interpolator silently read out
  // of bounds, so a mismatch has to be caught here.
  std::vector<AttributeSource> sources;
  sources.reserve(attribute_names.size());
  for (const TfToken& name : attribute_names) {
    auto it = primvars.find(name);
    if (it == primvars.end()) continue;

    const size_t dim = PrimvarDim(it->second.value);
    if (dim == 0) continue;

    const size_t size = PrimvarSize(it->second.value);
    if (size == 0) continue;

    const HdInterpolation interpolation = it->second.descriptor.interpolation;
    const size_t expected =
        ExpectedValueCount(interpolation, num_points, num_faces, num_corners);
    if (expected == 0) {
      TF_WARN("Mesh %s: dropping primvar '%s' with unsupported interpolation %d.",
              id.GetText(), name.GetText(), static_cast<int>(interpolation));
      continue;
    }
    if (expected != std::numeric_limits<size_t>::max() && size != expected) {
      TF_WARN("Mesh %s: dropping primvar '%s' (interpolation %d) with %zu "
              "values, expected %zu.",
              id.GetText(), name.GetText(), static_cast<int>(interpolation),
              size, expected);
      continue;
    }
    sources.push_back({name, it->second.value, dim, interpolation});
  }

  const bool single_material = material_ids.size() <= 1;
  const size_t num_buckets = single_material ? 1 : material_ids.size();
  spec.sub_meshes.reserve(num_buckets);

  // Scratch, reused across buckets.
  std::vector<int> record_corner, record_face;

  for (size_t m = 0; m < num_buckets; ++m) {
    record_corner.clear();
    record_face.clear();

    SubMeshSpec sub;
    VtIntArray corner_vertex, face_offsets, triangle_face, tri_corner_record;
    face_offsets.push_back(0);

    for (size_t f = 0; f < num_faces; ++f) {
      if (!single_material) {
        if (f >= face_material_indices.size()) break;
        const int material_index = face_material_indices[f];
        if (!TF_VERIFY(material_index >= 0 &&
                       static_cast<size_t>(material_index) < num_buckets,
                       "Mesh %s: face %zu has out-of-range material index %d.",
                       id.GetText(), f, material_index)) {
          continue;
        }
        if (static_cast<size_t>(material_index) != m) continue;
      }

      const int count = std::max(0, face_vertex_counts[f]);
      const int base = face_corner_base[f];
      const int record_base = static_cast<int>(record_corner.size());

      for (int c = 0; c < count; ++c) {
        record_corner.push_back(base + c);
        record_face.push_back(static_cast<int>(f));
        corner_vertex.push_back(face_vertex_indices[base + c]);
      }
      face_offsets.push_back(static_cast<int>(record_corner.size()));

      // Mirror the fan triangulation `from_corners()` performs internally:
      // faces with fewer than three corners contribute nothing.
      for (int i = 0; i + 2 < count; ++i) {
        tri_corner_record.push_back(record_base);
        tri_corner_record.push_back(record_base + i + 1);
        tri_corner_record.push_back(record_base + i + 2);
        triangle_face.push_back(static_cast<int>(f));
      }
    }

    // A bucket with no triangles produces no mesh, matching the old
    // triangle-derived material buckets.
    if (triangle_face.empty()) continue;

    sub.material_id = single_material && material_ids.empty()
                          ? SdfPath()
                          : material_ids[m];
    if (single_material) {
      sub.id = id;
    } else {
      const std::string mat_name =
          material_ids[m].IsEmpty()
              ? "mat_" + std::to_string(m)
              : absl::StrReplaceAll(material_ids[m].GetAsString(),
                                    {{"/", "_"}, {":", "_"}});
      sub.id = id.AppendChild(TfToken(mat_name));
    }

    sub.corner_vertex = std::move(corner_vertex);
    sub.face_offsets = std::move(face_offsets);
    sub.triangle_face = std::move(triangle_face);
    sub.tri_corner_record = std::move(tri_corner_record);

    // `from_corners()` allocates one position row per referenced source vertex,
    // ascending, dropping the unreferenced ones. Only triangle corners count.
    {
      std::vector<int> used;
      used.reserve(sub.tri_corner_record.size());
      for (int r : sub.tri_corner_record) used.push_back(sub.corner_vertex[r]);
      std::sort(used.begin(), used.end());
      used.erase(std::unique(used.begin(), used.end()), used.end());
      sub.used_vertices.assign(used.begin(), used.end());
    }

    const size_t record_count = sub.RecordCount();
    sub.attributes.reserve(sources.size());
    for (const AttributeSource& src : sources) {
      CornerAttributeSpec attr;
      attr.name = src.name;
      attr.values = src.values;
      attr.dim = src.dim;
      attr.indices.reserve(record_count);
      for (size_t r = 0; r < record_count; ++r) {
        switch (src.interpolation) {
          case HdInterpolationConstant:
            attr.indices.push_back(0);
            break;
          case HdInterpolationUniform:
            attr.indices.push_back(record_face[r]);
            break;
          case HdInterpolationVertex:
          case HdInterpolationVarying:
            attr.indices.push_back(sub.corner_vertex[r]);
            break;
          default:  // HdInterpolationFaceVarying, the only remaining case
            attr.indices.push_back(record_corner[r]);
            break;
        }
      }
      sub.attributes.push_back(std::move(attr));
    }

    spec.sub_meshes.push_back(std::move(sub));
  }

  return spec;
}

PXR_NAMESPACE_CLOSE_SCOPE
