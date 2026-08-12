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
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <drjit/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
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

#include "hdmitsuba/debug_codes.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

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

void GeometryProcessor::NormalizeNormals(PrimvarMap& primvars) {
  auto it = primvars.find(HdTokens->normals);
  if (it == primvars.end() || !it->second.value.IsHolding<VtVec3fArray>()) {
    return;
  }
  VtVec3fArray normals = it->second.value.Get<VtVec3fArray>();
  for (GfVec3f& n : normals) {
    const float length = n.GetLength();
    if (length > 0.0f) n /= length;
  }
  it->second.value = VtValue(std::move(normals));
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


PXR_NAMESPACE_CLOSE_SCOPE
