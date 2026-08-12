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

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <absl/types/span.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

struct PrimvarState {
  VtValue value;
  HdPrimvarDescriptor descriptor;
};

using PrimvarMap =
    std::unordered_map<TfToken, PrimvarState, TfToken::HashFunctor>;

class GeometryProcessor {
 public:
  GeometryProcessor() = delete;
  ~GeometryProcessor() = delete;

  template <typename T>
  using Interpolator =
      std::function<T(int global_face, int local_corner, int global_corner,
                      const VtIntArray& vertex_indices)>;

  // Returns an interpolation function for the given primvar data and
  // interpolation method. `data` is captured by value: VtArray is a refcounted
  // copy-on-write handle, so this is cheap, and the returned std::function
  // routinely outlives the caller's array.
  template <typename T>
  static Interpolator<T> GetInterpolator(const VtArray<T>& data,
                                         HdInterpolation interpolation) {
    if (data.empty()) {
      return [](int, int, int, const VtIntArray&) { return T(0.f); };
    }
    switch (interpolation) {
      case HdInterpolationConstant:
        return [data](int, int, int, const VtIntArray&) { return data[0]; };
      case HdInterpolationUniform:
        return [data](int global_face, int, int, const VtIntArray&) {
          return data[global_face];
        };
      case HdInterpolationVertex:
      case HdInterpolationVarying:
        return [data](int, int local_corner, int,
                      const VtIntArray& vertex_indices) {
          return data[vertex_indices[local_corner]];
        };
      case HdInterpolationFaceVarying:
        return [data](int, int, int global_corner, const VtIntArray&) {
          return data[global_corner];
        };
      default:
        return [](int, int, int, const VtIntArray&) { return T(0.f); };
    }
  }

  // Computes smooth vertex normals for a mesh topology.
  static void ComputeNormals(PrimvarMap& primvars,
                             const HdMeshTopology& topology);

  // Computes smooth vertex normals using raw topology arrays.
  static void ComputeNormals(PrimvarMap& primvars,
                             const VtIntArray& face_vertex_indices,
                             const VtIntArray& face_vertex_counts);

  // Transforms all vector and normal primvars by the given matrix.
  static void TransformPrimvars(PrimvarMap& primvars,
                                const GfMatrix4d& transform);

  // Scales the "normals" primvar to unit length in place.
  //
  // Mitsuba welds face corners by comparing the raw float payload and only
  // normalizes afterwards, so two corners of one vertex that agree in direction
  // but differ in magnitude would be split apart for no reason. USD frequently
  // carries unnormalized normals, so this has to run before handing them over.
  static void NormalizeNormals(PrimvarMap& primvars);
};

PXR_NAMESPACE_CLOSE_SCOPE
