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

#include <string>
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

struct SubMeshOutput {
  std::string id;
  SdfPath material_id;
  VtIntArray triangles;
  PrimvarMap primvars;
};

// Polygon mesh (after subdivision) processed by `RunGeometryPipeline`.
struct MeshGeometry {
  VtIntArray face_vertex_counts;
  // Point index of each face corner.
  VtIntArray face_vertex_indices;
  // Whether the polygons are wound clockwise.
  bool left_handed = false;
  // Whether the mesh was refined by subdivision, which requires smooth normals.
  bool is_subdivided = false;
  // Materials of the faces, indexed by `face_material_indices`.
  std::vector<SdfPath> material_ids;
  VtIntArray face_material_indices;
  PrimvarMap primvars;
};

// Polygon mesh whose primvars were expanded to per-vertex arrays, where a
// vertex is a unique combination of a point and the values of the
// face-varying and uniform primvars at it.
struct ExpandedMesh {
  // Vertex index of each face corner.
  VtIntArray face_vertices;
  // Point index of each vertex.
  std::vector<int> vertex_points;
  PrimvarMap primvars;
};

class GeometryProcessor {
 public:
  GeometryProcessor() = delete;
  ~GeometryProcessor() = delete;

  // Triangulates a mesh topology and returns the triangle indices and primitive
  // parameters mapping each triangle to its parent polygon index. Triangles are
  // always wound counter-clockwise (right-handed), so the winding of
  // `left_handed` polygons is reversed.
  static std::pair<VtIntArray, VtIntArray> TriangulateWithFaceMapping(
      const VtIntArray& face_vertex_counts,
      const VtIntArray& face_vertex_indices, bool left_handed);

  // Returns the smooth (angle-weighted) normal of each point of a polygon mesh,
  // whose polygons are wound clockwise if `left_handed`.
  static VtVec3fArray ComputeNormals(const VtVec3fArray& points,
                                     const VtIntArray& face_vertex_counts,
                                     const VtIntArray& face_vertex_indices,
                                     bool left_handed);

  // Returns the normal of each face of a polygon mesh, whose polygons are
  // wound clockwise if `left_handed`.
  static VtVec3fArray ComputeFaceNormals(const VtVec3fArray& points,
                                         const VtIntArray& face_vertex_counts,
                                         const VtIntArray& face_vertex_indices,
                                         bool left_handed);

  // Transforms the point and normal primvars by the given matrix, and flips
  // the V coordinate of texture coordinate primvars to Mitsuba's convention.
  static void TransformPrimvars(PrimvarMap& primvars,
                                const GfMatrix4d& transform);

  // Expands the primvars of a polygon mesh with `num_points` points into
  // per-vertex arrays, merging the corners of each point whose face-varying
  // and uniform primvars agree into one vertex.
  static ExpandedMesh ExpandPrimData(const VtIntArray& face_vertex_counts,
                                     const VtIntArray& face_vertex_indices,
                                     size_t num_points,
                                     const PrimvarMap& primvars);

  // Constructs the shape ID for a multi-material mesh subset. The material
  // suffix is delimited by '#', which cannot appear in an `SdfPath`, so that
  // sub-mesh IDs never collide with the paths of child prims.
  static std::string MakeSubMeshId(const SdfPath& mesh_id,
                                   const SdfPath& material_id,
                                   size_t material_index);

  // Splits a triangulated mesh into multiple sub-meshes by material index and
  // compacts unused vertices from the primvar buffers.
  static std::vector<SubMeshOutput> SplitAndCompactMeshes(
      const SdfPath& id, const VtIntArray& triangles,
      const VtIntArray& primitive_params, const PrimvarMap& final_primvars,
      absl::Span<const SdfPath> material_ids,
      const VtIntArray& face_material_indices);
};

PXR_NAMESPACE_CLOSE_SCOPE
