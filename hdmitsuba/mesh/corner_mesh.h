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
#include <vector>

#include <absl/types/span.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh/geometry_processor.h"

PXR_NAMESPACE_OPEN_SCOPE

// Describes a mesh the way Mitsuba's `Mesh::from_corners()` wants it: positions
// per source vertex, everything else per *face corner*. Mitsuba then welds the
// corners of each vertex, splitting only where the per-corner data disagrees,
// and records which copies still share a position (`position_index`) or a
// normal (`normal_index`). Seams therefore stop being geometric cuts, which is
// what lets us hand shading-normal generation back to Mitsuba.
//
// Nothing here copies primvar data: `values` and `positions` are refcounted
// VtArray handles onto the mesh-level arrays, and each sub-mesh only owns a few
// small index arrays.

// One per-corner attribute. `indices` maps each corner record of the owning
// sub-mesh to a row of `values`, which is what turns a USD interpolation mode
// into a plain gather.
struct CornerAttributeSpec {
  TfToken name;        // "normals", "st", or a "vertex_"-prefixed custom name
  VtValue values;      // mesh-level VtVec3fArray / VtVec2fArray
  size_t dim = 0;      // 3 or 2
  VtIntArray indices;  // RecordCount() entries
};

// One Mitsuba mesh: the faces of a single material subset.
struct SubMeshSpec {
  SdfPath id;
  SdfPath material_id;

  // One record per face corner of this sub-mesh, in face order.
  VtIntArray corner_vertex;  // record -> source vertex
  VtIntArray face_offsets;   // FaceCount() + 1 prefix sums over records

  // Fan triangulation of the above, matching what `from_corners()` performs
  // internally. Faces with fewer than three corners contribute no triangle.
  VtIntArray triangle_face;      // triangle -> source (global) face
  VtIntArray tri_corner_record;  // 3 per triangle -> record

  // Mitsuba's `positions` rows in order: `from_corners()` allocates one per
  // referenced source vertex, ascending, skipping unreferenced ones. Lets an
  // in-place update gather straight into Mitsuba's row order without re-welding.
  VtIntArray used_vertices;  // position row -> source vertex

  // Filled in after `from_corners()` has welded the mesh, since only Mitsuba
  // knows how the corners collapsed. `normal_record` names one representative
  // record per normal group; all records of a group carry the same value by
  // construction, so any of them reproduces it. Empty until the mesh is built.
  VtIntArray normal_record;  // normal group -> record

  std::vector<CornerAttributeSpec> attributes;

  size_t RecordCount() const { return corner_vertex.size(); }
  size_t FaceCount() const {
    return face_offsets.empty() ? 0 : face_offsets.size() - 1;
  }
  size_t TriangleCount() const { return triangle_face.size(); }
};

struct CornerMeshSpec {
  // Shared by every sub-mesh; `from_corners()` drops the vertices a sub-mesh
  // does not reference, so the array is passed whole each time.
  VtVec3fArray positions;

  // False makes the mesh flat shaded (Mitsuba property `face_normals`). When
  // true and no "normals" attribute is present, Mitsuba generates smooth
  // normals itself, continuous across seams.
  bool smooth_normals = false;

  // Mesh-level primvars, kept for consumers that look at them as a whole (BSDF
  // resolution reads `displayColor` from here).
  PrimvarMap primvars;

  std::vector<SubMeshSpec> sub_meshes;
};

// Who supplies the shading normals, and whether the mesh is smooth at all.
struct NormalPolicy {
  // Drives Mitsuba's `face_normals` property (as its negation). False leaves the
  // mesh flat shaded.
  bool smooth_normals = false;

  // Whether the caller has to synthesize normals with
  // `GeometryProcessor::ComputeNormals` before building. When false and no
  // normals are authored, Mitsuba generates them itself -- continuously across
  // seams, which is the whole point of the corner representation.
  bool compute_by_hand = false;
};

// The single place both geometry pipelines decide what to do about normals.
//
// `smooth_normals` reproduces the historical `face_normals = !has_normals`,
// where the primvar was present either because it was authored or because
// ComputeNormals had just synthesized it. Hand computation survives for exactly
// two cases: displacement reads normals *before* the mesh exists, and a
// multi-material mesh becomes several Mitsuba meshes whose independently
// generated normals would crease at the GeomSubset boundaries.
NormalPolicy DecideNormals(bool had_authored_normals, bool is_subdivided,
                           bool has_displacement, bool multi_material);

// Post-displacement normal handling, shared by both pipelines. A multi-material
// mesh keeps whole-mesh normals; anything else drops them so that Mitsuba
// derives them from the displaced positions.
void RefreshNormalsAfterDisplacement(PrimvarMap& primvars,
                                     const VtIntArray& face_vertex_counts,
                                     const VtIntArray& face_vertex_indices,
                                     bool multi_material);

class CornerMeshBuilder {
 public:
  CornerMeshBuilder() = delete;
  ~CornerMeshBuilder() = delete;

  // Partitions `face_material_indices` into one sub-mesh per material and
  // describes each as corner-indexed data over the shared position array.
  //
  // `primvars` are the mesh-level, *unexpanded* primvars; "points" supplies the
  // positions and every other entry named in `attribute_names` becomes a
  // CornerAttributeSpec. An attribute whose array length disagrees with its
  // declared interpolation is dropped with a warning rather than passed on,
  // since `from_corners()` treats an out-of-range index as fatal.
  static CornerMeshSpec Build(const SdfPath& id,
                              const VtIntArray& face_vertex_counts,
                              const VtIntArray& face_vertex_indices,
                              const PrimvarMap& primvars,
                              absl::Span<const SdfPath> material_ids,
                              const VtIntArray& face_material_indices,
                              bool smooth_normals,
                              absl::Span<const TfToken> attribute_names);

  // Number of values an array must hold to be addressable under `interpolation`.
  // `SIZE_MAX` means "no constraint" (constant, which only needs one entry).
  static size_t ExpectedValueCount(HdInterpolation interpolation,
                                   size_t point_count, size_t face_count,
                                   size_t corner_count);
};

PXR_NAMESPACE_CLOSE_SCOPE
