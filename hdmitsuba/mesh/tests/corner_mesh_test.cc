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

#include <vector>

#include <gtest/gtest.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec2f;
using pxr::GfVec3f;
using pxr::HdPrimvarDescriptor;
using pxr::HdTokens;
using pxr::SdfPath;
using pxr::TfToken;
using pxr::VtIntArray;
using pxr::VtValue;
using pxr::VtVec2fArray;
using pxr::VtVec3fArray;

const TfToken kSt("st");

// Two quads sharing the edge (1, 2): 6 points, 8 corners, 2 faces.
//
//   3---2---5
//   |   |   |
//   0---1---4
PrimvarMap TwoQuadPrimvars() {
  VtVec3fArray points = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0),
                         GfVec3f(0, 1, 0), GfVec3f(2, 0, 0), GfVec3f(2, 1, 0)};
  HdPrimvarDescriptor desc;
  desc.interpolation = pxr::HdInterpolationVertex;
  PrimvarMap primvars;
  primvars[HdTokens->points] = {VtValue(points), desc};
  return primvars;
}

VtIntArray TwoQuadCounts() { return VtIntArray{4, 4}; }
VtIntArray TwoQuadIndices() { return VtIntArray{0, 1, 2, 3, 1, 4, 5, 2}; }

void AddPrimvar(PrimvarMap& primvars, const TfToken& name, const VtValue& value,
                HdInterpolation interpolation) {
  HdPrimvarDescriptor desc;
  desc.interpolation = interpolation;
  primvars[name] = {value, desc};
}

CornerMeshSpec BuildTwoQuads(const PrimvarMap& primvars,
                             absl::Span<const SdfPath> material_ids,
                             const VtIntArray& face_material_indices,
                             absl::Span<const TfToken> attrs = {}) {
  return CornerMeshBuilder::Build(SdfPath("/mesh"), TwoQuadCounts(),
                                  TwoQuadIndices(), primvars, material_ids,
                                  face_material_indices,
                                  /*smooth_normals=*/true, attrs);
}

TEST(CornerMeshBuilderTest, SingleMaterialKeepsTopologyVerbatim) {
  const SdfPath material("/mat");
  CornerMeshSpec spec = BuildTwoQuads(TwoQuadPrimvars(), {&material, 1}, {});

  ASSERT_EQ(spec.sub_meshes.size(), 1);
  const SubMeshSpec& sub = spec.sub_meshes[0];

  // A single material keeps the prim's own path -- SceneModel looks up
  // sub-meshes by this id when updating in place.
  EXPECT_EQ(sub.id, SdfPath("/mesh"));
  EXPECT_EQ(sub.corner_vertex, TwoQuadIndices());
  EXPECT_EQ(sub.face_offsets, (VtIntArray{0, 4, 8}));
  // Each quad fans into two triangles around its first corner.
  EXPECT_EQ(sub.triangle_face, (VtIntArray{0, 0, 1, 1}));
  EXPECT_EQ(sub.tri_corner_record,
            (VtIntArray{0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7}));
  EXPECT_EQ(sub.used_vertices, (VtIntArray{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(spec.positions.size(), 6);
  EXPECT_TRUE(spec.smooth_normals);
}

TEST(CornerMeshBuilderTest, SplitsFacesByMaterial) {
  const SdfPath materials[] = {SdfPath("/root/mats/red"),
                               SdfPath("/root/mats/blue")};
  CornerMeshSpec spec =
      BuildTwoQuads(TwoQuadPrimvars(), materials, VtIntArray{0, 1});

  ASSERT_EQ(spec.sub_meshes.size(), 2);

  // Multi-material meshes get a child path per material, with the separators
  // flattened, exactly as the previous splitting code named them.
  EXPECT_EQ(spec.sub_meshes[0].id, SdfPath("/mesh/_root_mats_red"));
  EXPECT_EQ(spec.sub_meshes[1].id, SdfPath("/mesh/_root_mats_blue"));
  EXPECT_EQ(spec.sub_meshes[0].material_id, materials[0]);
  EXPECT_EQ(spec.sub_meshes[1].material_id, materials[1]);

  // Positions stay shared; each sub-mesh just references fewer of them, and
  // from_corners() drops the rest.
  EXPECT_EQ(spec.sub_meshes[0].corner_vertex, (VtIntArray{0, 1, 2, 3}));
  EXPECT_EQ(spec.sub_meshes[0].used_vertices, (VtIntArray{0, 1, 2, 3}));
  EXPECT_EQ(spec.sub_meshes[1].corner_vertex, (VtIntArray{1, 4, 5, 2}));
  EXPECT_EQ(spec.sub_meshes[1].used_vertices, (VtIntArray{1, 2, 4, 5}));
  EXPECT_EQ(spec.sub_meshes[0].face_offsets, (VtIntArray{0, 4}));
  EXPECT_EQ(spec.sub_meshes[1].face_offsets, (VtIntArray{0, 4}));

  // triangle_face stays in the *global* face numbering.
  EXPECT_EQ(spec.sub_meshes[0].triangle_face, (VtIntArray{0, 0}));
  EXPECT_EQ(spec.sub_meshes[1].triangle_face, (VtIntArray{1, 1}));
}

TEST(CornerMeshBuilderTest, SkipsMaterialsWithoutTriangles) {
  const SdfPath materials[] = {SdfPath("/a"), SdfPath("/b")};
  // Nothing is bound to material 1, so it must not produce an empty sub-mesh.
  CornerMeshSpec spec =
      BuildTwoQuads(TwoQuadPrimvars(), materials, VtIntArray{0, 0});

  ASSERT_EQ(spec.sub_meshes.size(), 1);
  EXPECT_EQ(spec.sub_meshes[0].id, SdfPath("/mesh/_a"));
  EXPECT_EQ(spec.sub_meshes[0].triangle_face, (VtIntArray{0, 0, 1, 1}));
}

TEST(CornerMeshBuilderTest, DegenerateFacesProduceNoTriangles) {
  // Point 6 is touched only by a dangling two-corner face.
  VtVec3fArray points = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0),
                         GfVec3f(0, 1, 0), GfVec3f(2, 0, 0), GfVec3f(2, 1, 0),
                         GfVec3f(9, 9, 9)};
  HdPrimvarDescriptor desc;
  desc.interpolation = pxr::HdInterpolationVertex;
  PrimvarMap primvars;
  primvars[HdTokens->points] = {VtValue(points), desc};

  // The dangling edge yields no triangle, but its records still have to be
  // present so that face_offsets covers every corner.
  VtIntArray counts{4, 2, 4};
  VtIntArray indices{0, 1, 2, 3, 0, 6, 1, 4, 5, 2};

  CornerMeshSpec spec = CornerMeshBuilder::Build(
      SdfPath("/mesh"), counts, indices, primvars, {}, {},
      /*smooth_normals=*/true, {});

  ASSERT_EQ(spec.sub_meshes.size(), 1);
  const SubMeshSpec& sub = spec.sub_meshes[0];
  EXPECT_EQ(sub.RecordCount(), 10);
  EXPECT_EQ(sub.face_offsets, (VtIntArray{0, 4, 6, 10}));
  EXPECT_EQ(sub.triangle_face, (VtIntArray{0, 0, 2, 2}));
  // Point 6 is never reached by a triangle corner, so from_corners() drops it
  // and it must not claim a position row.
  EXPECT_EQ(sub.used_vertices, (VtIntArray{0, 1, 2, 3, 4, 5}));
}

TEST(CornerMeshBuilderTest, MapsEveryInterpolationModeToIndices) {
  PrimvarMap primvars = TwoQuadPrimvars();
  const TfToken constant("cst"), uniform("uni"), vertex("vtx"),
      face_varying("fv");

  AddPrimvar(primvars, constant, VtValue(VtVec3fArray(1)),
             pxr::HdInterpolationConstant);
  AddPrimvar(primvars, uniform, VtValue(VtVec3fArray(2)),
             pxr::HdInterpolationUniform);
  AddPrimvar(primvars, vertex, VtValue(VtVec3fArray(6)),
             pxr::HdInterpolationVertex);
  AddPrimvar(primvars, face_varying, VtValue(VtVec2fArray(8)),
             pxr::HdInterpolationFaceVarying);

  const TfToken names[] = {constant, uniform, vertex, face_varying};
  CornerMeshSpec spec = BuildTwoQuads(primvars, {}, {}, names);

  ASSERT_EQ(spec.sub_meshes.size(), 1);
  const SubMeshSpec& sub = spec.sub_meshes[0];
  ASSERT_EQ(sub.attributes.size(), 4);

  auto find = [&](const TfToken& name) -> const CornerAttributeSpec* {
    for (const auto& a : sub.attributes) {
      if (a.name == name) return &a;
    }
    return nullptr;
  };

  EXPECT_EQ(find(constant)->indices, (VtIntArray{0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(find(uniform)->indices, (VtIntArray{0, 0, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(find(vertex)->indices, TwoQuadIndices());
  EXPECT_EQ(find(face_varying)->indices, (VtIntArray{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(find(face_varying)->dim, 2);
  EXPECT_EQ(find(vertex)->dim, 3);
}

TEST(CornerMeshBuilderTest, DropsPrimvarsOfTheWrongLength) {
  PrimvarMap primvars = TwoQuadPrimvars();
  // Face-varying needs 8 values (one per corner), not 6.
  AddPrimvar(primvars, kSt, VtValue(VtVec2fArray(6)),
             pxr::HdInterpolationFaceVarying);

  const TfToken names[] = {kSt};
  CornerMeshSpec spec = BuildTwoQuads(primvars, {}, {}, names);

  ASSERT_EQ(spec.sub_meshes.size(), 1);
  // Dropped rather than forwarded: an out-of-range index is fatal inside
  // from_corners(), so a malformed primvar must never reach it.
  EXPECT_TRUE(spec.sub_meshes[0].attributes.empty());
}

TEST(CornerMeshBuilderTest, IndicesStayPerSubMeshButValuesStayGlobal) {
  PrimvarMap primvars = TwoQuadPrimvars();
  AddPrimvar(primvars, kSt, VtValue(VtVec2fArray(8)),
             pxr::HdInterpolationFaceVarying);

  const SdfPath materials[] = {SdfPath("/a"), SdfPath("/b")};
  const TfToken names[] = {kSt};
  CornerMeshSpec spec =
      BuildTwoQuads(primvars, materials, VtIntArray{0, 1}, names);

  ASSERT_EQ(spec.sub_meshes.size(), 2);
  ASSERT_EQ(spec.sub_meshes[0].attributes.size(), 1);
  ASSERT_EQ(spec.sub_meshes[1].attributes.size(), 1);

  // The second sub-mesh indexes the back half of the shared, un-split array.
  EXPECT_EQ(spec.sub_meshes[0].attributes[0].indices, (VtIntArray{0, 1, 2, 3}));
  EXPECT_EQ(spec.sub_meshes[1].attributes[0].indices, (VtIntArray{4, 5, 6, 7}));
  EXPECT_EQ(spec.sub_meshes[0].attributes[0].values.Get<VtVec2fArray>().size(),
            8);
}

TEST(CornerMeshBuilderTest, EmptyPointsProduceNoSubMeshes) {
  PrimvarMap primvars;
  HdPrimvarDescriptor desc;
  desc.interpolation = pxr::HdInterpolationVertex;
  primvars[HdTokens->points] = {VtValue(VtVec3fArray()), desc};

  CornerMeshSpec spec = CornerMeshBuilder::Build(
      SdfPath("/mesh"), TwoQuadCounts(), TwoQuadIndices(), primvars, {}, {},
      /*smooth_normals=*/true, {});
  EXPECT_TRUE(spec.sub_meshes.empty());
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
