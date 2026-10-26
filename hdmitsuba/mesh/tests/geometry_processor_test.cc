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

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pxr/base/gf/matrix4d.h>
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

using ::testing::ElementsAre;

const TfToken kSt("st");

void AddPrimvar(PrimvarMap& primvars, const TfToken& name, VtValue value,
                HdInterpolation interpolation, const TfToken& role = {}) {
  primvars[name] = {std::move(value),
                    HdPrimvarDescriptor(name, interpolation, role)};
}

template <typename T>
const T& GetPrimvar(const PrimvarMap& primvars, const TfToken& name) {
  return primvars.at(name).value.Get<T>();
}

// Two quads sharing the edge between points 2 and 3.
const VtIntArray kQuadCounts = {4, 4};
const VtIntArray kQuadIndices = {0, 1, 2, 3, 3, 2, 4, 5};

PrimvarMap QuadPoints() {
  PrimvarMap primvars;
  AddPrimvar(primvars, HdTokens->points,
             VtValue(VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
                                  GfVec3f(1, 1, 0), GfVec3f(0, 1, 0),
                                  GfVec3f(0, 2, 0), GfVec3f(1, 2, 0)}),
             HdInterpolationVertex, HdPrimvarRoleTokens->point);
  return primvars;
}

TEST(GeometryProcessorTest, TriangulateWithFaceMapping) {
  auto [triangles, primitive_params] =
      GeometryProcessor::TriangulateWithFaceMapping(
          VtIntArray{3, 4, 5},
          VtIntArray{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  EXPECT_THAT(triangles, ElementsAre(0, 1, 2,           //
                                     3, 4, 5, 3, 5, 6,  //
                                     7, 8, 9, 7, 9, 10, 7, 10, 11));
  EXPECT_THAT(primitive_params, ElementsAre(0, 1, 1, 2, 2, 2));
}

TEST(GeometryProcessorTest, ExpandPrimDataSplitsUniform) {
  PrimvarMap primvars = QuadPoints();
  AddPrimvar(primvars, kSt, VtValue(VtFloatArray{0.25f, 0.75f}),
             HdInterpolationUniform);

  auto [indices, expanded] =
      GeometryProcessor::ExpandPrimData(kQuadIndices, kQuadCounts, primvars);

  // Differing uniform values split the shared points.
  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_EQ(GetPrimvar<VtVec3fArray>(expanded, HdTokens->points).size(), 8);
  EXPECT_THAT(
      GetPrimvar<VtFloatArray>(expanded, kSt),
      ElementsAre(0.25f, 0.25f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f, 0.75f));
}

TEST(GeometryProcessorTest, ExpandPrimDataDeduplicatesFaceVarying) {
  PrimvarMap primvars = QuadPoints();
  // Shared point 3 has matching UVs on both faces, shared point 2 does not.
  AddPrimvar(primvars, kSt,
             VtValue(VtVec2fArray{GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(1, 1),
                                  GfVec2f(0, 1), GfVec2f(0, 1), GfVec2f(5, 5),
                                  GfVec2f(6, 6), GfVec2f(7, 7)}),
             HdInterpolationFaceVarying);

  auto [indices, expanded] =
      GeometryProcessor::ExpandPrimData(kQuadIndices, kQuadCounts, primvars);

  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 3, 4, 5, 6));
  EXPECT_THAT(GetPrimvar<VtVec3fArray>(expanded, HdTokens->points),
              ElementsAre(GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0),
                          GfVec3f(0, 1, 0), GfVec3f(1, 1, 0), GfVec3f(0, 2, 0),
                          GfVec3f(1, 2, 0)));
  EXPECT_THAT(
      GetPrimvar<VtVec2fArray>(expanded, kSt),
      ElementsAre(GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(1, 1), GfVec2f(0, 1),
                  GfVec2f(5, 5), GfVec2f(6, 6), GfVec2f(7, 7)));
}

TEST(GeometryProcessorTest, ExpandPrimDataIgnoresEmptyFaceVarying) {
  PrimvarMap primvars = QuadPoints();
  AddPrimvar(primvars, kSt, VtValue(VtVec2fArray()),
             HdInterpolationFaceVarying);

  auto [indices, expanded] =
      GeometryProcessor::ExpandPrimData(kQuadIndices, kQuadCounts, primvars);

  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 3, 2, 4, 5));
  EXPECT_EQ(GetPrimvar<VtVec3fArray>(expanded, HdTokens->points).size(), 6);
  EXPECT_EQ(expanded.count(kSt), 0);
}

TEST(GeometryProcessorTest, SplitAndCompactMeshes) {
  PrimvarMap primvars;
  AddPrimvar(primvars, HdTokens->points,
             VtValue(VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
                                  GfVec3f(0, 1, 0), GfVec3f(10, 10, 10),
                                  GfVec3f(11, 10, 10), GfVec3f(10, 11, 10)}),
             HdInterpolationVertex, HdPrimvarRoleTokens->point);
  AddPrimvar(primvars, kSt, VtValue(VtFloatArray{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}),
             HdInterpolationVertex);

  auto sub_meshes = GeometryProcessor::SplitAndCompactMeshes(
      SdfPath("/mesh"), VtIntArray{0, 1, 2, 3, 4, 5}, VtIntArray{0, 1},
      primvars, {SdfPath("/mat1"), SdfPath("/mat2")}, VtIntArray{0, 1});

  ASSERT_EQ(sub_meshes.size(), 2);
  EXPECT_EQ(sub_meshes[0].material_id, SdfPath("/mat1"));
  EXPECT_THAT(sub_meshes[0].triangles, ElementsAre(0, 1, 2));
  EXPECT_THAT(
      GetPrimvar<VtVec3fArray>(sub_meshes[0].primvars, HdTokens->points),
      ElementsAre(GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)));
  EXPECT_THAT(GetPrimvar<VtFloatArray>(sub_meshes[0].primvars, kSt),
              ElementsAre(1.f, 2.f, 3.f));

  EXPECT_EQ(sub_meshes[1].material_id, SdfPath("/mat2"));
  EXPECT_THAT(sub_meshes[1].triangles, ElementsAre(0, 1, 2));
  EXPECT_THAT(
      GetPrimvar<VtVec3fArray>(sub_meshes[1].primvars, HdTokens->points),
      ElementsAre(GfVec3f(10, 10, 10), GfVec3f(11, 10, 10),
                  GfVec3f(10, 11, 10)));
  EXPECT_THAT(GetPrimvar<VtFloatArray>(sub_meshes[1].primvars, kSt),
              ElementsAre(4.f, 5.f, 6.f));
}

TEST(GeometryProcessorTest, TransformPrimvars) {
  PrimvarMap primvars;
  AddPrimvar(primvars, HdTokens->points,
             VtValue(VtVec3fArray{GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)}),
             HdInterpolationVertex, HdPrimvarRoleTokens->point);
  AddPrimvar(primvars, HdTokens->normals,
             VtValue(VtVec3fArray{GfVec3f(1, 1, 0)}), HdInterpolationVertex,
             HdPrimvarRoleTokens->normal);
  AddPrimvar(primvars, kSt, VtValue(VtVec2fArray{GfVec2f(0.25f, 0.75f)}),
             HdInterpolationFaceVarying,
             HdPrimvarRoleTokens->textureCoordinate);

  GfMatrix4d transform(1.0);
  transform.SetScale(GfVec3d(2.0, 0.5, 1.0));
  GeometryProcessor::TransformPrimvars(primvars, transform);

  EXPECT_THAT(GetPrimvar<VtVec3fArray>(primvars, HdTokens->points),
              ElementsAre(GfVec3f(2, 0, 0), GfVec3f(0, 0.5f, 0)));
  // Normals transform by the inverse transpose, i.e. scale (0.5, 2, 1).
  const GfVec3f normal =
      GetPrimvar<VtVec3fArray>(primvars, HdTokens->normals)[0];
  EXPECT_TRUE(GfIsClose(normal, GfVec3f(0.5f, 2, 0).GetNormalized(), 1e-6))
      << normal;
  // Texture coordinates are flipped vertically.
  EXPECT_THAT(GetPrimvar<VtVec2fArray>(primvars, kSt),
              ElementsAre(GfVec2f(0.25f, 0.25f)));
}

TEST(GeometryProcessorTest, TransformPrimvarsIdentityKeepsPoints) {
  PrimvarMap primvars;
  const VtVec3fArray points = {GfVec3f(1, 2, 3)};
  AddPrimvar(primvars, HdTokens->points, VtValue(points), HdInterpolationVertex,
             HdPrimvarRoleTokens->point);

  GeometryProcessor::TransformPrimvars(primvars, GfMatrix4d(1.0));

  // The identity transform shares the input buffer instead of copying it.
  EXPECT_TRUE(
      GetPrimvar<VtVec3fArray>(primvars, HdTokens->points).IsIdentical(points));
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
