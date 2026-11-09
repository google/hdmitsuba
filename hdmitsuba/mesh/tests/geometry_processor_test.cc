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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using ::testing::ElementsAre;

const TfToken kSt("st");

template <typename T>
const T& GetPrimvar(const VertexPrimvarMap& primvars, const TfToken& name) {
  return primvars.at(name).Get<T>();
}

// Two quads sharing the edge between points 2 and 3.
const VtIntArray kQuadCounts = {4, 4};
const VtIntArray kQuadIndices = {0, 1, 2, 3, 3, 2, 4, 5};

PrimvarMap QuadPoints() {
  return {{HdTokens->points,
           {VtValue(VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
                                 GfVec3f(1, 1, 0), GfVec3f(0, 1, 0),
                                 GfVec3f(0, 2, 0), GfVec3f(1, 2, 0)}),
            HdInterpolationVertex}}};
}

TEST(GeometryProcessorTest, ExpandPrimDataSplitsUniform) {
  PrimvarMap primvars = QuadPoints();
  primvars[kSt] = {VtValue(VtFloatArray{0.25f, 0.75f}), HdInterpolationUniform};

  auto [indices, vertex_points, expanded] = GeometryProcessor::ExpandPrimData(
      kQuadCounts, kQuadIndices, /*num_points=*/6, primvars);

  // Differing uniform values split the shared points.
  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_EQ(GetPrimvar<VtVec3fArray>(expanded, HdTokens->points).size(), 8);
  EXPECT_THAT(
      GetPrimvar<VtFloatArray>(expanded, kSt),
      ElementsAre(0.25f, 0.25f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f, 0.75f));
  EXPECT_THAT(vertex_points, ElementsAre(0, 1, 2, 3, 3, 2, 4, 5));
}

TEST(GeometryProcessorTest, ExpandPrimDataDeduplicatesFaceVarying) {
  PrimvarMap primvars = QuadPoints();
  // Shared point 3 has matching UVs on both faces, shared point 2 does not.
  primvars[kSt] = {
      VtValue(VtVec2fArray{GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(1, 1),
                           GfVec2f(0, 1), GfVec2f(0, 1), GfVec2f(5, 5),
                           GfVec2f(6, 6), GfVec2f(7, 7)}),
      HdInterpolationFaceVarying};

  auto [indices, vertex_points, expanded] = GeometryProcessor::ExpandPrimData(
      kQuadCounts, kQuadIndices, /*num_points=*/6, primvars);

  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 3, 4, 5, 6));
  EXPECT_THAT(vertex_points, ElementsAre(0, 1, 2, 3, 2, 4, 5));
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
  primvars[kSt] = {VtValue(VtVec2fArray()), HdInterpolationFaceVarying};

  auto [indices, vertex_points, expanded] = GeometryProcessor::ExpandPrimData(
      kQuadCounts, kQuadIndices, /*num_points=*/6, primvars);

  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 3, 2, 4, 5));
  EXPECT_EQ(GetPrimvar<VtVec3fArray>(expanded, HdTokens->points).size(), 6);
  EXPECT_EQ(expanded.count(kSt), 0);
}

TEST(GeometryProcessorTest, TransformPrimvarsIdentityKeepsPoints) {
  const VtVec3fArray points = {GfVec3f(1, 2, 3)};
  VertexPrimvarMap primvars = {{HdTokens->points, VtValue(points)}};

  GeometryProcessor::TransformPrimvars(primvars, GfMatrix4d(1.0));

  // The identity transform shares the input buffer instead of copying it.
  EXPECT_TRUE(
      GetPrimvar<VtVec3fArray>(primvars, HdTokens->points).IsIdentical(points));
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
