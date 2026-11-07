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

#include "hdmitsuba/mesh/subdivision.h"

#include <vector>

#include <gtest/gtest.h>
#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec3f;
using pxr::HdTokens;
using pxr::TfToken;
using pxr::VtIntArray;
using pxr::VtValue;
using pxr::VtVec3fArray;

TEST(SubdivisionEvaluatorTest, RefinePrimvar) {
  VtIntArray face_counts = {4};
  VtIntArray face_indices = {0, 1, 2, 3};

  pxr::PxOsdSubdivTags tags(pxr::PxOsdOpenSubdivTokens->edgeAndCorner,
                            pxr::PxOsdOpenSubdivTokens->cornersPlus1, TfToken(),
                            TfToken(), VtIntArray(), VtIntArray(),
                            pxr::VtFloatArray(), VtIntArray(),
                            pxr::VtFloatArray());
  pxr::PxOsdMeshTopology topo(pxr::UsdGeomTokens->catmullClark,
                              pxr::HdTokens->rightHanded, face_counts,
                              face_indices, VtIntArray(), tags);

  VtVec3fArray points = {GfVec3f(-1, -1, 1), GfVec3f(1, -1, 1),
                         GfVec3f(1, 1, 1), GfVec3f(-1, 1, 1)};

  SubdivisionEvaluator subdiv;
  subdiv.Initialize(topo, 1, /*fvar_indices=*/{}, TfToken("quad"));

  ASSERT_TRUE(subdiv.IsSubdivided());

  VtValue refined_val = subdiv.RefinePrimvar(
      VtValue(points), pxr::HdInterpolationVertex, HdTokens->points);
  ASSERT_TRUE(refined_val.IsHolding<VtVec3fArray>());
  const VtVec3fArray& refined_points = refined_val.Get<VtVec3fArray>();

  EXPECT_EQ(refined_points.size(), 13);

  pxr::VtFloatArray scalars = {0.f, 1.f, 2.f, 3.f};
  VtValue refined_scalars_val = subdiv.RefinePrimvar(
      VtValue(scalars), pxr::HdInterpolationVertex, TfToken("scalar"));
  ASSERT_TRUE(refined_scalars_val.IsHolding<pxr::VtFloatArray>());
  EXPECT_EQ(refined_scalars_val.Get<pxr::VtFloatArray>().size(), 13);
}

TEST(SubdivisionEvaluatorTest, RefineUniformPrimvar) {
  // Two quads sharing an edge.
  VtIntArray face_counts = {4, 4};
  VtIntArray face_indices = {0, 1, 4, 3, 1, 2, 5, 4};
  pxr::PxOsdMeshTopology topo(pxr::UsdGeomTokens->catmullClark,
                              pxr::HdTokens->rightHanded, face_counts,
                              face_indices);

  SubdivisionEvaluator subdiv;
  subdiv.Initialize(topo, 1, /*fvar_indices=*/{}, TfToken("quads"));
  ASSERT_TRUE(subdiv.IsSubdivided());
  // Each quad is refined into 4 quads.
  const VtIntArray coarse_faces = {0, 1};
  const VtIntArray refined_faces =
      subdiv.MapRefinedMaterialIndices(coarse_faces);
  ASSERT_EQ(refined_faces.size(), 8);

  const pxr::VtFloatArray colors = {0.25f, 0.75f};
  VtValue refined_colors = subdiv.RefinePrimvar(
      VtValue(colors), pxr::HdInterpolationUniform, TfToken("color"));
  ASSERT_TRUE(refined_colors.IsHolding<pxr::VtFloatArray>());
  const auto& refined_color_array = refined_colors.Get<pxr::VtFloatArray>();
  ASSERT_EQ(refined_color_array.size(), refined_faces.size());

  const VtVec3fArray vectors = {GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
  VtValue refined_vectors = subdiv.RefinePrimvar(
      VtValue(vectors), pxr::HdInterpolationUniform, TfToken("vector"));
  ASSERT_TRUE(refined_vectors.IsHolding<VtVec3fArray>());
  const auto& refined_vector_array = refined_vectors.Get<VtVec3fArray>();
  ASSERT_EQ(refined_vector_array.size(), refined_faces.size());

  const pxr::VtBoolArray flags = {true, false};
  VtValue refined_flags = subdiv.RefinePrimvar(
      VtValue(flags), pxr::HdInterpolationUniform, TfToken("flag"));
  ASSERT_TRUE(refined_flags.IsHolding<pxr::VtBoolArray>());
  const auto& refined_flag_array = refined_flags.Get<pxr::VtBoolArray>();
  ASSERT_EQ(refined_flag_array.size(), refined_faces.size());

  for (size_t i = 0; i < refined_faces.size(); ++i) {
    const int parent = refined_faces[i];
    EXPECT_EQ(refined_color_array[i], colors[parent]);
    EXPECT_EQ(refined_vector_array[i], vectors[parent]);
    EXPECT_EQ(refined_flag_array[i], flags[parent]);
  }
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
