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

  std::vector<VtIntArray> fvar_topologies;
  absl::flat_hash_map<TfToken, int, TfToken::HashFunctor> fvar_map;

  SubdivisionEvaluator subdiv;
  subdiv.Initialize(topo, 1, pxr::UsdGeomTokens->catmullClark, tags,
                    fvar_topologies, fvar_map, TfToken("quad"));

  ASSERT_TRUE(subdiv.IsSubdivided());

  VtValue refined_val = subdiv.RefinePrimvar(
      VtValue(points), pxr::HdInterpolationVertex, HdTokens->points);
  ASSERT_TRUE(refined_val.IsHolding<VtVec3fArray>());
  const VtVec3fArray& refined_points = refined_val.Get<VtVec3fArray>();

  EXPECT_EQ(refined_points.size(), 13);
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
