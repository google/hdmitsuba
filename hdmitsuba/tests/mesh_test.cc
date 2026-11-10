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

#include "hdmitsuba/mesh.h"

#include <gtest/gtest.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/scene.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/tokens.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include "hdmitsuba/tests/test_util.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec3f;
using pxr::HdMitsubaMesh;
using pxr::TfToken;
using pxr::VtIntArray;
using pxr::VtValue;
using pxr::VtVec3fArray;

using Float = float;
using Spectrum = mitsuba::Color<Float, 3>;
using Scene = mitsuba::Scene<Float, Spectrum>;
static MitsubaStaticState<Scene> global_static_state;

TEST(HdMitsubaMeshTest, GeomSubsetsSkipUnusedBaseMaterialSlot) {
  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();
  pxr::UsdGeomMesh usd_mesh = pxr::UsdGeomMesh::Define(stage, SdfPath("/mesh"));
  usd_mesh.CreatePointsAttr(VtValue(VtVec3fArray{
      GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0), GfVec3f(0, 1, 0)}));
  usd_mesh.CreateFaceVertexCountsAttr(VtValue(VtIntArray{4}));
  usd_mesh.CreateFaceVertexIndicesAttr(VtValue(VtIntArray{0, 1, 2, 3}));

  // A GeomSubset covering all faces leaves the base-mesh material unused, so
  // the mesh should not be split.
  pxr::UsdGeomSubset subset = pxr::UsdGeomSubset::CreateGeomSubset(
      usd_mesh, TfToken("subset"), pxr::UsdGeomTokens->face, VtIntArray{0},
      pxr::UsdShadeTokens->materialBind);
  pxr::UsdShadeMaterialBindingAPI::Apply(subset.GetPrim())
      .Bind(pxr::UsdShadeMaterial::Define(stage, SdfPath("/Looks/Mat")));

  auto [render_delegate, render_index, scene_delegate, scene_manager,
        render_param] = CreateRenderDelegateStateObjects<Scene>();
  scene_delegate->Populate(stage->GetPseudoRoot());

  pxr::HdMitsubaMesh mitsuba_mesh(SdfPath("/mesh"));
  pxr::HdDirtyBits dirty_bits = pxr::HdChangeTracker::AllDirty;
  mitsuba_mesh.Sync(scene_delegate.get(), render_param, &dirty_bits,
                    pxr::HdReprTokens->hull);
  scene_manager->CommitResources();

  Scene* scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(scene, nullptr);
  ASSERT_EQ(scene->shapes().size(), 1);
  EXPECT_EQ(scene->shapes()[0]->id(), "/mesh");

  scene_delegate.reset();
  render_index.reset();
  render_delegate.reset();
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
