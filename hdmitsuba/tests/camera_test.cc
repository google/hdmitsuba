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

#include "hdmitsuba/camera.h"

#include <drjit/matrix.h>
#include <gtest/gtest.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/scene.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/tests/test_util.h"
#include "hdmitsuba/traversal.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

constexpr float kTolerance = 1e-6f;

using Float = float;
using Scene = mitsuba::Scene<Float, mitsuba::Color<Float, 3>>;
static MitsubaStaticState<Scene> global_static_state;

TEST(HdMitsubaCameraTest, Sync) {
  using ScalarFloat = float;

  // Create a render delegate and a render index.

  // Create a camera prim in the USD stage.
  UsdStageRefPtr stage = UsdStage::CreateInMemory();
  SdfPath camera_path("/World/camera");
  UsdGeomCamera camera = UsdGeomCamera::Define(stage, camera_path);
  camera.GetProjectionAttr().Set(UsdGeomTokens->perspective);
  camera.GetFocalLengthAttr().Set(50.0f);
  camera.GetHorizontalApertureAttr().Set(36.0f);
  camera.GetVerticalApertureAttr().Set(24.0f);
  camera.GetClippingRangeAttr().Set(GfVec2f(0.5f, 50.0f));

  // Create a scene delegate and populate it with the camera prim.
  auto [render_delegate, render_index, scene_delegate, scene_manager,
        render_param] = CreateRenderDelegateStateObjects<Scene>();

  scene_delegate->Populate(stage->GetPseudoRoot());

  // Check camera sync/conversion.
  HdMitsubaCamera mitsuba_camera(camera_path);

  HdDirtyBits dirty_bits = HdCamera::DirtyBits::AllDirty;

  mitsuba_camera.Sync(scene_delegate.get(), render_param, &dirty_bits);
  EXPECT_EQ(dirty_bits, HdChangeTracker::Clean);

  scene_manager->CommitResources();  // Creates the scene.

  Scene* scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  EXPECT_NE(scene, nullptr);
  EXPECT_EQ(scene->sensors().size(), 1);
  EXPECT_EQ(scene->sensors()[0]->class_name(), "PerspectiveCamera");
  EXPECT_EQ(scene->sensors()[0]->id(), "/World/camera");
  TraversalCallback cb;
  scene->traverse(&cb);

  Float* value = cb.get<Float>("/World/camera.x_fov");
  EXPECT_EQ(*value, 39.5977554f);
  ScalarFloat* near_clip = cb.get<ScalarFloat>("/World/camera.near_clip");
  EXPECT_EQ(*near_clip, 0.5f);
  ScalarFloat* far_clip = cb.get<ScalarFloat>("/World/camera.far_clip");
  EXPECT_EQ(*far_clip, 50.0f);

  mitsuba::AffineTransform<mitsuba::Point<Float, 3>> transform =
      scene->sensors()[0]->world_transform();
  drjit::Matrix<Float, 4> matrix = transform.matrix;
  EXPECT_NEAR(matrix[0][0], -1.0f, kTolerance);
  EXPECT_NEAR(matrix[0][1], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[0][2], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[0][3], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[1][0], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[1][1], 1.0f, kTolerance);
  EXPECT_NEAR(matrix[1][2], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[1][3], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[2][0], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[2][1], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[2][2], -1.0f, kTolerance);
  EXPECT_NEAR(matrix[2][3], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[3][0], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[3][1], 0.0f, kTolerance);
  EXPECT_NEAR(matrix[3][2], 0.0f, kTolerance);
  scene_delegate.reset();
  render_index.reset();
  render_delegate.reset();
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
