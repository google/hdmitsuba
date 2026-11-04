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
#include <pxr/imaging/hd/task.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usdImaging/usdImaging/delegate.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

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

TEST(HdMitsubaCameraTest, SensorPluginSpecInPlaceAndRebuildFallback) {
  using ScalarFloat = float;

  UsdStageRefPtr stage = UsdStage::CreateInMemory();
  SdfPath camera_path("/World/camera");
  UsdGeomCamera camera = UsdGeomCamera::Define(stage, camera_path);
  camera.GetProjectionAttr().Set(UsdGeomTokens->perspective);
  camera.GetFocalLengthAttr().Set(50.0f);
  camera.GetHorizontalApertureAttr().Set(36.0f);
  camera.GetVerticalApertureAttr().Set(24.0f);
  camera.GetClippingRangeAttr().Set(GfVec2f(0.5f, 50.0f));

  UsdPrim cam_prim = camera.GetPrim();
  UsdAttribute type_attr = cam_prim.CreateAttribute(
      TfToken("mitsuba:sensor:type"), SdfValueTypeNames->String);
  type_attr.Set(std::string("thinlens"));
  UsdAttribute aperture_attr = cam_prim.CreateAttribute(
      TfToken("mitsuba:sensor:aperture_radius"), SdfValueTypeNames->Float);
  aperture_attr.Set(0.05f);
  UsdAttribute focus_attr = cam_prim.CreateAttribute(
      TfToken("mitsuba:sensor:focus_distance"), SdfValueTypeNames->Float);
  focus_attr.Set(3.0f);
  UsdAttribute fov_attr = cam_prim.CreateAttribute(
      TfToken("mitsuba:sensor:fov"), SdfValueTypeNames->Float);
  fov_attr.Set(45.0f);

  auto render_delegate = std::make_unique<HdMitsubaRenderDelegate>();
  std::unique_ptr<HdRenderIndex> render_index(
      HdRenderIndex::New(render_delegate.get(), {}));
  ASSERT_NE(render_index, nullptr);

  UsdImagingCreateSceneIndicesInfo create_info;
  create_info.stage = stage;
  const UsdImagingSceneIndices scene_indices =
      UsdImagingCreateSceneIndices(create_info);
  render_index->InsertSceneIndex(scene_indices.finalSceneIndex,
                                 SdfPath::AbsoluteRootPath());

  SceneManager* scene_manager =
      static_cast<HdMitsubaRenderParam*>(render_delegate->GetRenderParam())
          ->GetScene();
  auto sync_and_commit = [&]() {
    scene_indices.stageSceneIndex->ApplyPendingUpdates();
    HdTaskSharedPtrVector tasks;
    HdTaskContext task_context;
    render_index->SyncAll(&tasks, &task_context);
    scene_manager->CommitResources();
  };

  sync_and_commit();

  Scene* initial_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(initial_scene, nullptr);
  ASSERT_EQ(initial_scene->sensors().size(), 1);
  const mitsuba::Sensor<Float, mitsuba::Color<Float, 3>>* initial_sensor =
      initial_scene->sensors()[0].get();
  ASSERT_EQ(initial_sensor->class_name(), "ThinLensCamera");

  // Mutate traversable plugin leaves and USD clipping planes: both the Scene
  // and the ThinLensCamera instance must update in-place without rebuilding.
  aperture_attr.Set(0.25f);
  focus_attr.Set(6.5f);
  fov_attr.Set(60.0f);
  camera.GetClippingRangeAttr().Set(GfVec2f(0.25f, 120.0f));

  sync_and_commit();

  EXPECT_EQ(scene_manager->GetScene(), initial_scene);
  EXPECT_EQ(initial_scene->sensors()[0].get(), initial_sensor);

  TraversalCallback cb;
  initial_scene->traverse(&cb);
  Float* aperture = cb.get<Float>("/World/camera.aperture_radius");
  ASSERT_NE(aperture, nullptr);
  EXPECT_NEAR(*aperture, 0.25f, kTolerance);
  Float* focus = cb.get<Float>("/World/camera.focus_distance");
  ASSERT_NE(focus, nullptr);
  EXPECT_NEAR(*focus, 6.5f, kTolerance);
  Float* x_fov = cb.get<Float>("/World/camera.x_fov");
  ASSERT_NE(x_fov, nullptr);
  EXPECT_NEAR(*x_fov, 60.0f, kTolerance);
  ScalarFloat* near_clip = cb.get<ScalarFloat>("/World/camera.near_clip");
  ASSERT_NE(near_clip, nullptr);
  EXPECT_NEAR(*near_clip, 0.25f, kTolerance);
  ScalarFloat* far_clip = cb.get<ScalarFloat>("/World/camera.far_clip");
  ASSERT_NE(far_clip, nullptr);
  EXPECT_NEAR(*far_clip, 120.0f, kTolerance);

  // Structural change (switching plugin_type and removing attributes) must
  // rebuild the sensor cleanly.
  type_attr.Set(std::string("perspective"));
  cam_prim.RemoveProperty(TfToken("mitsuba:sensor:aperture_radius"));
  cam_prim.RemoveProperty(TfToken("mitsuba:sensor:focus_distance"));
  cam_prim.CreateAttribute(TfToken("mitsuba:sensor:film:pixel_filter:type"),
                           SdfValueTypeNames->String)
      .Set(std::string("gaussian"));
  UsdAttribute stddev_attr = cam_prim.CreateAttribute(
      TfToken("mitsuba:sensor:film:pixel_filter:stddev"),
      SdfValueTypeNames->Float);
  stddev_attr.Set(0.5f);

  sync_and_commit();

  Scene* rebuilt_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(rebuilt_scene, nullptr);
  const auto* perspective_sensor = rebuilt_scene->sensors()[0].get();
  EXPECT_NE(perspective_sensor, initial_sensor);
  EXPECT_EQ(perspective_sensor->class_name(), "PerspectiveCamera");

  // Non-traversable constructor-only leaf (GaussianFilter stddev) must fall
  // back to rebuilding the sensor when modified.
  stddev_attr.Set(1.5f);
  sync_and_commit();

  Scene* filter_rebuilt_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(filter_rebuilt_scene, nullptr);
  EXPECT_NE(filter_rebuilt_scene->sensors()[0].get(), perspective_sensor);

  render_index.reset();
  render_delegate.reset();
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
