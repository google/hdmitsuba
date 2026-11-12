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
#include <pxr/imaging/hd/repr.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/task.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/tokens.h>
#include <pxr/usdImaging/usdImaging/delegate.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/tests/test_util.h"
#include "hdmitsuba/traversal.h"

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

TEST(HdMitsubaMeshTest, EmitterAndShapePluginSpecsUpdateInPlaceAndFallback) {
  constexpr float kTolerance = 1e-5f;

  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();

  // 1. Mesh with standalone area emitter and nested homogeneous interior
  // medium + HG phase function.
  pxr::UsdGeomMesh usd_mesh =
      pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/World/Mesh"));
  usd_mesh.CreatePointsAttr(VtValue(VtVec3fArray{
      GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0), GfVec3f(0, 1, 0)}));
  usd_mesh.CreateFaceVertexCountsAttr(VtValue(VtIntArray{4}));
  usd_mesh.CreateFaceVertexIndicesAttr(VtValue(VtIntArray{0, 1, 2, 3}));
  pxr::UsdPrim mesh_prim = usd_mesh.GetPrim();
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:emitter:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("area"));
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:emitter:radiance:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("rgb"));
  pxr::UsdAttribute mesh_radiance_attr = mesh_prim.CreateAttribute(
      TfToken("mitsuba:emitter:radiance:value"),
      pxr::SdfValueTypeNames->Color3f);
  mesh_radiance_attr.Set(GfVec3f(1.0f, 2.0f, 3.0f));
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:shape:interior:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("homogeneous"));
  pxr::UsdAttribute medium_scale_attr = mesh_prim.CreateAttribute(
      TfToken("mitsuba:shape:interior:scale"), pxr::SdfValueTypeNames->Float);
  medium_scale_attr.Set(2.0f);
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:shape:interior:sigma_t:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("rgb"));
  pxr::UsdAttribute medium_sigma_t_attr = mesh_prim.CreateAttribute(
      TfToken("mitsuba:shape:interior:sigma_t:value"),
      pxr::SdfValueTypeNames->Color3f);
  medium_sigma_t_attr.Set(GfVec3f(0.2f, 0.4f, 0.6f));
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:shape:interior:phase:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("hg"));
  pxr::UsdAttribute phase_g_attr = mesh_prim.CreateAttribute(
      TfToken("mitsuba:shape:interior:phase:g"), pxr::SdfValueTypeNames->Float);
  phase_g_attr.Set(0.25f);

  // 2. Implicit sphere with `mitsuba:shape:to_world`.
  pxr::UsdGeomSphere usd_sphere =
      pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/World/Sphere"));
  usd_sphere.CreateRadiusAttr().Set(1.0);
  pxr::UsdPrim sphere_prim = usd_sphere.GetPrim();
  pxr::UsdAttribute sphere_to_world_attr = sphere_prim.CreateAttribute(
      TfToken("mitsuba:shape:to_world"), pxr::SdfValueTypeNames->Matrix4d);
  sphere_to_world_attr.Set(pxr::GfMatrix4d(1.0));

  // 3. Point SphereLight overridden to `mitsuba:emitter:type = "spot"` without
  // explicit `beam_width` (exercises SpotLight default beam_width = 0.75 *
  // cutoff_angle).
  pxr::UsdPrim spot_prim =
      stage->DefinePrim(pxr::SdfPath("/World/Spot"), TfToken("SphereLight"));
  spot_prim
      .CreateAttribute(TfToken("inputs:radius"), pxr::SdfValueTypeNames->Float)
      .Set(0.0f);
  spot_prim
      .CreateAttribute(TfToken("inputs:intensity"),
                       pxr::SdfValueTypeNames->Float)
      .Set(5.0f);
  spot_prim
      .CreateAttribute(TfToken("mitsuba:emitter:type"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("spot"));
  pxr::UsdAttribute spot_cutoff_attr = spot_prim.CreateAttribute(
      TfToken("mitsuba:emitter:cutoff_angle"), pxr::SdfValueTypeNames->Float);
  spot_cutoff_attr.Set(30.0f);

  auto render_delegate = std::make_unique<pxr::HdMitsubaRenderDelegate>();
  std::unique_ptr<pxr::HdRenderIndex> render_index(
      pxr::HdRenderIndex::New(render_delegate.get(), {}));
  ASSERT_NE(render_index, nullptr);

  pxr::UsdImagingCreateSceneIndicesInfo create_info;
  create_info.stage = stage;
  const pxr::UsdImagingSceneIndices scene_indices =
      pxr::UsdImagingCreateSceneIndices(create_info);
  render_index->InsertSceneIndex(scene_indices.finalSceneIndex,
                                 pxr::SdfPath::AbsoluteRootPath());

  class SyncGeometryTask final : public pxr::HdTask {
   public:
    explicit SyncGeometryTask(pxr::HdRenderIndex* index)
        : pxr::HdTask(pxr::SdfPath::EmptyPath()),
          index_(index),
          collection_(pxr::HdTokens->geometry,
                      pxr::HdReprSelector(pxr::HdReprTokens->hull)),
          tags_({pxr::HdRenderTagTokens->geometry}) {
      collection_.SetRootPath(pxr::SdfPath::AbsoluteRootPath());
    }
    void Sync(pxr::HdSceneDelegate*, pxr::HdTaskContext*,
              pxr::HdDirtyBits*) override {
      index_->EnqueueCollectionToSync(collection_);
    }
    void Prepare(pxr::HdTaskContext*, pxr::HdRenderIndex*) override {}
    void Execute(pxr::HdTaskContext*) override {}
    const pxr::TfTokenVector& GetRenderTags() const override { return tags_; }

   private:
    pxr::HdRenderIndex* index_;
    pxr::HdRprimCollection collection_;
    pxr::TfTokenVector tags_;
  };
  auto sync_task = std::make_shared<SyncGeometryTask>(render_index.get());

  pxr::SceneManager* scene_manager =
      static_cast<pxr::HdMitsubaRenderParam*>(render_delegate->GetRenderParam())
          ->GetScene();
  auto sync_and_commit = [&]() {
    scene_indices.stageSceneIndex->ApplyPendingUpdates();
    pxr::HdTaskSharedPtrVector tasks = {sync_task};
    pxr::HdTaskContext task_context;
    render_index->SyncAll(&tasks, &task_context);
    scene_manager->CommitResources();
  };

  sync_and_commit();

  Scene* initial_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(initial_scene, nullptr);

  auto find_shape =
      [](Scene* s,
         std::string_view id) -> const mitsuba::Shape<Float, Spectrum>* {
    for (const auto& shape : s->shapes()) {
      if (shape->id() == id) return shape.get();
    }
    return nullptr;
  };
  auto find_emitter =
      [](Scene* s,
         std::string_view id) -> const mitsuba::Emitter<Float, Spectrum>* {
    for (const auto& em : s->emitters()) {
      if (em->id() == id) return em.get();
    }
    return nullptr;
  };

  const auto* initial_mesh_shape = find_shape(initial_scene, "/World/Mesh");
  const auto* initial_sphere_shape = find_shape(initial_scene, "/World/Sphere");
  const auto* initial_spot_emitter = find_emitter(initial_scene, "/World/Spot");
  ASSERT_NE(initial_mesh_shape, nullptr);
  ASSERT_NE(initial_sphere_shape, nullptr);
  ASSERT_NE(initial_spot_emitter, nullptr);
  const auto* initial_mesh_emitter = initial_mesh_shape->emitter();
  ASSERT_NE(initial_mesh_emitter, nullptr);

  // Mutate leaf parameters across all three prims: Scene, Shape, and Emitter
  // instances must update in-place via TraversalCallback + parameters_changed.
  mesh_radiance_attr.Set(GfVec3f(4.0f, 5.0f, 6.0f));
  medium_scale_attr.Set(7.5f);
  medium_sigma_t_attr.Set(GfVec3f(0.7f, 0.3f, 0.1f));
  phase_g_attr.Set(-0.5f);
  pxr::GfMatrix4d translated(1.0);
  translated.SetTranslate(pxr::GfVec3d(2.0, 0.0, 0.0));
  sphere_to_world_attr.Set(translated);
  spot_cutoff_attr.Set(12.0f);

  sync_and_commit();

  EXPECT_EQ(scene_manager->GetScene(), initial_scene);
  EXPECT_EQ(find_shape(initial_scene, "/World/Mesh"), initial_mesh_shape);
  EXPECT_EQ(initial_mesh_shape->emitter(), initial_mesh_emitter);
  EXPECT_EQ(find_shape(initial_scene, "/World/Sphere"), initial_sphere_shape);
  EXPECT_EQ(find_emitter(initial_scene, "/World/Spot"), initial_spot_emitter);

  pxr::TraversalCallback mesh_cb;
  const_cast<mitsuba::Shape<Float, Spectrum>*>(initial_mesh_shape)
      ->traverse(&mesh_cb);
  Spectrum* radiance = mesh_cb.get<Spectrum>("emitter.radiance.value");
  ASSERT_NE(radiance, nullptr);
  EXPECT_NEAR((*radiance)[0], 4.0f, kTolerance);
  EXPECT_NEAR((*radiance)[1], 5.0f, kTolerance);
  EXPECT_NEAR((*radiance)[2], 6.0f, kTolerance);
  Float* scale = mesh_cb.get<Float>("interior_medium.scale");
  ASSERT_NE(scale, nullptr);
  EXPECT_NEAR(*scale, 7.5f, kTolerance);
  Spectrum* sigma_t =
      mesh_cb.get<Spectrum>("interior_medium.sigma_t.value.value");
  ASSERT_NE(sigma_t, nullptr);
  EXPECT_NEAR((*sigma_t)[0], 0.7f, kTolerance);
  EXPECT_NEAR((*sigma_t)[1], 0.3f, kTolerance);
  EXPECT_NEAR((*sigma_t)[2], 0.1f, kTolerance);
  Float* g = mesh_cb.get<Float>("interior_medium.phase_function.g");
  ASSERT_NE(g, nullptr);
  EXPECT_NEAR(*g, -0.5f, kTolerance);

  pxr::TraversalCallback spot_cb;
  const_cast<mitsuba::Emitter<Float, Spectrum>*>(initial_spot_emitter)
      ->traverse(&spot_cb);
  Float* cutoff = spot_cb.get<Float>("cutoff_angle");
  Float* beam = spot_cb.get<Float>("beam_width");
  ASSERT_NE(cutoff, nullptr);
  ASSERT_NE(beam, nullptr);
  EXPECT_NEAR(*cutoff, 12.0f, kTolerance);
  EXPECT_NEAR(*beam, 9.0f, kTolerance);

  // Constructor-only sphere override (`mitsuba:shape:radius`) must rebuild only
  // `/World/Sphere`, preserving `/World/Mesh` and `/World/Spot` instances.
  sphere_prim
      .CreateAttribute(TfToken("mitsuba:shape:radius"),
                       pxr::SdfValueTypeNames->Float)
      .Set(2.5f);
  sync_and_commit();

  Scene* rebuilt_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(rebuilt_scene, nullptr);
  EXPECT_EQ(find_shape(rebuilt_scene, "/World/Mesh"), initial_mesh_shape);
  EXPECT_EQ(find_emitter(rebuilt_scene, "/World/Spot"), initial_spot_emitter);
  EXPECT_NE(find_shape(rebuilt_scene, "/World/Sphere"), initial_sphere_shape);

  // Authored `mitsuba:shape:id` must be honored on both Mesh and implicit
  // Sphere shapes without breaking in-place updates or removal in `shapes_`.
  sphere_prim.RemoveProperty(TfToken("mitsuba:shape:radius"));
  mesh_prim
      .CreateAttribute(TfToken("mitsuba:shape:id"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("custom_mesh_id"));
  sphere_prim
      .CreateAttribute(TfToken("mitsuba:shape:id"),
                       pxr::SdfValueTypeNames->String)
      .Set(std::string("custom_sphere_id"));
  sync_and_commit();

  mitsuba::ref<Scene> id_scene =
      dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(id_scene.get(), nullptr);
  const auto* custom_mesh_shape = find_shape(id_scene.get(), "custom_mesh_id");
  const auto* custom_sphere_shape =
      find_shape(id_scene.get(), "custom_sphere_id");
  ASSERT_NE(custom_mesh_shape, nullptr);
  ASSERT_NE(custom_sphere_shape, nullptr);
  EXPECT_EQ(find_shape(id_scene.get(), "/World/Mesh"), nullptr);
  EXPECT_EQ(find_shape(id_scene.get(), "/World/Sphere"), nullptr);

  // Subsequent in-place updates on custom-ID shapes must find them by USD path.
  medium_scale_attr.Set(3.0f);
  translated.SetTranslate(pxr::GfVec3d(4.0, 0.0, 0.0));
  sphere_to_world_attr.Set(translated);
  sync_and_commit();
  EXPECT_EQ(static_cast<const void*>(scene_manager->GetScene()),
            static_cast<const void*>(id_scene.get()));
  EXPECT_EQ(find_shape(id_scene.get(), "custom_mesh_id"), custom_mesh_shape);
  EXPECT_EQ(find_shape(id_scene.get(), "custom_sphere_id"),
            custom_sphere_shape);

  // Deactivating custom-ID shapes must cleanly remove them from the scene.
  mesh_prim.SetActive(false);
  sphere_prim.SetActive(false);
  sync_and_commit();
  Scene* empty_shapes_scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(empty_shapes_scene, nullptr);
  EXPECT_EQ(find_shape(empty_shapes_scene, "custom_mesh_id"), nullptr);
  EXPECT_EQ(find_shape(empty_shapes_scene, "custom_sphere_id"), nullptr);

  render_index.reset();
  render_delegate.reset();
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
