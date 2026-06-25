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

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/tests/test_util.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec3f;
using pxr::HdMeshTopology;
using pxr::HdMitsubaMesh;
using pxr::HdPrimvarDescriptor;
using pxr::HdPrimvarRoleTokens;
using pxr::HdTokens;
using pxr::TfToken;
using pxr::VtIntArray;
using pxr::VtValue;
using pxr::VtVec3fArray;
using ::testing::Pointwise;

using Float = float;
using Spectrum = mitsuba::Color<Float, 3>;
using Scene = mitsuba::Scene<Float, Spectrum>;
static MitsubaStaticState<Scene> global_static_state;

MI_IMPORT_TYPES(Mesh)

MATCHER_P(Vec3fNear, tolerance, "") {
  return (std::abs(std::get<0>(arg)[0] - std::get<1>(arg)[0]) < tolerance) &&
         (std::abs(std::get<0>(arg)[1] - std::get<1>(arg)[1]) < tolerance) &&
         (std::abs(std::get<0>(arg)[2] - std::get<1>(arg)[2]) < tolerance);
}

TEST(HdMitsubaMeshTest, ComputeNormalsMatchesMitsuba) {
  // This checks that our custom normal computation matches Mitsuba's on a
  // simple triangle mesh.

  VtVec3fArray points = {GfVec3f(0.5, 1.0, 0.0), GfVec3f(4, 5, 6),
                         GfVec3f(7, 8, 9), GfVec3f(-1, 2, 3)};
  VtIntArray face_vertex_indices = {0, 1, 2, 0, 2, 3, 0, 3, 1};
  VtIntArray face_vertex_counts;
  size_t face_count = face_vertex_indices.size() / 3;
  for (size_t i = 0; i < face_count; ++i) {
    face_vertex_counts.push_back(3);
  }

  HdMitsubaMesh::PrimvarMap primvars;
  HdPrimvarDescriptor points_descriptor;
  points_descriptor.interpolation = pxr::HdInterpolationVertex;
  points_descriptor.role = HdPrimvarRoleTokens->point;
  primvars[HdTokens->points] = {VtValue(points), points_descriptor};
  HdMeshTopology topology(pxr::PxOsdOpenSubdivTokens->none,
                          pxr::HdTokens->rightHanded, face_vertex_counts,
                          face_vertex_indices);
  GeometryProcessor::ComputeNormals(primvars, topology);

  ASSERT_TRUE(primvars.find(HdTokens->normals) != primvars.end());
  const VtVec3fArray& hdmitsuba_normals =
      primvars.at(HdTokens->normals).value.Get<VtVec3fArray>();

  // Compute the normals using Mitsuba too.
  size_t vertex_count = points.size();
  mitsuba::Properties props;
  mitsuba::ref<Mesh> mitsuba_mesh =
      new Mesh("test_mesh", vertex_count, face_count, props, true, false);

  mitsuba_mesh->vertex_positions_buffer() =
      dr::load<typename Mesh::FloatStorage>(points.data(), vertex_count * 3);
  mitsuba_mesh->faces_buffer() = dr::load<mitsuba::DynamicBuffer<UInt32>>(
      face_vertex_indices.data(), face_vertex_indices.size());
  mitsuba_mesh->parameters_changed();

  VtVec3fArray mitsuba_computed_normals;
  const auto& mitsuba_normals_data = mitsuba_mesh->vertex_normals_buffer();
  for (size_t i = 0; i < mitsuba_normals_data.size() / 3; ++i) {
    mitsuba_computed_normals.push_back(
        GfVec3f(mitsuba_normals_data[3 * i], mitsuba_normals_data[3 * i + 1],
                mitsuba_normals_data[3 * i + 2]));
  }

  // Ensure hdmitsuba and Mitsuba computed normals are the same.
  ASSERT_EQ(hdmitsuba_normals.size(), mitsuba_computed_normals.size());
  EXPECT_THAT(hdmitsuba_normals,
              Pointwise(Vec3fNear(1e-3), mitsuba_computed_normals));
}

TEST(HdMitsubaMeshTest, SubdivisionLevelAttribute) {
  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();
  pxr::SdfPath mesh_path("/mesh");
  pxr::UsdGeomMesh usd_mesh = pxr::UsdGeomMesh::Define(stage, mesh_path);

  pxr::VtVec3fArray points = {GfVec3f(-1, -1, 1), GfVec3f(1, -1, 1),
                              GfVec3f(1, 1, 1), GfVec3f(-1, 1, 1)};
  pxr::VtIntArray face_counts = {4};
  pxr::VtIntArray face_indices = {0, 1, 2, 3};

  usd_mesh.CreatePointsAttr(pxr::VtValue(points));
  usd_mesh.CreateFaceVertexCountsAttr(pxr::VtValue(face_counts));
  usd_mesh.CreateFaceVertexIndicesAttr(pxr::VtValue(face_indices));
  usd_mesh.CreateSubdivisionSchemeAttr().Set(pxr::UsdGeomTokens->catmullClark);

  auto [render_delegate, render_index, scene_delegate, scene_manager,
        render_param] = CreateRenderDelegateStateObjects<Scene>();

  scene_delegate->Populate(stage->GetPseudoRoot());

  pxr::HdMitsubaMesh mitsuba_mesh(mesh_path);
  pxr::HdDirtyBits dirty_bits = pxr::HdChangeTracker::AllDirty;

  // By default, refinement level is 0 if not specified in DisplayStyle.
  // We'll set it to 1 via the attribute.
  usd_mesh.GetPrim()
      .CreateAttribute(pxr::TfToken("mitsuba:subdivision_level"),
                       pxr::SdfValueTypeNames->Int)
      .Set(1);

  mitsuba_mesh.Sync(scene_delegate.get(), render_param, &dirty_bits,
                    pxr::HdReprTokens->hull);

  scene_manager->CommitResources();
  Scene* scene = dynamic_cast<Scene*>(scene_manager->GetScene());
  ASSERT_NE(scene, nullptr);
  ASSERT_EQ(scene->shapes().size(), 1);

  Mesh* mitsuba_mesh_obj = dynamic_cast<Mesh*>(scene->shapes()[0].get());
  ASSERT_NE(mitsuba_mesh_obj, nullptr);

  // A single quad subdivided at level 1 Catmull-Clark should result in 4 quads.
  // These are triangulated into 8 triangles.
  EXPECT_EQ(mitsuba_mesh_obj->face_count(), 8);

  scene_delegate.reset();
  render_index.reset();
  render_delegate.reset();
}

TEST(HdMitsubaMeshTest, SubdivisionLevelReactivity) {
  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();
  pxr::SdfPath mesh_path("/mesh");
  pxr::UsdGeomMesh usd_mesh = pxr::UsdGeomMesh::Define(stage, mesh_path);

  pxr::VtVec3fArray points = {GfVec3f(-1, -1, 1), GfVec3f(1, -1, 1),
                              GfVec3f(1, 1, 1), GfVec3f(-1, 1, 1)};
  pxr::VtIntArray face_counts = {4};
  pxr::VtIntArray face_indices = {0, 1, 2, 3};

  usd_mesh.CreatePointsAttr(pxr::VtValue(points));
  usd_mesh.CreateFaceVertexCountsAttr(pxr::VtValue(face_counts));
  usd_mesh.CreateFaceVertexIndicesAttr(pxr::VtValue(face_indices));
  usd_mesh.CreateSubdivisionSchemeAttr().Set(pxr::UsdGeomTokens->catmullClark);

  auto [render_delegate, render_index, scene_delegate, scene_manager,
        render_param] = CreateRenderDelegateStateObjects<Scene>();

  scene_delegate->Populate(stage->GetPseudoRoot());

  pxr::HdMitsubaMesh mitsuba_mesh(mesh_path);
  pxr::HdDirtyBits dirty_bits = pxr::HdChangeTracker::AllDirty;

  // Set initial level to 1.
  usd_mesh.GetPrim()
      .CreateAttribute(pxr::TfToken("mitsuba:subdivision_level"),
                       pxr::SdfValueTypeNames->Int)
      .Set(1);

  mitsuba_mesh.Sync(scene_delegate.get(), render_param, &dirty_bits,
                    pxr::HdReprTokens->hull);
  scene_manager->CommitResources();

  {
    Scene* scene = dynamic_cast<Scene*>(scene_manager->GetScene());
    Mesh* mitsuba_mesh_obj = dynamic_cast<Mesh*>(scene->shapes()[0].get());
    ASSERT_EQ(mitsuba_mesh_obj->face_count(), 8);
  }

  // Change level to 2.
  usd_mesh.GetPrim()
      .GetAttribute(pxr::TfToken("mitsuba:subdivision_level"))
      .Set(2);

  // In a real scenario, UsdImaging would mark this as dirty.
  // We'll simulate it by manually marking it dirty in the render index.
  render_index->GetChangeTracker().MarkRprimDirty(
      mesh_path, pxr::HdChangeTracker::DirtyPrimvar);

  dirty_bits = render_index->GetChangeTracker().GetRprimDirtyBits(mesh_path);
  mitsuba_mesh.Sync(scene_delegate.get(), render_param, &dirty_bits,
                    pxr::HdReprTokens->hull);
  scene_manager->CommitResources();

  {
    Scene* scene = dynamic_cast<Scene*>(scene_manager->GetScene());
    Mesh* mitsuba_mesh_obj = dynamic_cast<Mesh*>(scene->shapes()[0].get());
    // At level 2, 1 quad -> 16 quads -> 32 triangles.
    EXPECT_EQ(mitsuba_mesh_obj->face_count(), 32);
  }

  scene_delegate.reset();
  render_index.reset();
  render_delegate.reset();
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
