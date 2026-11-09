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

#include <cmath>
#include <cstddef>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/mesh.h>
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

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/tests/test_util.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec3f;
using pxr::HdMitsubaMesh;
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

  const VtVec3fArray hdmitsuba_normals = GeometryProcessor::ComputeNormals(
      points, face_vertex_counts, face_vertex_indices, /*left_handed=*/false);

  // Compute the normals using Mitsuba too.
  size_t vertex_count = points.size();
  // Leaving the shading normals unspecified makes Mitsuba derive them from
  // the positions and the face winding order.
  mitsuba::ref<Mesh> mitsuba_mesh = new Mesh(
      "test_mesh",
      typename Mesh::TensorXu32(
          dr::load<typename Mesh::IndexBuffer>(face_vertex_indices.data(),
                                               face_vertex_indices.size()),
          {face_count, size_t(3)}),
      typename Mesh::TensorXf32(
          dr::load<typename Mesh::FloatBuffer>(points.data(), vertex_count * 3),
          {vertex_count, size_t(3)}));

  VtVec3fArray mitsuba_computed_normals;
  const auto& mitsuba_normals_data = mitsuba_mesh->normals().array();
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
