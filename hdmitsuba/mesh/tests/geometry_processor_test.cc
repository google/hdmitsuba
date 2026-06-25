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

#include <vector>

#include <gtest/gtest.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/enums.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include "hdmitsuba/mesh.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using pxr::GfVec3f;
using pxr::HdMeshTopology;
using pxr::HdMitsubaMesh;
using pxr::HdPrimvarDescriptor;
using pxr::HdPrimvarRoleTokens;
using pxr::HdTokens;
using pxr::SdfPath;
using pxr::TfToken;
using pxr::VtIntArray;
using pxr::VtValue;
using pxr::VtVec3fArray;

TEST(GeometryProcessorTest, TriangulateWithFaceMapping) {
  VtIntArray face_vertex_counts = {3, 4, 5};
  VtIntArray face_vertex_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto [triangles, primitive_params] =
      GeometryProcessor::TriangulateWithFaceMapping(face_vertex_counts,
                                                    face_vertex_indices);

  EXPECT_EQ(triangles.size(), 18);
  ASSERT_EQ(primitive_params.size(), 6);

  EXPECT_EQ(primitive_params[0], 0);
  EXPECT_EQ(primitive_params[1], 1);
  EXPECT_EQ(primitive_params[2], 1);
  EXPECT_EQ(primitive_params[3], 2);
  EXPECT_EQ(primitive_params[4], 2);
  EXPECT_EQ(primitive_params[5], 2);
}

TEST(GeometryProcessorTest, ExpandPrimData) {
  VtIntArray face_vertex_counts = {4, 4};
  VtIntArray face_vertex_indices = {0, 1, 2, 3, 3, 2, 4, 5};

  HdMeshTopology topology(pxr::PxOsdOpenSubdivTokens->none,
                          pxr::HdTokens->rightHanded, face_vertex_counts,
                          face_vertex_indices);

  HdMitsubaMesh::PrimvarMap primvars;
  HdPrimvarDescriptor points_desc;
  points_desc.name = HdTokens->points;
  points_desc.interpolation = pxr::HdInterpolationVertex;
  points_desc.role = HdPrimvarRoleTokens->point;
  VtVec3fArray points = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0),
                         GfVec3f(0, 1, 0), GfVec3f(2, 1, 0), GfVec3f(2, 0, 0)};
  primvars[HdTokens->points] = {VtValue(points), points_desc};

  auto [expanded_indices, expanded_primvars] =
      GeometryProcessor::ExpandPrimData(topology, primvars);

  EXPECT_EQ(expanded_indices.size(), 8);
  ASSERT_TRUE(expanded_primvars.find(HdTokens->points) != expanded_primvars.end());
  const VtVec3fArray& expanded_points =
      expanded_primvars.at(HdTokens->points).value.Get<VtVec3fArray>();
  EXPECT_EQ(expanded_points.size(), 6);
}

TEST(GeometryProcessorTest, SplitAndCompactMeshes) {
  SdfPath base_id("/mesh");
  VtIntArray triangles = {0, 1, 2, 3, 4, 5};
  VtIntArray primitive_params = {0, 1};

  HdMitsubaMesh::PrimvarMap primvars;
  HdPrimvarDescriptor points_desc;
  points_desc.name = HdTokens->points;
  points_desc.interpolation = pxr::HdInterpolationVertex;
  points_desc.role = HdPrimvarRoleTokens->point;
  VtVec3fArray points = {GfVec3f(0, 0, 0),    GfVec3f(1, 0, 0),
                         GfVec3f(0, 1, 0),    GfVec3f(10, 10, 10),
                         GfVec3f(11, 10, 10), GfVec3f(10, 11, 10)};
  primvars[HdTokens->points] = {VtValue(points), points_desc};

  std::vector<SdfPath> material_ids = {SdfPath("/mat1"), SdfPath("/mat2")};
  VtIntArray face_material_indices = {0, 1};
  auto sub_meshes = GeometryProcessor::SplitAndCompactMeshes(
      base_id, triangles, primitive_params, primvars, material_ids,
      face_material_indices);

  ASSERT_EQ(sub_meshes.size(), 2);

  EXPECT_EQ(sub_meshes[0].material_id, SdfPath("/mat1"));
  EXPECT_EQ(sub_meshes[0].triangles.size(), 3);
  const VtVec3fArray& sub0_points =
      sub_meshes[0].primvars.at(HdTokens->points).value.Get<VtVec3fArray>();
  ASSERT_EQ(sub0_points.size(), 3);
  EXPECT_EQ(sub0_points[0], GfVec3f(0, 0, 0));

  EXPECT_EQ(sub_meshes[1].material_id, SdfPath("/mat2"));
  EXPECT_EQ(sub_meshes[1].triangles.size(), 3);
  const VtVec3fArray& sub1_points =
      sub_meshes[1].primvars.at(HdTokens->points).value.Get<VtVec3fArray>();
  ASSERT_EQ(sub1_points.size(), 3);
  EXPECT_EQ(sub1_points[0], GfVec3f(10, 10, 10));
}

TEST(GeometryProcessorTest, TransformPrimvars) {
  HdMitsubaMesh::PrimvarMap primvars;
  HdPrimvarDescriptor points_desc;
  points_desc.name = HdTokens->points;
  points_desc.role = HdPrimvarRoleTokens->point;
  VtVec3fArray points = {GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
  primvars[HdTokens->points] = {VtValue(points), points_desc};

  HdPrimvarDescriptor normals_desc;
  normals_desc.name = HdTokens->normals;
  normals_desc.role = HdPrimvarRoleTokens->normal;
  VtVec3fArray normals = {GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
  primvars[HdTokens->normals] = {VtValue(normals), normals_desc};

  pxr::GfMatrix4d transform;
  transform.SetScale(pxr::GfVec3d(2.0, 0.5, 1.0));

  GeometryProcessor::TransformPrimvars(primvars, transform);

  const VtVec3fArray& transformed_points =
      primvars.at(HdTokens->points).value.Get<VtVec3fArray>();
  ASSERT_EQ(transformed_points.size(), 2);
  EXPECT_EQ(transformed_points[0], GfVec3f(2.0, 0.0, 0.0));
  EXPECT_EQ(transformed_points[1], GfVec3f(0.0, 0.5, 0.0));

  const VtVec3fArray& transformed_normals =
      primvars.at(HdTokens->normals).value.Get<VtVec3fArray>();
  ASSERT_EQ(transformed_normals.size(), 2);
  EXPECT_EQ(transformed_normals[0], GfVec3f(1.0, 0.0, 0.0));
  EXPECT_EQ(transformed_normals[1], GfVec3f(0.0, 1.0, 0.0));
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
