# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import Vt
from usd_mitsuba import translator


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant("llvm_ad_rgb")


def _create_quad_mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
  mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
  mesh.GetPointsAttr().Set(Vt.Vec3fArray([
      Gf.Vec3f(-1, -1, 0),
      Gf.Vec3f(1, -1,  0),
      Gf.Vec3f(1, 1, 0),
      Gf.Vec3f(-1, 1, 0),
  ]))
  mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
  mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
  return mesh


def test_translate_point_instancer():
  stage = Usd.Stage.CreateInMemory()

  # Create prototype
  _create_quad_mesh(stage, "/root/Prototypes/ProtoQuad")

  # Create instancer
  instancer = UsdGeom.PointInstancer.Define(stage, "/root/Instancer")
  instancer.GetPrototypesRel().SetTargets(
      [Sdf.Path("/root/Prototypes/ProtoQuad")])

  positions = Vt.Vec3fArray([Gf.Vec3f(-2.0, 0.5, 0), Gf.Vec3f(2.0, -0.5, 0)])
  proto_indices = Vt.IntArray([0, 0])
  instancer.GetPositionsAttr().Set(positions)
  instancer.GetProtoIndicesAttr().Set(proto_indices)

  # Translate
  scene_dict = translator.convert_to_mitsuba(stage)

  # Verify shapegroup exists
  group_id = "proto_group__root_Instancer_0"
  assert group_id in scene_dict
  group = scene_dict[group_id]
  assert group["type"] == "shapegroup"
  assert "shape_0" in group
  assert isinstance(group["shape_0"], mi.Mesh)

  # Verify prototype mesh is NOT at top level
  assert "_root_Prototypes_ProtoQuad" not in scene_dict

  # Verify instances exist
  inst0_id = "instance__root_Instancer_0_0"
  assert inst0_id in scene_dict
  inst0 = scene_dict[inst0_id]
  assert inst0["type"] == "instance"
  assert inst0["shapegroup"]["id"] == group_id

  # Verify transform of instance 0
  expected_trans0 = mi.ScalarTransform4f().translate([-2.0, 0.5, 0.0])
  np.testing.assert_allclose(
      inst0["to_world"].matrix, expected_trans0.matrix, atol=1e-6)

  inst1_id = "instance__root_Instancer_0_1"
  assert inst1_id in scene_dict
  inst1 = scene_dict[inst1_id]
  assert inst1["type"] == "instance"
  assert inst1["shapegroup"]["id"] == group_id

  # Verify transform of instance 1
  expected_trans1 = mi.ScalarTransform4f().translate([2.0, -0.5, 0.0])
  np.testing.assert_allclose(
      inst1["to_world"].matrix, expected_trans1.matrix, atol=1e-6)

  # Verify we can load the scene (at least the geometry part)
  # We need to add a dummy integrator to make it a valid scene if we load it.
  # But convert_to_mitsuba already adds default integrator/sensor if they are in stage.
  # Since they are not, it might have default integrator.
  # Let's try loading it.
  scene = mi.load_dict(scene_dict)
  assert scene is not None


def test_translate_native_instancing():
  stage = Usd.Stage.CreateInMemory()

  _create_quad_mesh(stage, "/Prototypes/Chair/Seat")
  seat_prim = stage.GetPrimAtPath("/Prototypes/Chair/Seat")
  seat_xform = UsdGeom.Xformable(seat_prim)
  seat_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0.5, 0))

  instance1 = UsdGeom.Xform.Define(stage, "/World/ChairInstance1")
  instance1.GetPrim().SetInstanceable(True)
  instance1.GetPrim().GetReferences().AddInternalReference(
      Sdf.Path("/Prototypes/Chair"))
  instance1.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))

  instance2 = UsdGeom.Xform.Define(stage, "/World/ChairInstance2")
  instance2.GetPrim().SetInstanceable(True)
  instance2.GetPrim().GetReferences().AddInternalReference(
      Sdf.Path("/Prototypes/Chair"))
  instance2.AddTranslateOp().Set(Gf.Vec3f(2.0, 0, 0))

  # Deactivate the prototypes scope
  prototypes_prim = stage.GetPrimAtPath("/Prototypes")
  prototypes_prim.SetActive(False)

  scene_dict = translator.convert_to_mitsuba(stage)

  mesh1_id = "_World_ChairInstance1_Seat"
  mesh2_id = "_World_ChairInstance2_Seat"

  assert mesh1_id in scene_dict
  assert mesh2_id in scene_dict
  mesh1 = scene_dict[mesh1_id]
  mesh2 = scene_dict[mesh2_id]
  assert isinstance(mesh1, mi.Mesh)
  assert isinstance(mesh2, mi.Mesh)

  params1 = mi.traverse(mesh1)
  positions1 = np.array(params1['vertex_positions']).reshape(-1, 3)
  centroid1 = np.mean(positions1, axis=0)

  params2 = mi.traverse(mesh2)
  positions2 = np.array(params2['vertex_positions']).reshape(-1, 3)
  centroid2 = np.mean(positions2, axis=0)

  np.testing.assert_allclose(centroid1, [0.0, 0.5, 0.0], atol=1e-6)
  np.testing.assert_allclose(centroid2, [2.0, 0.5, 0.0], atol=1e-6)

  scene = mi.load_dict(scene_dict)
  assert scene is not None
