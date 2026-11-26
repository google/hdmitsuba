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
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import Vt
import pytest

from usd_mitsuba import translator


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant("llvm_ad_rgb")


def _create_quad_mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
  mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
  mesh.GetPointsAttr().Set(
      Vt.Vec3fArray([
          Gf.Vec3f(-1, -1, 0),
          Gf.Vec3f(1, -1, 0),
          Gf.Vec3f(1, 1, 0),
          Gf.Vec3f(-1, 1, 0),
      ])
  )
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
      [Sdf.Path("/root/Prototypes/ProtoQuad")]
  )

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
      inst0["to_world"].matrix, expected_trans0.matrix, atol=1e-6
  )

  inst1_id = "instance__root_Instancer_0_1"
  assert inst1_id in scene_dict
  inst1 = scene_dict[inst1_id]
  assert inst1["type"] == "instance"
  assert inst1["shapegroup"]["id"] == group_id

  # Verify transform of instance 1
  expected_trans1 = mi.ScalarTransform4f().translate([2.0, -0.5, 0.0])
  np.testing.assert_allclose(
      inst1["to_world"].matrix, expected_trans1.matrix, atol=1e-6
  )

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
      Sdf.Path("/Prototypes/Chair")
  )
  instance1.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))

  instance2 = UsdGeom.Xform.Define(stage, "/World/ChairInstance2")
  instance2.GetPrim().SetInstanceable(True)
  instance2.GetPrim().GetReferences().AddInternalReference(
      Sdf.Path("/Prototypes/Chair")
  )
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
  positions1 = np.array(params1["positions"]).reshape(-1, 3)
  centroid1 = np.mean(positions1, axis=0)

  params2 = mi.traverse(mesh2)
  positions2 = np.array(params2["positions"]).reshape(-1, 3)
  centroid2 = np.mean(positions2, axis=0)

  np.testing.assert_allclose(centroid1, [0.0, 0.5, 0.0], atol=1e-6)
  np.testing.assert_allclose(centroid2, [2.0, 0.5, 0.0], atol=1e-6)

  scene = mi.load_dict(scene_dict)
  assert scene is not None


def test_translate_animated_mesh_and_point_instancer():
  stage = Usd.Stage.CreateInMemory()

  camera = UsdGeom.Camera.Define(stage, "/root/Camera")
  camera.GetShutterOpenAttr().Set(-0.5)
  camera.GetShutterCloseAttr().Set(0.5)

  _create_quad_mesh(stage, "/root/StaticQuad")
  for shape in (
      _create_quad_mesh(stage, "/root/AnimatedQuad"),
      UsdGeom.Cube.Define(stage, "/root/AnimatedCube"),
  ):
    op = shape.AddTranslateOp()
    op.Set((-1.0, 0.0, 0.0), 0.0)
    op.Set((1.0, 0.0, 0.0), 1.0)

  _create_quad_mesh(stage, "/root/Prototypes/ProtoQuad")
  instancer = UsdGeom.PointInstancer.Define(stage, "/root/Instancer")
  instancer.GetPrototypesRel().SetTargets(["/root/Prototypes/ProtoQuad"])
  instancer.GetProtoIndicesAttr().Set([0, 0])
  instancer.GetPositionsAttr().Set([(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 0.0)
  instancer.GetPositionsAttr().Set([(-2.0, 1.0, 0.0), (2.0, -1.0, 0.0)], 1.0)

  scene_dict = translator.convert_to_mitsuba(stage, time=Usd.TimeCode(0.5))

  assert isinstance(scene_dict["_root_StaticQuad"], mi.Mesh)
  for prim_id in ("_root_AnimatedQuad", "_root_AnimatedCube"):
    assert scene_dict[f"proto_group_{prim_id}"]["type"] == "shapegroup"
    assert isinstance(scene_dict[prim_id]["to_world"], mi.AnimatedTransform4f)
  for i in (0, 1):
    assert isinstance(
        scene_dict[f"instance__root_Instancer_0_{i}"]["to_world"],
        mi.AnimatedTransform4f,
    )
  mi.load_dict(scene_dict)


def test_cylinder_geometry_overrides_align_endcaps():
  stage = Usd.Stage.CreateInMemory()

  def _configure_cylinder(prim: Usd.Prim) -> None:
    cylinder = UsdGeom.Cylinder(prim)
    cylinder.CreateRadiusAttr().Set(0.5)
    cylinder.CreateHeightAttr().Set(2.0)
    prim.CreateAttribute("mitsuba:shape:type", Sdf.ValueTypeNames.String).Set(
        "cylinder"
    )
    prim.CreateAttribute("mitsuba:shape:radius", Sdf.ValueTypeNames.Float).Set(
        0.4
    )
    prim.CreateAttribute("mitsuba:shape:p0", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.25)
    )
    prim.CreateAttribute("mitsuba:shape:p1", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.75)
    )

  top_cylinder = UsdGeom.Cylinder.Define(stage, "/root/Cylinder")
  _configure_cylinder(top_cylinder.GetPrim())

  proto_cylinder = UsdGeom.Cylinder.Define(
      stage, "/root/Prototypes/Proto/Cylinder"
  )
  _configure_cylinder(proto_cylinder.GetPrim())
  instancer = UsdGeom.PointInstancer.Define(stage, "/root/Instancer")
  instancer.GetPrototypesRel().SetTargets([Sdf.Path("/root/Prototypes/Proto")])
  instancer.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(2.0, 0.0, 0.0)]))
  instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0]))

  scene_dict = translator.convert_to_mitsuba(stage)
  group = scene_dict["proto_group__root_Instancer_0"]
  for body, cap_top, cap_bottom in (
      (
          scene_dict["_root_Cylinder_body"],
          scene_dict["_root_Cylinder_cap_top"],
          scene_dict["_root_Cylinder_cap_bottom"],
      ),
      (group["shape_0"], group["shape_1"], group["shape_2"]),
  ):
    assert [body["type"], cap_top["type"], cap_bottom["type"]] == [
        "cylinder",
        "disk",
        "disk",
    ]
    for part in (body, cap_top, cap_bottom):
      for key in ("radius", "p0", "p1"):
        assert key not in part
    # Cylinder spans z in [-1, 1]; p0=0.25 -> z=-0.5, p1=0.75 -> z=+0.5,
    # overridden radius = 0.5 * 0.4 = 0.2.
    np.testing.assert_allclose(
        cap_bottom["to_world"] @ mi.ScalarPoint3f(0, 0, 0),
        [0.0, 0.0, -0.5],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        cap_top["to_world"] @ mi.ScalarPoint3f(0, 0, 0),
        [0.0, 0.0, 0.5],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.linalg.norm(cap_top["to_world"] @ mi.ScalarVector3f(1, 0, 0)),
        0.2,
        atol=1e-5,
    )
  mi.load_dict(scene_dict)


def test_implicit_surface_shape_sensor_override_precedence():
  stage = Usd.Stage.CreateInMemory()
  sphere_prim = UsdGeom.Sphere.Define(stage, "/root/Sphere").GetPrim()
  sphere_prim.CreateAttribute(
      "mitsuba:shape:sensor:type", Sdf.ValueTypeNames.String
  ).Set("irradiancemeter")
  sphere_prim.CreateAttribute(
      "mitsuba:shape:sensor:film:type", Sdf.ValueTypeNames.String
  ).Set("hdrfilm")
  sphere_prim.CreateAttribute(
      "mitsuba:shape:sensor:film:width", Sdf.ValueTypeNames.Int
  ).Set(32)
  camera_prim = UsdGeom.Camera.Define(stage, "/root/Meter").GetPrim()
  camera_prim.CreateAttribute(
      "mitsuba:sensor:type", Sdf.ValueTypeNames.String
  ).Set("irradiancemeter")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:shape", Sdf.ValueTypeNames.String
  ).Set("/root/Sphere")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:film:width", Sdf.ValueTypeNames.Int
  ).Set(16)
  camera_prim.CreateAttribute(
      "mitsuba:sensor:film:height", Sdf.ValueTypeNames.Int
  ).Set(24)

  scene_dict = translator.convert_to_mitsuba(stage)
  sensor_dict = scene_dict["_root_Sphere"]["sensor"]
  assert sensor_dict["type"] == "irradiancemeter"
  assert sensor_dict["film"]["width"] == 32
  assert "height" not in sensor_dict["film"]
  mi.load_dict(scene_dict)

  # Without `mitsuba:shape:sensor:type`, the nested override is skipped.
  sphere_prim.RemoveProperty("mitsuba:shape:sensor:type")
  fallback_dict = translator.convert_to_mitsuba(stage)
  fallback_sensor = fallback_dict["_root_Sphere"]["sensor"]
  assert fallback_sensor["film"]["width"] == 16
  assert fallback_sensor["film"]["height"] == 24
  mi.load_dict(fallback_dict)


def test_cylinder_multipart_strips_replicated_child_ids():
  stage = Usd.Stage.CreateInMemory()
  top_prim = UsdGeom.Cylinder.Define(stage, "/root/Cylinder").GetPrim()
  for name, value in (
      ("mitsuba:emitter:type", "area"),
      ("mitsuba:emitter:id", "cyl_light"),
      ("mitsuba:shape:interior:type", "homogeneous"),
      ("mitsuba:shape:interior:id", "cyl_medium"),
      ("mitsuba:shape:interior:phase:type", "hg"),
      ("mitsuba:shape:interior:phase:id", "cyl_phase"),
  ):
    top_prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)

  proto_prim = UsdGeom.Cylinder.Define(
      stage, "/root/Prototypes/Proto/Cylinder"
  ).GetPrim()
  proto_prim.CreateAttribute(
      "mitsuba:shape:bsdf:type", Sdf.ValueTypeNames.String
  ).Set("diffuse")
  proto_prim.CreateAttribute(
      "mitsuba:shape:bsdf:id", Sdf.ValueTypeNames.String
  ).Set("cyl_bsdf")
  instancer = UsdGeom.PointInstancer.Define(stage, "/root/Instancer")
  instancer.GetPrototypesRel().SetTargets([Sdf.Path("/root/Prototypes/Proto")])
  instancer.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(2.0, 0.0, 0.0)]))
  instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0]))

  scene_dict = translator.convert_to_mitsuba(stage)
  body = scene_dict["_root_Cylinder_body"]
  assert body["emitter"]["id"] == "cyl_light"
  assert body["interior"]["id"] == "cyl_medium"
  assert body["interior"]["phase"]["id"] == "cyl_phase"
  for suffix in ("cap_top", "cap_bottom"):
    cap = scene_dict[f"_root_Cylinder_{suffix}"]
    assert "id" not in cap["emitter"]
    assert "id" not in cap["interior"]
    assert "id" not in cap["interior"]["phase"]

  group = scene_dict["proto_group__root_Instancer_0"]
  assert group["shape_0"]["bsdf"]["id"] == "cyl_bsdf"
  for shape_key in ("shape_1", "shape_2"):
    assert "id" not in group[shape_key]["bsdf"]
  mi.load_dict(scene_dict)


def test_animated_cube_extra_to_world_applied_in_object_space():
  stage = Usd.Stage.CreateInMemory()
  camera = UsdGeom.Camera.Define(stage, "/root/Camera")
  camera.GetShutterOpenAttr().Set(-0.5)
  camera.GetShutterCloseAttr().Set(0.5)

  cube = UsdGeom.Cube.Define(stage, "/root/AnimatedCube")
  translate_op = cube.AddTranslateOp()
  translate_op.Set((-1.0, 0.0, 0.0), 0.0)
  translate_op.Set((1.0, 0.0, 0.0), 1.0)
  # Sheared object-space transform must live on the prototype shape rather than
  # inside `AnimatedTransform4f`, which rejects shear.
  sheared_matrix = Gf.Matrix4d(1.0)
  sheared_matrix[1, 0] = 0.5
  cube.GetPrim().CreateAttribute(
      "mitsuba:shape:to_world", Sdf.ValueTypeNames.Matrix4d
  ).Set(sheared_matrix)

  scene_dict = translator.convert_to_mitsuba(stage, time=Usd.TimeCode(0.5))
  assert isinstance(
      scene_dict["_root_AnimatedCube"]["to_world"], mi.AnimatedTransform4f
  )
  proto_shape = scene_dict["proto_group__root_AnimatedCube"]["shape_0"]
  np.testing.assert_allclose(
      proto_shape["to_world"] @ mi.ScalarPoint3f(0.0, 1.0, 0.0),
      [0.5, 1.0, 0.0],
      atol=1e-5,
  )
  mi.load_dict(scene_dict)
