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

"""Tests for geometry instancing in hdMitsuba."""

from __future__ import annotations

import os
import mitsuba as mi
import numpy as np
import pytest

from pxr import Usd
from pxr import UsdGeom
from pxr import UsdShade
from pxr import Vt
from pxr import Gf
from pxr import Sdf
import usd_render
from hdmitsuba.tests import test_helpers


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  # Define Instancer first so CopySpec has parent
  instancer = UsdGeom.PointInstancer.Define(stage, '/root/Instancer')

  # Copy /root/Cube to /root/Instancer/ProtoCube
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Instancer/ProtoCube')
  stage.RemovePrim('/root/Cube')

  instancer.GetPrototypesRel().SetTargets(['/root/Instancer/ProtoCube'])

  positions = Vt.Vec3fArray([Gf.Vec3f(-1.5, 0, 0), Gf.Vec3f(1.5, 0, 0)])
  proto_indices = Vt.IntArray([0, 0])
  instancer.GetPositionsAttr().Set(positions)
  instancer.GetProtoIndicesAttr().Set(proto_indices)

  output_path = os.path.join(tmp_path, 'instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  # Copy /root/Cube to /root/Cube1 and /root/Cube2
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube2')
  stage.RemovePrim('/root/Cube')

  cube1 = UsdGeom.Xformable.Get(stage, '/root/Cube1')
  cube1.AddTranslateOp().Set(Gf.Vec3d(-1.5, 0, 0))

  cube2 = UsdGeom.Xformable.Get(stage, '/root/Cube2')
  cube2.AddTranslateOp().Set(Gf.Vec3d(1.5, 0, 0))

  output_path = os.path.join(tmp_path, 'flattened.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_instancing_vs_flattened(tmp_path):
  instanced_path = _create_instanced_stage(tmp_path)
  flattened_path = _create_flattened_stage(tmp_path)

  # Render instanced
  stage_inst = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_instanced.png')

  # Render flattened
  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(img_flat, 'test_instancing_flattened.png')

  # Compare
  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def _create_modified_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube2')
  stage.RemovePrim('/root/Cube')

  cube1 = UsdGeom.Xformable.Get(stage, '/root/Cube1')
  cube1.AddTranslateOp().Set(Gf.Vec3d(-1.0, 0.5, 0))

  cube2 = UsdGeom.Xformable.Get(stage, '/root/Cube2')
  cube2.AddTranslateOp().Set(Gf.Vec3d(1.0, -0.5, 0))

  output_path = os.path.join(tmp_path, 'flattened_modified.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_modify_instance_transforms(tmp_path):
  instanced_path = _create_instanced_stage(tmp_path)
  stage = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  img_orig = engine.render()['color']
  test_helpers.write_image(img_orig, 'test_modify_instances_original.png')

  # Modify positions
  instancer = UsdGeom.PointInstancer.Get(stage, '/root/Instancer')
  positions = Vt.Vec3fArray([Gf.Vec3f(-1.0, 0.5, 0), Gf.Vec3f(1.0, -0.5, 0)])
  instancer.GetPositionsAttr().Set(positions)

  img_mod = engine.render()['color']
  test_helpers.write_image(img_mod, 'test_modify_instances_modified.png')

  # Compare with expected modified flattened render
  flat_mod_path = _create_modified_flattened_stage(tmp_path)
  stage_flat_mod = Usd.Stage.Open(flat_mod_path)
  test_helpers.create_render_settings(stage_flat_mod, resolution=(256, 256))
  engine_flat_mod = usd_render.RenderEngine(stage_flat_mod)
  engine_flat_mod.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat_mod = engine_flat_mod.render()['color']

  np.testing.assert_allclose(img_mod, img_flat_mod, atol=0.05)


def _create_native_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  stage.DefinePrim('/root/Prototypes', 'Scope')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Prototypes/CubePrototype')
  stage.RemovePrim('/root/Cube')

  instance1 = stage.DefinePrim('/root/Instance1', 'Xform')
  instance1.SetInstanceable(True)
  instance1.GetReferences().AddInternalReference('/root/Prototypes/CubePrototype')
  UsdGeom.Xformable(instance1).AddTranslateOp().Set(Gf.Vec3d(-1.5, 0, 0))

  instance2 = stage.DefinePrim('/root/Instance2', 'Xform')
  instance2.SetInstanceable(True)
  instance2.GetReferences().AddInternalReference('/root/Prototypes/CubePrototype')
  UsdGeom.Xformable(instance2).AddTranslateOp().Set(Gf.Vec3d(1.5, 0, 0))

  stage.GetPrimAtPath('/root/Prototypes').SetActive(False)

  output_path = os.path.join(tmp_path, 'native_instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_native_instancing(tmp_path):
  native_instanced_path = _create_native_instanced_stage(tmp_path)
  flattened_path = _create_flattened_stage(tmp_path)

  # Render native instanced
  stage_inst = Usd.Stage.Open(native_instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_native.png')

  # Render flattened
  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(img_flat, 'test_instancing_native_flattened.png')

  # Compare
  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def _create_custom_flattened_stage(tmp_path, positions: list[Gf.Vec3d], name: str = 'flattened') -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  for i in range(len(positions)):
    Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
                 stage.GetRootLayer(), f'/root/Cube{i}')
  stage.RemovePrim('/root/Cube')

  for i, pos in enumerate(positions):
    cube = UsdGeom.Xformable.Get(stage, f'/root/Cube{i}')
    cube.AddTranslateOp().Set(pos)

  output_path = os.path.join(tmp_path, f'{name}.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_nested_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  parent_instancer = UsdGeom.PointInstancer.Define(
      stage, '/root/ParentInstancer')

  stage.DefinePrim('/root/ParentInstancer/ChildInstancer', 'Scope')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube', stage.GetRootLayer(),
               '/root/ParentInstancer/ChildInstancer/ProtoCube')
  stage.RemovePrim('/root/Cube')

  child_instancer = UsdGeom.PointInstancer.Define(
      stage, '/root/ParentInstancer/ChildInstancer/Instancer')
  child_instancer.GetPrototypesRel().SetTargets(
      ['/root/ParentInstancer/ChildInstancer/ProtoCube'])
  child_instancer.GetPositionsAttr().Set(
      Vt.Vec3fArray([Gf.Vec3f(0, -0.5, 0), Gf.Vec3f(0, 0.5, 0)]))
  child_instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0, 0]))

  parent_instancer.GetPrototypesRel().SetTargets(
      ['/root/ParentInstancer/ChildInstancer/Instancer'])
  parent_instancer.GetPositionsAttr().Set(
      Vt.Vec3fArray([Gf.Vec3f(-2.0, 0, 0), Gf.Vec3f(2.0, 0, 0)]))
  parent_instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0, 0]))

  output_path = os.path.join(tmp_path, 'nested_instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_nested_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  for i in range(1, 5):
    Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
                 stage.GetRootLayer(), f'/root/Cube{i}')
  stage.RemovePrim('/root/Cube')

  UsdGeom.Xformable.Get(
      stage, '/root/Cube1').AddTranslateOp().Set(Gf.Vec3d(-2.0, -0.5, 0))
  UsdGeom.Xformable.Get(
      stage, '/root/Cube2').AddTranslateOp().Set(Gf.Vec3d(-2.0, 0.5, 0))
  UsdGeom.Xformable.Get(
      stage, '/root/Cube3').AddTranslateOp().Set(Gf.Vec3d(2.0, -0.5, 0))
  UsdGeom.Xformable.Get(
      stage, '/root/Cube4').AddTranslateOp().Set(Gf.Vec3d(2.0, 0.5, 0))

  output_path = os.path.join(tmp_path, 'nested_flattened.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_nested_instancing(tmp_path):
  instanced_path = _create_nested_instanced_stage(tmp_path)
  flattened_path = _create_nested_flattened_stage(tmp_path)

  stage_inst = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_nested.png')

  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(img_flat, 'test_instancing_nested_flattened.png')

  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def _create_scale_rotate_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  instancer = UsdGeom.PointInstancer.Define(stage, '/root/Instancer')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Instancer/ProtoCube')
  stage.RemovePrim('/root/Cube')
  instancer.GetPrototypesRel().SetTargets(['/root/Instancer/ProtoCube'])

  positions = Vt.Vec3fArray([Gf.Vec3f(-2.0, 0, 0), Gf.Vec3f(2.0, 0, 0)])
  scales = Vt.Vec3fArray([Gf.Vec3f(1.0, 2.0, 0.5), Gf.Vec3f(0.5, 1.0, 2.0)])
  q1 = Gf.Quath(Gf.Rotation(Gf.Vec3d(0, 0, 1), 45).GetQuat())
  q2 = Gf.Quath(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat())
  rotations = Vt.QuathArray([q1, q2])

  instancer.GetPositionsAttr().Set(positions)
  instancer.GetScalesAttr().Set(scales)
  instancer.GetOrientationsAttr().Set(rotations)
  instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0, 0]))

  output_path = os.path.join(tmp_path, 'scale_rotate_instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_scale_rotate_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube2')
  stage.RemovePrim('/root/Cube')

  cube1 = UsdGeom.Xformable.Get(stage, '/root/Cube1')
  cube1.AddTranslateOp().Set(Gf.Vec3d(-2.0, 0, 0))
  cube1.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 45).GetQuat()))
  cube1.AddScaleOp().Set(Gf.Vec3d(1.0, 2.0, 0.5))

  cube2 = UsdGeom.Xformable.Get(stage, '/root/Cube2')
  cube2.AddTranslateOp().Set(Gf.Vec3d(2.0, 0, 0))
  cube2.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat()))
  cube2.AddScaleOp().Set(Gf.Vec3d(0.5, 1.0, 2.0))

  output_path = os.path.join(tmp_path, 'scale_rotate_flattened.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_instancing_scale_rotate(tmp_path):
  instanced_path = _create_scale_rotate_instanced_stage(tmp_path)
  flattened_path = _create_scale_rotate_flattened_stage(tmp_path)

  stage_inst = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_scale_rotate.png')

  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(
      img_flat, 'test_instancing_scale_rotate_flattened.png')

  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def _create_multi_proto_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  instancer = UsdGeom.PointInstancer.Define(stage, '/root/Instancer')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Instancer/ProtoCube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Instancer/ProtoCube2')
  slab = UsdGeom.Xformable.Get(stage, '/root/Instancer/ProtoCube2')
  slab.AddScaleOp().Set(Gf.Vec3d(1.0, 0.2, 1.0))
  stage.RemovePrim('/root/Cube')

  instancer.GetPrototypesRel().SetTargets(
      ['/root/Instancer/ProtoCube1', '/root/Instancer/ProtoCube2'])

  positions = Vt.Vec3fArray(
      [Gf.Vec3f(-2.0, 0, 0), Gf.Vec3f(0, 0, 0), Gf.Vec3f(2.0, 0, 0)])
  proto_indices = Vt.IntArray([0, 1, 0])

  instancer.GetPositionsAttr().Set(positions)
  instancer.GetProtoIndicesAttr().Set(proto_indices)

  output_path = os.path.join(tmp_path, 'multi_proto_instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_multi_proto_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube2')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube3')
  stage.RemovePrim('/root/Cube')

  UsdGeom.Xformable.Get(
      stage, '/root/Cube1').AddTranslateOp().Set(Gf.Vec3d(-2.0, 0, 0))

  cube2 = UsdGeom.Xformable.Get(stage, '/root/Cube2')
  cube2.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
  cube2.AddScaleOp().Set(Gf.Vec3d(1.0, 0.2, 1.0))

  UsdGeom.Xformable.Get(
      stage, '/root/Cube3').AddTranslateOp().Set(Gf.Vec3d(2.0, 0, 0))

  output_path = os.path.join(tmp_path, 'multi_proto_flattened.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_multiple_prototypes(tmp_path):
  instanced_path = _create_multi_proto_instanced_stage(tmp_path)
  flattened_path = _create_multi_proto_flattened_stage(tmp_path)

  stage_inst = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_multi_proto.png')

  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(
      img_flat, 'test_instancing_multi_proto_flattened.png')

  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def _create_invisible_instanced_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  instancer = UsdGeom.PointInstancer.Define(stage, '/root/Instancer')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Instancer/ProtoCube')
  stage.RemovePrim('/root/Cube')
  instancer.GetPrototypesRel().SetTargets(['/root/Instancer/ProtoCube'])

  positions = Vt.Vec3fArray(
      [Gf.Vec3f(-2.0, 0, 0), Gf.Vec3f(0, 0, 0), Gf.Vec3f(2.0, 0, 0)])
  proto_indices = Vt.IntArray([0, 0, 0])
  instancer.GetPositionsAttr().Set(positions)
  instancer.GetProtoIndicesAttr().Set(proto_indices)
  instancer.GetInvisibleIdsAttr().Set(Vt.Int64Array([1]))

  output_path = os.path.join(tmp_path, 'invisible_instanced.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def _create_invisible_flattened_stage(tmp_path) -> str:
  cube_path = f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda'
  layer = Sdf.Layer.FindOrOpen(cube_path)
  anon_layer = Sdf.Layer.CreateAnonymous()
  anon_layer.TransferContent(layer)
  stage = Usd.Stage.Open(anon_layer)

  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube1')
  Sdf.CopySpec(stage.GetRootLayer(), '/root/Cube',
               stage.GetRootLayer(), '/root/Cube3')
  stage.RemovePrim('/root/Cube')

  UsdGeom.Xformable.Get(
      stage, '/root/Cube1').AddTranslateOp().Set(Gf.Vec3d(-2.0, 0, 0))
  UsdGeom.Xformable.Get(
      stage, '/root/Cube3').AddTranslateOp().Set(Gf.Vec3d(2.0, 0, 0))

  output_path = os.path.join(tmp_path, 'invisible_flattened.usda')
  stage.GetRootLayer().Export(output_path)
  return output_path


def test_invisible_ids(tmp_path):
  instanced_path = _create_invisible_instanced_stage(tmp_path)
  flattened_path = _create_invisible_flattened_stage(tmp_path)

  stage_inst = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage_inst, resolution=(256, 256))
  engine_inst = usd_render.RenderEngine(stage_inst)
  engine_inst.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_inst = engine_inst.render()['color']
  test_helpers.write_image(img_inst, 'test_instancing_invisible.png')

  stage_flat = Usd.Stage.Open(flattened_path)
  test_helpers.create_render_settings(stage_flat, resolution=(256, 256))
  engine_flat = usd_render.RenderEngine(stage_flat)
  engine_flat.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat = engine_flat.render()['color']
  test_helpers.write_image(img_flat, 'test_instancing_invisible_flattened.png')

  np.testing.assert_allclose(img_inst, img_flat, atol=0.05)


def test_change_instance_count(tmp_path):
  instanced_path = _create_instanced_stage(tmp_path)
  stage = Usd.Stage.Open(instanced_path)
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  img_orig = engine.render()['color']
  test_helpers.write_image(img_orig, 'test_change_count_original.png')

  # 1. Add an instance (change count from 2 to 3)
  instancer = UsdGeom.PointInstancer.Get(stage, '/root/Instancer')
  positions_3 = Vt.Vec3fArray(
      [Gf.Vec3f(-1.5, 0, 0), Gf.Vec3f(1.5, 0, 0), Gf.Vec3f(0.0, 1.0, 0)])
  proto_indices_3 = Vt.IntArray([0, 0, 0])
  instancer.GetPositionsAttr().Set(positions_3)
  instancer.GetProtoIndicesAttr().Set(proto_indices_3)

  img_3 = engine.render()['color']
  test_helpers.write_image(img_3, 'test_change_count_3_instances.png')

  # Compare with expected 3-instance flattened render
  flat_3_path = _create_custom_flattened_stage(
      tmp_path,
      [Gf.Vec3d(-1.5, 0, 0), Gf.Vec3d(1.5, 0, 0), Gf.Vec3d(0.0, 1.0, 0)],
      'flat_3',
  )
  stage_flat_3 = Usd.Stage.Open(flat_3_path)
  test_helpers.create_render_settings(stage_flat_3, resolution=(256, 256))
  engine_flat_3 = usd_render.RenderEngine(stage_flat_3)
  engine_flat_3.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat_3 = engine_flat_3.render()['color']

  np.testing.assert_allclose(img_3, img_flat_3, atol=0.05)

  # 2. Remove instances (change count from 3 to 1)
  positions_1 = Vt.Vec3fArray([Gf.Vec3f(0.0, 1.0, 0)])
  proto_indices_1 = Vt.IntArray([0])
  instancer.GetPositionsAttr().Set(positions_1)
  instancer.GetProtoIndicesAttr().Set(proto_indices_1)

  img_1 = engine.render()['color']
  test_helpers.write_image(img_1, 'test_change_count_1_instance.png')

  # Compare with expected 1-instance flattened render
  flat_1_path = _create_custom_flattened_stage(
      tmp_path, [Gf.Vec3d(0.0, 1.0, 0)], 'flat_1'
  )
  stage_flat_1 = Usd.Stage.Open(flat_1_path)
  test_helpers.create_render_settings(stage_flat_1, resolution=(256, 256))
  engine_flat_1 = usd_render.RenderEngine(stage_flat_1)
  engine_flat_1.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  img_flat_1 = engine_flat_1.render()['color']

  np.testing.assert_allclose(img_1, img_flat_1, atol=0.05)


def test_instancing_emitter_warning(tmp_path, capfd):
  instanced_path = _create_instanced_stage(tmp_path)
  stage = Usd.Stage.Open(instanced_path)

  # Make the material of the prototype emissive
  shader = UsdShade.Shader.Get(
      stage, '/root/_materials/Material/Principled_BSDF')
  shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
      Gf.Vec3f(1.0, 1.0, 1.0))

  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  # Render to trigger Sync and Commit
  engine.render()

  captured = capfd.readouterr()
  assert "is instanced but has an emitter attached" in captured.err
