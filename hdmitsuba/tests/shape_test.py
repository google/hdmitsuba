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

"""Tests for the hdMitsuba render delegate shapes/meshes."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Usd
from pxr import UsdGeom
from pxr import Vt
import usd_render
from hdmitsuba.tests import test_helpers


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


@pytest.mark.parametrize(
    'scene_name, time',
    [
        ('subdiv_cube', Usd.TimeCode.Default()),
        ('subdiv_plane', Usd.TimeCode.Default()),
        ('subdiv_two_quads', Usd.TimeCode.Default()),
        ('displacement_preview_surface', Usd.TimeCode.Default()),
        ('curve', Usd.TimeCode.Default()),
        ('geom_subsets', Usd.TimeCode.Default()),
        ('empty_mesh', Usd.TimeCode.Default()),
        ('skinned_mesh', Usd.TimeCode(20)),
    ],
)
def test_render(scene_name: str, time: Usd.TimeCode):
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/{scene_name}.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(512, 512))
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix=f'test_render_{scene_name}',
      atol=0.05,
      subdivision_level=4,
      refine_level_fallback=4,
      time=time,
  )


def test_modify_shape_transform():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image = engine.render()['color']
  test_helpers.write_image(image, 'test_modify_shape_transform_original.png')

  cube_xform = UsdGeom.Xformable.Get(stage, '/root/Cube')
  cube_xform.AddTranslateOp(opSuffix='new_translation').Set((0.9, 0.1, -0.1))
  cube_xform.AddScaleOp(opSuffix='new_scale').Set((1.0, 1.0, 0.5))
  cube_xform.AddRotateZOp(opSuffix='new_rotation').Set(-40)
  image_modified = engine.render()['color']
  test_helpers.write_image(
      image_modified, 'test_modify_shape_transform_modified.png'
  )
  assert np.mean(np.abs(image_modified[..., :3] - image[..., :3])) > 0.02

  # Re-render with a new render engine and check that the results match.
  engine_new = usd_render.RenderEngine(stage)
  engine_new.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_modified_2 = engine_new.render()['color']
  test_helpers.write_image(
      image_modified_2, 'test_modify_shape_transform_new.png'
  )
  np.testing.assert_allclose(image_modified, image_modified_2, atol=0.05)


def test_modify_points():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image = engine.render()['color']
  test_helpers.write_image(image, 'test_modify_points_original.png')

  mesh = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  points = np.array(mesh.GetPointsAttr().Get())
  points[0] = np.array([3.0, 1.0, 1.0])
  points[7] = np.array([-2.0, -2.0, -3.0])
  points *= 1.3
  mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

  image_modified = engine.render()['color']
  test_helpers.write_image(image_modified, 'test_modify_points_modified.png')
  assert np.mean(np.abs(image_modified[..., :3] - image[..., :3])) > 0.02

  engine_new = usd_render.RenderEngine(stage)
  engine_new.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_modified_2 = engine_new.render()['color']
  test_helpers.write_image(image_modified_2, 'test_modify_points_new.png')
  # np.testing.assert_allclose(image_modified, image_modified_2, atol=0.05)


def test_visibility():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Hide the cube
  cube_prim = stage.GetPrimAtPath('/root/Cube')
  imageable = UsdGeom.Imageable(cube_prim)
  imageable.MakeInvisible()

  image_invisible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_invisible',
      spp=8,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_invisible[..., :3]) < 0.01

  # Make it visible again
  imageable.MakeVisible()
  image_visible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_visible',
      atol=0.05,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_visible, atol=0.05)


def test_active():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Inactivate the cube
  cube_prim = stage.GetPrimAtPath('/root/Cube')
  cube_prim.SetActive(False)

  image_inactive, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_inactive',
      spp=8,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_inactive[..., :3]) < 0.01

  # Reactivate it
  cube_prim.SetActive(True)
  image_active, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_active',
      atol=0.05,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_active, atol=0.05)


def test_modify_curve_width():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(64, 64))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  _ = engine.render()['color']

  curve = UsdGeom.BasisCurves.Get(stage, '/root/mycurve/curve')
  widths = np.array(curve.GetWidthsAttr().Get()) * 2.5
  curve.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths))

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_modify_curve_width',
      atol=0.1,
      engine=engine,
  )


def test_modify_curve_transform_only():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(64, 64))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  _ = engine.render()['color']

  curve_xform = UsdGeom.Xformable.Get(stage, '/root/mycurve')
  curve_xform.AddRotateZOp().Set(45.0)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_modify_curve_transform_only',
      atol=0.1,
      engine=engine,
  )
