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

"""Tests for the hdMitsuba render delegate lights."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdShade
import usd_render
from hdmitsuba.tests import test_helpers

_RENDER_RESOLUTION = (64, 64)


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_plane_geometry(stage: Usd.Stage):
  geometry = UsdGeom.Mesh.Define(stage, '/plane')
  points = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]
  face_vertex_counts = [4]
  face_vertex_indices = [3, 2, 1, 0]
  geometry.GetPointsAttr().Set(points)
  geometry.GetFaceVertexCountsAttr().Set(face_vertex_counts)
  geometry.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
  return geometry


def _add_material(stage: Usd.Stage, geometry: UsdGeom.Mesh):
  material = UsdShade.Material.Define(stage, '/plane/material')
  shader = UsdShade.Shader.Define(stage, '/plane/material/surface')
  shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
      (1.0, 0.5, 0.3)
  )
  material.CreateSurfaceOutput().ConnectToSource(
      shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )
  UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _create_stage() -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()

  geometry = _create_plane_geometry(stage)
  _add_material(stage, geometry)

  camera = UsdGeom.Camera.Define(stage, '/camera')
  camera.AddTranslateOp().Set((4, 5, 6))
  camera.AddRotateXOp().Set(-40)
  camera.AddRotateYOp().Set(30)

  dome_light = UsdLux.DomeLight.Define(stage, '/domeLight')
  dome_light.CreateColorAttr().Set((1.0, 0.5, 0.5))

  test_helpers.create_render_settings(stage)

  return stage


def test_dome_light_untextured():
  stage = _create_stage()
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_dome_light_untextured',
      atol=0.1,
  )


@pytest.mark.parametrize(
    'scene_name',
    [
        'area',
        'constant',
        'directional',
        'disk',
        'envmap',
        'envmap_1x1',
        'mesh',
        'mesh_light_api',
        'point',
        'sphere',
        'spot',
        'textured_light',
        # 'ies' # TODO: Re-enable once IES support is added.
    ],
)
def test_render(scene_name: str):
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/{scene_name}.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix=f'test_render_{scene_name}',
      atol=0.1,
  )


def test_change_light_type():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/sphere.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_original = engine.render()['color']

  test_helpers.write_image(
      image_original, 'test_change_light_type_original.png'
  )

  # Change the light type to `Dome`
  light_path = '/root/Area/Area'
  light_prim = stage.GetPrimAtPath(light_path)
  light_prim.SetTypeName('DomeLight')

  image_modified = engine.render()['color']

  test_helpers.write_image(
      image_modified, 'test_change_light_type_modified.png'
  )


def test_visibility():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/sphere.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Hide the light
  light_prim = stage.GetPrimAtPath('/root/Area/Area')
  imageable = UsdGeom.Imageable(light_prim)
  imageable.MakeInvisible()

  image_invisible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_invisible',
      atol=0.1,
      engine=engine,
  )
  assert np.max(image_invisible[..., :3]) < 0.01

  # Make it visible again
  imageable.MakeVisible()
  image_visible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_visible',
      atol=0.1,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_visible, atol=0.05)


def test_active():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/sphere.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Inactivate the light
  light_prim = stage.GetPrimAtPath('/root/Area/Area')
  light_prim.SetActive(False)

  image_inactive, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_inactive',
      atol=0.1,
      engine=engine,
  )
  assert np.max(image_inactive[..., :3]) < 0.01

  # Reactivate it
  light_prim.SetActive(True)
  image_active, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_active',
      atol=0.1,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_active, atol=0.05)


def test_update_treat_as_point():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/sphere.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  # 1. Initial render (treatAsPoint = False by default)
  image_hd_before, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_treat_as_point_before',
      atol=0.1,
      engine=engine,
  )

  # 2. Toggle treatAsPoint to True
  light_prim = stage.GetPrimAtPath('/root/Area/Area')
  UsdLux.SphereLight(light_prim).CreateTreatAsPointAttr().Set(True)
  image_hd_after, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_treat_as_point_after',
      atol=0.1,
      engine=engine,
  )

  # 3. Toggle treatAsPoint back to False
  UsdLux.SphereLight(light_prim).CreateTreatAsPointAttr().Set(False)
  image_hd_restored = engine.render()['color']
  test_helpers.robust_assert_close(
      image_hd_restored, image_hd_before, atol=0.05
  )


def test_update_dome_light_intensity():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/envmap.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  _ = engine.render()['color'][..., :3]

  env_light = UsdLux.DomeLight.Get(stage, '/root/env_light')
  env_light.GetIntensityAttr().Set(3.0)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_dome_light_intensity',
      atol=0.2,
      engine=engine,
  )


def test_update_spot_shaping_softness():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/spot.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  _ = engine.render()['color'][..., :3]

  spot_light = UsdLux.SphereLight.Get(stage, '/root/Area/Area')
  shaping = UsdLux.ShapingAPI(spot_light.GetPrim())
  shaping.GetShapingConeSoftnessAttr().Set(0.0)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_spot_shaping_softness',
      atol=0.2,
      engine=engine,
  )


def test_update_mesh_light_intensity():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/mesh_light_api.usda'
  )
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  # 1. Initial render (intensity = 5 by default)
  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # 2. Update intensity to 0 (turn off the light)
  mesh_light_path = '/root/MeshLight/MeshLight'
  mesh_light_prim = stage.GetPrimAtPath(mesh_light_path)

  # Set intensity to 0
  intensity_attr = mesh_light_prim.GetAttribute('inputs:intensity')
  assert intensity_attr.IsValid()
  intensity_attr.Set(0.0)

  image_modified, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_mesh_light_intensity_off',
      atol=0.1,
      engine=engine,
  )
  # The image should be completely black (or very close to it)
  assert np.max(image_modified[..., :3]) < 0.01

  # 3. Update intensity to 20 (much brighter)
  intensity_attr.Set(20.0)
  image_bright, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_mesh_light_intensity_bright',
      atol=0.1,
      engine=engine,
  )
  # The bright image should be brighter than the original
  assert np.max(image_bright[..., :3]) > np.max(image_original[..., :3])
