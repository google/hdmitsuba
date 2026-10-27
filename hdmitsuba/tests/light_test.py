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

from typing import Any

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


def _open_ies_stage(light_type: str) -> Usd.Stage:
  """Opens the `ies_<light_type>.usda` scene with 256 spp render settings."""
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/ies_{light_type}.usda'
  )
  test_helpers.create_render_settings(
      stage, resolution=_RENDER_RESOLUTION, spp=256
  )
  return stage


def _render(
    stage: Usd.Stage,
    renderer: str = 'HdMitsubaRendererPlugin',
    overrides: dict[str, Any] | None = None,
) -> np.ndarray:
  """Renders `stage` with the Hydra `renderer` and render setting overrides."""
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id=renderer, overrides=overrides)
  return engine.render()['color']


def test_dome_light_untextured():
  stage = _create_stage()
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_dome_light_untextured',
      atol=0.1,
  )


def test_dome_light_corrupt_texture_falls_back_to_constant():
  stage = _create_stage()
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  reference = engine.render()['color'][..., :3]

  # Invalid envmap path should get ignored.
  dome_light = UsdLux.DomeLight.Get(stage, '/domeLight')
  dome_light.CreateTextureFileAttr().Set(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/envmap.usda'
  )
  image = engine.render()['color'][..., :3]

  np.testing.assert_allclose(image, reference, atol=0.1)


@pytest.mark.parametrize(
    'scene_name',
    [
        'area',
        'area_shadow_passthrough',
        'constant',
        'directional',
        'disk',
        'envmap',
        'envmap_1x1',
        'mesh',
        'mesh_light_api',
        'point',
        'point_unnormalized',
        'point_zero_radius',
        'sphere',
        'spot',
        'textured_light',
        'ies',
        'ies_disk',
        'ies_mesh',
        'ies_mesh_subsets',
        'ies_point',
        'ies_rect',
        'ies_sphere',
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

  # 4. Remove and re-apply MeshLightAPI to exercise emitter_spec presence transitions.
  mesh_light_prim.RemoveAPI(UsdLux.MeshLightAPI)
  mesh_light_prim.RemoveProperty('inputs:intensity')
  assert np.max(engine.render()['color'][..., :3]) < 0.01

  UsdLux.MeshLightAPI.Apply(mesh_light_prim)
  UsdLux.LightAPI(mesh_light_prim).CreateIntensityAttr().Set(5.0)
  assert np.max(engine.render()['color'][..., :3]) > 0.1


def test_area_light_primary_visibility():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/lights/sphere.usda')
  UsdGeom.Xformable(stage.GetPrimAtPath('/root/Area')).ClearXformOpOrder()
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_default, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_area_light_vis_default',
      atol=0.1,
      engine=engine,
  )
  assert 0.1 < np.max(image_default[..., :3]) < 6.0

  vis_attr = stage.GetPrimAtPath('/root/Area/Area').CreateAttribute(
      'mitsuba:visibility', Sdf.ValueTypeNames.String
  )
  vis_attr.Set('all')
  image_all, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, output_prefix='test_area_light_vis_all', atol=0.1, engine=engine
  )
  assert np.max(image_all[..., :3]) > 10.0

  vis_attr.Set('secondary')
  test_helpers.robust_assert_close(
      engine.render()['color'], image_default, atol=0.05
  )


def test_dome_light_selection_by_light_link():
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.Camera.Define(stage, '/Camera')
  # The first DomeLight in path order has a restricted lightLink.
  background = UsdLux.DomeLight.Define(stage, '/Lights/Background')
  background.CreateColorAttr((1, 0, 0))
  link = UsdLux.LightAPI(background).GetLightLinkCollectionAPI()
  link.GetExcludesRel().AddTarget('/World')
  UsdLux.DomeLight.Define(stage, '/Lights/Sky').CreateColorAttr((0, 0, 1))
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
  )
  # Only the blue sky dome is visible.
  assert np.allclose(engine.render()['color'][..., :3], [0, 0, 1], atol=1e-5)


def test_update_ies_light():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/lights/ies.usda')
  test_helpers.create_render_settings(stage, resolution=_RENDER_RESOLUTION)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  engine.render()

  # 1. Update intensity in place on the IES DiskLight.
  disk_light = UsdLux.DiskLight.Get(stage, '/root/Area/Area')
  disk_light.GetIntensityAttr().Set(6.0)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_ies_light_intensity',
      atol=1e-3,
      engine=engine,
  )

  # 2. Update shaping:ies:angleScale and turn off shaping:ies:normalize, with
  # the intensity scaled by ~1 / 1000 cd, the average of `step.ies`.
  shaping = UsdLux.ShapingAPI(disk_light.GetPrim())
  shaping.GetShapingIesAngleScaleAttr().Set(0.75)
  shaping.GetShapingIesNormalizeAttr().Set(False)
  disk_light.GetIntensityAttr().Set(0.006)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_ies_light_shaping',
      atol=1e-3,
      engine=engine,
  )

  # 3. Dynamically clear the IES file path (switches from 'ies' back to 'area').
  shaping.GetShapingIesFileAttr().Set(Sdf.AssetPath(''))
  disk_light.GetIntensityAttr().Set(6.0)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_ies_light_cleared',
      atol=1e-3,
      engine=engine,
  )


def test_ies_mesh_light_matches_rect_light():
  """Quad mesh lights with an IES profile match the same `RectLight`."""
  images = {
      light_type: _render(_open_ies_stage(light_type))
      for light_type in ('rect', 'mesh', 'mesh_subsets')
  }
  for light_type, image in images.items():
    test_helpers.write_image(image, f'test_ies_mesh_light_{light_type}.png')
  for light_type in ('mesh', 'mesh_subsets'):
    test_helpers.robust_assert_close(
        images[light_type], images['rect'], atol=1e-3
    )


def test_update_ies_mesh_light_transform():
  """Rotating an IES mesh light updates its light frame."""
  stage = _open_ies_stage('mesh')
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  engine.render()

  light = UsdGeom.Xformable(stage.GetPrimAtPath('/light'))
  light.GetOrderedXformOps()[1].Set(120.0)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_update_ies_mesh_light_transform',
      atol=1e-3,
      spp=256,
      engine=engine,
  )


@pytest.mark.parametrize(
    ('light_type', 'angle_scale', 'normalize'),
    [
        ('disk', 0.0, False),
        ('rect', 0.0, False),
        ('sphere', 0.0, False),
        ('disk', 0.75, True),
    ],
)
def test_ies_matches_hdembree(
    light_type: str, angle_scale: float, normalize: bool
):
  """Checks IES lights against OpenUSD's hdEmbree reference.

  The profile brightens with the horizontal angle and drops back at 360 deg,
  so any rotation or mirroring of the light frame is visible on the floor.
  This also checks `shaping:ies:angleScale` cone scaling and
  `shaping:ies:normalize` total-power scaling.

  Args:
    light_type: The USD light type.
    angle_scale: The `shaping:ies:angleScale` value.
    normalize: The `shaping:ies:normalize` value.
  """
  if 'HdEmbreeRendererPlugin' not in usd_render.get_registered_renderers():
    pytest.skip('HdEmbreeRendererPlugin not available')
  stage = _open_ies_stage(light_type)
  shaping = UsdLux.ShapingAPI(stage.GetPrimAtPath('/light'))
  shaping.GetShapingIesAngleScaleAttr().Set(angle_scale)
  shaping.GetShapingIesNormalizeAttr().Set(normalize)
  name = f'{light_type}_{angle_scale}_{normalize}'

  def normalized_luminance(image: np.ndarray) -> np.ndarray:
    luminance = np.array(image)[..., :3].mean(axis=-1)
    # Average 4x4 pixel blocks to suppress hdEmbree's sampling noise, and only
    # compare the angular distribution, not absolute intensities.
    luminance = luminance.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    return luminance / np.median(luminance[luminance > 0])

  images = {}
  for renderer, overrides in (
      (
          'HdEmbreeRendererPlugin',
          {
              'enableLighting': True,
              'ambientOcclusionSamples': 0,
              'convergedSamplesPerPixel': 256,
          },
      ),
      ('HdMitsubaRendererPlugin', None),
  ):
    images[renderer] = normalized_luminance(
        _render(stage, renderer, overrides)
    )
    test_helpers.write_image(
        images[renderer] / images[renderer].max(),
        f'test_ies_matches_hdembree_{name}_{renderer}.png',
    )
  test_helpers.robust_assert_close(
      images['HdMitsubaRendererPlugin'],
      images['HdEmbreeRendererPlugin'],
      quantile=0.98,
      atol=0.05,
      rtol=0.1,
  )
