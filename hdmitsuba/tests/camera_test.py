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

"""Tests for the hdMitsuba render delegate."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender
from pxr import UsdShade
import pytest

from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba
import usd_render


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_plane_geometry(stage: Usd.Stage, path: str = '/mesh') -> None:
  geometry = UsdGeom.Mesh.Define(stage, path)
  points = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]
  face_vertex_counts = [4]
  face_vertex_indices = [3, 2, 1, 0]
  geometry.GetPointsAttr().Set(points)
  geometry.GetFaceVertexCountsAttr().Set(face_vertex_counts)
  geometry.GetFaceVertexIndicesAttr().Set(face_vertex_indices)


def _define_camera(stage: Usd.Stage, path: str = '/camera') -> None:
  camera = UsdGeom.Camera.Define(stage, path)
  camera.AddTranslateOp().Set((4, 5, 6))
  camera.AddRotateXOp().Set(-40)
  camera.AddRotateYOp().Set(30)
  camera.GetHorizontalApertureOffsetAttr().Set(0.1)
  camera.GetVerticalApertureOffsetAttr().Set(-0.3)


def _define_light(stage: Usd.Stage, path: str = '/light') -> None:
  light = UsdLux.DomeLight.Define(stage, path)
  light.CreateIntensityAttr(1.3)
  light.CreateColorAttr((1.0, 1.0, 1.0))


def _setup_stage(
    stage: Usd.Stage,
    define_camera: bool = True,
    define_light: bool = True,
) -> None:
  _create_plane_geometry(stage)
  material = UsdShade.Material.Define(stage, '/mesh/material')
  shader = UsdShade.Shader.Define(stage, '/mesh/material/surface')
  shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
      (0.0, 0.0, 0.0)
  )
  material.CreateSurfaceOutput().ConnectToSource(
      shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )
  geometry = UsdGeom.Mesh.Get(stage, '/mesh')
  UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim())
  UsdShade.MaterialBindingAPI(geometry).Bind(material)

  if define_camera:
    _define_camera(stage)
  if define_light:
    _define_light(stage)
  render_settings_path = '/Render/PrimarySettings'
  render_settings = UsdRender.Settings.Define(stage, render_settings_path)
  render_settings.GetPrim().CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())


def _modify_camera(
    stage: Usd.Stage,
    camera_path: str,
    focal_length_scale: float = 1.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
  camera = UsdGeom.Camera.Get(stage, camera_path)
  focal_length = camera.GetFocalLengthAttr().Get()
  camera.GetFocalLengthAttr().Set(focal_length * focal_length_scale)
  xformable = UsdGeom.Xformable.Get(stage, camera_path)
  xformable.AddTranslateOp(opSuffix='new_translation').Set(translation)


def test_compare_to_hdembree():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, color_data_type='color4f')

  engine = usd_render.RenderEngine(stage)

  if 'HdEmbreeRendererPlugin' not in usd_render.get_registered_renderers():
    pytest.skip('HdEmbreeRendererPlugin not available')
  engine.configure(
      hydra_delegate_id='HdEmbreeRendererPlugin',
      width=512,
  )
  embree_image = engine.render()['color']
  assert embree_image.shape[2] == 4
  embree_mask = embree_image[..., 3] == 0

  # Recreate engine for switching delegate
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      width=512,
  )
  mitsuba_image = engine.render()['color']

  test_helpers.write_image(embree_image, 'test_compare_to_hdembree_embree.png')
  test_helpers.write_image(
      mitsuba_image, 'test_compare_to_hdembree_mitsuba.png'
  )

  mitsuba_mask = mitsuba_image[..., 0] > 0.5
  fraction_mismatch = np.mean(embree_mask != mitsuba_mask)
  assert fraction_mismatch < 0.005


def test_simple_render():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage, output_prefix='test_simple_render', atol=0.05
  )


@pytest.mark.parametrize('depth_of_field', [False, True])
@pytest.mark.parametrize('call_configure_twice', [False, True])
def test_update_camera(call_configure_twice: bool, depth_of_field: bool):
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage)

  camera_path = '/camera'
  camera = UsdGeom.Camera.Get(stage, camera_path)
  suffix = '_dof' if depth_of_field else ''
  if depth_of_field:
    camera.GetFStopAttr().Set(8.0)
    camera.GetFocusDistanceAttr().Set(8.0)

  # Render once with original camera settings.
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path=camera_path,
  )
  outputs = engine.render()
  image_hd_original = outputs['color']

  test_helpers.write_image(
      image_hd_original, f'test_update_camera{suffix}_original_hd.png'
  )

  # Modify the camera in the USD stage and render again.
  if depth_of_field:
    camera.GetFocusDistanceAttr().Set(6.0)
  focal_length = camera.GetFocalLengthAttr().Get()
  camera.GetFocalLengthAttr().Set(focal_length * 1.4)
  xformable = UsdGeom.Xformable.Get(stage, camera_path)
  xformable.AddTranslateOp(opSuffix='new_translation').Set((0.2, 0.2, 0.2))
  if call_configure_twice:  # Everything should still work.
    engine.configure(
        hydra_delegate_id='HdMitsubaRendererPlugin',
        camera_path=camera_path,
    )

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path,
      f'test_update_camera{suffix}_modified',
      atol=0.05,
      engine=engine,
  )


def test_camera_update_translation():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage, define_camera=False)
  _define_light(stage)
  _define_camera(stage, '/camera')

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
  )

  # Apply translation
  _modify_camera(stage, '/camera', translation=(0.2, 0.2, 0.2))
  outputs = engine.render()
  image1 = outputs['color']
  test_helpers.write_image(image1, 'test_camera_update_translation.png')
  assert image1.shape[2] == 3


def test_camera_update_focal_length():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage, define_camera=False)
  _define_light(stage)
  _define_camera(stage, '/camera')

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
  )

  # Apply focal length change and different translation
  _modify_camera(
      stage, '/camera', focal_length_scale=1.4, translation=(0.4, -0.4, 0.4)
  )
  outputs = engine.render()
  image2 = outputs['color']
  test_helpers.write_image(image2, 'test_camera_update_focal_length.png')
  assert image2.shape[2] == 3


def test_irradiancemeter_render():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/irradiancemeter.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  test_helpers.assert_hydra_equal_to_offline(
      stage, '/root/Camera/Camera', 'test_irradiancemeter_render', atol=0.05
  )


def test_irradiancemeter_rebind():
  """Tests that changing `mitsuba:sensor:shape` rebinds the surface sensor."""
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/irradiancemeter.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  second_cube = UsdGeom.Mesh.Define(stage, '/root/Cube2/Cube2')
  source_cube = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  second_cube.GetPointsAttr().Set(source_cube.GetPointsAttr().Get())
  second_cube.GetFaceVertexCountsAttr().Set(
      source_cube.GetFaceVertexCountsAttr().Get()
  )
  second_cube.GetFaceVertexIndicesAttr().Set(
      source_cube.GetFaceVertexIndicesAttr().Get()
  )
  second_cube.GetSubdivisionSchemeAttr().Set(
      source_cube.GetSubdivisionSchemeAttr().Get()
  )
  UsdGeom.Xformable(stage.GetPrimAtPath('/root/Cube2')).AddTranslateOp().Set(
      (0.0, 0.0, -6.0)
  )

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/Camera/Camera',
  )
  image_initial = engine.render()['color']
  test_helpers.write_image(image_initial, 'test_irradiancemeter_rebind_a.png')

  camera_prim = stage.GetPrimAtPath('/root/Camera/Camera')
  camera_prim.GetAttribute('mitsuba:sensor:shape').Set('/root/Cube2/Cube2')

  image_rebound = engine.render()['color']
  test_helpers.write_image(image_rebound, 'test_irradiancemeter_rebind_b.png')

  assert not np.allclose(
      image_initial[..., :3], image_rebound[..., :3], atol=1e-3
  )
  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_offline = np.array(mi.render(scene, spp=128))
  test_helpers.robust_assert_close(
      image_rebound[..., :3], image_offline, atol=0.05
  )


def test_irradiancemeter_target_mesh_rebuild():
  """Tests that rebuilding the target mesh of an irradiancemeter succeeds."""
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/irradiancemeter.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/Camera/Camera',
  )
  image_initial = engine.render()['color']

  # Trigger a mesh rebuild on the target shape without modifying the camera.
  target_prim = stage.GetPrimAtPath('/root/Cube/Cube')
  target_prim.SetActive(False)
  target_prim.SetActive(True)

  image_rebuilt = engine.render()['color']
  test_helpers.robust_assert_close(
      image_initial[..., :3], image_rebuilt[..., :3], atol=0.05
  )


@pytest.mark.parametrize(
    'camera_path',
    ['/root/ThinLensCamera', '/root/RadianceMeter', '/root/DistantSensor'],
)
def test_sensor_specs_render(camera_path: str):
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/sensor_specs.usda'
  )
  is_signal = camera_path != '/root/ThinLensCamera'
  test_helpers.create_render_settings(
      stage, resolution=(8, 8) if is_signal else (64, 32), spp=16
  )

  if not is_signal:
    image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
        stage, camera_path, 'test_sensor_specs_thinlens', spp=None, atol=0.05
    )
    assert np.max(image_hd[..., :3]) > 0.05
    return

  if camera_path == '/root/DistantSensor':
    UsdGeom.Sphere.Define(stage, '/root/Marker').CreateRadiusAttr(0.1)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path=camera_path
  )
  image_hd = engine.render()['color']
  assert image_hd.shape[:2] == (8, 8)

  mi_id = camera_path.replace('/', '_')
  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  sensor = next(s for s in scene.sensors() if s.id() == mi_id)
  image_mi = np.broadcast_to(
      np.array(mi.render(scene, sensor=sensor, spp=16)), (8, 8, 3)
  )
  test_helpers.robust_assert_close(image_hd[..., :3], image_mi, atol=0.05)

  if camera_path == '/root/DistantSensor':
    # Expanding the scene bounding box in place must re-notify `DistantSensor`
    # via `set_scene` so its bounding sphere stays accurate.
    UsdGeom.Xformable(stage.GetPrimAtPath('/root/Marker')).AddTranslateOp().Set(
        (6.0, 0.0, 0.0)
    )
    moved_hd = engine.render()['color']
    moved_scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
    moved_sensor = next(s for s in moved_scene.sensors() if s.id() == mi_id)
    moved_mi = np.broadcast_to(
        np.array(mi.render(moved_scene, sensor=moved_sensor, spp=16)), (8, 8, 3)
    )
    assert not np.allclose(image_hd[..., :3], moved_hd[..., :3], atol=0.005)
    test_helpers.robust_assert_close(moved_hd[..., :3], moved_mi, atol=0.005)


def test_sensor_dynamic_updates_and_crop():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=16)

  # Unattached `irradiancemeter` without `mitsuba:sensor:shape` is skipped.
  UsdGeom.Camera.Define(stage, '/orphan_meter').GetPrim().CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('irradiancemeter')

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('thinlens')
  aperture_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:aperture_radius', Sdf.ValueTypeNames.Float
  )
  aperture_attr.Set(0.0)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:focus_distance', Sdf.ValueTypeNames.Float
  ).Set(1.5)
  fov_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:fov', Sdf.ValueTypeNames.Float
  )
  fov_attr.Set(45.0)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:rfilter:type', Sdf.ValueTypeNames.String
  ).Set('gaussian')
  stddev_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:film:rfilter:stddev', Sdf.ValueTypeNames.Float
  )
  stddev_attr.Set(0.25)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  sharp = engine.render()['color']

  # Live in-place leaf update, attribute removal fallback, and filter rebuild.
  aperture_attr.Set(1.2)
  fov_attr.Set(60.0)
  blurred = engine.render()['color']
  assert not np.allclose(sharp[..., :3], blurred[..., :3], atol=0.02)

  camera_prim.RemoveProperty('mitsuba:sensor:fov')
  stddev_attr.Set(2.5)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/camera',
      'test_sensor_dynamic_updates',
      spp=16,
      atol=0.05,
      engine=engine,
  )

  # Structural plugin-type switch (`thinlens` -> `perspective`) rebuilds sensor.
  camera_prim.GetAttribute('mitsuba:sensor:type').Set('perspective')
  camera_prim.RemoveProperty('mitsuba:sensor:aperture_radius')
  camera_prim.RemoveProperty('mitsuba:sensor:focus_distance')
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/camera',
      'test_sensor_type_switch_perspective',
      spp=16,
      atol=0.05,
      engine=engine,
  )

  # Authored `mitsuba:sensor:film:crop_*` and live crop resize.
  for key, val in (
      ('crop_offset_x', 8),
      ('crop_offset_y', 4),
      ('crop_width', 16),
      ('crop_height', 12),
  ):
    camera_prim.CreateAttribute(
        f'mitsuba:sensor:film:{key}', Sdf.ValueTypeNames.Int
    ).Set(val)
  cropped_hd = engine.render()['color']
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert '_orphan_meter' not in scene_dict
  cropped_mi = np.array(mi.render(mi.load_dict(scene_dict), spp=16))
  assert cropped_mi.shape == (12, 16, 3)
  test_helpers.robust_assert_close(
      cropped_hd[4:16, 8:24, :3], cropped_mi, atol=0.05
  )
  np.testing.assert_array_equal(cropped_hd[:4, :, :3], 0.0)


def test_sensor_film_formats_and_interactive_sampler():
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/sensor_specs.usda'
  )
  settings = test_helpers.create_render_settings(
      stage, resolution=(16, 16), spp=16, color_data_type='color4f'
  )
  settings_prim = settings.GetPrim()
  cam_prim = stage.GetPrimAtPath('/root/ThinLensCamera')
  cam_prim.CreateAttribute(
      'mitsuba:sensor:film:sample_border', Sdf.ValueTypeNames.Bool
  ).Set(True)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/ThinLensCamera',
  )
  rgba = engine.render()['color']
  assert rgba.shape == (16, 16, 4)
  np.testing.assert_allclose(rgba[0, 0, 3], 0.0, atol=1e-5)
  np.testing.assert_allclose(rgba[8, 8, 3], 1.0, atol=1e-3)

  # Explicit `pixel_format` is preserved across 3/4-channel AOV toggles.
  fmt_attr = cam_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_format', Sdf.ValueTypeNames.String
  )
  fmt_attr.Set('xyz')
  xyz = engine.render()['color']
  offline_xyz = np.array(
      mi.render(mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage)))
  )
  test_helpers.robust_assert_close(xyz[..., :3], offline_xyz, atol=0.05)

  # RGBE 3-channel film + per-camera sampler SPP converge progressively, even
  # when an unrelated RenderSettings attribute changes mid-accumulation.
  cam_prim.RemoveProperty('mitsuba:sensor:film:pixel_format')
  cam_prim.CreateAttribute(
      'mitsuba:sensor:film:file_format', Sdf.ValueTypeNames.String
  ).Set('rgbe')
  cam_prim.GetAttribute('mitsuba:sensor:sampler:sample_count').Set(4)
  settings_prim.GetAttribute('mitsuba:sample_count').Set(2)
  settings_prim.CreateAttribute(
      'mitsuba:interactive_samples_per_pass', Sdf.ValueTypeNames.Int
  ).Set(2)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/ThinLensCamera',
      overrides={'enableInteractive': True},
  )
  engine.render()
  assert not engine.is_converged()
  engine.render()
  assert engine.is_converged()
  assert engine.get_render_stats()['totalSamples'] == 4
