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


def _sublayer_shape_scene(scene_name: str) -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/{scene_name}.usda'
  )
  return stage


@pytest.mark.parametrize(
    ('scene_name', 'camera_path', 'expected_irradiance'),
    [
        ('irradiancemeter', '/root/Camera/Camera', None),
        ('irradiancemeter_subsets', '/root/Camera/Camera', 0.5 * np.pi),
        ('irradiancemeter_subsets', '/root/PlaneMeter', 0.5 * np.pi),
    ],
)
def test_irradiancemeter_render(
    scene_name: str, camera_path: str, expected_irradiance: float | None
):
  stage = _sublayer_shape_scene(scene_name)
  target_camera = Sdf.Path(camera_path)
  for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera) and prim.GetPath() != target_camera:
      prim.SetActive(False)
  test_helpers.create_render_settings(stage, resolution=(64, 64))

  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path,
      f'test_{scene_name}_{target_camera.name}_render',
      atol=0.05,
  )
  if expected_irradiance is not None:
    np.testing.assert_allclose(
        image_hd[..., :3], expected_irradiance, atol=0.02
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


def _assert_signal_equal_to_offline(
    image_hd: np.ndarray,
    stage: Usd.Stage,
    camera_path: str,
    atol: float = 0.05,
) -> None:
  mitsuba_id = camera_path.replace('/', '_')
  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  sensor = next(s for s in scene.sensors() if s.id() == mitsuba_id)
  expected = np.broadcast_to(
      np.array(mi.render(scene, sensor=sensor)), image_hd[..., :3].shape
  )
  np.testing.assert_allclose(image_hd[..., :3], expected, atol=atol)


def test_thinlens_sensor_spec_render():
  stage = _sublayer_shape_scene('sensor_specs')
  test_helpers.create_render_settings(stage, resolution=(64, 32), spp=16)

  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/root/ThinLensCamera',
      'test_sensor_specs_thinlens',
      spp=None,
      atol=0.05,
  )
  assert np.max(image_hd[..., :3]) > 0.05


@pytest.mark.parametrize(
    'camera_path', ['/root/RadianceMeter', '/root/DistantSensor']
)
def test_signal_sensor_specs_render(camera_path: str):
  stage = _sublayer_shape_scene('sensor_specs')
  test_helpers.create_render_settings(stage, resolution=(8, 8), spp=16)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path=camera_path
  )
  image_hd = engine.render()['color']
  assert image_hd.shape[:2] == (8, 8)
  assert np.max(image_hd[..., :3]) > 0.01
  _assert_signal_equal_to_offline(image_hd, stage, camera_path)


def test_distant_sensor_bbox_update():
  stage = _sublayer_shape_scene('sensor_specs')
  test_helpers.create_render_settings(stage, resolution=(8, 8), spp=16)
  UsdGeom.Sphere.Define(stage, '/root/Marker').CreateRadiusAttr(0.1)

  camera_path = '/root/DistantSensor'
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path=camera_path
  )
  image_initial = engine.render()['color']
  _assert_signal_equal_to_offline(image_initial, stage, camera_path)

  # Expanding the scene bounding box in place must re-notify `DistantSensor`
  # via `set_scene` so its bounding sphere stays accurate.
  UsdGeom.Xformable(stage.GetPrimAtPath('/root/Marker')).AddTranslateOp().Set(
      (6.0, 0.0, 0.0)
  )
  image_moved = engine.render()['color']
  assert not np.allclose(
      image_initial[..., :3], image_moved[..., :3], atol=0.005
  )
  _assert_signal_equal_to_offline(image_moved, stage, camera_path, atol=0.005)


def test_sensor_spec_dynamic_updates():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=16)

  camera_prim = stage.GetPrimAtPath('/camera')
  for name, type_name, value in (
      ('mitsuba:sensor:type', Sdf.ValueTypeNames.String, 'thinlens'),
      ('mitsuba:sensor:aperture_radius', Sdf.ValueTypeNames.Float, 0.0),
      ('mitsuba:sensor:focus_distance', Sdf.ValueTypeNames.Float, 1.5),
      ('mitsuba:sensor:fov', Sdf.ValueTypeNames.Float, 45.0),
      (
          'mitsuba:sensor:film:rfilter:type',
          Sdf.ValueTypeNames.String,
          'gaussian',
      ),
      ('mitsuba:sensor:film:rfilter:stddev', Sdf.ValueTypeNames.Float, 0.25),
  ):
    camera_prim.CreateAttribute(name, type_name).Set(value)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  sharp = engine.render()['color']

  # Updating sensor attributes, removing overrides, and changing filter params.
  camera_prim.GetAttribute('mitsuba:sensor:aperture_radius').Set(1.2)
  camera_prim.GetAttribute('mitsuba:sensor:fov').Set(60.0)
  blurred = engine.render()['color']
  assert not np.allclose(sharp[..., :3], blurred[..., :3], atol=0.02)

  camera_prim.RemoveProperty('mitsuba:sensor:fov')
  camera_prim.GetAttribute('mitsuba:sensor:film:rfilter:stddev').Set(2.5)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/camera',
      'test_sensor_dynamic_updates',
      spp=16,
      atol=0.05,
      engine=engine,
  )

  # Switching sensor plugin type (`thinlens` -> `perspective`) rebuilds sensor.
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


def test_sensor_film_crop_and_orphan_meter():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=16)

  # Unattached `irradiancemeter` without `mitsuba:sensor:shape` is skipped.
  UsdGeom.Camera.Define(stage, '/orphan_meter').GetPrim().CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('irradiancemeter')

  camera_prim = stage.GetPrimAtPath('/camera')
  for key, value in (
      ('crop_offset_x', 8),
      ('crop_offset_y', 4),
      ('crop_width', 16),
      ('crop_height', 12),
  ):
    camera_prim.CreateAttribute(
        f'mitsuba:sensor:film:{key}', Sdf.ValueTypeNames.Int
    ).Set(value)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  cropped_hd = engine.render()['color']
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert '_orphan_meter' not in scene_dict
  cropped_offline = np.array(mi.render(mi.load_dict(scene_dict)))
  assert cropped_offline.shape == (12, 16, 3)
  test_helpers.robust_assert_close(
      cropped_hd[4:16, 8:24, :3], cropped_offline, atol=0.05
  )
  outside = np.ones((32, 32), dtype=bool)
  outside[4:16, 8:24] = False
  np.testing.assert_array_equal(cropped_hd[..., :3][outside], 0.0)


def test_invalid_sensor_type_falls_back_to_perspective():
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=16)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  perspective_image = engine.render()['color']

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('nonexistent_sensor')
  fallback_image = engine.render()['color']
  test_helpers.robust_assert_close(
      fallback_image[..., :3], perspective_image[..., :3], atol=0.05
  )


def test_sensor_film_alpha_and_pixel_format():
  stage = _sublayer_shape_scene('sensor_specs')
  test_helpers.create_render_settings(
      stage, resolution=(16, 16), spp=16, color_data_type='color4f'
  )

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/ThinLensCamera',
  )
  rgba = engine.render()['color']
  assert rgba.shape == (16, 16, 4)
  np.testing.assert_allclose(rgba[0, 0, 3], 0.0, atol=1e-5)
  np.testing.assert_allclose(rgba[8, 8, 3], 1.0, atol=1e-3)

  # Explicit `pixel_format` overrides the default RGBA film format.
  stage.GetPrimAtPath('/root/ThinLensCamera').CreateAttribute(
      'mitsuba:sensor:film:pixel_format', Sdf.ValueTypeNames.String
  ).Set('xyz')
  xyz = engine.render()['color']
  offline_xyz = np.array(
      mi.render(mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage)))
  )
  test_helpers.robust_assert_close(xyz[..., :3], offline_xyz, atol=0.05)


def test_per_camera_sampler_progressive_convergence():
  stage = _sublayer_shape_scene('sensor_specs')
  settings_prim = test_helpers.create_render_settings(
      stage, resolution=(16, 16), spp=2, color_data_type='color4f'
  ).GetPrim()
  camera_prim = stage.GetPrimAtPath('/root/ThinLensCamera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:file_format', Sdf.ValueTypeNames.String
  ).Set('rgbe')
  camera_prim.GetAttribute('mitsuba:sensor:sampler:sample_count').Set(4)
  settings_prim.CreateAttribute(
      'mitsuba:interactive_samples_per_pass', Sdf.ValueTypeNames.Int
  ).Set(2)

  # Per-camera sampler SPP (4) overrides RenderSettings SPP (2) on a
  # 3-channel RGBE film in progressive interactive mode.
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/root/ThinLensCamera',
      overrides={'enableInteractive': True},
  )
  engine.render()
  assert not engine.is_converged()
  color = engine.render()['color']
  assert engine.is_converged()
  assert engine.get_render_stats()['totalSamples'] == 4
  np.testing.assert_array_equal(color[..., 3], 1.0)
