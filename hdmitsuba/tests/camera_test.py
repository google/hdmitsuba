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
import pytest

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender
from pxr import UsdShade
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

  test_helpers.write_image(
      embree_image, 'test_compare_to_hdembree_embree.png'
  )
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
  _modify_camera(stage, '/camera', focal_length_scale=1.4,
                 translation=(0.4, -0.4, 0.4))
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


def test_thinlens_sensor_and_dynamic_update():
  """Tests thinlens sensor parameters and dynamic Hydra 2.0 invalidation."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=32)

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('thinlens')
  aperture_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:aperture_radius', Sdf.ValueTypeNames.Float
  )
  aperture_attr.Set(0.0)
  focus_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:focus_distance', Sdf.ValueTypeNames.Float
  )
  focus_attr.Set(1.5)
  fov_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:fov', Sdf.ValueTypeNames.Float
  )
  fov_attr.Set(45.0)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
  )
  image_sharp = engine.render()['color']

  # Dynamically update traversable thinlens leaves in-place.
  aperture_attr.Set(1.2)
  focus_attr.Set(2.5)
  fov_attr.Set(60.0)
  image_blurred = engine.render()['color']

  assert not np.allclose(
      image_sharp[..., :3], image_blurred[..., :3], atol=0.02
  )
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/camera',
      'test_thinlens_sensor_and_dynamic_update',
      atol=0.05,
      engine=engine,
      spp=32,
  )

  # Removing an authored override falls back to the USD camera FOV cleanly.
  camera_prim.RemoveProperty('mitsuba:sensor:fov')
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      '/camera',
      'test_thinlens_sensor_removed_fov_fallback',
      atol=0.05,
      engine=engine,
      spp=32,
  )


def test_nested_sensor_film_pixel_filter():
  """Tests nested `mitsuba:sensor:film:pixel_filter:*` parameters and parity."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=32)

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('gaussian')
  stddev_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:stddev', Sdf.ValueTypeNames.Float
  )
  stddev_attr.Set(0.25)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
  )
  image_narrow = engine.render()['color']

  # Dynamically widen the Gaussian reconstruction filter stddev.
  stddev_attr.Set(2.5)
  image_wide = engine.render()['color']

  assert not np.allclose(image_narrow[..., :3], image_wide[..., :3], atol=0.01)

  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_offline = np.array(mi.render(scene, spp=32))
  test_helpers.robust_assert_close(
      image_wide[..., :3], image_offline, atol=0.05
  )


def test_sensor_film_overrides_with_alpha_and_pixel_format():
  """Tests film overrides across RGBA buffer toggles and explicit formats."""
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  sphere = UsdGeom.Sphere.Define(stage, '/sphere')
  material = UsdShade.Material.Define(stage, '/sphere/material')
  shader = UsdShade.Shader.Define(stage, '/sphere/material/surface')
  shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
      (0.9, 0.05, 0.05)
  )
  material.CreateSurfaceOutput().ConnectToSource(
      shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )
  UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
  UsdGeom.Camera.Define(stage, '/camera').AddTranslateOp().Set((0, 0, 5))
  distant = UsdLux.DistantLight.Define(stage, '/light')
  distant.CreateIntensityAttr(2.0)
  distant.CreateNormalizeAttr(True)
  test_helpers.create_render_settings(
      stage, resolution=(32, 32), spp=32, color_data_type='color4f'
  )

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:sample_border', Sdf.ValueTypeNames.Bool
  ).Set(True)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('gaussian')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:stddev', Sdf.ValueTypeNames.Float
  ).Set(1.0)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  rgba_image = engine.render()['color']
  assert rgba_image.shape == (32, 32, 4)
  np.testing.assert_allclose(rgba_image[0, 0, 3], 0.0, atol=1e-5)
  np.testing.assert_allclose(rgba_image[16, 16, 3], 1.0, atol=1e-5)

  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  scene_dict['_camera']['film']['pixel_format'] = 'rgba'
  offline_rgba = np.array(mi.render(mi.load_dict(scene_dict), spp=32))
  test_helpers.robust_assert_close(rgba_image, offline_rgba, atol=0.05)

  # Explicit pixel_format must not be overwritten by 3/4-channel AOV toggles.
  pixel_format_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_format', Sdf.ValueTypeNames.String
  )
  pixel_format_attr.Set('xyz')
  xyz_image = engine.render()['color']
  offline_xyz = np.array(
      mi.render(mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage)), spp=32)
  )
  test_helpers.robust_assert_close(xyz_image[..., :3], offline_xyz, atol=0.05)
  assert not np.allclose(
      xyz_image[16, 16, :3], rgba_image[16, 16, :3], atol=0.05
  )

  UsdRender.Var.Get(stage, '/Render/Vars/color').GetDataTypeAttr().Set(
      'color3f'
  )
  pixel_format_attr.Set('xyza')
  xyza_rgb_image = engine.render()['color']
  test_helpers.robust_assert_close(xyza_rgb_image, offline_xyz, atol=0.05)


def test_camera_sampler_override_and_radiancemeter():
  """Tests camera-local sampler precedence, fov_axis, and radiancemeter."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  settings = test_helpers.create_render_settings(
      stage, resolution=(64, 32), spp=4
  )
  settings.CreateCameraRel().SetTargets(['/camera'])

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:fov_axis', Sdf.ValueTypeNames.String
  ).Set('y')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:fov', Sdf.ValueTypeNames.Float
  ).Set(35.0)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:sampler:type', Sdf.ValueTypeNames.String
  ).Set('stratified')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:sampler:jitter', Sdf.ValueTypeNames.Bool
  ).Set(False)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:sampler:sample_count', Sdf.ValueTypeNames.Int
  ).Set(16)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_hydra = engine.render()['color']
  assert engine.get_render_stats()['totalSamples'] == 16

  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_offline = np.array(mi.render(scene))
  test_helpers.robust_assert_close(
      image_hydra[..., :3], image_offline, atol=0.05
  )

  # Switch to a 1x1 radiancemeter using spatial Point3f/Vector3f attributes.
  test_helpers.create_render_settings(stage, resolution=(1, 1), spp=4)
  meter_cam = UsdGeom.Camera.Define(stage, '/meter')
  meter_prim = meter_cam.GetPrim()
  meter_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('radiancemeter')
  meter_prim.CreateAttribute(
      'mitsuba:sensor:origin', Sdf.ValueTypeNames.Point3f
  ).Set((0.0, 0.0, 5.0))
  meter_prim.CreateAttribute(
      'mitsuba:sensor:direction', Sdf.ValueTypeNames.Vector3f
  ).Set((0.0, 0.0, -1.0))

  settings.GetCameraRel().SetTargets(['/meter'])
  meter_hydra = engine.render()['color']
  assert meter_hydra.shape[:2] == (1, 1)

  meter_scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  meter_sensor = next(s for s in meter_scene.sensors() if s.id() == '_meter')
  meter_offline = np.array(mi.render(meter_scene, sensor=meter_sensor, spp=4))
  test_helpers.robust_assert_close(
      meter_hydra[..., :3], meter_offline, atol=0.05
  )


def test_rgbe_film_with_rgba_aov_converges():
  """Tests that 3-channel RGBE film with color4f AOV converges progressively."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  settings = test_helpers.create_render_settings(
      stage, resolution=(16, 16), spp=4, color_data_type='color4f'
  )
  settings.GetPrim().CreateAttribute(
      'mitsuba:interactive_samples_per_pass', Sdf.ValueTypeNames.Int
  ).Set(2)
  stage.GetPrimAtPath('/camera').CreateAttribute(
      'mitsuba:sensor:film:file_format', Sdf.ValueTypeNames.String
  ).Set('rgbe')

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
      overrides={'enableInteractive': True},
  )
  engine.render()
  assert not engine.is_converged()
  engine.render()
  assert engine.is_converged()


def test_per_camera_sampler_convergence_and_standalone_defaults():
  """Tests per-camera sampler SPP convergence and standalone default SPP."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  settings = test_helpers.create_render_settings(
      stage, resolution=(16, 16), spp=2
  )
  settings_prim = settings.GetPrim()
  settings_prim.CreateAttribute(
      'mitsuba:sampler:type', Sdf.ValueTypeNames.String
  ).Set('stratified')
  settings_prim.CreateAttribute(
      'mitsuba:sampler:jitter', Sdf.ValueTypeNames.Bool
  ).Set(False)
  settings_prim.CreateAttribute(
      'mitsuba:interactive_samples_per_pass', Sdf.ValueTypeNames.Int
  ).Set(2)

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:sampler:sample_count', Sdf.ValueTypeNames.Int
  ).Set(4)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
      overrides={'enableInteractive': True},
  )
  # Pass 1 renders 2 of 4 camera samples (exceeding RenderSettings spp=2) and
  # must remain unconverged until pass 2 reaches active_sample_count=4.
  engine.render()
  assert not engine.is_converged()
  engine.render()
  assert engine.is_converged()
  assert engine.get_render_stats()['totalSamples'] == 4

  # A camera specifying only `mitsuba:sensor:sampler:type` uses Mitsuba's
  # default sample_count=4 rather than inheriting RenderSettings spp=32.
  settings_prim.GetAttribute('mitsuba:sample_count').Set(32)
  mj_cam = UsdGeom.Camera.Define(stage, '/cam_mj')
  mj_cam.AddTranslateOp().Set((0, 0, 5))
  mj_cam.GetPrim().CreateAttribute(
      'mitsuba:sensor:sampler:type', Sdf.ValueTypeNames.String
  ).Set('multijitter')

  batch_engine = usd_render.RenderEngine(stage)
  batch_engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/cam_mj'
  )
  mj_hydra = batch_engine.render()['color']
  assert batch_engine.get_render_stats()['totalSamples'] == 4
  mj_scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  mj_sensor = next(s for s in mj_scene.sensors() if s.id() == '_cam_mj')
  mj_offline = np.array(mi.render(mj_scene, sensor=mj_sensor))
  test_helpers.robust_assert_close(mj_hydra[..., :3], mj_offline, atol=0.05)


def test_sensor_film_crop_window_and_unattached_irradiancemeter():
  """Tests authored `mitsuba:sensor:film:crop_*` and unattached irradiancemeter."""
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=8)

  # An unattached `irradiancemeter` without `mitsuba:sensor:shape` must be
  # skipped cleanly in both Hydra and offline translation.
  orphan = UsdGeom.Camera.Define(stage, '/orphan_meter')
  orphan.GetPrim().CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('irradiancemeter')

  camera_prim = stage.GetPrimAtPath('/camera')
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:crop_offset_x', Sdf.ValueTypeNames.Int
  ).Set(8)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:crop_offset_y', Sdf.ValueTypeNames.Int
  ).Set(4)
  crop_w_attr = camera_prim.CreateAttribute(
      'mitsuba:sensor:film:crop_width', Sdf.ValueTypeNames.Int
  )
  crop_w_attr.Set(16)
  camera_prim.CreateAttribute(
      'mitsuba:sensor:film:crop_height', Sdf.ValueTypeNames.Int
  ).Set(12)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  image_hydra = engine.render()['color']
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert '_orphan_meter' not in scene_dict
  image_offline = np.array(mi.render(mi.load_dict(scene_dict), spp=8))
  assert image_offline.shape == (12, 16, 3)
  test_helpers.robust_assert_close(
      image_hydra[4:16, 8:24, :3], image_offline, atol=0.05
  )
  mask = np.ones((32, 32), dtype=bool)
  mask[4:16, 8:24] = False
  np.testing.assert_array_equal(image_hydra[..., :3][mask], 0.0)

  # Dynamic in-place crop update without rebuilding the sensor.
  crop_w_attr.Set(20)
  image_hydra_wider = engine.render()['color']
  image_offline_wider = np.array(
      mi.render(mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage)), spp=8)
  )
  assert image_offline_wider.shape == (12, 20, 3)
  test_helpers.robust_assert_close(
      image_hydra_wider[4:16, 8:28, :3], image_offline_wider, atol=0.05
  )
  np.testing.assert_array_equal(image_hydra_wider[:, 28:, :3], 0.0)
