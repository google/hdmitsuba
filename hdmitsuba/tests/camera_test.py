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
  test_helpers.create_render_settings(stage)

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


@pytest.mark.parametrize('call_configure_twice', [False, True])
def test_update_camera(call_configure_twice: bool):
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage)
  test_helpers.create_render_settings(stage)

  camera_path = '/camera'

  # Render once with original camera settings.
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path=camera_path,
  )
  outputs = engine.render()
  image_hd_original = outputs['color']

  test_helpers.write_image(
      image_hd_original, 'test_update_camera_original_hd.png'
  )

  # Modify the camera in the USD stage and render again.
  camera = UsdGeom.Camera.Get(stage, camera_path)
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
      'test_update_camera_modified',
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
  assert image1.shape[2] == 4


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
  assert image2.shape[2] == 4


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


def _render_translated_camera(
    translations: list[tuple[float | None, tuple[float, float, float]]],
    shutter: tuple[float, float] | None = None,
) -> np.ndarray:
  """Renders the test scene with a camera translate op at time code 0.

  Args:
    translations: (time, value) pairs for the translate op. A time of None
      authors the default value instead of a time sample.
    shutter: Optional (shutterOpen, shutterClose) interval of the camera.

  Returns:
    The RGB channels of the rendered color output.
  """
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage, define_camera=False)
  camera = UsdGeom.Camera.Define(stage, '/camera')
  if shutter is not None:
    camera.CreateShutterOpenAttr().Set(shutter[0])
    camera.CreateShutterCloseAttr().Set(shutter[1])
  translate_op = UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp()
  for time, value in translations:
    translate_op.Set(
        value, time=Usd.TimeCode.Default() if time is None else time
    )
  camera.AddRotateXOp().Set(-40)
  camera.AddRotateYOp().Set(30)
  test_helpers.create_render_settings(stage)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
  )
  # Time-sampled attributes have no value at the default time code, so the
  # frame must be rendered at an explicit time for the keyframes to resolve.
  return engine.render(time_code=0)['color'][..., :3]


def test_animated_camera():
  shutter = (-0.5, 0.5)
  start, end = (4.0, 5.0, 6.0), (4.5, 5.0, 6.0)
  mid = (4.25, 5.0, 6.0)

  def dark_fraction(image: np.ndarray) -> float:
    # Pixels covered by the black plane, as opposed to the dome light.
    return float((image.max(axis=-1) < 0.5).mean())

  image_static = _render_translated_camera([(None, start)])
  assert dark_fraction(image_static) > 0.1

  # Identical keyframes must reproduce the static camera.
  image_same = _render_translated_camera(
      [(shutter[0], start), (shutter[1], start)], shutter
  )
  test_helpers.robust_assert_close(image_same, image_static, atol=0.02)

  # A moving camera keeps the plane in view.
  image_moving = _render_translated_camera(
      [(shutter[0], start), (shutter[1], end)], shutter
  )
  test_helpers.write_image(image_moving, 'test_animated_camera.png')
  assert dark_fraction(image_moving) > 0.75 * dark_fraction(image_static)

  # Time sampling over the shutter interval changes the pixels along the plane
  # edges, but not the rest of the image, relative to the static mid-shutter
  # pose.
  image_mid = _render_translated_camera([(None, mid)])
  blurred = (np.abs(image_moving - image_mid).max(axis=-1) > 0.05).mean()
  assert 0.02 < blurred < 0.3, blurred
