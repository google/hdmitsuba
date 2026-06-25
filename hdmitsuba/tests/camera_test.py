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
