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
import usd_render
from hdmitsuba.tests import test_helpers

_RENDER_SETTINGS_PATH = '/renderSettings'


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_stage() -> Usd.Stage:
  """Creates an in-memory USD stage with a sphere, camera, and light."""
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  geometry = UsdGeom.Sphere.Define(stage, '/sphere')
  material = UsdShade.Material.Define(stage, '/sphere/material')
  shader = UsdShade.Shader.Define(stage, '/sphere/material/surface')
  shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
      (0.1, 0.1, 0.9)
  )
  material.CreateSurfaceOutput().ConnectToSource(
      shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )
  UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)

  camera = UsdGeom.Camera.Define(stage, '/camera')
  camera.AddTranslateOp().Set((5, 5, 5))
  camera.AddRotateXOp().Set(-40)
  camera.AddRotateYOp().Set(30)
  sphere_light = UsdLux.SphereLight.Define(stage, '/light')
  sphere_light.CreateIntensityAttr(1.0)
  sphere_light.CreateColorAttr((0.0, 1.0, 0.0))
  sphere_light.CreateRadiusAttr(0.5)
  sphere_light.AddTranslateOp().Set((0, 3.0, 0))

  dome_light = UsdLux.DomeLight.Define(stage, '/domeLight')
  dome_light.CreateIntensityAttr(0.1)
  dome_light.CreateColorAttr((1.0, 1.0, 1.0))

  # Set mitsuba variant in render settings.
  render_settings = UsdRender.Settings.Define(stage, _RENDER_SETTINGS_PATH)
  render_settings.GetPrim().CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, _RENDER_SETTINGS_PATH
  )
  return stage


@pytest.mark.parametrize(
    'renderer',
    ['HdEmbreeRendererPlugin', 'HdMitsubaRendererPlugin'],
    ids=['HdEmbree', 'HdMitsuba']
)
def test_render_simple_scene(renderer):
  stage = _create_stage()
  assert stage is not None

  engine = usd_render.RenderEngine(stage)
  if renderer not in usd_render.get_registered_renderers():
    pytest.skip(f'{renderer} not available')
  engine.configure(
      hydra_delegate_id=renderer,
      width=512,
      camera_path='/camera',
  )
  outputs = engine.render()
  image = outputs['color']
  extension = 'png' if image.dtype == np.uint8 else 'exr'
  test_helpers.write_image(image, f'simple_{renderer}.{extension}')


def _create_flat_stage() -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()
  UsdLux.DomeLight.Define(stage, '/World/dome_light')
  camera = UsdGeom.Camera.Define(stage, '/World/camera')
  camera.AddTranslateOp().Set((0, 0, 5))
  settings = UsdRender.Settings.Define(stage, _RENDER_SETTINGS_PATH)
  settings.CreateCameraRel().SetTargets(['/World/camera'])
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, _RENDER_SETTINGS_PATH
  )
  return stage


def test_progressive_vs_non_progressive_convergence():
  stage = _create_flat_stage()
  # Configure non-progressive engine with 16 SPP
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id="HdMitsubaRendererPlugin",
      width=16,
      overrides={
          'mitsuba:sample_count': 16,
          'enableInteractive': False,
      }
  )
  images_single = engine.render()
  img_single = images_single['color']

  # Configure progressive engine with 16 SPP (rendered as 16 x 1 SPP)
  engine_prog = usd_render.RenderEngine(stage)
  engine_prog.configure(
      hydra_delegate_id="HdMitsubaRendererPlugin",
      width=16,
      overrides={
          'mitsuba:sample_count': 16,
          'enableInteractive': True,
      }
  )
  images_prog = engine_prog.render()
  img_prog = images_prog['color']
  np.testing.assert_allclose(img_single, img_prog, atol=1e-6, rtol=1e-6)
