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

from __future__ import annotations

import numpy as np
import pytest
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdRender
from pxr import UsdLux
import usd_render

_RENDER_SETTINGS_PATH = '/Render/PrimarySettings'
_DELEGATES = [
    'HdMitsubaRendererPlugin',
    'HdEmbreeRendererPlugin',
]


def _skip_missing_delegate(delegate_id: str):
  if delegate_id not in usd_render.get_registered_renderers():
    pytest.skip(f'{delegate_id} not available')


def _create_stage() -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()

  mesh = UsdGeom.Mesh.Define(stage, '/mesh')
  mesh.GetPointsAttr().Set(
      [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)])
  mesh.GetFaceVertexCountsAttr().Set([4])
  mesh.GetFaceVertexIndicesAttr().Set([3, 2, 1, 0])

  UsdLux.DomeLight.Define(stage, '/World/dome_light')

  camera = UsdGeom.Camera.Define(stage, '/World/camera')
  camera.AddTranslateOp().Set((4, 5, 6))
  camera.AddRotateXOp().Set(-40)
  camera.AddRotateYOp().Set(30)
  camera.GetHorizontalApertureOffsetAttr().Set(0.1)
  camera.GetVerticalApertureOffsetAttr().Set(-0.3)
  settings = UsdRender.Settings.Define(stage, _RENDER_SETTINGS_PATH)
  settings.CreateCameraRel().SetTargets(['/World/camera'])
  stage.GetRootLayer().renderSettingsPrimPath = _RENDER_SETTINGS_PATH
  return stage


def test_instantiate_engine():
  stage = _create_stage()
  engine = usd_render.RenderEngine(stage)
  assert engine is not None


@pytest.mark.parametrize("delegate_id", _DELEGATES)
def test_basic_render(delegate_id: str):
  stage = _create_stage()
  engine = usd_render.RenderEngine(stage)
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=100,
  )
  images = engine.render()
  assert images is not None
  assert 'color' in images
  assert np.mean(images['color']) > 0.1


@pytest.mark.parametrize("delegate_id", _DELEGATES)
def test_render_different_cameras(delegate_id: str):
  stage = _create_stage()
  engine = usd_render.RenderEngine(stage)
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=100,
      camera_path='/World/camera',
  )
  assert engine is not None
  images = engine.render()
  assert images is not None
  assert 'color' in images
  color_image = images['color']
  assert color_image.shape == (73, 100, 4)

  # Modify camera position and render again.
  xformable = UsdGeom.Xformable.Get(stage, '/World/camera')
  xformable.AddTranslateOp(opSuffix='new_translation').Set((1.0, 1.0, 1.0))
  images_mod = engine.render()
  assert images_mod is not None
  assert 'color' in images_mod
  color_image_modified = images_mod['color']
  assert color_image_modified.shape == (73, 100, 4)

  assert np.mean(np.abs(color_image_modified - color_image)) > 0.1


@pytest.mark.parametrize("force_rebuild", [True, False])
@pytest.mark.parametrize("delegate_id", _DELEGATES)
def test_change_width(force_rebuild: bool, delegate_id: str):
  stage = _create_stage()

  engine = usd_render.RenderEngine(stage)
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=100,
  )
  assert engine is not None
  images = engine.render()
  assert images is not None
  assert images['color'].shape == (73, 100, 4)

  render_settings_path = _RENDER_SETTINGS_PATH if force_rebuild else None
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=50,
      render_settings_path=render_settings_path,
  )
  images = engine.render()
  assert images is not None
  assert images['color'].shape == (36, 50, 4)


@pytest.mark.parametrize("delegate_id", _DELEGATES)
def test_render_resolution_updates_on_camera_aspect_ratio_change(delegate_id: str):
  stage = _create_stage()
  camera = UsdGeom.Camera.Get(stage, '/World/camera')
  camera.GetHorizontalApertureAttr().Set(36.0)
  camera.GetVerticalApertureAttr().Set(24.0)

  engine = usd_render.RenderEngine(stage)
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=300,
      camera_path='/World/camera',
  )
  images = engine.render()
  assert images['color'].shape == (200, 300, 4)

  # Change camera aspect ratio to 1.0
  camera.GetVerticalApertureAttr().Set(36.0)

  # Re-configure to trigger resolution update
  _skip_missing_delegate(delegate_id)
  engine.configure(
      hydra_delegate_id=delegate_id,
      width=300,
      camera_path='/World/camera',
  )
  assert engine.render()['color'].shape == (300, 300, 4)


def test_invalid_camera_throws():
  stage = Usd.Stage.CreateInMemory()
  engine = usd_render.RenderEngine(stage)
  renderers = usd_render.get_registered_renderers()
  if not renderers:
    pytest.skip("No renderers registered")
  with pytest.raises(RuntimeError, match="No camera found"):
    engine.configure(hydra_delegate_id=renderers[0])
