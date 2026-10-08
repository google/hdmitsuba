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

"""Tests for transformation motion blur in hdMitsuba."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Usd
from pxr import UsdGeom
import usd_render
from hdmitsuba.tests import test_helpers

_CAMERA_PATH = '/root/Camera/Camera'
_SPP = 256
_TIME = Usd.TimeCode(0.5)


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _set_keyframes(stage: Usd.Stage, attr_path: str, start, end) -> Usd.Attribute:
  attr = stage.GetAttributeAtPath(attr_path)
  attr.Set(start, 0.0)
  attr.Set(end, 1.0)
  return attr


def _set_shutter(
    stage: Usd.Stage, shutter: tuple[float, float], path: str = _CAMERA_PATH
) -> None:
  camera = UsdGeom.Camera.Get(stage, path)
  camera.GetShutterOpenAttr().Set(shutter[0])
  camera.GetShutterCloseAttr().Set(shutter[1])


def _load_stage(scene_name: str = 'point') -> Usd.Stage:
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/{scene_name}.usda'
  )
  _set_shutter(stage, (-0.5, 0.5))
  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=_SPP)
  return stage


def _assert_hydra_equal_to_offline(
    stage: Usd.Stage, engine: usd_render.RenderEngine, output_prefix: str
) -> np.ndarray:
  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix=output_prefix,
      spp=_SPP,
      atol=1e-3,
      time=_TIME,
      engine=engine,
  )
  return image_hd[..., :3]


@pytest.mark.parametrize(
    'scene_name,attr_path,start,end',
    [
        ('point', '/root/Camera.xformOp:translate', (-1.5, -2, 2.5), (1.5, -2, 2.5)),
        ('directional', '/root/Area.xformOp:rotateXYZ', (-60, 0, 0), (60, 0, 0)),
        ('envmap', '/root/env_light.xformOp:rotateXYZ', (90, 0, 0), (90, 0, 300)),
        ('point', '/root/Area.xformOp:translate', (-2, 0, 1), (2, 0, 1)),
        ('spot', '/root/Area.xformOp:translate', (-2, 0, 1), (2, 0, 1)),
        ('sphere', '/root/Area.xformOp:translate', (-2, 0, 1), (2, 0, 1)),
    ],
)
def test_motion_blur(scene_name: str, attr_path: str, start, end):
  stage = _load_stage(scene_name)
  if scene_name == 'directional':
    # Non-uniform scale verifies that scale and induced shear are stripped.
    stage.GetAttributeAtPath('/root/Area.xformOp:scale').Set((1.0, 2.0, 1.0))
  _set_keyframes(stage, attr_path, start, end)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image = _assert_hydra_equal_to_offline(
      stage, engine, f'test_motion_{scene_name}'
  )
  _set_shutter(stage, (0.0, 0.0))
  image_static = engine.render(time_code=_TIME)['color'][..., :3]
  # Area lights ('sphere') do not support motion blur and render at frame time.
  if scene_name != 'sphere':
    assert np.mean(np.abs(image - image_static)) > 0.02
  else:
    test_helpers.robust_assert_close(image, image_static, atol=0.1)


def test_interactive_motion_updates():
  stage = _load_stage()
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path=_CAMERA_PATH
  )

  # Static -> animated.
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_static')
  light_translate = _set_keyframes(
      stage, '/root/Area.xformOp:translate', (-1.0, 0.0, 1.3), (1.0, 0.0, 1.3)
  )
  camera_translate = _set_keyframes(
      stage, '/root/Camera.xformOp:translate', (-0.5, -2, 2.5), (0.5, -2, 2.5)
  )
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_animated')

  # Animated -> animated with the same keyframe count (records frozen kernel),
  # then with a different keyframe count (replays with updated keyframe count).
  camera_translate.Set((-1.5, -2, 2.5), 0.0)
  engine.render(time_code=_TIME)
  light_translate.Set((0.0, 0.8, 1.3), 0.3)
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_keyframes')

  # Adding another camera widens the scene motion interval.
  _set_shutter(stage, (0.0, 0.5))
  camera_2 = UsdGeom.Camera.Define(stage, '/camera2')
  _set_shutter(stage, (-1.0, 0.5), '/camera2')
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_camera2')

  # Inverted shutter on the rendered camera shoots rays at shutter offset 0,
  # even while camera_2 keeps the scene motion interval non-empty.
  _set_shutter(stage, (0.5, -0.5))
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_inverted_shutter')

  # Removing camera_2 leaves no valid shutter (animated -> static).
  camera_2.GetPrim().SetActive(False)
  _assert_hydra_equal_to_offline(stage, engine, 'test_updates_static_end')
