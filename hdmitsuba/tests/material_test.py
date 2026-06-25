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

"""Tests for the hdMitsuba render delegate Materials."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Usd
from pxr import UsdShade
import usd_render
from hdmitsuba.tests import test_helpers


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


@pytest.mark.parametrize(
    'scene_name',
    ['materials', 'checkerboard', 'normalmap', 'nodegraphs', 'display_color', 'mitsuba_bitmap']
)
def test_render(scene_name: str):
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/{scene_name}.usda'
  )
  camera_path = '/root/Camera/Camera'
  test_helpers.create_render_settings(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage, camera_path, f'test_render_{scene_name}', atol=0.05
  )


def test_texture_fallback():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/texture_fallback.usda'
  )
  camera_path = '/root/Camera/Camera'
  test_helpers.create_render_settings(stage)

  image_hd, image_usd = test_helpers.assert_hydra_equal_to_offline(
      stage, camera_path, 'test_fallback_texture', spp=8, atol=0.05
  )

  info_msg = (
      'The fallback color being green central pixel should have its green'
      ' component dominant.'
  )
  center_pixel_hd = image_hd[70, 100, :3]
  assert center_pixel_hd[1] > center_pixel_hd[0], info_msg
  assert center_pixel_hd[1] > center_pixel_hd[2], info_msg

  center_pixel_usd = image_usd[70, 100, :3]
  assert center_pixel_usd[1] > center_pixel_usd[0], info_msg
  assert center_pixel_usd[1] > center_pixel_usd[2], info_msg


def test_texture_fallback_corrupt():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/texture_fallback_corrupt.usda'
  )
  camera_path = '/root/Camera/Camera'
  test_helpers.create_render_settings(stage)

  image_hd, image_usd = test_helpers.assert_hydra_equal_to_offline(
      stage, camera_path, 'test_fallback_texture_corrupt', spp=8, atol=0.05
  )

  info_msg = (
      'The fallback color being blue central pixel should have its blue'
      ' component dominant.'
  )
  center_pixel_hd = image_hd[70, 100, :3]
  assert center_pixel_hd[2] > center_pixel_hd[0], info_msg
  assert center_pixel_hd[2] > center_pixel_hd[1], info_msg

  center_pixel_usd = image_usd[70, 100, :3]
  assert center_pixel_usd[2] > center_pixel_usd[0], info_msg
  assert center_pixel_usd[2] > center_pixel_usd[1], info_msg


def test_modify_material():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/materials.usda'
  )
  session_layer = stage.GetSessionLayer()
  stage.SetEditTarget(session_layer)

  camera_path = '/root/Camera/Camera'
  test_helpers.create_render_settings(stage)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path=camera_path,
  )

  image_original = engine.render()['color']

  # Modify the Material by disconnecting the texture of the diffuse ground.
  # "ModifyMaterial" step
  shader = UsdShade.Shader.Get(
      stage, '/root/_materials/ground/Principled_BSDF'
  )
  assert shader is not None
  diffuse_color = shader.GetInput('diffuseColor')
  assert diffuse_color is not None
  diffuse_color.ClearSources()
  diffuse_color.Set(Gf.Vec3f(0.0, 0.0, 1.0))

  image_modified = engine.render()['color']
  test_helpers.write_image(
      image_original, 'test_modify_material_original.png'
  )
  test_helpers.write_image(
      image_modified, 'test_modify_material_modified.png'
  )

  assert np.mean(np.abs(image_modified - image_original)) > 0.02

  # "ResetMaterial" step
  # Reset the Material to its original state by removing the session override.
  diffuse_color.GetAttr().Clear()
  image_reset = engine.render()['color']
  test_helpers.robust_assert_close(image_reset, image_original, atol=0.05)
