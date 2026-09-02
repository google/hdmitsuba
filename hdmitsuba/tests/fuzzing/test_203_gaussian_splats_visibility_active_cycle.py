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

"""Fuzz testing: visibility and active/inactive toggle cycle on Gaussian Splats in USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_gaussian_splats_visibility_active_cycle():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/gaussian_splats.usda'
  )
  render_settings = test_helpers.create_render_settings(stage, resolution=(128, 128))
  render_settings.GetPrim().CreateAttribute(
      'mitsuba:integrator:type', Sdf.ValueTypeNames.String
  ).Set('volprim_rf_basic')

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/World/camera')

  # Initial render
  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))

  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.05
  )
  assert np.max(image_hd_initial[..., :3]) > 0.05

  # Step 1: Make the particle field invisible
  splats_prim = stage.GetPrimAtPath('/World/gaussian_splats')
  imageable = UsdGeom.Imageable(splats_prim)
  imageable.MakeInvisible()

  image_hd_invis = engine.render()['color']
  scene_invis = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_invis = np.array(mi.render(scene_invis, spp=128))

  test_helpers.robust_assert_close(
      image_hd_invis[..., :3], image_usd_invis, atol=0.05
  )
  assert np.max(image_hd_invis[..., :3]) < 0.01

  # Step 2: Make the particle field visible again
  imageable.MakeVisible()
  image_hd_vis = engine.render()['color']
  scene_vis = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_vis = np.array(mi.render(scene_vis, spp=128))

  test_helpers.robust_assert_close(
      image_hd_vis[..., :3], image_usd_vis, atol=0.05
  )
  np.testing.assert_allclose(image_hd_initial, image_hd_vis, atol=0.05)

  # Step 3: Set active = False
  splats_prim.SetActive(False)
  image_hd_inactive = engine.render()['color']
  scene_inactive = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_inactive = np.array(mi.render(scene_inactive, spp=128))

  test_helpers.robust_assert_close(
      image_hd_inactive[..., :3], image_usd_inactive, atol=0.05
  )
  assert np.max(image_hd_inactive[..., :3]) < 0.01
