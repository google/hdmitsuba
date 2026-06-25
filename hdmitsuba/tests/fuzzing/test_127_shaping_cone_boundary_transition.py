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

"""Fuzz testing: dynamic boundary state transition between spot and point light on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_shaping_cone_boundary_transition():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/spot.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_shaping_cone_boundary_transition_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial,
      'test_shaping_cone_boundary_transition_initial_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification 1: Set cone angle to 0 (transition to point light)
  spot_light = UsdLux.SphereLight.Get(stage, '/root/Area/Area')
  shaping = UsdLux.ShapingAPI(spot_light.GetPrim())
  shaping.GetShapingConeAngleAttr().Set(0.0)

  image_hd_mod1 = engine.render()['color']
  scene_mod1 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod1 = np.array(mi.render(scene_mod1, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod1[..., :3], image_usd_mod1, atol=0.2
  )

  # Modification 2: Set cone angle back to 45.0 (transition back to spot light)
  shaping.GetShapingConeAngleAttr().Set(45.0)

  image_hd_mod2 = engine.render()['color']
  scene_mod2 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod2 = np.array(mi.render(scene_mod2, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod2[..., :3], image_usd_mod2, atol=0.2
  )
