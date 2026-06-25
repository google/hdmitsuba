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

"""Fuzz testing: dual frame intensity oscillation modification on Environment Map in USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_envmap_dome_intensity_sawtooth():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/envmap.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  env_light = UsdLux.DomeLight.Get(stage, '/root/env_light')

  # Mod 1: High sawtooth peak
  env_light.GetIntensityAttr().Set(10.0)

  image_hd_mod1 = engine.render()['color']
  scene_mod1 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod1 = np.array(mi.render(scene_mod1, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod1[..., :3], image_usd_mod1, atol=0.2
  )

  # Mod 2: Low sawtooth valley
  env_light.GetIntensityAttr().Set(0.1)

  image_hd_mod2 = engine.render()['color']
  scene_mod2 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod2 = np.array(mi.render(scene_mod2, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod2[..., :3], image_usd_mod2, atol=0.2
  )
