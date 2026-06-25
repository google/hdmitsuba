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

"""Fuzz testing: consecutive RGB color wheel rotation modification across point lights on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_dynamic_light_color_rainbow():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  p1 = UsdLux.SphereLight.Define(stage, '/root/Rain1')
  p1.CreateTreatAsPointAttr(True)
  p1.CreateIntensityAttr(50.0)
  p1.CreateColorAttr(Gf.Vec3f(1, 0, 0))
  p1.AddTranslateOp().Set((3, 0, 3))

  p2 = UsdLux.SphereLight.Define(stage, '/root/Rain2')
  p2.CreateTreatAsPointAttr(True)
  p2.CreateIntensityAttr(50.0)
  p2.CreateColorAttr(Gf.Vec3f(0, 1, 0))
  p2.AddTranslateOp().Set((-3, 0, 3))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  # Mod 1: rainbow step 1
  p1.GetColorAttr().Set(Gf.Vec3f(1, 1, 0))
  p2.GetColorAttr().Set(Gf.Vec3f(0, 1, 1))

  image_hd_mod1 = engine.render()['color']
  scene_mod1 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod1 = np.array(mi.render(scene_mod1, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod1[..., :3], image_usd_mod1, atol=0.2
  )

  # Mod 2: rainbow step 2
  p1.GetColorAttr().Set(Gf.Vec3f(1, 0, 1))
  p2.GetColorAttr().Set(Gf.Vec3f(0, 0, 1))

  image_hd_mod2 = engine.render()['color']
  scene_mod2 = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_mod2 = np.array(mi.render(scene_mod2, spp=128))
  test_helpers.robust_assert_close(
      image_hd_mod2[..., :3], image_usd_mod2, atol=0.2
  )
