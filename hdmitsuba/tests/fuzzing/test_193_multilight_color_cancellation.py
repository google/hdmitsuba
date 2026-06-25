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

"""Fuzz testing: complimentary color cancellation modification across point lights on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_multilight_color_cancellation():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  l1 = UsdLux.SphereLight.Define(stage, '/root/CancelYellow')
  l1.CreateTreatAsPointAttr(True)
  l1.CreateIntensityAttr(50.0)
  l1.CreateColorAttr(Gf.Vec3f(1, 1, 0))
  l1.AddTranslateOp().Set((2, 2, 2))

  l2 = UsdLux.SphereLight.Define(stage, '/root/CancelCyan')
  l2.CreateTreatAsPointAttr(True)
  l2.CreateIntensityAttr(50.0)
  l2.CreateColorAttr(Gf.Vec3f(0, 1, 1))
  l2.AddTranslateOp().Set((-2, -2, 2))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_multilight_color_cancellation_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_multilight_color_cancellation_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Invert colors to Magenta and White
  l1.GetColorAttr().Set(Gf.Vec3f(1, 0, 1))
  l2.GetColorAttr().Set(Gf.Vec3f(1, 1, 1))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_multilight_color_cancellation_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified,
      'test_multilight_color_cancellation_modified_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
