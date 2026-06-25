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

"""Fuzz testing: multi-chroma color spectrum inversion across area lights in USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_multilight_compound_chroma_shift():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  light_r = UsdLux.SphereLight.Define(stage, '/root/LightRed')
  light_r.CreateTreatAsPointAttr(False)
  light_r.CreateRadiusAttr(0.5)
  light_r.CreateColorAttr(Gf.Vec3f(1, 0, 0))
  light_r.AddTranslateOp().Set((2, 0, 2))

  light_g = UsdLux.SphereLight.Define(stage, '/root/LightGreen')
  light_g.CreateTreatAsPointAttr(False)
  light_g.CreateRadiusAttr(0.5)
  light_g.CreateColorAttr(Gf.Vec3f(0, 1, 0))
  light_g.AddTranslateOp().Set((-2, 0, 2))

  light_b = UsdLux.SphereLight.Define(stage, '/root/LightBlue')
  light_b.CreateTreatAsPointAttr(False)
  light_b.CreateRadiusAttr(0.5)
  light_b.CreateColorAttr(Gf.Vec3f(0, 0, 1))
  light_b.AddTranslateOp().Set((0, 2, 2))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_multilight_compound_chroma_shift_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial,
      'test_multilight_compound_chroma_shift_initial_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Shift colors across spectrum
  light_r.GetColorAttr().Set(Gf.Vec3f(0, 1, 1))
  light_g.GetColorAttr().Set(Gf.Vec3f(1, 0, 1))
  light_b.GetColorAttr().Set(Gf.Vec3f(1, 1, 0))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified,
      'test_multilight_compound_chroma_shift_modified_hd.png',
  )
  test_helpers.write_image(
      image_usd_modified,
      'test_multilight_compound_chroma_shift_modified_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
