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

"""Fuzz testing: three-point shaping cone convergence modification on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_multilight_shaping_convergence():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  s1 = UsdLux.SphereLight.Define(stage, '/root/Conv1')
  s1.CreateIntensityAttr(50.0)
  s1.AddTranslateOp().Set((2, 0, 3))
  s1.AddRotateXYZOp().Set((0, -30, 0))
  shaping1 = UsdLux.ShapingAPI.Apply(s1.GetPrim())
  shaping1.CreateShapingConeAngleAttr(15.0)

  s2 = UsdLux.SphereLight.Define(stage, '/root/Conv2')
  s2.CreateIntensityAttr(50.0)
  s2.AddTranslateOp().Set((-2, 0, 3))
  s2.AddRotateXYZOp().Set((0, 30, 0))
  shaping2 = UsdLux.ShapingAPI.Apply(s2.GetPrim())
  shaping2.CreateShapingConeAngleAttr(15.0)

  s3 = UsdLux.SphereLight.Define(stage, '/root/Conv3')
  s3.CreateIntensityAttr(50.0)
  s3.AddTranslateOp().Set((0, 2, 3))
  s3.AddRotateXYZOp().Set((30, 0, 0))
  shaping3 = UsdLux.ShapingAPI.Apply(s3.GetPrim())
  shaping3.CreateShapingConeAngleAttr(15.0)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_multilight_shaping_convergence_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_multilight_shaping_convergence_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Shift intensities and cone angles
  shaping1.GetShapingConeAngleAttr().Set(30.0)
  shaping2.GetShapingConeAngleAttr().Set(30.0)
  shaping3.GetShapingConeAngleAttr().Set(30.0)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_multilight_shaping_convergence_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified,
      'test_multilight_shaping_convergence_modified_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
