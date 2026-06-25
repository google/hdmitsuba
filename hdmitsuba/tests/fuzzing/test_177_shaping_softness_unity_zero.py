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

"""Fuzz testing: anti-parallel softness extreme boundaries modification across co-located spot lights on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdLux  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_shaping_softness_unity_zero():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/lights/spot.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  orig_spot = UsdLux.SphereLight.Get(stage, '/root/Area/Area')
  spot2 = UsdLux.SphereLight.Define(stage, '/root/FuzzSpotColocated')
  spot2.CreateTreatAsPointAttr(True)
  spot2.CreateIntensityAttr(50.0)
  spot2.AddTranslateOp().Set((0, 0, 3))
  spot2.AddRotateXYZOp().Set((0, 45, 0))
  xform_prim = UsdGeom.Xformable(spot2.GetPrim())
  xform_prim.SetXformOpOrder([
      UsdGeom.XformOp(spot2.GetPrim().GetAttribute('xformOp:translate')),
      UsdGeom.XformOp(spot2.GetPrim().GetAttribute('xformOp:rotateXYZ')),
  ])
  shaping2 = UsdLux.ShapingAPI.Apply(spot2.GetPrim())
  shaping2.CreateShapingConeAngleAttr(45.0)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_shaping_softness_unity_zero_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_shaping_softness_unity_zero_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: co-located softness 0 and 1
  shaping1 = UsdLux.ShapingAPI(orig_spot.GetPrim())
  shaping1.GetShapingConeSoftnessAttr().Set(0.0)
  shaping2.GetShapingConeSoftnessAttr().Set(1.0)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_shaping_softness_unity_zero_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_shaping_softness_unity_zero_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
