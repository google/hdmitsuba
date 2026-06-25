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

"""Fuzz testing: mega compound stress test across USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdLux  # type: ignore
from pxr import UsdShade  # type: ignore
from pxr import Vt  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_ultra_compound_stress_fuzz():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  # Add extra light and material setup
  dome_light = UsdLux.DomeLight.Define(stage, '/root/FuzzDome')
  dome_light.CreateIntensityAttr(0.2)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_ultra_compound_stress_fuzz_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_ultra_compound_stress_fuzz_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: ultra compound stress mutations
  # 1. Cube transform
  cube_xform = UsdGeom.Xformable.Get(stage, '/root/Cube')
  cube_xform.AddTranslateOp(opSuffix='stress_trans').Set((0.1, 0.1, -0.1))
  cube_xform.AddScaleOp(opSuffix='stress_scale').Set((1.1, 0.9, 1.1))

  # 2. Mesh points
  mesh = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  points = np.array(mesh.GetPointsAttr().Get()) * 1.05
  mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

  # 3. Material attributes
  shader = UsdShade.Shader.Get(
      stage, '/root/_materials/Material/Principled_BSDF'
  )
  shader.GetInput('diffuseColor').Set(Gf.Vec3f(0.8, 0.2, 0.8))
  shader.GetInput('roughness').Set(0.15)
  shader.GetInput('metallic').Set(0.8)

  # 4. Multiple light attributes
  sphere_light = UsdLux.SphereLight.Get(stage, '/root/Light/Light')
  sphere_light.GetIntensityAttr().Set(450.0)
  sphere_light.GetColorAttr().Set(Gf.Vec3f(0.8, 1.0, 0.8))
  dome_light.GetIntensityAttr().Set(0.4)

  # 5. Camera focal length and clipping
  camera = UsdGeom.Camera.Get(stage, '/root/Camera/Camera')
  camera.GetFocalLengthAttr().Set(0.45)
  camera.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, 50.0))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_ultra_compound_stress_fuzz_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_ultra_compound_stress_fuzz_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
