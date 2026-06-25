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

"""Fuzz testing: absolute maximum multi-prim macro property permutation overload on USD stage."""

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


def test_absolute_maximum_fuzz_overload():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_absolute_maximum_fuzz_overload_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_absolute_maximum_fuzz_overload_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: absolute maximum fuzz overload modifying 30+ stage attributes
  # 1. Geometry points and transform
  mesh = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  points = np.array(mesh.GetPointsAttr().Get()) * 1.1
  mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
  xform = UsdGeom.Xformable.Get(stage, '/root/Cube')
  xform.AddRotateXYZOp(opSuffix='ovl_rot').Set((5.0, 10.0, 15.0))

  # 2. Material parameter grid
  shader = UsdShade.Shader.Get(
      stage, '/root/_materials/Material/Principled_BSDF'
  )
  shader.GetInput('diffuseColor').Set(Gf.Vec3f(0.8, 0.9, 0.2))
  shader.GetInput('roughness').Set(0.3)
  shader.GetInput('metallic').Set(0.7)
  shader.GetInput('clearcoat').Set(0.8)
  shader.GetInput('clearcoatRoughness').Set(0.15)
  shader.GetInput('specular').Set(0.4)

  # 3. Camera attributes and transform
  camera = UsdGeom.Camera.Get(stage, '/root/Camera/Camera')
  camera.GetFocalLengthAttr().Set(0.55)
  camera.GetHorizontalApertureOffsetAttr().Set(0.05)
  camera.GetVerticalApertureOffsetAttr().Set(-0.05)
  camera.GetClippingRangeAttr().Set(Gf.Vec2f(2.0, 50.0))
  cam_xform = UsdGeom.Xformable.Get(stage, '/root/Camera')
  cam_xform.AddTranslateOp(opSuffix='ovl_trans').Set((0.1, -0.1, 0.1))

  # 4. Light properties and new prim insertion
  sphere_light = UsdLux.SphereLight.Get(stage, '/root/Light/Light')
  sphere_light.GetIntensityAttr().Set(300.0)
  sphere_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.8, 0.8))

  extra_light = UsdLux.SphereLight.Define(stage, '/root/OverloadSphere')
  extra_light.CreateTreatAsPointAttr(True)
  extra_light.CreateIntensityAttr(150.0)
  extra_light.CreateColorAttr(Gf.Vec3f(0.2, 0.8, 0.8))
  extra_light.AddTranslateOp().Set((-3, 3, 3))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_absolute_maximum_fuzz_overload_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified,
      'test_absolute_maximum_fuzz_overload_modified_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
