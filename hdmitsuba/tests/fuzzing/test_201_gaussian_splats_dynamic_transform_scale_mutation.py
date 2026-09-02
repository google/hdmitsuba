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

"""Fuzz testing: dynamic transform and primvar (scale, SH, opacity) mutation on Gaussian Splats in USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdVol
from pxr import Vt
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_gaussian_splats_dynamic_transform_primvar_mutation():
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

  # Modification 1: Mutate the particle field transform (translate, rotate,
  # uniform scale).
  # The rotation exercises the quaternion composition path; the scale stays
  # uniform since non-uniform transform scales are unsupported (warn-only).
  splats_xform = UsdGeom.Xformable.Get(stage, '/World/gaussian_splats')
  splats_xform.AddTranslateOp(opSuffix='fuzz_trans').Set(Gf.Vec3d(0.2, -0.3, 0.1))
  splats_xform.AddRotateXYZOp(opSuffix='fuzz_rot').Set(Gf.Vec3f(25.0, -40.0, 10.0))
  splats_xform.AddScaleOp(opSuffix='fuzz_scale').Set(Gf.Vec3f(1.3, 1.3, 1.3))

  # Modification 2: Mutate individual particle scales primvar
  splats_prim = UsdVol.ParticleField3DGaussianSplat.Get(
      stage, '/World/gaussian_splats'
  )
  scales = np.array(splats_prim.GetScalesAttr().Get()) * 1.3
  splats_prim.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales))

  # Modification 3: Invert spherical harmonics colors (R <-> B channels)
  sh_attr = splats_prim.GetRadianceSphericalHarmonicsCoefficientsAttr()
  sh_coeffs = np.array(sh_attr.Get())
  sh_coeffs[..., 0], sh_coeffs[..., 2] = (
      sh_coeffs[..., 2].copy(),
      sh_coeffs[..., 0].copy(),
  )
  sh_attr.Set(Vt.Vec3fArray.FromNumpy(sh_coeffs))

  # Modification 4: Change opacities to partial transparency
  opacities = np.full(len(sh_coeffs), 0.5, dtype=np.float32)
  splats_prim.GetOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(opacities))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))

  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.05
  )
  assert np.mean(np.abs(image_hd_modified[..., :3] - image_hd_initial[..., :3])) > 0.01
