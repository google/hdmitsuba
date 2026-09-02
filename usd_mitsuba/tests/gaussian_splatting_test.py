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

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdVol
from pxr import Vt

from usd_mitsuba import gaussian_splatting
from usd_mitsuba import translator
from usd_mitsuba import util


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('llvm_ad_rgb')


def test_convert_gaussian_splats():
  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim('/World/Splat', 'ParticleField3DGaussianSplat')
  usd_splat = UsdVol.ParticleField3DGaussianSplat(prim)

  positions = Vt.Vec3fArray([Gf.Vec3f(1.0, 2.0, 3.0), Gf.Vec3f(4.0, 5.0, 6.0)])
  scales = Vt.Vec3fArray([Gf.Vec3f(0.1, 0.2, 0.3), Gf.Vec3f(0.4, 0.5, 0.6)])
  orientations = Vt.QuatfArray([Gf.Quatf(1.0, 0, 0, 0), Gf.Quatf(0.7071, 0, 0.7071, 0)])
  opacities = Vt.FloatArray([0.8, 0.9])
  sh_coeffs = Vt.Vec3fArray([Gf.Vec3f(1.0, 0.5, 0.2), Gf.Vec3f(0.1, 0.2, 0.3)])

  usd_splat.GetPositionsAttr().Set(positions)
  usd_splat.GetScalesAttr().Set(scales)
  usd_splat.GetOrientationsAttr().Set(orientations)
  usd_splat.GetOpacitiesAttr().Set(opacities)
  usd_splat.GetRadianceSphericalHarmonicsDegreeAttr().Set(0)
  usd_splat.GetRadianceSphericalHarmonicsCoefficientsAttr().Set(sh_coeffs)

  splat_id = util.get_mitsuba_id(prim)
  res = gaussian_splatting.convert_gaussian_splats(prim, Usd.TimeCode.Default())
  assert splat_id in res
  splats_dict = res[splat_id]
  assert splats_dict['type'] == 'ellipsoidsmesh'
  assert np.allclose(np.array(splats_dict['centers']), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  assert np.allclose(np.array(splats_dict['scales']), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
  assert np.allclose(np.array(splats_dict['opacities']), [[0.8], [0.9]])
  assert splats_dict['sh_coeffs'].shape == (2, 3)

  # Test full stage conversion
  scene_dict = translator.convert_to_mitsuba(stage)
  assert splat_id in scene_dict
