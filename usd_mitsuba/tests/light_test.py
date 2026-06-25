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

"""Tests for light translation."""

from __future__ import annotations

import os

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux

from usd_mitsuba import light

_EXR_PATH = os.path.abspath(
    'hdmitsuba/test_assets/lights/textures/envmap.png'
)
_IES_PATH = os.path.abspath(
    'hdmitsuba/test_assets/lights/textures/step.ies'
)


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_stage_with_light(
    light_type: type[UsdLux.BoundableLightBase]
    | type[UsdLux.NonboundableLightBase],
    name: str = 'light',
) -> tuple[Usd.Stage, Usd.Prim]:
  stage = Usd.Stage.CreateInMemory()
  light_prim = light_type.Define(stage, Sdf.Path(f'/{name}'))
  return stage, light_prim.GetPrim()


def test_distant_light():
  stage, prim = _create_stage_with_light(UsdLux.DistantLight)
  usd_light = UsdLux.DistantLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.5, 0.25))
  usd_light.GetIntensityAttr().Set(2.0)
  usd_light.GetExposureAttr().Set(2.0)  # 2.0 * 2^2 = 8.0 intensity
  transform_api = UsdGeom.Xformable(prim)
  transform_api.AddRotateXOp().Set(90.0)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'directional'
  np.testing.assert_allclose(
      emitter_dict['irradiance']['value'],
      mi.ScalarColor3f(8.0, 4.0, 2.0),
      atol=1e-6,
  )

  # Check transform: USD default direction is -Z, while Mitsuba directional
  # emitter points along +Z. The translation rotates by 180 degrees around X.
  # With the added 90 degrees rotation around X in USD, the final direction in
  # Mitsuba should point along +Y.
  to_world = emitter_dict['to_world']
  direction = to_world @ mi.ScalarVector3f(0, 0, 1)
  np.testing.assert_allclose(direction, mi.ScalarVector3f(0, 1, 0), atol=1e-6)


def test_dome_light_constant():
  stage, prim = _create_stage_with_light(UsdLux.DomeLight)
  usd_light = UsdLux.DomeLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(0.5, 0.5, 0.5))
  usd_light.GetIntensityAttr().Set(1.5)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'constant'
  np.testing.assert_allclose(
      emitter_dict['radiance']['value'], mi.ScalarColor3f(0.75, 0.75, 0.75), atol=1e-6
  )


def test_dome_light_textured():
  stage, prim = _create_stage_with_light(UsdLux.DomeLight)
  usd_light = UsdLux.DomeLight(prim)
  usd_light.GetTextureFileAttr().Set(Sdf.AssetPath(_EXR_PATH))
  usd_light.GetIntensityAttr().Set(2.5)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'envmap'
  assert emitter_dict['filename'] == _EXR_PATH
  assert emitter_dict['scale'] == 2.5


def test_sphere_light_point():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  usd_light = UsdLux.SphereLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
  usd_light.GetIntensityAttr().Set(10.0)
  usd_light.GetRadiusAttr().Set(0.0)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'point'
  np.testing.assert_allclose(
      emitter_dict['intensity']['value'], mi.ScalarColor3f(10.0, 10.0, 10.0)
  )


def test_sphere_light_point_normalized():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  usd_light = UsdLux.SphereLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
  usd_light.GetIntensityAttr().Set(10.0)
  usd_light.GetRadiusAttr().Set(0.0)
  usd_light.GetNormalizeAttr().Set(True)  # Should multiply by 0.25 for point

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'point'
  np.testing.assert_allclose(
      emitter_dict['intensity']['value'], mi.ScalarColor3f(2.5, 2.5, 2.5)
  )


def test_sphere_light_spot():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  usd_light = UsdLux.SphereLight(prim)
  usd_light.GetRadiusAttr().Set(0.0)
  usd_light.GetIntensityAttr().Set(5.0)
  shaping_api = UsdLux.ShapingAPI.Apply(prim)
  shaping_api.GetShapingConeAngleAttr().Set(45.0)
  shaping_api.GetShapingConeSoftnessAttr().Set(0.1)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'spot'
  np.testing.assert_allclose(
      emitter_dict['intensity']['value'], mi.ScalarColor3f(5.0, 5.0, 5.0)
  )
  assert emitter_dict['cutoff_angle'] == 45.0
  # beam_width = cone_angle * (1 - softness) = 45 * 0.9 = 40.5
  assert abs(emitter_dict['beam_width'] - 40.5) < 1e-6


def test_sphere_light_area():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  usd_light = UsdLux.SphereLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
  usd_light.GetIntensityAttr().Set(5.0)
  usd_light.GetRadiusAttr().Set(2.0)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'sphere'
  # Radius should scale the to_world transform
  scale = emitter_dict['to_world'] @ mi.ScalarVector3f(1, 0, 0)
  assert abs(dr.norm(scale) - 2.0) < 1e-6

  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'], mi.ScalarColor3f(
          5.0, 0.0, 0.0)
  )


def test_sphere_light_area_normalized():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  usd_light = UsdLux.SphereLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
  usd_light.GetIntensityAttr().Set(5.0)
  usd_light.GetRadiusAttr().Set(2.0)
  usd_light.GetNormalizeAttr().Set(True)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'sphere'
  # Radiance should be divided by 4 * pi * radius^2
  # 5.0 / (4 * pi * 4) = 5.0 / (16 * pi) = 5.0 / 50.26548 = 0.09947
  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'],
      mi.ScalarColor3f(5.0 / (4 * dr.pi * 4), 0.0, 0.0),
      atol=1e-5,
  )


def test_rect_light():
  stage, prim = _create_stage_with_light(UsdLux.RectLight)
  usd_light = UsdLux.RectLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(0.0, 1.0, 0.0))
  usd_light.GetIntensityAttr().Set(3.0)
  usd_light.GetWidthAttr().Set(2.0)
  usd_light.GetHeightAttr().Set(4.0)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'rectangle'
  # Width/Height should scale the to_world transform (by half width/height)
  to_world = emitter_dict['to_world']
  sx = dr.norm(to_world @ mi.ScalarVector3f(1, 0, 0))
  sy = dr.norm(to_world @ mi.ScalarVector3f(0, 1, 0))
  assert abs(sx - 1.0) < 1e-6  # 0.5 * 2.0
  assert abs(sy - 2.0) < 1e-6  # 0.5 * 4.0

  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'], mi.ScalarColor3f(
          0.0, 3.0, 0.0)
  )


def test_rect_light_normalized():
  stage, prim = _create_stage_with_light(UsdLux.RectLight)
  usd_light = UsdLux.RectLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(0.0, 1.0, 0.0))
  usd_light.GetIntensityAttr().Set(3.0)
  usd_light.GetWidthAttr().Set(2.0)
  usd_light.GetHeightAttr().Set(4.0)
  usd_light.GetNormalizeAttr().Set(True)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'rectangle'
  # Area of 2x4 rectangle is 8.
  # Radiance = 3.0 / 8.0 = 0.375
  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'],
      mi.ScalarColor3f(0.0, 0.375, 0.0),
  )


def test_disk_light():
  stage, prim = _create_stage_with_light(UsdLux.DiskLight)
  usd_light = UsdLux.DiskLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
  usd_light.GetIntensityAttr().Set(4.0)
  usd_light.GetRadiusAttr().Set(3.0)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'disk'
  to_world = emitter_dict['to_world']
  sx = dr.norm(to_world @ mi.ScalarVector3f(1, 0, 0))
  assert abs(sx - 3.0) < 1e-6

  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'], mi.ScalarColor3f(
          0.0, 0.0, 4.0)
  )


def test_disk_light_normalized():
  stage, prim = _create_stage_with_light(UsdLux.DiskLight)
  usd_light = UsdLux.DiskLight(prim)
  usd_light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
  usd_light.GetIntensityAttr().Set(4.0)
  usd_light.GetRadiusAttr().Set(3.0)
  usd_light.GetNormalizeAttr().Set(True)

  emitter_dict = light.convert_light(prim)

  assert emitter_dict['type'] == 'disk'
  # Area of radius 3 disk is pi * 9.
  # Radiance = 4.0 / (9 * pi) = 0.14147
  np.testing.assert_allclose(
      emitter_dict['emitter']['radiance']['value'],
      mi.ScalarColor3f(0.0, 0.0, 4.0 / (9 * dr.pi)),
  )


def test_ies_profile_error():
  stage, prim = _create_stage_with_light(UsdLux.SphereLight)
  shaping_api = UsdLux.ShapingAPI.Apply(prim)
  shaping_api.GetShapingIesFileAttr().Set(Sdf.AssetPath(_IES_PATH))

  with pytest.raises(ValueError, match='IES files are not yet supported.'):
    light.convert_light(prim)
