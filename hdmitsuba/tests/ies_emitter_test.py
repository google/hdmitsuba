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

"""Standalone tests for the Mitsuba `ies` emitter plugin."""

from __future__ import annotations

import math

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

# Importing `geometry_ext` loads libhdMitsuba, which registers the `ies` plugin.
from hdmitsuba import geometry_ext  # pylint: disable=unused-import
from hdmitsuba.tests import test_helpers


def _ies_path(name: str) -> str:
  """Returns the path of the test IES profile `name`."""
  return f'{test_helpers.TEST_ASSETS_PATH}/lights/textures/ies/{name}.ies'


def _si_at(theta_deg: float, phi_deg: float) -> mi.SurfaceInteraction3f:
  """Returns an interaction seen from `(theta_deg, phi_deg)` (IES frame)."""
  theta, phi = math.radians(theta_deg), math.radians(phi_deg)
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.sh_frame = mi.Frame3f(mi.Vector3f(0.0, 0.0, 1.0))
  si.wi = mi.Vector3f(
      -math.sin(theta) * math.cos(phi),
      math.sin(theta) * math.sin(phi),
      math.cos(theta),
  )
  return si


def _eval_ies_at(
    emitter: mi.Emitter, theta_deg: float, phi_deg: float
) -> np.ndarray:
  """Evaluates `emitter` at polar angle `theta_deg` and azimuth `phi_deg`."""
  return np.array(emitter.eval(_si_at(theta_deg, phi_deg))).ravel()


def _load_ies(filename: str) -> mi.Emitter:
  """Loads `filename` as an `ies` emitter wrapping a unit `area` emitter."""
  return mi.load_dict({
      'type': 'ies',
      'filename': filename,
      'nested_emitter': {'type': 'area'},
  })


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


@pytest.mark.parametrize('shape', ['disk', 'sphere', 'point'])
def test_uniform_ies_renders_like_plain_light(shape: str):
  """A uniform 1 cd profile over the full sphere doesn't change the render."""
  to_world = mi.ScalarTransform4f().translate([0, 0, 1]).rotate([1, 0, 0], 30)

  def render(with_ies: bool) -> np.ndarray:
    light = {'type': 'point', 'to_world': to_world}
    if shape != 'point':
      light = {'type': 'area'}
    if with_ies:
      light = {
          'type': 'ies',
          'filename': _ies_path('uniform_sphere'),
          'to_world': to_world,
          'nested_emitter': light,
      }
    if shape != 'point':
      light = {
          'type': shape,
          'to_world': to_world @ mi.ScalarTransform4f().scale(0.1),
          'emitter': light,
      }
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'direct'},
        'sensor': {
            'type': 'orthographic',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]
            ),
            'film': {'type': 'hdrfilm', 'width': 32, 'height': 32},
        },
        'floor': {'type': 'rectangle', 'bsdf': {'type': 'diffuse'}},
        'light': light,
    })
    return np.array(mi.render(scene, spp=16))

  reference = render(with_ies=False)
  assert reference.max() > 0.0
  np.testing.assert_allclose(
      render(with_ies=True), reference, rtol=1e-5, atol=1e-6
  )


# `(theta_deg, phi_deg, expected_cd)` samples of the valid test IES profiles.
_VALID_PROFILE_SAMPLES = {
    # Keyword header and non-uniform knots.
    'keyword_header': [(0, 0, 5526), (60, 0, 10.75)],
    # No header, with values wrapped over lines.
    'headerless': [(0, 0, 100), (45, 0, 50), (90, 0, 0)],
    # Duplicate knots, which keep steps sharp.
    'duplicate_knots': [(0, 0, 100), (30, 0, 100), (60, 0, 0)],
    # Candela multiplier 2, ballast factor 1.5, ballast-lamp factor 0.5,
    # and two lamps (ignored), after a `TILT=INCLUDE` block.
    'multipliers_tilt': [(0, 0, 150), (45, 0, 105), (90, 0, 60)],
    # Non-uniform vertical (`10 30 90`) and horizontal (`0 30 90 180`)
    # axes, and no emission below the first vertical knot.
    'non_uniform': [(5, 15, 0), (20, 60, 280), (60, 300, 105)],
    # Type C horizontal symmetry domains, at `theta = 45 deg` (half of the
    # `theta = 0` values) and `phi = 0, 90, 180, 270 deg`.
    **{
        name: [(45, 90 * i, 0.5 * v) for i, v in enumerate(values)]
        for name, values in (
            ('symmetry_0_90', [100, 300, 100, 300]),
            ('symmetry_0_180', [100, 200, 400, 200]),
            ('symmetry_90_270', [100, 200, 400, 200]),
            ('symmetry_0_360', [100, 200, 300, 400]),
            ('symmetry_no_360', [100, 200, 300, 400]),
        )
    },
    # Used by `ies.usda`: alternating 0 / 2000 cd rings every 4.5 deg.
    'step': [((i + 0.5) * 4.5, 0, 2000 * (i % 2)) for i in range(20)],
}


@pytest.mark.parametrize('name', _VALID_PROFILE_SAMPLES)
def test_ies_parser_valid_profiles(name: str):
  emitter = _load_ies(_ies_path(name))
  for theta_deg, phi_deg, expected in _VALID_PROFILE_SAMPLES[name]:
    np.testing.assert_allclose(
        _eval_ies_at(emitter, theta_deg, phi_deg),
        [expected] * 3,
        rtol=1e-5,
        err_msg=f'at ({theta_deg}, {phi_deg}) deg',
    )


@pytest.mark.parametrize(
    'filename',
    ['/nonexistent/profile.ies']
    + [
        _ies_path(f'invalid/{name}')
        for name in (
            'missing_tilt',
            'non_numeric',
            'not_type_c_symmetric',
            'single_vertical_angle',
            'truncated_header',
            'truncated_values',
            'type_b',
        )
    ],
    ids=lambda filename: filename.rsplit('/', 1)[-1],
)
def test_ies_parser_invalid_profiles(filename: str):
  """Missing, unsupported, and malformed IES files raise."""
  with pytest.raises(RuntimeError):
    _load_ies(filename)


def test_ies_profile_update_and_gradient():
  """`ies_profile` and the nested emitter can be updated and differentiated."""
  emitter = mi.load_dict({
      'type': 'ies',
      # 1 cd at `theta = 0, 90 deg` and `phi = 0, 90, 180, 270, 360 deg`.
      'filename': _ies_path('ones_2d'),
      'nested_emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
      },
  })
  params = mi.traverse(emitter)
  params['ies_profile'] = mi.TensorXf(
      [2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 2.0, 0.0], (5, 2)
  )
  params['nested_emitter.radiance.value'] = mi.Color3f(4.0, 2.0, 1.0)
  params.update()
  # theta = 60 deg interpolates 1/3 of the way from the 90 deg knot (0 cd).
  np.testing.assert_allclose(
      _eval_ies_at(emitter, 60.0, 90.0), [16.0 / 3, 8.0 / 3, 4.0 / 3], rtol=1e-5
  )
  np.testing.assert_allclose(
      _eval_ies_at(emitter, 60.0, 180.0), [8.0, 4.0, 2.0], rtol=1e-5
  )
  if not dr.is_diff_v(mi.Float):
    return
  dr.enable_grad(params['ies_profile'])
  params.update()
  dr.backward(dr.sum(emitter.eval(_si_at(60.0, 90.0))))
  # At (theta = 60 deg, phi = 90 deg -> knot row 1):
  # eval = (4 + 2 + 1) * (prof[1, 0] / 3 + 2 * prof[1, 1] / 3).
  expected_grad = np.zeros((5, 2))
  expected_grad[1] = [7.0 / 3, 14.0 / 3]
  np.testing.assert_allclose(
      dr.grad(params['ies_profile']), expected_grad, rtol=1e-5, atol=1e-6
  )


def test_ies_area_sampling_consistency():
  """Sampled weights, PDFs, and evaluations of an IES area light agree."""
  # Keep the shape alive, as the emitter only references it.
  shape = mi.load_dict({
      'type': 'rectangle',
      'to_world': mi.ScalarTransform4f().scale(0.5),
      'emitter': {
          'type': 'ies',
          'filename': _ies_path('ramp'),
          'nested_emitter': {
              'type': 'area',
              'radiance': {'type': 'rgb', 'value': [2.0, 4.0, 6.0]},
          },
      },
  })
  emitter = shape.emitter()
  it = dr.zeros(mi.Interaction3f)
  it.p = mi.Point3f(0.0, 0.0, 2.0)
  ds, weight = emitter.sample_direction(it, mi.Point2f(0.3, 0.7))
  pdf = emitter.pdf_direction(it, ds)
  assert ds.pdf[0] > 0.0
  np.testing.assert_allclose(pdf, ds.pdf, rtol=1e-5)
  np.testing.assert_allclose(
      emitter.eval_direction(it, ds) / pdf, weight, rtol=1e-5
  )


def test_ies_point_light():
  """An IES point light is a delta light modulated by the profile."""
  # Pointing down from `z = 2`.
  to_world = mi.ScalarTransform4f().translate([0, 0, 2]).rotate([1, 0, 0], 180)
  emitter = mi.load_dict({
      'type': 'ies',
      # 1 cd at `theta = 0` to 0.2 cd at `theta = 90 deg`.
      'filename': _ies_path('ramp'),
      'to_world': to_world,
      'nested_emitter': {
          'type': 'point',
          'intensity': {'type': 'rgb', 'value': [8.0, 4.0, 2.0]},
          'to_world': to_world,
      },
  })
  assert emitter.flags() == int(mi.EmitterFlags.DeltaPosition)

  # Straight below the light (`theta = 0`), at distance 2.
  it = dr.zeros(mi.Interaction3f)
  ds, weight = emitter.sample_direction(it, mi.Point2f(0.5, 0.5))
  assert ds.delta[0]
  np.testing.assert_allclose(weight, [[2.0], [1.0], [0.5]], rtol=1e-5)
  np.testing.assert_allclose(
      emitter.eval_direction(it, ds), [[2.0], [1.0], [0.5]], rtol=1e-5
  )
  assert emitter.pdf_direction(it, ds)[0] == 0.0

  # Rays into the lower (lit) and upper (unlit) hemispheres of the light.
  ray, ray_weight = emitter.sample_ray(
      0.0, 0.5, mi.Point2f(0.5, 0.5), mi.Point2f([0.8, 0.8], [0.7, 0.2])
  )
  theta = np.arccos(np.clip(-np.array(ray.d.z), -1.0, 1.0))
  profile = np.where(theta <= 0.5 * np.pi, 1.0 - 0.8 * theta / (0.5 * np.pi), 0)
  np.testing.assert_allclose(
      ray_weight,
      4.0 * np.pi * np.outer([8.0, 4.0, 2.0], profile),
      rtol=1e-5,
      atol=1e-6,
  )


def test_ies_sampling_weight_turns_light_off():
  """A zero `sampling_weight` turns the light off."""
  emitter = _load_ies(_ies_path('uniform_sphere'))
  np.testing.assert_allclose(_eval_ies_at(emitter, 0.0, 0.0), [1.0] * 3)
  params = mi.traverse(emitter)
  params['sampling_weight'] = 0.0
  params.update()
  np.testing.assert_allclose(_eval_ies_at(emitter, 0.0, 0.0), [0.0] * 3)


def test_ies_angle_scale_and_normalize():
  """`angle_scale` scales the profile's angles, `normalize` its power."""
  for angle_scale, theta_deg, expected in (
      # theta = 22.5 deg / 0.5 = 45 deg -> midpoint of [100, 0] = 50;
      # theta = 60 deg / 0.5 = 120 deg > 90 deg -> 0.
      (0.5, 22.5, 50.0),
      (0.5, 60.0, 0.0),
      (-2.0, 90.0, 0.0),
      (-1.0, 45.0, 50.0),
  ):
    scaled = mi.load_dict({
        'type': 'ies',
        # 100 cd at `theta = 0` to 0 cd at `theta = 90 deg`.
        'filename': _ies_path('linear'),
        'angle_scale': angle_scale,
        'nested_emitter': {'type': 'area'},
    })
    np.testing.assert_allclose(
        _eval_ies_at(scaled, theta_deg, 0.0),
        [expected] * 3,
        rtol=1e-5,
        atol=1e-6,
    )

  # `normalize = True` on a constant 200 cd hemispherical profile:
  # single vertical interval [0, pi/2] has midpoint pi/4, so
  # power = 2 pi * (pi / 2) * sin(pi / 4) * 200 / (2 pi).
  norm_emitter = mi.load_dict({
      'type': 'ies',
      'filename': _ies_path('const'),
      'normalize': True,
      'nested_emitter': {'type': 'area'},
  })
  expected_power = 200.0 * (0.5 * np.pi) * np.sin(0.25 * np.pi)
  np.testing.assert_allclose(
      _eval_ies_at(norm_emitter, 30.0, 0.0),
      [200.0 / expected_power] * 3,
      rtol=1e-5,
  )
