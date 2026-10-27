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

import pathlib
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from hdmitsuba.tests import test_helpers
from usd_mitsuba import util

_IES_RESOLUTION = 256
_IES_LIGHT_HEIGHT = 1.0
_IES_LIGHT_SIZE = 0.005


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  util.ensure_plugin_registration()


@pytest.mark.parametrize('profile', ['uniform', 'step', 'azimuthal'])
@pytest.mark.parametrize(
    'shape', ['disk', 'rectangle', 'sphere', 'cylinder', 'cube', 'point']
)
def test_ies_emitter_render(tmp_path, shape: str, profile: str):
  """Renders a small IES light above a floor and checks the cast pattern.

  The light points down (-Z) and is rotated about Z so that azimuthal frame
  errors are visible. The floor radiance must equal the radiance cast by the
  same light without a profile, times the IES weight in the emission direction.

  Args:
    tmp_path: Temporary directory for the IES files.
    shape: Parent shape of the `ies` emitter, or `point` for a point light.
    profile: `uniform` (must match the plain light exactly), `step`
      (rotationally symmetric, alternating 0 / 2000 cd rings every 4.5 deg,
      which exposes theta errors) or `azimuthal` (varies linearly in phi and
      theta, which exposes rotation and handedness errors).
  """
  to_world = (
      mi.ScalarTransform4f()
      .translate([0.0, 0.0, _IES_LIGHT_HEIGHT])
      .rotate([0.0, 0.0, 1.0], 30.0)
      .rotate([1.0, 0.0, 0.0], 180.0)
  )
  if profile == 'uniform':
    filename = test_helpers.write_ies_file(
        tmp_path / 'uniform.ies', [0.0, 90.0], [1.0, 1.0]
    )
  elif profile == 'step':
    filename = f'{test_helpers.TEST_ASSETS_PATH}/lights/textures/step.ies'
  else:
    # Linear in theta over [0, 90] deg and in phi.
    filename = test_helpers.write_ies_file(
        tmp_path / 'azimuthal.ies',
        [0.0, 90.0],
        np.outer([1.0, 2.0, 3.0, 4.0, 1.0], [1.0, 0.0]),
        [0.0, 90.0, 180.0, 270.0, 360.0],
    )

  image, pos = _render_ies_scene(
      shape, to_world, {'type': 'ies', 'filename': filename}
  )
  reference, _ = _render_ies_scene(shape, to_world, None)
  # Ignore pixels that see the light itself rather than the floor.
  floor = pos[..., 2] < 1e-3
  image[~floor] = reference[~floor] = 0.0
  test_helpers.write_image(
      image / image.max(), f'test_ies_emitter_render_{shape}_{profile}.png'
  )
  if profile == 'uniform':
    # Same samples and a unit profile: identical up to float rounding.
    np.testing.assert_allclose(
        image, reference, rtol=1e-4, atol=1e-6 * reference.max()
    )
    return

  # Emission direction from the light to each floor point in the light frame.
  frame = np.array(to_world.matrix)[:3, :3]
  local = (pos - [0.0, 0.0, _IES_LIGHT_HEIGHT]) @ frame
  cos_theta = local[..., 2] / np.linalg.norm(local, axis=-1)
  theta = np.degrees(np.arccos(cos_theta))
  phi = np.degrees(np.arctan2(local[..., 1], -local[..., 0])) % 360.0
  if profile == 'step':
    band = 20.0 * theta / 90.0
    weight = 2000.0 * (np.floor(band) % 2)
    # Ignore ring boundaries, which are blurred by the light and pixel extent.
    valid = np.abs(band % 1.0 - 0.5) < 0.3
  else:
    weight = (1.0 - theta / 90.0) * np.interp(
        phi, [0.0, 90.0, 180.0, 270.0, 360.0], [1, 2, 3, 4, 1]
    )
    valid = np.ones_like(weight, dtype=bool)
  valid &= reference > 1e-2 * reference.max()

  expected = reference * weight
  test_helpers.robust_assert_close(
      image[valid], expected[valid], atol=1e-3 * expected.max(), rtol=0.05
  )


def test_ies_parser_valid_and_invalid_profiles(tmp_path):
  """Tests IES profile parsing, symmetry expansion, non-uniform axes, and errors."""
  # 1. Non-uniform 1D profile in meters (`0 10 15 22 ...`), headerless wrapped,
  # single-knot constant profile, and duplicate-knot step profiles.
  valid_1d_cases = [
      (
          (
              'IESNA:LM-63-2002\n[TEST] 2345123\nTILT=NONE\n'
              '1 -1 1 10 1 1 2 -0.29 -0.29 0\n1 1 40.0\n'
              '0 10 15 22 30 40 50 60 70 90\n0\n'
              '5526 5506 87 48 28 18 13 10.75 10 0\n'
          ),
          [(0.0, 5526.0), (60.0, 10.75)],
      ),
      (
          (
              '\nTILT=NONE\n1 1000 1 3 1 1 2 -0.1 -0.1 0\n1 1 0\n0 45\n90\n0\n'
              '100 50\n0\n'
          ),
          [(0.0, 100.0), (45.0, 50.0), (90.0, 0.0)],
      ),
      (
          (
              'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 1 1 1 2 0.1 0.1 0\n1 1 0\n'
              '90\n0\n125\n'
          ),
          [(0.0, 125.0), (120.0, 0.0)],
      ),
      (
          (
              'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 5 1 1 2 0.1 0.1 0\n1 1 0\n'
              '0 0 45 45 90\n0\n100 100 100 0 0\n'
          ),
          [(0.0, 100.0), (30.0, 100.0), (60.0, 0.0)],
      ),
  ]
  for idx, (ies_text, samples) in enumerate(valid_1d_cases):
    emitter = _load_ies_text(tmp_path / f'valid_1d_{idx}.ies', ies_text)
    for theta_deg, expected in samples:
      np.testing.assert_allclose(
          _eval_ies_at(emitter, theta_deg, 0.0),
          [expected] * 3,
          rtol=1e-4,
          atol=1e-6,
      )

  # 2. All 20 alternating concentric rings of `step.ies` (0 cd / 2000 cd).
  step_emitter = mi.load_dict({
      'type': 'ies',
      'filename': f'{test_helpers.TEST_ASSETS_PATH}/lights/textures/step.ies',
      'nested_emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': [0.5, 1.0, 2.0]},
      },
  })
  for band in range(20):
    theta_deg = (band + 0.5) * (90.0 / 20.0)
    expected_cd = 2000.0 if (band % 2 == 1) else 0.0
    np.testing.assert_allclose(
        _eval_ies_at(step_emitter, theta_deg, 0.0),
        [0.5 * expected_cd, 1.0 * expected_cd, 2.0 * expected_cd],
        atol=1e-2,
    )

  # 3. All 5 Type C 2D horizontal symmetry domains (`[0, 90]`, `[0, 180]`,
  # `[90, 270]`, `[0, 360]`, and full circle omitting `360 deg`).
  symmetry_cases = [
      (
          '1 -1 1 2 2 1 2 0.2 0.2 0\n1 1 40.0\n0 90\n0 90\n100 0\n300 0\n',
          [100.0, 300.0, 100.0, 300.0],
      ),
      (
          (
              '1 -1 1 2 3 1 2 0.2 0.2 0\n1 1 40.0\n0 90\n0 90 180\n'
              '100 0\n200 0\n400 0\n'
          ),
          [100.0, 200.0, 400.0, 200.0],
      ),
      (
          (
              '1 -1 1 2 3 1 2 0.2 0.2 0\n1 1 40.0\n0 90\n90 180 270\n'
              '100 0\n200 0\n400 0\n'
          ),
          [100.0, 200.0, 400.0, 200.0],
      ),
      (
          (
              '1 -1 1 2 5 1 2 0.2 0.2 0\n1 1 40.0\n0 90\n0 90 180 270 360\n'
              '100 0\n200 0\n300 0\n400 0\n100 0\n'
          ),
          [100.0, 200.0, 300.0, 400.0],
      ),
      (
          (
              '1 -1 1 2 4 1 2 0.2 0.2 0\n1 1 40.0\n0 90\n0 90 180 270\n'
              '100 0\n200 0\n300 0\n400 0\n'
          ),
          [100.0, 200.0, 300.0, 400.0],
      ),
  ]
  for idx, (body, expected_phi) in enumerate(symmetry_cases):
    sym_emitter = _load_ies_text(
        tmp_path / f'sym_{idx}.ies', f'IESNA:LM-63-2002\nTILT=NONE\n{body}'
    )
    for phi_deg, expected in zip(
        (0.0, 90.0, 180.0, 270.0), expected_phi, strict=True
    ):
      np.testing.assert_allclose(
          _eval_ies_at(sym_emitter, 45.0, phi_deg),
          [0.5 * expected] * 3,
          rtol=1e-4,
      )

  # 4. Non-uniform vertical (`10 30 90`) and horizontal (`0 30 90 180`) axes
  # and endpoint clamping (`theta = 5 < 10`).
  nu_emitter = _load_ies_text(
      tmp_path / 'non_uniform.ies',
      'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 3 4 1 2 0.2 0.2 0\n1 1 40.0\n'
      '10 30 90\n0 30 90 180\n100 60 0\n200 120 0\n500 300 0\n900 540 0\n',
  )
  for theta_deg, phi_deg, expected in (
      (5.0, 15.0, 150.0),
      (20.0, 60.0, 280.0),
      (60.0, 300.0, 105.0),
  ):
    np.testing.assert_allclose(
        _eval_ies_at(nu_emitter, theta_deg, phi_deg), [expected] * 3, rtol=1e-4
    )

  # 5. Invalid/unsupported IES files and nonexistent paths must raise.
  for bad_path in ('', '/nonexistent/path/to/profile.ies'):
    with pytest.raises(RuntimeError):
      mi.load_dict({
          'type': 'ies',
          'filename': bad_path,
          'nested_emitter': {'type': 'area'},
      })

  invalid_profiles = [
      # Missing TILT=NONE.
      (
          'IESNA:LM-63-2002\n1 -1 1 2 2 1 1 -0.29 -0.29 0\n1 1 40.0\n0 90\n0'
          ' 85\n5526 5506\n5526 5503\n'
      ),
      # Unsupported TILT=INCLUDE.
      (
          'IESNA:LM-63-2002\nTILT=INCLUDE\n1 -1 1 2 1 1 1 -0.29 -0.29 0\n'
          '1 1 40.0\n0 90\n0\n5526 0\n'
      ),
      # Unsupported num_lamps != 1.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n2 -1 1 2 1 1 1 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0\n5526 0\n'
      ),
      # Unsupported candela_multiplier != 1.0.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 2.0 2 1 1 1 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0\n5526 5506\n'
      ),
      # Unsupported photometric_type != 1 (Type B = 2).
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 1 2 1 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0\n5526 0\n'
      ),
      # Truncated header.
      'IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 2 1\n',
      # Decreasing vertical angles.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 3 1 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 60 30\n0\n100 50 0\n'
      ),
      # Vertical angles outside [0, 180].
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 1 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n-10 90\n0\n100 0\n'
      ),
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 1 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 190\n0\n100 0\n'
      ),
      # Non-increasing horizontal angles.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 3 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0 90 90\n100 0\n200 0\n300 0\n'
      ),
      # Horizontal angles outside [0, 360].
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 2 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0 370\n100 0\n200 0\n'
      ),
      # Negative candela value.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 1 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0\n-100 0\n'
      ),
      # Non-symmetric horizontal slices outside Type C symmetry domain.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 2 1 1 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0 85\n5526 5506\n9999 1111\n'
      ),
      # Truncated candela data.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 2 1 1 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0 85\n5526 5506\n'
      ),
      # Non-numeric candela token.
      (
          'IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 2 1 1 2 -0.29 -0.29 0\n1 1'
          ' 40.0\n0 90\n0\n5526 invalid_float\n'
      ),
  ]
  for idx, bad_ies in enumerate(invalid_profiles):
    with pytest.raises(RuntimeError):
      _load_ies_text(tmp_path / f'invalid_{idx}.ies', bad_ies)


def test_ies_emitter_plugin_evaluation_and_sampling(tmp_path):
  """Tests `ies` plugin evaluation, sampling, traversal, and property checks."""
  # 1. Direct evaluation of a 1D profile, linear in theta over [0, 90] deg.
  linear_1d_ies = test_helpers.write_ies_file(
      tmp_path / 'linear_1d.ies', [0.0, 90.0], [0.9, 0.3]
  )
  emitter = mi.load_dict({
      'type': 'ies',
      'filename': linear_1d_ies,
      'nested_emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': [2.0, 4.0, 6.0]},
      },
  })
  for theta_deg, expected in (
      (0.0, [1.8, 3.6, 5.4]),
      (60.0, [1.0, 2.0, 3.0]),
      (180.0, [0.0, 0.0, 0.0]),
  ):
    np.testing.assert_allclose(
        _eval_ies_at(emitter, theta_deg, 0.0), expected, rtol=1e-5, atol=1e-6
    )

  # In-place traversal update of a 2D `(5, 2)` profile.
  emitter = mi.load_dict({
      'type': 'ies',
      'filename': test_helpers.write_ies_file(
          tmp_path / 'ones_2d.ies',
          [0.0, 90.0],
          np.ones((5, 2)),
          [0.0, 90.0, 180.0, 270.0, 360.0],
      ),
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
      _eval_ies_at(emitter, 60.0, 90.0), [16.0 / 3, 8.0 / 3, 4.0 / 3], rtol=1e-4
  )
  np.testing.assert_allclose(
      _eval_ies_at(emitter, 60.0, 180.0), [8.0, 4.0, 2.0], rtol=1e-4
  )

  if dr.is_diff_v(mi.Float):
    dr.enable_grad(params['ies_profile'])
    params.update()
    si_grad = dr.zeros(mi.SurfaceInteraction3f)
    si_grad.sh_frame = mi.Frame3f(mi.Vector3f(0.0, 0.0, 1.0))
    theta_rad = np.radians(60.0)
    phi_rad = np.radians(90.0)
    si_grad.wi = mi.Vector3f(
        float(-np.sin(theta_rad) * np.cos(phi_rad)),
        float(np.sin(theta_rad) * np.sin(phi_rad)),
        float(np.cos(theta_rad)),
    )
    dr.backward(dr.sum(emitter.eval(si_grad)))
    grad = np.asarray(dr.grad(params['ies_profile']))
    # At (theta = 60 deg, phi = 90 deg -> knot row 1):
    # eval = (4 + 2 + 1) * (prof[1, 0] / 3 + 2 * prof[1, 1] / 3), so
    # d(sum(eval))/d(prof[1, 0]) = 7 / 3 and d(sum(eval))/d(prof[1, 1]) = 14 / 3.
    expected_grad = np.zeros((5, 2), dtype=np.float32)
    expected_grad[1, 0] = 7.0 / 3
    expected_grad[1, 1] = 14.0 / 3
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)

  # 2. Attached shape sampling, PDF, and `sample_ray` consistency.
  ramp_ies = test_helpers.write_ies_file(
      tmp_path / 'ramp.ies', [0.0, 90.0], [1.0, 0.2]
  )
  shape_cases = [
      ('disk', [0.0, 0.0, 2.0], [0.3, 0.7]),
      ('rectangle', [0.0, 0.0, 2.0], [0.3, 0.7]),
      ('sphere', [0.0, 0.0, 2.0], [0.3, 0.7]),
      ('cylinder', [2.0, 0.0, 0.25], [0.5, 0.0]),
  ]
  for shape_type, receiver_p, sample_uv in shape_cases:
    shape = mi.load_dict({
        'type': shape_type,
        'to_world': mi.ScalarTransform4f().scale(0.5),
        'emitter': {
            'type': 'ies',
            'filename': ramp_ies,
            'nested_emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [2.0, 4.0, 6.0]},
            },
        },
    })
    sh_emitter = shape.emitter()
    assert sh_emitter.bbox().valid()
    assert 'IesEmitter[' in str(sh_emitter)

    it = dr.zeros(mi.Interaction3f)
    it.p = mi.Point3f(*receiver_p)
    ds, weight = sh_emitter.sample_direction(it, mi.Point2f(*sample_uv))
    pdf = sh_emitter.pdf_direction(it, ds)
    eval_dir = sh_emitter.eval_direction(it, ds)
    assert float(ds.pdf[0]) > 0.0
    np.testing.assert_allclose(float(pdf[0]), float(ds.pdf[0]), rtol=1e-5)
    np.testing.assert_allclose(
        np.array(eval_dir / pdf).ravel(), np.array(weight).ravel(), rtol=1e-5
    )
    ps, pos_weight = sh_emitter.sample_position(0.0, mi.Point2f(*sample_uv))
    np.testing.assert_allclose(
        float(pos_weight[0]), 1.0 / float(ps.pdf[0]), rtol=1e-5
    )

  # 3. Point-light wrapper mode (`DeltaPosition`, delegates to `'point'`).
  pt_emitter = mi.load_dict({
      'type': 'ies',
      'filename': ramp_ies,
      'nested_emitter': {
          'type': 'point',
          'intensity': {'type': 'rgb', 'value': [8.0, 4.0, 2.0]},
          'to_world': (
              mi.ScalarTransform4f()
              .translate([0.0, 0.0, 2.0])
              .rotate([1.0, 0.0, 0.0], 180.0)
          ),
      },
  })
  assert pt_emitter.flags() == int(mi.EmitterFlags.DeltaPosition)
  it = dr.zeros(mi.Interaction3f)
  it.p = mi.Point3f(0.0, 0.0, 0.0)
  ds, weight = pt_emitter.sample_direction(it, mi.Point2f(0.5, 0.5))
  assert bool(ds.delta[0])
  np.testing.assert_allclose(
      np.array(weight).ravel(), [2.0, 1.0, 0.5], rtol=1e-5
  )
  np.testing.assert_allclose(
      np.array(pt_emitter.eval_direction(it, ds)).ravel(),
      [2.0, 1.0, 0.5],
      rtol=1e-5,
  )
  assert float(pt_emitter.pdf_direction(it, ds)[0]) == 0.0
  ray, ray_weight = pt_emitter.sample_ray(
      0.0, 0.5, mi.Point2f(0.5, 0.5), mi.Point2f(0.8, 0.7)
  )
  theta = np.arccos(float(-ray.d.z[0]))
  expected_profile = 1.0 - 0.8 * theta / (0.5 * np.pi)
  np.testing.assert_allclose(
      np.array(ray_weight).ravel(),
      4.0 * np.pi * np.array([8.0, 4.0, 2.0]) * expected_profile,
      rtol=1e-4,
  )
  _, back_weight = pt_emitter.sample_ray(
      0.0, 0.5, mi.Point2f(0.5, 0.5), mi.Point2f(0.8, 0.2)
  )
  np.testing.assert_allclose(
      np.array(back_weight).ravel(), [0.0] * 3, atol=1e-6
  )

  # 4. `visibility = 'primary'` vs `sampling_weight = 0.0`.
  vis_disk = mi.load_dict({
      'type': 'disk',
      'visibility': 'primary',
      'emitter': {
          'type': 'ies',
          'filename': test_helpers.write_ies_file(
              tmp_path / 'uniform.ies', [0.0, 90.0], [1.0, 1.0]
          ),
          'nested_emitter': {
              'type': 'area',
              'radiance': {'type': 'rgb', 'value': [2.0, 2.0, 2.0]},
          },
      },
  })
  vis_emitter = vis_disk.emitter()
  np.testing.assert_allclose(
      _eval_ies_at(vis_emitter, 0.0, 0.0), [2.0] * 3, rtol=1e-5
  )
  vis_params = mi.traverse(vis_emitter)
  vis_params['sampling_weight'] = 0.0
  vis_params.update()
  np.testing.assert_allclose(
      _eval_ies_at(vis_emitter, 0.0, 0.0), [0.0] * 3, atol=1e-6
  )

  # 5. `angle_scale` (positive and negative) and `normalize`.
  linear_ies = test_helpers.write_ies_file(
      tmp_path / 'linear.ies', [0.0, 90.0], [100.0, 0.0]
  )
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
        'filename': linear_ies,
        'angle_scale': angle_scale,
        'nested_emitter': {'type': 'area'},
    })
    np.testing.assert_allclose(
        _eval_ies_at(scaled, theta_deg, 0.0),
        [expected] * 3,
        rtol=1e-4,
        atol=1e-6,
    )

  # `normalize = True` on a constant 200 cd hemispherical profile:
  # single vertical interval [0, pi/2] has midpoint pi/4, so
  # power = (2*pi * (pi/2) * sin(pi/4) * 200) / (2*pi) = 200 * (pi/2) * sin(pi/4).
  norm_emitter = mi.load_dict({
      'type': 'ies',
      'filename': test_helpers.write_ies_file(
          tmp_path / 'const.ies', [0.0, 90.0], [200.0, 200.0]
      ),
      'normalize': True,
      'nested_emitter': {'type': 'area'},
  })
  expected_power = 200.0 * (0.5 * np.pi) * np.sin(0.25 * np.pi)
  np.testing.assert_allclose(
      _eval_ies_at(norm_emitter, 30.0, 0.0),
      [200.0 / expected_power] * 3,
      rtol=1e-4,
  )

  # 6. Invalid plugin properties and profile updates must raise.
  invalid_dicts = [
      # Missing nested emitter.
      {
          'type': 'ies',
          'filename': linear_ies,
      },
      # Multiple nested emitters.
      {
          'type': 'ies',
          'filename': linear_ies,
          'e0': {'type': 'area'},
          'e1': {'type': 'area'},
      },
      # Missing profile file.
      {
          'type': 'ies',
          'nested_emitter': {'type': 'area'},
      },
  ]
  for invalid_dict in invalid_dicts:
    with pytest.raises(RuntimeError):
      mi.load_dict(invalid_dict)
  # `ies_profile` must keep the shape of the IES table.
  params = mi.traverse(norm_emitter)
  params['ies_profile'] = mi.TensorXf([1.0, 1.0, 1.0], (3,))
  with pytest.raises(RuntimeError):
    params.update()


def _render_ies_scene(
    shape: str, to_world: mi.ScalarTransform4f, emitter: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray]:
  """Renders a top-down view of a unit floor lit by a single small light.

  Args:
    shape: Parent shape of `emitter`, or `point` for a standalone emitter.
    to_world: Light transform (unscaled).
    emitter: Emitter dict, or `None` for a plain `area` / `point` light.

  Returns:
    The floor radiance (H, W) and the world position seen by each pixel
    (H, W, 3).
  """
  if shape == 'point':
    point = {'type': 'point', 'to_world': to_world}
    light = {**emitter, 'nested_emitter': point} if emitter else point
  else:
    if emitter:
      emitter = {
          **emitter,
          'nested_emitter': {'type': 'area'},
          **({'to_world': to_world} if shape == 'cube' else {}),
      }
    else:
      emitter = {'type': 'area'}
    light = {
        'type': shape,
        'to_world': to_world @ mi.ScalarTransform4f().scale(_IES_LIGHT_SIZE),
        'emitter': emitter,
    }
  scene = mi.load_dict({
      'type': 'scene',
      'integrator': {
          'type': 'aov',
          'aovs': 'pos:position',
          'integrator': {'type': 'direct', 'bsdf_samples': 0},
      },
      'sensor': {
          'type': 'orthographic',
          'to_world': (
              mi.ScalarTransform4f().look_at(
                  origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]
              )
          ),
          'film': {
              'type': 'hdrfilm',
              'width': _IES_RESOLUTION,
              'height': _IES_RESOLUTION,
              'rfilter': {'type': 'box'},
          },
      },
      'floor': {'type': 'rectangle', 'bsdf': {'type': 'diffuse'}},
      'light': light,
  })
  image = np.array(mi.render(scene, spp=16))
  return image[..., 0], image[..., -3:]


def _eval_ies_at(
    emitter: mi.Emitter, theta_deg: float, phi_deg: float
) -> np.ndarray:
  """Evaluates `emitter` at polar angle `theta_deg` and Type C azimuth `phi_deg`."""
  theta = float(np.radians(theta_deg))
  phi = float(np.radians(phi_deg))
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.sh_frame = mi.Frame3f(mi.Vector3f(0.0, 0.0, 1.0))
  si.wi = mi.Vector3f(
      float(-np.sin(theta) * np.cos(phi)),
      float(np.sin(theta) * np.sin(phi)),
      float(np.cos(theta)),
  )
  return np.array(emitter.eval(si)).ravel()


def _load_ies_text(path: pathlib.Path, ies_text: str) -> mi.Emitter:
  """Writes `ies_text` to `path` and loads it as an `ies` area emitter."""
  path.write_text(ies_text)
  return mi.load_dict({
      'type': 'ies',
      'filename': str(path),
      'nested_emitter': {'type': 'area'},
  })
