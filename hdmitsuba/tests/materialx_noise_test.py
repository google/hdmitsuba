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

"""Tests for MaterialX 2D/3D procedural noises."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Usd
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx


@pytest.fixture(autouse=True)
def _set_default_variant() -> None:
  mtlx.set_default_variant()


def test_worley_solid_large_jitter() -> None:
  """Verifies that Solid Worley with `jitter > 1` hashes the jittered cell."""
  stage = Usd.Stage.CreateInMemory()
  # At `texcoord = (0.98, 0.5)` with `jitter = 1.8`, the closest feature point
  # originates in cell `(0, 0)` and jitters across the boundary into `(0, -1)`;
  # `style=1` hashes cell `(0, -1)` (`floor(minpos + p)`).
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_worleynoise2d_float',
              {'texcoord': (0.98, 0.5), 'jitter': 1.8, 'style': 1},
          ),
          stage=stage,
          channels=1,
      ),
      mtlx.eval_emission_expr(
          ('ND_cellnoise2d_float', {'texcoord': (0.5, -0.5)}),
          stage=stage,
          channels=1,
      ),
      atol=1e-6,
  )


@pytest.mark.parametrize(
    'node_id, param, values, inputs',
    [
        (
            'ND_fractal3d_float',
            'octaves',
            (1, 4),
            {'position': (0.3, 0.7, 0.2)},
        ),
        (
            'ND_worleynoise2d_float',
            'style',
            (0, 1),
            {'texcoord': (0.3, 0.7), 'jitter': 0.8},
        ),
    ],
)
def test_dynamic_noise_parameter(
    node_id: str,
    param: str,
    values: tuple[int, int],
    inputs: dict[str, object],
) -> None:
  """Tests per-sample varying fractal octaves and Worley style.

  Args:
    node_id: MaterialX noise NodeDef identifier.
    param: Name of the varying integer parameter.
    values: `(low, high)` parameter values selected by `p.x > 0`.
    inputs: Additional constant inputs for `node_id`.
  """
  stage = Usd.Stage.CreateInMemory()
  si = dr.zeros(mi.SurfaceInteraction3f, 2)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.p = mi.Point3f([-1.0, 1.0], 0.0, 0.0)
  pos = mtlx.define_shader(stage, 'Pos', 'ND_position_vector3', space='world')
  px = mtlx.define_shader(stage, 'Px', 'ND_extract_vector3', in_=pos, index=0)
  varying = (
      'ND_ifgreater_integer',
      {'value1': px, 'value2': 0.0, 'in1': values[1], 'in2': values[0]},
  )
  expected = np.concatenate([
      mtlx.eval_emission_expr(
          (node_id, {**inputs, param: v}), stage=stage, channels=1
      )
      for v in values
  ])
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (node_id, {**inputs, param: varying}), stage=stage, si=si, channels=1
      ),
      expected,
      atol=1e-6,
  )


def test_constant_coord_noise_not_spatially_varying() -> None:
  """Verifies constant-coordinate noise omits `SpatiallyVarying`."""
  _, em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=('ND_noise3d_color3', {'position': (0.3, 0.7, 0.2)}),
  )
  assert em is not None
  assert not (em.flags() & int(mi.EmitterFlags.SpatiallyVarying))


def test_dynamic_fractal_ad() -> None:
  """Verifies forward and reverse AD for symbolic `while_loop` fractal noise."""
  stage = Usd.Stage.CreateInMemory()
  pos = mtlx.define_shader(stage, 'Pos', 'ND_position_vector3', space='world')
  px = mtlx.define_shader(stage, 'Px', 'ND_extract_vector3', in_=pos, index=0)
  # Lanes with `p.x <= 0` run 2 octaves; lanes with `p.x > 0` run 4 octaves.
  xs = [-0.2, 0.5, -0.8, 1.1]
  hi_mask = np.array(xs) > 0.0
  si4 = dr.zeros(mi.SurfaceInteraction3f, 4)
  si4.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si4.p = mi.Point3f(xs, 0.3, 0.4)

  dyn_octaves = (
      'ND_ifgreater_integer',
      {'value1': px, 'value2': 0.0, 'in1': 4, 'in2': 2},
  )

  def compile_fractals(**inputs: object) -> list[mi.Emitter]:
    ems = []
    for octaves in (dyn_octaves, 2, 4):
      _, em = mtlx.compile_mtlx_material(
          'ND_surface_unlit',
          stage=stage,
          emission_color=(
              'ND_fractal3d_color3',
              {
                  'octaves': octaves,
                  'lacunarity': 2.1,
                  'diminish': 0.5,
                  **inputs,
              },
          ),
      )
      assert em is not None
      ems.append(em)
    return ems

  # A constant (width 1) position checks that the loop state is widened.
  ems = compile_fractals(position=(0.3, 0.7, 0.2))
  for val in (2.1, 0.5):
    fwd = []
    for em in ems[1:]:
      p = mi.traverse(em)
      key = next(
          k
          for k in p.keys()
          if k.startswith('radiance.param') and np.allclose(p[k], val)
      )
      dr.enable_grad(p[key])
      dr.set_grad(p[key], 1.0)
      p.update()
      y = em.eval(si4)
      dr.forward_to(y)
      fwd.append(dr.grad(y)[0])
      dr.disable_grad(p[key])
      p.update()
    expected_fwd = np.where(hi_mask, fwd[1], fwd[0])

    p = mi.traverse(ems[0])
    key = next(
        k
        for k in p.keys()
        if k.startswith('radiance.param') and np.allclose(p[k], val)
    )
    dr.enable_grad(p[key])
    dr.set_grad(p[key], 1.0)
    p.update()
    y = ems[0].eval(si4)
    dr.forward_to(y)
    np.testing.assert_allclose(
        dr.grad(y)[0], expected_fwd, rtol=1e-5, atol=1e-6
    )
    dr.set_grad(p[key], 0.0)
    dr.backward(dr.sum(dr.full(mi.EmitterPtr, ems[0], 4).eval(si4)[0]))
    np.testing.assert_allclose(
        dr.grad(p[key]), np.sum(expected_fwd), rtol=1e-5, atol=1e-6
    )
    dr.disable_grad(p[key])
    p.update()

  pos_grads = []
  for em in compile_fractals():
    px_g = mi.Float(xs)
    dr.enable_grad(px_g)
    si4.p = mi.Point3f(px_g, 0.3, 0.4)
    dr.backward(dr.sum(em.eval(si4)[0]))
    pos_grads.append(dr.grad(px_g))
  np.testing.assert_allclose(
      pos_grads[0],
      np.where(hi_mask, pos_grads[2], pos_grads[1]),
      rtol=1e-5,
      atol=1e-6,
  )


def test_worley_ad() -> None:
  """Verifies Worley reverse AD for static `style=0` and dynamic `style`."""
  stage = Usd.Stage.CreateInMemory()
  pos = mtlx.define_shader(stage, 'PosW', 'ND_position_vector3', space='world')
  px = mtlx.define_shader(stage, 'PxW', 'ND_extract_vector3', in_=pos, index=0)
  xs = [-0.25, 0.35, -0.75, 0.85]
  cell_mask = np.array(xs) > 0.0
  dyn_style = (
      'ND_ifgreater_integer',
      {'value1': px, 'value2': 0.0, 'in1': 1, 'in2': 0},
  )
  ems = [
      mtlx.compile_mtlx_material(
          'ND_surface_unlit',
          stage=stage,
          emission_color=(
              'ND_worleynoise3d_float',
              {'position': pos, 'jitter': 0.8, 'style': style},
          ),
      )[1]
      for style in (0, dyn_style)
  ]
  assert ems[0] is not None and ems[1] is not None

  eps = 1e-4
  si = dr.zeros(mi.SurfaceInteraction3f, 4)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.p = mi.Point3f([x + eps for x in xs], 0.3, 0.4)
  y_hi = np.array(ems[0].eval(si)[0])
  si.p = mi.Point3f([x - eps for x in xs], 0.3, 0.4)
  fd_px = (y_hi - np.array(ems[0].eval(si)[0])) / (2.0 * eps)

  px_grads = []
  for em in ems:
    px_g = mi.Float(xs)
    dr.enable_grad(px_g)
    si.p = mi.Point3f(px_g, 0.3, 0.4)
    dr.backward(dr.sum(em.eval(si)[0]))
    px_grads.append(np.array(dr.grad(px_g)))
  np.testing.assert_allclose(px_grads[0], fd_px, rtol=1e-2, atol=1e-3)
  np.testing.assert_allclose(
      px_grads[1], np.where(cell_mask, 0.0, px_grads[0]), rtol=1e-5, atol=1e-6
  )

  # Symbolic vcall reverse AD wrt `jitter`.
  si.p = mi.Point3f(xs, 0.3, 0.4)
  jit_grads = []
  for em in ems:
    p = mi.traverse(em)
    key = next(k for k in p.keys() if k.startswith('radiance.param'))
    dr.enable_grad(p[key])
    p.update()
    dr.backward(dr.sum(dr.full(mi.EmitterPtr, em, 4).eval(si)[0]))
    jit_grads.append(float(dr.grad(p[key])[0]))
    dr.disable_grad(p[key])
    p.update()
  p = mi.traverse(ems[0])
  key = next(k for k in p.keys() if k.startswith('radiance.param'))
  p[key] = mi.Float(0.8 + eps)
  p.update()
  j_hi = np.array(ems[0].eval(si)[0])
  p[key] = mi.Float(0.8 - eps)
  p.update()
  fd_jit = (j_hi - np.array(ems[0].eval(si)[0])) / (2.0 * eps)
  np.testing.assert_allclose(jit_grads[0], np.sum(fd_jit), rtol=1e-2, atol=1e-3)
  np.testing.assert_allclose(
      jit_grads[1], np.sum(fd_jit[~cell_mask]), rtol=1e-2, atol=1e-3
  )
