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

"""Tests for MaterialX math, vector, channel, logic, matrix, and emission ops."""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdShade
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx
from hdmitsuba.tests import test_helpers


@pytest.fixture(autouse=True)
def _set_default_variant() -> None:
  mtlx.set_default_variant()


def _num_params(obj: Any, prefix: str = 'radiance.param') -> int:
  return sum(k.startswith(prefix) for k in mi.traverse(obj).keys())


def _code_size(obj: Any) -> int:
  match = re.search(r'code_size = (\d+)', str(obj))
  assert match is not None, str(obj)
  return int(match.group(1))


def _make_uv_batch(us: Any, vs: Any = 0.0) -> tuple[
    mi.SurfaceInteraction3f,
    tuple[str, dict[str, Any]],
    tuple[str, dict[str, Any]],
]:
  us_arr = np.asarray(us, dtype=np.float64)
  vs_arr = np.broadcast_to(np.asarray(vs, dtype=np.float64), us_arr.shape)
  si = dr.zeros(mi.SurfaceInteraction3f, len(us_arr))
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.uv = mi.Point2f(us_arr, 1.0 - vs_arr)
  tc = ('ND_texcoord_vector2', {})
  return (
      si,
      ('ND_extract_vector2', {'in_': tc, 'index': 0}),
      ('ND_extract_vector2', {'in_': tc, 'index': 1}),
  )


def _compile_emission(expr: Any, stage: Usd.Stage | None = None) -> mi.Emitter:
  _, em = mtlx.compile_mtlx_material(
      'ND_surface_unlit', stage=stage, emission_color=expr
  )
  assert em is not None
  return em


def _compile_float_emission(
    expr: Any, stage: Usd.Stage | None = None
) -> mi.Emitter:
  return _compile_emission(('ND_convert_float_color3', {'in_': expr}), stage)


def _const_ratio(num: int, den: int = 1) -> tuple[str, dict[str, Any]]:
  n = ('ND_convert_integer_float', {'in_': num})
  if den == 1:
    return n
  return (
      'ND_divide_float',
      {'in1': n, 'in2': ('ND_convert_integer_float', {'in_': den})},
  )


def _check_op(
    stage: Usd.Stage, node_id: str, expected: Any, **inputs: Any
) -> None:
  """Evaluates a MaterialX node via area-light emission and checks its output."""
  expected_arr = np.atleast_1d(np.asarray(expected, dtype=np.float64))
  actual = mtlx.eval_emission_expr(
      (node_id, inputs), stage=stage, channels=len(expected_arr)
  )
  np.testing.assert_allclose(
      actual, expected_arr, rtol=1e-4, atol=1e-4, err_msg=node_id
  )


def _make_position_batch(
    stage: Usd.Stage,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> tuple[
    mi.SurfaceInteraction3f, UsdShade.Shader, UsdShade.Shader, UsdShade.Shader
]:
  """Creates a batched `SurfaceInteraction3f` and world-position extractors."""
  si = dr.zeros(mi.SurfaceInteraction3f, len(xs))
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.p = mi.Point3f(xs, ys, zs)
  pos = mtlx.define_shader(stage, 'Pos', 'ND_position_vector3', space='world')
  px = mtlx.define_shader(stage, 'Px', 'ND_extract_vector3', in_=pos, index=0)
  py = mtlx.define_shader(stage, 'Py', 'ND_extract_vector3', in_=pos, index=1)
  pz = mtlx.define_shader(stage, 'Pz', 'ND_extract_vector3', in_=pos, index=2)
  return si, px, py, pz


def test_elementary_math_and_interpolation() -> None:
  """Tests elementary math, domain edge cases, and interpolation ops over dynamic batches."""
  stage = Usd.Stage.CreateInMemory()
  x = np.array([-2.0, -1.5, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 3.25])
  y = np.array([3.0, 2.0, 0.5, 0.0, -1.0, -2.0, 0.0, -0.5, 3.0, 0.5])
  z = np.array([-1.0, 0.0, -0.5, 0.5, -1.0, 0.0, 0.5, 1.0, 1.0, 2.0])
  si, px, py, pz = _make_position_batch(
      stage, x.tolist(), y.tolist(), z.tolist()
  )

  def eval_batch(node_id: str, channels: int = 1, **kw: Any) -> np.ndarray:
    return mtlx.eval_emission_expr(
        (node_id, kw), stage=stage, si=si, channels=channels
    )

  # Unary math & domain edge cases (halfway rounding away from zero, sign(0)=0,
  # sqrt/ln on non-positive inputs -> 0).
  for node_id, expected in [
      ('ND_absval_float', np.abs(x)),
      ('ND_sign_float', [-1, -1, -1, 0, 0, 1, 1, 1, 1, 1]),
      ('ND_floor_float', np.floor(x)),
      ('ND_ceil_float', np.ceil(x)),
      ('ND_round_float', [-2, -2, -1, 0, 0, 0, 1, 1, 2, 3]),
      ('ND_fract_float', x - np.floor(x)),
      ('ND_sin_float', np.sin(x)),
      ('ND_cos_float', np.cos(x)),
      ('ND_asin_float', np.arcsin(np.clip(x, -1.0, 1.0))),
      ('ND_acos_float', np.arccos(np.clip(x, -1.0, 1.0))),
      ('ND_sqrt_float', np.sqrt(np.maximum(x, 0.0))),
      (
          'ND_ln_float',
          np.where(x > 0.0, np.log(np.where(x > 0.0, x, 1.0)), 0.0),
      ),
  ]:
    np.testing.assert_allclose(
        eval_batch(node_id, in_=px),
        expected,
        rtol=1e-5,
        atol=1e-5,
        err_msg=node_id,
    )
  np.testing.assert_allclose(
      eval_batch('ND_invert_float', in_=px, amount=1.0), 1.0 - x, atol=1e-5
  )
  for node_id, x_clip, fn in [
      ('ND_tan_float', np.clip(x, -1.2, 1.2), np.tan),
      ('ND_exp_float', np.clip(x, -4.0, 4.0), np.exp),
  ]:
    si.p = mi.Point3f(x_clip, y, z)
    np.testing.assert_allclose(
        eval_batch(node_id, in_=px), fn(x_clip), rtol=1e-5, atol=1e-5
    )
  si.p = mi.Point3f(x, y, z)

  # Binary arithmetic & singularities: divide by 0, modulo by <= 0, atan2(0, 0),
  # OSL safe_pow (negative base with odd/even/non-integer exponent, 0^0), and
  # MaterialX safepower.
  safe_div = np.divide(x, y, out=np.zeros_like(x), where=y != 0.0)
  expected_pow = [
      -8.0,  # (-2)^3
      2.25,  # (-1.5)^2
      0.0,  # (-0.5)^0.5 -> 0
      1.0,  # (-0)^0 -> 1
      0.0,  # 0^(-1) -> 0
      16.0,  # 0.25^(-2)
      1.0,  # 0.5^0
      1.0,  # 1^(-0.5)
      8.0,  # 2^3
      np.sqrt(3.25),
  ]
  for node_id, expected in [
      ('ND_add_float', x + y),
      ('ND_subtract_float', x - y),
      ('ND_multiply_float', x * y),
      ('ND_divide_float', safe_div),
      ('ND_modulo_float', x - y * np.floor(safe_div)),
      ('ND_min_float', np.minimum(x, y)),
      ('ND_max_float', np.maximum(x, y)),
      ('ND_power_float', expected_pow),
  ]:
    np.testing.assert_allclose(
        eval_batch(node_id, in1=px, in2=py),
        expected,
        rtol=1e-5,
        atol=1e-5,
        err_msg=node_id,
    )
  np.testing.assert_allclose(
      eval_batch('ND_inside_float', in_=px, mask=py), x * y, atol=1e-5
  )
  np.testing.assert_allclose(
      eval_batch('ND_outside_float', in_=px, mask=py), x * (1.0 - y), atol=1e-5
  )
  np.testing.assert_allclose(
      eval_batch('ND_atan2_float', iny=py, inx=px),
      np.where((x == 0.0) & (y == 0.0), 0.0, np.arctan2(y, x)),
      atol=1e-5,
  )
  nz_x = np.where(x == 0.0, 0.5, x)
  si.p = mi.Point3f(nz_x, y, z)
  np.testing.assert_allclose(
      eval_batch('ND_safepower_float', in1=px, in2=py),
      np.sign(nz_x) * np.power(np.abs(nz_x), y),
      rtol=1e-5,
      atol=1e-5,
  )
  si.p = mi.Point3f(x, y, z)

  # Interpolation (`mix`, `smoothstep` including `low >= high`, `remap`
  # including `inlow == inhigh`, and `splitlr` exact step at `center`).
  np.testing.assert_allclose(
      eval_batch('ND_mix_float', fg=px, bg=py, mix=0.25),
      0.75 * y + 0.25 * x,
      atol=1e-5,
  )
  # Lanes cover: x<lo, x<lo, x==lo<hi, x<lo with lo>hi, x>lo>hi, x>lo>hi,
  # x==lo>hi, x==lo>hi, lo<x<hi (t=0.5), x>lo>hi.
  np.testing.assert_allclose(
      eval_batch('ND_smoothstep_float', in_=px, low=pz, high=py),
      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
      atol=1e-5,
  )
  np.testing.assert_allclose(
      eval_batch(
          'ND_remap_float',
          in_=px,
          inlow=0.0,
          inhigh=py,
          outlow=2.0,
          outhigh=6.0,
      ),
      2.0 + safe_div * 4.0,
      rtol=1e-5,
      atol=1e-5,
  )
  _check_op(
      stage,
      'ND_splitlr_float',
      1.0,
      valuel=0.0,
      valuer=1.0,
      center=0.5,
      texcoord=(0.5, 0.5),
  )
  _check_op(
      stage,
      'ND_splitlr_float',
      0.0,
      valuel=0.0,
      valuer=1.0,
      center=0.5,
      texcoord=(0.49999, 0.5),
  )

  # Multi-channel elementwise dispatch: `vector2` (`.z == 0` on promotion) and
  # `color4` (`FA` scalar broadcast).
  v2 = mtlx.define_shader(stage, 'V2', 'ND_combine2_vector2', in1=px, in2=py)
  np.testing.assert_allclose(
      eval_batch('ND_fract_vector2', channels=3, in_=v2),
      np.column_stack([x - np.floor(x), y - np.floor(y), np.zeros_like(x)]),
      atol=1e-5,
  )
  c4 = mtlx.define_shader(
      stage, 'C4', 'ND_combine4_color4', in1=px, in2=py, in3=pz, in4=px
  )
  np.testing.assert_allclose(
      eval_batch('ND_clamp_color4FA', channels=4, in_=c4, low=-0.5, high=1.5),
      np.clip(np.column_stack([x, y, z, x]), -0.5, 1.5),
      atol=1e-5,
  )

  # Differentiable `atan2(0, 0)` must not poison AD gradients.
  _, atan2_em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=stage,
      emission_color=(
          'ND_convert_float_color3',
          {'in_': ('ND_atan2_float', {'iny': py, 'inx': px})},
      ),
  )
  assert atan2_em is not None
  si_zero = dr.zeros(mi.SurfaceInteraction3f, 2)
  si_zero.wi = mi.Vector3f(0.0, 0.0, 1.0)
  p_x = mi.Float([0.0, 1.0])
  dr.enable_grad(p_x)
  si_zero.p = mi.Point3f(p_x, [0.0, 1.0], 0.0)
  dr.backward(dr.sum(atan2_em.eval(si_zero)))
  np.testing.assert_allclose(dr.grad(p_x), [0.0, -1.5], atol=1e-5)


def test_conditionals_logic_and_switch() -> None:
  """Tests dynamic conditionals, boolean logic, constant folding, and switch."""
  stage = Usd.Stage.CreateInMemory()
  x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 9.5, 10.0])
  y = np.array([-1.0, 0.5, 0.0, -0.5, 2.0, 1.0, 5.0, 1.0, -1.0])
  z = np.linspace(-1.0, 1.0, len(x))
  si, px, py, pz = _make_position_batch(
      stage, x.tolist(), y.tolist(), z.tolist()
  )

  def eval_batch(node_id: str, channels: int = 1, **kw: Any) -> np.ndarray:
    return mtlx.eval_emission_expr(
        (node_id, kw), stage=stage, si=si, channels=channels
    )

  c4 = mtlx.define_shader(
      stage, 'C4', 'ND_combine4_color4', in1=px, in2=py, in3=pz, in4=px
  )
  np.testing.assert_allclose(
      eval_batch(
          'ND_ifgreater_color4',
          channels=4,
          value1=px,
          value2=py,
          in1=c4,
          in2=(0.1, 0.2, 0.3, 0.4),
      ),
      np.where(
          (x > y)[:, None],
          np.column_stack([x, y, z, x]),
          np.array([0.1, 0.2, 0.3, 0.4]),
      ),
      atol=1e-5,
  )
  np.testing.assert_allclose(
      eval_batch(
          'ND_ifgreatereq_float', value1=px, value2=py, in1=2.5, in2=-1.0
      ),
      np.where(x >= y, 2.5, -1.0),
      atol=1e-5,
  )
  np.testing.assert_allclose(
      eval_batch('ND_ifequal_float', value1=px, value2=py, in1=2.5, in2=-1.0),
      np.where(x == y, 2.5, -1.0),
      atol=1e-5,
  )

  b1 = mtlx.define_shader(
      stage, 'B1', 'ND_ifgreater_boolean', value1=px, value2=0.0
  )
  b2 = mtlx.define_shader(
      stage, 'B2', 'ND_ifgreater_boolean', value1=py, value2=0.0
  )
  bx, by = x > 0.0, y > 0.0
  np.testing.assert_allclose(
      eval_batch(
          'ND_convert_boolean_float',
          in_=(
              'ND_logical_and',
              {'in1': b1, 'in2': ('ND_logical_not', {'in_': b2})},
          ),
      ),
      (bx & ~by).astype(np.float64),
      atol=1e-5,
  )
  np.testing.assert_allclose(
      eval_batch(
          'ND_convert_boolean_float',
          in_=(
              'ND_logical_xor',
              {'in1': b1, 'in2': ('ND_logical_or', {'in1': b1, 'in2': b2})},
          ),
      ),
      (bx ^ (bx | by)).astype(np.float64),
      atol=1e-5,
  )

  # Constant-zero folding through compositing and logical ops elides emitters.
  _, zero_plus = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=stage,
      emission_color=(
          'ND_plus_color3',
          {'fg': (0.0, 0.0, 0.0), 'bg': (0.0, 0.0, 0.0)},
      ),
  )
  assert zero_plus is None
  _, folded_and = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=stage,
      emission_color=(
          'ND_ifequal_color3B',
          {
              'value1': ('ND_logical_and', {'in1': False, 'in2': b1}),
              'value2': True,
              'in1': (1.0, 1.0, 1.0),
              'in2': (0.0, 0.0, 0.0),
          },
      ),
  )
  assert folded_and is None
  # Equal branches fold in `select()` and `mix()`, so coated non-emissive
  # `standard_surface` (`blended_coat_emission_edf = mix(0, 0, coat)`) elides
  # its emitter.
  _, folded_select = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=stage,
      emission_color=(
          'ND_ifgreater_color3',
          {
              'value1': px,
              'value2': 0.5,
              'in1': (0.0, 0.0, 0.0),
              'in2': (0.0, 0.0, 0.0),
          },
      ),
  )
  assert folded_select is None
  _, coated_non_emissive = mtlx.compile_mtlx_material(
      'ND_standard_surface_surfaceshader',
      stage=stage,
      coat=0.5,
      emission=0.0,
  )
  assert coated_non_emissive is None

  # Dynamic `ND_switch_float` across `< 1`, `1..9`, and `>= 10` (`-> 0.0`).
  sw_vals = np.array([float(i * 10) for i in range(1, 11)])
  sw_idx = np.clip(np.floor(x), 0, 9).astype(np.int32)
  np.testing.assert_allclose(
      eval_batch(
          'ND_switch_float',
          channels=1,
          which=px,
          **{f'in{i}': float(i * 10) for i in range(1, 11)},
      ),
      np.where(x < 10.0, sw_vals[sw_idx], 0.0),
      atol=1e-5,
  )


def test_vector_and_channel_ops() -> None:
  """Tests 2D/3D/4D vector ops, channel packing/separation, and constant conditionals."""
  stage = Usd.Stage.CreateInMemory()

  # 2D, 3D, and 4D dotproduct, crossproduct, magnitude, and normalize (testing
  # both non-zero and zero-vector safe normalization in a single 2-lane batch).
  v1, v2 = Gf.Vec3f(1.0, 2.0, 3.0), Gf.Vec3f(4.0, -5.0, 6.0)
  for node_id, expected, inputs in [
      ('ND_constant_color3', (0.2, 0.5, 0.8), {'value': (0.2, 0.5, 0.8)}),
      ('ND_dot_color3', (0.3, 0.6, 0.9), {'in_': (0.3, 0.6, 0.9)}),
      ('ND_dotproduct_vector2', 5.0, {'in1': (2.0, 3.0), 'in2': (4.0, -1.0)}),
      ('ND_dotproduct_vector3', 12.0, {'in1': v1, 'in2': v2}),
      (
          'ND_dotproduct_vector4',
          7.5,
          {'in1': (1.0, 2.0, 3.0, 4.0), 'in2': (2.0, -1.0, 0.5, 1.5)},
      ),
      ('ND_crossproduct_vector3', (27.0, 6.0, -13.0), {'in1': v1, 'in2': v2}),
      ('ND_magnitude_vector2', 5.0, {'in_': (3.0, 4.0)}),
      ('ND_magnitude_vector3', 7.0, {'in_': Gf.Vec3f(2.0, 3.0, 6.0)}),
      ('ND_magnitude_vector4', 5.0, {'in_': (1.0, 2.0, 2.0, 4.0)}),
  ]:
    _check_op(stage, node_id, expected, **inputs)

  si2 = dr.zeros(mi.SurfaceInteraction3f, 2)
  si2.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si2.p = mi.Point3f([3.0, 0.0], [4.0, 0.0], [0.0, 0.0])
  pos = mtlx.define_shader(stage, 'Pos2', 'ND_position_vector3', space='world')
  px = mtlx.define_shader(stage, 'Px2', 'ND_extract_vector3', in_=pos, index=0)
  py = mtlx.define_shader(stage, 'Py2', 'ND_extract_vector3', in_=pos, index=1)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_normalize_vector2',
              {'in_': ('ND_combine2_vector2', {'in1': px, 'in2': py})},
          ),
          stage=stage,
          si=si2,
          channels=2,
      ),
      [[0.6, 0.8], [0.0, 0.0]],
      atol=1e-5,
  )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          ('ND_normalize_vector3', {'in_': pos}),
          stage=stage,
          si=si2,
          channels=3,
      ),
      [[0.6, 0.8, 0.0], [0.0, 0.0, 0.0]],
      atol=1e-5,
  )
  si2.p = mi.Point3f([1.0, 0.0], [2.0, 0.0], [2.0, 0.0])
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_normalize_vector4',
              {
                  'in_': (
                      'ND_combine2_vector4VF',
                      {
                          'in1': pos,
                          'in2': ('ND_multiply_float', {'in1': py, 'in2': 2.0}),
                      },
                  )
              },
          ),
          stage=stage,
          si=si2,
          channels=4,
      ),
      [[0.2, 0.4, 0.4, 0.8], [0.0, 0.0, 0.0, 0.0]],
      atol=1e-5,
  )

  # Channel packing (`combine2_vector4VV`), multi-output separation (`separate4`),
  # legacy `swizzle` upgrade, and 1.39 `ND_convert_integer_boolean`.
  sep4 = mtlx.define_shader(
      stage, 'Sep4', 'ND_separate4_color4', in_=(0.15, 0.35, 0.65, 0.85)
  )
  for node_id, expected, inputs in [
      (
          'ND_convert_boolean_float',
          0.0,
          {'in_': ('ND_convert_integer_boolean', {'in_': 0})},
      ),
      (
          'ND_convert_boolean_float',
          1.0,
          {'in_': ('ND_convert_integer_boolean', {'in_': 3})},
      ),
      (
          'ND_combine2_vector4VV',
          (0.1, 0.2, 0.3, 0.4),
          {'in1': (0.1, 0.2), 'in2': (0.3, 0.4)},
      ),
      (
          'ND_combine4_color4',
          (0.85, 0.65, 0.35, 0.15),
          {
              'in1': sep4.CreateOutput('outa', Sdf.ValueTypeNames.Float),
              'in2': sep4.CreateOutput('outb', Sdf.ValueTypeNames.Float),
              'in3': sep4.CreateOutput('outg', Sdf.ValueTypeNames.Float),
              'in4': sep4.CreateOutput('outr', Sdf.ValueTypeNames.Float),
          },
      ),
      (
          'ND_swizzle_color4_color3',
          (0.9, 0.8, 0.2),
          {'in_': (0.1, 0.2, 0.8, 0.9), 'channels': 'abg'},
      ),
      # Compile-time boolean/integer condition folding (`B` and `I` variants),
      # plus 10-input 1.39 vs 5-input 1.38 `switch`.
      (
          'ND_ifequal_color4B',
          (0.5, 0.6, 0.7, 0.8),
          {
              'value1': True,
              'value2': False,
              'in1': (0.1, 0.2, 0.3, 0.4),
              'in2': (0.5, 0.6, 0.7, 0.8),
          },
      ),
      (
          'ND_switch_color3I',
          (1.0, 0.0, 0.0),
          {'in1': (1.0, 0.0, 0.0), 'in2': (0.0, 1.0, 0.0), 'which': -1},
      ),
      (
          'ND_switch_color3I',
          (0.0, 1.0, 0.0),
          {'in1': (1.0, 0.0, 0.0), 'in6': (0.0, 1.0, 0.0), 'which': 5},
      ),
      (
          'ND_switch_color3I',
          (1.0, 0.0, 0.0),
          {'in1': (1.0, 0.0, 0.0), 'in5': (0.0, 1.0, 0.0), 'which': 5},
      ),
  ]:
    _check_op(stage, node_id, expected, **inputs)


def test_colorspace_and_cycle_handling(
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests UsdColorSpaceAPI folding, cycles, and warning dedup.

  Args:
    capfd: Pytest fd capture fixture.
  """
  # `UsdColorSpaceAPI` converts sRGB `(0.5, 0.5, 0.5)` to linear Rec.709 while
  # keeping opaque `opacity = (1, 1, 1)` exact so cutout opacity (`Null`) is
  # elided.
  cs_stage = Usd.Stage.CreateInMemory()
  cs_mat = UsdShade.Material.Define(cs_stage, '/mesh/Material')
  Usd.ColorSpaceAPI.Apply(cs_mat.GetPrim()).CreateColorSpaceNameAttr(
      'srgb_rec709_scene'
  )
  bsdf, emitter = mtlx.compile_mtlx_material(
      'ND_standard_surface_surfaceshader',
      stage=cs_stage,
      emission=1.0,
      emission_color=(0.5, 0.5, 0.5),
      opacity=(1.0, 1.0, 1.0),
  )
  assert bsdf is not None and bsdf.class_name() == 'MtlxBsdf'
  assert not mi.has_flag(bsdf.flags(), mi.BSDFFlags.Null)
  assert emitter is not None
  expected_lin = Gf.ColorSpace(Gf.ColorSpaceNames.LinearRec709).Convert(
      Gf.ColorSpace(Gf.ColorSpaceNames.SRGBRec709), Gf.Vec3f(0.5, 0.5, 0.5)
  )
  np.testing.assert_allclose(
      emitter.eval(mtlx.make_test_si(wi=(0.0, 0.0, 1.0))),
      mi.Color3f(*expected_lin.GetRGB()),
      atol=1e-5,
  )

  # Cyclic shader connections break cleanly with a warning instead of recursing;
  # unsupported nodes and unknown NodeDefs warn only once on fan-out.
  cyc_stage = Usd.Stage.CreateInMemory()
  node_a = mtlx.define_shader(
      cyc_stage, 'NodeA', 'ND_add_color3', in2=(0.2, 0.3, 0.4)
  )
  node_b = mtlx.define_shader(
      cyc_stage,
      'NodeB',
      'ND_add_color3',
      in2=(0.1, 0.2, 0.3),
  )
  node_b.CreateInput('in1', Sdf.ValueTypeNames.Color3f).ConnectToSource(
      node_a.CreateOutput('out', Sdf.ValueTypeNames.Color3f)
  )
  node_a.CreateInput('in1', Sdf.ValueTypeNames.Color3f).ConnectToSource(
      node_b.CreateOutput('out', Sdf.ValueTypeNames.Color3f)
  )
  # `ND_texcoord_vector2` index=1 triggers `WarnUnsupported` in both emission
  # and opacity channels, which must be deduplicated per compiled shader.
  uv1 = mtlx.define_shader(cyc_stage, 'UV1', 'ND_texcoord_vector2', index=1)
  uv1_x = mtlx.define_shader(
      cyc_stage, 'UV1X', 'ND_extract_vector2', in_=uv1, index=0
  )
  unknown = mtlx.define_shader(
      cyc_stage, 'Unknown', 'ND_hdmitsuba_nonexistent_node'
  )
  warn_sum = (
      'ND_add_color3',
      {
          'in1': ('ND_convert_float_color3', {'in_': uv1_x}),
          'in2': ('ND_add_color3', {'in1': unknown, 'in2': unknown}),
      },
  )
  capfd.readouterr()
  _, cyc_em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=cyc_stage,
      emission_color=('ND_add_color3', {'in1': node_a, 'in2': warn_sum}),
      opacity=uv1_x,
  )
  err = capfd.readouterr().err
  assert 'Cycle detected' in err
  assert err.count("Unsupported MaterialX 'texcoord' index") == 1
  assert err.count('No MaterialX NodeDef') == 1
  assert cyc_em is not None
  # Breaking `NodeB.in1 -> NodeA` falls back to `in1 = 0`:
  # `NodeA = (0.1, 0.2, 0.3) + (0.2, 0.3, 0.4) = (0.3, 0.5, 0.7)`.
  si_uv1 = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si_uv1.uv = mi.Point2f(1.0, 0.5)
  np.testing.assert_allclose(
      cyc_em.eval(si_uv1), mi.Color3f(1.3, 1.5, 1.7), atol=1e-5
  )


@pytest.mark.skipif(
    Sdf.FileFormat.FindByExtension('mtlx') is None,
    reason='Requires an OpenUSD build with MaterialX support.',
)
def test_custom_source_asset_shader(capfd: pytest.CaptureFixture[str]) -> None:
  """Tests custom sourceAsset .mtlx NodeGraphs, multi-output ports, and helper textures.

  Args:
    capfd: Pytest file-descriptor capture fixture.
  """
  asset_stage, asset_mesh = mtlx.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  asset_sh = UsdShade.Shader.Define(asset_stage, '/mesh/Material/CustomGlow')
  asset_sh.SetSourceAsset(
      Sdf.AssetPath(
          f'{test_helpers.TEST_ASSETS_PATH}/materials/custom_glow.mtlx'
      ),
      'mtlx',
  )
  asset_sh.SetSourceAssetSubIdentifier('ND_hdmitsuba_custom_glow', 'mtlx')
  asset_sh.CreateInput('tint', Sdf.ValueTypeNames.Color3f).Set((0.3, 0.4, 0.2))
  mtlx.bind_mtlx_shader(
      asset_stage,
      asset_mesh,
      'ND_surfacematerial',
      surfaceshader=asset_sh.CreateOutput(
          'out_surface', Sdf.ValueTypeNames.Token
      ),
  )
  capfd.readouterr()
  img_asset, _ = test_helpers.assert_hydra_equal_to_offline(
      asset_stage,
      output_prefix='test_mtlx_source_asset',
      spp=4,
      subdivision_level=0,
  )
  assert 'Unsupported' not in capfd.readouterr().err
  # tint=(0.3,0.4,0.2) + lum=(0.2,0.2,0.2) + ng_tint=(0.05,0,0) +
  # mat*norm=(0,0,0.5) + srgb_to_linear(uv_quadrants(0.25,0.25)=(60,80,220)/255)
  np.testing.assert_allclose(
      mtlx.render_center(img_asset),
      np.array([0.55, 0.6, 0.9])
      + ((np.array([60.0, 80.0, 220.0]) / 255.0 + 0.055) / 1.055) ** 2.4,
      atol=2e-2,
  )


def test_hsv_premult_and_refract_tir() -> None:
  """Tests HSV <-> RGB, premult/unpremult, color4 alpha isolation, and refract TIR."""
  stage = Usd.Stage.CreateInMemory()

  _check_op(
      stage,
      'ND_rgbtohsv_color4',
      (13.0 / 30.0, 2.0 / 3.0, 0.75, 0.65),
      in_=(0.25, 0.75, 0.55, 0.65),
  )
  _check_op(
      stage,
      'ND_hsvtorgb_color4',
      (0.25, 0.75, 0.55, 0.4),
      in_=(13.0 / 30.0, 2.0 / 3.0, 0.75, 0.4),
  )
  _check_op(
      stage, 'ND_premult_color4', (0.2, 0.3, 0.4, 0.5), in_=(0.4, 0.6, 0.8, 0.5)
  )
  _check_op(
      stage,
      'ND_unpremult_color4',
      (0.8, 0.4, 0.2, 0.5),
      in_=(0.4, 0.2, 0.1, 0.5),
  )
  # `ND_convert_float_color4` propagates scalar into alpha, and a `place2d`
  # (`transformmatrix`) inside a 4-channel node's scalar `mix` input does not
  # clobber `fg.a` / `bg.a`.
  _check_op(
      stage,
      'ND_mix_color4',
      (0.4, 0.5, 0.6, 0.55),
      fg=('ND_convert_float_color4', {'in_': 0.8}),
      bg=(0.0, 0.2, 0.4, 0.3),
      mix=(
          'ND_extract_vector2',
          {
              'in_': (
                  'ND_place2d_vector2',
                  {'texcoord': (0.25, 0.5), 'offset': (-0.25, 0.0)},
              ),
              'index': 0,
          },
      ),
  )
  # Deterministic total internal reflection (`ior = 2.0` at grazing angle -> `(0, 0, 0)`).
  _check_op(
      stage,
      'ND_refract_vector3',
      (0.0, 0.0, 0.0),
      in_=Gf.Vec3f(0.8, 0.0, -0.6),
      normal=Gf.Vec3f(0.0, 0.0, 1.0),
      ior=2.0,
  )


def _parse_osl_input(val: str) -> Any:
  if val in ('true', 'false'):
    return val == 'true'
  if ',' in val:
    return tuple(float(x) for x in val.split(','))
  return float(val)


def test_osl_arithmetic_and_pattern_nodes() -> None:
  """Compares non-standard MaterialX pattern, color, and shader nodes against OSL reference values."""
  ref_path = (
      pathlib.Path(__file__).parent / 'materialx_reference/osl_reference.json'
  )
  reference = json.loads(ref_path.read_text())
  with mi.variant_context('scalar_rgb'):
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = mi.Vector3f(0.0, 0.0, 1.0)
    stage = Usd.Stage.CreateInMemory()
    failures = []
    for entry in reference['nodedefs']:
      nodedef = entry['nodedef']
      tol = 5e-3 if nodedef == 'ND_blackbody' else 1e-4
      first_inputs = {
          k: _parse_osl_input(v)
          for k, v in entry['samples'][0]['inputs'].items()
      }
      node = mtlx.define_shader(stage, 'Node', nodedef, **first_inputs)
      for sample in entry['samples']:
        for k, v in sample['inputs'].items():
          node.GetInput(k).Set(_parse_osl_input(v))
        for out_name, expected in sample['expected'].items():
          out = node.CreateOutput(out_name, Sdf.ValueTypeNames.Token)
          actual = mtlx.eval_emission_expr(
              out, stage=stage, si=si, channels=len(expected)
          )
          if not np.allclose(actual, expected, rtol=tol, atol=tol):
            failures.append(
                f'{nodedef}.{out_name}({sample["inputs"]}): '
                f'expected {expected}, got {actual}'
            )
            break
    assert not failures, 'OSL reference mismatches:\n' + '\n'.join(failures)


def test_expr_matrices() -> None:
  """Tests MaterialX 3x3 and 4x4 matrix algebra and transforms."""
  stage = Usd.Stage.CreateInMemory()

  # `creatematrix` + `transpose` + linear `transformmatrix_vector3`, plus 2D
  # (`vector2 * matrix33`) and 3D (`vector3 * matrix44`) affine transforms.
  m = (
      'ND_creatematrix_vector3_matrix33',
      {
          'in1': Gf.Vec3f(1.0, 2.0, 3.0),
          'in2': Gf.Vec3f(4.0, 5.0, 6.0),
          'in3': Gf.Vec3f(7.0, 8.0, 9.0),
      },
  )
  _check_op(
      stage,
      'ND_transformmatrix_vector3',
      (14.0, 32.0, 50.0),
      in_=Gf.Vec3f(1.0, 2.0, 3.0),
      mat=('ND_transpose_matrix33', {'in_': m}),
  )
  _check_op(
      stage,
      'ND_transformmatrix_vector2M3',
      (12.0, 16.0),
      in_=(2.0, 3.0),
      mat=Gf.Matrix3d(2.0, -1.0, 0.0, 1.0, 4.0, 0.0, 5.0, 6.0, 1.0),
  )
  _check_op(
      stage,
      'ND_transformmatrix_vector3M4',
      (9.0, 13.0, 21.0),
      in_=Gf.Vec3f(1.0, 2.0, 3.0),
      mat=Gf.Matrix4d(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 7, 7, 9, 1),
  )

  # 3x3 determinant, `divide` (`A * B^-1`), `add` (`FA`), `subtract`, and
  # `extract`.
  a3 = np.array([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [0.5, 1.5, 2.5]])
  b3 = np.array([[1.5, 0.5, -1.0], [-0.5, 2.0, 1.0], [1.0, -1.5, 3.0]])
  a3_gf, b3_gf = Gf.Matrix3d(*a3.flat), Gf.Matrix3d(*b3.flat)
  v3 = np.array([1.2, -0.7, 2.3])
  _check_op(stage, 'ND_determinant_matrix33', np.linalg.det(a3), in_=a3_gf)
  _check_op(
      stage,
      'ND_transformmatrix_vector3',
      v3 @ a3 @ np.linalg.inv(b3),
      in_=Gf.Vec3f(*v3),
      mat=('ND_divide_matrix33', {'in1': a3_gf, 'in2': b3_gf}),
  )
  _check_op(
      stage,
      'ND_extract_matrix33',
      (a3 + 1.25 - b3)[2],
      in_=(
          'ND_subtract_matrix33',
          {
              'in1': ('ND_add_matrix33FA', {'in1': a3_gf, 'in2': 1.25}),
              'in2': b3_gf,
          },
      ),
      index=2,
  )

  # 4x4 determinant, `divide` (`A * B^-1` via `transformmatrix_vector4`),
  # `creatematrix` (`vector4`), `subtract` (`FA`), dynamic `ifgreater`, and
  # `extract`.
  a4 = np.array([
      [2.0, 0.5, -1.0, 0.25],
      [-0.5, 3.0, 1.0, -1.5],
      [1.0, -1.0, 2.5, 0.5],
      [0.5, 1.5, -0.5, 2.0],
  ])
  b4 = np.array([
      [1.5, -0.5, 0.0, 1.0],
      [0.5, 2.0, -1.0, 0.5],
      [-1.0, 0.5, 3.0, -0.5],
      [0.25, -1.0, 0.5, 1.5],
  ])
  a4_gf, b4_gf = Gf.Matrix4d(*a4.flat), Gf.Matrix4d(*b4.flat)
  v4 = np.array([1.0, -2.0, 3.0, 1.5])
  _check_op(stage, 'ND_determinant_matrix44', np.linalg.det(a4), in_=a4_gf)
  _check_op(
      stage,
      'ND_transformmatrix_vector4',
      v4 @ a4 @ np.linalg.inv(b4),
      in_=tuple(v4.tolist()),
      mat=('ND_divide_matrix44', {'in1': a4_gf, 'in2': b4_gf}),
  )
  m4_from_rows = (
      'ND_creatematrix_vector4_matrix44',
      {f'in{i + 1}': tuple(a4[i].tolist()) for i in range(4)},
  )
  si2, px2, _, _ = _make_position_batch(
      stage, [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]
  )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_extract_matrix44',
              {
                  'in_': (
                      'ND_ifgreater_matrix44',
                      {
                          'value1': px2,
                          'value2': 0.5,
                          'in1': m4_from_rows,
                          'in2': (
                              'ND_subtract_matrix44FA',
                              {'in1': b4_gf, 'in2': 0.5},
                          ),
                      },
                  ),
                  'index': 2,
              },
          ),
          stage=stage,
          si=si2,
          channels=4,
      ),
      np.vstack([a4[2], (b4 - 0.5)[2]]),
      atol=1e-5,
  )

  # Singular matrix inversion returns zero without poisoning AD gradients.
  _, inv_em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      stage=stage,
      emission_color=(
          'ND_extract_matrix33',
          {
              'in_': (
                  'ND_invertmatrix_matrix33',
                  {
                      'in_': (
                          'ND_creatematrix_vector3_matrix33',
                          {
                              'in1': (
                                  'ND_position_vector3',
                                  {'space': 'world'},
                              ),
                              'in2': Gf.Vec3f(0.0, 1.0, 0.0),
                              'in3': Gf.Vec3f(0.0, 0.0, 1.0),
                          },
                      )
                  },
              ),
              'index': 0,
          },
      ),
  )
  assert inv_em is not None
  si_inv = dr.zeros(mi.SurfaceInteraction3f, 2)
  si_inv.wi = mi.Vector3f(0.0, 0.0, 1.0)
  p_x = mi.Float([0.0, 2.0])
  dr.enable_grad(p_x)
  si_inv.p = mi.Point3f(p_x, 0.0, 0.0)
  out_inv = inv_em.eval(si_inv)
  np.testing.assert_allclose(out_inv[0], [0.0, 0.5], atol=1e-5)
  dr.backward(dr.sum(out_inv))
  np.testing.assert_allclose(dr.grad(p_x), [0.0, -0.25], atol=1e-5)


def test_edf_and_emissive_surface_shaders(
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests EDF nodes, combinators, and surface shader emission models.

  Args:
    capfd: Pytest fd capture fixture.
  """
  stage = Usd.Stage.CreateInMemory()
  si = mtlx.make_test_si()

  fg_edf = (
      'ND_add_edf',
      {
          'in1': ('ND_uniform_edf', {'color': (0.2, 0.4, 0.6)}),
          'in2': ('ND_conical_edf', {'color': (0.8, 0.2, 0.4)}),
      },
  )
  bg_edf = (
      'ND_multiply_edfC',
      {
          'in1': ('ND_uniform_edf', {'color': (2.0, 1.0, 0.5)}),
          'in2': (0.5, 0.4, 0.8),
      },
  )
  for shader_id, expected, kw in [
      # `ND_generalized_schlick_edf` cosine-weighted hemispherical average with
      # `ND_surface` opacity: 0.5 * (3, 6, 9) * mix((0, 1, 0), (1, 0, 0), 2/6).
      (
          'ND_surface',
          (1.0, 1.0, 0.0),
          {
              'edf': (
                  'ND_generalized_schlick_edf',
                  {
                      'color0': (1.0, 0.0, 0.0),
                      'color90': (0.0, 1.0, 0.0),
                      'exponent': 1.0,
                      'base': ('ND_uniform_edf', {'color': (3.0, 6.0, 9.0)}),
                  },
              ),
              'opacity': 0.5,
          },
      ),
      # `ND_add_edf`, `ND_multiply_edfC`, `ND_mix_edf`, and `ND_conical_edf`.
      (
          'ND_surface',
          (1.0, 0.45, 0.55),
          {'edf': ('ND_mix_edf', {'fg': fg_edf, 'bg': bg_edf, 'mix': 0.25})},
      ),
      # `ND_mix_surfaceshader` blending OpenPBR and glTF PBR emissions.
      (
          'ND_mix_surfaceshader',
          (0.525, 1.1625, 1.8),
          {
              'fg': (
                  'ND_open_pbr_surface_surfaceshader',
                  {
                      'emission_luminance': 1.5,
                      'emission_color': (0.4, 0.6, 0.8),
                  },
              ),
              'bg': (
                  'ND_gltf_pbr_surfaceshader',
                  {'emissive': (0.2, 0.5, 0.8), 'emissive_strength': 2.5},
              ),
              'mix': 0.25,
          },
      ),
  ]:
    _, emitter = mtlx.compile_mtlx_material(shader_id, stage=stage, **kw)
    assert emitter is not None
    np.testing.assert_allclose(
        emitter.eval(si), mi.Color3f(*expected), atol=1e-5
    )

  # `ND_measured_edf` with a non-empty IES `file` warns and falls back to `color`.
  capfd.readouterr()
  _, measured_emitter = mtlx.compile_mtlx_material(
      'ND_surface',
      stage=stage,
      edf=(
          'ND_measured_edf',
          {'color': (0.3, 0.6, 0.9), 'file': Sdf.AssetPath('spot.ies')},
      ),
  )
  assert 'measured_edf' in capfd.readouterr().err
  assert measured_emitter is not None
  np.testing.assert_allclose(
      measured_emitter.eval(si), mi.Color3f(0.3, 0.6, 0.9), atol=1e-5
  )


def test_surface_unlit_opacity() -> None:
  """Tests ND_surface_unlit cutout opacity, transmission attenuation, and doubleSided."""
  stage, plane = mtlx.create_base_scene(
      with_lights=False, quad_half_size=0.8, resolution=(16, 16), spp=16
  )
  backdrop = mtlx.define_quad(stage, '/backdrop', half_size=2.0)
  backdrop.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.5))
  mtlx.bind_mtlx_shader(
      stage,
      backdrop,
      'ND_surface_unlit',
      emission=1.0,
      emission_color=(0.2, 0.8, 0.4),
  )

  # Point the front quad's normals away from the camera: single-sided area
  # emitters emit zero on the back face while still transmitting `(1 - opacity)`
  # of the backdrop, whereas `doubleSided` enables two-sided emission.
  plane.GetNormalsAttr().Set([(0.0, 0.0, -1.0)] * 4)
  surf = mtlx.bind_mtlx_shader(
      stage,
      plane,
      'ND_surface_unlit',
      emission=3.0,
      emission_color=(0.9, 0.2, 0.8),
      opacity=0.5,
  )
  engine = mtlx.create_engine(stage)
  np.testing.assert_allclose(
      mtlx.render_center(engine), (0.1, 0.4, 0.2), rtol=0.05, atol=0.02
  )
  plane.CreateDoubleSidedAttr(True)
  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_surface_unlit_opacity',
      engine=engine,
      spp=16,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(image_hd),
      (1.45, 0.7, 1.4),
      rtol=0.05,
      atol=0.03,
  )
  # `transmission` attenuates `surface_unlit` emission by `(1 - transmission)`.
  surf.CreateInput('transmission', Sdf.ValueTypeNames.Float).Set(0.25)
  np.testing.assert_allclose(
      mtlx.render_center(engine),
      0.75 * np.array([1.35, 0.3, 1.2]) + np.array([0.1, 0.4, 0.2]),
      rtol=0.05,
      atol=0.03,
  )


@pytest.mark.parametrize(
    'variants',
    [('cuda_ad_rgb', 'llvm_ad_rgb'), ('cuda_ad_spectral', 'llvm_ad_spectral')],
)
def test_dynamic_material_edits_and_spectral_freezing(
    variants: tuple[str, str],
) -> None:
  """Tests dynamic Hydra MaterialX edits and FrozenRender replay (including spectral).

  Args:
    variants: Candidate Mitsuba variants to test.
  """
  mtlx.set_variant_or_skip(*variants)
  stage, mesh = mtlx.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  c4 = mtlx.define_shader(
      stage,
      '/mesh/Material/C4',
      'ND_constant_color4',
      value=(0.05, 0.1, 0.425, 2.0),
  )
  shader = mtlx.bind_mtlx_shader(
      stage,
      mesh,
      'ND_standard_surface_surfaceshader',
      base=0.0,
      emission=('ND_extract_color4', {'in_': c4, 'index': 3}),
      emission_color=('ND_convert_color4_color3', {'in_': c4}),
      opacity=(0.5, 0.5, 0.5),
  )

  engine = mtlx.create_engine(stage)
  np.testing.assert_allclose(
      mtlx.render_center(engine),
      (0.05, 0.1, 0.425),
      rtol=0.05,
      atol=0.02,
  )
  c4.GetInput('value').Set((0.4, 0.15, 0.05, 2.0))
  shader.GetInput('opacity').Set((1.0, 1.0, 1.0))
  np.testing.assert_allclose(
      mtlx.render_center(engine), (0.8, 0.3, 0.1), rtol=0.05, atol=0.03
  )

  # `mitsuba:use_kernel_freezing` records and replays across frames (including
  # in spectral variants, exercising `MtlxExprTexture`'s sRGB LUT traversal).
  rs = stage.GetPrimAtPath(stage.GetMetadata('renderSettingsPrimPath'))
  rs.CreateAttribute(
      'mitsuba:use_kernel_freezing', Sdf.ValueTypeNames.Bool
  ).Set(True)
  engine.render()  # Warm-up pass
  engine.render()  # Recording pass
  with dr.kernel_history() as frozen_history:
    frozen_center = mtlx.render_center(engine)
  modes = {k.recording_mode for k in frozen_history}
  assert dr.KernelRecordingMode.Replayed in modes, frozen_history
  assert dr.KernelRecordingMode.Recorded not in modes, frozen_history
  np.testing.assert_allclose(
      frozen_center, (0.8, 0.3, 0.1), rtol=0.05, atol=0.03
  )


def test_spectral_upsampling_matches_mitsuba() -> None:
  """Verifies spectral upsampling matches Mitsuba's own D65/SRGBModel evaluation."""
  mtlx.set_variant_or_skip('cuda_ad_spectral', 'llvm_ad_spectral')
  _, mtlx_em = mtlx.compile_mtlx_material(
      'ND_surface_unlit', emission_color=(0.2, 0.4, 0.6)
  )
  assert mtlx_em is not None
  ref_em = mi.load_dict(
      {'type': 'area', 'radiance': {'type': 'd65', 'color': [0.2, 0.4, 0.6]}}
  )
  mtlx_params = mi.traverse(mtlx_em)
  ref_params = mi.traverse(ref_em)

  si = dr.zeros(mi.SurfaceInteraction3f, 16)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.wavelengths = mi.Spectrum(np.linspace(360.0, 830.0, 64).reshape(4, 16))

  for rgb in [
      (0.0, 0.0, 0.0),
      (0.2, 0.4, 0.6),
      (1.8, 0.4, 0.9),
      (0.1, 2.3, 0.05),
      (0.35, 0.35, 0.35),
  ]:
    for i in range(3):
      mtlx_params[f'radiance.nested_texture.param{i}'] = mi.Float(rgb[i])
    mtlx_params.update()
    ref_params['radiance.value'] = mi.Color3f(*rgb)
    ref_params.update()
    np.testing.assert_allclose(
        mtlx_em.eval(si), ref_em.eval(si), rtol=1e-5, atol=1e-6
    )

  # Also check the `result_uniform` scalar emission path against neutral D65.
  _, gray_em = mtlx.compile_mtlx_material('ND_surface_unlit', emission=0.35)
  assert gray_em is not None
  np.testing.assert_allclose(
      gray_em.eval(si), ref_em.eval(si), rtol=1e-5, atol=1e-6
  )


def test_mi_traverse_emitter_param_update() -> None:
  """Verifies mi.traverse updates, param order, size-1 validation, and @dr.freeze state isolation."""
  _, c4_em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=(
          'ND_convert_color4_color3',
          {'in_': ('ND_constant_color4', {'value': (0.2, 0.4, 0.6, 0.8)})},
      ),
  )
  assert c4_em is not None
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  params = mi.traverse(c4_em)
  np.testing.assert_allclose(
      [float(params[f'radiance.param{i}'][0]) for i in range(4)],
      [0.2, 0.4, 0.6, 0.8],
  )
  params['radiance.param0'] = mi.Float(0.8)
  params['radiance.param2'] = mi.Float(0.1)
  params.update()
  np.testing.assert_allclose(
      c4_em.eval(si), mi.Color3f(0.8, 0.4, 0.1), rtol=1e-4
  )

  with pytest.raises(RuntimeError, match='size 1'):
    params['radiance.param0'] = mi.Float([0.1, 0.2])
    params.update()

  # `@dr.freeze` must distinguish two `MtlxExprTexture` instances with identical
  # parameter counts and different bytecodes via `traverse_state`.
  _, em_lr = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=(
          'ND_ramplr_color3',
          {'valuel': (0.0, 0.0, 0.0), 'valuer': (1.0, 0.0, 0.0)},
      ),
  )
  _, em_tb = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=(
          'ND_ramptb_color3',
          {'valuet': (0.0, 0.0, 0.0), 'valueb': (1.0, 0.0, 0.0)},
      ),
  )

  @dr.freeze
  def eval_frozen(em: mi.Emitter, uv: mi.Point2f) -> mi.Spectrum:
    si_f = dr.zeros(mi.SurfaceInteraction3f, 2)
    si_f.wi = mi.Vector3f(0.0, 0.0, 1.0)
    si_f.uv = uv
    return em.eval(si_f)

  uv = mi.Point2f([0.9, 0.1], [0.5, 0.5])
  for _ in range(3):
    out_lr = eval_frozen(em_lr, uv)
  out_tb = eval_frozen(em_tb, uv)
  np.testing.assert_allclose(out_lr[0], [0.9, 0.1], atol=1e-5)
  np.testing.assert_allclose(out_tb[0], [0.5, 0.5], atol=1e-5)


def test_mtlx_surface_precedence_and_preview_emissive_edit() -> None:
  """Verifies UsdPreviewSurface emissive edits and mtlx:surface precedence."""
  stage, quad = mtlx.create_base_scene(
      with_lights=False, quad_half_size=0.8, resolution=(16, 16), spp=4
  )
  material = UsdShade.Material.Define(stage, '/mesh/Material')
  preview = mtlx.define_shader(
      stage,
      'Preview',
      'UsdPreviewSurface',
      diffuseColor=(0.0, 0.0, 0.0),
      emissiveColor=(0.1, 0.8, 0.2),
  )
  material.CreateSurfaceOutput().ConnectToSource(
      preview.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )
  UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(material)

  engine = mtlx.create_engine(stage)
  img_preview, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_preview_emissive',
      engine=engine,
      spp=4,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_preview), (0.1, 0.8, 0.2), atol=2e-2
  )
  # Dynamic `UsdPreviewSurface.emissiveColor` edit in Hydra must rebuild the
  # emitter.
  preview.GetInput('emissiveColor').Set((0.2, 0.3, 0.9))
  np.testing.assert_allclose(
      mtlx.render_center(engine), (0.2, 0.3, 0.9), atol=2e-2
  )

  # Connecting an `mtlx:surface` terminal takes precedence over the universal
  # `UsdPreviewSurface` output.
  mtlx.bind_mtlx_shader(
      stage, quad, 'ND_surface_unlit', emission_color=(0.9, 0.1, 0.3)
  )
  img_mtlx, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_surface_precedence',
      spp=4,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_mtlx), (0.9, 0.1, 0.3), atol=2e-2
  )


def test_multi_shader_dedup_and_frozen_structural_edit() -> None:
  """Verifies distinct MaterialX bytecodes are not conflated by vcall deduplication or FrozenRender."""
  stage, left = mtlx.create_base_scene(
      with_lights=False,
      quad_half_size=0.45,
      resolution=(32, 32),
      spp=4,
      mesh_path='/left',
  )
  left.AddTranslateOp().Set(Gf.Vec3d(-0.5, 0.0, 0.0))
  right = mtlx.define_quad(stage, '/right', half_size=0.45)
  right.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))

  # Left quad ramps horizontally; right quad ramps vertically with identical
  # opaque value count so Dr.Jit CallDeduplication must distinguish `m_prog.code`.
  mtlx.bind_mtlx_shader(
      stage,
      left,
      'ND_surface_unlit',
      emission_color=(
          'ND_ramplr_color3',
          {'valuel': (0.0, 0.0, 1.0), 'valuer': (1.0, 0.0, 0.0)},
      ),
  )
  mtlx.bind_mtlx_shader(
      stage,
      right,
      'ND_surface_unlit',
      emission_color=(
          'ND_ramptb_color3',
          {'valuet': (0.0, 1.0, 0.0), 'valueb': (1.0, 1.0, 0.0)},
      ),
  )
  rs = stage.GetPrimAtPath(stage.GetMetadata('renderSettingsPrimPath'))
  rs.CreateAttribute(
      'mitsuba:use_kernel_freezing', Sdf.ValueTypeNames.Bool
  ).Set(True)

  engine = mtlx.create_engine(stage)
  img1 = np.array(engine.render()['color'])[..., :3]
  engine.render()  # Recording pass
  engine.render()  # Replay pass before structural edit
  assert abs(img1[16, 5, 0] - img1[16, 11, 0]) > 0.2
  assert abs(img1[10, 8, 0] - img1[21, 8, 0]) < 0.05
  assert abs(img1[10, 24, 0] - img1[21, 24, 0]) > 0.2
  assert abs(img1[16, 21, 0] - img1[16, 27, 0]) < 0.05

  # Structurally swap the left quad from `ramplr` to `ramptb` (same opaque
  # layout, different bytecode) while kernel freezing is active.
  mtlx.bind_mtlx_shader(
      stage,
      left,
      'ND_surface_unlit',
      emission_color=(
          'ND_ramptb_color3',
          {'valuet': (0.0, 0.0, 1.0), 'valueb': (1.0, 0.0, 0.0)},
      ),
  )
  img2 = np.array(engine.render()['color'])[..., :3]
  assert abs(img2[10, 8, 0] - img2[21, 8, 0]) > 0.2
  assert abs(img2[16, 5, 0] - img2[16, 11, 0]) < 0.05


def test_mixed_context_and_dynamic_material_switch() -> None:
  """Verifies Hydra updates BSDFs when editing mixed-context or dynamically switched materials."""
  stage, quad = mtlx.create_base_scene(
      with_lights=True, quad_half_size=0.8, resolution=(16, 16), spp=8
  )
  mtlx.bind_mtlx_shader(stage, quad, 'ND_surface_unlit', emission=0.0)
  mi_diff = mtlx.define_shader(
      stage, 'MiDiffuse', 'mitsuba_diffuse', reflectance=(0.9, 0.05, 0.05)
  )
  mat = UsdShade.Material.Get(stage, '/mesh/Material')
  mat.CreateSurfaceOutput('mitsuba').ConnectToSource(
      mi_diff.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )

  engine = mtlx.create_engine(stage)
  c_red = mtlx.render_center(engine)
  assert c_red[0] > c_red[1] * 3.0

  # Editing `mitsuba_diffuse` on a material that also has `outputs:mtlx:surface`
  # must update Hydra's active BSDF.
  mi_diff.GetInput('reflectance').Set((0.05, 0.9, 0.05))
  c_green = mtlx.render_center(engine)
  assert c_green[1] > c_green[0] * 3.0

  # Clearing `outputs:mitsuba:surface` dynamically switches the material to
  # `outputs:mtlx:surface`, replacing `mitsuba_diffuse` with black `diffuse(0)`.
  mat.GetSurfaceOutput('mitsuba').DisconnectSource()
  np.testing.assert_allclose(
      mtlx.render_center(engine), (0.0, 0.0, 0.0), atol=1e-3
  )


def test_constant_folding_and_opaque_compaction() -> None:
  """Tests unauthored const baking, independent shader params, and opaque compaction."""
  stage = Usd.Stage.CreateInMemory()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))

  bsdf_def, _ = mtlx.compile_mtlx_material(
      'ND_standard_surface_surfaceshader', stage=stage
  )
  assert bsdf_def is not None
  assert _num_params(bsdf_def, 'param') == 0

  a = mtlx.define_shader(stage, 'A', 'ND_constant_float', value=0.25)
  b = mtlx.define_shader(stage, 'B', 'ND_constant_float', value=0.25)
  _, em_pair = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=('ND_combine3_color3', {'in1': a, 'in2': b, 'in3': 0.0}),
      stage=stage,
  )
  assert em_pair is not None
  assert _num_params(em_pair) == 2

  # Dead `* 0` opaques (indices 0..2) are pruned and live `fg` opaques remap
  # from 3..5 -> 0..2; `mix` requires `FoldExpr` to fold `2.0 / 2.0 -> 1.0`.
  two = ('ND_convert_integer_float', {'in_': 2})
  _, em = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission_color=(
          'ND_add_color3',
          {
              'in1': (
                  'ND_multiply_color3FA',
                  {'in1': (0.8, 0.9, 0.2), 'in2': 0.0},
              ),
              'in2': (
                  'ND_mix_color3',
                  {
                      'fg': (0.2, 0.3, 0.4),
                      'bg': (0.5, 0.6, 0.7),
                      'mix': ('ND_divide_float', {'in1': two, 'in2': two}),
                  },
              ),
          },
      ),
      stage=stage,
  )
  assert em is not None
  assert _num_params(em) == 3
  np.testing.assert_allclose(em.eval(si), mi.Color3f(0.2, 0.3, 0.4), atol=1e-5)

  # Widening an equal non-zero `vector2` (both folded constants and non-uniform
  # ops like a missing `geompropvalue`) to `vector3` zero-pads `.z`, and
  # `combine2(0, 0)` canonicalizes to zero and prunes dead operands.
  half = (
      'ND_divide_float',
      {'in1': ('ND_convert_integer_float', {'in_': 1}), 'in2': two},
  )
  for v2_expr in [
      (
          'ND_add_vector2',
          {
              'in1': ('ND_combine2_vector2', {'in1': half, 'in2': half}),
              'in2': (
                  'ND_multiply_vector2',
                  {
                      'in1': ('ND_combine2_vector2', {'in1': 0.0, 'in2': 0.0}),
                      'in2': (0.7, 0.8),
                  },
              ),
          },
      ),
      (
          'ND_geompropvalue_vector2',
          {
              'geomprop': 'missing_vec2',
              'default': ('ND_combine2_vector2', {'in1': half, 'in2': half}),
          },
      ),
  ]:
    _, em_v2 = mtlx.compile_mtlx_material(
        'ND_surface_unlit',
        emission_color=(
            'ND_convert_vector3_color3',
            {'in_': ('ND_convert_vector2_vector3', {'in_': v2_expr})},
        ),
        stage=stage,
    )
    assert em_v2 is not None
    assert _num_params(em_v2) == 0
    np.testing.assert_allclose(
        em_v2.eval(si), mi.Color3f(0.5, 0.5, 0.0), atol=1e-5
    )

  # Folded `1 - saturate(transmission)`, `saturate(mix)`, and float32
  # overflow guards prune dead branches while preserving float32 runtime results.
  _, em_trans = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      transmission=1.0,
      emission_color=(0.3, 0.4, 0.5),
      stage=stage,
  )
  assert em_trans is None
  bsdf_mix, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      fg=(
          'ND_standard_surface_surfaceshader',
          {
              'transmission': 1.0,
              'specular_roughness': 0.0,
              'opacity': (1.0, 1.0, 1.0),
          },
      ),
      bg=(
          'ND_standard_surface_surfaceshader',
          {'thin_walled': True, 'opacity': (0.5, 0.5, 0.5)},
      ),
      mix=('ND_add_float', {'in1': two, 'in2': two}),
      stage=stage,
  )
  assert bsdf_mix is not None
  assert not mi.has_flag(bsdf_mix.flags(), mi.BSDFFlags.Null)
  assert _num_params(bsdf_mix, 'param') == 0
  bs_tir, _ = bsdf_mix.sample(
      mi.BSDFContext(),
      mtlx.make_test_si(wi=(0.8, 0.0, -0.6)),
      mi.Float(0.99),
      mi.Point2f(0.5, 0.5),
  )
  assert dr.all(bs_tir.wo.z < 0.0)

  exp100 = ('ND_exp_float', {'in_': ('ND_convert_integer_float', {'in_': 100})})
  exp_m70 = (
      'ND_exp_float',
      {'in_': ('ND_convert_integer_float', {'in_': -70})},
  )
  exp50 = ('ND_exp_float', {'in_': ('ND_convert_integer_float', {'in_': 50})})
  _, em_ovf = mtlx.compile_mtlx_material(
      'ND_surface_unlit',
      emission=(
          'ND_ifgreater_float',
          {
              'value1': (
                  'ND_multiply_float',
                  {'in1': exp100, 'in2': exp_m70},
              ),
              'value2': exp50,
              'in1': 1.0,
              'in2': 0.0,
          },
      ),
      stage=stage,
  )
  assert em_ovf is not None
  np.testing.assert_allclose(em_ovf.eval(si), mi.Color3f(1.0), atol=1e-5)


def _check_freeze_update_and_ad(
    expr: tuple[str, dict[str, Any]],
    si: mi.SurfaceInteraction3f,
    param_init: float,
    param_new: float,
    expected0: Any,
    expected1: Any,
    expected_grad: float,
    rtol: float = 1e-5,
) -> mi.Emitter:
  em = _compile_emission(expr)
  np.testing.assert_allclose(em.eval(si), expected0, atol=1e-5)

  @dr.freeze
  def eval_frozen(s: mi.SurfaceInteraction3f) -> mi.Color3f:
    return em.eval(s)

  np.testing.assert_allclose(eval_frozen(si), expected0, atol=1e-5)
  np.testing.assert_allclose(eval_frozen(si), expected0, atol=1e-5)

  params = mi.traverse(em)
  key = next(
      k
      for k in params.keys()
      if k.startswith('radiance.param') and np.allclose(params[k], param_init)
  )
  params[key] = mi.Float(param_new)
  params.update()
  np.testing.assert_allclose(eval_frozen(si), expected1, atol=1e-5)

  dr.enable_grad(params[key])
  params.update()
  dr.backward(dr.sum(em.eval(si).x))
  np.testing.assert_allclose(dr.grad(params[key]), expected_grad, rtol=rtol)
  return em


def test_invariant_hoisting_freeze_and_ad() -> None:
  """Verifies parameter updates, `@dr.freeze`, and AD through hoisted temporaries."""
  u = [0.15, 0.55, 0.85]
  si_uv, u_node, _ = _make_uv_batch(u)
  base_rgb = mi.Color3f(u, 0.2, 0.4)
  _check_freeze_update_and_ad(
      (
          'ND_colorcorrect_color3',
          {
              'in_': (
                  'ND_combine3_color3',
                  {'in1': u_node, 'in2': 0.2, 'in3': 0.4},
              ),
              'exposure': 1.5,
          },
      ),
      si_uv,
      1.5,
      2.0,
      base_rgb * 2.0**1.5,
      base_rgb * 4.0,
      sum(u) * 4.0 * math.log(2.0),
  )

  si_p = dr.zeros(mi.SurfaceInteraction3f, 2)
  si_p.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si_p.p = mi.Point3f([2.0, 4.0], [3.0, 6.0], [4.0, 8.0])
  _check_freeze_update_and_ad(
      (
          'ND_convert_vector3_color3',
          {
              'in_': (
                  'ND_transformmatrix_vector3',
                  {
                      'in_': ('ND_position_vector3', {'space': 'world'}),
                      'mat': (
                          'ND_invertmatrix_matrix33',
                          {
                              'in_': Gf.Matrix3d(
                                  2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0
                              )
                          },
                      ),
                  },
              )
          },
      ),
      si_p,
      2.0,
      4.0,
      mi.Color3f([1.0, 2.0], [1.0, 2.0], [1.0, 2.0]),
      mi.Color3f([0.5, 1.0], [1.0, 2.0], [1.0, 2.0]),
      -0.375,
  )


def test_dead_code_and_subprogram_pruning() -> None:
  """Verifies dead temporary pruning in main bytecode and HeightToNormal subprograms."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  si, u_node, _ = _make_uv_batch(u)
  dead_term = (
      'ND_multiply_float',
      {
          'in1': ('ND_add_float', {'in1': u_node, 'in2': 0.25}),
          'in2': ('ND_divide_float', {'in1': 0.0, 'in2': u_node}),
      },
  )
  em_base = _compile_float_emission(u_node)
  em_pruned = _compile_float_emission(
      ('ND_add_float', {'in1': u_node, 'in2': dead_term})
  )
  assert _num_params(em_pruned) == 0
  assert _code_size(em_pruned) == _code_size(em_base)
  np.testing.assert_allclose(em_pruned.eval(si), em_base.eval(si), atol=1e-6)

  em_h2n_clean = _compile_emission((
      'ND_convert_vector3_color3',
      {'in_': ('ND_heighttonormal_vector3', {'in_': u_node})},
  ))
  em_h2n_dead = _compile_emission((
      'ND_convert_vector3_color3',
      {
          'in_': (
              'ND_heighttonormal_vector3',
              {'in_': ('ND_add_float', {'in1': u_node, 'in2': dead_term})},
          )
      },
  ))
  assert _num_params(em_h2n_dead) == 0
  np.testing.assert_allclose(
      em_h2n_dead.eval(si), em_h2n_clean.eval(si), atol=1e-6
  )

  # Nested `heighttonormal` must keep the callee subprogram live.
  si_uv2, u2, v2 = _make_uv_batch([0.25, 0.75], [0.5, 0.75])
  h2n_inner = (
      'ND_heighttonormal_vector3',
      {'in_': ('ND_multiply_float', {'in1': u2, 'in2': v2}), 'scale': 10.0},
  )
  h2n_outer = (
      'ND_heighttonormal_vector3',
      {
          'in_': ('ND_extract_vector3', {'in_': h2n_inner, 'index': 0}),
          'scale': 20.0,
      },
  )
  em_nested = _compile_emission(
      ('ND_convert_vector3_color3', {'in_': h2n_outer})
  )
  np.testing.assert_allclose(
      em_nested.eval(si_uv2),
      [
          [0.46243557, 0.5],
          [0.67793226, 0.46303481],
          [0.96575642, 0.99863172],
      ],
      atol=1e-4,
  )


def test_affine_chain_collapse_freeze_and_ad() -> None:
  """Verifies chained affine collapse across `@dr.freeze`, updates, and AD."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  si, u_node, _ = _make_uv_batch(u)
  inner = (
      'ND_add_float',
      {
          'in1': (
              'ND_divide_float',
              {
                  'in1': (
                      'ND_multiply_float',
                      {'in1': u_node, 'in2': 3.0},
                  ),
                  'in2': 1.5,
              },
          ),
          'in2': 0.25,
      },
  )

  def make_chain(remapped_in: tuple[str, dict[str, Any]]) -> Any:
    return (
        'ND_mix_color3',
        {
            'fg': (0.9, 0.4, 0.2),
            'bg': (0.1, 0.6, 0.8),
            'mix': (
                'ND_invert_float',
                {
                    'amount': 1.2,
                    'in_': (
                        'ND_remap_float',
                        {
                            'in_': remapped_in,
                            'inlow': 0.25,
                            'inhigh': 2.25,
                            'outlow': 0.1,
                            'outhigh': 0.9,
                        },
                    ),
                },
            ),
        },
    )

  def expected_chain(s1: float, s2: float) -> np.ndarray:
    t = 0.1 + (((u * s1 / s2 + 0.25) - 0.25) / 2.0) * 0.8
    m = 1.2 - t
    fg = np.array([0.9, 0.4, 0.2])
    bg = np.array([0.1, 0.6, 0.8])
    return bg[:, None] + m * (fg - bg)[:, None]

  em_aff = _check_freeze_update_and_ad(
      make_chain(inner),
      si,
      1.5,
      2.5,
      expected_chain(3.0, 1.5),
      expected_chain(3.0, 2.5),
      # Red channel slope in `m`: `(0.9 - 0.1) = 0.8`; `dm/ds2 = +0.4 * u * 3 / s2^2`.
      float(np.sum(0.8 * 0.4 * u * 3.0 / (2.5**2))),
      rtol=1e-4,
  )
  em_uncollapsed = _compile_emission(
      make_chain(('ND_absval_float', {'in_': inner}))
  )
  assert _code_size(em_aff) < _code_size(em_uncollapsed)


def test_screen_and_clip01_fast_paths() -> None:
  """Verifies `screen` FMA/folding, `Clip01` elision, and `tmp_clipped01_` alignment."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  v = np.array([0.2, 0.7, 0.15, 0.9])
  si, u_node, v_node = _make_uv_batch(u, v)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          ('ND_screen_float', {'fg': u_node, 'bg': v_node}),
          si=si,
          channels=1,
      ),
      u + v - u * v,
      atol=1e-6,
  )
  em_screen_one = _compile_emission((
      'ND_screen_color3',
      {'fg': (1.0, 1.0, 1.0), 'bg': (0.2, 0.4, 0.6)},
  ))
  assert _num_params(em_screen_one) == 0
  np.testing.assert_allclose(em_screen_one.eval(si), mi.Color3f(1.0), atol=1e-6)

  # Matrix ops must keep `tmp_clipped01_` aligned for subsequent `clamp(0, 1)`.
  stage = Usd.Stage.CreateInMemory()
  mat_t = mtlx.define_shader(
      stage,
      'MatT',
      'ND_transpose_matrix33',
      in_=('ND_creatematrix_vector3_matrix33', {'in1': (2.0, 0.0, 0.0)}),
  )
  row0 = ('ND_extract_matrix33', {'in_': mat_t, 'index': 0})
  m00 = ('ND_extract_vector3', {'in_': row0, 'index': 0})
  clamped_mat = (
      'ND_clamp_float',
      {
          'in_': ('ND_add_float', {'in1': u_node, 'in2': m00}),
          'low': 0.0,
          'high': 1.0,
      },
  )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(clamped_mat, stage=stage, si=si, channels=1),
      np.ones(len(u)),
      atol=1e-6,
  )

  # Multi-channel constants in [0, 1] and repeated `clamp(0, 1)` elide `Op::Clip`.
  c01 = (
      'ND_combine3_color3',
      {'in1': _const_ratio(1, 2), 'in2': 0.0, 'in3': 1.0},
  )
  em_c01 = _compile_emission(c01)
  em_c01_clamped = _compile_emission((
      'ND_clamp_color3',
      {'in_': c01, 'low': (0.0, 0.0, 0.0), 'high': (1.0, 1.0, 1.0)},
  ))
  assert _code_size(em_c01_clamped) == _code_size(em_c01)


def test_artistic_ior_fast_paths() -> None:
  """Verifies `artistic_ior` exact fast paths (`edge_color` 0, 1, intermediate)."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  si, u_node, _ = _make_uv_batch(u)
  stage = Usd.Stage.CreateInMemory()

  def compile_ior(ec: float) -> tuple[mi.Emitter | None, mi.Emitter | None]:
    ior_node = mtlx.define_shader(
        stage,
        f'ArtIor_{int(ec * 100)}',
        'ND_artistic_ior',
        reflectivity=('ND_convert_float_color3', {'in_': u_node}),
        edge_color=(ec, ec, ec),
    )
    out_ior = ior_node.CreateOutput('ior', Sdf.ValueTypeNames.Color3f)
    out_ext = ior_node.CreateOutput('extinction', Sdf.ValueTypeNames.Color3f)
    _, em_i = mtlx.compile_mtlx_material(
        'ND_surface_unlit', stage=stage, emission_color=out_ior
    )
    _, em_e = mtlx.compile_mtlx_material(
        'ND_surface_unlit', stage=stage, emission_color=out_ext
    )
    return em_i, em_e

  em_mid_ior, em_mid_ext = compile_ior(0.35)
  assert em_mid_ior is not None and em_mid_ext is not None
  # Concrete values at u=[0.1, 0.35, 0.6, 0.85], edge_c=0.35.
  np.testing.assert_allclose(
      em_mid_ior.eval(si)[0],
      [1.5375815, 2.7017345, 5.204939, 16.035316],
      rtol=1e-5,
      atol=1e-5,
  )
  np.testing.assert_allclose(
      em_mid_ext.eval(si)[0],
      [0.62799335, 1.7097427, 3.6807773, 11.722049],
      rtol=1e-5,
      atol=1e-5,
  )

  em_ior0, em_ext0 = compile_ior(0.0)
  assert em_ior0 is not None and em_ext0 is None
  assert _code_size(em_ior0) < _code_size(em_mid_ior)
  np.testing.assert_allclose(
      em_ior0.eval(si)[0],
      (1.0 + np.sqrt(u)) / (1.0 - np.sqrt(u)),
      rtol=1e-5,
      atol=1e-5,
  )

  em_ior1, em_ext1 = compile_ior(1.0)
  assert em_ior1 is not None and em_ext1 is not None
  assert _code_size(em_ior1) < _code_size(em_mid_ior)
  assert _code_size(em_ext1) < _code_size(em_mid_ext)
  np.testing.assert_allclose(
      em_ior1.eval(si)[0], (1.0 - u) / (1.0 + u), rtol=1e-5, atol=1e-5
  )
  np.testing.assert_allclose(
      em_ext1.eval(si)[0],
      2.0 * np.sqrt(u) / (1.0 + u),
      rtol=1e-5,
      atol=1e-5,
  )


def test_range_smoothstep_and_if_fast_paths() -> None:
  """Verifies unit-gamma `range`, `smoothstep`, and `ifgreater` fast paths."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  si, u_node, _ = _make_uv_batch(u)
  gamma_15 = _const_ratio(3, 2)
  range_args = {
      'in_': u_node,
      'inlow': 0.1,
      'inhigh': 0.9,
      'outlow': -0.5,
      'outhigh': 1.5,
  }
  t_lin = -0.5 + (u - 0.1) / 0.8 * 2.0
  for doclamp, expected in [
      (False, t_lin),
      (True, np.clip(t_lin, -0.5, 1.5)),
  ]:
    em_fast = _compile_float_emission(
        ('ND_range_float', {**range_args, 'doclamp': doclamp})
    )
    em_slow = _compile_float_emission((
        'ND_range_float',
        {**range_args, 'gamma': gamma_15, 'doclamp': doclamp},
    ))
    assert _code_size(em_fast) < _code_size(em_slow)
    np.testing.assert_allclose(em_fast.eval(si)[0], expected, atol=1e-5)

  # Pre-NodeGraph `smoothstep` keeps multi-channel `color3` vectorized and
  # collapses `(x - lo) / (hi - lo)` via Affine.
  ss_c3 = (
      'ND_smoothstep_color3',
      {
          'in_': (
              'ND_combine3_color3',
              {'in1': u_node, 'in2': 0.3, 'in3': 0.7},
          ),
          'low': (0.2, 0.2, 0.2),
          'high': (0.8, 0.8, 0.8),
      },
  )
  ss_sep = (
      'ND_combine3_color3',
      {
          'in1': (
              'ND_smoothstep_float',
              {'in_': u_node, 'low': 0.2, 'high': 0.8},
          ),
          'in2': ('ND_smoothstep_float', {'in_': 0.3, 'low': 0.2, 'high': 0.8}),
          'in3': ('ND_smoothstep_float', {'in_': 0.7, 'low': 0.2, 'high': 0.8}),
      },
  )
  em_ss_c3 = _compile_emission(ss_c3)
  em_ss_sep = _compile_emission(ss_sep)
  assert _code_size(em_ss_c3) < _code_size(em_ss_sep)
  vals = np.stack([u, np.full_like(u, 0.3), np.full_like(u, 0.7)])
  ts = np.clip((vals - 0.2) / 0.6, 0.0, 1.0)
  np.testing.assert_allclose(
      em_ss_c3.eval(si), ts * ts * (3.0 - 2.0 * ts), atol=1e-5
  )

  # `ifgreater` / `ifgreatereq` with (1, 0) and (0, 1) omit `Op::Select`.
  for cat, in1, in2, expected in [
      ('ND_ifgreater_float', 1.0, 0.0, u > 0.5),
      ('ND_ifgreater_float', 0.0, 1.0, u <= 0.5),
      ('ND_ifgreatereq_float', 1.0, 0.0, u >= 0.5),
      ('ND_ifgreatereq_float', 0.0, 1.0, u < 0.5),
  ]:
    em_if = _compile_float_emission(
        (cat, {'value1': u_node, 'value2': 0.5, 'in1': in1, 'in2': in2})
    )
    em_sel = _compile_float_emission(
        (cat, {'value1': u_node, 'value2': 0.5, 'in1': 1.0, 'in2': 0.5})
    )
    assert _code_size(em_if) < _code_size(em_sel)
    np.testing.assert_allclose(em_if.eval(si)[0], expected, atol=1e-6)


@pytest.mark.parametrize(
    ('node_id', 'in_type', 'extra_inputs'),
    [
        pytest.param('ND_hsvadjust_color3', 'color3', {}, id='hsvadjust'),
        pytest.param(
            'ND_contrast_color3',
            'color3',
            {'amount': (1.0, 1.0, 1.0), 'pivot': (0.3, 0.4, 0.5)},
            id='contrast',
        ),
        pytest.param(
            'ND_rotate2d_vector2', 'vector2', {'amount': 0.0}, id='rotate2d'
        ),
    ],
)
def test_identity_fast_paths(
    node_id: str, in_type: str, extra_inputs: dict[str, Any]
) -> None:
  """Verifies identity stdlib fast paths elide ops and unused opaque inputs."""
  u = np.array([0.1, 0.35, 0.6, 0.85])
  v = np.array([0.2, 0.7, 0.15, 0.9])
  si, u_node, v_node = _make_uv_batch(u, v)
  if in_type == 'vector2':
    base_expr: Any = ('ND_texcoord_vector2', {})
    em_in: Any = (
        'ND_convert_vector3_color3',
        {'in_': ('ND_convert_vector2_vector3', {'in_': base_expr})},
    )
    fast_in: Any = (
        'ND_convert_vector3_color3',
        {
            'in_': (
                'ND_convert_vector2_vector3',
                {'in_': (node_id, {'in_': base_expr, **extra_inputs})},
            )
        },
    )
  else:
    base_expr = (
        'ND_combine3_color3',
        {'in1': u_node, 'in2': v_node, 'in3': 0.4},
    )
    em_in = base_expr
    fast_in = (node_id, {'in_': base_expr, **extra_inputs})
  em_base = _compile_emission(em_in)
  em_fast = _compile_emission(fast_in)
  assert _num_params(em_fast) == _num_params(em_base)
  assert _code_size(em_fast) == _code_size(em_base)
  np.testing.assert_allclose(em_fast.eval(si), em_base.eval(si), atol=1e-5)


def test_power_fast_paths() -> None:
  """Verifies constant-exponent `power` fast paths (0, 1, 0.5, 2, 3)."""
  xs = [-0.25, 0.0, 0.4, 2.0]
  stage = Usd.Stage.CreateInMemory()
  si, px, _, _ = _make_position_batch(stage, xs, [0.0] * 4, [0.0] * 4)
  xs_arr = np.array(xs)

  # `exp == 0` skips compiling `in1` and prunes its opaque parameters.
  em_pow0 = _compile_float_emission(
      (
          'ND_power_float',
          {'in1': ('ND_add_float', {'in1': px, 'in2': 0.25}), 'in2': 0.0},
      ),
      stage=stage,
  )
  assert _num_params(em_pow0) == 0
  np.testing.assert_allclose(em_pow0.eval(si), mi.Color3f(1.0), atol=1e-6)

  em_base = _compile_float_emission(px, stage=stage)
  em_pow1 = _compile_float_emission(
      ('ND_power_float', {'in1': px}), stage=stage
  )
  assert _code_size(em_pow1) == _code_size(em_base)
  np.testing.assert_allclose(em_pow1.eval(si), em_base.eval(si), atol=1e-6)

  em_pow25 = _compile_float_emission(
      ('ND_power_float', {'in1': px, 'in2': _const_ratio(5, 2)}), stage=stage
  )
  for exp_node, expected, expect_smaller in [
      (_const_ratio(1, 2), np.sqrt(np.maximum(xs_arr, 0.0)), True),
      (_const_ratio(2), xs_arr**2, False),
      (_const_ratio(3), xs_arr**3, False),
  ]:
    em_p = _compile_float_emission(
        ('ND_power_float', {'in1': px, 'in2': exp_node}), stage=stage
    )
    if expect_smaller:
      assert _code_size(em_p) < _code_size(em_pow25)
    np.testing.assert_allclose(em_p.eval(si)[0], expected, atol=1e-5)
