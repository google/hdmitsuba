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
import pathlib
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
def test_custom_source_asset_shader(tmp_path: pathlib.Path) -> None:
  """Tests custom sourceAsset .mtlx surface shaders.

  Args:
    tmp_path: Pytest temporary directory fixture.
  """
  # Custom `sourceAsset` `.mtlx` surface shader renders identically in Hydra and
  # offline `usd_mitsuba` (also exercising repeated Sdr library resolution).
  mtlx_file = tmp_path / 'custom_glow.mtlx'
  mtlx_file.write_text("""<?xml version="1.0"?>
<materialx version="1.38">
  <nodedef name="ND_hdmitsuba_custom_glow" node="hdmitsuba_custom_glow">
    <input name="tint" type="color3" value="0.2, 0.5, 0.8" />
    <output name="out" type="surfaceshader" />
  </nodedef>
  <nodegraph name="NG_hdmitsuba_custom_glow" nodedef="ND_hdmitsuba_custom_glow">
    <surface_unlit name="surf" type="surfaceshader">
      <input name="emission" type="float" value="2.0" />
      <input name="emission_color" type="color3" interfacename="tint" />
    </surface_unlit>
    <output name="out" type="surfaceshader" nodename="surf" />
  </nodegraph>
</materialx>
""")
  asset_stage, asset_mesh = mtlx.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  asset_mat = UsdShade.Material.Define(asset_stage, '/mesh/Material')
  asset_sh = UsdShade.Shader.Define(asset_stage, '/mesh/Material/CustomGlow')
  asset_sh.SetSourceAsset(Sdf.AssetPath(str(mtlx_file)), 'mtlx')
  asset_sh.SetSourceAssetSubIdentifier('ND_hdmitsuba_custom_glow', 'mtlx')
  asset_sh.CreateInput('tint', Sdf.ValueTypeNames.Color3f).Set((0.3, 0.4, 0.2))
  asset_mat.CreateSurfaceOutput('mtlx').ConnectToSource(
      asset_sh.CreateOutput('out', Sdf.ValueTypeNames.Token)
  )
  UsdShade.MaterialBindingAPI.Apply(asset_mesh.GetPrim()).Bind(asset_mat)
  img_asset, _ = test_helpers.assert_hydra_equal_to_offline(
      asset_stage,
      output_prefix='test_mtlx_source_asset',
      spp=4,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_asset), (0.6, 0.8, 0.4), atol=2e-2
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
