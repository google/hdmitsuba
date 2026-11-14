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

"""Tests for MaterialX math, arithmetic, trig, vector, logic, matrix, and emission ops."""

from __future__ import annotations

import math
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from hdmitsuba.tests import materialx_test_helpers as mtlx_helpers
from hdmitsuba.tests import test_helpers


@pytest.fixture(autouse=True)
def _set_mitsuba_variant() -> None:
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _check_op(
    stage: Usd.Stage, node_id: str, expected: Any, **inputs: Any
) -> None:
  """Evaluates a MaterialX node via area-light emission and checks its output."""
  expected_arr = np.atleast_1d(np.asarray(expected, dtype=np.float64))
  actual = mtlx_helpers.eval_emission_expr(
      (node_id, inputs), stage=stage, channels=len(expected_arr)
  )
  np.testing.assert_allclose(
      actual, expected_arr, rtol=1e-4, atol=1e-4, err_msg=node_id
  )


def test_elementary_arithmetic_rounding_and_interpolation_ops() -> None:
  """Tests binary arithmetic, rounding/sign, clamp, mix, smoothstep, and remap."""
  stage = Usd.Stage.CreateInMemory()

  # Binary arithmetic: add, subtract, multiply, divide, modulo, invert.
  _check_op(stage, 'ND_add_float', 0.75, in1=1.25, in2=-0.5)
  _check_op(
      stage,
      'ND_subtract_color3',
      (0.8, 0.3, -0.3),
      in1=(0.9, 0.5, 0.2),
      in2=(0.1, 0.2, 0.5),
  )
  _check_op(
      stage,
      'ND_multiply_vector2',
      (3.0, -1.0),
      in1=(1.5, -2.0),
      in2=(2.0, 0.5),
  )
  _check_op(stage, 'ND_divide_float', 1.75, in1=3.5, in2=2.0)
  _check_op(stage, 'ND_modulo_float', 1.25, in1=5.25, in2=2.0)
  _check_op(stage, 'ND_modulo_float', 1.25, in1=-0.75, in2=2.0)
  _check_op(
      stage,
      'ND_invert_color3',
      (0.8, 0.5, 0.2),
      in_=(0.2, 0.5, 0.8),
      amount=(1.0, 1.0, 1.0),
  )

  # Unary rounding and sign: absval, sign, floor, ceil, round.
  _check_op(
      stage,
      'ND_absval_vector3',
      (1.5, 0.0, 2.25),
      in_=Gf.Vec3f(-1.5, 0.0, 2.25),
  )
  _check_op(
      stage, 'ND_sign_vector3', (-1.0, 0.0, 1.0), in_=Gf.Vec3f(-2.5, 0.0, 3.5)
  )
  _check_op(
      stage, 'ND_floor_vector3', (1.0, -2.0, 2.0), in_=Gf.Vec3f(1.7, -1.2, 2.0)
  )
  _check_op(
      stage, 'ND_ceil_vector3', (2.0, -1.0, 2.0), in_=Gf.Vec3f(1.2, -1.7, 2.0)
  )
  _check_op(
      stage, 'ND_round_vector3', (1.0, 2.0, -1.0), in_=Gf.Vec3f(1.4, 1.6, -1.4)
  )

  # Range & interpolation: min, max, clamp, mix, smoothstep, remap, inside,
  # outside.
  _check_op(
      stage,
      'ND_min_color3',
      (0.2, 0.3, 0.5),
      in1=(0.2, 0.8, 0.5),
      in2=(0.4, 0.3, 0.9),
  )
  _check_op(
      stage,
      'ND_max_color3',
      (0.4, 0.8, 0.9),
      in1=(0.2, 0.8, 0.5),
      in2=(0.4, 0.3, 0.9),
  )
  _check_op(
      stage,
      'ND_clamp_color3',
      (0.1, 0.5, 0.9),
      in_=(-0.2, 0.5, 1.4),
      low=(0.1, 0.1, 0.1),
      high=(0.9, 0.9, 0.9),
  )
  _check_op(
      stage,
      'ND_mix_color3',
      (0.25, 0.75, 0.5),
      fg=(1.0, 0.0, 0.5),
      bg=(0.0, 1.0, 0.5),
      mix=0.25,
  )
  _check_op(stage, 'ND_smoothstep_float', 0.5, in_=0.5, low=0.0, high=1.0)
  _check_op(stage, 'ND_smoothstep_float', 0.15625, in_=0.25, low=0.0, high=1.0)
  _check_op(
      stage,
      'ND_remap_float',
      4.0,
      in_=0.5,
      inlow=0.0,
      inhigh=1.0,
      outlow=2.0,
      outhigh=6.0,
  )
  _check_op(stage, 'ND_inside_float', 0.2, in_=0.8, mask=0.25)
  _check_op(stage, 'ND_outside_float', 0.6, in_=0.8, mask=0.25)


def test_trigonometric_exponential_and_logarithmic_ops() -> None:
  """Tests trig (`sin`..`atan2`) and `sqrt`/`ln`/`exp`/`power` ops."""
  stage = Usd.Stage.CreateInMemory()

  _check_op(stage, 'ND_sin_float', math.sin(0.5), in_=0.5)
  _check_op(stage, 'ND_cos_float', math.cos(0.5), in_=0.5)
  _check_op(stage, 'ND_tan_float', math.tan(0.4), in_=0.4)
  _check_op(stage, 'ND_asin_float', math.asin(0.6), in_=0.6)
  _check_op(stage, 'ND_acos_float', math.acos(0.6), in_=0.6)
  _check_op(stage, 'ND_atan2_float', math.atan2(3.0, 4.0), iny=3.0, inx=4.0)
  _check_op(stage, 'ND_atan2_float', 0.0, iny=0.0, inx=0.0)
  _check_op(stage, 'ND_sqrt_float', 1.5, in_=2.25)
  _check_op(stage, 'ND_ln_float', 1.5, in_=math.e**1.5)
  _check_op(stage, 'ND_exp_float', math.exp(1.25), in_=1.25)

  # OSL safe_pow edge cases (negative base with integer/fractional exponent,
  # zero base with negative/zero exponent).
  _check_op(stage, 'ND_power_float', 0.25, in1=0.5, in2=2.0)
  _check_op(stage, 'ND_power_float', 0.25, in1=-0.5, in2=2.0)
  _check_op(stage, 'ND_power_float', -0.125, in1=-0.5, in2=3.0)
  _check_op(stage, 'ND_power_float', 0.0, in1=-0.5, in2=0.5)
  _check_op(stage, 'ND_power_float', 0.0, in1=0.0, in2=-1.0)
  _check_op(stage, 'ND_power_float', 1.0, in1=0.0, in2=0.0)


def test_vector_channel_and_conditional_logic_ops() -> None:
  """Tests 2D/3D/4D vector ops, channel ops, comparisons, and boolean logic."""
  stage = Usd.Stage.CreateInMemory()

  # 2D, 3D, and 4D dotproduct, crossproduct, magnitude, normalize, distance.
  _check_op(
      stage, 'ND_dotproduct_vector2', 5.0, in1=(2.0, 3.0), in2=(4.0, -1.0)
  )
  _check_op(
      stage,
      'ND_dotproduct_vector3',
      12.0,
      in1=Gf.Vec3f(1.0, 2.0, 3.0),
      in2=Gf.Vec3f(4.0, -5.0, 6.0),
  )
  _check_op(
      stage,
      'ND_dotproduct_vector4',
      7.5,
      in1=(1.0, 2.0, 3.0, 4.0),
      in2=(2.0, -1.0, 0.5, 1.5),
  )
  _check_op(
      stage,
      'ND_crossproduct_vector3',
      (0.0, 0.0, 1.0),
      in1=Gf.Vec3f(1.0, 0.0, 0.0),
      in2=Gf.Vec3f(0.0, 1.0, 0.0),
  )
  _check_op(stage, 'ND_magnitude_vector2', 5.0, in_=(3.0, 4.0))
  _check_op(stage, 'ND_magnitude_vector3', 7.0, in_=Gf.Vec3f(2.0, 3.0, 6.0))
  _check_op(stage, 'ND_magnitude_vector4', 5.0, in_=(1.0, 2.0, 2.0, 4.0))
  _check_op(stage, 'ND_normalize_vector2', (0.6, 0.8), in_=(3.0, 4.0))
  _check_op(
      stage,
      'ND_normalize_vector3',
      (0.0, 0.6, 0.8),
      in_=Gf.Vec3f(0.0, 3.0, 4.0),
  )
  _check_op(
      stage,
      'ND_normalize_vector4',
      (0.2, 0.4, 0.4, 0.8),
      in_=(1.0, 2.0, 2.0, 4.0),
  )
  _check_op(
      stage,
      'ND_distance_vector3',
      7.0,
      in1=Gf.Vec3f(1, 2, 3),
      in2=Gf.Vec3f(3, 5, 9),
  )

  # Channel packing (combine2/3/4) and extraction (extract index 0..3).
  _check_op(stage, 'ND_combine2_vector2', (0.25, 0.75), in1=0.25, in2=0.75)
  _check_op(
      stage,
      'ND_combine2_color4CF',
      (0.1, 0.2, 0.3, 0.85),
      in1=(0.1, 0.2, 0.3),
      in2=0.85,
  )
  _check_op(
      stage,
      'ND_combine2_vector4VV',
      (0.1, 0.2, 0.3, 0.4),
      in1=(0.1, 0.2),
      in2=(0.3, 0.4),
  )
  _check_op(
      stage,
      'ND_combine4_color4',
      (0.1, 0.2, 0.3, 0.4),
      in1=0.1,
      in2=0.2,
      in3=0.3,
      in4=0.4,
  )
  _check_op(stage, 'ND_extract_color4', 0.3, in_=(0.1, 0.2, 0.3, 0.75), index=2)
  _check_op(
      stage, 'ND_extract_color4', 0.75, in_=(0.1, 0.2, 0.3, 0.75), index=3
  )

  # Conditionals (ifgreater, ifgreatereq, ifequal) and boolean logic.
  _check_op(
      stage,
      'ND_ifgreater_float',
      2.5,
      value1=0.8,
      value2=0.3,
      in1=2.5,
      in2=-1.0,
  )
  _check_op(
      stage,
      'ND_ifgreater_float',
      -1.0,
      value1=0.3,
      value2=0.3,
      in1=2.5,
      in2=-1.0,
  )
  _check_op(
      stage,
      'ND_ifgreatereq_float',
      2.5,
      value1=0.3,
      value2=0.3,
      in1=2.5,
      in2=-1.0,
  )
  _check_op(
      stage,
      'ND_ifequal_color3',
      (0.1, 0.8, 0.3),
      value1=0.5,
      value2=0.5,
      in1=(0.1, 0.8, 0.3),
      in2=(0.9, 0.2, 0.4),
  )

  gt_true = ('ND_ifgreater_boolean', {'value1': 0.9, 'value2': 0.2})
  not_false = ('ND_logical_not', {'in_': False})
  or_true = ('ND_logical_or', {'in1': False, 'in2': True})
  _check_op(
      stage,
      'ND_convert_boolean_float',
      1.0,
      in_=('ND_logical_and', {'in1': gt_true, 'in2': not_false}),
  )
  _check_op(
      stage,
      'ND_convert_boolean_float',
      1.0,
      in_=('ND_logical_xor', {'in1': False, 'in2': or_true}),
  )
  _check_op(
      stage,
      'ND_convert_boolean_float',
      0.0,
      in_=('ND_logical_xor', {'in1': True, 'in2': True}),
  )


def test_osl_arithmetic_and_pattern_nodes() -> None:
  """Compares MaterialX math, color, channel, and pattern nodes against OSL reference values."""
  mtlx_helpers.check_osl_reference_nodes(include_noise=False)


def test_expr_matrices() -> None:
  """Tests MaterialX 2D/3D/4D matrix transforms and transposes via area-light emission."""
  stage = Usd.Stage.CreateInMemory()

  # Row-vector matrix chain:
  # (1, 2, 3) * M^T * Scale(0.5, 1.0, 2.0) = (7, 32, 100)
  m = (
      'ND_creatematrix_vector3_matrix33',
      {
          'in1': Gf.Vec3f(1.0, 2.0, 3.0),
          'in2': Gf.Vec3f(4.0, 5.0, 6.0),
          'in3': Gf.Vec3f(7.0, 8.0, 9.0),
      },
  )
  v1 = (
      'ND_transformmatrix_vector3',
      {
          'in_': Gf.Vec3f(1.0, 2.0, 3.0),
          'mat': ('ND_transpose_matrix33', {'in_': m}),
      },
  )
  _check_op(
      stage,
      'ND_transformmatrix_vector3',
      (7.0, 32.0, 100.0),
      in_=v1,
      mat=(0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0),
  )

  # 2D affine transformmatrix (vector2 * matrix33): (2, 3, 1) * M = (12, 16)
  _check_op(
      stage,
      'ND_transformmatrix_vector2M3',
      (12.0, 16.0),
      in_=(2.0, 3.0),
      mat=(2.0, -1.0, 0.0, 1.0, 4.0, 0.0, 5.0, 6.0, 1.0),
  )

  # 3D affine transformmatrix (vector3 * matrix44): (1, 2, 3, 1) * M
  _check_op(
      stage,
      'ND_transformmatrix_vector3M4',
      (9.0, 13.0, 21.0),
      in_=Gf.Vec3f(1.0, 2.0, 3.0),
      mat=(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 7, 7, 9, 1),
  )

  # 4D transformmatrix (vector4 * matrix44): (1, 2, 3, 2) * M
  _check_op(
      stage,
      'ND_transformmatrix_vector4',
      (14.0, 18.0, 28.0, 6.0),
      in_=(1.0, 2.0, 3.0, 2.0),
      mat=(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 6, 6, 8, 3),
  )


def test_translator_network_rewrites() -> None:
  """Tests PrepareMaterialXNetwork type promotions and stale-parameter cleanup in translator.cc."""
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8
  )
  # Connected input with stale authored parameter value + `ND_add_float`
  # promotion to 3-channel + `float -> vector3 -> color3` automatic
  # `ND_convert_*` insertion + integer `doclamp` coercion.
  c_src = mtlx_helpers.define_shader(
      stage, 'ColorSrc', 'ND_constant_color3', value=(0.2, 0.5, 0.8)
  )
  add_upgraded = mtlx_helpers.define_shader(
      stage, 'AddUpgraded', 'ND_add_float', in1=99.0, in2=0.1
  )
  add_upgraded.GetInput('in1').ConnectToSource(
      c_src.CreateOutput('out', Sdf.ValueTypeNames.Color3f)
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission_color=(
          'ND_range_color3',
          {
              'in_': (
                  'ND_multiply_vector3',
                  {
                      'in1': add_upgraded,
                      'in2': ('ND_constant_float', {'value': 2.0}),
                  },
              ),
              'inlow': (0.0, 0.0, 0.0),
              'inhigh': (1.0, 1.0, 1.0),
              'outlow': (0.0, 0.0, 0.0),
              'outhigh': (1.0, 0.0, 1.0),
              'doclamp': 1,
          },
      ),
  )
  # `clamp((c_src + 0.1) * 2.0, 0, 1) * (1, 0, 1) = (0.6, 0.0, 1.0)`.
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.6, 0.0, 1.0), rtol=1e-4, atol=1e-4
  )

  # Legacy `ND_normalmap` with a connected `vector2` `scale` input upgrades to
  # `ND_normalmap_vector2`.
  nm = mtlx_helpers.define_shader(
      stage,
      'NormalMapConnScale',
      'ND_normalmap',
      in_=(0.6, 0.75, 1.0),
      scale=('ND_constant_vector2', {'value': (2.0, 0.5)}),
  )
  expected_nm = np.array([0.4, 0.25, 1.0]) / np.linalg.norm([0.4, 0.25, 1.0])
  np.testing.assert_allclose(
      mtlx_helpers.eval_emission_expr(nm, stage=stage),
      expected_nm,
      rtol=1e-4,
      atol=1e-4,
  )


def test_edf_and_emissive_surface_shaders() -> None:
  """Tests EDF nodes, combinators, and surface shader emission models."""
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8
  )
  si = mtlx_helpers.make_test_si(wi=(0.3, 0.0, 0.9539392))

  # `ND_generalized_schlick_edf` cosine-weighted hemispherical average with
  # `ND_surface` opacity: 0.5 * (3, 6, 9) * mix((0, 1, 0), (1, 0, 0), 2 / 6) = (1, 1, 0).
  _, schlick_emitter = mtlx_helpers.compile_mtlx_material(
      'ND_surface',
      stage=stage,
      edf=(
          'ND_generalized_schlick_edf',
          {
              'color0': (1.0, 0.0, 0.0),
              'color90': (0.0, 1.0, 0.0),
              'exponent': 1.0,
              'base': ('ND_uniform_edf', {'color': (3.0, 6.0, 9.0)}),
          },
      ),
      opacity=0.5,
  )
  assert schlick_emitter is not None
  np.testing.assert_allclose(
      schlick_emitter.eval(si), mi.Color3f(1.0, 1.0, 0.0), atol=1e-5
  )

  # `ND_add_edf`, `ND_multiply_edf`, and `ND_mix_edf` combinators.
  _, combo_emitter = mtlx_helpers.compile_mtlx_material(
      'ND_surface',
      stage=stage,
      edf=(
          'ND_mix_edf',
          {
              'fg': (
                  'ND_add_edf',
                  {
                      'in1': ('ND_uniform_edf', {'color': (0.2, 0.4, 0.6)}),
                      'in2': ('ND_conical_edf', {'color': (0.8, 0.2, 0.4)}),
                  },
              ),
              'bg': (
                  'ND_multiply_edfC',
                  {
                      'in1': ('ND_uniform_edf', {'color': (2.0, 1.0, 0.5)}),
                      'in2': (0.5, 0.4, 0.8),
                  },
              ),
              'mix': 0.25,
          },
      ),
  )
  assert combo_emitter is not None
  # 0.25 * (1.0, 0.6, 1.0) + 0.75 * (1.0, 0.4, 0.4) = (1.0, 0.45, 0.55).
  np.testing.assert_allclose(
      combo_emitter.eval(si), mi.Color3f(1.0, 0.45, 0.55), atol=1e-5
  )

  # `ND_mix_surfaceshader` blending `ND_standard_surface_surfaceshader` and
  # `ND_surface_unlit` emissions.
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_mix_surfaceshader',
      fg=(
          'ND_standard_surface_surfaceshader',
          {'emission': 2.0, 'emission_color': (1.0, 0.0, 0.5)},
      ),
      bg=(
          'ND_surface_unlit',
          {'emission': 0.5, 'emission_color': (0.0, 1.0, 0.2)},
      ),
      mix=0.25,
  )
  expected_edf = 0.25 * 2.0 * np.array([1.0, 0.0, 0.5]) + 0.75 * 0.5 * np.array(
      [0.0, 1.0, 0.2]
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), expected_edf, rtol=1e-4
  )

  # `ND_open_pbr_surface_surfaceshader` and `ND_gltf_pbr_surfaceshader` emission.
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_open_pbr_surface_surfaceshader',
      emission_luminance=1.5,
      emission_color=(0.4, 0.6, 0.8),
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.6, 0.9, 1.2), rtol=1e-4
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_gltf_pbr_surfaceshader',
      emissive=(0.2, 0.5, 0.8),
      emissive_strength=2.5,
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.5, 1.25, 2.0), rtol=1e-4
  )


def test_surface_unlit_opacity() -> None:
  """Tests ND_surface_unlit with opacity modulating emission and doubleSided."""
  stage, plane = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8, resolution=(16, 16), spp=4
  )
  # Point the quad's normals away from the camera: single-sided area emitters
  # emit zero on the back face, whereas `doubleSided` enables two-sided
  # emission.
  plane.GetNormalsAttr().Set([(0.0, 0.0, -1.0)] * 4)
  mtlx_helpers.bind_mtlx_shader(
      stage,
      plane,
      'ND_surface_unlit',
      emission=3.0,
      emission_color=(0.9, 0.2, 0.8),
      opacity=0.5,
  )
  engine = mtlx_helpers.create_engine(stage)
  np.testing.assert_allclose(
      mtlx_helpers.render_center(engine), (0.0, 0.0, 0.0), atol=1e-4
  )
  plane.CreateDoubleSidedAttr(True)
  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, engine=engine, spp=4, subdivision_level=0
  )
  np.testing.assert_allclose(
      mtlx_helpers.render_center(image_hd),
      (1.35, 0.3, 1.2),
      rtol=0.05,
      atol=0.02,
  )


@pytest.mark.parametrize(
    'variants',
    [('cuda_ad_rgb', 'llvm_ad_rgb'), ('cuda_ad_spectral', 'llvm_ad_spectral')],
)
def test_parameter_edit_reuses_kernels(variants: tuple[str, str]) -> None:
  """Tests that editing emission parameter values reuses compiled kernels.

  Args:
    variants: Candidate Mitsuba variants to test.
  """
  mi.set_variant(*variants)
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  shader = mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_standard_surface_surfaceshader',
      base=0.0,
      emission=2.0,
      emission_color=(1.0, 0.0, 0.2),
  )

  engine = mtlx_helpers.create_engine(stage)
  engine.render()

  shader.GetInput('emission_color').Set((0.1, 0.2, 0.85))
  with dr.kernel_history((dr.KernelType.JIT,)) as history:
    center = mtlx_helpers.render_center(engine)
  assert len(history) > 0
  assert all(k.cache_hit for k in history), history
  # The edited scene renders like a freshly loaded one.
  np.testing.assert_allclose(
      center, mtlx_helpers.render_center(stage), rtol=1e-3
  )


def test_mi_traverse_emitter_param_update() -> None:
  """Verifies mi.traverse exposes VM parameters on MtlxExprTexture and updates eval()."""
  _, emitter = mtlx_helpers.compile_mtlx_material(
      'ND_surface_unlit', emission_color=(0.2, 0.4, 0.6)
  )
  assert emitter is not None
  si = mtlx_helpers.make_test_si(wi=(0.0, 0.0, 1.0))
  np.testing.assert_allclose(
      emitter.eval(si), mi.Color3f(0.2, 0.4, 0.6), rtol=1e-4
  )
  params = mi.traverse(emitter)
  assert 'radiance.param0' in params
  params['radiance.param0'] = mi.Float(0.8)
  params.update()
  np.testing.assert_allclose(
      emitter.eval(si), mi.Color3f(0.8, 0.4, 0.6), rtol=1e-4
  )

