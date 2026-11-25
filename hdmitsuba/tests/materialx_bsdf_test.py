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

"""Tests for MaterialX pbrlib BSDF closure evaluation, sampling, and surfaces."""

from __future__ import annotations

import math
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx
from usd_mitsuba import translator as usd_mitsuba_translator


@pytest.fixture(autouse=True)
def _set_default_variant() -> None:
  mtlx.set_default_variant()


def _compile_bsdf(
    node_id: str,
    *,
    opacity: Any = 1.0,
    thin_walled: Any = False,
    **inputs: Any,
) -> mi.BSDF:
  bsdf, _ = mtlx.compile_mtlx_material(
      'ND_surface',
      bsdf=(node_id, inputs),
      opacity=opacity,
      thin_walled=thin_walled,
  )
  assert bsdf is not None
  return bsdf


def _sample_inputs(
    n: int = 16384, seed: int = 12345
) -> tuple[mi.Float, mi.Point2f]:
  rng = mi.PCG32(size=n, initstate=seed)
  return rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32())


_CONDUCTOR_KW: dict[str, Any] = {
    'ior': (0.18, 0.42, 1.37),
    'extinction': (3.42, 2.35, 1.77),
    'roughness': (0.25, 0.45),
}


def _diel_kw(mode: str, roughness: float = 0.3, **extra: Any) -> dict[str, Any]:
  return {
      'ior': 1.5,
      'roughness': (roughness, roughness),
      'scatter_mode': mode,
      **extra,
  }


_LOBE_CASES = [
    pytest.param(
        'ND_oren_nayar_diffuse_bsdf',
        {'roughness': 0.5, 'energy_compensation': True},
        {},
        id='oren_nayar_comp',
    ),
    pytest.param(
        'ND_oren_nayar_diffuse_bsdf',
        {'roughness': 0.5, 'energy_compensation': False},
        {},
        id='oren_nayar_raw',
    ),
    pytest.param(
        'ND_oren_nayar_diffuse_bsdf',
        {'roughness': 0.5},
        {'wi': (0.3, -0.4, -0.8660254)},
        id='oren_nayar_backside',
    ),
    pytest.param(
        'ND_oren_nayar_diffuse_bsdf',
        {'roughness': 0.4, 'normal': Gf.Vec3f(0.2, -0.3, 0.9327379)},
        {'opacity': 0.75},
        id='oren_nayar_normal',
    ),
    pytest.param(
        'ND_translucent_bsdf',
        {'color': (0.7, 0.5, 0.3)},
        {},
        id='translucent',
    ),
    pytest.param(
        'ND_subsurface_bsdf',
        {'color': (0.8, 0.4, 0.3), 'anisotropy': 0.5},
        {},
        id='subsurface',
    ),
    pytest.param(
        'ND_sheen_bsdf',
        {'color': (0.8, 0.7, 0.9), 'roughness': 0.35, 'mode': 'zeltner'},
        {},
        id='sheen_zeltner',
    ),
    pytest.param(
        'ND_conductor_bsdf',
        {
            **_CONDUCTOR_KW,
            'normal': Gf.Vec3f(0.2, -0.3, 0.9327379),
            'tangent': Gf.Vec3f(0.0, 1.0, 0.0),
        },
        {},
        id='conductor_frame',
    ),
    pytest.param(
        'ND_conductor_bsdf',
        {**_CONDUCTOR_KW, 'retroreflective': True},
        {},
        id='conductor_retroreflective',
    ),
    pytest.param(
        'ND_generalized_schlick_bsdf',
        {
            'color0': (0.04, 0.08, 0.2),
            'color82': (0.7, 0.5, 0.9),
            'color90': (1.0, 0.9, 0.8),
            'exponent': 4.0,
            'roughness': (0.35, 0.45),
            'scatter_mode': 'RT',
        },
        {'wi': (0.6, -0.4, 0.6928203)},
        id='generalized_schlick_RT',
    ),
    pytest.param(
        'ND_chiang_hair_bsdf',
        {
            'tint_R': (0.9, 0.85, 0.8),
            'tint_TT': (0.8, 0.5, 0.3),
            'tint_TRT': (0.7, 0.4, 0.2),
            'roughness_R': (0.15, 0.25),
            'roughness_TT': (0.08, 0.25),
            'roughness_TRT': (0.3, 0.3),
            'ior': 1.55,
            'cuticle_angle': 0.52,
            'absorption_coefficient': Gf.Vec3f(0.2, 0.5, 0.9),
        },
        {},
        id='chiang_hair',
    ),
    pytest.param(
        'ND_dielectric_bsdf',
        _diel_kw('RT', 0.25),
        {'wi': (0.8, 0.0, -0.6)},
        id='dielectric_RT_tir',
    ),
    pytest.param(
        'ND_dielectric_bsdf',
        _diel_kw('T', 0.35),
        {'wi': (0.8, 0.0, -0.6)},
        id='dielectric_T_tir',
    ),
    pytest.param(
        'ND_dielectric_bsdf',
        _diel_kw('RT'),
        {'thin_walled': True, 'opacity': 0.75},
        id='dielectric_thin',
    ),
    pytest.param(
        'ND_dielectric_bsdf',
        _diel_kw('RT', thinfilm_thickness=400.0, thinfilm_ior=1.3),
        {},
        id='dielectric_RT_thinfilm',
    ),
    pytest.param(
        'ND_layer_bsdf',
        {
            'top': (
                'ND_dielectric_bsdf',
                _diel_kw('R', 0.25, ior=2.5),
            ),
            'base': (
                'ND_oren_nayar_diffuse_bsdf',
                {'color': (0.7, 0.3, 0.2), 'roughness': 0.4},
            ),
        },
        {},
        id='layer_dielectric',
    ),
    pytest.param(
        'ND_mix_bsdf',
        {
            'fg': ('ND_dielectric_bsdf', _diel_kw('RT')),
            'bg': (
                'ND_oren_nayar_diffuse_bsdf',
                {'color': (0.2, 0.7, 0.4), 'roughness': 0.4},
            ),
            'mix': 0.4,
        },
        {'opacity': 0.75},
        id='mix_opacity',
    ),
    pytest.param(
        'ND_layer_bsdf',
        {
            'top': (
                'ND_mix_bsdf',
                {
                    'fg': (
                        'ND_multiply_bsdfC',
                        {
                            'in1': (
                                'ND_oren_nayar_diffuse_bsdf',
                                {'color': (0.6, 0.4, 0.2), 'roughness': 0.4},
                            ),
                            'in2': (0.8, 0.5, 0.2),
                        },
                    ),
                    'mix': 0.4,
                },
            ),
            'base': (
                'ND_add_bsdf',
                {
                    'in1': ('ND_translucent_bsdf', {'color': (0.3, 0.5, 0.4)}),
                    'in2': (
                        'ND_oren_nayar_diffuse_bsdf',
                        {
                            'color': (0.2, 0.2, 0.2),
                            'normal': Gf.Vec3f(0.2, -0.3, 0.9327379),
                        },
                    ),
                },
            ),
        },
        {},
        id='combinator_dag_multi_normal',
    ),
]


def _check_bsdf_sampling(
    bsdf: mi.BSDF, si: mi.SurfaceInteraction3f | None = None
) -> mi.BSDFSample3f:
  u1, u2 = _sample_inputs()
  if si is None:
    si = mtlx.make_test_si()
  ctx = mi.BSDFContext()

  bs, weight = bsdf.sample(ctx, si, u1, u2)
  val, pdf = bsdf.eval_pdf(ctx, si, bs.wo)
  val2, pdf2, bs2, weight2 = bsdf.eval_pdf_sample(ctx, si, bs.wo, u1, u2)
  for a, b in (
      (val, val2),
      (pdf, pdf2),
      (weight, weight2),
      (bs.wo, bs2.wo),
      (bs.eta, bs2.eta),
  ):
    assert dr.allclose(a, b, rtol=1e-4, atol=1e-4)

  valid = (bs.pdf > 1e-5) & ~bs.is_null()
  assert dr.all(dr.isfinite(bs.pdf)) and dr.all(dr.isfinite(weight), axis=None)
  assert dr.any(valid)
  assert dr.allclose(
      dr.select(valid, bs.pdf, 0.0),
      dr.select(valid, pdf, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )
  assert dr.allclose(
      dr.select(valid, weight * bs.pdf, 0.0),
      dr.select(valid, val, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )
  assert dr.all(dr.mean(weight * dr.square(bs.eta), axis=1) <= 1.02, axis=None)
  return bs


@pytest.mark.parametrize('node_id, lobe_kw, extra', _LOBE_CASES)
def test_bsdf_lobe_sampling_consistency_and_conservation(
    node_id: str, lobe_kw: dict[str, Any], extra: dict[str, Any]
) -> None:
  """Tests sample/eval_pdf consistency, chi2 goodness-of-fit, and energy.

  Args:
    node_id: MaterialX BSDF NodeDef identifier.
    lobe_kw: Input parameters passed to the BSDF node.
    extra: Optional overrides (`opacity`, `thin_walled`, `wi`).
  """
  bsdf = _compile_bsdf(
      node_id,
      opacity=extra.get('opacity', 1.0),
      thin_walled=extra.get('thin_walled', False),
      **lobe_kw,
  )
  si = mtlx.make_test_si(extra['wi']) if 'wi' in extra else mtlx.make_test_si()
  _check_bsdf_sampling(bsdf, si)
  ctx = mi.BSDFContext()

  def _sample_func(sample: Any) -> tuple[mi.Vector3f, mi.Float]:
    bs_c, w_c = bsdf.sample(
        ctx, si, sample[0], mi.Point2f(sample[1], sample[2])
    )
    return bs_c.wo, dr.select((~bs_c.is_null()) & dr.any(w_c != 0.0), 1.0, 0.0)

  chi2 = mi.chi2.ChiSquareTest(
      domain=mi.chi2.SphericalDomain(),
      sample_func=_sample_func,
      pdf_func=lambda wo: bsdf.pdf(ctx, si, wo),
      sample_dim=3,
  )
  assert chi2.run(test_count=len(_LOBE_CASES), quiet=True), chi2.messages


def test_bsdf_reciprocity() -> None:
  """Tests Oren-Nayar BSDF Helmholtz reciprocity."""
  ctx = mi.BSDFContext()
  bsdf = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', color=(0.8, 0.5, 0.3), roughness=0.5
  )
  si_ab = mtlx.make_test_si()
  si_ba = mtlx.make_test_si(wi=(-0.5, 0.2, 0.842615))
  assert dr.allclose(
      bsdf.eval(ctx, si_ab, si_ba.wi) / si_ba.wi.z,
      bsdf.eval(ctx, si_ba, si_ab.wi) / si_ab.wi.z,
      rtol=1e-4,
      atol=1e-5,
  )


def test_bsdf_layer_throughput() -> None:
  """Tests layer throughput for opaque, dielectric, multiplied, and mixed tops."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo = mi.Vector3f(0.3, 0.4, 0.8660254)
  top = ('ND_oren_nayar_diffuse_bsdf', {'color': (0.2, 0.2, 0.2)})
  zero_top = (
      'ND_oren_nayar_diffuse_bsdf',
      {'color': (0.2, 0.2, 0.2), 'weight': 0.0},
  )
  base = ('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)})

  cases = (
      # Opaque top (even at weight=0) and multiplied tops block base (tp = 0).
      (top, (0.2, 0.2, 0.2), False),
      (zero_top, (0.0, 0.0, 0.0), False),
      (
          ('ND_multiply_bsdfC', {'in1': top, 'in2': (1.0, 0.0, 0.0)}),
          (0.2, 0.0, 0.0),
          False,
      ),
      (('ND_multiply_bsdfF', {'in1': top, 'in2': 0.0}), (0.0, 0.0, 0.0), False),
      # Partial mix (unconnected bg = Null, tp = 1), runtime OpaqueBlocker in a
      # partial mix (`tp = lerp(1, 0, 0.5) = 0.5`), and add throughputs.
      (('ND_mix_bsdf', {'fg': top, 'mix': 0.5}), (0.6, 0.6, 0.6), True),
      (
          ('ND_mix_bsdf', {'fg': zero_top, 'mix': 0.5}),
          (0.5, 0.5, 0.5),
          True,
      ),
      (
          (
              'ND_add_bsdf',
              {
                  'in1': ('ND_mix_bsdf', {'fg': top, 'mix': 0.3}),
                  'in2': ('ND_mix_bsdf', {'fg': top, 'mix': 0.4}),
              },
          ),
          (0.44, 0.44, 0.44),
          True,
      ),
  )
  for top_spec, expected, is_layered in cases:
    bsdf = _compile_bsdf('ND_layer_bsdf', top=top_spec, base=base)
    assert mi.has_flag(bsdf.flags(), mi.BSDFFlags.NonSymmetric) == is_layered
    assert dr.allclose(
        bsdf.eval(ctx, si, wo),
        mi.Color3f(*expected) * (wo.z * dr.inv_pi),
        rtol=1e-4,
        atol=1e-5,
    )

  # Pruning an opaque-top layer drops authored base opaque parameters, and a
  # root `OpaqueBlocker` (`weight=0` diffuse) emits empty flags and 0 params.
  colored_base = ('ND_oren_nayar_diffuse_bsdf', {'color': (0.3, 0.4, 0.5)})
  assert len(
      mi.traverse(_compile_bsdf('ND_layer_bsdf', top=top, base=colored_base))
  ) == len(mi.traverse(_compile_bsdf(top[0], **top[1])))
  bsdf_blocker = _compile_bsdf('ND_layer_bsdf', top=zero_top, base=colored_base)
  assert bsdf_blocker.flags() == 0
  assert not mi.traverse(bsdf_blocker)

  # Transmissive dielectric top blocks the base layer (`throughput = 0`).
  kw_rt = _diel_kw('RT', 0.2)
  bsdf_blocked = _compile_bsdf(
      'ND_layer_bsdf', top=('ND_dielectric_bsdf', kw_rt), base=base
  )
  bsdf_top = _compile_bsdf('ND_dielectric_bsdf', **kw_rt)
  assert dr.allclose(
      bsdf_blocked.eval(ctx, si, wo),
      bsdf_top.eval(ctx, si, wo),
      rtol=1e-5,
      atol=1e-5,
  )

  # Tinted dielectric coat over white diffuse base does not doubly tint the base
  # and stays two-sided from the back face (no TIR blackout).
  u1, u2 = _sample_inputs(seed=7)
  coat = (
      'ND_dielectric_bsdf',
      _diel_kw('R', 0.1, tint=(1.0, 0.0, 0.0)),
  )
  bsdf_coat = _compile_bsdf('ND_layer_bsdf', top=coat, base=base)
  _, w_front = bsdf_coat.sample(ctx, si, u1, u2)
  assert dr.allclose(
      dr.mean(w_front, axis=1), mi.Color3f(1.0, 0.96, 0.96), atol=0.015
  )
  bs_back, w_back = bsdf_coat.sample(
      ctx, mtlx.make_test_si(wi=(0.8, 0.0, -0.6)), u1, u2
  )
  assert dr.all((bs_back.wo.z < 0.0) | (bs_back.pdf <= 0.0))
  assert dr.allclose(
      dr.mean(w_back, axis=1), mi.Color3f(1.01, 0.948, 0.948), atol=0.015
  )

  # Height-correlated G2 on rough dielectric coat preserves white-furnace
  # energy at grazing incidence against `detail::ggx_dir_albedo_ab`.
  bsdf_rough_coat = _compile_bsdf(
      'ND_layer_bsdf',
      top=('ND_dielectric_bsdf', _diel_kw('R', 1.0, ior=100.0)),
      base=base,
  )
  _, w_furnace = bsdf_rough_coat.sample(
      ctx, mtlx.make_test_si(wi=(math.sqrt(0.96), 0.0, 0.2)), u1, u2
  )
  assert dr.allclose(dr.mean(w_furnace, axis=1), 1.0, atol=0.03)


def test_conductor_bsdf() -> None:
  """Tests conductor zero-IOR, height-correlated G2, tangent, and thinfilm."""
  ctx = mi.BSDFContext()
  si_n = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  bsdf_zero = _compile_bsdf(
      'ND_conductor_bsdf',
      ior=(0.0, 0.0, 0.0),
      extinction=(0.0, 0.0, 0.0),
      roughness=(0.2, 0.2),
  )
  val_zero, pdf_zero = bsdf_zero.eval_pdf(ctx, si_n, mi.Vector3f(0.0, 0.0, 1.0))
  assert dr.allclose(pdf_zero, dr.inv_pi / (4.0 * 0.2**2), rtol=1e-4)
  assert dr.allclose(val_zero, 2.08759, rtol=1e-4)

  # Height-correlated Smith G2 at grazing incidence (G2 = 1/5 vs separable 1/9).
  bsdf_white = _compile_bsdf(
      'ND_conductor_bsdf',
      ior=(1.0, 1.0, 1.0),
      extinction=(1e4, 1e4, 1e4),
      roughness=(1.0, 1.0),
  )
  sin_g = math.sqrt(1.0 - 0.2**2)
  val_g = bsdf_white.eval(
      ctx,
      mtlx.make_test_si(wi=(sin_g, 0.0, 0.2)),
      mi.Vector3f(-sin_g, 0.0, 0.2),
  )
  assert dr.allclose(val_g, 0.124298, rtol=1e-4)

  # Grazing incidence with alpha -> 0: clamped `ess <= 1` keeps energy
  # compensation >= 1 (exact `G2 / G1` when `F = 1`, `alpha = 1e-3`).
  bsdf_smooth = _compile_bsdf(
      'ND_conductor_bsdf',
      ior=(1.0, 1.0, 1.0),
      extinction=(1e4, 1e4, 1e4),
      roughness=(0.0, 0.0),
  )
  sin_e = math.sqrt(1.0 - 1e-8)
  val_e, pdf_e = bsdf_smooth.eval_pdf(
      ctx,
      mtlx.make_test_si(wi=(sin_e, 0.0, 1e-4)),
      mi.Vector3f(-sin_e, 0.0, 1e-4),
  )
  g1_e = 2.0 / (1.0 + math.sqrt(1.0 + (1e-3 * sin_e / 1e-4) ** 2))
  assert dr.allclose(val_e / pdf_e, 1.0 / (2.0 - g1_e), rtol=1e-4)

  # Rotating `tangent` by 90° matches swapping `(alpha_x, alpha_y)`.
  bsdf_ref = _compile_bsdf('ND_conductor_bsdf', **_CONDUCTOR_KW)
  bsdf_ty = _compile_bsdf(
      'ND_conductor_bsdf', **_CONDUCTOR_KW, tangent=Gf.Vec3f(0.0, 1.0, 0.0)
  )
  bsdf_swap = _compile_bsdf(
      'ND_conductor_bsdf', **{**_CONDUCTOR_KW, 'roughness': (0.45, 0.25)}
  )
  wo_x = mi.Vector3f(0.5, 0.0, math.sqrt(0.75))
  assert dr.allclose(
      bsdf_ty.eval_pdf(ctx, si_n, wo_x),
      bsdf_swap.eval_pdf(ctx, si_n, wo_x),
      rtol=1e-4,
  )
  assert not dr.allclose(
      bsdf_ty.eval(ctx, si_n, wo_x), bsdf_ref.eval(ctx, si_n, wo_x), rtol=1e-2
  )

  # Zero `thinfilm_thickness` emits no dead opaques.
  bsdf_tf0 = _compile_bsdf(
      'ND_conductor_bsdf', thinfilm_thickness=0.0, **_CONDUCTOR_KW
  )
  assert len(mi.traverse(bsdf_tf0)) == len(mi.traverse(bsdf_ref))


def test_bsdf_combinator_context_masking() -> None:
  """Tests BSDFContext lobe masking on mixed BSDFs."""
  bsdf = _compile_bsdf(
      'ND_mix_bsdf',
      fg=('ND_translucent_bsdf', {'color': (0.6, 0.4, 0.2)}),
      bg=('ND_oren_nayar_diffuse_bsdf', {'color': (0.2, 0.7, 0.4)}),
      mix=0.5,
  )
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  ctx_refl = mi.BSDFContext()
  ctx_refl.type_mask = int(mi.BSDFFlags.DiffuseReflection)
  u1, u2 = _sample_inputs(n=256, seed=7)
  bs, weight = bsdf.sample(ctx_refl, si, u1, u2)
  assert dr.all(bs.pdf > 0.0)
  assert dr.all(bs.sampled_type == int(mi.BSDFFlags.DiffuseReflection))
  assert dr.allclose(bs.pdf, bs.wo.z * dr.inv_pi, rtol=1e-4, atol=1e-5)
  assert dr.allclose(weight, mi.Color3f(0.1, 0.35, 0.2), rtol=1e-4, atol=1e-5)


def test_mix_surfaceshader_opacity() -> None:
  """Tests mixing surfaces with distinct opacities."""
  bsdf_mix, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      fg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_oren_nayar_diffuse_bsdf',
                  {'color': (1.0, 0.0, 0.0)},
              ),
              'opacity': 0.2,
          },
      ),
      bg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_oren_nayar_diffuse_bsdf',
                  {'color': (0.0, 1.0, 0.0)},
              ),
              'opacity': 1.0,
          },
      ),
      mix=0.5,
  )
  assert bsdf_mix is not None
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo = mi.Vector3f(0.0, 0.0, 1.0)
  assert dr.allclose(bsdf_mix.eval_null(si), 0.4, atol=1e-5)
  val, pdf = bsdf_mix.eval_pdf(ctx, si, wo)
  assert dr.allclose(
      val, mi.Color3f(0.1, 0.5, 0.0) * dr.inv_pi, rtol=1e-4, atol=1e-5
  )
  assert dr.allclose(pdf, 0.6 * dr.inv_pi, rtol=1e-4, atol=1e-5)
  feat = bsdf_mix.eval_features(si)
  assert dr.allclose(feat.diffuse_albedo, [0.1, 0.5, 0.0], atol=1e-5)
  assert dr.allclose(feat.specular_transmittance, 0.4, atol=1e-5)


def test_bsdf_cutout_opacity() -> None:
  """Tests stochastic cutout opacity on spatially varying UV opacity."""
  ctx = mi.BSDFContext()
  op_uv = (
      'ND_extract_vector2',
      {'in_': ('ND_texcoord_vector2', {}), 'index': 0},
  )
  bsdf_uv = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', opacity=op_uv, color=(0.8, 0.4, 0.2)
  )
  assert bsdf_uv.component_count() == 2
  assert mi.has_flag(bsdf_uv.flags(0), mi.BSDFFlags.DiffuseReflection)
  assert mi.has_flag(bsdf_uv.flags(1), mi.BSDFFlags.Null)

  si_uv = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si_uv.uv = mi.Point2f([0.0, 0.3, 0.7], [0.5, 0.5, 0.5])
  op_lanes = mi.Float(0.0, 0.3, 0.7)
  assert dr.allclose(bsdf_uv.eval_null(si_uv), mi.Color3f(1.0 - op_lanes))

  bs_s, w_s = bsdf_uv.sample(ctx, si_uv, mi.Float(0.5), mi.Point2f(0.5, 0.5))
  assert list(bs_s.sampled_type) == [
      int(mi.BSDFFlags.Null),
      int(mi.BSDFFlags.Null),
      int(mi.BSDFFlags.DiffuseReflection),
  ]
  assert list(bs_s.sampled_component) == [1, 1, 0]
  pass_mask = bs_s.is_null()
  assert dr.allclose(
      dr.select(pass_mask, bs_s.wo, mi.Vector3f(0.0, 0.0, -1.0)),
      mi.Vector3f(0.0, 0.0, -1.0),
  )
  assert dr.allclose(dr.select(pass_mask, bs_s.pdf, 0.0), [1.0, 0.7, 0.0])
  assert dr.allclose(dr.select(pass_mask, w_s, mi.Color3f(1.0)), 1.0)

  wo_n = mi.Vector3f(0.0, 0.0, 1.0)
  val_uv, pdf_uv = bsdf_uv.eval_pdf(ctx, si_uv, wo_n)
  assert dr.allclose(
      val_uv,
      mi.Color3f(0.8, 0.4, 0.2) * (op_lanes * dr.inv_pi),
      rtol=1e-4,
      atol=1e-5,
  )
  assert dr.allclose(pdf_uv, op_lanes * dr.inv_pi, rtol=1e-4, atol=1e-5)

  ctx_no_null = mi.BSDFContext()
  ctx_no_null.type_mask = int(mi.BSDFFlags.All) & ~int(mi.BSDFFlags.Null)
  _, pdf_nn = bsdf_uv.eval_pdf(ctx_no_null, si_uv, wo_n)
  bs_nn, _ = bsdf_uv.sample(
      ctx_no_null, si_uv, mi.Float(0.5), mi.Point2f(0.5, 0.5)
  )
  assert dr.allclose(pdf_nn, [0.0, dr.inv_pi, dr.inv_pi], rtol=1e-4, atol=1e-5)
  assert list(bs_nn.pdf > 0.0) == [False, True, True]


def test_spectral_bsdf() -> None:
  """Tests spectral upsampling and bounded reflectance clipping."""
  mtlx.set_variant_or_skip('cuda_ad_spectral', 'llvm_ad_spectral')
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo = mi.Vector3f(0.0, 0.0, 1.0)

  v_red = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', color=(0.9, 0.1, 0.05)
  ).eval(ctx, si, wo)
  ref_red = mi.load_dict({
      'type': 'diffuse',
      'reflectance': {'type': 'rgb', 'value': [0.9, 0.1, 0.05]},
  }).eval(ctx, si, wo)
  assert dr.allclose(v_red, ref_red, rtol=1e-4, atol=1e-5)

  # Bounded diffuse reflectance stays <= 1 per wavelength for RGB and uniform.
  v_sat = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', color=(1.0, 0.0, 0.0)
  ).eval(ctx, si, wo)
  assert dr.all((v_sat > 0.0) & (v_sat * dr.pi <= 1.0 + 1e-4), axis=None)
  v_uni = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      color=('ND_convert_float_color3', {'in_': 2.0}),
  ).eval(ctx, si, wo)
  assert dr.allclose(v_uni * dr.pi, 1.0, atol=1e-5)

  # Spectral combinator weight upsampling and spectral selection_weight.
  bsdf_comb = _compile_bsdf(
      'ND_mix_bsdf',
      fg=(
          'ND_multiply_bsdfC',
          {
              'in1': ('ND_oren_nayar_diffuse_bsdf', {'color': (0.8, 0.4, 0.2)}),
              'in2': (0.5, 0.7, 0.9),
          },
      ),
      bg=('ND_translucent_bsdf', {'color': (0.2, 0.6, 0.3)}),
      mix=0.6,
  )
  up = lambda c: mi.srgb_model_eval(
      mi.srgb_model_fetch(mi.Color3f(*c)), si.wavelengths
  )
  s_fg = 0.6 * up((0.8, 0.4, 0.2)) * up((0.5, 0.7, 0.9))
  p_fg = dr.mean(s_fg) / (dr.mean(s_fg) + dr.mean(0.4 * up((0.2, 0.6, 0.3))))
  val_c, pdf_c = bsdf_comb.eval_pdf(ctx, si, wo)
  assert dr.allclose(val_c, s_fg * dr.inv_pi, rtol=1e-4, atol=1e-5)
  assert dr.allclose(pdf_c, p_fg * dr.inv_pi, rtol=1e-4, atol=1e-5)
  bs_c, w_c = bsdf_comb.sample(
      ctx, si, mi.Float(0.25, 0.75), mi.Point2f(0.5, 0.5)
  )
  val_s, pdf_s = bsdf_comb.eval_pdf(ctx, si, bs_c.wo)
  assert dr.allclose(bs_c.pdf, pdf_s, rtol=1e-4, atol=1e-5)
  assert dr.allclose(w_c * bs_c.pdf, val_s, rtol=1e-4, atol=1e-5)

  # Unbounded spectral upsampling of conductor extinction (also covers
  # extinction-only sRGB LUT loading when `ior` is uniform).
  bsdf_cond = _compile_bsdf(
      'ND_conductor_bsdf',
      ior=('ND_convert_float_color3', {'in_': 0.5}),
      extinction=(3.42, 2.35, 1.77),
      roughness=(0.25, 0.25),
  )
  z_k = 2.0 * 3.42
  k_spec = up((3.42 / z_k, 2.35 / z_k, 1.77 / z_k)) * z_k
  f0_spec = (0.25 + dr.square(k_spec)) / (2.25 + dr.square(k_spec))
  ess = 0.9187085
  val_cond, pdf_cond = bsdf_cond.eval_pdf(ctx, si, wo)
  assert dr.allclose(pdf_cond, dr.inv_pi / (4.0 * 0.25**2), rtol=1e-4)
  assert dr.allclose(
      val_cond,
      f0_spec * (1.0 + f0_spec * (1.0 - ess) / ess) * pdf_cond,
      rtol=1e-4,
  )
  bs_k, w_k = bsdf_cond.sample(
      ctx, si, mi.Float(0.5), mi.Point2f([0.2, 0.8], [0.3, 0.7])
  )
  val_ks, pdf_ks = bsdf_cond.eval_pdf(ctx, si, bs_k.wo)
  assert dr.allclose(bs_k.pdf, pdf_ks, rtol=1e-4, atol=1e-5)
  assert dr.allclose(w_k * bs_k.pdf, val_ks, rtol=1e-4, atol=1e-5)


def test_bsdf_ad_gradients() -> None:
  """Tests detached AD gradients for roughness, IOR, opacity, and mix."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si()
  u1, u2 = _sample_inputs(n=65536, seed=42)

  def _enable_param(bsdf: mi.BSDF, val: float) -> tuple[Any, str]:
    params = mi.traverse(bsdf)
    key = next(
        k for k in params.keys() if dr.allclose(params[k], val, atol=1e-6)
    )
    dr.enable_grad(params[key])
    params.update()
    return params, key

  def _check_fd_grad(
      bsdf: mi.BSDF,
      val0: float,
      objective: Any,
      *,
      eps: float = 2e-3,
      atol: float = 0.02,
  ) -> None:
    params, key = _enable_param(bsdf, val0)
    val_p, pdf_p = bsdf.eval_pdf(ctx, si, mi.Vector3f(0.0, 0.0, 1.0))
    assert dr.grad_enabled(val_p) and not dr.grad_enabled(pdf_p)
    bs_p, w_p = bsdf.sample(ctx, si, u1, u2)
    assert not dr.grad_enabled(bs_p.wo)
    dr.backward(dr.mean(objective(bs_p, w_p), axis=None))
    ad_grad = dr.grad(params[key])

    vals = []
    for delta in (-eps, eps):
      params[key] = mi.Float(val0 + delta)
      params.update()
      bs_fd, w_fd = bsdf.sample(ctx, si, u1, u2)
      vals.append(dr.mean(objective(bs_fd, w_fd), axis=None))
    assert dr.allclose(
        ad_grad, (vals[1] - vals[0]) / (2.0 * eps), rtol=0.15, atol=atol
    )

  # Detached `wo` and `pdf` yield unbiased roughness and `ior` gradients.
  # Directional-cosine target `E[w * wo.z]` has non-zero roughness sensitivity.
  _check_fd_grad(
      _compile_bsdf('ND_conductor_bsdf', **_CONDUCTOR_KW),
      _CONDUCTOR_KW['roughness'][0],
      lambda bs, w: w * bs.wo.z,
  )
  _check_fd_grad(
      _compile_bsdf('ND_dielectric_bsdf', **_diel_kw('RT')),
      1.5,
      lambda _, w: w,
  )
  _check_fd_grad(
      _compile_bsdf('ND_sheen_bsdf', roughness=0.5, mode='zeltner'),
      0.5,
      lambda bs, w: w * bs.wo.z,
  )
  _check_fd_grad(
      _compile_bsdf(
          'ND_dielectric_bsdf',
          **_diel_kw('R', 0.2, thinfilm_thickness=420.0, thinfilm_ior=1.3),
      ),
      420.0,
      lambda _, w: w.x - w.z,
      eps=1.0,
      atol=1e-4,
  )
  _check_fd_grad(
      _compile_bsdf('ND_chiang_hair_bsdf', cuticle_angle=0.55),
      0.55,
      lambda bs, w: w * bs.wo.x,
  )

  bsdf_op = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', opacity=0.6, color=(0.4, 0.4, 0.4)
  )
  params_op, op_key = _enable_param(bsdf_op, 0.6)
  assert not dr.grad_enabled(bsdf_op.pdf(ctx, si, mi.Vector3f(0.0, 0.0, 1.0)))
  bs_op, w_op = bsdf_op.sample(ctx, si, u1, u2)
  assert not dr.grad_enabled(bs_op.wo)
  # E[w_op] = opacity * 0.4 + (1 - opacity) * 1.0 => d/d(opacity) = -0.6.
  dr.backward(dr.mean(w_op, axis=None))
  assert dr.allclose(dr.grad(params_op[op_key]), -0.6, atol=0.02)

  bsdf_mix = _compile_bsdf(
      'ND_mix_bsdf',
      fg=('ND_translucent_bsdf', {'color': (0.8, 0.8, 0.8)}),
      bg=('ND_oren_nayar_diffuse_bsdf', {'color': (0.2, 0.2, 0.2)}),
      mix=0.5,
  )
  params_mix, mix_key = _enable_param(bsdf_mix, 0.5)
  val_m, pdf_m = bsdf_mix.eval_pdf(ctx, si, mi.Vector3f(0.0, 0.0, 1.0))
  assert dr.grad_enabled(val_m) and not dr.grad_enabled(pdf_m)
  # E[w_mix] = mix * 0.8 + (1 - mix) * 0.2 => d/d(mix) = 0.6.
  _, w_mix = bsdf_mix.sample(ctx, si, u1, u2)
  dr.backward(dr.mean(w_mix, axis=None))
  assert dr.allclose(dr.grad(params_mix[mix_key]), 0.6, atol=0.02)

  # Grazing/inactive conductor lane with ior=1, k=0 must not NaN AD gradients.
  bsdf_c0 = _compile_bsdf(
      'ND_conductor_bsdf',
      ior=(1.0, 1.0, 1.0),
      extinction=(0.0, 0.0, 0.0),
      roughness=(0.2, 0.2),
  )
  params_c0 = mi.traverse(bsdf_c0)
  for k in params_c0.keys():
    dr.enable_grad(params_c0[k])
  params_c0.update()
  si_g = mtlx.make_test_si()
  si_g.wi = mi.Vector3f([0.6, 1.0], [0.0, 0.0], [0.8, 0.0])
  wo_g = mi.Vector3f([-0.6, -1.0], [0.0, 0.0], [0.8, 0.0])
  dr.backward(dr.sum(bsdf_c0.eval(ctx, si_g, wo_g)))
  for k in params_c0.keys():
    assert dr.allclose(dr.grad(params_c0[k]), 0.0, atol=1e-6)


def test_eval_features() -> None:
  """Tests denoiser features for BSDF lobes and cutout opacity."""
  si = mtlx.make_test_si(wi=(0.2, -0.3, 0.9327379))

  # Cutout + perturbed normal: albedo scales by opacity, transparency sets wt
  # and roughness=0, and sh_frame reflects the perturbed world normal.
  n_tilt = (0.0, 0.5, math.sqrt(0.75))
  bsdf = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      color=(0.8, 0.4, 0.2),
      opacity=0.4,
      normal=n_tilt,
  )
  feat = bsdf.eval_features(si)
  assert dr.allclose(feat.diffuse_albedo, [0.32, 0.16, 0.08], atol=1e-5)
  assert dr.allclose(feat.specular_reflectance, 0.0, atol=1e-6)
  assert dr.allclose(feat.specular_transmittance, 0.6, atol=1e-5)
  assert dr.allclose(feat.roughness, 0.0, atol=1e-6)
  assert dr.allclose(feat.wt, -si.to_world(si.wi), atol=1e-5)
  assert dr.allclose(feat.sh_frame.n, n_tilt, atol=1e-5)

  # Multi-normal mix selects the dominant leaf's shading frame per lane.
  n_bg = (0.5, 0.0, math.sqrt(0.75))
  bsdf_mix_n = _compile_bsdf(
      'ND_mix_bsdf',
      fg=('ND_oren_nayar_diffuse_bsdf', {'normal': n_tilt}),
      bg=('ND_oren_nayar_diffuse_bsdf', {'normal': n_bg}),
      mix=(
          'ND_extract_vector2',
          {'in_': ('ND_texcoord_vector2', {}), 'index': 0},
      ),
  )
  si_uv = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si_uv.uv = mi.Point2f([0.0, 1.0], [0.5, 0.5])
  feat_n = bsdf_mix_n.eval_features(si_uv)
  assert dr.allclose(
      feat_n.sh_frame.n,
      mi.Vector3f(
          [n_bg[0], n_tilt[0]], [n_bg[1], n_tilt[1]], [n_bg[2], n_tilt[2]]
      ),
      atol=1e-5,
  )

  # Translucent reports diffuse transmission in diffuse_albedo and roughness=1.
  bsdf_tr = _compile_bsdf('ND_translucent_bsdf', color=(0.5, 0.3, 0.2))
  feat_tr = bsdf_tr.eval_features(si)
  assert dr.allclose(feat_tr.diffuse_albedo, [0.5, 0.3, 0.2], atol=1e-5)
  assert dr.allclose(feat_tr.specular_transmittance, 0.0, atol=1e-6)
  assert dr.allclose(feat_tr.roughness, 1.0, atol=1e-6)
  assert dr.allclose(feat_tr.wt, 0.0, atol=1e-6)

  # Conductor reports specular_reflectance and max(alpha_x, alpha_y) roughness.
  bsdf_cond = _compile_bsdf('ND_conductor_bsdf', **_CONDUCTOR_KW)
  feat_c = bsdf_cond.eval_features(si)
  assert dr.allclose(feat_c.diffuse_albedo, 0.0, atol=1e-6)
  assert dr.allclose(
      feat_c.specular_reflectance, [0.93676, 0.74977, 0.33681], atol=1e-4
  )
  assert dr.allclose(feat_c.roughness, 0.45, atol=1e-6)

  # Dielectric RT splits reflectance/transmittance and reports refracted `wt`.
  bsdf_diel = _compile_bsdf(
      'ND_dielectric_bsdf', ior=1.5, roughness=(0.15, 0.25), scatter_mode='RT'
  )
  feat_d = bsdf_diel.eval_features(si)
  assert dr.allclose(feat_d.diffuse_albedo, 0.0, atol=1e-6)
  assert dr.allclose(feat_d.specular_reflectance, 0.04034, atol=1e-4)
  assert dr.allclose(feat_d.specular_transmittance, 0.95966, atol=1e-4)
  assert dr.allclose(feat_d.roughness, 0.25, atol=1e-6)
  assert dr.allclose(
      feat_d.wt,
      [-0.2 / 1.5, 0.3 / 1.5, -math.sqrt(1.0 - (0.2**2 + 0.3**2) / 1.5**2)],
      atol=1e-4,
  )


def test_dielectric_bsdf(capfd: pytest.CaptureFixture[str]) -> None:
  """Tests dielectric energy, unit IOR, relative eta, TIR, and ignored VDFs.

  Args:
    capfd: Pytest file-descriptor capture fixture.
  """
  ctx = mi.BSDFContext()
  u1, u2 = _sample_inputs(seed=7)

  # Smooth dielectric conserves energy under front incidence and back-side TIR.
  bsdf_glass = _compile_bsdf('ND_dielectric_bsdf', **_diel_kw('RT', 0.0))
  for wi, is_tir in [((0.6, 0.0, 0.8), False), ((0.8, 0.0, -0.6), True)]:
    bs, w = bsdf_glass.sample(ctx, mtlx.make_test_si(wi=wi), u1, u2)
    scaled = w * dr.square(bs.eta)
    assert dr.allclose(
        scaled if is_tir else dr.mean(scaled, axis=1), 1.0, atol=0.015
    )

  # Unit IOR produces exact zero reflection on reflection-only coats and
  # thin-walled transmissive dielectrics (even at grazing incidence where
  # `1 - cos_i^2` rounds to `1` in float32).
  si_grazing = mtlx.make_test_si(wi=(math.sqrt(1.0 - 1e-8), 0.0, 1e-4))
  bsdf_unit = _compile_bsdf(
      'ND_layer_bsdf',
      top=('ND_dielectric_bsdf', _diel_kw('R', 0.1, ior=1.0)),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)}),
  )
  assert dr.allclose(
      bsdf_unit.eval(ctx, si_grazing, mi.Vector3f(0.0, 0.0, 1.0)),
      dr.inv_pi,
      atol=1e-5,
  )
  feat_u = bsdf_unit.eval_features(si_grazing)
  assert dr.allclose(feat_u.specular_reflectance, 0.0, atol=1e-7)
  assert dr.allclose(feat_u.diffuse_albedo, 1.0, atol=1e-5)

  feat_thin = _compile_bsdf(
      'ND_dielectric_bsdf',
      thin_walled=True,
      **_diel_kw('RT', 0.1, ior=1.0),
  ).eval_features(si_grazing)
  assert dr.allclose(feat_thin.specular_reflectance, 0.0, atol=1e-7)
  assert dr.allclose(feat_thin.specular_transmittance, 1.0, atol=1e-5)

  for thin, ior, mode, wi, exp_eta, exp_wz_pos in [
      (False, 1.5, 'T', (0.0, 0.0, 1.0), 1.5, False),
      (False, 1.5, 'T', (0.0, 0.0, -1.0), 1.0 / 1.5, True),
      (True, 1.5, 'T', (0.0, 0.0, 1.0), 1.0, False),
      (False, 0.75, 'RT', (0.8, 0.0, 0.6), 1.0, True),
      (False, 0.75, 'RT', (0.0, 0.0, 1.0), 0.75, False),
  ]:
    bsdf = _compile_bsdf(
        'ND_dielectric_bsdf',
        thin_walled=thin,
        **_diel_kw(mode, 0.0, ior=ior),
    )
    bs, w = bsdf.sample(
        ctx, mtlx.make_test_si(wi=wi), mi.Float(0.99), mi.Point2f(0.5, 0.5)
    )
    assert dr.all(bs.wo.z > 0.0) == exp_wz_pos
    assert dr.allclose(bs.eta, exp_eta, atol=1e-5)
    assert dr.allclose(w * dr.square(bs.eta), 1.0, atol=2e-3)

  # Ignored VDF absorption warns once and emits no dead opaque parameters.
  bsdf_vdf_ref = _compile_bsdf('ND_dielectric_bsdf', **_diel_kw('T'))
  capfd.readouterr()
  bsdf_vdf = _compile_bsdf(
      'ND_layer_vdf',
      top=('ND_dielectric_bsdf', _diel_kw('T')),
      base=(
          'ND_absorption_vdf',
          {
              'absorption': (
                  'ND_add_vector3',
                  {
                      'in1': Gf.Vec3f(0.1, 0.2, 0.3),
                      'in2': Gf.Vec3f(0.4, 0.5, 0.6),
                  },
              )
          },
      ),
  )
  assert 'Unsupported MaterialX VDF closure' in capfd.readouterr().err
  assert len(mi.traverse(bsdf_vdf)) == len(mi.traverse(bsdf_vdf_ref))


def test_dielectric_mixing() -> None:
  """Tests mixed-surface dielectric two-sidedness, thin_walled, and TIR PDF."""
  ctx = mi.BSDFContext()

  # Mixing a solid glass surface with an opaque coated surface keeps the opaque
  # surface's reflection-only dielectric two-sided on the back face.
  bsdf_mixed, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      fg=('ND_surface', {'bsdf': ('ND_dielectric_bsdf', _diel_kw('RT', 0.1))}),
      bg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_layer_bsdf',
                  {
                      'top': ('ND_dielectric_bsdf', _diel_kw('R', 0.1)),
                      'base': (
                          'ND_oren_nayar_diffuse_bsdf',
                          {'color': (1.0, 1.0, 1.0)},
                      ),
                  },
              )
          },
      ),
      mix=0.01,
  )
  assert bsdf_mixed is not None
  feat_back = bsdf_mixed.eval_features(mtlx.make_test_si(wi=(0.8, 0.0, -0.6)))
  assert dr.allclose(feat_back.diffuse_albedo, 0.99 * 0.948, atol=2e-3)

  # MaterialX PBR spec: `thin_walled` takes precedence across mixed surfaces.
  bsdf_mix_tw, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      fg=(
          'ND_surface',
          {
              'bsdf': ('ND_dielectric_bsdf', _diel_kw('T', 0.0)),
              'thin_walled': True,
          },
      ),
      bg=(
          'ND_surface',
          {
              'bsdf': ('ND_dielectric_bsdf', _diel_kw('T', 0.0)),
              'thin_walled': False,
          },
      ),
      mix=0.5,
  )
  assert bsdf_mix_tw is not None
  bs_tw, _ = bsdf_mix_tw.sample(
      ctx,
      mtlx.make_test_si(wi=(0.0, 0.0, 1.0)),
      mi.Float(0.25, 0.75),
      mi.Point2f(0.5, 0.5),
  )
  assert dr.all(bs_tw.wo.z < 0.0)
  assert dr.allclose(bs_tw.eta, 1.0, atol=1e-5)

  # Multi-leaf rough `scatter_mode='T'` keeps non-zero PDF under macrosurface
  # TIR so rough microfacet transmission remains importance-sampled.
  bsdf_mix_t = _compile_bsdf(
      'ND_mix_bsdf',
      fg=('ND_dielectric_bsdf', _diel_kw('T', 0.35)),
      bg=('ND_oren_nayar_diffuse_bsdf', {'color': (0.5, 0.5, 0.5)}),
      mix=0.5,
  )
  si_tir = mtlx.make_test_si(wi=(0.8, 0.0, -0.6))
  wo_tir = mi.Vector3f(-0.933029, -0.000816, 0.359801)
  val_t, pdf_t = _compile_bsdf(
      'ND_dielectric_bsdf', **_diel_kw('T', 0.35)
  ).eval_pdf(ctx, si_tir, wo_tir)
  val_tir, pdf_tir = bsdf_mix_t.eval_pdf(ctx, si_tir, wo_tir)
  assert dr.allclose(val_t, 2.4628, rtol=1e-3)
  assert dr.allclose(val_tir, 0.5 * val_t, rtol=1e-4)
  assert dr.allclose(pdf_tir, (2.0 / 3.0) * pdf_t, rtol=1e-4)


_STANDARD_SURFACE_TRANS_KW: dict[str, Any] = {
    'base_color': (0.2, 0.6, 0.8),
    'specular_roughness': 0.3,
    'transmission': 0.5,
}


@pytest.mark.parametrize(
    'kwargs, exp_trans_eta',
    [
        pytest.param(
            {
                'base_color': (0.7, 0.2, 0.1),
                'metalness': 0.3,
                'specular_roughness': 0.25,
                'coat': 0.5,
            },
            None,
            id='opaque_coated',
        ),
        pytest.param(
            {**_STANDARD_SURFACE_TRANS_KW, 'thin_walled': False},
            1.5,
            id='transmissive_solid',
        ),
        pytest.param(
            {**_STANDARD_SURFACE_TRANS_KW, 'thin_walled': True},
            1.0,
            id='transmissive_thin_walled',
        ),
    ],
)
def test_standard_surface(
    kwargs: dict[str, Any], exp_trans_eta: float | None
) -> None:
  """Tests `standard_surface` compilation and sampling (solid and thin-walled).

  Args:
    kwargs: Input parameters passed to `ND_standard_surface_surfaceshader`.
    exp_trans_eta: Expected relative IOR on transmitted samples, or None if
      opaque.
  """
  bsdf, _ = mtlx.compile_mtlx_material(
      'ND_standard_surface_surfaceshader', **kwargs
  )
  assert bsdf is not None
  bs = _check_bsdf_sampling(bsdf)
  trans = bs.wo.z < 0.0
  assert dr.any(trans) == (exp_trans_eta is not None)
  assert dr.allclose(
      bs.eta, dr.select(trans, exp_trans_eta or 1.0, 1.0), atol=1e-5
  )


def test_burley_diffuse_bsdf() -> None:
  """Tests `burley_diffuse_bsdf` analytical values, reciprocity, and layering."""
  ctx = mi.BSDFContext()
  color = (0.7, 0.4, 0.2)
  si, si_rev = mtlx.make_test_si(), mtlx.make_test_si()
  si.wi = mi.Vector3f([0.6, 0.6, 0.6], [0.0, 0.0, 0.0], [0.8, 0.8, -0.8])
  si_rev.wi = mi.Vector3f([-0.8, 0.8, 0.8], [0.0, 0.0, 0.0], [0.6, 0.6, -0.6])

  for roughness, exp_fwd, exp_retro in (
      (0.0, 0.9947208, 0.9947208),
      (0.85, 1.0036964, 1.0123174),
  ):
    bsdf = _compile_bsdf(
        'ND_burley_diffuse_bsdf', color=color, roughness=roughness
    )
    assert mi.has_flag(bsdf.flags(), mi.BSDFFlags.DiffuseReflection)
    val, pdf = bsdf.eval_pdf(ctx, si, si_rev.wi)
    assert dr.allclose(
        val,
        mi.Color3f(*color)
        * (0.6 * dr.inv_pi * mi.Float(exp_fwd, exp_retro, exp_retro)),
        rtol=1e-4,
    )
    assert dr.allclose(pdf, 0.6 * dr.inv_pi, rtol=1e-5)
    # Reciprocity: f(wi, wo) / |cos_o| == f(wo, wi) / |cos_i|.
    assert dr.allclose(
        val / dr.abs(si_rev.wi.z),
        bsdf.eval(ctx, si_rev, si.wi) / dr.abs(si.wi.z),
        rtol=1e-5,
    )
    _check_bsdf_sampling(bsdf)

  # Zero-weight Burley diffuse still blocks layered bases as an opaque lobe.
  bsdf_blocked = _compile_bsdf(
      'ND_layer_bsdf',
      top=('ND_burley_diffuse_bsdf', {'weight': 0.0, 'color': color}),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)}),
  )
  assert dr.allclose(bsdf_blocked.eval(ctx, si, si_rev.wi), 0.0)


def test_sheen_bsdf() -> None:
  """Tests Conty-Kulla and Zeltner LTC `sheen_bsdf` eval, reciprocity, and layering."""
  ctx = mi.BSDFContext()
  color = (0.8, 0.6, 0.4)
  si = mtlx.make_test_si(wi=(0.6, 0.0, 0.8))
  si_rev = mtlx.make_test_si(wi=(-0.5, 0.3, math.sqrt(1.0 - 0.25 - 0.09)))

  # Conty-Kulla: analytical value, exact mirror zero, reciprocity, and flags.
  bsdf_ck = _compile_bsdf(
      'ND_sheen_bsdf', color=color, roughness=0.4, mode='conty_kulla'
  )
  assert mi.has_flag(bsdf_ck.flags(), mi.BSDFFlags.DiffuseReflection)
  assert not mi.has_flag(bsdf_ck.flags(), mi.BSDFFlags.NonSymmetric)
  val_ck, pdf_ck = bsdf_ck.eval_pdf(ctx, si, si_rev.wi)
  assert dr.allclose(val_ck, mi.Color3f(*color) * 0.0024557, rtol=1e-4)
  assert dr.allclose(pdf_ck, si_rev.wi.z * dr.inv_pi, rtol=1e-5)
  val_ck_rev = bsdf_ck.eval(ctx, si_rev, si.wi)
  assert dr.allclose(val_ck / si_rev.wi.z, val_ck_rev / si.wi.z, rtol=1e-5)
  assert dr.allclose(
      bsdf_ck.eval(
          ctx, mtlx.make_test_si(wi=(0.0, 0.0, 1.0)), mi.Vector3f(0.0, 0.0, 1.0)
      ),
      0.0,
  )
  _check_bsdf_sampling(bsdf_ck)

  # Zeltner LTC: NonSymmetric flag and exact-density zero-variance sampling.
  bsdf_z = _compile_bsdf(
      'ND_sheen_bsdf', color=color, roughness=0.35, mode='zeltner'
  )
  assert mi.has_flag(bsdf_z.flags(), mi.BSDFFlags.NonSymmetric)
  u1, u2 = _sample_inputs(2048)
  bs_z, w_z = bsdf_z.sample(ctx, si, u1, u2)
  assert dr.all(bs_z.pdf > 0.0)
  feat_z = bsdf_z.eval_features(si)
  assert dr.allclose(
      feat_z.diffuse_albedo, mi.Color3f(*color) * 0.0062649, rtol=1e-4
  )
  assert dr.allclose(w_z, feat_z.diffuse_albedo, rtol=1e-4, atol=1e-4)

  # Layering white sheen over a white translucent base conserves energy and
  # attenuates the base by `1 - dir_albedo`.
  si_g = mtlx.make_test_si(wi=(0.95, 0.0, 0.31225))
  for mode, exp_tp in (('conty_kulla', 0.6303415), ('zeltner', 0.8400126)):
    layered = _compile_bsdf(
        'ND_layer_bsdf',
        top=(
            'ND_sheen_bsdf',
            {'color': (1.0, 1.0, 1.0), 'roughness': 0.35, 'mode': mode},
        ),
        base=('ND_translucent_bsdf', {'color': (1.0, 1.0, 1.0)}),
    )
    assert dr.allclose(
        layered.eval(ctx, si_g, mi.Vector3f(0.0, 0.0, -1.0)),
        exp_tp * dr.inv_pi,
        rtol=1e-4,
    )
    _check_bsdf_sampling(layered, si_g)

  # Low-roughness Conty-Kulla keeps non-zero sampling PDF even where the
  # analytic albedo fit clips to 0 (e.g. n_dot_v = 0.5, roughness = 0.01).
  layered_low_r = _compile_bsdf(
      'ND_layer_bsdf',
      top=('ND_sheen_bsdf', {'roughness': 0.01}),
      base=('ND_translucent_bsdf', {}),
  )
  val_lr, pdf_lr = layered_low_r.eval_pdf(
      ctx,
      mtlx.make_test_si(wi=(math.sqrt(0.75), 0.0, 0.5)),
      mi.Vector3f(0.0, 0.0, 1.0),
  )
  assert dr.all(val_lr > 0.0)
  assert dr.allclose(pdf_lr, (1e-3 / 1.001) * dr.inv_pi, rtol=1e-4)

  # Zero-weight sheen passes the base through unattenuated.
  passthru = _compile_bsdf(
      'ND_layer_bsdf',
      top=('ND_sheen_bsdf', {'weight': 0.0, 'color': color}),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (0.6, 0.4, 0.2)}),
  )
  assert dr.allclose(
      passthru.eval(ctx, si, si_rev.wi),
      mi.Color3f(0.6, 0.4, 0.2) * (si_rev.wi.z * dr.inv_pi),
      rtol=1e-4,
  )


def test_generalized_schlick_bsdf() -> None:
  """Tests `generalized_schlick_bsdf` Fresnel, IOR, layering, and two-sidedness."""
  ctx = mi.BSDFContext()
  c0, c82, c90 = (0.8, 0.4, 0.1), (0.6, 0.8, 0.5), (0.95, 0.9, 0.85)
  bsdf = _compile_bsdf(
      'ND_generalized_schlick_bsdf',
      color0=c0,
      color82=c82,
      color90=c90,
      exponent=5.0,
      roughness=(0.05, 0.05),
      scatter_mode='R',
  )
  assert mi.has_flag(bsdf.flags(), mi.BSDFFlags.GlossyReflection)
  assert not mi.has_flag(bsdf.flags(), mi.BSDFFlags.GlossyTransmission)

  # At normal incidence (cos=1) Hoffman-Schlick equals `color0`; at cos=1/7
  # (~82 deg) the F82 dip scales `lerp(c0, c90, (6/7)^5)` by `color82`.
  ref_white = _compile_bsdf(
      'ND_generalized_schlick_bsdf', roughness=(0.05, 0.05)
  )
  si_norm = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo_norm = mi.Vector3f(0.0, 0.0, 1.0)
  ref_norm = ref_white.eval(ctx, si_norm, wo_norm)
  assert dr.allclose(
      bsdf.eval(ctx, si_norm, wo_norm) / ref_norm, mi.Color3f(*c0), rtol=1e-4
  )

  # Small exponent (`< 1`) still evaluates exactly to `color0` at normal incidence.
  low_exp = _compile_bsdf(
      'ND_generalized_schlick_bsdf',
      color0=(0.04, 0.04, 0.04),
      exponent=0.1,
      roughness=(0.05, 0.05),
  )
  assert dr.allclose(
      low_exp.eval(ctx, si_norm, wo_norm) / ref_norm, 0.04, rtol=1e-4
  )

  cos_82 = 1.0 / 7.0
  sin_82 = math.sqrt(1.0 - cos_82 * cos_82)
  si_82 = mtlx.make_test_si(wi=(sin_82, 0.0, cos_82))
  wo_82 = mi.Vector3f(-sin_82, 0.0, cos_82)
  f_at_82 = dr.lerp(
      mi.Color3f(*c0), mi.Color3f(*c90), (6.0 / 7.0) ** 5
  ) * mi.Color3f(*c82)
  assert dr.allclose(
      bsdf.eval(ctx, si_82, wo_82) / ref_white.eval(ctx, si_82, wo_82),
      f_at_82,
      rtol=4e-2,
  )

  # Effective IOR for solid `RT` uses MaterialX's `dot(color0, 1/3)` mean.
  blue_rt = _compile_bsdf(
      'ND_generalized_schlick_bsdf',
      color0=(0.0, 0.0, 0.5),
      roughness=(0.1, 0.1),
      scatter_mode='RT',
  )
  bs_blue, _ = blue_rt.sample(
      ctx, si_norm, mi.Float(0.01, 0.99), mi.Point2f(0.5, 0.5)
  )
  assert list(bs_blue.wo.z < 0.0) == [False, True]
  assert dr.allclose(bs_blue.eta, [1.0, 2.379796], rtol=1e-4)

  # Layering R-only generalized Schlick over diffuse attenuates the base.
  layered = _compile_bsdf(
      'ND_layer_bsdf',
      top=(
          'ND_generalized_schlick_bsdf',
          {'color0': (0.5, 0.2, 0.1), 'roughness': (0.2, 0.2)},
      ),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)}),
  )
  feat = layered.eval_features(si_norm)
  assert dr.allclose(feat.specular_reflectance, (0.5, 0.2, 0.1), atol=2e-2)
  assert dr.allclose(feat.diffuse_albedo, (0.5, 0.8, 0.9), atol=2e-2)
  _check_bsdf_sampling(layered)

  # Reflection-only generalized Schlick stays two-sided (no bogus back-face TIR)
  # even when mixed with a transmissive lobe on the same surface.
  mixed_trans = _compile_bsdf(
      'ND_mix_bsdf',
      fg=(
          'ND_generalized_schlick_bsdf',
          {'color0': c0, 'roughness': (0.1, 0.1), 'scatter_mode': 'R'},
      ),
      bg=('ND_dielectric_bsdf', _diel_kw('T', 0.1)),
      mix=0.5,
  )
  si_back = mtlx.make_test_si(wi=(0.6, 0.0, -0.8))
  wo_back = mi.Vector3f(-0.6, 0.0, -0.8)
  solo_r = _compile_bsdf(
      'ND_generalized_schlick_bsdf', color0=c0, roughness=(0.1, 0.1)
  )
  assert dr.allclose(
      mixed_trans.eval(ctx, si_back, wo_back),
      0.5 * solo_r.eval(ctx, si_back, wo_back),
      rtol=1e-4,
  )


def test_retroreflective_microfacet() -> None:
  """Tests `retroreflective=True` reflection reversal across microfacet lobes."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.6, 0.0, 0.8))
  wo_mirror = mi.Vector3f(-0.6, 0.0, 0.8)
  wo_retro = mi.Vector3f(0.6, 0.0, 0.8)

  for node_id, kw in (
      ('ND_dielectric_bsdf', _diel_kw('R', 0.15)),
      ('ND_conductor_bsdf', _CONDUCTOR_KW),
      (
          'ND_generalized_schlick_bsdf',
          {'color0': (0.4, 0.6, 0.8), 'roughness': (0.15, 0.15)},
      ),
  ):
    std = _compile_bsdf(node_id, **kw)
    retro = _compile_bsdf(node_id, retroreflective=True, **kw)
    f_retro = retro.eval(ctx, si, wo_retro)
    f_mirror = retro.eval(ctx, si, wo_mirror)
    assert dr.allclose(f_retro, std.eval(ctx, si, wo_mirror), rtol=1e-5)
    assert dr.allclose(f_mirror, std.eval(ctx, si, wo_retro), rtol=1e-5)
    assert dr.all(f_retro > 10.0 * f_mirror)

  # Thin-walled RT retroreflective keeps transmission along the forward ray.
  retro_rt = _compile_bsdf(
      'ND_dielectric_bsdf',
      thin_walled=True,
      retroreflective=True,
      **_diel_kw('RT', 0.15),
  )
  std_rt = _compile_bsdf(
      'ND_dielectric_bsdf', thin_walled=True, **_diel_kw('RT', 0.15)
  )
  wo_trans = mi.Vector3f(-0.6, 0.0, -0.8)
  assert dr.allclose(
      retro_rt.eval(ctx, si, wo_trans),
      std_rt.eval(ctx, si, wo_trans),
      rtol=1e-5,
  )
  _check_bsdf_sampling(retro_rt, si)


def test_thin_film_interference() -> None:
  """Tests Belcour-Barla Airy thin-film interference in RGB mode."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo = mi.Vector3f(0.0, 0.0, 1.0)

  # Belcour-Barla RGB interference ratios and thickness -> 0+ continuity across
  # dielectric, conductor, and generalized Schlick BSDFs at normal incidence.
  for node_id, kw, tf_kw, exp_ratio, rtol in (
      (
          'ND_dielectric_bsdf',
          _diel_kw('R', 0.1),
          {'thinfilm_thickness': 420.0, 'thinfilm_ior': 1.3},
          (0.68340, 0.87306, 0.25791),
          1e-3,
      ),
      (
          'ND_dielectric_bsdf',
          _diel_kw('R', 0.1),
          {'thinfilm_thickness': 1e-4, 'thinfilm_ior': 1.3},
          (1.0, 1.0, 1.0),
          2e-2,
      ),
      (
          'ND_conductor_bsdf',
          _CONDUCTOR_KW,
          {'thinfilm_thickness': 400.0, 'thinfilm_ior': 1.4},
          (1.01394, 0.94954, 0.64012),
          1e-3,
      ),
      (
          'ND_generalized_schlick_bsdf',
          {'color0': (0.5, 0.5, 0.5), 'roughness': (0.15, 0.15)},
          {'thinfilm_thickness': 400.0, 'thinfilm_ior': 1.4},
          (1.17344, 1.15874, 0.83705),
          1e-3,
      ),
  ):
    bare_val = _compile_bsdf(node_id, **kw).eval(ctx, si, wo)
    tf_val = _compile_bsdf(node_id, **kw, **tf_kw).eval(ctx, si, wo)
    assert dr.allclose(tf_val / bare_val, mi.Color3f(*exp_ratio), rtol=rtol)

  # Zero cross-channel phase mixing on colored `generalized_schlick_bsdf` with
  # mixed phase signs.
  gs_kw = {
      'roughness': (0.1, 0.1),
      'thinfilm_thickness': 1e-4,
      'thinfilm_ior': 1.5,
  }
  v_sc = _compile_bsdf(
      'ND_generalized_schlick_bsdf', color0=(0.8, 0.8, 0.01), **gs_kw
  ).eval(ctx, si, wo)
  v_sg = _compile_bsdf(
      'ND_generalized_schlick_bsdf', color0=(0.8, 0.8, 0.8), **gs_kw
  ).eval(ctx, si, wo)
  assert dr.allclose(v_sc.x, v_sg.x, rtol=1e-3)
  assert dr.allclose(v_sc.y, v_sg.y, rtol=1e-3)

  # Back-side Snell reciprocity on solid RT dielectric with thin film: ratio
  # against the bare dielectric cancels GGX geometry and matches at Snell angles.
  diel_rt_bare = _compile_bsdf('ND_dielectric_bsdf', **_diel_kw('RT', 0.05))
  diel_rt_tf = _compile_bsdf(
      'ND_dielectric_bsdf',
      **_diel_kw('RT', 0.05, thinfilm_thickness=350.0, thinfilm_ior=1.3),
  )
  cos_t = math.sqrt(1.0 - 0.16)
  si_ext, wo_ext = mtlx.make_test_si(wi=(0.6, 0.0, 0.8)), mi.Vector3f(
      -0.6, 0.0, 0.8
  )
  si_int, wo_int = mtlx.make_test_si(wi=(0.4, 0.0, -cos_t)), mi.Vector3f(
      -0.4, 0.0, -cos_t
  )
  ratio_ext = diel_rt_tf.eval(ctx, si_ext, wo_ext) / diel_rt_bare.eval(
      ctx, si_ext, wo_ext
  )
  ratio_int = diel_rt_tf.eval(ctx, si_int, wo_int) / diel_rt_bare.eval(
      ctx, si_int, wo_int
  )
  assert dr.allclose(ratio_ext, ratio_int, rtol=1e-2)

  # Thin-film coat layered over white diffuse conserves energy at low roughness,
  # unit thin-film IOR matches bare albedo, and thin-film `generalized_schlick`
  # overrides grazing `f90` to 1.
  layered_tf = _compile_bsdf(
      'ND_layer_bsdf',
      top=(
          'ND_dielectric_bsdf',
          _diel_kw('R', 0.02, thinfilm_thickness=400.0, thinfilm_ior=1.8),
      ),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)}),
  )
  _check_bsdf_sampling(layered_tf)
  feat_bare = _compile_bsdf(
      'ND_dielectric_bsdf', **_diel_kw('R', 0.05)
  ).eval_features(si_ext)
  feat_tf1 = _compile_bsdf(
      'ND_dielectric_bsdf',
      **_diel_kw('R', 0.05, thinfilm_thickness=400.0, thinfilm_ior=1.0),
  ).eval_features(si_ext)
  assert dr.allclose(
      feat_tf1.specular_reflectance, feat_bare.specular_reflectance, rtol=1e-5
  )
  layered_gs_tf = _compile_bsdf(
      'ND_layer_bsdf',
      top=(
          'ND_generalized_schlick_bsdf',
          {
              'color0': (0.0, 0.0, 0.0),
              'color90': (0.0, 0.0, 0.0),
              'roughness': (0.8, 0.8),
              'thinfilm_thickness': 400.0,
              'thinfilm_ior': 1.5,
          },
      ),
      base=('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)}),
  )
  feat_gs = layered_gs_tf.eval_features(
      mtlx.make_test_si(wi=(math.sqrt(1.0 - 0.01), 0.0, 0.1))
  )
  assert dr.allclose(feat_gs.specular_reflectance, 0.12481, rtol=1e-3)
  assert dr.allclose(feat_gs.diffuse_albedo, 0.87519, rtol=1e-3)


def test_thin_film_spectral() -> None:
  """Tests exact monochromatic spectral Airy interference."""
  mtlx.set_variant_or_skip('cuda_ad_spectral', 'llvm_ad_spectral')
  ctx = mi.BSDFContext()
  ior_sub = 1.5
  eta2 = math.sqrt(ior_sub)
  wl_nm = 550.0
  d_qw = wl_nm / (4.0 * eta2)
  si_spec = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si_spec.wavelengths = mi.UnpolarizedSpectrum(wl_nm)
  wo_spec = mi.Vector3f(0.0, 0.0, 1.0)
  spec_bare = _compile_bsdf(
      'ND_dielectric_bsdf', **_diel_kw('R', 0.1, ior=ior_sub)
  )
  spec_qw = _compile_bsdf(
      'ND_dielectric_bsdf',
      **_diel_kw(
          'R', 0.1, ior=ior_sub, thinfilm_thickness=d_qw, thinfilm_ior=eta2
      ),
  )
  spec_hw = _compile_bsdf(
      'ND_dielectric_bsdf',
      **_diel_kw(
          'R',
          0.1,
          ior=ior_sub,
          thinfilm_thickness=2.0 * d_qw,
          thinfilm_ior=eta2,
      ),
  )
  assert dr.allclose(spec_qw.eval(ctx, si_spec, wo_spec), 0.0, atol=1e-4)
  assert dr.allclose(
      spec_hw.eval(ctx, si_spec, wo_spec),
      spec_bare.eval(ctx, si_spec, wo_spec),
      rtol=1e-3,
  )


def test_chiang_hair_bsdf() -> None:
  """Tests `chiang_hair_bsdf` lobes, cuticle tilt, sharpness, and features."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si()
  r_only_kw = {
      'tint_TT': (0.0, 0.0, 0.0),
      'tint_TRT': (0.0, 0.0, 0.0),
      'roughness_R': (0.1, 0.2),
  }

  # `tint_TT` tints transmission proportionally while R-only has zero TT.
  hair_r = _compile_bsdf('ND_chiang_hair_bsdf', **r_only_kw)
  hair_tt = _compile_bsdf(
      'ND_chiang_hair_bsdf',
      **{**r_only_kw, 'tint_TT': (0.2, 0.6, 0.9), 'roughness_TT': (0.05, 0.2)},
  )
  wo_trans = mi.Vector3f(-0.3, 0.4, -0.8660254)
  assert dr.allclose(hair_r.eval(ctx, si, wo_trans), 0.0, atol=1e-6)
  val_tt = hair_tt.eval(ctx, si, wo_trans)
  assert dr.all(val_tt.z > 0.0)
  assert dr.allclose(
      val_tt / val_tt.z, mi.Color3f(2.0 / 9.0, 6.0 / 9.0, 1.0), rtol=1e-4
  )

  # `cuticle_angle` shifts the longitudinal reflection lobe, and very low
  # roughness stays finite and non-zero (no `sinh`/`i0` overflow).
  wo_spec = mi.Vector3f(-0.3, 0.4, 0.8660254)
  hair_tilt = _compile_bsdf(
      'ND_chiang_hair_bsdf', cuticle_angle=0.58, **r_only_kw
  )
  assert dr.allclose(hair_r.eval(ctx, si, wo_spec), 0.12884, rtol=1e-3)
  assert dr.allclose(hair_tilt.eval(ctx, si, wo_spec), 0.04420, rtol=1e-3)
  hair_sharp = _compile_bsdf(
      'ND_chiang_hair_bsdf',
      roughness_R=(0.001, 0.001),
      roughness_TT=(0.001, 0.001),
      roughness_TRT=(0.001, 0.001),
  )
  val_s, pdf_s = hair_sharp.eval_pdf(ctx, si, wo_spec)
  assert dr.allclose(val_s, 254.0, rtol=1e-2)
  assert dr.allclose(pdf_s, 254.0, rtol=1e-2)

  # Wire one shared multi-output `chiang_hair_roughness` node + melanin
  # absorption into `chiang_hair_bsdf` and verify denoiser features and `wt`.
  stage = Usd.Stage.CreateInMemory()
  rough_node = mtlx.define_shader(
      stage,
      'HairRough',
      'ND_chiang_hair_roughness',
      longitudinal=0.2,
      azimuthal=0.3,
  )
  hair_wired, _ = mtlx.compile_mtlx_material(
      'ND_surface',
      stage=stage,
      bsdf=(
          'ND_chiang_hair_bsdf',
          {
              **{
                  f'roughness_{l}': rough_node.CreateOutput(
                      f'roughness_{l}', Sdf.ValueTypeNames.Token
                  )
                  for l in ('R', 'TT', 'TRT')
              },
              'absorption_coefficient': (
                  'ND_deon_hair_absorption_from_melanin',
                  {'melanin_concentration': 0.3, 'melanin_redness': 0.4},
              ),
          },
      ),
  )
  assert hair_wired is not None
  feat = hair_wired.eval_features(si)
  assert dr.allclose(feat.wt, -si.to_world(si.wi), atol=1e-5)
  assert dr.allclose(
      feat.specular_reflectance, (0.07629, 0.06784, 0.05585), rtol=1e-3
  )
  assert dr.allclose(
      feat.specular_transmittance, (0.72127, 0.60426, 0.37919), rtol=1e-3
  )

  # Default `curve_direction` on Mitsuba curve shapes follows `dp_dv` (fiber
  # axis) rather than circumferential `dp_du`.
  curve_stage = Usd.Stage.CreateInMemory()
  curves = UsdGeom.BasisCurves.Define(curve_stage, '/Curve')
  curves.CreateTypeAttr(UsdGeom.Tokens.linear)
  curves.CreateCurveVertexCountsAttr([2])
  curves.CreatePointsAttr([(0.0, -1.0, 0.0), (0.0, 1.0, 0.0)])
  curves.CreateWidthsAttr([0.2, 0.2])
  curve_scene = mi.load_dict(
      usd_mitsuba_translator.convert_to_mitsuba(curve_stage)
  )
  si_curve = curve_scene.ray_intersect(
      mi.Ray3f(
          mi.Point3f(0.0, -0.5, 1.0), dr.normalize(mi.Vector3f(0.0, 0.5, -1.0))
      )
  )
  hair_y = _compile_bsdf(
      'ND_chiang_hair_bsdf',
      curve_direction=Gf.Vec3f(0.0, 1.0, 0.0),
      **r_only_kw,
  )
  hair_x = _compile_bsdf(
      'ND_chiang_hair_bsdf',
      curve_direction=Gf.Vec3f(1.0, 0.0, 0.0),
      **r_only_kw,
  )
  wo_curve = si_curve.to_local(dr.normalize(mi.Vector3f(0.0, 0.5, 1.0)))
  assert dr.allclose(
      hair_r.eval(ctx, si_curve, wo_curve),
      hair_y.eval(ctx, si_curve, wo_curve),
      rtol=1e-4,
  )
  assert not dr.allclose(
      hair_r.eval(ctx, si_curve, wo_curve),
      hair_x.eval(ctx, si_curve, wo_curve),
      rtol=5e-2,
  )


def test_chiang_hair_helpers() -> None:
  """Tests Chiang/d'Eon hair roughness and absorption helper nodes."""
  for out_name, exp_v in (
      ('roughness_R', 0.0539401),
      ('roughness_TT', 0.0134850),
      ('roughness_TRT', 0.2157603),
  ):
    np.testing.assert_allclose(
        mtlx.eval_emission_expr((
            'ND_convert_vector2_color3',
            {
                'in': (
                    'ND_chiang_hair_roughness',
                    {'longitudinal': 0.25, 'azimuthal': 0.4},
                    out_name,
                )
            },
        )),
        (exp_v, 0.2970400, 0.0),
        rtol=1e-4,
    )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr((
          'ND_convert_vector3_color3',
          {
              'in': (
                  'ND_chiang_hair_absorption_from_color',
                  {'color': (0.5, 0.2, 0.1), 'azimuthal_roughness': 0.3},
              )
          },
      )),
      (0.0138565, 0.0747053, 0.1529095),
      rtol=1e-4,
  )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr((
          'ND_convert_vector3_color3',
          {
              'in': (
                  'ND_deon_hair_absorption_from_melanin',
                  {'melanin_concentration': 0.4, 'melanin_redness': 0.3},
              )
          },
      )),
      (0.1784825, 0.3105311, 0.6507917),
      rtol=1e-4,
  )


@pytest.mark.parametrize(
    'shader_id, kwargs, exp_diffuse, exp_spec',
    [
        pytest.param(
            'ND_open_pbr_surface_surfaceshader',
            {
                'base_color': (0.6, 0.3, 0.2),
                'base_metalness': 0.4,
                'specular_roughness': 0.25,
                'coat_weight': 0.5,
                'fuzz_weight': 0.4,
                'fuzz_roughness': 0.5,
                'thin_film_weight': 0.6,
                'thin_film_thickness': 0.4,
            },
            (0.281, 0.134, 0.090),
            (0.235, 0.136, 0.084),
            id='open_pbr_full',
        ),
        pytest.param(
            'ND_gltf_pbr_surfaceshader',
            {
                'base_color': (0.3, 0.6, 0.8),
                'metallic': 0.5,
                'roughness': 0.3,
                'sheen_color': (0.8, 0.7, 0.9),
                'sheen_roughness': 0.4,
                'iridescence': 0.7,
                'iridescence_thickness': 380.0,
            },
            (0.162, 0.290, 0.387),
            (0.122, 0.338, 0.430),
            id='gltf_pbr_full',
        ),
        pytest.param(
            'ND_disney_principled',
            {
                'baseColor': (0.7, 0.4, 0.2),
                'metallic': 0.3,
                'roughness': 0.35,
                'sheen': 0.5,
                'clearcoat': 0.4,
            },
            (0.330, 0.192, 0.100),
            (0.250, 0.161, 0.103),
            id='disney_principled',
        ),
    ],
)
def test_uber_surface_shaders(
    shader_id: str,
    kwargs: dict[str, Any],
    exp_diffuse: tuple[float, float, float],
    exp_spec: tuple[float, float, float],
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests OpenPBR, glTF PBR, and Disney Principled uber-shader compilation.

  Args:
    shader_id: MaterialX surface shader NodeDef identifier.
    kwargs: Input parameters passed to the surface shader.
    exp_diffuse: Expected normal-incidence diffuse albedo feature.
    exp_spec: Expected normal-incidence specular reflectance feature.
    capfd: Pytest file-descriptor capture fixture for checking compiler logs.
  """
  bsdf, _ = mtlx.compile_mtlx_material(shader_id, **kwargs)
  assert bsdf is not None
  assert 'Unsupported MaterialX' not in capfd.readouterr().err
  feat = bsdf.eval_features(mtlx.make_test_si(wi=(0.0, 0.0, 1.0)))
  assert dr.allclose(feat.diffuse_albedo, exp_diffuse, atol=1e-3)
  assert dr.allclose(feat.specular_reflectance, exp_spec, atol=1e-3)
  _check_bsdf_sampling(bsdf)
