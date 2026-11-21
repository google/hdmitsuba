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
from pxr import Gf
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx


@pytest.fixture(autouse=True)
def _enable_materialx_and_variant(monkeypatch: pytest.MonkeyPatch) -> None:
  mtlx.enable_materialx_and_default_variant(monkeypatch)


def _compile_bsdf(
    node_id: str, *, opacity: Any = 1.0, **inputs: Any
) -> mi.BSDF:
  bsdf, _ = mtlx.compile_mtlx_material(
      'ND_surface', bsdf=(node_id, inputs), opacity=opacity
  )
  assert bsdf is not None
  return bsdf


def _sample_inputs(
    n: int = 16384, seed: int = 12345
) -> tuple[mi.Float, mi.Point2f]:
  rng = mi.PCG32(size=n, initstate=seed)
  return rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32())


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
]


@pytest.mark.parametrize('node_id, lobe_kw, extra', _LOBE_CASES)
def test_bsdf_lobe_sampling_consistency_and_conservation(
    node_id: str, lobe_kw: dict[str, Any], extra: dict[str, Any]
) -> None:
  """Tests sample/eval_pdf consistency, chi2 goodness-of-fit, and energy.

  Args:
    node_id: MaterialX BSDF NodeDef identifier.
    lobe_kw: Input parameters passed to the BSDF node.
    extra: Optional overrides (`opacity`, `wi`).
  """
  bsdf = _compile_bsdf(node_id, opacity=extra.get('opacity', 1.0), **lobe_kw)
  u1, u2 = _sample_inputs()
  si = mtlx.make_test_si(extra['wi']) if 'wi' in extra else mtlx.make_test_si()
  ctx = mi.BSDFContext()

  bs, weight = bsdf.sample(ctx, si, u1, u2)
  val, pdf = bsdf.eval_pdf(ctx, si, bs.wo)
  val2, pdf2, bs2, weight2 = bsdf.eval_pdf_sample(ctx, si, bs.wo, u1, u2)
  for a, b in ((val, val2), (pdf, pdf2), (weight, weight2), (bs.wo, bs2.wo)):
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


def test_bsdf_ad_gradients() -> None:
  """Tests unbiased AD gradients for cutout opacity."""
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si()
  u1, u2 = _sample_inputs(n=65536, seed=42)

  bsdf_op = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', opacity=0.6, color=(0.4, 0.4, 0.4)
  )
  params_op = mi.traverse(bsdf_op)
  op_key = next(
      k for k in params_op.keys() if dr.allclose(params_op[k], 0.6, atol=1e-6)
  )
  dr.enable_grad(params_op[op_key])
  params_op.update()
  bs_op, w_op = bsdf_op.sample(ctx, si, u1, u2)
  assert not dr.grad_enabled(bs_op.wo)
  # E[w_op] = opacity * 0.4 + (1 - opacity) * 1.0 => d/d(opacity) = -0.6.
  dr.backward(dr.mean(w_op, axis=None))
  assert dr.allclose(dr.grad(params_op[op_key]), -0.6, atol=0.02)


def test_eval_features() -> None:
  """Tests denoiser feature extraction for diffuse, translucent, and opacity."""
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

  # Translucent reports diffuse transmission in diffuse_albedo and roughness=1.
  bsdf_tr = _compile_bsdf('ND_translucent_bsdf', color=(0.5, 0.3, 0.2))
  feat_tr = bsdf_tr.eval_features(si)
  assert dr.allclose(feat_tr.diffuse_albedo, [0.5, 0.3, 0.2], atol=1e-5)
  assert dr.allclose(feat_tr.specular_transmittance, 0.0, atol=1e-6)
  assert dr.allclose(feat_tr.roughness, 1.0, atol=1e-6)
  assert dr.allclose(feat_tr.wt, 0.0, atol=1e-6)
