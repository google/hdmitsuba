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

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Gf
from pxr import Usd
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx

_NUM_SAMPLES = 16384


@pytest.fixture(autouse=True)
def _enable_materialx_and_variant(monkeypatch: pytest.MonkeyPatch) -> None:
  mtlx.enable_materialx_and_default_variant(monkeypatch)


def _compile_bsdf(
    node_id: str,
    *,
    stage: Usd.Stage | None = None,
    opacity: Any = 1.0,
    thin_walled: bool = False,
    **inputs: Any,
) -> mi.BSDF:
  bsdf, _ = mtlx.compile_mtlx_material(
      'ND_surface',
      stage=stage,
      bsdf=(node_id, inputs),
      opacity=opacity,
      thin_walled=thin_walled,
  )
  assert bsdf is not None
  return bsdf


def _sample_inputs(
    n: int = _NUM_SAMPLES, seed: int = 12345
) -> tuple[mi.Float, mi.Point2f]:
  u = (
      np.random.default_rng(seed)
      .uniform(1e-4, 1.0 - 1e-4, size=(3, n))
      .astype(np.float32)
  )
  return mi.Float(u[0]), mi.Point2f(u[1], u[2])


def _check_bsdf_sampling(
    bsdf: mi.BSDF, *, wi: tuple[float, float, float] = (0.3, -0.4, 0.8660254)
) -> None:
  """Verifies sample/eval_pdf consistency, unbiasedness, and energy conservation."""
  u1, u2 = _sample_inputs()
  si = mtlx.make_test_si(wi=wi)
  ctx = mi.BSDFContext()

  bs, weight = bsdf.sample(ctx, si, u1, u2)
  val, pdf = bsdf.eval_pdf(ctx, si, bs.wo)
  val2, pdf2, bs2, weight2 = bsdf.eval_pdf_sample(ctx, si, bs.wo, u1, u2)
  for a, b in ((val, val2), (pdf, pdf2), (weight, weight2), (bs.wo, bs2.wo)):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

  is_null = (bs.sampled_type & int(mi.BSDFFlags.Null)) != 0
  valid = (bs.pdf > 1e-5) & ~is_null

  assert dr.all(dr.isfinite(bs.pdf)) and dr.all(dr.isfinite(weight), axis=None)
  assert dr.any(valid)
  np.testing.assert_allclose(
      dr.select(valid, bs.pdf, 0.0),
      dr.select(valid, pdf, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )
  np.testing.assert_allclose(
      dr.select(valid, weight * bs.pdf, 0.0),
      dr.select(valid, val, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )
  assert dr.all(dr.mean(weight * dr.square(bs.eta), axis=1) <= 1.02, axis=None)

  v_sph = np.asarray(
      bsdf.eval(ctx, si, mi.warp.square_to_uniform_sphere(u2)), dtype=np.float64
  )
  est_sph = 4.0 * np.pi * np.mean(v_sph, axis=-1)
  err_sph = 4.0 * np.pi * np.std(v_sph, axis=-1) / np.sqrt(_NUM_SAMPLES)
  non_null_w = np.asarray(dr.select(is_null, 0.0, weight), dtype=np.float64)
  est_s = np.mean(non_null_w, axis=-1)
  err_s = np.std(non_null_w, axis=-1) / np.sqrt(_NUM_SAMPLES)
  tol = 3.5 * np.hypot(err_s, err_sph) + 1.5e-2
  assert np.all(
      np.abs(est_s - est_sph) <= tol
  ), f'Unbiasedness mismatch: sample={est_s}, sphere={est_sph}, tol={tol}'


def _lobe_cases() -> list[Any]:
  def diff(c=(0.7, 0.3, 0.2), r=0.4):
    return ('ND_oren_nayar_diffuse_bsdf', {'color': c, 'roughness': r})

  def diel(mode, r=0.3):
    return (
        'ND_dielectric_bsdf',
        {'ior': 1.5, 'roughness': (r, r), 'scatter_mode': mode},
    )

  cond = (
      'ND_conductor_bsdf',
      {
          'ior': (0.18, 0.42, 1.37),
          'extinction': (3.42, 2.35, 1.77),
          'roughness': (0.25, 0.45),
      },
  )
  cases = [
      (
          'oren_nayar_comp',
          'ND_oren_nayar_diffuse_bsdf',
          {'roughness': 0.5, 'energy_compensation': True},
          {},
      ),
      (
          'oren_nayar_raw',
          'ND_oren_nayar_diffuse_bsdf',
          {'roughness': 0.5, 'energy_compensation': False},
          {},
      ),
      ('translucent', 'ND_translucent_bsdf', {'color': (0.7, 0.5, 0.3)}, {}),
      (
          'subsurface',
          'ND_subsurface_bsdf',
          {'color': (0.8, 0.4, 0.3), 'anisotropy': 0.5},
          {},
      ),
      ('conductor', *cond, {}),
      (
          'conductor_frame',
          'ND_conductor_bsdf',
          {
              **cond[1],
              'normal': Gf.Vec3f(0.2, -0.3, 0.9327379),
              'tangent': Gf.Vec3f(0.0, 1.0, 0.0),
          },
          {},
      ),
      ('dielectric_R', *diel('R'), {}),
      ('dielectric_T', *diel('T'), {}),
      ('dielectric_RT', *diel('RT'), {}),
      ('dielectric_RT_tir', *diel('RT', 0.25), {'wi': (0.8, 0.0, -0.6)}),
      ('dielectric_T_tir', *diel('T', 0.35), {'wi': (0.8, 0.0, -0.6)}),
      ('dielectric_thin', *diel('RT'), {'thin_walled': True}),
      ('layer', 'ND_layer_bsdf', {'top': diel('R', 0.25), 'base': diff()}, {}),
      (
          'mix_opacity',
          'ND_mix_bsdf',
          {'fg': cond, 'bg': diff((0.2, 0.7, 0.4)), 'mix': 0.4},
          {'opacity': 0.75},
      ),
      (
          'add_multiply',
          'ND_add_bsdf',
          {
              'in1': (
                  'ND_multiply_bsdfC',
                  {'in1': diff((0.6, 0.4, 0.2)), 'in2': (0.5, 0.5, 0.5)},
              ),
              'in2': ('ND_translucent_bsdf', {'color': (0.2, 0.3, 0.4)}),
          },
          {},
      ),
  ]
  return [pytest.param(nid, kw, ex, id=cid) for cid, nid, kw, ex in cases]


@pytest.mark.parametrize('node_id, lobe_kw, extra', _lobe_cases())
def test_bsdf_lobe_sampling_consistency_and_conservation(
    node_id: str, lobe_kw: dict[str, Any], extra: dict[str, Any]
) -> None:
  """Tests sample/eval_pdf consistency, unbiasedness, and energy conservation.

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
  _check_bsdf_sampling(bsdf, wi=extra.get('wi', (0.3, -0.4, 0.8660254)))


def test_bsdf_reciprocity_and_layer_energy_conservation() -> None:
  """Tests BSDF reciprocity and layered/dielectric energy conservation."""
  stage = Usd.Stage.CreateInMemory()
  ctx = mi.BSDFContext()
  bsdf = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      stage=stage,
      color=(0.8, 0.5, 0.3),
      roughness=0.5,
  )
  si_ab = mtlx.make_test_si(wi=(0.3, -0.4, 0.8660254))
  si_ba = mtlx.make_test_si(wi=(-0.5, 0.2, 0.842615))
  np.testing.assert_allclose(
      bsdf.eval(ctx, si_ab, si_ba.wi) / si_ba.wi.z,
      bsdf.eval(ctx, si_ba, si_ab.wi) / si_ab.wi.z,
      rtol=1e-4,
      atol=1e-5,
  )

  u1, u2 = _sample_inputs(seed=7)
  # Tinted dielectric coat over white diffuse base does not doubly tint the base
  # and stays two-sided from the back face (no TIR blackout).
  coat = (
      'ND_dielectric_bsdf',
      {
          'ior': 1.5,
          'roughness': (0.1, 0.1),
          'scatter_mode': 'R',
          'tint': (1.0, 0.0, 0.0),
      },
  )
  base = ('ND_oren_nayar_diffuse_bsdf', {'color': (1.0, 1.0, 1.0)})
  bsdf_coat = _compile_bsdf('ND_layer_bsdf', stage=stage, top=coat, base=base)
  _, w_front = bsdf_coat.sample(
      ctx, mtlx.make_test_si(wi=(0.0, 0.0, 1.0)), u1, u2
  )
  np.testing.assert_allclose(
      dr.mean(w_front, axis=1), mi.Color3f(1.0, 0.96, 0.96), atol=0.03
  )

  bs_back, w_back = bsdf_coat.sample(
      ctx, mtlx.make_test_si(wi=(0.8, 0.0, -0.6)), u1, u2
  )
  assert dr.all((bs_back.wo.z < 0.0) | (bs_back.pdf <= 0.0))
  np.testing.assert_allclose(
      dr.mean(w_back, axis=1), mi.Color3f(1.0, 0.93, 0.93), atol=0.03
  )

  # Opaque or transmissive top lobes block the base layer (`throughput = 0`).
  si_n = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo_n = mi.Vector3f(0.3, 0.4, 0.8660254)
  for top in [
      ('ND_oren_nayar_diffuse_bsdf', {'color': (0.2, 0.2, 0.2)}),
      (
          'ND_dielectric_bsdf',
          {'ior': 1.5, 'roughness': (0.2, 0.2), 'scatter_mode': 'RT'},
      ),
  ]:
    bsdf_blocked = _compile_bsdf(
        'ND_layer_bsdf', stage=stage, top=top, base=base
    )
    bsdf_top, _ = mtlx.compile_mtlx_material(
        'ND_surface', stage=stage, bsdf=top
    )
    assert bsdf_top is not None
    np.testing.assert_allclose(
        bsdf_blocked.eval(ctx, si_n, wo_n),
        bsdf_top.eval(ctx, si_n, wo_n),
        rtol=1e-5,
        atol=1e-5,
    )

  # Zero-IOR/extinction conductor stays finite at normal incidence (no 0/0 NaN).
  bsdf_zero_cond = _compile_bsdf(
      'ND_conductor_bsdf',
      stage=stage,
      ior=(0.0, 0.0, 0.0),
      extinction=(0.0, 0.0, 0.0),
      roughness=(0.2, 0.2),
  )
  val_zero = bsdf_zero_cond.eval(
      ctx, mtlx.make_test_si(wi=(0.0, 0.0, 1.0)), mi.Vector3f(0.0, 0.0, 1.0)
  )
  assert dr.all(dr.isfinite(val_zero), axis=None) and dr.all(
      val_zero > 0.0, axis=None
  )

  # Smooth dielectric conserves energy under front incidence and back-side TIR.
  bsdf_glass = _compile_bsdf(
      'ND_dielectric_bsdf',
      stage=stage,
      ior=1.5,
      roughness=(0.0, 0.0),
      scatter_mode='RT',
  )
  for wi, is_tir in [((0.6, 0.0, 0.8), False), ((0.8, 0.0, -0.6), True)]:
    bs, w = bsdf_glass.sample(ctx, mtlx.make_test_si(wi=wi), u1, u2)
    scaled = w * dr.square(bs.eta)
    np.testing.assert_allclose(
        scaled if is_tir else dr.mean(scaled, axis=1),
        1.0 if is_tir else mi.Color3f(1.0),
        atol=0.015,
    )


def test_bsdf_cutout_opacity() -> None:
  """Tests stochastic cutout opacity and mixed-opacity surface shaders."""
  stage = Usd.Stage.CreateInMemory()
  ctx = mi.BSDFContext()

  # Spatially varying UV opacity: null pass-through, eval_null, eval/eval_pdf
  # scaling, and opacity == 0 lane consistency when BSDFFlags.Null is disabled.
  op_uv = (
      'ND_extract_vector2',
      {'in_': ('ND_texcoord_vector2', {}), 'index': 0},
  )
  bsdf_uv = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      stage=stage,
      opacity=op_uv,
      color=(0.8, 0.4, 0.2),
  )
  assert (bsdf_uv.flags() & int(mi.BSDFFlags.Null)) != 0
  si_uv = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si_uv.uv = mi.Point2f([0.0, 0.3, 0.7], [0.5, 0.5, 0.5])
  op_lanes = np.array([0.0, 0.3, 0.7])
  np.testing.assert_allclose(
      bsdf_uv.eval_null(si_uv),
      np.broadcast_to(1.0 - op_lanes, (3, 3)),
      atol=1e-5,
  )
  bs_s, w_s = bsdf_uv.sample(ctx, si_uv, mi.Float(0.5), mi.Point2f(0.5, 0.5))
  np.testing.assert_array_equal(
      bs_s.sampled_type,
      [
          int(mi.BSDFFlags.Null),
          int(mi.BSDFFlags.Null),
          int(mi.BSDFFlags.DiffuseReflection),
      ],
  )
  np.testing.assert_allclose(
      np.asarray(bs_s.wo)[:, :2],
      [[0.0, 0.0], [0.0, 0.0], [-1.0, -1.0]],
      atol=1e-5,
  )
  np.testing.assert_allclose(bs_s.pdf[:2], [1.0, 0.7], atol=1e-5)
  np.testing.assert_allclose(np.asarray(w_s)[:, :2], 1.0, atol=1e-5)

  wo_n = mi.Vector3f(0.0, 0.0, 1.0)
  val_uv, pdf_uv = bsdf_uv.eval_pdf(ctx, si_uv, wo_n)
  np.testing.assert_allclose(
      val_uv,
      np.outer(np.array([0.8, 0.4, 0.2]) / np.pi, op_lanes),
      rtol=1e-4,
      atol=1e-5,
  )
  np.testing.assert_allclose(pdf_uv, op_lanes / np.pi, rtol=1e-4, atol=1e-5)

  ctx_no_null = mi.BSDFContext()
  ctx_no_null.type_mask = int(mi.BSDFFlags.All) & ~int(mi.BSDFFlags.Null)
  _, pdf_nn = bsdf_uv.eval_pdf(ctx_no_null, si_uv, wo_n)
  bs_nn, _ = bsdf_uv.sample(
      ctx_no_null, si_uv, mi.Float(0.5), mi.Point2f(0.5, 0.5)
  )
  np.testing.assert_allclose(
      pdf_nn, [0.0, 1.0 / np.pi, 1.0 / np.pi], rtol=1e-4, atol=1e-5
  )
  np.testing.assert_array_equal(bs_nn.pdf > 0.0, [False, True, True])

  # Mixing surfaces with different opacities weights each surface by its own
  # opacity without dividing by the mixed opacity.
  bsdf_mix, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      stage=stage,
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
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  np.testing.assert_allclose(bsdf_mix.eval_null(si), 0.4, atol=1e-5)
  np.testing.assert_allclose(
      bsdf_mix.eval(ctx, si, wo_n),
      mi.Color3f(0.1, 0.5, 0.0) / dr.pi,
      rtol=1e-4,
      atol=1e-5,
  )


def test_dielectric_transmission_ior() -> None:
  """Tests solid vs thin-walled relative IOR, sub-unit IOR, and mixed surfaces."""
  stage = Usd.Stage.CreateInMemory()
  ctx = mi.BSDFContext()
  for thin, ior, mode, wi, exp_eta, exp_wz_pos in [
      (False, 1.5, 'T', (0.0, 0.0, 1.0), 1.5, False),
      (False, 1.5, 'T', (0.0, 0.0, -1.0), 1.0 / 1.5, True),
      (True, 1.5, 'T', (0.0, 0.0, 1.0), 1.0, False),
      (False, 0.75, 'RT', (0.8, 0.0, 0.6), 1.0, True),
      (False, 0.75, 'RT', (0.0, 0.0, 1.0), 0.75, False),
  ]:
    bsdf = _compile_bsdf(
        'ND_dielectric_bsdf',
        stage=stage,
        thin_walled=thin,
        ior=ior,
        roughness=(0.0, 0.0),
        scatter_mode=mode,
    )
    bs, w = bsdf.sample(
        ctx, mtlx.make_test_si(wi=wi), mi.Float(0.99), mi.Point2f(0.5, 0.5)
    )
    assert (bs.wo.z > 0.0) == exp_wz_pos
    np.testing.assert_allclose(bs.eta, exp_eta, atol=1e-5)
    if ior == 0.75 and exp_wz_pos:
      np.testing.assert_allclose(w, 1.0, atol=2e-3)

  # Mixing a solid glass surface with an opaque coated surface keeps the opaque
  # surface's reflection-only dielectric two-sided on the back face.
  bsdf_mixed, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      stage=stage,
      fg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_dielectric_bsdf',
                  {'ior': 1.5, 'roughness': (0.1, 0.1), 'scatter_mode': 'RT'},
              )
          },
      ),
      bg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_layer_bsdf',
                  {
                      'top': (
                          'ND_dielectric_bsdf',
                          {
                              'ior': 1.5,
                              'roughness': (0.1, 0.1),
                              'scatter_mode': 'R',
                          },
                      ),
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
  u1, u2 = _sample_inputs(seed=11)
  _, w_back = bsdf_mixed.sample(
      ctx, mtlx.make_test_si(wi=(0.8, 0.0, -0.6)), u1, u2
  )
  assert dr.all(dr.mean(w_back, axis=1) > 0.85, axis=None)

  # MaterialX PBR spec: `thin_walled` takes precedence across mixed surfaces.
  bsdf_mix_tw, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      stage=stage,
      fg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_dielectric_bsdf',
                  {'ior': 1.5, 'roughness': (0.0, 0.0), 'scatter_mode': 'T'},
              ),
              'thin_walled': True,
          },
      ),
      bg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_dielectric_bsdf',
                  {'ior': 1.5, 'roughness': (0.0, 0.0), 'scatter_mode': 'T'},
              ),
              'thin_walled': False,
          },
      ),
      mix=0.5,
  )
  assert bsdf_mix_tw is not None
  bs_tw, _ = bsdf_mix_tw.sample(
      ctx, mtlx.make_test_si(wi=(0.0, 0.0, 1.0)), u1, u2
  )
  assert dr.all(bs_tw.wo.z < 0.0)
  np.testing.assert_allclose(bs_tw.eta, 1.0, atol=1e-5)


def test_standard_surface() -> None:
  """Tests `standard_surface` compilation and sampling (solid and thin-walled)."""
  stage = Usd.Stage.CreateInMemory()
  for kwargs in [
      {
          'base_color': (0.7, 0.2, 0.1),
          'metalness': 0.3,
          'specular_roughness': 0.25,
          'coat': 0.5,
      },
      {
          'base_color': (0.2, 0.6, 0.8),
          'specular_roughness': 0.3,
          'transmission': 0.5,
          'thin_walled': True,
      },
  ]:
    bsdf, _ = mtlx.compile_mtlx_material(
        'ND_standard_surface_surfaceshader', stage=stage, **kwargs
    )
    assert bsdf is not None
    _check_bsdf_sampling(bsdf)
    if kwargs.get('thin_walled'):
      u1, u2 = _sample_inputs(seed=19)
      bs, _ = bsdf.sample(
          mi.BSDFContext(), mtlx.make_test_si(wi=(0.0, 0.0, 1.0)), u1, u2
      )
      assert dr.any(bs.wo.z < 0.0)
      np.testing.assert_allclose(bs.eta, 1.0, atol=1e-5)


def test_spectral_bsdf() -> None:
  """Tests spectral upsampling, bounded clipping, combinators, and opacity."""
  mtlx.set_variant_or_skip('cuda_ad_spectral', 'llvm_ad_spectral')
  stage = Usd.Stage.CreateInMemory()
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  wo = mi.Vector3f(0.0, 0.0, 1.0)

  v_red = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', stage=stage, color=(0.9, 0.1, 0.05)
  ).eval(ctx, si, wo)
  ref_red = mi.load_dict({
      'type': 'diffuse',
      'reflectance': {'type': 'rgb', 'value': [0.9, 0.1, 0.05]},
  }).eval(ctx, si, wo)
  np.testing.assert_allclose(v_red, ref_red, rtol=1e-4, atol=1e-5)

  # Bounded diffuse reflectance stays <= 1 per wavelength for RGB and uniform.
  v_sat = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf', stage=stage, color=(1.0, 0.0, 0.0)
  ).eval(ctx, si, wo)
  assert dr.all((v_sat > 0.0) & (v_sat * dr.pi <= 1.0 + 1e-4), axis=None)
  v_uni = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      stage=stage,
      color=('ND_convert_float_color3', {'in_': 2.0}),
  ).eval(ctx, si, wo)
  np.testing.assert_allclose(v_uni * dr.pi, 1.0, atol=1e-5)

  bsdf_spec, _ = mtlx.compile_mtlx_material(
      'ND_mix_surfaceshader',
      stage=stage,
      fg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_conductor_bsdf',
                  {
                      'ior': (0.18, 0.42, 1.37),
                      'extinction': (3.42, 2.35, 1.77),
                      'roughness': (0.25, 0.25),
                  },
              ),
              'opacity': 0.4,
          },
      ),
      bg=(
          'ND_surface',
          {
              'bsdf': (
                  'ND_oren_nayar_diffuse_bsdf',
                  {'color': (0.7, 0.3, 0.2)},
              ),
              'opacity': 1.0,
          },
      ),
      mix=0.5,
  )
  assert bsdf_spec is not None
  np.testing.assert_allclose(bsdf_spec.eval_null(si), 0.3, atol=1e-5)
  _check_bsdf_sampling(bsdf_spec)


def test_bsdf_ad_gradients() -> None:
  """Tests unbiased detached AD gradients for roughness and cutout opacity."""
  stage = Usd.Stage.CreateInMemory()
  ctx = mi.BSDFContext()
  si = mtlx.make_test_si(wi=(0.3, -0.4, 0.8660254))
  u1, u2 = _sample_inputs(n=65536, seed=42)

  # Detached `wo` in `sample()` yields unbiased roughness gradients.
  bsdf = _compile_bsdf(
      'ND_conductor_bsdf',
      stage=stage,
      ior=(0.18, 0.42, 1.37),
      extinction=(3.42, 2.35, 1.77),
      roughness=(0.35, 0.45),
  )
  params = mi.traverse(bsdf)
  r_key = next(k for k in params.keys() if np.isclose(params[k], 0.35))
  dr.enable_grad(params[r_key])
  params.update()
  bs, w = bsdf.sample(ctx, si, u1, u2)
  assert not dr.grad_enabled(bs.wo)
  # Directional-cosine target `E[w * wo.z]` has non-zero roughness sensitivity.
  dr.backward(dr.mean(w * bs.wo.z, axis=None))
  ad_grad = np.asarray(dr.grad(params[r_key]))

  eps = 2e-3
  vals = []
  for delta in (-eps, eps):
    params[r_key] = mi.Float(0.35 + delta)
    params.update()
    bs_fd, w_fd = bsdf.sample(ctx, si, u1, u2)
    vals.append(np.asarray(dr.mean(w_fd * bs_fd.wo.z, axis=None)))
  np.testing.assert_allclose(
      ad_grad, (vals[1] - vals[0]) / (2.0 * eps), rtol=0.15, atol=0.02
  )

  # Mixed pass-through and surface lanes differentiate cutout opacity cleanly.
  bsdf_op = _compile_bsdf(
      'ND_oren_nayar_diffuse_bsdf',
      stage=stage,
      opacity=0.6,
      color=(0.4, 0.4, 0.4),
  )
  params_op = mi.traverse(bsdf_op)
  op_key = next(k for k in params_op.keys() if np.isclose(params_op[k], 0.6))
  dr.enable_grad(params_op[op_key])
  params_op.update()
  _, w_op = bsdf_op.sample(ctx, si, u1, u2)
  # E[w_op] = opacity * 0.4 + (1 - opacity) * 1.0 => d/d(opacity) = -0.6.
  dr.backward(dr.mean(w_op, axis=None))
  np.testing.assert_allclose(dr.grad(params_op[op_key]), -0.6, atol=0.02)
