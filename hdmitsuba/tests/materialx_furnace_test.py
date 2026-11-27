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

"""White-furnace energy conservation and sampling tests for MaterialX BSDFs."""

from __future__ import annotations

import math
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Gf
from pxr import Usd
import pytest

from hdmitsuba.tests import materialx_test_helpers as mtlx
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba_translator

_FRONT_COS = (0.02, 0.1, 0.25, 0.5, 0.75, 0.99)
_BOTH_COS = _FRONT_COS + (-0.99, -0.75, -0.5, -0.25, -0.1, -0.02)
_WHITE = (1.0, 1.0, 1.0)
_WHITE_DIFF = ('ND_oren_nayar_diffuse_bsdf', {'color': _WHITE})
_OPENPBR = 'ND_open_pbr_surface_surfaceshader'
_STDSURF = 'ND_standard_surface_surfaceshader'
_GLTF = 'ND_gltf_pbr_surfaceshader'


@pytest.fixture(autouse=True)
def _set_default_variant() -> None:
  mtlx.set_default_variant()


def _cond(r: float, **extra: Any) -> dict[str, Any]:
  return {
      'ior': _WHITE,
      'extinction': (1e4, 1e4, 1e4),
      'roughness': (r, r),
      **extra,
  }


def _diel(
    mode: str, r: float, ior: float = 1.5, **extra: Any
) -> dict[str, Any]:
  return {'ior': ior, 'roughness': (r, r), 'scatter_mode': mode, **extra}


def _case(
    name: str,
    shader_id: str,
    kwargs: dict[str, Any],
    cos: tuple[float, ...] = _FRONT_COS,
    bounds: tuple[float, float] = (0.985, 1.02),
    **opts: Any,
) -> Any:
  return pytest.param(shader_id, kwargs, cos, bounds, opts, id=name)


def _assert_weak_furnace(
    bsdf: mi.BSDF,
    cos_values: tuple[float, ...],
    bounds: tuple[float, float],
    *,
    importance: bool = False,
    n_side: int = 128,
    max_weight: float | None = None,
) -> None:
  n_per = n_side * n_side
  n_groups = len(cos_values)
  n_total = n_groups * n_per
  cos_arr = np.repeat(np.asarray(cos_values, dtype=np.float32), n_per)
  sin_arr = np.sqrt(np.maximum(0.0, 1.0 - cos_arr * cos_arr))
  idx = np.arange(n_total, dtype=np.int32)

  si = dr.zeros(mi.SurfaceInteraction3f, n_total)
  si.n = mi.Normal3f(0.0, 0.0, 1.0)
  si.sh_frame = mi.Frame3f(
      mi.Vector3f(1.0, 0.0, 0.0),
      mi.Vector3f(0.0, 1.0, 0.0),
      mi.Normal3f(0.0, 0.0, 1.0),
  )
  si.dp_du, si.dp_dv = mi.Vector3f(1.0, 0.0, 0.0), mi.Vector3f(0.0, 1.0, 0.0)
  si.uv = mi.Point2f(
      ((idx % 4) / 3.0).astype(np.float32),
      (((idx // 4) % 4) / 3.0).astype(np.float32),
  )
  si.wi = dr.normalize(
      mi.Vector3f(sin_arr * math.cos(0.37), sin_arr * math.sin(0.37), cos_arr)
  )
  if mi.is_spectral:
    si.wavelengths = mi.Spectrum(450.0, 530.0, 610.0, 680.0)

  rng = mi.PCG32(size=n_total, initstate=42)
  local = dr.arange(mi.UInt32, n_total) % n_per
  u1 = (mi.Float((local * 7919 + 104729) % n_per) + rng.next_float32()) / n_per
  u2 = mi.Point2f(
      (mi.Float(local % n_side) + rng.next_float32()) / n_side,
      (mi.Float(local // n_side) + rng.next_float32()) / n_side,
  )
  ctx = mi.BSDFContext(
      mi.TransportMode.Importance if importance else mi.TransportMode.Radiance
  )
  bs, w = bsdf.sample(ctx, si, u1, u2)
  val, pdf = bsdf.eval_pdf(ctx, si, bs.wo)

  for arr_v in (w, val):
    assert dr.all(dr.isfinite(arr_v), axis=None)
    assert dr.all(arr_v >= -1e-6, axis=None)
  assert dr.all(dr.isfinite(bs.pdf)) and dr.all(dr.isfinite(pdf))
  valid = (bs.pdf > 1e-5) & ~bs.is_null()
  assert dr.allclose(
      dr.select(valid, bs.pdf, 0.0),
      dr.select(valid, pdf, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )
  assert dr.allclose(
      dr.select(valid, w * bs.pdf, 0.0),
      dr.select(valid, val, 0.0),
      rtol=1e-3,
      atol=1e-4,
  )

  arr = np.asarray(w if importance else w * dr.square(bs.eta), dtype=np.float64)
  if arr.shape[0] != n_total:
    arr = arr.T
  rho = arr.reshape(n_groups, n_per, -1).mean(axis=1)
  lo, hi = bounds
  assert lo <= float(rho.min()) and float(rho.max()) <= hi, (
      f'Weak furnace albedo out of [{lo}, {hi}]: '
      f'{dict(zip(cos_values, rho.mean(axis=1).round(4)))}'
  )
  if max_weight is not None:
    assert (
        float(arr.max()) <= max_weight
    ), f'Weight spike {float(arr.max()):.2f}'


# Targeted weak-furnace cases complementing the 9 materials in
# `test_assets/materials/materialx_furnace.usda`.
_WEAK_CASES = [
    _case(
        'oren_nayar_comp_r1',
        'ND_oren_nayar_diffuse_bsdf',
        {'color': _WHITE, 'roughness': 1.0, 'energy_compensation': True},
        _BOTH_COS,
        (0.99, 1.015),
    ),
    _case(
        'subsurface',
        'ND_subsurface_bsdf',
        {'color': _WHITE},
        _BOTH_COS,
        (0.99, 1.01),
    ),
    _case(
        'conductor_smooth_grazing',
        'ND_conductor_bsdf',
        _cond(0.0),
        _BOTH_COS,
        (0.99, 1.01),
    ),
    _case('conductor_r1', 'ND_conductor_bsdf', _cond(1.0), bounds=(0.97, 1.04)),
    _case(
        'conductor_retro', 'ND_conductor_bsdf', _cond(0.5, retroreflective=True)
    ),
    _case(
        'dielectric_RT_thin_retro',
        'ND_dielectric_bsdf',
        _diel('RT', 0.0, retroreflective=True),
        _BOTH_COS,
        (0.985, 1.035),
        thin_walled=True,
    ),
    _case(
        'dielectric_RT_ior_lt_1',
        'ND_dielectric_bsdf',
        _diel('RT', 0.0, ior=0.67),
        (0.99, 0.75, 0.5, -0.99, -0.5, -0.25),
    ),
    _case(
        'dielectric_RT_importance',
        'ND_dielectric_bsdf',
        _diel('RT', 0.0),
        (0.8, 0.5, -0.8, -0.5),
        importance=True,
    ),
    _case(
        'layer_diel_smooth_diffuse',
        'ND_layer_bsdf',
        {'top': ('ND_dielectric_bsdf', _diel('R', 0.0)), 'base': _WHITE_DIFF},
        _BOTH_COS,
        (0.99, 1.01),
    ),
    _case(
        'layer_diel_r1_ior25_diffuse',
        'ND_layer_bsdf',
        {
            'top': ('ND_dielectric_bsdf', _diel('R', 1.0, ior=2.5)),
            'base': _WHITE_DIFF,
        },
        _BOTH_COS,
        (0.975, 1.02),
    ),
    _case(
        'layer_diel_translucent',
        'ND_layer_bsdf',
        {
            'top': ('ND_dielectric_bsdf', _diel('R', 0.5)),
            'base': ('ND_translucent_bsdf', {'color': _WHITE}),
        },
        _BOTH_COS,
    ),
    _case(
        'layer_mixed_top_diffuse',
        'ND_layer_bsdf',
        {
            'top': (
                'ND_mix_bsdf',
                {
                    'fg': ('ND_dielectric_bsdf', _diel('R', 0.5)),
                    'bg': (
                        'ND_sheen_bsdf',
                        {'color': _WHITE, 'roughness': 0.5, 'mode': 'zeltner'},
                    ),
                    'mix': 0.5,
                },
            ),
            'base': _WHITE_DIFF,
        },
    ),
    _case(
        'chiang_hair_tilted_sharp',
        'ND_chiang_hair_bsdf',
        {
            'tint_R': _WHITE,
            'tint_TT': _WHITE,
            'tint_TRT': _WHITE,
            'roughness_R': (0.05, 0.1),
            'roughness_TT': (0.05, 0.1),
            'roughness_TRT': (0.1, 0.1),
            'absorption_coefficient': Gf.Vec3f(0.0, 0.0, 0.0),
            'cuticle_angle': 0.55,
        },
        bounds=(0.99, 1.01),
    ),
    _case(
        'mix_surfaceshader_opacity',
        'ND_mix_surfaceshader',
        {
            'fg': ('ND_surface', {'bsdf': _WHITE_DIFF, 'opacity': 0.3}),
            'bg': ('ND_surface', {'bsdf': _WHITE_DIFF, 'opacity': 0.8}),
            'mix': 0.4,
        },
        bounds=(0.99, 1.01),
    ),
    _case(
        'openpbr_dielectric_specw2',
        _OPENPBR,
        {
            'base_color': _WHITE,
            'specular_weight': 2.0,
            'specular_roughness': 0.5,
        },
    ),
    _case(
        'openpbr_rough_glass_critical_angle',
        _OPENPBR,
        {
            'transmission_weight': 1.0,
            'specular_roughness': 0.3,
            'specular_ior': 1.5,
        },
        (-0.99, -0.85, -0.74, -0.5, 0.5, 0.99),
        (0.94, 1.03),
        max_weight=3.0,
    ),
    _case(
        'openpbr_partial_glass',
        _OPENPBR,
        {
            'base_color': _WHITE,
            'transmission_weight': 0.5,
            'specular_roughness': 0.3,
            'specular_ior': 1.5,
        },
        (-0.99, -0.5, 0.5, 0.99),
        (0.97, 1.02),
        max_weight=3.0,
    ),
    _case(
        'std_surface_partial_crystal_tir',
        _STDSURF,
        {
            'base': 0.0,
            'transmission': 0.94,
            'specular_roughness': 0.01,
            'specular_IOR': 1.7,
        },
        (-0.75, -0.5, -0.2),
        (0.99, 1.01),
    ),
    # `gltf_pbr.mtlx` uses default Conty-Kulla sheen (~0.967 at grazing).
    _case(
        'gltf_pbr_dielectric_sheen',
        _GLTF,
        {
            'base_color': _WHITE,
            'metallic': 0.0,
            'roughness': 0.5,
            'sheen_color': _WHITE,
            'sheen_roughness': 0.5,
        },
        bounds=(0.96, 1.03),
    ),
]


@pytest.mark.parametrize(
    'shader_id, kwargs, cos_values, bounds, opts', _WEAK_CASES
)
def test_weak_furnace(
    shader_id: str,
    kwargs: dict[str, Any],
    cos_values: tuple[float, ...],
    bounds: tuple[float, float],
    opts: dict[str, Any],
) -> None:
  """Tests directional-hemispherical albedo and sampling across angles.

  Args:
    shader_id: MaterialX BSDF or surface shader NodeDef identifier.
    kwargs: Input parameters passed to the shader.
    cos_values: Incident `cos(theta_i)` values to evaluate.
    bounds: Allowed `(min, max)` directional albedo range.
    opts: Optional `thin_walled`, `importance`, or `max_weight` overrides.
  """
  if shader_id.endswith('_bsdf'):
    surface_kw = {'thin_walled': True} if opts.get('thin_walled') else {}
    bsdf, _ = mtlx.compile_mtlx_material(
        'ND_surface', bsdf=(shader_id, kwargs), **surface_kw
    )
  else:
    bsdf, _ = mtlx.compile_mtlx_material(shader_id, **kwargs)
  assert bsdf is not None
  _assert_weak_furnace(
      bsdf,
      cos_values,
      bounds,
      importance=opts.get('importance', False),
      max_weight=opts.get('max_weight'),
  )


@pytest.mark.parametrize('spectral', [False, True], ids=['rgb', 'spectral'])
def test_strong_furnace_scene(spectral: bool) -> None:
  """Tests weak and multi-bounce furnace on the on-disk 9-sphere USDA scene.

  Args:
    spectral: Whether to run on a spectral variant instead of RGB.
  """
  if spectral:
    mtlx.set_variant_or_skip('cuda_ad_spectral', 'llvm_ad_spectral')

  res = 128
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/materialx_furnace.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(res, res), spp=128)
  scene_dict = usd_mitsuba_translator.convert_to_mitsuba(stage)
  scene_dict['integrator'] = {'type': 'path', 'max_depth': 32, 'rr_depth': 32}
  scene = mi.load_dict(scene_dict)
  img = np.asarray(mi.render(scene, spp=128, seed=123), dtype=np.float64)[
      ..., :3
  ]
  assert np.all(np.isfinite(img)) and np.all(img >= -1e-6)

  ys, xs = np.mgrid[0:res, 0:res]
  pos = mi.Point2f(
      ((xs.ravel() + 0.5) / res).astype(np.float32),
      ((ys.ravel() + 0.5) / res).astype(np.float32),
  )
  rays, _ = scene.sensors()[0].sample_ray(0.0, 0.5, pos, mi.Point2f(0.5, 0.5))
  si = scene.ray_intersect(rays)
  hit = np.asarray(si.is_valid()).reshape(res, res)
  cos_v = np.asarray(dr.abs(si.wi.z)).reshape(res, res)
  assert np.allclose(img[~hit].mean(axis=0), 1.0, atol=0.015)

  shapes = scene.shapes()
  assert len(shapes) == 9
  for shape in shapes:
    mask = np.asarray(si.shape == shape).reshape(res, res)
    assert mask.sum() > 200, f'Sphere {shape.id()} not visible in camera'
    is_nmap = shape.id().endswith('openpbr_normalmap')
    int_lo, rim_lo = (0.90, 0.87) if is_nmap else (0.975, 0.95)
    int_rgb = img[mask & (cos_v > 0.3)].mean(axis=0)
    rim_rgb = img[mask & (cos_v <= 0.3)].mean(axis=0)
    assert np.all((int_rgb >= int_lo) & (int_rgb <= 1.025)), (
        shape.id(),
        int_rgb,
    )
    assert np.all((rim_rgb >= rim_lo) & (rim_rgb <= 1.03)), (
        shape.id(),
        rim_rgb,
    )
    if not is_nmap:
      _assert_weak_furnace(
          shape.bsdf(), (0.1, 0.5, 0.99), (0.95, 1.035), n_side=96
      )
