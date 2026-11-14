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

"""Tests for MaterialX image textures, UV address modes, alpha, and spectral upsampling."""

from __future__ import annotations

import pathlib
from typing import Any

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


def test_image_tiledimage_triplanar_and_normalmap() -> None:
  """Tests Hydra/offline parity for tiledimage, triplanarprojection, and normalmap via emission."""
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  albedo = Sdf.AssetPath(mtlx_helpers.ALBEDO_PATH)
  normal_vec = (
      'ND_normalmap',
      {
          'in_': (
              'ND_image_vector3',
              {'file': Sdf.AssetPath(mtlx_helpers.NORMAL_PATH)},
          ),
          'scale': 0.5,
      },
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission_color=(
          'ND_mix_color3',
          {
              'fg': (
                  'ND_mix_color3',
                  {
                      'fg': (
                          'ND_tiledimage_color3',
                          {'file': albedo, 'uvtiling': (2.0, 2.0)},
                      ),
                      'bg': (
                          'ND_triplanarprojection_color3',
                          {'filex': albedo, 'filey': albedo, 'filez': albedo},
                      ),
                      'mix': 0.5,
                  },
              ),
              'bg': ('ND_convert_vector3_color3', {'in_': normal_vec}),
              'mix': 0.25,
          },
      ),
  )
  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, spp=4, subdivision_level=0
  )
  assert np.mean(mtlx_helpers.render_center(image_hd, half_size=4)) > 0.05


def test_place2d_triplanar_and_tiledimage_exact(tmp_path: pathlib.Path) -> None:
  """Tests exact analytical outputs of ND_place2d, ND_triplanarprojection, and ND_tiledimage.

  Args:
    tmp_path: Temporary directory fixture.
  """
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8
  )
  # ND_place2d_vector2 SRT (operationorder=0) vs TRS (operationorder=1).
  p2d_kw = {
      'texcoord': (0.5, 0.0),
      'pivot': (0.0, 0.0),
      'scale': (2.0, 1.0),
      'rotate': 90.0,
      'offset': (0.1, 0.2),
  }
  np.testing.assert_allclose(
      mtlx_helpers.eval_emission_expr(
          ('ND_place2d_vector2', {**p2d_kw, 'operationorder': 0}),
          stage=stage,
          channels=2,
      ),
      (-0.1, -0.45),
      atol=1e-4,
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_emission_expr(
          ('ND_place2d_vector2', {**p2d_kw, 'operationorder': 1}),
          stage=stage,
          channels=2,
      ),
      (-0.1, -0.4),
      atol=1e-4,
  )

  # 2x2 EXR: top row [Red, Green], bottom row [Blue, Yellow].
  tex_path = Sdf.AssetPath(str(tmp_path / 'quad2x2.exr'))
  tex_arr = np.array(
      [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]], dtype=np.float32
  )
  mi.Bitmap(tex_arr).write(tex_path.path)

  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission_color=(
          'ND_tiledimage_color3',
          {
              'file': tex_path,
              'texcoord': (0.125, 0.375),
              'uvtiling': (2.0, 2.0),
              'uvoffset': (0.0, 0.5),
              'filtertype': 'closest',
          },
      ),
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.0, 0.0, 1.0), atol=1e-4
  )

  tri = mtlx_helpers.define_shader(
      stage,
      'Triplanar',
      'ND_triplanarprojection_color3',
      filex=tex_path,
      filey=tex_path,
      filez=tex_path,
      position=Gf.Vec3f(0.25, 0.75, 0.0),
      normal=Gf.Vec3f(0.0, 0.0, 1.0),
      upaxis=2,
      filtertype='closest',
  )
  mtlx_helpers.bind_mtlx_shader(
      stage, mesh, 'ND_surface_unlit', emission_color=tri
  )
  # Z-up samples filez at (0.25, 0.75) -> Red; X-up uses (-0.75, 0.25) -> Blue.
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (1.0, 0.0, 0.0), atol=1e-4
  )
  tri.GetInput('upaxis').Set(0)
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.0, 0.0, 1.0), atol=1e-4
  )


def test_image_address_modes(tmp_path: pathlib.Path) -> None:
  """Tests UV address modes (`clamp`, `constant`, `periodic`, `mirror`) and colorspace.

  Args:
    tmp_path: Temporary directory fixture.
  """
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8
  )
  tex_path = str(tmp_path / 'ramp2x1.exr')
  mi.Bitmap(
      np.array([[[0.2, 0.4, 0.6], [0.8, 0.2, 0.1]]], dtype=np.float32)
  ).write(tex_path)
  uv_node = mtlx_helpers.define_shader(
      stage, 'UvConst', 'ND_constant_vector2', value=(1.5, 0.5)
  )
  img_node = mtlx_helpers.define_shader(
      stage,
      'Img',
      'ND_image_color3',
      file=Sdf.AssetPath(tex_path),
      texcoord=uv_node,
      uaddressmode='clamp',
      vaddressmode='constant',
      default=(0.9, 0.7, 0.3),
      filtertype='closest',
  )
  mtlx_helpers.bind_mtlx_shader(
      stage, mesh, 'ND_surface_unlit', emission_color=img_node
  )
  # UV (1.5, 0.5) clamps U to 1.0 (right texel) and is in-bounds in V.
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.8, 0.2, 0.1), atol=1e-4
  )
  # UV (0.5, 1.5) is out-of-bounds in V (`constant` mode -> returns `default`).
  uv_node.GetInput('value').Set((0.5, 1.5))
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.9, 0.7, 0.3), atol=1e-4
  )

  # Mirror vs periodic wrap modes along U.
  img_node.GetInput('uaddressmode').Set('mirror')
  img_node.GetInput('vaddressmode').Set('periodic')
  uv_node.GetInput('value').Set((1.25, 0.5))
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.8, 0.2, 0.1), atol=1e-4
  )
  uv_node.GetInput('value').Set((-0.25, 0.5))
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.2, 0.4, 0.6), atol=1e-4
  )
  img_node.GetInput('uaddressmode').Set('periodic')
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), (0.8, 0.2, 0.1), atol=1e-4
  )

  # Colorspace: raw skips sRGB decoding whereas srgb_texture applies it.
  gray_path = str(tmp_path / 'gray.png')
  mi.Bitmap(
      np.full((1, 1, 3), 128, dtype=np.uint8), mi.Bitmap.PixelFormat.RGB
  ).write(gray_path)
  file_in = img_node.GetInput('file')
  file_in.Set(Sdf.AssetPath(gray_path))
  file_in.GetAttr().SetColorSpace('raw')
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage), 128.0 / 255.0, atol=1e-3
  )
  file_in.GetAttr().SetColorSpace('srgb_texture')
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage),
      ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4,
      atol=1e-3,
  )


@pytest.mark.parametrize(
    'variants',
    [('cuda_ad_rgb', 'llvm_ad_rgb'), ('cuda_ad_spectral', 'llvm_ad_spectral')],
)
def test_image_alpha(variants: tuple[str, str], tmp_path: pathlib.Path) -> None:
  """Tests that the alpha channel of color4 images is read from the file.

  Args:
    variants: Candidate Mitsuba variants to test.
    tmp_path: Temporary directory fixture.
  """
  mi.set_variant(*variants)
  rgba_path = str(tmp_path / 'rgba.png')
  rgba = np.full((4, 4, 4), (255, 255, 255, 128), dtype=np.uint8)
  mi.Bitmap(rgba, mi.Bitmap.PixelFormat.RGBA).write(rgba_path)

  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, quad_half_size=0.8, resolution=(16, 16), spp=4
  )

  def _alpha(path: str) -> tuple[str, dict[str, Any]]:
    return (
        'ND_extract_color4',
        {'in_': ('ND_image_color4', {'file': Sdf.AssetPath(path)}), 'index': 3},
    )

  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission_color=(
          'ND_combine3_color3',
          {
              'in1': _alpha(rgba_path),
              'in2': _alpha(mtlx_helpers.ALBEDO_PATH),
              'in3': 0.2,
          },
      ),
  )
  if mi.is_spectral:
    np.testing.assert_allclose(
        mtlx_helpers.render_center(stage),
        (128.0 / 255.0, 1.0, 0.2),
        rtol=0.05,
        atol=0.02,
    )
  else:
    np.testing.assert_allclose(
        mtlx_helpers.eval_unlit(stage), (128.0 / 255.0, 1.0, 0.2), atol=1e-4
    )


def test_spectral_variant_textures() -> None:
  """Tests image textures and colors in a spectral variant via emission."""
  mi.set_variant('cuda_ad_spectral', 'llvm_ad_spectral')

  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  albedo = Sdf.AssetPath(mtlx_helpers.ALBEDO_PATH)
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_standard_surface_surfaceshader',
      emission=('ND_image_float', {'file': albedo}),
      emission_color=(
          'ND_multiply_color3',
          {
              'in1': ('ND_image_color3', {'file': albedo}),
              'in2': (1.0, 0.2, 0.2),
          },
      ),
  )

  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, spp=4, subdivision_level=0
  )
  center = mtlx_helpers.render_center(image_hd, half_size=4)
  assert center[0] > 0.02
  assert center[0] > center[1] * 2.0


def test_spectral_upsampling_and_d65_emitter() -> None:
  """Tests spectral upsampling and D65 emitter wrapping in spectral variants."""
  mi.set_variant('cuda_ad_spectral', 'llvm_ad_spectral')
  stage = Usd.Stage.CreateInMemory()

  def _compile_emitter(rgb: tuple[float, float, float]) -> mi.Emitter:
    _, emitter = mtlx_helpers.compile_mtlx_material(
        'ND_surface_unlit', stage=stage, emission_color=rgb
    )
    assert emitter is not None
    return emitter

  emitter_hdr = _compile_emitter((2.0, 0.8, 0.4))
  emitter_hi = _compile_emitter((1.0, 0.4, 0.2))
  emitter_lo = _compile_emitter((0.25, 0.1, 0.05))
  emitter_gray = _compile_emitter((0.3, 0.3, 0.3))
  emitter_white = _compile_emitter((1.0, 1.0, 1.0))

  si = mtlx_helpers.make_test_si(wi=(0.0, 0.0, 1.0))
  si.wavelengths = mi.Spectrum(420.0, 500.0, 580.0, 660.0)
  coeffs = mi.srgb_model_fetch(mi.Color3f(1.0, 0.4, 0.2))
  expected_refl = mi.srgb_model_eval(coeffs, si.wavelengths)
  np.testing.assert_allclose(
      emitter_hi.eval(si),
      emitter_white.eval(si) * expected_refl,
      rtol=1e-4,
      atol=1e-4,
  )
  # Homogeneous upsampling commutes with positive scalar scaling (including > 1).
  np.testing.assert_allclose(
      emitter_hdr.eval(si), 2.0 * emitter_hi.eval(si), rtol=1e-4, atol=1e-4
  )
  np.testing.assert_allclose(
      emitter_hi.eval(si), 4.0 * emitter_lo.eval(si), rtol=1e-4, atol=1e-4
  )
  # Neutral gray upsamples to a flat 0.3x multiple of D65 across wavelengths.
  np.testing.assert_allclose(
      emitter_gray.eval(si) / emitter_white.eval(si), 0.3, rtol=1e-3, atol=1e-3
  )

