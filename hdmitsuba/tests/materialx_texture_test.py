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

"""Tests for MaterialX image textures, color spaces, primvars, and geometric intrinsics."""

from __future__ import annotations

import pathlib

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
def _enable_materialx_and_default_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  mtlx.enable_materialx_and_default_variant(monkeypatch)


_UV_QUADRANTS = Sdf.AssetPath(
    f'{test_helpers.TEST_ASSETS_PATH}/materials/textures/uv_quadrants.png'
)


def _write_bitmap(
    path: pathlib.Path, data: object, dtype: type = np.float32
) -> Sdf.AssetPath:
  mi.Bitmap(np.asarray(data, dtype=dtype)).write(path.as_posix())
  return Sdf.AssetPath(path.as_posix())


def test_place2d_tiledimage_and_triplanar() -> None:
  """Tests ND_place2d, ND_tiledimage, and ND_triplanarprojection expression evaluation."""
  stage = Usd.Stage.CreateInMemory()
  p2d_kw = {
      'texcoord': (0.5, 0.0),
      'scale': (2.0, 1.0),
      'rotate': 90.0,
      'offset': (0.1, 0.2),
  }
  for order, expected_uv in [(0, (-0.1, -0.45)), (1, (-0.1, -0.4))]:
    np.testing.assert_allclose(
        mtlx.eval_emission_expr(
            ('ND_place2d_vector2', {**p2d_kw, 'operationorder': order}),
            stage=stage,
            channels=2,
        ),
        expected_uv,
        atol=1e-4,
    )

  # `uv_quadrants.png`: TL (200, 40, 80), TR (40, 200, 120),
  # BL (60, 80, 220), BR (160, 160, 60).
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_tiledimage_color3',
              {
                  'file': _UV_QUADRANTS,
                  'texcoord': (0.375, 0.3),
                  'uvtiling': (2.0, 2.0),
                  'uvoffset': (0.0, 0.25),
                  'filtertype': 'closest',
              },
          ),
          stage=stage,
      ),
      np.array([160.0, 160.0, 60.0]) / 255.0,
      atol=1e-4,
  )

  for upaxis, expected_rgb in [
      (2, np.array([200.0, 40.0, 80.0]) / 255.0),
      (0, np.array([60.0, 80.0, 220.0]) / 255.0),
  ]:
    np.testing.assert_allclose(
        mtlx.eval_emission_expr(
            (
                'ND_triplanarprojection_color3',
                {
                    'filex': _UV_QUADRANTS,
                    'filey': _UV_QUADRANTS,
                    'filez': _UV_QUADRANTS,
                    'position': Gf.Vec3f(0.25, 0.75, 0.0),
                    'normal': Gf.Vec3f(0.0, 0.0, 1.0),
                    'upaxis': upaxis,
                    'filtertype': 'closest',
                },
            ),
            stage=stage,
        ),
        expected_rgb,
        atol=1e-4,
    )


def test_ifgreater_distinct_images() -> None:
  """Tests that distinct `<image>` inputs in `ifgreater` stay distinct during discovery."""

  def _img(uv: tuple[float, float]) -> dict[str, object]:
    return {'file': _UV_QUADRANTS, 'texcoord': uv, 'filtertype': 'closest'}

  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_ifgreater_color3',
              {
                  'value1': ('ND_image_float', _img((0.25, 0.75))),
                  'value2': ('ND_image_float', _img((0.75, 0.75))),
                  'in1': ('ND_image_color3', _img((0.25, 0.25))),
                  'in2': (0.0, 0.0, 0.0),
              },
          ),
      ),
      np.array([60.0, 80.0, 220.0]) / 255.0,
      atol=1e-4,
  )


def test_image_channels_and_dead_branch_discovery(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
  """Tests grayscale/RGB channel mapping and dead-branch texture discovery.

  Args:
    tmp_path: Temporary directory fixture.
    capfd: Pytest fd capture fixture.
  """
  # Single-channel grayscale bitmap read by `ND_image_color3` broadcasts across
  # RGB, while `ND_image_float` on a 3-channel RGB bitmap extracts `.r` (not
  # luminance). Dead `<image>` inputs (`base_color`, `0 * image`) are not
  # preloaded.
  mono_path = _write_bitmap(tmp_path / 'mono1x1.exr', [[[0.65]]])
  rgb_asym_path = _write_bitmap(tmp_path / 'rgb_asym.exr', [[[0.8, 0.1, 0.1]]])
  unused_missing = Sdf.AssetPath((tmp_path / 'unused_missing.exr').as_posix())

  capfd.readouterr()
  _, emitter = mtlx.compile_mtlx_material(
      'ND_standard_surface_surfaceshader',
      base=0.0,
      base_color=('ND_image_color3', {'file': unused_missing}),
      emission=(
          'ND_add_float',
          {
              'in1': ('ND_image_float', {'file': rgb_asym_path}),
              'in2': (
                  'ND_multiply_float',
                  {
                      'in1': ('ND_constant_float', {'value': 0.0}),
                      'in2': ('ND_image_float', {'file': unused_missing}),
                  },
              ),
          },
      ),
      emission_color=(
          'ND_image_color3',
          {
              'file': (
                  'ND_dot_filename',
                  {
                      'in_': (
                          'ND_constant_filename',
                          {'value': mono_path},
                      )
                  },
              )
          },
      ),
  )
  assert 'unused_missing.exr' not in capfd.readouterr().err
  assert emitter is not None
  np.testing.assert_allclose(
      emitter.eval(mtlx.make_test_si(wi=(0.0, 0.0, 1.0))),
      mi.Color3f(0.52, 0.52, 0.52),
      atol=1e-5,
  )


@pytest.mark.skipif(
    Sdf.FileFormat.FindByExtension('mtlx') is None,
    reason='Requires an OpenUSD build with MaterialX support.',
)
def test_source_asset_fileprefix(capfd: pytest.CaptureFixture[str]) -> None:
  """Tests custom .mtlx sourceAsset graphs with fileprefix, overrides, and empty filenames.

  Args:
    capfd: Pytest fd capture fixture.
  """
  stage = Usd.Stage.CreateInMemory()
  custom = UsdShade.Shader.Define(stage, '/mesh/Material/CustomFp')
  custom.SetSourceAsset(
      Sdf.AssetPath(
          f'{test_helpers.TEST_ASSETS_PATH}/materials/fileprefix_test.mtlx'
      ),
      'mtlx',
  )
  custom.SetSourceAssetSubIdentifier('ND_hdmitsuba_fp_test', 'mtlx')
  custom.CreateInput('override_tex', Sdf.ValueTypeNames.Asset).Set(
      _UV_QUADRANTS
  )
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si.uv = mi.Point2f(0.25, 0.25)
  capfd.readouterr()
  actual = mtlx.eval_emission_expr(
      custom.CreateOutput('out', Sdf.ValueTypeNames.Color3f),
      stage=stage,
      si=si,
  )
  assert 'Failed to load texture' not in capfd.readouterr().err
  # `(60, 80, 220)/255 + (200, 40, 80)/255 + (0.1, 0.1, 0.1)`.
  np.testing.assert_allclose(
      actual, np.array([260.0, 120.0, 300.0]) / 255.0 + 0.1, atol=1e-4
  )


def test_image_address_modes(tmp_path: pathlib.Path) -> None:
  """Tests UV address modes (`clamp`, `constant`, `black`, `periodic`, `mirror`).

  Args:
    tmp_path: Temporary directory fixture.
  """
  stage = Usd.Stage.CreateInMemory()
  tex_path = _write_bitmap(
      tmp_path / 'ramp2x1.exr', [[[0.2, 0.4, 0.6], [0.8, 0.2, 0.1]]]
  )
  fallback_path = _write_bitmap(
      tmp_path / 'fallback1x1.exr', [[[0.9, 0.7, 0.3]]]
  )

  # Unauthored `texcoord` hits the default-UV `Op::SampleInput` fast path for
  # uniform wrap modes (`periodic`, `mirror`, `clamp`) and `Op::SampleInputUV`
  # for mixed / `constant` modes.
  si = dr.zeros(mi.SurfaceInteraction3f, 2)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  img_node = mtlx.define_shader(
      stage,
      'Img',
      'ND_image_color3',
      file=tex_path,
      default=('ND_image_color3', {'file': fallback_path}),
      filtertype='closest',
  )
  for umode, vmode, uvs, expected in [
      (
          'clamp',
          'constant',
          ([1.5, 0.5], [0.5, -0.5]),
          [(0.8, 0.2, 0.1), (0.9, 0.7, 0.3)],
      ),
      (
          'black',
          'constant',
          ([1.5, 0.5], [0.5, -0.5]),
          [(0.0, 0.0, 0.0), (0.9, 0.7, 0.3)],
      ),
      (
          'mirror',
          'periodic',
          ([1.75, -0.75], [0.5, 0.5]),
          [(0.2, 0.4, 0.6), (0.8, 0.2, 0.1)],
      ),
      (
          'periodic',
          'periodic',
          ([-0.25, 0.25], [0.5, 0.5]),
          [(0.8, 0.2, 0.1), (0.2, 0.4, 0.6)],
      ),
  ]:
    img_node.CreateInput('uaddressmode', Sdf.ValueTypeNames.String).Set(umode)
    img_node.CreateInput('vaddressmode', Sdf.ValueTypeNames.String).Set(vmode)
    si.uv = mi.Point2f(*uvs)
    np.testing.assert_allclose(
        mtlx.eval_emission_expr(img_node, stage=stage, si=si),
        expected,
        atol=1e-4,
    )


def test_image_and_value_colorspaces(tmp_path: pathlib.Path) -> None:
  """Tests image and value colorspace conversions (including floating-point images).

  Args:
    tmp_path: Temporary directory fixture.
  """
  stage = Usd.Stage.CreateInMemory()
  gray_path = _write_bitmap(
      tmp_path / 'gray.png', [[[128, 128, 128]]], dtype=np.uint8
  )
  aces = np.array([0.5, 0.25, 0.1])
  aces_path = _write_bitmap(tmp_path / 'aces.exr', aces[None, None])
  gray = 128.0 / 255.0
  img_node = mtlx.define_shader(stage, 'Img', 'ND_image_color3')
  file_in = img_node.CreateInput('file', Sdf.ValueTypeNames.Asset)
  for path, color_space, expected in [
      (gray_path, None, (gray,) * 3),
      (gray_path, 'raw', (gray,) * 3),
      (gray_path, 'srgb_texture', (((gray + 0.055) / 1.055) ** 2.4,) * 3),
      (gray_path, 'g22_rec709', (gray**2.2,) * 3),
      (aces_path, 'acescg', (0.68875, 0.21902, 0.07105)),
      (aces_path, 'srgb_texture', ((aces + 0.055) / 1.055) ** 2.4),
  ]:
    file_in.Set(path)
    if color_space:
      file_in.GetAttr().SetColorSpace(color_space)
    else:
      file_in.GetAttr().ClearColorSpace()
    np.testing.assert_allclose(
        mtlx.eval_emission_expr(img_node, stage=stage),
        expected,
        atol=1e-3,
    )

  # Value `<input>` color space conversion to linear Rec.709.
  c_node = mtlx.define_shader(
      stage, 'ConstCs', 'ND_constant_color3', value=(0.5, 0.25, 0.1)
  )
  c_node.GetInput('value').GetAttr().SetColorSpace('lin_ciexyzd65_scene')
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(c_node, stage=stage),
      (1.10155, 0.01291, 0.09055),
      atol=1e-4,
  )


@pytest.mark.parametrize(
    'variants',
    [('cuda_ad_rgb', 'llvm_ad_rgb'), ('cuda_ad_spectral', 'llvm_ad_spectral')],
)
def test_image_alpha_and_spectral_textures(
    variants: tuple[str, str], tmp_path: pathlib.Path
) -> None:
  """Tests RGBA/RGB color4 alpha, spectral image upsampling, and interactive commits.

  Args:
    variants: Candidate Mitsuba variants to test.
    tmp_path: Temporary directory fixture.
  """
  mtlx.set_variant_or_skip(*variants)
  rgba_path = _write_bitmap(
      tmp_path / 'rgba.png', [[[204, 102, 51, 128]]], dtype=np.uint8
  )
  rgb_path = _write_bitmap(
      tmp_path / 'rgb.png', [[[77, 153, 204]]], dtype=np.uint8
  )

  stage, mesh = mtlx.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  rgba = mtlx.define_shader(stage, 'Rgba', 'ND_image_color4', file=rgba_path)
  surf = mtlx.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission=1.0,
      emission_color=(
          'ND_combine3_color3',
          {
              'in1': ('ND_extract_color4', {'in_': rgba, 'index': 3}),
              'in2': (
                  'ND_multiply_float',
                  {
                      'in1': (
                          'ND_extract_color4',
                          {
                              'in_': ('ND_image_color4', {'file': rgb_path}),
                              'index': 3,
                          },
                      ),
                      'in2': ('ND_extract_color4', {'in_': rgba, 'index': 1}),
                  },
              ),
              'in3': ('ND_image_float', {'file': rgb_path}),
          },
      ),
  )
  engine = mtlx.create_engine(stage)
  img_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix=f'test_mtlx_alpha_{mi.variant()}',
      engine=engine,
      spp=4,
      subdivision_level=0,
  )
  expected = np.array([128.0, 102.0, 77.0]) / 255.0
  np.testing.assert_allclose(
      mtlx.render_center(img_hd), expected, rtol=0.05, atol=0.02
  )

  # Interactive edit keeps RGB `color4` alpha working without error.
  surf.GetInput('emission').Set(0.5)
  np.testing.assert_allclose(
      mtlx.render_center(engine), 0.5 * expected, rtol=0.05, atol=0.02
  )


def test_geompropvalue_and_geomcolor() -> None:
  """Tests primvar defaults (including color4 & missing st) and geomcolor4."""
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/materialx_geomprops.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(16, 16), spp=4)
  # `(0.5 * (1.0 + 0.2)) * (0.75 * (0.2, 0.4, 0.6) + (0.1, 0.2, 0.3) + (0.3, 0.25, 0.0))`
  # `= (0.33, 0.45, 0.45)`.
  img_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, output_prefix='test_mtlx_geomprops', spp=4, subdivision_level=0
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_hd), (0.33, 0.45, 0.45), atol=1e-2
  )


def test_view_and_normal_ops() -> None:
  """Tests facingratio, normalmap, transformnormal, bitangent handedness, and tangent space."""
  stage = Usd.Stage.CreateInMemory()
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          ('ND_facingratio_float', {'faceforward': False}),
          stage=stage,
          si=mtlx.make_test_si(wi=(0.6, 0.0, 0.8)),
          channels=1,
      ),
      0.8,
      atol=1e-5,
  )
  for shader_id, kw, expected in [
      (
          'ND_normalmap_float',
          {
              'in_': Gf.Vec3f(0.5, 0.8, 0.9),
              'normal': Gf.Vec3f(0.0, 0.0, 1.0),
              'tangent': Gf.Vec3f(1.0, 0.0, 0.0),
              'bitangent': Gf.Vec3f(0.0, -1.0, 0.0),
          },
          (0.0, -0.6, 0.8),
      ),
      # `transformnormal` passes unnormalized vectors through without normalizing.
      (
          'ND_transformnormal_vector3',
          {'in_': Gf.Vec3f(0.0, 0.0, 2.0)},
          (0.0, 0.0, 2.0),
      ),
      ('ND_normal_vector3', {'space': 'tangent'}, (0.0, 0.0, 1.0)),
      ('ND_tangent_vector3', {'space': 'tangent'}, (1.0, 0.0, 0.0)),
  ]:
    np.testing.assert_allclose(
        mtlx.eval_emission_expr((shader_id, kw), stage=stage),
        expected,
        atol=1e-5,
    )

  # World bitangent aligns with MaterialX `+v` (`-si.dp_dv`) on both right-handed
  # and mirrored UV charts.
  si = dr.zeros(mi.SurfaceInteraction3f, 2)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.sh_frame = mi.Frame3f(
      mi.Vector3f(1.0, 0.0, 0.0),
      mi.Vector3f(0.0, 1.0, 0.0),
      mi.Normal3f(0.0, 0.0, 1.0),
  )
  si.dp_du = mi.Vector3f([1.0, -1.0], [0.0, 0.0], [0.0, 0.0])
  si.dp_dv = mi.Vector3f(0.0, -1.0, 0.0)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          ('ND_bitangent_vector3', {'space': 'world'}), stage=stage, si=si
      ),
      [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
      atol=1e-5,
  )


def test_object_vs_world_space_position_and_normal() -> None:
  """Verifies object/world space intrinsics under non-uniform scale and rotation."""
  stage, quad = mtlx.create_base_scene(
      with_lights=False, quad_half_size=1.2, resolution=(16, 16), spp=4
  )
  # Under `translate(0.225, 0.2, 0.4) * rotateZ(180) * scale(9/16, 1, 1)`, the
  # camera center hit `p_world = (0, 0, 0.4)` corresponds to `p_obj = (0.4, 0.2, 0)`,
  # and tilted object normals `(0.6, 0, 0.8)` map to `n_world = (-0.8, 0, 0.6)`,
  # `t_world = (-0.6, 0, -0.8)`, and `b_world = (0, -1, 0)`.
  quad.AddTranslateOp().Set(Gf.Vec3d(0.225, 0.2, 0.4))
  quad.AddRotateZOp().Set(180.0)
  quad.AddScaleOp().Set(Gf.Vec3f(0.5625, 1.0, 1.0))
  quad.GetNormalsAttr().Set([(0.6, 0.0, 0.8)] * 4)

  engine = mtlx.create_engine(stage)
  for shader_id, space, expected in [
      ('ND_position_vector3', 'world', (0.5, 0.5, 0.7)),
      ('ND_normal_vector3', 'world', (0.1, 0.5, 0.8)),
      ('ND_tangent_vector3', 'world', (0.2, 0.5, 0.1)),
      ('ND_bitangent_vector3', 'world', (0.5, 0.0, 0.5)),
      ('ND_position_vector3', 'object', (0.7, 0.6, 0.5)),
      ('ND_normal_vector3', 'object', (0.8, 0.5, 0.9)),
  ]:
    mtlx.bind_mtlx_shader(
        stage,
        quad,
        'ND_surface_unlit',
        emission_color=(
            'ND_convert_vector3_color3',
            {
                'in_': (
                    'ND_add_vector3',
                    {
                        'in1': (
                            'ND_multiply_vector3FA',
                            {
                                'in1': (shader_id, {'space': space}),
                                'in2': 0.5,
                            },
                        ),
                        'in2': Gf.Vec3f(0.5, 0.5, 0.5),
                    },
                )
            },
        ),
    )
    np.testing.assert_allclose(
        mtlx.render_center(engine), expected, atol=1.5e-2
    )

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_intrinsics_parity',
      engine=engine,
      spp=4,
      subdivision_level=0,
  )
