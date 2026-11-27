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
from pxr import UsdGeom
from pxr import UsdShade
import pytest

from hdmitsuba import geometry_ext as geom_lib
from hdmitsuba.tests import materialx_test_helpers as mtlx
from hdmitsuba.tests import test_helpers
from usd_mitsuba import material as material_lib
from usd_mitsuba import translator as usd_mitsuba


@pytest.fixture(autouse=True)
def _set_default_variant() -> None:
  mtlx.set_default_variant()


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
      (gray_path, 'srgb_tx', (((gray + 0.055) / 1.055) ** 2.4,) * 3),
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
  for node_spec, expected in [
      (('ND_position_vector3', {'space': 'world'}), (0.5, 0.5, 0.7)),
      (('ND_normal_vector3', {'space': 'world'}), (0.1, 0.5, 0.8)),
      (('ND_tangent_vector3', {'space': 'world'}), (0.2, 0.5, 0.1)),
      (('ND_bitangent_vector3', {'space': 'world'}), (0.5, 0.0, 0.5)),
      (('ND_position_vector3', {'space': 'object'}), (0.7, 0.6, 0.5)),
      (('ND_normal_vector3', {'space': 'object'}), (0.8, 0.5, 0.9)),
      (('ND_geompropvalue_vector3', {'geomprop': 'Pobject'}), (0.7, 0.6, 0.5)),
      (('ND_geompropvalue_vector3', {'geomprop': 'Nobject'}), (0.8, 0.5, 0.9)),
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
                                'in1': node_spec,
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


def test_geomprop_alias_nodes(
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests built-in geometric `geompropvalue` aliases and channel coercion.

  Args:
    capfd: Pytest file-descriptor capture fixture for verifying warnings.
  """
  stage = Usd.Stage.CreateInMemory()
  si = mtlx.make_test_si()
  si.p = mi.Point3f(0.2, 0.4, 0.6)
  si.dp_dv = mi.Vector3f(0.0, -1.0, 0.0)
  si.uv = mi.Point2f(0.25, 0.75)

  for node_id, prop, channels, expected, warn in (
      ('ND_geompropvalue_vector3', 'Pworld', 3, (0.2, 0.4, 0.6), None),
      ('ND_geompropvalue_vector3', 'Nworld', 3, (0.0, 0.0, 1.0), None),
      ('ND_geompropvalue_vector3', 'Tworld', 3, (1.0, 0.0, 0.0), None),
      ('ND_geompropvalue_vector3', 'Bworld', 3, (0.0, 1.0, 0.0), None),
      (
          'ND_geompropvalue_vector3',
          'Tobject',
          3,
          (1.0, 0.0, 0.0),
          'Unsupported MaterialX tangent space',
      ),
      ('ND_geompropvalue_vector2', 'UV0', 2, (0.25, 0.25), None),
  ):
    np.testing.assert_allclose(
        mtlx.eval_emission_expr(
            (node_id, {'geomprop': prop}), stage=stage, si=si, channels=channels
        ),
        expected,
        atol=1e-5,
    )
    err = capfd.readouterr().err
    assert (warn in err) if warn else (not err)

  # Mismatched channel counts preserve scalar splatting and upper-lane ops.
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_add_vector3FA',
              {
                  'in1': ('ND_geompropvalue_vector3', {'geomprop': 'UV0'}),
                  'in2': 0.5,
              },
          ),
          stage=stage,
          si=si,
      ),
      (0.75, 0.75, 0.5),
      atol=1e-5,
  )
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_add_vector3FA',
              {
                  'in1': (0.1, 0.2, 0.3),
                  'in2': ('ND_geompropvalue_float', {'geomprop': 'Pworld'}),
              },
          ),
          stage=stage,
          si=si,
      ),
      (0.3, 0.4, 0.5),
      atol=1e-5,
  )


def _define_mtlx_disp(
    stage: Usd.Stage, path: str, disp_id: str, **inputs: object
) -> UsdShade.Material:
  mat = UsdShade.Material.Define(stage, path)
  disp = mtlx.define_shader(stage, f'{path}/Disp', disp_id, **inputs)
  mat.CreateDisplacementOutput('mtlx').ConnectToSource(
      disp.CreateOutput('out', Sdf.ValueTypeNames.Token)
  )
  return mat


def _convert_mtlx_disp(
    disp_id: str, **inputs: object
) -> material_lib.MeshMaterial:
  stage = Usd.Stage.CreateInMemory()
  mat = _define_mtlx_disp(stage, '/Mat', disp_id, **inputs)
  return material_lib.convert_mesh_material(mat.GetPrim())


@pytest.mark.parametrize(
    ('disp_id', 'inputs', 'requires_uv', 'is_vector', 'expected'),
    [
        pytest.param(
            'ND_displacement_float',
            {'displacement': -0.3, 'scale': 2.0},
            False,
            False,
            -0.6,
            id='scalar',
        ),
        pytest.param(
            'ND_displacement_float',
            {
                'displacement': ('ND_image_float', {'file': _UV_QUADRANTS}),
                'scale': 0.5,
            },
            True,
            False,
            0.5 * 60.0 / 255.0,
            id='image',
        ),
        pytest.param(
            'ND_displacement_vector3',
            {'displacement': Gf.Vec3f(0.2, -0.4, 0.6), 'scale': 0.5},
            False,
            True,
            (0.1, -0.2, 0.3),
            id='vector3',
        ),
        pytest.param(
            'ND_mix_displacementshader',
            {
                'fg': (
                    'ND_displacement_vector3',
                    {'displacement': Gf.Vec3f(0.4, -0.2, 0.8)},
                ),
                'bg': ('ND_displacement_float', {'displacement': 0.2}),
                'mix': 0.25,
            },
            False,
            True,
            (0.1, -0.05, 0.35),
            id='mix_vector_scalar',
        ),
        # Zero displacement with a UV-dependent scale does not leak `requires_uv`.
        pytest.param(
            'ND_mix_displacementshader',
            {
                'fg': (
                    'ND_displacement_float',
                    {
                        'displacement': 0.0,
                        'scale': ('ND_image_float', {'file': _UV_QUADRANTS}),
                    },
                ),
                'bg': ('ND_displacement_float', {'displacement': 0.4}),
                'mix': 0.5,
            },
            False,
            False,
            0.2,
            id='zero_disp_uv_scale',
        ),
        # Dead UV-reading temporaries from folded branches do not leak `requires_uv`.
        pytest.param(
            'ND_mix_displacementshader',
            {
                'fg': (
                    'ND_mix_displacementshader',
                    {
                        'fg': ('ND_displacement_float', {'displacement': 0.0}),
                        'bg': ('ND_displacement_float', {'displacement': 0.0}),
                        'mix': (
                            'ND_ramplr_float',
                            {'valuel': 0.0, 'valuer': 1.0},
                        ),
                    },
                ),
                'bg': ('ND_displacement_float', {'displacement': 0.4}),
                'mix': 0.5,
            },
            False,
            False,
            0.2,
            id='dead_uv_temp',
        ),
        pytest.param(
            'ND_displacement_float',
            {
                'displacement': ('ND_image_float', {'file': _UV_QUADRANTS}),
                'scale': 0.0,
            },
            False,
            False,
            None,
            id='zero_scale',
        ),
    ],
)
def test_materialx_displacement_eval(
    disp_id: str,
    inputs: dict[str, object],
    requires_uv: bool,
    is_vector: bool,
    expected: float | tuple[float, float, float] | None,
) -> None:
  """Tests MaterialX scalar, image, vector, mixed, and zero-scale displacement.

  Args:
    disp_id: MaterialX displacement shader node identifier.
    inputs: Shader inputs passed to `define_shader`.
    requires_uv: Expected `MeshMaterial.requires_uv` flag.
    is_vector: Expected `MeshMaterial.is_vector` flag.
    expected: Expected scalar/vector displacement value, or None if inactive.
  """
  mat = _convert_mtlx_disp(disp_id, **inputs)
  if expected is None:
    assert mat.displacement is None
    return
  assert mat.displacement is not None
  assert (mat.requires_uv, mat.is_vector) == (requires_uv, is_vector)
  si = mtlx.make_test_si()
  si.uv = mi.Point2f(0.25, 0.75)
  if isinstance(expected, tuple):
    np.testing.assert_allclose(
        mat.displacement.eval_3(si), mi.Color3f(*expected), atol=1e-4
    )
  else:
    np.testing.assert_allclose(mat.displacement.eval_1(si), expected, atol=1e-4)


@pytest.mark.parametrize(
    'mirror_uv', [False, True], ids=['rh_uv', 'mirrored_uv']
)
@pytest.mark.parametrize('scale_x', [1.0, -1.0], ids=['pos_x', 'neg_x'])
def test_materialx_vector_displacement_bitangent(
    mirror_uv: bool, scale_x: float
) -> None:
  """Tests tangent-frame bitangent handedness under mirrored UVs and transforms.

  Args:
    mirror_uv: Whether to mirror the quad's U coordinates.
    scale_x: X scale factor applied by the object-to-world transform.
  """
  stage = Usd.Stage.CreateInMemory()
  quad = mtlx.define_quad(stage, '/Quad')
  if mirror_uv:
    quad.GetPrim().GetAttribute('primvars:st').Set(
        [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    )
  mesh_data = geom_lib.extract_mesh_data(
      stage, quad.GetPath(), Usd.TimeCode.Default()
  )
  base_pts = np.array(quad.GetPointsAttr().Get())
  for disp in (
      Gf.Vec3f(0.0, 1.0, 0.0),
      ('ND_bitangent_vector3', {'space': 'world'}),
  ):
    mat = _convert_mtlx_disp(
        'ND_displacement_vector3', displacement=disp, scale=0.25
    )
    (sub,) = geom_lib.process_mesh(
        mesh_data,
        0,
        Gf.Matrix4d().SetScale(Gf.Vec3d(scale_x, 1.0, 1.0)),
        [(
            mat.displacement,
            mat.requires_uv,
            mat.is_vector,
            mat.displacement_attributes,
        )],
        set(),
    )
    np.testing.assert_allclose(
        sub.primvars['points'],
        base_pts * (scale_x, 1.0, 1.0) + (0.0, 0.25, 0.0),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ('surf_ctx', 'disp_ctx', 'disp_shader', 'scale', 'expected'),
    [
        pytest.param(
            'mtlx',
            '',
            ('mitsuba_uniform', {'value': 0.15}),
            1.0,
            0.4,
            id='mtlx_over_univ',
        ),
        pytest.param(
            'mtlx',
            '',
            ('mitsuba_uniform', {'value': 0.15}),
            0.0,
            None,
            id='zero_scale_no_fallback',
        ),
        pytest.param(
            'mtlx',
            '',
            ('ND_displacement_float', {'displacement': 0.1}),
            1.0,
            0.4,
            id='mtlx_over_univ_mtlx',
        ),
        pytest.param(
            '',
            'glslfx',
            ('mitsuba_uniform', {'value': 0.15}),
            1.0,
            0.4,
            id='univ_over_glslfx',
        ),
        pytest.param(
            'mtlx',
            'mitsuba',
            ('mitsuba_uniform', {'value': 0.15}),
            1.0,
            0.15,
            id='mitsuba_over_mtlx',
        ),
    ],
)
def test_materialx_surfacematerial_displacement_precedence(
    surf_ctx: str,
    disp_ctx: str,
    disp_shader: tuple[str, dict[str, object]],
    scale: float,
    expected: float | None,
) -> None:
  """Tests `surfacematerial.displacementshader` render-context precedence.

  Args:
    surf_ctx: Render context for the `ND_surfacematerial` surface output.
    disp_ctx: Render context for the competing displacement output.
    disp_shader: `(shader_id, inputs)` for the competing displacement shader.
    scale: Scale input on the `surfacematerial`'s `ND_displacement_float`.
    expected: Expected scalar displacement value, or None if inactive.
  """
  stage = Usd.Stage.CreateInMemory()
  mat = UsdShade.Material.Define(stage, '/Mat')
  sm_node = mtlx.define_shader(
      stage,
      '/Mat/SurfMat',
      'ND_surfacematerial',
      surfaceshader=(
          'ND_surface',
          {'bsdf': ('ND_oren_nayar_diffuse_bsdf', {})},
      ),
      displacementshader=(
          'ND_displacement_float',
          {'displacement': 0.4, 'scale': scale},
      ),
  )
  mat.CreateSurfaceOutput(surf_ctx).ConnectToSource(
      sm_node.CreateOutput('out', Sdf.ValueTypeNames.Token)
  )
  competing = mtlx.define_shader(
      stage, '/Mat/OtherDisp', disp_shader[0], **disp_shader[1]
  )
  mat.CreateDisplacementOutput(disp_ctx).ConnectToSource(
      competing.CreateOutput('out', Sdf.ValueTypeNames.Token)
  )
  res = material_lib.convert_mesh_material(mat.GetPrim())
  assert res.bsdf is not None
  if expected is None:
    assert res.displacement is None
  else:
    assert res.displacement is not None
    np.testing.assert_allclose(
        res.displacement.eval_1(mtlx.make_test_si()), expected, atol=1e-5
    )


def test_materialx_displacement_terminals(
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests `mitsuba:displacement` precedence and unsupported non-fallback.

  Args:
    capfd: Pytest file-descriptor capture fixture for verifying warnings.
  """
  stage = Usd.Stage.CreateInMemory()
  mi_out = mtlx.define_shader(
      stage, '/MiD', 'mitsuba_uniform', value=0.15
  ).CreateOutput('out', Sdf.ValueTypeNames.Token)

  # `mitsuba:displacement` takes precedence over `mtlx:displacement`.
  mat = _define_mtlx_disp(
      stage, '/PrecMat', 'ND_displacement_float', displacement=0.9
  )
  mat.CreateDisplacementOutput('mitsuba').ConnectToSource(mi_out)
  assert (
      geom_lib.translate_materialx(mat.GetPrim(), Usd.TimeCode.Default())
      is None
  )
  res = material_lib.convert_mesh_material(mat.GetPrim())
  assert res.displacement is not None
  np.testing.assert_allclose(
      res.displacement.eval_1(mtlx.make_test_si()), 0.15, atol=1e-5
  )

  # Unsupported `mtlx:displacement` warns without falling back to universal.
  mat = _define_mtlx_disp(stage, '/BadDisp', 'ND_oren_nayar_diffuse_bsdf')
  mat.CreateDisplacementOutput().ConnectToSource(mi_out)
  capfd.readouterr()
  assert material_lib.convert_mesh_material(mat.GetPrim()).displacement is None
  assert 'Unsupported MaterialX displacement shader' in capfd.readouterr().err


@pytest.mark.parametrize(
    ('disp_expr', 'primvar', 'transform', 'expected_disp'),
    [
        # Object-space position ignores the translation and scale in `transform`.
        pytest.param(
            (
                'ND_extract_vector3',
                {
                    'in_': ('ND_position_vector3', {'space': 'object'}),
                    'index': 0,
                },
            ),
            None,
            Gf.Matrix4d().SetScale(Gf.Vec3d(2.0, 2.0, 1.0))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(5.0, -3.0, 2.0)),
            [-0.8, 0.8, 0.8, -0.8],
            id='position_object',
        ),
        # Object-space normal stays (0, 0, 1) even when rotated 90 deg around X.
        pytest.param(
            (
                'ND_extract_vector3',
                {'in_': ('ND_normal_vector3', {'space': 'object'}), 'index': 2},
            ),
            None,
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0)),
            [1.0, 1.0, 1.0, 1.0],
            id='normal_object',
        ),
        pytest.param(
            ('ND_geompropvalue_float', {'geomprop': 'mask'}),
            (
                'mask',
                Sdf.ValueTypeNames.FloatArray,
                'vertex',
                [0.1, 0.2, 0.3, 0.4],
            ),
            Gf.Matrix4d(1.0),
            [0.1, 0.2, 0.3, 0.4],
            id='primvar_vertex',
        ),
        pytest.param(
            ('ND_geompropvalue_float', {'geomprop': 'gain'}),
            ('gain', Sdf.ValueTypeNames.Float, 'constant', 0.25),
            Gf.Matrix4d(1.0),
            [0.25, 0.25, 0.25, 0.25],
            id='primvar_constant_scalar',
        ),
        pytest.param(
            ('ND_geompropvalue_float', {'geomprop': 'face_h'}),
            ('face_h', Sdf.ValueTypeNames.FloatArray, 'uniform', [0.2, 0.5]),
            Gf.Matrix4d(1.0),
            [0.2, 0.2, 0.2, 0.5],
            id='primvar_uniform',
        ),
        pytest.param(
            ('ND_geompropvalue_float', {'geomprop': 'fv_h'}),
            (
                'fv_h',
                Sdf.ValueTypeNames.FloatArray,
                'faceVarying',
                [0.15, 0.25, 0.35, 0.99, 0.88, 0.45],
            ),
            Gf.Matrix4d(1.0),
            [0.15, 0.25, 0.35, 0.45],
            id='primvar_facevarying',
        ),
        pytest.param(
            (
                'ND_extract_color3',
                {'in_': ('ND_geomcolor_color3', {}), 'index': 0},
            ),
            (
                'displayColor',
                Sdf.ValueTypeNames.Color3fArray,
                'vertex',
                [
                    (0.2, 0.0, 0.0),
                    (0.4, 0.0, 0.0),
                    (0.6, 0.0, 0.0),
                    (0.8, 0.0, 0.0),
                ],
            ),
            Gf.Matrix4d(1.0),
            [0.2, 0.4, 0.6, 0.8],
            id='geomcolor',
        ),
        pytest.param(
            ('ND_geompropvalue_float', {'geomprop': 'missing', 'default': 0.3}),
            None,
            Gf.Matrix4d(1.0),
            [0.3, 0.3, 0.3, 0.3],
            id='primvar_missing_default',
        ),
    ],
)
def test_materialx_displacement_object_space_and_primvars(
    disp_expr: tuple[str, dict[str, object]],
    primvar: tuple[str, object, str, object] | None,
    transform: Gf.Matrix4d,
    expected_disp: list[float],
) -> None:
  """Tests object-space intrinsics and mesh primvars in MaterialX displacement.

  Args:
    disp_expr: MaterialX scalar expression driving `ND_displacement_float`.
    primvar: Optional `(name, sdf_type, interpolation, value)` to author.
    transform: Object-to-world transform passed to `process_mesh`.
    expected_disp: Expected scalar displacement at each of the quad's 4 corners.
  """
  stage = Usd.Stage.CreateInMemory()
  quad = mtlx.define_quad(stage, '/Quad')
  # Drop UVs to verify object-space intrinsics and primvars do not require `st`.
  quad.GetPrim().RemoveProperty('primvars:st')
  if primvar is not None:
    name, sdf_type, interp, value = primvar
    if interp in ('uniform', 'faceVarying'):
      quad.CreateFaceVertexCountsAttr([3, 3])
      quad.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    UsdGeom.PrimvarsAPI(quad).CreatePrimvar(name, sdf_type, interp).Set(value)
  mat = _convert_mtlx_disp('ND_displacement_float', displacement=disp_expr)
  assert not mat.requires_uv
  mesh_data = geom_lib.extract_mesh_data(
      stage, quad.GetPath(), Usd.TimeCode.Default()
  )
  (sub,) = geom_lib.process_mesh(
      mesh_data,
      0,
      transform,
      [(
          mat.displacement,
          mat.requires_uv,
          mat.is_vector,
          mat.displacement_attributes,
      )],
      set(),
  )
  expected_world = np.array([
      transform.TransformAffine(p + Gf.Vec3f(0.0, 0.0, d))
      for p, d in zip(quad.GetPointsAttr().Get(), expected_disp)
  ])
  actual = np.unique(np.round(sub.primvars['points'], 6), axis=0)
  expected = np.unique(np.round(expected_world, 6), axis=0)
  np.testing.assert_allclose(actual, expected, atol=1e-5)


def _h2n_expected(gx: float, gy: float) -> np.ndarray:
  n = np.array([-gx, -gy, 1.0])
  return (n / np.linalg.norm(n)) * 0.5 + 0.5


def _uv_linear_expr(a: float, b: float) -> tuple[str, dict[str, object]]:
  """Builds MaterialX expression `a * UV0.x + b * UV0.y`."""
  return (
      'ND_dotproduct_vector2',
      {'in1': ('ND_texcoord_vector2', {}), 'in2': (a, b)},
  )


@pytest.mark.parametrize(
    ('node_id', 'inputs', 'expected'),
    [
        pytest.param(
            'ND_heighttonormal_vector3',
            {'in_': _uv_linear_expr(16.0, 32.0)},
            _h2n_expected(1.0, 2.0),
            id='diagonal_default_scale',
        ),
        pytest.param(
            'ND_heighttonormal_vector3',
            {'in_': _uv_linear_expr(8.0, 16.0), 'scale': -2.0},
            _h2n_expected(-1.0, -2.0),
            id='diagonal_negative_scale',
        ),
        pytest.param(
            'ND_heighttonormal_vector3',
            {'in_': _uv_linear_expr(16.0, 0.0), 'scale': 0.0},
            (0.5, 0.5, 1.0),
            id='zero_scale',
        ),
        pytest.param(
            'ND_heighttonormal_vector3',
            {'in_': 0.75, 'scale': 5.0},
            (0.5, 0.5, 1.0),
            id='constant_height',
        ),
        # Scaling both `in` and `texcoord` by 4 cancels out in `dH/d(texcoord)`.
        pytest.param(
            'ND_heighttonormal_vector3',
            {
                'in_': _uv_linear_expr(64.0, 0.0),
                'texcoord': (
                    'ND_multiply_vector2FA',
                    {'in1': ('ND_texcoord_vector2', {}), 'in2': 4.0},
                ),
            },
            _h2n_expected(1.0, 0.0),
            id='scaled_texcoord',
        ),
        # `rotate2d(UV0, 90) = (v, -u)`, so `16 * u = -16 * tc.y`.
        pytest.param(
            'ND_heighttonormal_vector3',
            {
                'in_': _uv_linear_expr(16.0, 0.0),
                'texcoord': (
                    'ND_rotate2d_vector2',
                    {'in_': ('ND_texcoord_vector2', {}), 'amount': 90.0},
                ),
            },
            _h2n_expected(0.0, -1.0),
            id='rotated_texcoord',
        ),
        # Mirrored texcoord `(-u, v)` flips `dH/d(tc.x)` while keeping `n.z > 0`.
        pytest.param(
            'ND_heighttonormal_vector3',
            {
                'in_': _uv_linear_expr(16.0, 0.0),
                'texcoord': (
                    'ND_multiply_vector2',
                    {'in1': ('ND_texcoord_vector2', {}), 'in2': (-1.0, 1.0)},
                ),
            },
            _h2n_expected(-1.0, 0.0),
            id='mirrored_texcoord',
        ),
        pytest.param(
            'ND_heighttonormal_vector3',
            {'in_': _uv_linear_expr(16.0, 0.0), 'texcoord': (0.5, 0.5)},
            (0.5, 0.5, 1.0),
            id='degenerate_texcoord',
        ),
        # `ND_bump_vector3` flattens to `heighttonormal` + `normalmap`.
        pytest.param(
            'ND_bump_vector3',
            {'height': _uv_linear_expr(16.0, 32.0), 'scale': 0.5},
            np.array([-0.5, -1.0, 1.0]) / np.linalg.norm([-0.5, -1.0, 1.0]),
            id='bump_world_normal',
        ),
    ],
)
def test_heighttonormal_and_bump(
    node_id: str,
    inputs: dict[str, object],
    expected: tuple[float, float, float] | np.ndarray,
) -> None:
  """Tests MaterialX 1.39 `heighttonormal` and `bump` normal perturbation.

  Args:
    node_id: MaterialX NodeDef identifier (`ND_heighttonormal_vector3` or
      `ND_bump_vector3`).
    inputs: Node inputs passed to `define_shader`.
    expected: Expected 3-channel tangent-space or world-space normal.
  """
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si.uv = mi.Point2f(0.4, 0.6)
  # Match a standard USD XY quad where MaterialX +V points along +Y (`-dp_dv`).
  si.dp_dv = mi.Vector3f(0.0, -1.0, 0.0)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr((node_id, inputs), si=si),
      expected,
      atol=1e-3,
  )


@pytest.mark.parametrize(
    ('node_id', 'inputs'),
    [
        pytest.param(
            'ND_bump_vector3',
            {
                'height': ('ND_image_float', {'file': _UV_QUADRANTS}),
                'scale': 0.0,
            },
            id='zero_scale_bump',
        ),
        pytest.param(
            'ND_heighttonormal_vector3', {'in_': 0.75}, id='constant_height'
        ),
        pytest.param(
            'ND_heighttonormal_vector3',
            {
                'in_': (
                    'ND_image_float',
                    {
                        'file': Sdf.AssetPath('/nonexistent/height.exr'),
                        'default': 0.5,
                    },
                )
            },
            id='missing_image_height',
        ),
    ],
)
def test_heighttonormal_unused_height_inputs(
    node_id: str, inputs: dict[str, object]
) -> None:
  """Tests that heights which do not affect the normal add no parameters.

  Args:
    node_id: MaterialX NodeDef identifier.
    inputs: Node inputs passed to `define_shader`.
  """
  _, emitter = mtlx.compile_mtlx_material(
      'ND_surface_unlit', emission_color=(node_id, inputs)
  )
  assert not [
      k
      for k in mi.traverse(emitter).keys()
      if k.startswith(('radiance.in', 'radiance.param'))
  ]


def test_heighttonormal_image_and_object_space(
    tmp_path: pathlib.Path,
) -> None:
  """Tests `heighttonormal` on a 2D image and a transformed mesh's object position.

  Args:
    tmp_path: Temporary directory fixture for writing a linear height ramp EXR.
  """
  # Linear horizontal ramp image: texel centers span `[0, 16]` across U.
  ramp = ((np.arange(64) + 0.5) * (16.0 / 64))[None, :]
  img_path = _write_bitmap(tmp_path / 'ramp.exr', ramp)
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si.uv = mi.Point2f(0.5, 0.5)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          (
              'ND_heighttonormal_vector3',
              {'in_': ('ND_image_float', {'file': img_path})},
          ),
          si=si,
      ),
      _h2n_expected(1.0, 0.0),
      atol=1e-3,
  )

  # 3D object-space position height on a rotated mesh: object X maps to world Y,
  # while quad U spans `[-0.8, 0.8]` (`dx_obj/du = 1.6`). With `h = 10 * P_obj.x`,
  # `dh/du = 16`, so `bump(scale=1)` tilts the world normal along `-T = (0, -1, 0)`.
  stage = Usd.Stage.CreateInMemory()
  quad = mtlx.define_quad(stage, '/Quad')
  quad.AddRotateZOp().Set(90.0)
  mtlx.bind_mtlx_shader(
      stage,
      quad,
      'ND_surface_unlit',
      emission_color=(
          'ND_bump_vector3',
          {
              'height': (
                  'ND_multiply_float',
                  {
                      'in1': (
                          'ND_extract_vector3',
                          {
                              'in_': (
                                  'ND_position_vector3',
                                  {'space': 'object'},
                              ),
                              'index': 0,
                          },
                      ),
                      'in2': 10.0,
                  },
              ),
          },
      ),
  )
  mi_scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  hit = mi_scene.ray_intersect(
      mi.Ray3f(mi.Point3f(0.0, 0.0, 1.0), mi.Vector3f(0.0, 0.0, -1.0))
  )
  np.testing.assert_allclose(
      hit.emitter(mi_scene).eval(hit),
      dr.normalize(mi.Color3f(0.0, -1.0, 1.0)),
      atol=1e-3,
  )


def test_heighttonormal_shared_authored_inputs() -> None:
  """Tests authored inputs shared between outer scope and `heighttonormal`."""
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si.uv = mi.Point2f(0.5, 0.5)
  stage = Usd.Stage.CreateInMemory()

  def check(
      expr: tuple[str, dict[str, object]],
      n_params: int,
      expected: tuple[float, float, float] | np.ndarray,
      atol: float = 1e-3,
  ) -> None:
    _, em = mtlx.compile_mtlx_material(
        'ND_surface_unlit',
        emission_color=('ND_convert_vector3_color3', {'in_': expr}),
        stage=stage,
    )
    assert (
        em is not None
        and sum(k.startswith('radiance.param') for k in mi.traverse(em).keys())
        == n_params
    )
    np.testing.assert_allclose(
        em.eval(si), mi.Color3f(*map(float, expected)), atol=atol
    )

  slope = mtlx.define_shader(
      stage, 'Slope', 'ND_constant_vector2', value=(16.0, 32.0)
  )
  check(
      (
          'ND_add_vector3FA',
          {
              'in1': (
                  'ND_heighttonormal_vector3',
                  {
                      'in_': (
                          'ND_dotproduct_vector2',
                          {
                              'in1': ('ND_texcoord_vector2', {}),
                              'in2': slope,
                          },
                      )
                  },
              ),
              'in2': ('ND_extract_vector2', {'in_': slope, 'index': 0}),
          },
      ),
      2,
      _h2n_expected(1.0, 2.0) + 16.0,
  )

  c = mtlx.define_shader(stage, 'C', 'ND_constant_float', value=0.25)
  check(
      (
          'ND_add_vector3FA',
          {'in1': ('ND_heighttonormal_vector3', {'in_': c}), 'in2': c},
      ),
      1,
      (0.75, 0.75, 1.25),
      atol=1e-4,
  )

  m3 = mtlx.define_shader(
      stage,
      'M3',
      'ND_constant_matrix33',
      value=Gf.Matrix3d(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
  )
  pos_x = (
      'ND_extract_vector3',
      {
          'in_': (
              'ND_transformmatrix_vector3',
              {
                  'in_': ('ND_position_vector3', {'space': 'world'}),
                  'mat': m3,
              },
          ),
          'index': 0,
      },
  )
  check(
      (
          'ND_add_vector3',
          {
              'in1': ('ND_heighttonormal_vector3', {'in_': pos_x}),
              'in2': (
                  'ND_transformmatrix_vector3',
                  {'in_': Gf.Vec3f(1.0, 1.0, 1.0), 'mat': m3},
              ),
          },
      ),
      9,
      _h2n_expected(0.125, 0.0) + (2.0, 3.0, 4.0),
  )


def test_default_frame_normalmap() -> None:
  """Verifies default-frame `normalmap` matches explicit world-frame inputs."""
  si = dr.zeros(mi.SurfaceInteraction3f, 3)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  n0 = dr.normalize(mi.Vector3f(0.2, -0.3, 0.9))
  si.sh_frame = mi.Frame3f(n0)
  # Lane 0: right-handed UV, lane 1: mirrored V (`dot(n x t, dp_dv) > 0`),
  # lane 2: degenerate `dp_du = 0` falling back to `si.sh_frame.s`.
  si.dp_du = mi.Vector3f([1.0, -1.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0])
  si.dp_dv = mi.Vector3f([0.0, 0.1, 0.0], [-1.0, -1.0, 0.0], [0.0, 0.0, 0.0])

  base = {'in_': Gf.Vec3f(0.65, 0.3, 0.85), 'scale': 0.7}
  nm_def = mtlx.eval_emission_expr(('ND_normalmap_float', base), si=si)
  # Wrap `normal` in `ND_normalize_vector3` so the reference takes the 5-arg path.
  nm_exp = mtlx.eval_emission_expr(
      (
          'ND_normalmap_float',
          {
              **base,
              'normal': (
                  'ND_normalize_vector3',
                  {'in_': ('ND_normal_vector3', {'space': 'world'})},
              ),
              'tangent': ('ND_tangent_vector3', {'space': 'world'}),
              'bitangent': ('ND_bitangent_vector3', {'space': 'world'}),
          },
      ),
      si=si,
  )
  np.testing.assert_allclose(nm_def, nm_exp, atol=1e-5)
  np.testing.assert_allclose(
      mtlx.eval_emission_expr(
          ('ND_normalmap_float', {'in_': Gf.Vec3f(0.0, 0.0, 0.0)}), si=si
      ),
      np.broadcast_to(np.asarray(n0).T, (3, 3)),
      atol=1e-5,
  )


def test_heighttonormal_traverse_freeze_and_ad() -> None:
  """Verifies `mi.traverse()`, `@dr.freeze`, and AD through `heighttonormal`."""
  si = mtlx.make_test_si(wi=(0.0, 0.0, 1.0))
  si.uv = mi.Point2f(0.5, 0.5)
  # `h = u * (k * 16 + 8)` has `dh_du = k + 0.5`, `dh_dv = 0`.
  expr = (
      'ND_heighttonormal_vector3',
      {
          'in_': (
              'ND_multiply_float',
              {
                  'in1': (
                      'ND_extract_vector2',
                      {'in_': ('ND_texcoord_vector2', {}), 'index': 0},
                  ),
                  'in2': (
                      'ND_add_float',
                      {
                          'in1': (
                              'ND_multiply_float',
                              {
                                  'in1': ('ND_constant_float', {'value': 0.5}),
                                  'in2': 16.0,
                              },
                          ),
                          'in2': 8.0,
                      },
                  ),
              },
          ),
      },
  )
  _, em = mtlx.compile_mtlx_material('ND_surface_unlit', emission_color=expr)
  assert em is not None
  expected0 = mi.Color3f(*map(float, _h2n_expected(1.0, 0.0)))
  expected1 = mi.Color3f(*map(float, _h2n_expected(2.0, 0.0)))
  np.testing.assert_allclose(em.eval(si), expected0, atol=1e-4)

  @dr.freeze
  def eval_frozen(s: mi.SurfaceInteraction3f) -> mi.Color3f:
    return em.eval(s)

  for _ in range(2):
    np.testing.assert_allclose(eval_frozen(si), expected0, atol=1e-4)

  params = mi.traverse(em)
  key = next(
      k
      for k in params.keys()
      if k.startswith('radiance.param') and np.allclose(params[k], 0.5)
  )
  params[key] = mi.Float(1.5)
  params.update()
  np.testing.assert_allclose(eval_frozen(si), expected1, atol=1e-4)

  # `out.x = 0.5 - 0.5 * s / sqrt(1 + s^2)` with `s = k + 0.5 = 2` -> `-0.5 / 5^1.5`.
  dr.enable_grad(params[key])
  params.update()
  dr.backward(em.eval(si).x)
  np.testing.assert_allclose(dr.grad(params[key]), -0.5 / 5.0**1.5, rtol=1e-3)
