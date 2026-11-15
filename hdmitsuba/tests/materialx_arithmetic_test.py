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

"""Tests for MaterialX VM core, EDFs, cutout opacity, and kernel freezing."""

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
def _enable_materialx_and_variant(monkeypatch: pytest.MonkeyPatch) -> None:
  mtlx.enable_materialx_and_default_variant(monkeypatch)


def test_core_pattern_ops() -> None:
  """Tests core VM arithmetic, channel, conditional, switch, ramp, and luminance ops."""
  stage = Usd.Stage.CreateInMemory()

  # Constants, routing, conversions (including 1.39 `ND_convert_integer_boolean`
  # which flattens to `ifequal` + `not`), packing, swizzle, 10-input 1.39
  # `switch` (`which >= 5` when `in6..in10` authored), and `luminance` (`color4`
  # preserves alpha).
  lum = 0.2 * 0.2722287 + 0.5 * 0.6740818 + 0.8 * 0.0536895
  for node_id, expected, kw in [
      ('ND_constant_color3', (0.2, 0.5, 0.8), {'value': (0.2, 0.5, 0.8)}),
      ('ND_dot_color3', (0.3, 0.6, 0.9), {'in_': (0.3, 0.6, 0.9)}),
      (
          'ND_convert_boolean_float',
          (0.0,),
          {'in_': ('ND_convert_integer_boolean', {'in_': 0})},
      ),
      (
          'ND_convert_boolean_float',
          (1.0,),
          {'in_': ('ND_convert_integer_boolean', {'in_': 3})},
      ),
      (
          'ND_combine2_vector4VV',
          (0.1, 0.2, 0.3, 0.4),
          {'in1': (0.1, 0.2), 'in2': (0.3, 0.4)},
      ),
      (
          'ND_combine4_color4',
          (0.4, 0.3, 0.2, 0.7),
          {'in1': 0.4, 'in2': 0.3, 'in3': 0.2, 'in4': 0.7},
      ),
      (
          'ND_swizzle_color4_color3',
          (0.9, 0.8, 0.2),
          {'in_': (0.1, 0.2, 0.8, 0.9), 'channels': 'abg'},
      ),
      (
          'ND_switch_color3I',
          (0.0, 1.0, 0.0),
          {'in1': (1.0, 0.0, 0.0), 'in6': (0.0, 1.0, 0.0), 'which': 5},
      ),
      (
          'ND_luminance_color4',
          (lum, lum, lum, 0.65),
          {'in_': (0.2, 0.5, 0.8, 0.65)},
      ),
  ]:
    actual = mtlx.eval_emission_expr(
        (node_id, kw), stage=stage, channels=len(expected)
    )
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-4, err_msg=node_id
    )

  # Dynamic batched arithmetic, conditionals, and `ramp4` bilinear interpolation
  # driven by UV coordinates across 4 lanes (`u` in `[0, 0.25, 0.5, 1.0]`).
  u = mi.Float([0.0, 0.25, 0.5, 1.0])
  v = mi.Float([0.0, 0.5, 0.25, 1.0])
  si = dr.zeros(mi.SurfaceInteraction3f, 4)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)
  si.uv = mi.Point2f(
      u, 1.0 - v
  )  # `DefaultTexcoord` flips `1 - si.uv.y` -> `v`.

  tc = mtlx.define_shader(stage, 'TC', 'ND_texcoord_vector2')
  tx = mtlx.define_shader(stage, 'TX', 'ND_extract_vector2', in_=tc, index=0)
  ty = mtlx.define_shader(stage, 'TY', 'ND_extract_vector2', in_=tc, index=1)
  u_np, v_np = np.asarray(u), np.asarray(v)

  for node_id, expected, kw in [
      ('ND_add_float', u_np + v_np, {'in1': tx, 'in2': ty}),
      ('ND_subtract_float', u_np - v_np, {'in1': tx, 'in2': ty}),
      ('ND_multiply_float', u_np * v_np, {'in1': tx, 'in2': ty}),
      (
          'ND_divide_float',
          np.divide(u_np, v_np, out=np.zeros_like(u_np), where=v_np != 0),
          {'in1': tx, 'in2': ty},
      ),
      (
          'ND_clamp_float',
          np.clip(u_np, 0.2, 0.7),
          {'in_': tx, 'low': 0.2, 'high': 0.7},
      ),
      ('ND_invert_float', 1.0 - u_np, {'in_': tx, 'amount': 1.0}),
      (
          'ND_mix_float',
          (1.0 - u_np) * 0.2 + u_np * 0.8,
          {'fg': 0.8, 'bg': 0.2, 'mix': tx},
      ),
      (
          'ND_ifgreater_float',
          np.where(u_np > v_np, 2.0, 0.5),
          {'value1': tx, 'value2': ty, 'in1': 2.0, 'in2': 0.5},
      ),
      (
          'ND_ifgreatereq_float',
          np.where(u_np >= v_np, 2.0, 0.5),
          {'value1': tx, 'value2': ty, 'in1': 2.0, 'in2': 0.5},
      ),
      (
          'ND_ifequal_float',
          np.where(u_np == v_np, 2.0, 0.5),
          {'value1': tx, 'value2': ty, 'in1': 2.0, 'in2': 0.5},
      ),
      (
          'ND_ramp4_float',
          (1.0 - v_np) * (2.0 + u_np) + v_np * u_np,
          {
              'valuetl': 0.0,
              'valuetr': 1.0,
              'valuebl': 2.0,
              'valuebr': 3.0,
              'texcoord': tc,
          },
      ),
  ]:
    actual = mtlx.eval_emission_expr(
        (node_id, kw), stage=stage, si=si, channels=1
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, err_msg=node_id)


def test_colorspace_and_cycle_handling(
    capfd: pytest.CaptureFixture[str],
) -> None:
  """Tests UsdColorSpaceAPI folding, cycles, and warning dedup.

  Args:
    capfd: Pytest fd capture fixture.
  """
  # `UsdColorSpaceAPI` converts sRGB `(0.5, 0.5, 0.5)` to linear Rec.709 while
  # keeping opaque `opacity = (1, 1, 1)` exact so `MaskBSDF` is elided.
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
  assert bsdf is not None and bsdf.class_name() == 'SmoothDiffuse'
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


def test_feature_flag_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies MaterialX is opt-in via HDMITSUBA_ENABLE_MATERIALX and falls back when unset.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
  """
  stage, quad = mtlx.create_base_scene(
      with_lights=False, quad_half_size=0.8, resolution=(16, 16), spp=4
  )
  mtlx.bind_mtlx_shader(
      stage, quad, 'ND_surface_unlit', emission_color=(0.9, 0.1, 0.3)
  )
  preview = mtlx.define_shader(
      stage,
      'Preview',
      'UsdPreviewSurface',
      diffuseColor=(0.0, 0.0, 0.0),
      emissiveColor=(0.1, 0.8, 0.2),
  )
  UsdShade.Material.Get(
      stage, '/mesh/Material'
  ).CreateSurfaceOutput().ConnectToSource(
      preview.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )

  # TODO(vicini): Remove `HDMITSUBA_ENABLE_MATERIALX` (and this test) once
  # MaterialX support is complete.
  monkeypatch.delenv('HDMITSUBA_ENABLE_MATERIALX', raising=False)
  engine = mtlx.create_engine(stage)
  img_off, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_flag_off',
      engine=engine,
      spp=4,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_off), (0.1, 0.8, 0.2), atol=2e-2
  )
  # Dynamic `UsdPreviewSurface.emissiveColor` edit in Hydra must rebuild the
  # emitter.
  preview.GetInput('emissiveColor').Set((0.2, 0.3, 0.9))
  np.testing.assert_allclose(
      mtlx.render_center(engine), (0.2, 0.3, 0.9), atol=2e-2
  )

  monkeypatch.setenv('HDMITSUBA_ENABLE_MATERIALX', '1')
  img_on, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_mtlx_flag_on',
      spp=4,
      subdivision_level=0,
  )
  np.testing.assert_allclose(
      mtlx.render_center(img_on), (0.9, 0.1, 0.3), atol=2e-2
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
