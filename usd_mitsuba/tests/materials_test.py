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

import os
import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from pxr import Sdf
from pxr import Usd
from pxr import UsdShade
from usd_mitsuba import material
from usd_mitsuba import material_authoring

_ALBEDO_PNG = os.path.abspath(
    'hdmitsuba/test_assets/materials/textures/albedo.png')
_NORMALMAP_PNG = os.path.abspath(
    'hdmitsuba/test_assets/materials/textures/normalmap.png')

MITSUBA_EXAMPLE_BSDFS = [
    {'type': 'diffuse'},
    {
        'type': 'diffuse',
        'reflectance': {'type': 'srgb', 'color': (0.2, 0.5, 0.2)},
    },
    {
        'type': 'roughplastic',
        'diffuse_reflectance': (0.2, 0.5, 0.2),
        'specular_reflectance': 0.2,
    },
    {'type': 'roughconductor', 'material': 'Au', 'alpha': 0.2},
    {
        'type': 'principled',
        'base_color': {
            'type': 'checkerboard',
            'to_uv': mi.scalar_rgb.ScalarAffineTransform4f.scale(10.0),
        },
    },
    {
        'type': 'principled',
        'base_color': {'type': 'bitmap', 'filename': _ALBEDO_PNG},
    },
    {
        'type': 'normalmap',
        'normalmap': {'type': 'bitmap', 'filename': _NORMALMAP_PNG, 'raw': True},
        'nested_bsdf': {
            'type': 'roughplastic',
            'diffuse_reflectance': {
                'type': 'srgb',
                'color': (0.2, 0.5, 0.2),
            },
            'specular_reflectance': {
                'type': 'bitmap',
                'filename': _ALBEDO_PNG,
            },
            'alpha': 0.2,
            'nonlinear': True,
        },
    },
]


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _assert_pytree_equals(a, b, atol: float = 1e-6, rtol: float = 1e-6):
  """Asserts that two pytrees (e.g. trees of NumPy arrays) are equal."""
  if not isinstance(a, type(b)):
    raise ValueError(
        f'Types of a and b do not match: {type(a)} and {type(b)}'
    )

  if isinstance(a, dict):
    assert set(a.keys()) == set(b.keys())
    for k in a:
      if k not in b:
        raise AssertionError(f'Key {k} not found in b: {b.keys()}')
      _assert_pytree_equals(a[k], b[k], atol, rtol)
    return
  elif isinstance(a, (list, tuple)):
    if len(a) != len(b):
      raise AssertionError(
          f'Lengths of a and b do not match: {len(a)} and {len(b)}'
      )
    for i in range(len(a)):
      _assert_pytree_equals(a[i], b[i], atol, rtol)
      return
  elif isinstance(a, (float, np.ndarray, dr.ArrayBase)):
    np.testing.assert_allclose(
        np.array(a), np.array(b), atol=atol, rtol=rtol
    )
    return
  elif isinstance(a, (int, bool, str)):
    assert a == b


@pytest.mark.parametrize("mitsuba_bsdf", MITSUBA_EXAMPLE_BSDFS)
def test_mitsuba_to_usd_conversion(mitsuba_bsdf):
  stage = Usd.Stage.CreateInMemory()
  material_authoring.add_mitsuba_shader(stage, Sdf.Path('/test'), mitsuba_bsdf)

  prim = stage.GetPrimAtPath(Sdf.Path('/test'))
  assert prim is not None
  shader = UsdShade.Shader(prim)
  assert shader.GetShaderId() == 'mitsuba_' + mitsuba_bsdf['type']

  result_dict = material.usd_mitsuba_material_to_dict(shader)
  _assert_pytree_equals(result_dict, mitsuba_bsdf)


@pytest.mark.parametrize("bsdf_dict", [
    {
        'type': 'bumpmap',
        'bumpmap': {'type': 'bitmap', 'filename': _NORMALMAP_PNG},
        'nested_bsdf': {
            'type': 'principled',
            'base_color': {'type': 'bitmap', 'filename': _ALBEDO_PNG},
            'roughness': 0.2,
        },
    },
    {
        'type': 'normalmap',
        'normalmap': {'type': 'bitmap', 'filename': _NORMALMAP_PNG, 'raw': True},
        'nested_bsdf': {
            'type': 'principled',
            'base_color': {'type': 'bitmap', 'filename': _ALBEDO_PNG},
            'roughness': 0.2,
        },
    },
    {
        'type': 'principled',
        'base_color': {'type': 'bitmap', 'filename': _ALBEDO_PNG},
        'roughness': 0.2,
        'spec_trans': 0.3,
    },
    {
        'type': 'principled',
        'base_color': {'type': 'bitmap', 'filename': _ALBEDO_PNG},
        'roughness': 0.2,
        'clearcoat': 0.5,
        'clearcoat_gloss': 0.8,
    },
])
def test_add_mitsuba_shader_as_usd_preview_surface(bsdf_dict):
  stage = Usd.Stage.CreateInMemory()
  prim_path = Sdf.Path('/test')
  shader = material_authoring.add_mitsuba_shader_as_usd_preview_surface(
      stage, bsdf_dict, prim_path
  )
  assert shader is not None

  assert shader.GetShaderId() == 'UsdPreviewSurface'
  assert shader.GetPrim().GetPath() == prim_path
  result_dict, _ = material.usd_preview_surface_to_mitsuba(shader)

  if 'bumpmap' in bsdf_dict:  # Bumpmap is not supported in preview surface.
    ref_dict = bsdf_dict['nested_bsdf']
  elif 'normalmap' in bsdf_dict:
    # The normal map round-trips through a `[-1, 1]` UsdUVTexture encoding.
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.uv = mi.Point2f(0.25, 0.75)
    np.testing.assert_allclose(
        mi.load_dict(result_dict.pop('normalmap')).eval_3(si),
        mi.load_dict(bsdf_dict['normalmap']).eval_3(si),
        atol=1e-5,
    )
    ref_dict = {'type': 'normalmap', 'nested_bsdf': bsdf_dict['nested_bsdf']}
  else:
    ref_dict = bsdf_dict

  _assert_pytree_equals(result_dict, ref_dict)


def test_usd_preview_surface_to_mitsuba_specular_eta_resolution():
  stage = Usd.Stage.CreateInMemory()
  shader = UsdShade.Shader.Define(stage, Sdf.Path('/test_shader'))
  shader.CreateIdAttr('UsdPreviewSurface')

  # opacity maps to spec_trans by inverting (spec_trans = 1.0 - opacity)
  shader.CreateInput('specular', Sdf.ValueTypeNames.Float).Set(0.5)
  shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.5)

  # Test 1: spec_trans = 0.5 (opacity = 0.5) > 0.0, keep eta
  shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(0.5)
  bsdf_dict, _ = material.usd_preview_surface_to_mitsuba(shader)
  assert 'eta' in bsdf_dict
  assert 'specular' not in bsdf_dict
  assert bsdf_dict['eta'] == 1.5
  assert bsdf_dict['spec_trans'] == 0.5

  # Test 2: spec_trans = 0.0 (opacity = 1.0), keep specular
  shader.GetInput('opacity').Set(1.0)
  bsdf_dict, _ = material.usd_preview_surface_to_mitsuba(shader)
  assert 'eta' not in bsdf_dict
  assert 'specular' in bsdf_dict
  assert bsdf_dict['specular'] == 0.5
  assert bsdf_dict['spec_trans'] == 0.0


def test_displacement_texture_translation():
  stage = Usd.Stage.CreateInMemory()
  shader = UsdShade.Shader.Define(
      stage, Sdf.Path('/Material/displacement_texture')
  )
  shader.CreateIdAttr('UsdUVTexture')
  shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(_ALBEDO_PNG)

  result_dict = material.usd_mitsuba_material_to_dict(shader, 'displacement')
  assert result_dict['raw']


@pytest.mark.parametrize(
    'medium_dict',
    [
        {
            'type': 'homogeneous',
            'sigma_t': 2.5,
            'scale': 1.5,
            'albedo': {'type': 'srgb', 'color': (0.8, 0.2, 0.2)},
            'phase': {'type': 'hg', 'g': 0.5},
        },
        {
            'type': 'heterogeneous',
            'sigma_t': {'type': 'constvolume', 'value': 2.0},
            'albedo': {
                'type': 'constvolume',
                'value': {'type': 'srgb', 'color': (0.4, 0.6, 0.8)},
            },
            'phase': {'type': 'isotropic'},
            'scale': 1.25,
        },
    ],
)
def test_mitsuba_medium_to_usd_conversion(medium_dict):
  stage = Usd.Stage.CreateInMemory()
  shader = material_authoring.add_mitsuba_shader(
      stage, Sdf.Path('/Material/VolumeShader'), medium_dict
  )
  assert shader.GetShaderId() == 'mitsuba_' + medium_dict['type']

  result_dict = material.usd_mitsuba_material_to_dict(shader, 'volume')
  _assert_pytree_equals(result_dict, medium_dict)
  assert isinstance(mi.load_dict(result_dict), mi.Medium)


def test_volume_material_terminal_and_gridvolume(tmp_path):
  vol_path = str(tmp_path / 'density.vol')
  mi.VolumeGrid(np.ones((2, 2, 2, 1), dtype=np.float32)).write(vol_path)

  stage = Usd.Stage.CreateInMemory()
  shader = material_authoring.add_mitsuba_shader(
      stage,
      Sdf.Path('/Material/VolumeShader'),
      {
          'type': 'heterogeneous',
          'sigma_t': {'type': 'gridvolume', 'filename': vol_path},
      },
  )
  usd_material = UsdShade.Material.Define(stage, Sdf.Path('/Material'))
  usd_material.CreateVolumeOutput().ConnectToSource(
      shader.ConnectableAPI(), 'volume'
  )
  cube = stage.DefinePrim('/Cube', 'Cube')
  UsdShade.MaterialBindingAPI.Apply(cube).Bind(usd_material)

  bsdf, emitter, displacement, medium = material.convert_material(cube)
  assert bsdf is not None and bsdf.class_name() == 'Null'
  assert emitter is None and displacement is None
  assert isinstance(medium, mi.Medium)

  surface_shader = material_authoring.add_mitsuba_shader(
      stage, Sdf.Path('/Material/SurfaceShader'), {'type': 'diffuse'}
  )
  usd_material.CreateSurfaceOutput('mitsuba').ConnectToSource(
      surface_shader.ConnectableAPI(), 'surface'
  )
  bsdf_with_surface, _, _, medium_with_surface = material.convert_material(cube)
  assert (
      bsdf_with_surface is not None
      and bsdf_with_surface.class_name() == 'SmoothDiffuse'
  )
  assert isinstance(medium_with_surface, mi.Medium)

  usd_material.GetVolumeOutput().ClearSources()
  _, _, _, cleared_medium = material.convert_material(cube)
  assert cleared_medium is None


def test_custom_python_plugins_to_usd_conversion():
  class _CustomBSDF(mi.BSDF):

    def __init__(self, props: mi.Properties):
      super().__init__(props)
      props.mark_queried('reflectance')
      props.mark_queried('nested')
      self.m_flags = int(mi.BSDFFlags.DiffuseReflection)
      self.m_components = [self.m_flags]

  class _CustomTexture(mi.Texture):

    def __init__(self, props: mi.Properties):
      super().__init__(props)
      props.mark_queried('value')

  mi.register_bsdf('custom_python_bsdf', _CustomBSDF)
  mi.register_texture('custom_python_tex', _CustomTexture)

  bsdf_dict = {
      'type': 'custom_python_bsdf',
      'reflectance': {'type': 'custom_python_tex', 'value': 0.5},
      'nested': {'type': 'custom_python_bsdf', 'reflectance': 0.1},
  }
  stage = Usd.Stage.CreateInMemory()
  shader = material_authoring.add_mitsuba_shader(
      stage, Sdf.Path('/test'), bsdf_dict
  )
  _assert_pytree_equals(
      material.usd_mitsuba_material_to_dict(shader), bsdf_dict
  )

  with pytest.raises(ValueError, match='ObjectType.Texture'):
    material_authoring.add_mitsuba_shader(
        Usd.Stage.CreateInMemory(),
        Sdf.Path('/invalid'),
        {'type': 'custom_python_tex', 'value': 0.5},
    )


def test_invalid_normal_and_bump_fallbacks():
  stage = Usd.Stage.Open('hdmitsuba/test_assets/materials/uv_texture.usda')
  stage.SetEditTarget(stage.GetSessionLayer())
  preview = UsdShade.Shader.Get(stage, '/root/_materials/preview_orm/Shader')
  missing = UsdShade.Shader.Get(
      stage, '/root/_materials/mitsuba_channels/MissingTex'
  )
  preview.CreateInput('bump', Sdf.ValueTypeNames.Float).ConnectToSource(
      missing.ConnectableAPI(), 'rgb'
  )
  preview_dict, _ = material.usd_preview_surface_to_mitsuba(preview)
  assert preview_dict['type'] == 'principled'

  mi_dict = material.usd_mitsuba_material_to_dict(
      UsdShade.Shader.Get(stage, '/root/_materials/mitsuba_channels/BumpMap')
  )
  assert mi_dict['bumpmap'] == {'type': 'uniform', 'value': 0.0}
  assert mi_dict['bsdf']['normalmap'] == {
      'type': 'rgb',
      'value': (0.5, 0.5, 1.0),
  }
  assert isinstance(mi.load_dict(mi_dict), mi.BSDF)


@pytest.mark.parametrize(
    'input_name, output_name, scale, bias',
    [
        ('displacement', 'r', (0.5, 1, 1, 1), (-0.25, 0, 0, 0)),
        ('diffuseColor', None, (1, 2, 3, 1), (0, 0, 0.5, 0)),
    ],
)
def test_uv_texture_scale_bias(input_name, output_name, scale, bias):
  stage = Usd.Stage.CreateInMemory()
  shader = UsdShade.Shader.Define(stage, Sdf.Path('/Material/texture'))
  shader.CreateIdAttr('UsdUVTexture')
  shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(_ALBEDO_PNG)
  shader.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(scale)
  shader.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(bias)

  tex = mi.load_dict(
      material.usd_mitsuba_material_to_dict(shader, input_name, output_name)
  )
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.uv = mi.Point2f(0.25, 0.75)
  raw_tex = mi.load_dict({
      'type': 'bitmap',
      'filename': _ALBEDO_PNG,
      'raw': input_name != 'diffuseColor',
  })
  expected = np.array(raw_tex.eval_3(si)).squeeze() * scale[:3] + bias[:3]
  if output_name == 'r':
    actual, expected = np.array(tex.eval_1(si)), expected[0]
  else:
    actual = np.array(tex.eval_3(si)).squeeze()
  np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_displacement_through_preview_surface():
  stage = Usd.Stage.CreateInMemory()
  usd_material = UsdShade.Material.Define(stage, Sdf.Path('/Material'))
  surface = UsdShade.Shader.Define(stage, Sdf.Path('/Material/surface'))
  surface.CreateIdAttr('UsdPreviewSurface')
  texture = UsdShade.Shader.Define(stage, Sdf.Path('/Material/texture'))
  texture.CreateIdAttr('UsdUVTexture')
  texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(_ALBEDO_PNG)
  surface.CreateInput('displacement', Sdf.ValueTypeNames.Float).ConnectToSource(
      texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
  )
  usd_material.CreateDisplacementOutput().ConnectToSource(
      surface.CreateOutput('displacement', Sdf.ValueTypeNames.Token)
  )
  mesh = stage.DefinePrim('/Mesh', 'Mesh')
  UsdShade.MaterialBindingAPI.Apply(mesh).Bind(usd_material)
  assert material.has_displacement(mesh)
  assert isinstance(material.convert_material(mesh)[2], mi.Texture)

  surface.GetInput('displacement').DisconnectSource()
  assert not material.has_displacement(mesh)
  assert material.convert_material(mesh)[2] is None
