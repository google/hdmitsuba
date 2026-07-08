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

