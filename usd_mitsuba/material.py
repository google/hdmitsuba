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

"""Handles material conversions for Mitsuba rendering.

This module provides the routines to both generate and read USD definitions
of common Mitsuba BSDFs. This translation layer simplifies creation and use
of USD assets for Mitsuba rendering.
"""

from __future__ import annotations

from typing import Any
import os

import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdShade


USD_TO_MI_PRINCIPLED_RENAMING = {
    'diffuseColor': 'base_color',
    'roughness': 'roughness',
    'clearcoat': 'clearcoat',
    'clearcoatRoughness': 'clearcoat_gloss',
    'metallic': 'metallic',
    'specular': 'specular',
    'ior': 'eta',
    'opacity': 'spec_trans',
}

MI_TO_USD_PRINCIPLED_RENAMING = {
    v: k for k, v in USD_TO_MI_PRINCIPLED_RENAMING.items()
}

# Standard non-color shader inputs defined in the UsdPreviewSurface
# specification. Ref:
# third_party/usd/v23/pxr/usdImaging/plugin/usdShaders/shaders/shaderDefs.usda
_USD_NON_COLOR_INPUTS = frozenset({
    'normal',
    'bump',
    'roughness',
    'metallic',
    'displacement',
    'specular',
    'clearcoat',
    'clearcoatRoughness',
    'ior',
    'opacity',
})

_SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    '.exr',
    '.png',
    '.jpg',
    '.jpeg',
    '.hdr',
    '.rgbe',
    '.pfm',
    '.ppm',
    '.tga',
    '.bmp',
})

_MI_PREFIX = 'mitsuba_'


def _get_surface_output(material: UsdShade.Material) -> UsdShade.Output | None:
  """Gets the Mitsuba-specific surface output or falls back to default."""
  return material.GetSurfaceOutput('mitsuba') or material.GetSurfaceOutput()


def _get_displacement_output(
    material: UsdShade.Material,
) -> UsdShade.Output | None:
  """Gets the Mitsuba-specific displacement output or falls back to default."""
  return (
      material.GetDisplacementOutput('mitsuba')
      or material.GetDisplacementOutput()
  )


def find_shader(
    prop: UsdShade.Input | UsdShade.Output,
) -> UsdShade.Shader | None:
  """Returns the first upstream UsdShade.Shader.

  This function finds the first UsdShade.Shader connected to a given input or
  output.

  Args:
    prop: The UsdShade.Input or UsdShade.Output to find the shader for.
  """
  attributes = prop.GetValueProducingAttributes(shaderOutputsOnly=True)
  return UsdShade.Shader(attributes[0].GetPrim()) if attributes else None


def _convert_uv_texture(
    shader: UsdShade.Shader,
    input_name: str | None = None,
) -> dict[str, Any]:
  """Extracts texture information from a UsdUVTexture shader."""
  file_input = shader.GetInput('file')
  asset_path = file_input.Get() if file_input else None
  fallback_input = shader.GetInput('fallback')

  if asset_path and asset_path.resolvedPath:
    filename = asset_path.resolvedPath
    _, ext = os.path.splitext(filename)

    # This check is not identical to hdmitsuba's more strict check,
    # but should be sufficient for the Python conversion.
    if ext.lower() in _SUPPORTED_IMAGE_EXTENSIONS:
      texture = {'type': 'bitmap', 'filename': filename}

      # Handle color space if we have a bitmap.
      if source_color_space := shader.GetInput('sourceColorSpace').Get():
        is_non_color = input_name in _USD_NON_COLOR_INPUTS
        if source_color_space == 'auto':
          if is_non_color:
            texture['raw'] = True
        else:
          texture['raw'] = (source_color_space == 'raw') and is_non_color
      return texture
    else:
      Tf.Warn(
          f"Texture file '{filename}' has unsupported extension '{ext}'."
          " Using fallback."
      )

  # 2. If the file is missing/unresolved and we have a fallback color, use it.
  if fallback_input:
    val = fallback_input.Get()
    if val is not None:
      if isinstance(val, (Gf.Vec3f, Gf.Vec4f)):
        val = tuple(val)[:3]
      return {'type': 'rgb', 'value': val}

  # 3. Absolute fallback to magenta.
  return {'type': 'rgb', 'value': (1.0, 0.0, 1.0)}


def usd_mitsuba_material_to_dict(shader: UsdShade.Shader) -> dict[str, Any]:
  """Converts a USD shader to a Mitsuba BSDF dictionary."""

  def _traverse(
      shader: UsdShade.Shader, input_name: str | None = None
  ) -> dict[str, Any]:
    shader_id = shader.GetIdAttr().Get()
    if shader_id == 'UsdUVTexture':
      return _convert_uv_texture(shader, input_name)

    if not shader_id.startswith('mitsuba_'):
      raise ValueError(f'The shader {shader_id} is not a Mitsuba shader.')

    result = {'type': shader_id.replace('mitsuba_', '', 1)}
    for usd_input in shader.GetInputs():
      name = usd_input.GetBaseName()
      if connected_shader := find_shader(usd_input):
        result[name] = _traverse(connected_shader, name)
      else:
        attr = usd_input.GetAttr()
        value = attr.Get()
        if value is not None:
          type_name = attr.GetTypeName()
          if type_name == Sdf.ValueTypeNames.Color3f:
            result[name] = mi.scalar_rgb.ScalarColor3f(*value)
          elif type_name == Sdf.ValueTypeNames.Float3:
            result[name] = tuple(value)
          elif type_name == Sdf.ValueTypeNames.Matrix3d:
            result[name] = mi.scalar_rgb.ScalarAffineTransform3f(
                mi.scalar_rgb.ScalarMatrix3f(np.array(value).T)
            )
          elif type_name == Sdf.ValueTypeNames.Matrix4d:
            result[name] = mi.scalar_rgb.ScalarAffineTransform4f(
                mi.scalar_rgb.ScalarMatrix4f(np.array(value).T)
            )
          elif isinstance(value, Sdf.AssetPath):
            resolved = value.resolvedPath
            result[name] = resolved if resolved else value.path
          else:
            result[name] = value

    return result

  return _traverse(shader)


def usd_preview_surface_to_mitsuba(
    shader: UsdShade.Shader,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Converts a USD preview surface to a Mitsuba principled BSDF.

  Args:
    shader: The UsdPreviewSurface shader to convert.

  Returns:
    A tuple containing:
      - The Mitsuba BSDF dictionary.
      - The Mitsuba Emitter dictionary (if the shader has emission), or None.
  """

  def _get_value(value: Any, invert: bool = False) -> Any:
    if isinstance(value, Sdf.AssetPath):
      return value.path
    if invert:
      value = 1 - value
    if isinstance(value, Gf.Vec3f):
      return {'type': 'rgb', 'value': tuple(value)}
    return value

  bsdf_dict: dict[str, Any] = {'type': 'principled'}
  parent_bsdf_dict = {}
  renaming = USD_TO_MI_PRINCIPLED_RENAMING
  emitter_dict: dict[str, Any] = {'type': 'area'}
  for usd_input in shader.GetInputs():
    name = usd_input.GetBaseName()
    val = None
    if connected_shader := find_shader(usd_input):
      if connected_shader.GetIdAttr().Get() == 'UsdUVTexture':
        val = _convert_uv_texture(connected_shader, name)
        if name in ('opacity', 'clearcoatRoughness'):
          Tf.Warn(
              f"Texture inversion for {name} is not supported. The resulting"
              " Mitsuba material may look incorrect."
          )
    else:
      raw_val = usd_input.GetAttr().Get()
      if raw_val is not None:
        if name == 'emissiveColor' and (raw_val == Gf.Vec3f(0) or raw_val == 0):
          continue
        val = _get_value(raw_val, name in ('opacity', 'clearcoatRoughness'))

    if val is None:
      continue

    if name == 'emissiveColor':
      emitter_dict['radiance'] = val
    elif name in renaming:
      bsdf_dict[renaming[name]] = val
    elif name == 'normal':
      parent_bsdf_dict = {'type': 'normalmap', 'normalmap': val}
    elif name == 'bump':
      parent_bsdf_dict = {'type': 'bumpmap', 'bumpmap': val}
    else:
      Tf.Warn(f"Unsupported input {name} in preview surface shader.")

  # Mitsuba's principled BSDF throws an error if both eta and specular are
  # provided. For transmissive materials (spec_trans > 0), we must keep eta
  # for proper refraction. Otherwise, we keep specular.
  if 'specular' in bsdf_dict and 'eta' in bsdf_dict:
    spec_trans = bsdf_dict.get('spec_trans', 0.0)
    if isinstance(spec_trans, dict) or (
        isinstance(spec_trans, (int, float)) and spec_trans > 0.0
    ):
      del bsdf_dict['specular']
    else:
      del bsdf_dict['eta']

  if parent_bsdf_dict:
    bsdf_dict = parent_bsdf_dict | {'nested_bsdf': bsdf_dict}

  # Only return an emitter dict if a radiance value was found.
  res_emitter_dict: dict[str, Any] | None = (
      emitter_dict if 'radiance' in emitter_dict else None
  )
  return bsdf_dict, res_emitter_dict


def has_displacement(prim: Usd.Prim) -> bool:
  """Returns True if the bound material or any subset material has a displacement shader connected."""
  prims_to_check = [prim] + [
      child for child in prim.GetChildren() if child.IsA(UsdGeom.Subset)
  ]

  for p in prims_to_check:
    material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
    if material:
      if find_shader(_get_displacement_output(material)) is not None:
        return True

  return False


def convert_material(
    prim: Usd.Prim,
) -> tuple[mi.BSDF | None, mi.Emitter | None, mi.Texture | None]:
  """Parses a USD geometry prim and returns Mitsuba material properties.

  This function will use the Mitsuba material variant if available, and
  otherwise fall back to using the USD preview surface shader, or even just
  the display color attribute.

  If the material is emissive, a Mitsuba `area` emitter will also be returned.

  Args:
    prim: The USD prim to extract the material from.

  Returns:
    A tuple of the Mitsuba BSDF, emitter and displacement texture (each may
    be None if not found).
  """

  bsdf, emitter, displacement = None, None, None
  material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
  if material:
    if shader := find_shader(_get_surface_output(material)):
      if shader.GetShaderId() == 'UsdPreviewSurface':
        bsdf_dict, emitter_dict = usd_preview_surface_to_mitsuba(shader)
        bsdf = mi.load_dict(bsdf_dict)
        emitter = mi.load_dict(emitter_dict) if emitter_dict else None
      elif shader.GetShaderId().startswith('mitsuba_'):
        bsdf_dict = usd_mitsuba_material_to_dict(shader)
        bsdf = mi.load_dict(bsdf_dict)

    if shader := find_shader(_get_displacement_output(material)):
      bsdf_dict = usd_mitsuba_material_to_dict(shader)
      displacement = mi.load_dict(bsdf_dict)
      if not isinstance(displacement, mi.Texture):
        raise ValueError(
            'Displacement texture is not a Mitsuba texture. Please use a'
            ' Mitsuba texture.'
        )

  elif display_color_attr := prim.GetAttribute('primvars:displayColor'):
    if colors := display_color_attr.Get():
      bsdf = mi.load_dict({
          'type': 'diffuse',
          'reflectance': {
              'type': 'rgb',
              'value': tuple(colors[0]),
          },
      })

  return bsdf, emitter, displacement
