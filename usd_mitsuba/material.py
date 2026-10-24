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

from pxr import Ar
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
_USD_PREVIEW_SURFACE_NON_COLOR_INPUTS = frozenset({
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

# Mitsuba-specific non-color shader inputs for custom shaders.
_MITSUBA_NON_COLOR_INPUTS = frozenset({
    'normalmap',
    'bumpmap',
})

_NON_COLOR_INPUTS = (
    _USD_PREVIEW_SURFACE_NON_COLOR_INPUTS | _MITSUBA_NON_COLOR_INPUTS
)

_CHANNEL_INDICES = {'r': 0, 'g': 1, 'b': 2, 'a': 3}

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


def _find_shader_output(
    prop: UsdShade.Input | UsdShade.Output | None,
) -> tuple[UsdShade.Shader, str] | None:
  """Returns the first upstream UsdShade.Shader and its output base name."""
  if prop and (
      attrs := prop.GetValueProducingAttributes(shaderOutputsOnly=True)
  ):
    return UsdShade.Shader(attrs[0].GetPrim()), attrs[0].GetBaseName()
  return None


def find_shader(
    prop: UsdShade.Input | UsdShade.Output,
) -> UsdShade.Shader | None:
  """Returns the first upstream UsdShade.Shader.

  This function finds the first UsdShade.Shader connected to a given input or
  output.

  Args:
    prop: The UsdShade.Input or UsdShade.Output to find the shader for.
  """
  return conn[0] if (conn := _find_shader_output(prop)) else None


def _get_volume_shader(
    material: UsdShade.Material,
) -> UsdShade.Shader | None:
  """Gets the Mitsuba-specific volume shader or falls back to default."""
  shader = find_shader(material.GetVolumeOutput('mitsuba')) or find_shader(
      material.GetVolumeOutput()
  )
  if shader is not None and shader.GetShaderId().startswith(_MI_PREFIX):
    return shader
  return None


def _use_raw_bitmap(
    cs: str, input_name: str | None, ch_idx: int
) -> bool:
  """Returns whether a UsdUVTexture bitmap should skip sRGB decoding.

  Covers the Gf.ColorSpace names (e.g. `lin_rec709_scene`, `srgb_ap1_scene`,
  `g22_rec709_scene`, `raw`) as well as UsdUVTexture's `sRGB` token and common
  OCIO aliases (e.g. `lin_srgb`, `srgb_texture`). Alpha is always linear.

  Args:
    cs: The source color space name (may be empty or 'auto').
    input_name: The downstream input name, used if `cs` is unset or 'auto'.
    ch_idx: The extracted channel index (3 = alpha), or -1 for all channels.

  Returns:
    True if the bitmap should be treated as linear data.
  """
  if ch_idx == 3:
    return True
  lower = cs.lower()
  if lower and lower not in ('auto', 'unknown'):
    if lower in ('raw', 'data', 'identity', 'linear', 'acescg') or (
        lower.startswith('lin_')
    ):
      return True
    if lower == 'srgb' or lower.startswith(('srgb_', 'g18_', 'g22_', 'g24_')):
      return False
  return input_name in _NON_COLOR_INPUTS


def _open_asset_stream(filename: str) -> mi.MemoryStream | None:
  """Opens an Ar-resolved asset into an in-memory Mitsuba stream."""
  asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(filename))
  if asset is not None and (buffer := asset.GetBuffer()) is not None:
    stream = mi.MemoryStream(asset.GetSize())
    stream.write(buffer)
    stream.seek(0)
    return stream
  return None


def _extract_usd_transform2d(
    shader: UsdShade.Shader,
) -> mi.scalar_rgb.ScalarAffineTransform3f | None:
  """Extracts a Mitsuba UV transform from upstream UsdTransform2d nodes."""
  identity = mi.scalar_rgb.ScalarAffineTransform3f()
  xf = identity
  curr = find_shader(shader.GetInput('st'))
  while curr is not None and curr.GetIdAttr().Get() == 'UsdTransform2d':
    if (t := curr.GetInput('translation').Get()) is not None:
      xf = xf.translate(tuple(t))
    if r := curr.GetInput('rotation').Get():
      xf = xf.rotate(r)
    if (s := curr.GetInput('scale').Get()) is not None:
      xf = xf.scale(tuple(s))
    curr = find_shader(curr.GetInput('in'))
  if xf == identity:
    return None
  # Mesh UVs are V-flipped on load (uv_mi = F @ uv_usd, with F = F^-1), so
  # conjugate the USD-space transform into Mitsuba UV space: F @ xf @ F.
  flip = identity.translate([0, 1]).scale([1, -1])
  return flip @ xf @ flip


def _load_uv_texture_image(
    filename: str, ch_idx: int
) -> dict[str, Any] | None:
  """Returns the texture dict for a UsdUVTexture image, or None if unusable."""
  # Strip the closing bracket of package paths like `a.usdz[tex.png]`.
  _, ext = os.path.splitext(filename.rstrip(']'))
  # This check is not identical to hdmitsuba's more strict check,
  # but should be sufficient for the Python conversion.
  if ext.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
    Tf.Warn(
        f"Texture file '{filename}' has unsupported extension '{ext}'."
        ' Using fallback.'
    )
    return None
  if os.path.exists(filename) and ch_idx < 0:
    return {'type': 'bitmap', 'filename': filename}
  try:
    stream = None if os.path.exists(filename) else _open_asset_stream(filename)
    bmp = mi.Bitmap(stream or filename)
  except RuntimeError as e:
    Tf.Warn(f"Failed to load texture '{filename}': {e}. Using fallback.")
    return None
  if ch_idx == 3 and not bmp.has_alpha():
    return {'type': 'uniform', 'value': 1.0}
  if ch_idx >= 0 and bmp.pixel_format() != mi.Bitmap.PixelFormat.Y:
    # Alpha is the last channel. Color channels missing from the image
    # (e.g. `g` of a grayscale+alpha image) read the first channel.
    color_ch = bmp.channel_count() - bmp.has_alpha()
    src_ch = color_ch if ch_idx == 3 else (ch_idx if ch_idx < color_ch else 0)
    arr = np.ascontiguousarray(np.array(bmp, copy=False)[..., src_ch])
    ch_bmp = mi.Bitmap(arr, mi.Bitmap.PixelFormat.Y)
    # Alpha stays linear since _use_raw_bitmap() always sets `raw` for it.
    ch_bmp.set_srgb_gamma(bmp.srgb_gamma())
    bmp = ch_bmp
  return {'type': 'bitmap', 'bitmap': bmp}


def _convert_uv_texture(
    shader: UsdShade.Shader,
    input_name: str | None,
    output_name: str | None,
) -> dict[str, Any] | None:
  """Extracts texture information from a UsdUVTexture shader."""
  is_normal = input_name in ('normal', 'normalmap')
  ch_idx = _CHANNEL_INDICES.get(output_name or '', -1)
  if is_normal and ch_idx >= 0:
    Tf.Warn(
        f"Normal map input '{input_name}' is connected to single-channel"
        f" output '{output_name}'. Normal maps require at least 3 channels."
    )
    return None

  # 1. Use the texture image if it is valid.
  file_input = shader.GetInput('file')
  asset_path = file_input.Get()
  filename = asset_path.resolvedPath if asset_path else None
  if filename and (texture := _load_uv_texture_image(filename, ch_idx)):
    if texture['type'] != 'bitmap':
      return texture
    # Like UsdImaging, only consider the `colorSpace` metadata of the file
    # attribute itself if `sourceColorSpace` is unset or 'auto'.
    cs = shader.GetInput('sourceColorSpace').Get()
    if not cs or cs == 'auto':
      cs = file_input.GetAttr().GetColorSpace()
    if _use_raw_bitmap(cs, input_name, ch_idx):
      texture['raw'] = True

    # Mitsuba's bitmap only has a single wrap_mode: use wrapS, falling back to
    # wrapT if wrapS is unauthored, 'useMetadata' or unsupported.
    for param in ('wrapS', 'wrapT'):
      w = shader.GetInput(param).Get()
      if w in ('repeat', 'clamp', 'mirror'):
        texture['wrap_mode'] = w
        break
      if w and w != 'useMetadata':
        Tf.Warn(
            f"Unsupported wrap mode '{w}' on '{param}' for UsdUVTexture"
            f" '{filename}'; ignoring."
        )

    if to_uv := _extract_usd_transform2d(shader):
      texture['to_uv'] = to_uv
    return texture

  if is_normal:
    return None

  # 2. If the file is missing/unresolved, use the fallback color (or magenta).
  fb = shader.GetInput('fallback').Get()
  fb = tuple(fb) if fb is not None else (1.0, 0.0, 1.0, 1.0)
  if ch_idx >= 0:
    return {
        'type': 'uniform',
        'value': float(fb[ch_idx]) if ch_idx < len(fb) else 1.0,
    }
  return {'type': 'rgb', 'value': fb[:3]}


def usd_mitsuba_material_to_dict(
    shader: UsdShade.Shader,
    input_name: str | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
  """Converts a USD shader to a Mitsuba BSDF dictionary.

  Args:
    shader: The USD shader to convert.
    input_name: Optional downstream input name used for color-space heuristics.
    output_name: Optional output channel name on `shader` (e.g. 'r', 'rgb').

  Returns:
    The Mitsuba dictionary representation of the shader, or an empty dictionary
    if the shader is rejected (e.g. a single-channel texture as a normal map).
  """

  def _traverse(
      shader: UsdShade.Shader,
      input_name: str | None = None,
      output_name: str | None = None,
  ) -> dict[str, Any] | None:
    shader_id = shader.GetIdAttr().Get()
    if shader_id == 'UsdUVTexture':
      return _convert_uv_texture(shader, input_name, output_name)

    if not shader_id.startswith(_MI_PREFIX):
      raise ValueError(f'The shader {shader_id} is not a Mitsuba shader.')

    result = {'type': shader_id.removeprefix(_MI_PREFIX)}
    for usd_input in shader.GetInputs():
      name = usd_input.GetBaseName()
      if conn := _find_shader_output(usd_input):
        if (child := _traverse(conn[0], name, conn[1])) is not None:
          result[name] = child
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

    if (
        result['type'] == 'gridvolume'
        and 'filename' in result
        and 'grid' not in result
        and not os.path.exists(result['filename'])
        and (stream := _open_asset_stream(result['filename']))
    ):
      result['grid'] = mi.VolumeGrid(stream)
      del result['filename']

    return result

  return _traverse(shader, input_name, output_name) or {}


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
    if conn := _find_shader_output(usd_input):
      if conn[0].GetIdAttr().Get() == 'UsdUVTexture':
        val = _convert_uv_texture(conn[0], name, conn[1])
        if val is not None and name in ('opacity', 'clearcoatRoughness'):
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
) -> tuple[
    mi.BSDF | None, mi.Emitter | None, mi.Texture | None, mi.Medium | None
]:
  """Parses a USD geometry prim and returns Mitsuba material properties.

  This function will use the Mitsuba material variant if available, and
  otherwise fall back to using the USD preview surface shader, or even just
  the display color attribute.

  If the material is emissive, a Mitsuba `area` emitter will also be returned.

  Args:
    prim: The USD prim to extract the material from.

  Returns:
    A tuple of the Mitsuba BSDF, emitter, displacement texture, and interior
    medium (each may be None if not found).
  """

  bsdf, emitter, displacement, interior_medium = None, None, None, None
  material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
  if material:
    if shader := find_shader(_get_surface_output(material)):
      if shader.GetShaderId() == 'UsdPreviewSurface':
        bsdf_dict, emitter_dict = usd_preview_surface_to_mitsuba(shader)
        bsdf = mi.load_dict(bsdf_dict)
        emitter = mi.load_dict(emitter_dict) if emitter_dict else None
      elif shader.GetShaderId().startswith(_MI_PREFIX):
        if bsdf_dict := usd_mitsuba_material_to_dict(shader):
          bsdf = mi.load_dict(bsdf_dict)

    if shader := _get_volume_shader(material):
      medium_dict = usd_mitsuba_material_to_dict(shader, 'volume')
      medium = mi.load_dict(medium_dict)
      if not isinstance(medium, mi.Medium):
        raise ValueError(
            'Volume shader is not a Mitsuba medium. Please use a Mitsuba'
            ' medium.'
        )
      interior_medium = medium
      if bsdf is None:
        bsdf = mi.load_dict({'type': 'null'})

    if conn := _find_shader_output(_get_displacement_output(material)):
      bsdf_dict = usd_mitsuba_material_to_dict(conn[0], 'displacement', conn[1])
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

  return bsdf, emitter, displacement, interior_medium

