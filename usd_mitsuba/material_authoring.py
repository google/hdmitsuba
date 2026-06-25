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

"""Authoring of USD materials from Mitsuba material descriptions.

This module provides functionality to convert Mitsuba BSDF descriptions
(in dictionary format) into USD shader networks. This is useful to 
create USD stages that use specific Mituba BSDFs & textures.
"""

from __future__ import annotations

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdShade

from usd_mitsuba.material import USD_TO_MI_PRINCIPLED_RENAMING

MI_SINGLE_CHANNEL_PROPERTY = ('roughness', 'eta', 'scale')

MI_TO_USD_PRINCIPLED_RENAMING = {
    v: k for k, v in USD_TO_MI_PRINCIPLED_RENAMING.items()
}

_USD_INPUT_TYPES = {
    'diffuseColor': Sdf.ValueTypeNames.Color3f,
    'emissiveColor': Sdf.ValueTypeNames.Color3f,
    'roughness': Sdf.ValueTypeNames.Float,
    'clearcoat': Sdf.ValueTypeNames.Float,
    'clearcoatRoughness': Sdf.ValueTypeNames.Float,
    'metallic': Sdf.ValueTypeNames.Float,
    'specular': Sdf.ValueTypeNames.Float,
    'ior': Sdf.ValueTypeNames.Float,
    'opacity': Sdf.ValueTypeNames.Float,
    'normal': Sdf.ValueTypeNames.Normal3f,
}


def _get_sdf_type_and_value(prop_val: Any) -> tuple[Any, Any]:
    """Return USD value type and value from Mitsuba property value."""
    if isinstance(prop_val, float):
        return Sdf.ValueTypeNames.Float, prop_val
    # Note: bool check must precede int because bool is a subclass of int.
    if isinstance(prop_val, bool):
        return Sdf.ValueTypeNames.Bool, prop_val
    if isinstance(prop_val, int):
        return Sdf.ValueTypeNames.Int, prop_val
    if isinstance(prop_val, str):
        return Sdf.ValueTypeNames.String, prop_val
    if isinstance(prop_val, mi.scalar_rgb.ScalarColor3d):
        return Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*prop_val)
    if isinstance(
        prop_val,
        (
            mi.scalar_rgb.Vector3f,
            mi.scalar_rgb.Point3f,
            mi.scalar_rgb.Vector3d,
            dr.scalar.Array3f64,
        ),
    ):
        return Sdf.ValueTypeNames.Float3, Gf.Vec3f(*prop_val)
    if isinstance(prop_val, mi.scalar_rgb.ScalarAffineTransform3d):
        return (
            Sdf.ValueTypeNames.Matrix3d,
            Gf.Matrix3d(np.array(prop_val.matrix).T),
        )
    if isinstance(prop_val, mi.scalar_rgb.ScalarAffineTransform4d):
        return (
            Sdf.ValueTypeNames.Matrix4d,
            Gf.Matrix4d(np.array(prop_val.matrix).T),
        )
    raise ValueError(f'Unsupported property value type: {type(prop_val)}')


def _add_texture_to_shader(
    shader: UsdShade.Shader,
    input_name: str,
    output_channel: str,
    texture_path: str,
) -> None:
    """Creates and connects a texture sampler network to a given shader input.

    This helper function sets up a UsdPrimvarReader_float2 to read UV
    coordinates and a UsdUVTexture sampler to read the texture file. It then
    connects the appropriate output channel (e.g., 'rgb', 'r') of the texture
    sampler to the specified input on the main shader.

    Args:
        shader: The shader to which the texture will be connected.
        input_name: The name of the input on the shader (e.g., 'diffuseColor').
        output_channel: The output channel to use from the texture ('rgb', 'r',
          'g', 'b').
        texture_path: The file path to the texture asset.
    """
    material_path = shader.GetPath().GetParentPath()
    texture = UsdShade.Shader.Define(
        shader.GetPrim().GetStage(), material_path.AppendPath(input_name)
    )
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_path)
    output_type = (
        Sdf.ValueTypeNames.Float3
        if output_channel == 'rgb'
        else Sdf.ValueTypeNames.Float
    )
    shader.CreateInput(input_name, output_type).ConnectToSource(
        texture.ConnectableAPI(), output_channel
    )


def _convert_mitsuba_node_to_usd(
    stage: Usd.Stage,
    material_path: Sdf.Path,
    shader: UsdShade.Shader,
    state: Any,
    node: Any,
) -> None:
    """Converts a Mitsuba node to a USD shader.

    Args:
      stage: The USD stage to define the shader on.
      material_path: The Sdf.Path to define the shader at.
      shader: The USD shader to populate.
      state: The Mitsuba parser state (for resolving child references).
      node: The current Mitsuba node to convert.
    """
    shader.CreateIdAttr(f'mitsuba_{node.props.plugin_name()}')
    for key in node.props.keys():
        prop_type = node.props.type(key)
        prop_val = node.props[key]
        if prop_type == mi.Properties.Type.ResolvedReference:
            child_node = state.nodes[prop_val.index()]
            if child_node.type == mi.ObjectType.Texture:
                is_color = key not in MI_SINGLE_CHANNEL_PROPERTY
                output_channel = 'rgb' if is_color else 'r'
                if child_node.props.plugin_name() == 'bitmap':
                    texture_path = child_node.props['filename']
                    _add_texture_to_shader(
                        shader, key, output_channel, texture_path)
                else:
                    output_type = (
                        Sdf.ValueTypeNames.Float3
                        if is_color
                        else Sdf.ValueTypeNames.Float
                    )
                    tex_shader_path = material_path.AppendPath(key)
                    tex_shader = UsdShade.Shader.Define(stage, tex_shader_path)
                    _convert_mitsuba_node_to_usd(
                        stage, tex_shader_path, tex_shader, state, child_node
                    )
                    if is_color:
                        tex_shader.CreateOutput(
                            'rgb', Sdf.ValueTypeNames.Float3)
                    else:
                        tex_shader.CreateOutput('r', Sdf.ValueTypeNames.Float)

                    shader.CreateInput(key, output_type).ConnectToSource(
                        tex_shader.ConnectableAPI(), output_channel
                    )
            elif child_node.type == mi.ObjectType.BSDF:
                bsdf_shader_path = material_path.AppendPath(key)
                bsdf_shader = UsdShade.Shader.Define(stage, bsdf_shader_path)
                _convert_mitsuba_node_to_usd(
                    stage, bsdf_shader_path, bsdf_shader, state, child_node
                )
                if not bsdf_shader.GetOutput('out'):
                    bsdf_shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
                shader.CreateInput(key, Sdf.ValueTypeNames.Token).ConnectToSource(
                    bsdf_shader.ConnectableAPI(), 'out'
                )
        else:
            sdf_type, sdf_val = _get_sdf_type_and_value(prop_val)
            if sdf_type:
                shader.CreateInput(key, sdf_type).Set(sdf_val)


def add_mitsuba_shader(
    stage: Usd.Stage,
    path: Sdf.Path,
    mitsuba_bsdf_dict: dict[str, Any],
) -> UsdShade.Shader:
    """Adds a Mitsuba shader to a USD stage.

    This function converts a Mitsuba BSDF in dictionary format to a USD shader.
    It supports all standard Mitsuba BSDFs, including normal and bumpmaps.

    Args:
      stage: The USD stage to work in.
      path: The path to the root of the shader.
      mitsuba_bsdf_dict: The Mitsuba BSDF dictionary. This is expected to be a
        standard Mitsuba BSDF dictionary.

    Returns:
      The USD shader.
    """
    shader = UsdShade.Shader.Define(stage, path)
    config = mi.parser.ParserConfig('scalar_rgb')
    state = mi.parser.parse_dict(config, mitsuba_bsdf_dict)

    if state.root.type != mi.ObjectType.BSDF:
        raise ValueError(
            'The Mitsuba BSDF dictionary must contain a BSDF object, but got'
            f' {state.root.type}.'
        )

    _convert_mitsuba_node_to_usd(stage, path, shader, state, state.root)
    return shader


def _parse_mi_node_to_usd_preview(
    mi_node: Any,
    mi_state: Any,
    usd_shader: UsdShade.Shader,
    stage: Usd.Stage,
    base_path: Sdf.Path,
) -> None:
    """Recursive helper to parse a Mitsuba node and populate a UsdPreviewSurface.

    Args:
      mi_node: The current Mitsuba node (from mi.parser.State).
      mi_state: The Mitsuba parser state (for resolving references).
      usd_shader: The UsdPreviewSurface shader to populate.
      stage: The Usd.Stage to define new nodes on.
      base_path: The Sdf.Path of the main shader (for scoping child nodes).
    """
    inverse_renaming = MI_TO_USD_PRINCIPLED_RENAMING
    plugin_name = mi_node.props.plugin_name()
    if plugin_name in ('normalmap', 'bumpmap'):
        normal_tex_node = None
        nested_bsdf_node = None
        for key in mi_node.props.keys():
            if mi_node.props.type(key) == mi.Properties.Type.ResolvedReference:
                child_node = mi_state.nodes[mi_node.props[key].index()]
                if child_node.type == mi.ObjectType.BSDF:
                    nested_bsdf_node = child_node
                elif child_node.type == mi.ObjectType.Texture:
                    normal_tex_node = child_node
        if not nested_bsdf_node or not normal_tex_node:
            raise ValueError(
                f"'{plugin_name}' at {base_path} misses nested BSDF or texture"
            )
        _parse_mi_node_to_usd_preview(
            nested_bsdf_node, mi_state, usd_shader, stage, base_path
        )
        if normal_tex_node.props.plugin_name() != 'bitmap':
            raise ValueError(
                f'Unsupported {plugin_name} texture type at {base_path}')
        if plugin_name == 'normalmap':
            _add_texture_to_shader(
                usd_shader, 'normal', 'rgb', normal_tex_node.props['filename']
            )
        else:
            pass  # Bumpmap texture is not supported in preview surface.
        return

    if plugin_name != 'principled':
        raise ValueError(
            f"Unsupported Mitsuba BSDF type '{plugin_name}' at {base_path}. "
            "Only 'principled', 'normalmap', and 'bumpmap' are supported."
        )

    for mi_name in mi_node.props.keys():
        if mi_name not in inverse_renaming:
            continue
        usd_name = inverse_renaming[mi_name]
        invert_value = usd_name in ('opacity', 'clearcoatRoughness')
        usd_type = _USD_INPUT_TYPES[usd_name]
        prop_type = mi_node.props.type(mi_name)
        prop_val = mi_node.props[mi_name]
        is_vector3 = usd_type in (
            Sdf.ValueTypeNames.Color3f,
            Sdf.ValueTypeNames.Float3,
        )
        if prop_type == mi.Properties.Type.ResolvedReference:
            child_node = mi_state.nodes[prop_val.index()]
            if (
                child_node.type == mi.ObjectType.Texture
                and child_node.props.plugin_name() == 'bitmap'
            ):
                _add_texture_to_shader(
                    usd_shader,
                    usd_name,
                    'rgb' if is_vector3 else 'r',
                    child_node.props['filename'],
                )
            else:
                raise ValueError(
                    f"Warning: Unsupported texture type for '{mi_name}' at {base_path}"
                )
        elif prop_type == mi.Properties.Type.Color:
            usd_input = usd_shader.CreateInput(usd_name, usd_type)
            if is_vector3:
                usd_input.Set(Gf.Vec3f(*prop_val))
            else:
                val = dr.mean(prop_val)
                if invert_value:
                    val = 1.0 - val
                usd_input.Set(val)
        elif prop_type == mi.Properties.Type.Float:
            usd_input = usd_shader.CreateInput(usd_name, usd_type)
            val = float(prop_val)
            if invert_value:
                val = 1.0 - val
            if is_vector3:
                usd_input.Set(Gf.Vec3f(val, val, val))
            else:
                usd_input.Set(val)
        else:
            raise ValueError(
                f"Unsupported prop type '{prop_type}' for '{mi_name}'")


def add_mitsuba_shader_as_usd_preview_surface(
    stage: Usd.Stage,
    bsdf_dict: dict[str, Any],
    prim_path: Sdf.Path,
    displacement_texture: str | None = None,
) -> UsdShade.Shader:
    """Converts a Mitsuba BSDF dictionary to a UsdPreviewSurface shader.

    Supports 'principled' BSDFs and 'normalmap' or 'bumpmap' wrappers. This
    function could also be extended to support approximate conversion for other
    Mitsuba BSDFs.

    Args:
        stage: The Usd.Stage to define the shader on.
        bsdf_dict: The Mitsuba BSDF dictionary.
        prim_path: The Sdf.Path to create the UsdPreviewSurface shader at.
        displacement_texture: Optional path to a displacement texture.

    Returns:
        The created UsdShade.Shader (UsdPreviewSurface) or None if failed.
    """
    config = mi.parser.ParserConfig('scalar_rgb')
    state = mi.parser.parse_dict(config, bsdf_dict)
    root_node = state.root
    if root_node.type != mi.ObjectType.BSDF:
        raise ValueError(
            f'Error: Root object in dict at {prim_path} is not a BSDF (got'
            f' {root_node.type}).'
        )

    shader = UsdShade.Shader.Define(stage, prim_path)
    shader.CreateIdAttr('UsdPreviewSurface')
    _parse_mi_node_to_usd_preview(root_node, state, shader, stage, prim_path)
    if displacement_texture is not None:
        _add_texture_to_shader(shader, 'displacement',
                               'out', displacement_texture)
    return shader
