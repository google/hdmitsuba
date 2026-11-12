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

"""Converts between USD lights and Mitsuba emitters."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from pxr import Gf
from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdLux

from usd_mitsuba import motion
from usd_mitsuba import util


def _get_or(attr: Usd.Attribute, time: Usd.TimeCode, default: Any) -> Any:
  """Returns the value of `attr`, or `default` if it has none.

  Like usdImaging, shaping attributes are read whenever the prim has them, even
  if `UsdLux.ShapingAPI` is not applied.
  """
  value = attr.Get(time)
  return default if value is None else value


def _has_explicit_emitter_frame(overrides: Mapping[str, Any]) -> bool:
  return any(
      key in overrides
      for key in ('to_world', 'direction', 'origin', 'position')
  )


def _maybe_wrap_shaping(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    emitter_dict: dict[str, Any],
    to_world: mi.ScalarTransform4f | mi.AnimatedTransform4f,
) -> dict[str, Any]:
  """Wraps `emitter_dict` in an `'ies'` emitter if `prim` has active shaping.

  Point lights realize the cone as `'spot'` emitters, so it only applies to
  area emitters. `to_world` is the rigid light frame of the `'ies'` emitter,
  whose +Z axis is the photometric axis (the light's -Z axis in USD).
  """
  shaping_api = UsdLux.ShapingAPI(prim)
  shaping: dict[str, Any] = {}
  if ies_file := shaping_api.GetShapingIesFileAttr().Get(time):
    if not ies_file.resolvedPath:
      raise ValueError(
          f'Could not resolve IES file {ies_file.path!r} of {prim.GetPath()}.'
      )
    shaping |= {
        'filename': ies_file.resolvedPath,
        'angle_scale': _get_or(
            shaping_api.GetShapingIesAngleScaleAttr(), time, 0.0
        ),
        'normalize': _get_or(
            shaping_api.GetShapingIesNormalizeAttr(), time, False
        ),
    }
  cone_angle = _get_or(shaping_api.GetShapingConeAngleAttr(), time, 180.0)
  softness = min(
      max(_get_or(shaping_api.GetShapingConeSoftnessAttr(), time, 0.0), 0.0),
      1.0,
  )
  if emitter_dict.get('type') not in ('point', 'spot') and (
      cone_angle < 180.0 or softness > 0.0
  ):
    shaping |= {'cone_angle': cone_angle, 'cone_softness': softness}
  if (focus := _get_or(shaping_api.GetShapingFocusAttr(), time, 0.0)) > 0.0:
    shaping |= {
        'focus': focus,
        'focus_tint': {
            'type': 'rgb',
            'value': mi.ScalarColor3f(
                *_get_or(shaping_api.GetShapingFocusTintAttr(), time, (0, 0, 0))
            ),
        },
    }
  if not shaping:
    return emitter_dict
  return {
      'type': 'ies',
      'nested_emitter': emitter_dict,
      'to_world': to_world,
      **shaping,
  }


def _area_emitter(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    radiance: mi.ScalarColor3f,
    to_world: mi.ScalarTransform4f,
    texture_file: Sdf.AssetPath | None = None,
    emitter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Returns an `'area'` emitter, wrapped in `'ies'` if `prim` has shaping.

  `to_world` is the transform of the emitting shape, whose +Z axis is the
  photometric axis. A `texture_file` modulates `radiance`.
  """
  has_custom_radiance = bool(
      emitter_overrides and 'radiance' in emitter_overrides
  )
  radiance_dict: dict[str, Any] = {'type': 'rgb', 'value': radiance}
  if texture_file and not has_custom_radiance:
    if not texture_file.resolvedPath:
      raise ValueError(
          f'Could not resolve texture {texture_file.path!r} of'
          f' {prim.GetPath()}.'
      )
    radiance_dict = {
        'type': 'math',
        'expr': 'in[0] * in[1]',
        'in0': {
            'type': 'bitmap',
            'bitmap': mi.Bitmap(texture_file.resolvedPath),
        },
        'in1': {'type': 'srgb', 'color': radiance, 'unbounded': True},
    }
  emitter_dict: dict[str, Any] = {
      'type': 'area',
      'radiance': radiance_dict,
  }
  if emitter_overrides and emitter_overrides.get('type') == 'ies':
    wrapped = _maybe_wrap_shaping(
        prim,
        time,
        emitter_dict,
        util.remove_scale_from_transform(to_world),
    )
    if wrapped.get('type') != 'ies':
      wrapped = {'type': 'ies'}
    elif _has_explicit_emitter_frame(emitter_overrides):
      wrapped.pop('to_world', None)
    util.deep_merge_dict(wrapped, emitter_overrides)
    return wrapped
  if emitter_overrides:
    custom_type = emitter_overrides.get('type')
    if custom_type and custom_type != 'area':
      emitter_dict = {'type': custom_type}
    util.deep_merge_dict(emitter_dict, emitter_overrides)
  return _maybe_wrap_shaping(
      prim,
      time,
      emitter_dict,
      util.remove_scale_from_transform(to_world),
  )


def _convert_dome_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f | mi.AnimatedTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
    emitter_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  dome_light = UsdLux.DomeLight(prim)
  texture_file_attr = dome_light.GetTextureFileAttr().Get(time)
  custom_type = (emitter_overrides or {}).get('type', '')
  if texture_file_attr and custom_type in ('', 'envmap'):
    mean = dr.mean(color)
    result: dict[str, Any] = {
        'type': 'envmap',
        'to_world': world_transform,
        'scale': intensity * mean,
    }
    if not (
        emitter_overrides
        and ('filename' in emitter_overrides or 'bitmap' in emitter_overrides)
    ):
      bitmap = mi.Bitmap(texture_file_attr.resolvedPath)
      # Mitsuba's `envmap` only has a scalar `scale` (the mean of `color`), so
      # the per-channel tint is baked into its texture.
      if mean > 0.0 and not dr.allclose(color, mean):
        linear = bitmap.convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False
        )
        bitmap = mi.Bitmap(np.array(linear) * np.array(color / mean))
      result['bitmap'] = bitmap
    return result
  return {
      'type': 'constant',
      'radiance': {'type': 'rgb', 'value': color * intensity},
  }


def _is_point_light(
    sphere_light: UsdLux.SphereLight, time: Usd.TimeCode
) -> bool:
  return (
      sphere_light.GetTreatAsPointAttr().Get(time)
      or sphere_light.GetRadiusAttr().Get(time) == 0
  )


def _is_area_light(prim: Usd.Prim, time: Usd.TimeCode) -> bool:
  """Returns whether the light `prim` is converted to a Mitsuba shape.

  Mirrors `LightSpec::IsAreaLight` in hdmitsuba/spec_types.h.

  Args:
    prim: The USD light prim.
    time: The time code to evaluate at.

  Returns:
    True if the light is converted to a shape with an area emitter.
  """
  if (
      prim.IsA(UsdLux.RectLight)
      or prim.IsA(UsdLux.DiskLight)
      or prim.IsA(UsdLux.CylinderLight)
  ):
    return True
  return prim.IsA(UsdLux.SphereLight) and not _is_point_light(
      UsdLux.SphereLight(prim), time
  )


def _disk_area(transform: mi.ScalarTransform4f, radius: float) -> float:
  """Returns the area of a disk of radius `radius` under `transform`."""
  return dr.pi * dr.norm(
      dr.cross(
          transform @ mi.ScalarVector3f(radius, 0, 0),
          transform @ mi.ScalarVector3f(0, radius, 0),
      )
  )


def _sphere_area(transform: mi.ScalarTransform4f, radius: float) -> float:
  """Returns the area of a sphere of radius `radius` under `transform`.

  Non-uniformly scaled spheres are ellipsoids, whose area is approximated as in
  hdEmbree.
  """
  a, b, c = (
      dr.norm(transform @ (radius * mi.ScalarVector3f(axis)))
      for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
  )
  p = 1.6
  return 4.0 * dr.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (
      1.0 / p
  )


def _convert_sphere_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f | mi.AnimatedTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  color = color * intensity
  sphere_light = UsdLux.SphereLight(prim)
  normalize = sphere_light.GetNormalizeAttr().Get(time)
  radius = sphere_light.GetRadiusAttr().Get(time)
  if _is_point_light(sphere_light, time):
    if normalize:
      color *= 0.25
    elif radius > 0:
      # A point light stands in for its sphere, whose radiant intensity is L
      # times its average projected area, a quarter of its surface area.
      # Zero-radius lights keep `intensity` as radiant intensity.
      # `world_transform` has its scale removed, so use the prim's.
      color *= 0.25 * _sphere_area(util.get_world_transform(prim, time), radius)
    emitter_dict: dict[str, Any] = {
        'type': 'point',
        'to_world': world_transform,
        'intensity': {'type': 'rgb', 'value': color},
    }
    shaping = UsdLux.ShapingAPI(prim)
    if cone_angle := shaping.GetShapingConeAngleAttr().Get(time):
      softness = _get_or(shaping.GetShapingConeSoftnessAttr(), time, 0.0)
      emitter_dict |= {
          'type': 'spot',
          'beam_width': cone_angle * (1.0 - softness),
          'cutoff_angle': cone_angle,
      }
    return _maybe_wrap_shaping(prim, time, emitter_dict, world_transform)

  area = _sphere_area(world_transform, radius)
  # Mitsuba's `sphere` does not support non-uniform scale, so use the radius of
  # the sphere with the same area.
  # The rotation leaves the sphere unchanged but aligns its IES frame.
  world_transform = util.remove_scale_from_transform(
      world_transform.rotate([1, 0, 0], 180)
  ).scale(dr.sqrt(area / (4 * dr.pi)))
  if normalize:
    color /= area
  return {
      'type': 'sphere',
      'to_world': world_transform,
      'emitter': _area_emitter(prim, time, color, world_transform),
  }


def _convert_distant_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f | mi.AnimatedTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  distant_light = UsdLux.DistantLight(prim)
  # Mitsuba's `directional` emitter requires angles below 180 deg.
  angle = min(max(distant_light.GetAngleAttr().Get(time), 0.0), 179.99)
  if angle > 0.0:
    cos_theta = math.cos(math.radians(0.5 * angle))
    if distant_light.GetNormalizeAttr().Get(time):
      intensity *= 2.0 / (1.0 + cos_theta)
    else:
      intensity *= 2.0 * math.pi * (1.0 - cos_theta)
  return {
      'type': 'directional',
      'to_world': world_transform,
      'irradiance': {'type': 'rgb', 'value': color * intensity},
      'angle': angle,
  }


def _convert_rect_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  rect_light = UsdLux.RectLight(prim)
  scale = mi.ScalarVector3f(
      0.5 * rect_light.GetWidthAttr().Get(time),
      0.5 * rect_light.GetHeightAttr().Get(time),
      1,
  )
  world_transform = world_transform.rotate([1, 0, 0], 180).scale(scale)
  if rect_light.GetNormalizeAttr().Get(time):
    area = dr.norm(
        dr.cross(
            world_transform @ mi.ScalarVector3f(2, 0, 0),
            world_transform @ mi.ScalarVector3f(0, 2, 0),
        )
    )
    color /= area

  return {
      'type': 'rectangle',
      'to_world': world_transform,
      'emitter': _area_emitter(
          prim,
          time,
          color * intensity,
          world_transform,
          rect_light.GetTextureFileAttr().Get(time),
      ),
  }


def _convert_disk_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  disk_light = UsdLux.DiskLight(prim)
  radius = disk_light.GetRadiusAttr().Get(time)
  if disk_light.GetNormalizeAttr().Get(time):
    color /= _disk_area(world_transform, radius)
  world_transform = world_transform.rotate([1, 0, 0], 180).scale(radius)

  return {
      'type': 'disk',
      'to_world': world_transform,
      'emitter': _area_emitter(
          prim, time, color * intensity, world_transform
      ),
  }


def _cylinder_perimeter(
    transform: mi.ScalarTransform4f, radius: float
) -> float:
  """Returns the perimeter of a circle of radius `radius` in the YZ plane.

  Non-uniformly scaled circles are ellipses, whose perimeter under `transform`
  is approximated as in hdEmbree.
  """
  a = dr.norm(transform @ mi.ScalarVector3f(0, radius, 0))
  b = dr.norm(transform @ mi.ScalarVector3f(0, 0, radius))
  return dr.pi * (3.0 * (a + b) - dr.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def _convert_cylinder_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  cylinder_light = UsdLux.CylinderLight(prim)
  length = dr.norm(
      world_transform
      @ mi.ScalarVector3f(cylinder_light.GetLengthAttr().Get(time), 0, 0)
  )
  perimeter = _cylinder_perimeter(
      world_transform, cylinder_light.GetRadiusAttr().Get(time)
  )
  if cylinder_light.GetNormalizeAttr().Get(time):
    color /= perimeter * length
  light_frame = util.remove_scale_from_transform(
      world_transform.rotate([1, 0, 0], 180)
  )
  # Mitsuba's `cylinder` spans z in [0, 1] and does not support non-uniform
  # scale of its cross-section, so use the radius of the cylinder with the
  # same area.
  radius = perimeter / (2.0 * dr.pi)
  return {
      'type': 'cylinder',
      'to_world': light_frame.scale([length, radius, radius])
      .rotate([0, 1, 0], 90)
      .translate([0, 0, -0.5]),
      'emitter': _area_emitter(prim, time, color * intensity, light_frame),
  }


def _base_emission(
    prim: Usd.Prim, time: Usd.TimeCode
) -> tuple[mi.ScalarColor3f, float]:
  """Returns the unnormalized color and intensity of the light `prim`."""
  light_api = UsdLux.LightAPI(prim)
  color = light_api.GetColorAttr().Get(time)
  if light_api.GetEnableColorTemperatureAttr().Get(time):
    color = Gf.CompMult(
        color,
        UsdLux.BlackbodyTemperatureAsRgb(
            light_api.GetColorTemperatureAttr().Get(time)
        ),
    )
  intensity = (
      light_api.GetDiffuseAttr().Get(time)
      * light_api.GetIntensityAttr().Get(time)
      * 2.0 ** light_api.GetExposureAttr().Get(time)
  )
  return mi.ScalarColor3f(*color), intensity


def convert_mesh_light(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    world_transform: Gf.Matrix4d,
    emitter_overrides: dict[str, Any] | None = None,
) -> mi.Emitter | None:
  """Returns the emitter of a `MeshLightAPI` mesh.

  Args:
    prim: The USD mesh prim.
    time: The time code to evaluate attributes at.
    world_transform: The world transform of `prim`.
    emitter_overrides: Optional `mitsuba:emitter:*` overrides from the shape.

  Returns:
    The `'area'` emitter, wrapped in an `'ies'` emitter if `prim` has active
    shaping, or None if `prim` has no `MeshLightAPI`.
  """
  if not prim.HasAPI(UsdLux.MeshLightAPI):
    return None
  color, intensity = _base_emission(prim, time)
  # Meshes bake their transform into the vertices, so the IES light frame
  # (USD lights emit along -Z) must be given explicitly.
  return mi.load_dict(
      _area_emitter(
          prim,
          time,
          color * intensity,
          util.to_mitsuba_transform(world_transform).rotate([1, 0, 0], 180),
          emitter_overrides=emitter_overrides,
      )
  )


_LIGHT_CONVERTERS = [
    (UsdLux.DomeLight, _convert_dome_light),
    (UsdLux.SphereLight, _convert_sphere_light),
    (UsdLux.DistantLight, _convert_distant_light),
    (UsdLux.RectLight, _convert_rect_light),
    (UsdLux.DiskLight, _convert_disk_light),
    (UsdLux.CylinderLight, _convert_cylinder_light),
]


def _sample_light_transform(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    motion_interval: tuple[float, float],
    emitter_plugin_type: str = '',
) -> mi.ScalarTransform4f | mi.AnimatedTransform4f:
  """Samples the light's world transform, falling back to static for area lights."""
  if _is_area_light(prim, time):
    transform = motion.sample_world_transform(prim, time, motion_interval)
    if isinstance(transform, mi.AnimatedTransform4f):
      Tf.Warn(f'Motion blur is not supported for light {prim.GetPath()}.')
      return util.get_world_transform(prim, time)
    return transform

  # USD distant, spot, and shaped point lights emit along -Z, while Mitsuba's
  # directional, spot, projector, and ies emitters point along +Z. Unshaped
  # point lights are isotropic, so they are flipped too.
  shaping_api = UsdLux.ShapingAPI(prim)
  has_shaping = bool(
      shaping_api.GetShapingIesFileAttr().Get(time)
      or _get_or(shaping_api.GetShapingConeSoftnessAttr(), time, 0.0) > 0.0
      or _get_or(shaping_api.GetShapingFocusAttr(), time, 0.0) > 0.0
  )
  if emitter_plugin_type:
    flip_z = emitter_plugin_type in (
        'spot',
        'projector',
        'directional',
        'ies',
    ) or (has_shaping and emitter_plugin_type == 'point')
  else:
    flip_z = prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.SphereLight)

  # Non-area emitters do not support scale or shear, and AnimatedTransform
  # rejects keyframe matrices with shear.
  def to_emitter_transform(m: Gf.Matrix4d) -> mi.ScalarTransform4f:
    t = util.remove_scale_from_transform(util.to_mitsuba_transform(m))
    return t.rotate([1, 0, 0], 180) if flip_z else t

  return motion.sample_world_transform(
      prim, time, motion_interval, to_emitter_transform
  )


def convert_light(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    motion_interval: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
  """Handles a light prim and returns the Mitsuba emitter dictionary.

  Args:
    prim: The USD light prim.
    time: The time code to evaluate at.
    motion_interval: The (open, close) shutter offsets to sample the light
      transform over. Motion is only supported for lights that are not shapes.

  Returns:
    The Mitsuba emitter dictionary.
  """
  emitter_overrides = util.extract_nested_dict(prim, 'mitsuba:emitter:', time)
  shape_overrides = util.extract_nested_dict(prim, 'mitsuba:shape:', time)
  world_transform = _sample_light_transform(
      prim, time, motion_interval, emitter_overrides.get('type', '')
  )
  color, intensity = _base_emission(prim, time)

  emitter_dict: dict[str, Any] | None = None
  for light_class, converter in _LIGHT_CONVERTERS:
    if prim.IsA(light_class):
      if light_class is UsdLux.DomeLight:
        emitter_dict = _convert_dome_light(
            prim, world_transform, color, intensity, time, emitter_overrides
        )
      else:
        emitter_dict = converter(prim, world_transform, color, intensity, time)
      break

  if emitter_dict is None:
    raise ValueError(f'Unsupported light prim: {prim.GetPath()}.')

  vis = prim.GetAttribute('mitsuba:visibility').Get(time)
  has_explicit_frame = _has_explicit_emitter_frame(emitter_overrides)
  if _is_area_light(prim, time):
    emitter_dict['bsdf'] = {'type': 'null'}
    emitter_dict['visibility'] = vis or 'secondary'
    if emitter_overrides:
      target = emitter_dict['emitter']
      custom_type = emitter_overrides.get('type')
      if (
          target.get('type') == 'ies'
          and 'nested_emitter' in target
          and custom_type != 'ies'
      ):
        nested = target['nested_emitter']
        if custom_type and custom_type != nested.get('type'):
          nested = {'type': custom_type}
        util.deep_merge_dict(nested, emitter_overrides)
        target['nested_emitter'] = nested
      else:
        if custom_type and custom_type != target.get('type'):
          target = {'type': custom_type}
        elif target.get('type') == 'ies' and has_explicit_frame:
          target.pop('to_world', None)
        util.deep_merge_dict(target, emitter_overrides)
        emitter_dict['emitter'] = target
    if shape_overrides:
      util.deep_merge_dict(emitter_dict, shape_overrides)
  else:
    if prim.IsA(UsdLux.DomeLight) and vis:
      emitter_dict['visibility'] = vis
    if emitter_overrides:
      custom_type = emitter_overrides.get('type')
      if (
          emitter_dict.get('type') == 'ies'
          and 'nested_emitter' in emitter_dict
          and custom_type != 'ies'
      ):
        nested = emitter_dict['nested_emitter']
        if custom_type and custom_type != nested.get('type'):
          nested = {'type': custom_type}
          if (
              not has_explicit_frame
              and 'to_world' in emitter_dict['nested_emitter']
          ):
            nested['to_world'] = emitter_dict['nested_emitter']['to_world']
        elif has_explicit_frame:
          nested.pop('to_world', None)
        util.deep_merge_dict(nested, emitter_overrides)
        emitter_dict['nested_emitter'] = nested
      else:
        if custom_type and custom_type != emitter_dict.get('type'):
          base: dict[str, Any] = {'type': custom_type}
          if (
              custom_type != 'constant'
              and not has_explicit_frame
              and 'to_world' in emitter_dict
          ):
            base['to_world'] = emitter_dict['to_world']
          if (
              custom_type in ('envmap', 'constant')
              and 'visibility' in emitter_dict
          ):
            base['visibility'] = emitter_dict['visibility']
          emitter_dict = base
        elif has_explicit_frame:
          emitter_dict.pop('to_world', None)
        util.deep_merge_dict(emitter_dict, emitter_overrides)

  return emitter_dict
