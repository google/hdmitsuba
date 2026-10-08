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

from typing import Any

import drjit as dr
import mitsuba as mi
from pxr import Gf
from pxr import Tf
from pxr import Usd
from pxr import UsdLux

from usd_mitsuba import motion
from usd_mitsuba import util


def _check_ies_profile(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
  """Checks if the prim has an IES profile and raises a ValueError if so."""
  if prim.HasAPI(UsdLux.ShapingAPI):
    shaping_api = UsdLux.ShapingAPI(prim)
    ies_file = shaping_api.GetShapingIesFileAttr().Get(time)
    if ies_file:
      raise ValueError('IES files are not yet supported.')


def _convert_dome_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f | mi.AnimatedTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  dome_light = UsdLux.DomeLight(prim)
  texture_file_attr = dome_light.GetTextureFileAttr().Get(time)
  if texture_file_attr:
    filename = texture_file_attr.resolvedPath
    return {
        'type': 'envmap',
        'bitmap': mi.Bitmap(filename),
        'to_world': world_transform,
        'scale': intensity,
    }
  else:
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
  if prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight):
    return True
  return prim.IsA(UsdLux.SphereLight) and not _is_point_light(
      UsdLux.SphereLight(prim), time
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
    emitter_dict: dict[str, Any] = {
        'type': 'point',
        'to_world': world_transform,
        'intensity': {'type': 'rgb', 'value': color},
    }
    if prim.HasAPI(UsdLux.ShapingAPI):
      shaping = UsdLux.ShapingAPI(prim)
      if cone_angle := shaping.GetShapingConeAngleAttr().Get(time):
        softness = shaping.GetShapingConeSoftnessAttr().Get(time)
        emitter_dict |= {
            'type': 'spot',
            'beam_width': cone_angle * (1.0 - softness),
            'cutoff_angle': cone_angle,
        }
    return emitter_dict

  world_transform = world_transform.scale(radius)
  if normalize:
    radius2 = dr.squared_norm(world_transform @ mi.ScalarVector3f(1, 0, 0))
    color /= 4 * dr.pi * radius2
  return {
      'type': 'sphere',
      'to_world': world_transform,
      'emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': color},
      },
  }


def _convert_distant_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f | mi.AnimatedTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  distant_light = UsdLux.DistantLight(prim)
  angle = distant_light.GetAngleAttr().Get(time)
  return {
      'type': 'directional',
      'to_world': world_transform,
      'irradiance': {'type': 'rgb', 'value': color * intensity},
      'angle': float(angle) if angle is not None else 0.0,
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
      'emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': color * intensity},
      },
  }


def _convert_disk_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  disk_light = UsdLux.DiskLight(prim)
  world_transform = world_transform.rotate([1, 0, 0], 180).scale(
      disk_light.GetRadiusAttr().Get(time)
  )
  if disk_light.GetNormalizeAttr().Get(time):
    area = (
        dr.squared_norm(world_transform @ mi.ScalarVector3f(1, 0, 0)) * dr.pi
    )
    color /= area

  return {
      'type': 'disk',
      'to_world': world_transform,
      'emitter': {
          'type': 'area',
          'radiance': {'type': 'rgb', 'value': color * intensity},
      },
  }


_LIGHT_CONVERTERS = [
    (UsdLux.DomeLight, _convert_dome_light),
    (UsdLux.SphereLight, _convert_sphere_light),
    (UsdLux.DistantLight, _convert_distant_light),
    (UsdLux.RectLight, _convert_rect_light),
    (UsdLux.DiskLight, _convert_disk_light),
]


def _sample_light_transform(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    motion_interval: tuple[float, float],
) -> mi.ScalarTransform4f | mi.AnimatedTransform4f:
  """Samples the light's world transform, falling back to static for area lights."""
  if _is_area_light(prim, time):
    transform = motion.sample_world_transform(prim, time, motion_interval)
    if isinstance(transform, mi.AnimatedTransform4f):
      Tf.Warn(f'Motion blur is not supported for light {prim.GetPath()}.')
      return util.get_world_transform(prim, time)
    return transform

  # USD distant and spot lights emit along -Z, while Mitsuba's directional and
  # spot emitters point along +Z.
  flip_z = prim.IsA(UsdLux.DistantLight) or (
      prim.HasAPI(UsdLux.ShapingAPI)
      and UsdLux.ShapingAPI(prim).GetShapingConeAngleAttr().Get(time) != 0
  )

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
  _check_ies_profile(prim, time)
  world_transform = _sample_light_transform(prim, time, motion_interval)

  usd_light = (
      UsdLux.BoundableLightBase(prim)
      if prim.IsA(UsdLux.BoundableLightBase)
      else UsdLux.NonboundableLightBase(prim)
  )
  color = usd_light.GetColorAttr().Get(time)
  intensity = usd_light.GetIntensityAttr().Get(time)
  exposure = usd_light.GetExposureAttr().Get(time)
  intensity = intensity * (2.0 ** exposure)
  color = mi.ScalarColor3f(color[0], color[1], color[2])

  for light_class, converter in _LIGHT_CONVERTERS:
    if prim.IsA(light_class):
      return converter(prim, world_transform, color, intensity, time)
  raise ValueError(f'Unsupported light prim: {prim.GetPath()}.')
