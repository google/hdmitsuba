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
from pxr import Usd
from pxr import UsdLux

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
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  dome_light = UsdLux.DomeLight(prim)
  texture_file_attr = dome_light.CreateTextureFileAttr().Get(time)
  if texture_file_attr:
    return {
        'type': 'envmap',
        'filename': texture_file_attr.resolvedPath,
        'to_world': world_transform,
        'scale': intensity,
    }
  else:
    return {
        'type': 'constant',
        'radiance': {'type': 'rgb', 'value': color * intensity},
    }


def _convert_sphere_light(
    prim: Usd.Prim,
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  color = color * intensity
  sphere_light = UsdLux.SphereLight(prim)
  normalize = sphere_light.GetNormalizeAttr().Get(time)
  radius = sphere_light.GetRadiusAttr().Get(time)
  if sphere_light.GetTreatAsPointAttr().Get(time) or radius == 0:
    if normalize:
      color *= 0.25
    emitter_dict = {
        'type': 'point',
        'to_world': world_transform,
        'intensity': {'type': 'rgb', 'value': color},
    }
    if prim.HasAPI(UsdLux.ShapingAPI):
      shaping_api = UsdLux.ShapingAPI(prim)
      shaping_cone_angle = shaping_api.GetShapingConeAngleAttr().Get(time)
      shaping_softness = shaping_api.GetShapingConeSoftnessAttr().Get(time)
      if shaping_cone_angle != 0:
        emitter_dict = {
            'type': 'spot',
            'to_world': world_transform.rotate([1, 0, 0], 180),
            'intensity': {'type': 'rgb', 'value': color},
            'beam_width': shaping_cone_angle * (1.0 - shaping_softness),
            'cutoff_angle': shaping_cone_angle,
        }
    return emitter_dict
  else:
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
    world_transform: mi.ScalarTransform4f,
    color: mi.ScalarColor3f,
    intensity: float,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  return {
      'type': 'directional',
      'to_world': world_transform.rotate([1, 0, 0], 180),
      'irradiance': {'type': 'rgb', 'value': color * intensity},
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


def convert_light(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> dict[str, Any]:
  """Handles a light prim and returns the Mitsuba emitter dictionary.

  Args:
    prim: The USD light prim.
    time: The time code to evaluate at.

  Returns:
    The Mitsuba emitter dictionary.
  """
  _check_ies_profile(prim, time)
  world_transform = util.get_world_transform(prim, time)

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

  emitter_dict = None
  for light_class, converter in _LIGHT_CONVERTERS:
    if prim.IsA(light_class):
      emitter_dict = converter(prim, world_transform, color, intensity, time)
      break

  if emitter_dict is None:
    raise ValueError(f'Unsupported light prim: {prim.GetPath()}.')

  return emitter_dict
