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

"""Converts between USD and Mitsuba cameras."""

from __future__ import annotations

from typing import Any

import drjit as dr
import mitsuba as mi
from pxr import Gf
from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom

from usd_mitsuba import motion
from usd_mitsuba import util
from usd_mitsuba.render_settings import get_render_settings


def _to_sensor_transform(world_transform: Gf.Matrix4d) -> mi.ScalarTransform4f:
  """Returns the Mitsuba sensor transform for a USD camera world transform."""
  # Mitsuba sensors generally do not support scaling.
  return util.remove_scale_from_transform(
      util.to_mitsuba_transform(world_transform)
  ).rotate([0, 1, 0], 180)


SENSOR_SHAPE_ATTR = 'mitsuba:sensor:shape'
_DEFAULT_SAMPLE_COUNT = 128
_FILM_CROP_KEYS = (
    'crop_offset_x',
    'crop_offset_y',
    'crop_width',
    'crop_height',
)


def get_target_shape_path(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> Sdf.Path | None:
  """Returns the shape path a sensor prim measures, if any."""
  attribute = prim.GetAttribute(SENSOR_SHAPE_ATTR)
  if not attribute:
    return None
  path_string = attribute.Get(time)
  if isinstance(path_string, Sdf.AssetPath):
    path_string = path_string.resolvedPath or path_string.path
  if not isinstance(path_string, str) or not path_string:
    return None
  if not Sdf.Path.IsValidPathString(path_string):
    Tf.Warn(
        f'Camera {prim.GetPath()}: mitsuba:sensor:shape is not a valid path:'
        f' {path_string}'
    )
    return None
  return Sdf.Path(path_string)


def get_surface_sensor_bindings(
    stage: Usd.Stage, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> dict[Sdf.Path, Usd.Prim]:
  """Maps each measured shape path to the camera prim that measures it."""
  bindings: dict[Sdf.Path, Usd.Prim] = {}
  for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Camera):
      continue
    shape_path = get_target_shape_path(prim, time)
    if shape_path is None:
      continue
    if shape_path in bindings:
      Tf.Warn(
          f'Shape {shape_path} is already measured by another sensor; ignoring'
          f' sensor {prim.GetPath()}.'
      )
    else:
      bindings[shape_path] = prim
  return bindings


def _clamped_film_crop(
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
    crop_width: int | None = None,
    crop_height: int | None = None,
) -> dict[str, int]:
  """Returns film crop properties clamped to `[0, width) x [0, height)`."""
  offset_x = max(0, min(offset_x, width - 1))
  offset_y = max(0, min(offset_y, height - 1))
  resolved_width = width - offset_x if crop_width is None else crop_width
  resolved_height = height - offset_y if crop_height is None else crop_height
  return {
      'crop_offset_x': offset_x,
      'crop_offset_y': offset_y,
      'crop_width': max(1, min(resolved_width, width - offset_x)),
      'crop_height': max(1, min(resolved_height, height - offset_y)),
  }


def usd_to_mitsuba(
    camera: UsdGeom.Camera,
    image_size: tuple[int, int] | None = None,
    reconstruction_filter: (
        mi.ReconstructionFilter | str | dict[str, Any] | None
    ) = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> dict[str, Any]:
  """Converts a USD camera to a Mitsuba sensor.

  Args:
    camera: The USD camera to convert.
    image_size: The desired image size of the Mitsuba sensor as (width, height).
    reconstruction_filter: The reconstruction filter to use for the Mitsuba
      sensor (either as plugin name, dictionary, or instantiated
      ReconstructionFilter object).
    time: The time code to evaluate the camera at.

  Returns:
    A dictionary representing the Mitsuba sensor.
  """
  prim = camera.GetPrim()
  shutter = motion.get_shutter_interval(camera, time)
  aperture_x = camera.GetHorizontalApertureAttr().Get(time)
  aperture_y = camera.GetVerticalApertureAttr().Get(time)
  focal_length = camera.GetFocalLengthAttr().Get(time)
  f_stop = camera.GetFStopAttr().Get(time)
  focus_distance = camera.GetFocusDistanceAttr().Get(time)
  has_usd_dof = bool(
      f_stop
      and f_stop > 0.0
      and focus_distance
      and focus_distance > 0.0
      and focal_length > 0.0
  )
  sensor_overrides = util.extract_nested_dict(prim, 'mitsuba:sensor:', time)
  shape_path = sensor_overrides.pop('shape', None)
  sensor_type = sensor_overrides.get('type') or (
      'thinlens' if has_usd_dof else 'perspective'
  )
  is_signal_sensor = sensor_type in ('radiancemeter', 'distant')
  render_settings = get_render_settings(prim.GetStage())
  if is_signal_sensor:
    image_size = (1, 1)
  elif image_size is None and render_settings:
    resolution_attr = render_settings.GetResolutionAttr()
    if resolution_attr.HasAuthoredValue():
      resolution = resolution_attr.Get(time)
      image_size = (resolution[0], resolution[1])
  if image_size is None:
    image_size = (512, int(dr.round(512 * aperture_y / aperture_x)))

  sensor_dict: dict[str, Any] = {'type': sensor_type}
  has_explicit_ray_frame = any(
      key in sensor_overrides for key in ('to_world', 'direction', 'origin')
  )
  is_surface_sensor = isinstance(
      shape_path, str
  ) and Sdf.Path.IsValidPathString(shape_path)
  if (
      sensor_type != 'irradiancemeter'
      and not is_surface_sensor
      and not has_explicit_ray_frame
  ):
    sensor_dict['to_world'] = motion.sample_world_transform(
        prim, time, shutter, _to_sensor_transform
    )

  if sensor_type in ('perspective', 'thinlens'):
    if 'focal_length' not in sensor_overrides:
      sensor_dict['fov'] = dr.rad2deg(
          2.0 * dr.atan(aperture_x / (2.0 * focal_length))
      )
      sensor_dict['fov_axis'] = 'x'
    sensor_dict['principal_point_offset_x'] = (
        camera.GetHorizontalApertureOffsetAttr().Get(time) / aperture_x
    )
    sensor_dict['principal_point_offset_y'] = (
        -camera.GetVerticalApertureOffsetAttr().Get(time) / aperture_y
    )
    if sensor_type == 'thinlens':
      if has_usd_dof:
        sensor_dict['aperture_radius'] = (
            0.5 * (focal_length * Gf.Camera.FOCAL_LENGTH_UNIT) / f_stop
        )
        sensor_dict['focus_distance'] = focus_distance
      else:
        sensor_dict['aperture_radius'] = 0.0

  if sensor_type in ('perspective', 'thinlens', 'orthographic'):
    clipping_range = camera.GetClippingRangeAttr().Get(time)
    if clipping_range:
      sensor_dict['near_clip'] = clipping_range[0]
      sensor_dict['far_clip'] = clipping_range[1]

  sensor_dict['film'] = {
      'type': 'hdrfilm',
      'width': image_size[0],
      'height': image_size[1],
  }
  sensor_dict['shutter_open'], sensor_dict['shutter_close'] = shutter
  util.deep_merge_dict(sensor_dict, sensor_overrides)
  film_dict = sensor_dict['film']
  assert isinstance(film_dict, dict)
  if is_signal_sensor:
    film_dict['width'] = 1
    film_dict['height'] = 1
    for crop_key in _FILM_CROP_KEYS:
      film_dict.pop(crop_key, None)
  else:
    film_dict.setdefault('width', image_size[0])
    film_dict.setdefault('height', image_size[1])
    width, height = int(film_dict['width']), int(film_dict['height'])
    if width > 0 and height > 0:
      ndc_attribute = (
          render_settings.GetDataWindowNDCAttr() if render_settings else None
      )
      if (
          ndc_attribute
          and ndc_attribute.HasAuthoredValue()
          and (data_window_ndc := tuple(ndc_attribute.Get(time)))
          != (0.0, 0.0, 1.0, 1.0)
      ):
        xmin, ymin, xmax, ymax = data_window_ndc
        offset_x = int(round(xmin * width))
        offset_y = int(round((1.0 - ymax) * height))
        film_dict.update(
            _clamped_film_crop(
                width,
                height,
                offset_x,
                offset_y,
                int(round(xmax * width)) - offset_x,
                int(round((1.0 - ymin) * height)) - offset_y,
            )
        )
      elif any(key in film_dict for key in _FILM_CROP_KEYS):
        film_dict.update(
            _clamped_film_crop(
                width,
                height,
                int(film_dict.get('crop_offset_x', 0)),
                int(film_dict.get('crop_offset_y', 0)),
                int(film_dict['crop_width'])
                if 'crop_width' in film_dict
                else None,
                int(film_dict['crop_height'])
                if 'crop_height' in film_dict
                else None,
            )
        )

  if 'sampler' not in sensor_dict:
    if render_settings:
      sampler_dict = util.extract_nested_dict(
          render_settings.GetPrim(), 'mitsuba:sampler:', time
      )
    else:
      sampler_dict = {}
    sampler_dict.setdefault('type', 'independent')
    sampler_dict.setdefault('sample_count', _DEFAULT_SAMPLE_COUNT)
    sensor_dict['sampler'] = sampler_dict
  elif isinstance(sensor_dict['sampler'], dict):
    sensor_dict['sampler'].setdefault('type', 'independent')

  if reconstruction_filter is not None:
    plugin_manager = mi.PluginManager.instance()
    for key, value in list(film_dict.items()):
      if (
          isinstance(value, dict)
          and isinstance(value.get('type'), str)
          and plugin_manager.plugin_type(value['type'])
          == mi.ObjectType.ReconstructionFilter
      ):
        del film_dict[key]
    film_dict['pixel_filter'] = (
        {'type': reconstruction_filter}
        if isinstance(reconstruction_filter, str)
        else reconstruction_filter
    )
  return sensor_dict
