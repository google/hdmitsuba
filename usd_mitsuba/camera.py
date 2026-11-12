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


def get_target_shape_path(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> Sdf.Path | None:
  """Returns the shape path a sensor prim measures, if any."""
  attr = prim.GetAttribute(SENSOR_SHAPE_ATTR)
  if not attr:
    return None
  value = attr.Get(time)
  if not value:
    return None
  if isinstance(value, Sdf.AssetPath):
    path_str = value.resolvedPath or value.path
  else:
    path_str = str(value)
  if not path_str or not Sdf.Path.IsValidPathString(path_str):
    return None
  path = Sdf.Path(path_str)
  return path if not path.isEmpty else None


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
    bindings.setdefault(shape_path, prim)
  return bindings


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
  world_transform = motion.sample_world_transform(
      prim, time, shutter, _to_sensor_transform
  )
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
  sensor_overrides.pop('shape', None)
  sensor_type = sensor_overrides.get('type') or (
      'thinlens' if has_usd_dof else 'perspective'
  )
  is_signal_sensor = sensor_type in ('radiancemeter', 'distant')
  render_settings = get_render_settings(prim.GetStage())
  if image_size is None:
    if is_signal_sensor:
      image_size = (1, 1)
    elif render_settings:
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
  is_surface_sensor = get_target_shape_path(prim, time) is not None
  if (
      sensor_type != 'irradiancemeter'
      and not is_surface_sensor
      and not has_explicit_ray_frame
  ):
    sensor_dict['to_world'] = world_transform

  if sensor_type in ('perspective', 'thinlens'):
    if 'focal_length' not in sensor_overrides:
      x_fov = 2.0 * dr.atan(aperture_x / (2.0 * focal_length))
      sensor_dict['fov'] = dr.rad2deg(x_fov)
      sensor_dict['fov_axis'] = 'x'
    sensor_dict['principal_point_offset_x'] = (
        camera.GetHorizontalApertureOffsetAttr().Get(time) / aperture_x
    )
    sensor_dict['principal_point_offset_y'] = (
        -camera.GetVerticalApertureOffsetAttr().Get(time) / aperture_y
    )
    if sensor_type == 'thinlens':
      sensor_dict['aperture_radius'] = (
          float(0.5 * (focal_length * Gf.Camera.FOCAL_LENGTH_UNIT) / f_stop)
          if has_usd_dof
          else 0.0
      )
      if has_usd_dof:
        sensor_dict['focus_distance'] = float(focus_distance)

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
  film_dict.setdefault('width', image_size[0])
  film_dict.setdefault('height', image_size[1])
  if is_signal_sensor:
    film_dict.pop('crop_offset_x', None)
    film_dict.pop('crop_offset_y', None)
    film_dict.pop('crop_width', None)
    film_dict.pop('crop_height', None)

  has_ndc_crop = False
  # Apply crop window if present in render settings
  if render_settings and not is_signal_sensor:
    ndc_attr = render_settings.GetDataWindowNDCAttr()
    if ndc_attr.HasAuthoredValue():
      ndc = ndc_attr.Get(time)
      ndc = tuple(ndc)
      if ndc != (0.0, 0.0, 1.0, 1.0):
        has_ndc_crop = True
        width, height = image_size
        xmin, ymin, xmax, ymax = ndc

        # Calculate pixel-space crop
        crop_offset_x = int(round(xmin * width))
        crop_offset_y = int(round((1.0 - ymax) * height))
        crop_width = int(round(xmax * width)) - crop_offset_x
        crop_height = int(round((1.0 - ymin) * height)) - crop_offset_y

        # Ensure values are within bounds
        crop_offset_x = max(0, min(crop_offset_x, width - 1))
        crop_offset_y = max(0, min(crop_offset_y, height - 1))
        crop_width = max(1, min(crop_width, width - crop_offset_x))
        crop_height = max(1, min(crop_height, height - crop_offset_y))

        film_dict['crop_width'] = crop_width
        film_dict['crop_height'] = crop_height
        film_dict['crop_offset_x'] = crop_offset_x
        film_dict['crop_offset_y'] = crop_offset_y
  if not has_ndc_crop and any(
      key in film_dict
      for key in ('crop_offset_x', 'crop_offset_y', 'crop_width', 'crop_height')
  ):
    width, height = int(film_dict['width']), int(film_dict['height'])
    if width > 0 and height > 0:
      crop_offset_x = max(
          0, min(int(film_dict.get('crop_offset_x', 0)), width - 1)
      )
      crop_offset_y = max(
          0, min(int(film_dict.get('crop_offset_y', 0)), height - 1)
      )
      crop_width = max(
          1,
          min(
              int(film_dict.get('crop_width', width - crop_offset_x)),
              width - crop_offset_x,
          ),
      )
      crop_height = max(
          1,
          min(
              int(film_dict.get('crop_height', height - crop_offset_y)),
              height - crop_offset_y,
          ),
      )
      film_dict['crop_offset_x'] = crop_offset_x
      film_dict['crop_offset_y'] = crop_offset_y
      film_dict['crop_width'] = crop_width
      film_dict['crop_height'] = crop_height

  sampler_dict: dict[str, Any] = {}
  if render_settings:
    settings_prim = render_settings.GetPrim()
    sampler_dict = util.extract_nested_dict(
        settings_prim, 'mitsuba:sampler:', time
    )
    sample_count_attr = settings_prim.GetAttribute('mitsuba:sample_count')
    if sample_count_attr and sample_count_attr.HasAuthoredValue():
      sampler_dict.setdefault('sample_count', sample_count_attr.Get(time))
  if not sampler_dict.get('type'):
    sampler_dict['type'] = 'independent'
  sampler_dict.setdefault('sample_count', _DEFAULT_SAMPLE_COUNT)
  if isinstance(sensor_dict.get('sampler'), dict):
    if not sensor_dict['sampler'].get('type'):
      sensor_dict['sampler']['type'] = 'independent'
  else:
    sensor_dict.setdefault('sampler', sampler_dict)

  if reconstruction_filter is not None:
    film_dict.pop('rfilter', None)
    film_dict.pop('reconstruction_filter', None)
    film_dict['pixel_filter'] = (
        {'type': reconstruction_filter}
        if isinstance(reconstruction_filter, str)
        else reconstruction_filter
    )
  return sensor_dict
