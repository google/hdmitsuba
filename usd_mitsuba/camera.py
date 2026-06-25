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
import numpy as np
from pxr import Usd
from pxr import UsdGeom

from usd_mitsuba import util
from usd_mitsuba.render_settings import get_render_settings


def _get_camera_transform(camera: UsdGeom.Camera, time: Usd.TimeCode) -> mi.ScalarTransform4f:
  """Returns the Mitsuba sensor transform for a given USD camera and time code."""
  world_transform = camera.ComputeLocalToWorldTransform(time)

  # Mitsuba sensors generally do not support scaling.
  _, rotation, translation = dr.transform_decompose(
      mi.ScalarMatrix4f(np.array(world_transform).T)
  )
  rotation = rotation * dr.rotate(
      mi.ScalarQuaternion4f, mi.ScalarVector3f(0, 1, 0), dr.deg2rad(180)
  )
  return mi.ScalarTransform4f(
      dr.transform_compose(mi.ScalarMatrix3f(1.0), rotation, translation)
  )


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

  world_transform = _get_camera_transform(camera, time)
  aperture_x = camera.GetHorizontalApertureAttr().Get(time)
  aperture_y = camera.GetVerticalApertureAttr().Get(time)
  render_settings = get_render_settings(camera.GetPrim().GetStage())
  if image_size is None and render_settings:
    resolution_attr = render_settings.GetResolutionAttr()
    if resolution_attr.HasAuthoredValue():
      resolution = resolution_attr.Get(time)
      image_size = (resolution[0], resolution[1])
  if image_size is None:
    image_size = (512, int(dr.round(512 * aperture_y / aperture_x)))

  prim = camera.GetPrim()

  if prim.GetAttribute('mitsuba:sensor:type').Get():
    sensor_dict = util.extract_nested_dict(prim, 'mitsuba:sensor:')
    # Irradiancemeter sensors cannot have a transform.
    if sensor_dict.get('type') != 'irradiancemeter':
      sensor_dict['to_world'] = world_transform
    film_dict = sensor_dict.setdefault('film', {})
    assert isinstance(film_dict, dict)
    film_dict.setdefault('type', 'hdrfilm')
    film_dict.setdefault('width', image_size[0])
    film_dict.setdefault('height', image_size[1])
  else:
    sensor_dict = {
        'type': 'perspective',
        'to_world': world_transform,
        'film': {
            'type': 'hdrfilm',
            'width': image_size[0],
            'height': image_size[1],
        },
    }
    focal_length = camera.GetFocalLengthAttr().Get(time)
    x_fov = 2.0 * dr.atan(aperture_x / (2.0 * focal_length))
    sensor_dict['fov'] = dr.rad2deg(x_fov)

    sensor_dict['principal_point_offset_x'] = (
        camera.GetHorizontalApertureOffsetAttr().Get(time) / aperture_x
    )
    sensor_dict['principal_point_offset_y'] = (
        -camera.GetVerticalApertureOffsetAttr().Get(time) / aperture_y
    )

    clipping_range = camera.GetClippingRangeAttr().Get(time)
    if clipping_range:
      sensor_dict['near_clip'] = clipping_range[0]
      sensor_dict['far_clip'] = clipping_range[1]

  # TODO: Modify this to also support specifying 'mitsuba:sampler:sample_count'.
  sample_count = 1
  if render_settings:
    sample_count_attr = render_settings.GetPrim().GetAttribute('mitsuba:sample_count')
    if sample_count_attr.HasAuthoredValue():
      sample_count = sample_count_attr.Get(time)

  sensor_dict['sampler'] = {
      'type': 'independent',
      'sample_count': sample_count,
  }

  if reconstruction_filter is not None:
    film_dict = sensor_dict.setdefault('film', {})
    assert isinstance(film_dict, dict)
    film_dict['reconstruction_filter'] = reconstruction_filter
  return sensor_dict
