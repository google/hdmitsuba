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

"""Handles Points translation for Mitsuba rendering."""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from pxr import Tf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdShade
from usd_mitsuba import material
from usd_mitsuba import util


def convert_points(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    to_world: mi.ScalarTransform4f | None = None,
) -> dict[str, Any] | None:
  """Converts a points prim to a Mitsuba `ellipsoids` shape.

  Like in hdMitsuba, the shape holds a sphere per point.

  Args:
    prim: The points prim.
    time: The time code to evaluate the prim at.
    to_world: The similarity transform to bake into the shape, which has no
      `to_world` parameter. Defaults to the identity.

  Returns:
    The shape dictionary, or None if `prim` has no points.
  """
  points = UsdGeom.Points(prim)
  positions = np.array(
      points.GetPointsAttr().Get(time) or [], dtype=np.float32
  ).reshape(-1, 3)
  count = len(positions)
  if count == 0:
    return None
  # Widths are diameters, either per point or constant, and default to 1. The
  # `widths` primvar takes precedence over the attribute.
  widths_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar('widths')
  widths = np.array(
      (widths_primvar.Get(time) if widths_primvar else None)
      or points.GetWidthsAttr().Get(time)
      or [1.0],
      dtype=np.float32,
  )
  matrix = (
      np.eye(4, dtype=np.float32)
      if to_world is None
      else np.array(to_world.matrix, dtype=np.float32)
  )
  # Each row holds the center, the three radii and the rotation quaternion
  # (with the real part last) of an ellipsoid.
  data = np.zeros((count, 10), dtype=np.float32)
  data[:, :3] = positions @ matrix[:3, :3].T + matrix[:3, 3]
  data[:, 3:6] = (
      0.5
      * np.linalg.norm(matrix[:3, 0])
      * (widths if len(widths) == count else widths[:1])
  )[:, None]
  data[:, 9] = 1.0
  shape: dict[str, Any] = {
      'type': 'ellipsoids',
      'extent': 1.0,
      'visibility': util.get_ray_visibility(prim, time),
      'data': mi.TensorXf(data),
  }

  bsdf, _, displacement, _ = material.convert_material(prim, time=time)
  if displacement is not None:
    Tf.Warn(f'Displacement on {prim.GetPath()} is not supported.')
  # Like the constant displayColor, per-point colors are a fallback for
  # geometry without a material.
  display_color = points.GetDisplayColorPrimvar()
  colors = display_color.Get(time)
  if (
      UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
          UsdShade.Tokens.full
      )[0]
      or display_color.GetInterpolation()
      not in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying)
      or len(colors) != count
  ):
    if bsdf is not None:
      shape['bsdf'] = bsdf
  elif count == 1:
    # The `lut` texture below requires at least two entries.
    shape['bsdf'] = {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': tuple(colors[0])},
    }
  else:
    # The `ellipsoids` shape does not expose per-ellipsoid colors to
    # `mesh_attribute` textures, so look them up by the point index instead.
    shape['vertex_point_index'] = mi.TensorXf(
        np.arange(count, dtype=np.float32).reshape(-1, 1)
    )
    shape['bsdf'] = {
        'type': 'diffuse',
        'reflectance': {
            'type': 'lut',
            'input': {'type': 'mesh_attribute', 'name': 'vertex_point_index'},
            'filter_type': 'nearest',
            'input_max': count - 1.0,
            'data': mi.TensorXf(np.array(colors, dtype=np.float32)),
        },
    }
  return shape
