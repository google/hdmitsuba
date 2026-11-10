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

"""Shared utility functions for Mitsuba rendering."""

from __future__ import annotations

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom


def get_mitsuba_id(prim: Usd.Prim) -> str:
  """Converts a USD prim's path to a Mitsuba scene dictionary key.

  Args:
    prim: The USD prim.

  Returns:
    A valid Mitsuba scene object ID.
  """
  return str(prim.GetPath()).replace('/', '_').replace('.', '_')


def extract_nested_dict(prim: Usd.Prim, prefix: str) -> dict[str, Any]:
  """Parses all attributes starting with the given prefix into a dictionary.

  For example, if the prefix is 'mitsuba:integrator:', then all attributes
  starting with that prefix will be parsed into a nested dictionary with the
  structure defined by the attribute names. Concretely, the following USD
  attributes:

    mitsuba:integrator:type = "path"
    mitsuba:integrator:max_depth = 10
    mitsuba:integrator:rr_depth = 0

  result in a Python dictionary of the form:

    {
      'type': 'path',
      'max_depth': 10,
      'rr_depth': 0,
    }


  Args:
    prim: The USD prim to parse.
    prefix: The prefix to match the attribute names against.

  Returns:
    A dictionary of attributes matching the prefix.
  """
  result = {}
  for attr in prim.GetAttributes():
    name = attr.GetName()
    if name.startswith(prefix):
      key_parts = name[len(prefix) :].split(':')
      current_level = result
      for part in key_parts[:-1]:
        current_level = current_level.setdefault(part, {})
        if not isinstance(current_level, dict):
          break
      else:
        current_level[key_parts[-1]] = attr.Get()
  return result


def parse_ray_visibility(value: str | None, fallback: str | None) -> str | None:
  """Validates the value of a `mitsuba:visibility` attribute or primvar.

  Mirrors `ParseMitsubaVisibility` in hdmitsuba/utils.h.

  Args:
    value: The authored value, if any.
    fallback: The value to return if `value` is not authored or invalid.

  Returns:
    The Mitsuba ray visibility ("all", "primary", "secondary" or "hidden"), or
    `fallback`.
  """
  if not value:
    return fallback
  if value not in ('all', 'primary', 'secondary', 'hidden'):
    Tf.Warn(
        f"Ignoring invalid mitsuba:visibility '{value}', expected"
        ' "all", "primary", "secondary" or "hidden".'
    )
    return fallback
  return value


def get_ray_visibility(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> str:
  """Returns the Mitsuba ray visibility of a shape.

  Args:
    prim: The USD shape prim.
    time: The time code to evaluate the primvar at.

  Returns:
    The value of the (inherited) `primvars:mitsuba:visibility` primvar ("all",
    "primary", "secondary" or "hidden"), or "all" if it is not authored or
    invalid.
  """
  primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(
      'mitsuba:visibility'
  )
  return parse_ray_visibility(primvar.Get(time) if primvar else None, 'all')


def get_world_transform(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> mi.ScalarTransform4f:
  """Computes the Mitsuba world transform from a USD prim.

  Args:
    prim: The USD prim to compute the world transform for.
    time: The time code to evaluate the transform at.

  Returns:
    The Mitsuba world transform as a `ScalarTransform4f`.
  """
  return to_mitsuba_transform(
      UsdGeom.Imageable(prim).ComputeLocalToWorldTransform(time)
  )


def to_mitsuba_transform(matrix: Gf.Matrix4d) -> mi.ScalarTransform4f:
  """Converts a USD (row-vector) matrix to a Mitsuba transform.

  Args:
    matrix: The USD matrix.

  Returns:
    The Mitsuba transform.
  """
  return mi.ScalarTransform4f(np.array(matrix, dtype=np.float32).T)


def remove_scale_from_transform(
    transform: mi.ScalarTransform4f,
) -> mi.ScalarTransform4f:
  """Strips scale and shear from `transform`, keeping rotation and translation.

  Args:
    transform: The Mitsuba transform.

  Returns:
    The transform with unit scale and no shear.
  """
  if not transform.has_scale():
    return transform
  _, rotation, translation = dr.transform_decompose(transform.matrix)
  return mi.ScalarTransform4f(
      dr.transform_compose(mi.ScalarMatrix3f(1.0), rotation, translation)
  )


def get_implicit_surface_parts(
    prim: Usd.Prim, time: Usd.TimeCode
) -> dict[str, dict[str, Any]]:
  """Returns the native Mitsuba shapes of an implicit surface in object space.

  Args:
    prim: The cube, plane, sphere or cylinder prim.
    time: The time code to evaluate the prim at.

  Returns:
    The shape dictionaries by ID suffix, empty if `prim` is none of the above.
  """
  if prim.IsA(UsdGeom.Cube):
    size = UsdGeom.Cube(prim).GetSizeAttr().Get(time)
    return {
        '': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f().scale(0.5 * size),
        }
    }
  if prim.IsA(UsdGeom.Plane):
    plane = UsdGeom.Plane(prim)
    width = plane.GetWidthAttr().Get(time)
    length = plane.GetLengthAttr().Get(time)
    return {
        '': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale(
                mi.ScalarVector3f(width * 0.5, length * 0.5, 1)
            ),
        }
    }
  if prim.IsA(UsdGeom.Sphere):
    radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get(time)
    return {
        '': {'type': 'sphere', 'to_world': mi.ScalarTransform4f().scale(radius)}
    }
  if prim.IsA(UsdGeom.Cylinder_1):
    cylinder = UsdGeom.Cylinder_1(prim)
    radius = cylinder.GetRadiusBottomAttr().Get(time)
    if cylinder.GetRadiusTopAttr().Get(time) != radius:
      Tf.Warn(
          f'Cylinder {prim.GetPath()} has differing top and bottom radii,'
          ' which is not supported. Rendering it with its bottom radius.'
      )
  elif prim.IsA(UsdGeom.Cylinder):
    cylinder = UsdGeom.Cylinder(prim)
    radius = cylinder.GetRadiusAttr().Get(time)
  else:
    return {}
  height = cylinder.GetHeightAttr().Get(time)
  # Rotates the Z-aligned Mitsuba shapes onto the spine axis.
  axis = {
      UsdGeom.Tokens.x: mi.ScalarTransform4f(
          [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
      ),
      UsdGeom.Tokens.y: mi.ScalarTransform4f(
          [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
      ),
      UsdGeom.Tokens.z: mi.ScalarTransform4f(),
  }[cylinder.GetAxisAttr().Get(time)]
  top = axis.translate([0, 0, 0.5 * height])
  bottom = axis.translate([0, 0, -0.5 * height])
  return {
      'body': {
          'type': 'cylinder',
          'to_world': bottom.scale([radius, radius, height]),
      },
      'cap_top': {'type': 'disk', 'to_world': top.scale([radius, radius, 1])},
      'cap_bottom': {
          'type': 'disk',
          'to_world': bottom.scale([radius, radius, 1]),
          'flip_normals': True,
      },
  }
