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

import os
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import Vt


def get_mitsuba_id(prim: Usd.Prim) -> str:
  """Converts a USD prim's path to a Mitsuba scene dictionary key.

  Args:
    prim: The USD prim.

  Returns:
    A valid Mitsuba scene object ID.
  """
  return str(prim.GetPath()).replace('/', '_').replace('.', '_')


def deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> None:
  """Recursively merges `overrides` into `base` in place."""
  for key, value in overrides.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      deep_merge_dict(base[key], value)
    else:
      base[key] = value


_VECTOR_ROLE_NAMES = frozenset({
    Sdf.ValueRoleNames.Point,
    Sdf.ValueRoleNames.Vector,
    Sdf.ValueRoleNames.Normal,
})


def _normalize_nested_dict(node: dict[str, Any], mi_mod: Any) -> None:
  """Normalizes nested dict entries for `mi.load_dict` compatibility."""
  for key, value in list(node.items()):
    if isinstance(value, dict):
      if value.get('type') == 'vector' and 'value' in value:
        node[key] = tuple(value['value'])
      elif value.get('type') == 'spectrum' and 'value' in value:
        spec_val = value['value']
        if isinstance(spec_val, str):
          raw = spec_val.strip()
          if ':' in raw:
            pairs = []
            for item in raw.split(','):
              wavelength_str, weight_str = item.split(':', 1)
              pairs.append((float(wavelength_str), float(weight_str)))
            value['value'] = pairs
          elif raw:
            value['value'] = float(raw)
        elif isinstance(spec_val, (Vt.Vec2fArray, Vt.Vec2dArray)):
          value['value'] = [(float(p[0]), float(p[1])) for p in spec_val]
      elif value.get('type') == 'spectrum' and 'filename' in value:
        filename = value['filename']
        if (
            isinstance(filename, str)
            and filename
            and not os.path.exists(filename)
        ):
          resolved = str(mi_mod.file_resolver().resolve(filename))
          if os.path.exists(resolved):
            value['filename'] = resolved
      else:
        _normalize_nested_dict(value, mi_mod)


def extract_nested_dict(
    prim: Usd.Prim,
    prefix: str,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> dict[str, Any]:
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
    time: The time code to evaluate the attributes at.

  Returns:
    A dictionary of attributes matching the prefix.
  """
  mi_mod = mi if mi.variant() is not None else mi.scalar_rgb
  result = {}
  for prop in prim.GetAuthoredPropertiesInNamespace(prefix):
    if not isinstance(prop, Usd.Attribute) or not prop.HasAuthoredValue():
      continue
    value: Any = prop.Get(time)
    if value is None or isinstance(value, Sdf.ValueBlock):
      continue
    if isinstance(value, Sdf.AssetPath):
      value = value.resolvedPath or value.path
    name = prop.GetName()
    key_parts = name[len(prefix) :].split(':')
    if any(not part for part in key_parts):
      continue
    leaf = key_parts[-1]
    if leaf in ('type', 'id'):
      if not isinstance(value, str) or not value:
        continue
    current_level = result
    for part in key_parts[:-1]:
      if not isinstance(current_level.get(part), dict):
        current_level[part] = {}
      current_level = current_level[part]
    if leaf not in ('type', 'id') and isinstance(current_level.get(leaf), dict):
      continue
    if isinstance(value, (Gf.Vec3f, Gf.Vec3d)):
      if prop.GetRoleName() in _VECTOR_ROLE_NAMES:
        value = tuple(value)
      else:
        value = mi_mod.ScalarColor3f(*value)
    elif isinstance(value, (Gf.Vec4f, Gf.Vec4d)):
      value = mi_mod.ScalarColor3f(*value[:3])
    elif isinstance(value, (Gf.Matrix3f, Gf.Matrix3d)):
      value = mi_mod.ScalarAffineTransform3f(
          mi_mod.ScalarMatrix3f(np.array(value, dtype=np.float32).T)
      )
    elif isinstance(value, (Gf.Matrix4f, Gf.Matrix4d)):
      value = mi_mod.ScalarAffineTransform4f(
          mi_mod.ScalarMatrix4f(np.array(value, dtype=np.float32).T)
      )
    elif isinstance(value, (Vt.FloatArray, Vt.DoubleArray)):
      if not value:
        continue
      value = ','.join(str(x) for x in value)
    current_level[leaf] = value
  _normalize_nested_dict(result, mi_mod)
  return result


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
