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

import mitsuba as mi
import numpy as np

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
  transform = UsdGeom.Imageable(prim).ComputeLocalToWorldTransform(time)
  return mi.ScalarTransform4f(np.array(transform, dtype=np.float32).T)
