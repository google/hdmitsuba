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

"""Handles PointInstancer translation for Mitsuba rendering."""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from usd_mitsuba import mesh
from usd_mitsuba import util


def get_prototype_paths(stage: Usd.Stage) -> set[Sdf.Path]:
  """Gathers all prototype paths that are targeted by PointInstancers.

  These paths should be skipped during normal stage traversal.

  Args:
    stage: The USD stage to traverse.

  Returns:
    A set of Sdf.Path objects representing prototype prims and their descendants.
  """
  prototype_paths = set()
  for prim in stage.TraverseAll():
    if prim.IsA(UsdGeom.PointInstancer):
      for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
        if proto_prim := stage.GetPrimAtPath(target):
          for p in Usd.PrimRange(proto_prim):
            prototype_paths.add(p.GetPath())
  return prototype_paths


def convert_point_instancer(
    instancer_prim: Usd.Prim,
    subdivision_level: int,
    time: Usd.TimeCode,
    mi_scene_dict: dict[str, Any],
) -> None:
  """Converts a PointInstancer prim to Mitsuba ShapeGroups and Instances.

  Args:
    instancer_prim: The PointInstancer USD prim.
    subdivision_level: The subdivision level for prototype meshes.
    time: The time code to evaluate at.
    mi_scene_dict: The top-level Mitsuba scene dictionary to populate.
  """
  instancer = UsdGeom.PointInstancer(instancer_prim)
  instancer_id = util.get_mitsuba_id(instancer_prim)
  transforms = instancer.ComputeInstanceTransformsAtTime(time, time)
  if not transforms:
    return
  proto_indices = instancer.GetProtoIndicesAttr().Get(time)
  if not proto_indices:
    return
  prototype_paths = instancer.GetPrototypesRel().GetTargets()
  used_proto_indices = set(proto_indices)
  shape_groups = {}
  for proto_idx in used_proto_indices:
    if proto_idx >= len(prototype_paths):
      continue
    proto_path = prototype_paths[proto_idx]
    proto_prim = instancer_prim.GetStage().GetPrimAtPath(proto_path)
    if not proto_prim:
      continue

    proto_to_local_transform = UsdGeom.Xformable(
        proto_prim).ComputeLocalToWorldTransform(time).GetInverse()
    group_id = f"proto_group_{instancer_id}_{proto_idx}"
    group_dict = {'type': 'shapegroup'}
    shape_idx = 0
    # TODO(vicini): Support other geometric prim types (curves, implicit shapes) in prototypes.
    # Currently only UsdGeom.Mesh is supported.
    for child_prim in Usd.PrimRange(proto_prim):
      if child_prim.IsA(UsdGeom.Mesh):
        child_xform = UsdGeom.Xformable(child_prim)
        relative_trans = (
            child_xform.ComputeLocalToWorldTransform(time)
            * proto_to_local_transform
        )
        meshes = mesh.convert_mesh(
            child_prim,
            subdivision_level,
            time,
            custom_transform=relative_trans,
        )
        for _, mesh_obj in meshes.items():
          group_dict[f"shape_{shape_idx}"] = mesh_obj
          shape_idx += 1
    if shape_idx > 0:
      mi_scene_dict[group_id] = group_dict
      shape_groups[proto_idx] = group_id

  num_instances = min(len(transforms), len(proto_indices))
  for inst_idx in range(num_instances):
    proto_idx = proto_indices[inst_idx]
    if proto_idx not in shape_groups:
      continue
    inst_id = f"instance_{instancer_id}_{proto_idx}_{inst_idx}"
    mi_scene_dict[inst_id] = {
        'type': 'instance',
        'shapegroup': {
            'type': 'ref',
            'id': shape_groups[proto_idx],
        },
        'to_world': mi.ScalarTransform4f(np.array(transforms[inst_idx], dtype=np.float32).T),
    }
