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

"""Handles mesh processing and conversions for Mitsuba rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdLux
from pxr import UsdGeom
from hdmitsuba import geometry_ext as geom_lib
from usd_mitsuba import camera
from usd_mitsuba import light
from usd_mitsuba import material
from usd_mitsuba import motion
from usd_mitsuba import util


def _to_mitsuba_mesh(
    sub: Any,
    properties: mi.Properties,
    mesh_attributes: Iterable[tuple[str, int]],
) -> mi.Mesh:
  """Creates a Mitsuba mesh from a sub-mesh object and properties.

  Args:
    sub: The `geometry_ext.SubMeshOutput` to convert.
    properties: The properties of the Mitsuba mesh.
    mesh_attributes: `(primvar name, channel count)` pairs of the primvars that
      are attached as vertex attributes. Primvars missing from `sub` are
      skipped.

  Returns:
    The Mitsuba mesh.
  """

  primvars = sub.primvars
  vertices = np.array(primvars['points'])
  faces = np.array(sub.triangles).reshape(-1, 3)
  has_vertex_normals = 'normals' in primvars
  # Without (authored or computed) normals the mesh is flat shaded. Requesting
  # face normals keeps Mitsuba from deriving smooth shading normals of its own.
  if 'face_normals' not in properties:
    properties['face_normals'] = not has_vertex_normals
  texture_coordinates = (
      np.array(primvars['st']) if 'st' in primvars else None
  )

  normals = mi.TensorXf()
  if has_vertex_normals and not properties['face_normals']:
    normals = mi.TensorXf(np.array(primvars['normals'], dtype=np.float32))
  texcoords = mi.TensorXf()
  if texture_coordinates is not None:
    texture_coordinates[:, 1] = 1 - texture_coordinates[:, 1]
    texcoords = mi.TensorXf(texture_coordinates.astype(np.float32))

  # The mesh fields are immutable after construction, so they all have to be
  # supplied up front.
  mi_mesh = mi.Mesh(properties)
  mi_mesh.from_fields(
      faces=mi.TensorXu(faces.astype(np.uint32)),
      positions=mi.TensorXf(vertices.astype(np.float32)),
      normals=normals,
      texcoords=texcoords,
  )
  geom_lib.add_mesh_attributes(mi_mesh, sub, mesh_attributes)
  return mi_mesh


def _set_plugin_properties(
    properties: mi.Properties, overrides: dict[str, Any]
) -> None:
  """Applies scalar parameters and instantiated child plugins onto `properties`."""
  for key, value in overrides.items():
    if key in ('type', 'id'):
      continue
    if isinstance(value, dict):
      if 'type' not in value:
        raise ValueError(
            f"Nested Mitsuba shape override '{key}' is missing '{key}:type'."
        )
      loaded = mi.load_dict(value)
      if isinstance(loaded, list):
        if len(loaded) == 1:
          properties[key] = loaded[0]
        elif len(loaded) > 1:
          for idx, item in enumerate(loaded):
            properties[f'{key}_{idx}'] = item
      else:
        properties[key] = loaded
    else:
      properties[key] = value


def convert_mesh(
    prim: Usd.Prim,
    subdivision_level: int,
    time: Usd.TimeCode,
    sensor_bindings: Mapping[Sdf.Path, Usd.Prim],
    custom_transform: Gf.Matrix4d | None = None,
    motion_interval: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
  """Converts a mesh prim and returns a dictionary of Mitsuba scene shapes.

  Args:
    prim: The USD prim.
    subdivision_level: The subdivision level.
    time: The time code.
    sensor_bindings: Precomputed shape path -> sensor prim map.
    custom_transform: Optional transform to use instead of local-to-world.
    motion_interval: Frame-relative shutter interval `(open, close)`.

  Returns:
    A dictionary mapping Mitsuba scene object IDs to Mitsuba shape objects or
    dictionaries.
  """
  stage = prim.GetStage()
  path = prim.GetPath()
  level_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(
      'mitsuba:subdivision_level'
  )
  if level_primvar and (level := level_primvar.Get()) is not None:
    subdivision_level = level

  mesh_data = geom_lib.extract_mesh_data(stage, path, time)
  # Faces not bound to a GeomSubset use the material of the mesh itself (an
  # empty material path).
  subprims = [
      prim if material_id.isEmpty else stage.GetPrimAtPath(material_id)
      for material_id in mesh_data.material_ids
  ]
  emitter_overrides = util.extract_nested_dict(prim, 'mitsuba:emitter:', time)
  shape_overrides = util.extract_nested_dict(prim, 'mitsuba:shape:', time)
  custom_id = shape_overrides.pop('id', None)

  # GeomSubsets share the sidedness of their mesh.
  double_sided = UsdGeom.Mesh(prim).GetDoubleSidedAttr().Get()
  materials = [
      material.convert_mesh_material(
          subprim, double_sided, emitter_overrides=emitter_overrides
      )
      for subprim in subprims
  ]

  # Emitters and sensors are only attached to top-level shapes: Mitsuba does
  # not support them on instances, so prototypes do without, and shapes
  # carrying them are not wrapped in an instance for motion blur below (they
  # render with their static transform at frame time, like area lights).
  sensor_prim = sensor_bindings.get(path)
  has_emitter_or_sensor = (
      prim.HasAPI(UsdLux.MeshLightAPI)
      or sensor_prim is not None
      or bool(emitter_overrides.get('type'))
      or 'emitter' in shape_overrides
      or 'sensor' in shape_overrides
      or any(mat.emitter is not None for mat in materials)
  )

  motion_transform = None
  if custom_transform is not None:
    world_transform = custom_transform
    shape_overrides.pop('emitter', None)
    shape_overrides.pop('sensor', None)
  elif has_emitter_or_sensor:
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
  else:
    samples = geom_lib.sample_world_transform(
        prim, time, motion_interval, uniform=True
    )
    if len(samples) == 1:
      world_transform = samples[0][1]
    else:
      world_transform = Gf.Matrix4d(1.0)
      motion_transform = motion.to_mitsuba_transform(samples)

  geom_transform = world_transform
  if 'to_world' in shape_overrides:
    extra_to_world = shape_overrides.pop('to_world')
    if hasattr(extra_to_world, 'matrix'):
      extra_matrix = Gf.Matrix4d(
          np.array(extra_to_world.matrix, dtype=np.float64).T.tolist()
      )
      geom_transform = extra_matrix * world_transform

  displacements = [
      (mat.displacement, mat.requires_uv, mat.is_vector)
      for mat in materials
  ]
  requested_attrs = {
      name for mat in materials for name, _ in mat.mesh_attributes
  }
  sub_meshes = geom_lib.process_mesh(
      mesh_data,
      subdivision_level,
      geom_transform,
      displacements,
      requested_attrs,
  )

  has_medium = 'interior' in shape_overrides or 'exterior' in shape_overrides
  if (
      custom_transform is None
      and sensor_prim is not None
      and len(sub_meshes) > 1
  ):
    Tf.Warn(
        f'The surface sensor of {prim.GetPath()} only measures its first'
        ' material subset.'
    )

  materials_by_id = {
      material_id: (subprim, mat)
      for material_id, subprim, mat in zip(
          mesh_data.material_ids, subprims, materials
      )
  }
  converted_meshes = {}
  for i, sub in enumerate(sub_meshes):
    subprim, mat = materials_by_id[sub.material_id]
    props = mi.Properties()
    if mat.interior_medium is not None:
      props['interior'] = mat.interior_medium
    if 'bsdf' not in shape_overrides:
      if mat.bsdf is not None:
        props['bsdf'] = mat.bsdf
      elif has_medium or mat.interior_medium is not None:
        props['bsdf'] = mi.load_dict({'type': 'null'})

    if custom_transform is None:
      # MeshLightAPI takes precedence over material-driven emission. Each
      # sub-mesh needs its own emitter instance, and the world transform is
      # also the IES light frame of a mesh light.
      mesh_light_emitter = light.convert_mesh_light(
          prim, time, world_transform, emitter_overrides
      )
      if mesh_light_emitter is not None:
        props['emitter'] = mesh_light_emitter
      elif mat.emitter is not None:
        props['emitter'] = mat.emitter
      elif emitter_overrides.get('type'):
        props['emitter'] = mi.load_dict(emitter_overrides)

      if sensor_prim is not None and i == 0:
        props['sensor'] = mi.load_dict(
            camera.usd_to_mitsuba(UsdGeom.Camera(sensor_prim), time=time)
        )

    if shape_overrides:
      sub_overrides = (
          {k: v for k, v in shape_overrides.items() if k != 'sensor'}
          if i > 0 and 'sensor' in shape_overrides
          else shape_overrides
      )
      _set_plugin_properties(props, sub_overrides)

    mesh_id = util.get_mitsuba_id(subprim)
    if (
        custom_transform is None
        and len(sub_meshes) == 1
        and isinstance(custom_id, str)
        and custom_id
    ):
      mesh_id = custom_id
    props.set_id(mesh_id)
    converted_meshes[mesh_id] = _to_mitsuba_mesh(
        sub, props, mat.mesh_attributes
    )

  if motion_transform is not None and converted_meshes:
    return motion.wrap_in_instance(
        util.get_mitsuba_id(prim), converted_meshes.values(), motion_transform
    )
  return converted_meshes
