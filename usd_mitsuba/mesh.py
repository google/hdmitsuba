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

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Tf
from pxr import Usd
from pxr import UsdLux
from pxr import UsdGeom
from pxr import Vt
from hdmitsuba import geometry_ext as geom_lib
from usd_mitsuba import camera
from usd_mitsuba import material
from usd_mitsuba import util


def _apply_displacement(
    sub: Any, displacement: mi.Texture, mesh_data: Any
) -> None:
  """Applies a displacement texture to a sub-mesh object."""
  uv_state = sub.primvars['st']
  uv = np.array(uv_state.value)
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.uv = mi.Point2f(uv[:, 0], 1.0 - uv[:, 1])
  offset = np.array(displacement.eval_1(si))[..., None] - 0.5
  points = np.array(sub.primvars['points'].value)
  points += offset * np.array(sub.primvars['normals'].value)
  sub.primvars['points'].value = Vt.Vec3fArray.FromNumpy(
      points.astype(np.float32)
  )
  sub_topology = geom_lib.HdMeshTopology(
      mesh_data.scheme,
      mesh_data.orientation,
      Vt.IntArray([3] * (len(sub.triangles) // 3)),
      sub.triangles,
  )
  sub.primvars = geom_lib.compute_normals(sub.primvars, sub_topology)


# TODO: Add support for vertex colors and other primvars.
def _to_mitsuba_mesh(sub: Any, properties: mi.Properties) -> mi.Mesh:
  """Creates a Mitsuba mesh from a sub-mesh object and properties."""

  vertices = np.array(sub.primvars['points'].value)
  faces = np.array(sub.triangles).reshape(-1, 3)
  has_vertex_normals = 'normals' in sub.primvars
  properties['face_normals'] = not has_vertex_normals
  texture_coordinates = (
      np.array(sub.primvars['st'].value) if 'st' in sub.primvars else None
  )
  mi_mesh = mi.Mesh(
      name='',
      vertex_count=vertices.shape[0],
      face_count=faces.shape[0],
      props=properties,
      has_vertex_normals=has_vertex_normals,
      has_vertex_texcoords=texture_coordinates is not None,
  )

  params = mi.traverse(mi_mesh)
  params['faces'] = np.ravel(faces)
  params['vertex_positions'] = np.ravel(vertices)
  if texture_coordinates is not None:
    texture_coordinates[:, 1] = 1 - texture_coordinates[:, 1]
    params['vertex_texcoords'] = np.ravel(texture_coordinates)
  if has_vertex_normals:
    params['vertex_normals'] = np.ravel(
        np.array(sub.primvars['normals'].value))

  params.update()
  return mi_mesh


def _get_mesh_light_emitter(
    prim: Usd.Prim,
    time: Usd.TimeCode,
) -> mi.Emitter | None:
  """Detects MeshLightAPI and returns a Mitsuba area emitter if present."""
  if not prim.HasAPI(UsdLux.MeshLightAPI):
    return None
  light_api = UsdLux.LightAPI(prim)
  intensity = light_api.GetIntensityAttr().Get(time)
  color = light_api.GetColorAttr().Get(time)
  exposure = light_api.GetExposureAttr().Get(time)
  return mi.load_dict({
      'type': 'area',
      'radiance': {
          'type': 'rgb',
          'value': mi.ScalarColor3f(*color) * intensity * (2.0**exposure),
      },
  })


def convert_mesh(
    prim: Usd.Prim,
    subdivision_level: int,
    time: Usd.TimeCode,
    custom_transform: Gf.Matrix4d | None = None,
) -> dict[str, mi.Mesh]:
  """Converts a mesh prim and returns a dictionary of Mitsuba meshes.

  Args:
    prim: The USD prim.
    subdivision_level: The subdivision level.
    time: The time code.
    custom_transform: Optional transform to use instead of local-to-world.

  Returns:
    A dictionary mapping Mitsuba scene object IDs to mi.Mesh objects.
  """
  stage = prim.GetStage()
  path = prim.GetPath()
  has_displacement = material.has_displacement(prim)
  mesh_prim = UsdGeom.Mesh(prim)
  if level_attr := prim.GetAttribute('mitsuba:subdivision_level'):
    if (level := level_attr.Get()) is not None:
      subdivision_level = level

  mesh_data, sub_meshes = geom_lib.extract_and_process_meshes(
      stage, path, time, subdivision_level, has_displacement
  )
  if custom_transform is not None:
    world_transform = custom_transform
  else:
    world_transform = mesh_prim.ComputeLocalToWorldTransform(time)

  converted_meshes = {}
  for sub in sub_meshes:
    if 'points' not in sub.primvars or len(sub.primvars['points'].value) == 0:
      Tf.Warn(f"Mesh {prim.GetPath()} has no points. Skipping translation.")
      continue
    subprim = (
        stage.GetPrimAtPath(sub.material_id)
        if not sub.material_id.isEmpty
        else prim
    )
    bsdf, material_emitter, displacement = material.convert_material(subprim)
    mesh_light_emitter = _get_mesh_light_emitter(prim, time)
    props = mi.Properties()
    if bsdf is not None:
      props['bsdf'] = bsdf

    # MeshLightAPI takes precedence over material-driven emission.
    if mesh_light_emitter is not None:
      props['emitter'] = mesh_light_emitter
    elif material_emitter is not None:
      props['emitter'] = material_emitter

    if (sensor_attr := prim.GetAttribute('mitsuba:sensor')) and (
        sensor_path := sensor_attr.Get()
    ):
      cam_prim = stage.GetPrimAtPath(sensor_path)
      if cam_prim and cam_prim.IsA(UsdGeom.Camera):
        props['sensor'] = mi.load_dict(
            camera.usd_to_mitsuba(UsdGeom.Camera(cam_prim), time=time)
        )

    if displacement is not None:
      _apply_displacement(sub, displacement, mesh_data)

    sub.primvars = geom_lib.transform_primvars(sub.primvars, world_transform)
    converted_meshes[util.get_mitsuba_id(
        subprim)] = _to_mitsuba_mesh(sub, props)
  return converted_meshes
