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
from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdLux
from pxr import UsdGeom
from pxr import Vt
from hdmitsuba import geometry_ext as geom_lib
from usd_mitsuba import camera
from usd_mitsuba import material
from usd_mitsuba import util


def _sub_mesh_for_material(spec: Any, material_id: Any) -> Any:
  """Returns the sub-mesh bound to `material_id`, or None."""
  for sub in spec.sub_meshes:
    if sub.material_id == material_id:
      return sub
  return None


def _apply_displacement(
    primvars: Any, sub: Any, displacement: mi.Texture
) -> None:
  """Displaces the source points reached by one material's faces.

  Mirrors the Hydra delegate: the whole, un-welded point array is displaced, and
  a point shared by several corners is displaced once, using the first corner
  that reaches it. `sub` supplies the addressing -- its per-record `indices`
  already encode the primvar's interpolation mode, so nothing here needs to know
  what that mode was.
  """
  attrs = {a.name: a for a in sub.attributes}
  if 'st' not in attrs or 'normals' not in attrs:
    return

  corner_vertex = np.asarray(sub.corner_vertex)
  # First corner wins, matching the delegate's "displace each vertex once".
  target, first = np.unique(corner_vertex, return_index=True)

  uv_values = np.asarray(attrs['st'].values, dtype=np.float32)
  uv = uv_values[np.asarray(attrs['st'].indices)[first]]
  normals = np.asarray(attrs['normals'].values, dtype=np.float32)[
      np.asarray(attrs['normals'].indices)[first]
  ]

  si = dr.zeros(mi.SurfaceInteraction3f)
  si.uv = mi.Point2f(uv[:, 0], 1.0 - uv[:, 1])
  offset = np.array(displacement.eval_1(si))[..., None] - 0.5

  points = np.array(primvars['points'].value)
  points[target] += offset * normals
  primvars['points'].value = Vt.Vec3fArray.FromNumpy(points.astype(np.float32))


def _to_mitsuba_mesh(
    spec: Any, sub: Any, properties: mi.Properties
) -> mi.Mesh:
  """Creates a Mitsuba mesh from one sub-mesh of a corner-mesh spec."""
  properties['face_normals'] = not spec.smooth_normals

  attrs = {}
  for attr in sub.attributes:
    values = np.asarray(attr.values, dtype=np.float32)
    # The Python binding exposes only the shared `corner_index`, not
    # per-attribute indices, so each attribute is gathered to one row per
    # record.
    attrs[attr.name] = np.ascontiguousarray(
        values[np.asarray(attr.indices)]
    )

  mi_mesh = mi.Mesh(properties)
  mi_mesh.from_corners(
      positions=np.ascontiguousarray(
          np.asarray(spec.positions, dtype=np.float32)
      ),
      # np.array (not asarray): Vt arrays expose a read-only buffer, and the
      # index parameters of from_corners are declared non-const, so nanobind
      # rejects anything unwritable. The float parameters are const and would
      # accept a view.
      corner_vertex=np.array(sub.corner_vertex, dtype=np.int32),
      face_offsets=np.array(sub.face_offsets, dtype=np.int32),
      normals=attrs.get('normals'),
      texcoords=attrs.get('st'),
  )
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

  mesh_data, policy = geom_lib.extract_and_process_meshes(
      stage, path, time, subdivision_level, has_displacement
  )
  if custom_transform is not None:
    world_transform = custom_transform
  else:
    world_transform = mesh_prim.ComputeLocalToWorldTransform(time)

  primvars = mesh_data.primvars
  if 'points' not in primvars or len(primvars['points'].value) == 0:
    Tf.Warn(f"Mesh {prim.GetPath()} has no points. Skipping translation.")
    return {}

  material_ids = list(mesh_data.material_ids)
  if not material_ids:
    material_ids = [Sdf.Path.emptyPath]

  # Resolve each bound material once, up front: displacement has to be applied
  # before the mesh is welded, and the sub-mesh split is what scopes it.
  resolved = {}
  for material_id in material_ids:
    subprim = (
        stage.GetPrimAtPath(material_id)
        if not material_id.isEmpty
        else prim
    )
    resolved[material_id] = (subprim, material.convert_material(subprim))

  # Build once for addressing only, so displacement can find each material's
  # points and their UVs/normals, then rebuild below from the displaced state.
  displaced = False
  if has_displacement:
    addressing = geom_lib.build_corner_mesh(
        mesh_data.id(),
        mesh_data.face_vertex_counts,
        mesh_data.face_vertex_indices,
        primvars,
        material_ids,
        mesh_data.face_material_indices,
        policy.smooth_normals,
    )
    for material_id, (_, (_, _, displacement)) in resolved.items():
      if displacement is None:
        continue
      sub = _sub_mesh_for_material(addressing, material_id)
      if sub is not None:
        _apply_displacement(primvars, sub, displacement)
        displaced = True

  primvars = geom_lib.transform_primvars(primvars, world_transform)
  if displaced:
    primvars = geom_lib.refresh_normals_after_displacement(
        primvars,
        mesh_data.face_vertex_counts,
        mesh_data.face_vertex_indices,
        len(material_ids) > 1,
    )
  primvars = geom_lib.normalize_normals(primvars)

  spec = geom_lib.build_corner_mesh(
      mesh_data.id(),
      mesh_data.face_vertex_counts,
      mesh_data.face_vertex_indices,
      primvars,
      material_ids,
      mesh_data.face_material_indices,
      policy.smooth_normals,
  )

  converted_meshes = {}
  for sub in spec.sub_meshes:
    subprim, (bsdf, material_emitter, _) = resolved[sub.material_id]
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

    converted_meshes[util.get_mitsuba_id(subprim)] = _to_mitsuba_mesh(
        spec, sub, props
    )
  return converted_meshes
