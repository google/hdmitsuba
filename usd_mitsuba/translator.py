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

"""Converts a USD stage to a Mitsuba scene for rendering."""

from __future__ import annotations

import tempfile
from typing import Any

import mitsuba as mi
import numpy as np

from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender

from usd_mitsuba import camera
from usd_mitsuba import instancing
from usd_mitsuba import light
from usd_mitsuba import material
from usd_mitsuba import mesh
from usd_mitsuba import render_settings as render_settings_lib
from usd_mitsuba import util


def _convert_cube(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> dict[str, Any]:
  """Handles a cube prim and returns the Mitsuba shape dictionary."""
  bsdf, emitter, displacement = material.convert_material(prim)

  if displacement is not None:
    raise ValueError(
        'Displacement is not supported for cubes. Please use a mesh.'
    )

  mi_cube: dict[str, Any] = {
      'type': 'cube',
      'to_world': util.get_world_transform(prim, time),
  }
  if bsdf is not None:
    mi_cube['bsdf'] = bsdf
  if emitter is not None:
    mi_cube['emitter'] = emitter

  return mi_cube


def _convert_plane(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> dict[str, Any]:
  """Handles a plane prim and returns the Mitsuba shape dictionary."""
  bsdf, emitter, displacement = material.convert_material(prim)
  if displacement is not None:
    raise ValueError(
        'Displacement is not supported for planes. Please use a mesh.'
    )

  plane_prim = UsdGeom.Plane(prim)
  width = plane_prim.GetWidthAttr().Get(time)
  height = plane_prim.GetLengthAttr().Get(time)
  mi_plane = {
      'type': 'rectangle',
      'to_world': (
          util.get_world_transform(prim, time).scale(
              mi.ScalarVector3f(width * 0.5, height * 0.5, 1)
          )
      ),
  }
  if bsdf is not None:
    mi_plane['bsdf'] = bsdf
  if emitter is not None:
    mi_plane['emitter'] = emitter
  return mi_plane


def _create_empty_mitsuba_curve(curve_type: str, bsdf: Any) -> mi.Object:
  """Creates an empty Mitsuba curve object.

  Mitsuba's curve plugins (linearcurve, bsplinecurve) require a filename
  containing control points for initialization. Since we want to define the
  points dynamically from USD, we create a temporary file with dummy points,
  load it, and then we can overwrite the parameters in memory.
  """
  # 4 points is the minimum required for B-splines.
  dummy_data = (
      "-1.0 0.0 0.0 0.5\n"
      "-0.5 0.0 0.0 0.5\n"
      " 0.5 0.0 0.0 0.5\n"
      " 1.5 0.0 0.0 0.5\n"
  )
  with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
    f.write(dummy_data)
    f.flush()
    curve_dict: dict[str, Any] = {
        'type': curve_type,
        'filename': f.name,
    }
    if bsdf is not None:
      curve_dict['bsdf'] = bsdf
    return mi.load_dict(curve_dict)


def _convert_curves(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> mi.Object:
  """Handles a curves prim and returns the Mitsuba curve object."""
  curve_prim = UsdGeom.BasisCurves(prim)
  curve_points = np.array(curve_prim.GetPointsAttr().Get(time))
  bsdf, _, _ = material.convert_material(prim)
  widths = curve_prim.GetWidthsAttr().Get(time)
  if widths is not None:
    widths = np.array(widths)
    if widths.ndim == 0 or widths.size == 1:
      radius = np.full((curve_points.shape[0], 1), widths.item())
    elif widths.size == curve_points.shape[0]:
      radius = widths.reshape(-1, 1)
    else:
      # Fallback to mean for other interpolations (e.g., uniform per curve)
      radius = np.full((curve_points.shape[0], 1), np.mean(widths))
  else:
    radius = np.full((curve_points.shape[0], 1), 0.01)
  curve_points = np.hstack([curve_points, radius])

  curve_point_counts = np.array(
      curve_prim.GetCurveVertexCountsAttr().Get(time))

  # TODO: Correctly parse this from the USD prim.
  curve_type = 'linearcurve'
  curve_obj = _create_empty_mitsuba_curve(curve_type, bsdf)
  degree = 3 if curve_type == 'bsplinecurve' else 1
  params = mi.traverse(curve_obj)
  params['control_point_count'] = curve_points.shape[0]

  # Mitsuba expects the segment_indices to be an array of point indices,
  # skipping the last "degree" points for each segment.
  n_indices = curve_points.shape[0] - degree * curve_point_counts.shape[0]
  indices = np.arange(n_indices, dtype=np.int32)
  per_vertex_offset = (
      np.arange(curve_point_counts.shape[0], dtype=np.int32) * degree
  )
  repeats = np.maximum(0, curve_point_counts - degree)
  curve_indices = indices + np.repeat(per_vertex_offset, repeats)

  params['segment_indices'] = mi.Int32(curve_indices)
  params['control_points'] = mi.Float(np.ravel(curve_points))
  params.update()
  return curve_obj


def _convert_render_settings(
    stage: Usd.Stage,
    render_settings: UsdRender.Settings | None,
    render_product_path: str | None = None,
    variant_name: str | None = None,
) -> dict[str, Any]:
  """Handles the render settings and returns the integrator dictionary."""
  resolved_variant = variant_name
  if not resolved_variant and render_settings:
    prim = render_settings.GetPrim()
    if variant_attr := prim.GetAttribute('mitsuba:variant'):
      resolved_variant = variant_attr.Get()

  if resolved_variant:
    if resolved_variant != mi.variant():
      mi.set_variant(resolved_variant)
  else:
    if not mi.variant():
      mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

  if not render_settings:
    return {'type': 'path'}

  prim = render_settings.GetPrim()
  integrator_dict = util.extract_nested_dict(prim, 'mitsuba:integrator:')
  if integrator_dict:
    if not integrator_dict.get('type'):
      raise ValueError(
          'mitsuba:integrator:type cannot be empty, if specified.')
  else:
    integrator_dict = {'type': 'path'}

  render_color, aovs = render_settings_lib.parse_aovs(
      stage, render_settings, render_product_path
  )
  if aovs:
    aov_integrator: dict[str, Any] = {'type': 'aov', 'aovs': ','.join(aovs)}
    if render_color:
      aov_integrator['integrator'] = dict(integrator_dict)
    integrator_dict = aov_integrator

  return integrator_dict


def convert_to_mitsuba(
    stage: Usd.Stage,
    subdivision_level: int = 1,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    render_product_path: str | None = None,
    variant_name: str | None = None,
) -> dict[str, Any]:
  """Converts a USD stage to a Mitsuba scene for rendering.

  This function converts to a Mitsuba scene dictionary, which can be used to
  can be modified further or loaded using `mi.load_dict`.

  Args:
    stage: The USD stage to convert.
    subdivision_level: The default subdivision level for meshes.
    time: The time code to evaluate the scene at.
    render_product_path: Optional path to the render product to use.
    variant_name: The Mitsuba variant to resolve and set.

  Returns:
    A Mitsuba scene dictionary.
  """
  render_settings = render_settings_lib.get_render_settings(stage)
  mi_scene_dict: dict[str, Any] = {'type': 'scene'}
  mi_scene_dict['integrator'] = _convert_render_settings(
      stage,
      render_settings,
      render_product_path,
      variant_name,
  )

  prototype_paths = instancing.get_prototype_paths(stage)
  # Traverse the stage using TraverseInstanceProxies. This flattens native USD
  # instances (instanceable=true), which means they will be duplicated in Mitsuba.
  # This is since Mitsuba's instancing does not support BSDF or emitter changes
  # across instances. Note that USD's PointInstancer, however, does translate
  # to actual Mitsuba instances.
  for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if not prim.IsActive():
      continue
    if prim.GetPath() in prototype_paths:
      continue
    if prim.IsA(UsdGeom.Imageable) and (
        UsdGeom.Imageable(prim).ComputeVisibility(time)
        == UsdGeom.Tokens.invisible
    ):
      continue
    mi_id = util.get_mitsuba_id(prim)
    if prim.IsA(UsdGeom.PointInstancer):
      instancing.convert_point_instancer(
          prim, subdivision_level, time, mi_scene_dict)
    elif prim.IsA(UsdGeom.Mesh):
      mi_scene_dict.update(mesh.convert_mesh(prim, subdivision_level, time))
    elif prim.IsA(UsdGeom.Cube):
      mi_scene_dict[mi_id] = _convert_cube(prim, time)
    elif prim.IsA(UsdGeom.Plane):
      mi_scene_dict[mi_id] = _convert_plane(prim, time)
    elif prim.IsA(UsdGeom.Camera):
      mi_scene_dict[mi_id] = camera.usd_to_mitsuba(
          UsdGeom.Camera(prim), time=time)
    elif prim.IsA(UsdLux.NonboundableLightBase) or prim.IsA(
        UsdLux.BoundableLightBase
    ):
      mi_scene_dict[mi_id] = light.convert_light(prim, time)
    elif prim.IsA(UsdGeom.BasisCurves):
      mi_scene_dict[mi_id] = _convert_curves(prim, time)
  return mi_scene_dict
