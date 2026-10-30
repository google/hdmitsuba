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

from collections.abc import Mapping
import tempfile
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender
from pxr import UsdVol

from usd_mitsuba import camera
from usd_mitsuba import gaussian_splatting
from usd_mitsuba import instancing
from usd_mitsuba import light
from usd_mitsuba import material
from usd_mitsuba import mesh
from usd_mitsuba import motion
from usd_mitsuba import render_settings as render_settings_lib
from usd_mitsuba import util


def _wrap_shape_if_animated(
    prim: Usd.Prim,
    shape_dict: dict[str, Any],
    time: Usd.TimeCode,
    motion_interval: tuple[float, float],
) -> dict[str, Any]:
  """Applies `prim`'s world transform, wrapping in an instance if animated.

  Emissive shapes are never wrapped since Mitsuba does not support emitters on
  instances; like area lights, they render with their static transform.
  """
  prim_id = util.get_mitsuba_id(prim)
  to_world = (
      util.get_world_transform(prim, time)
      if 'emitter' in shape_dict
      else motion.sample_world_transform(
          prim, time, motion_interval, uniform=True
      )
  )
  if isinstance(to_world, mi.AnimatedTransform4f):
    return motion.wrap_in_instance(prim_id, [shape_dict], to_world)
  shape_dict['to_world'] = to_world @ shape_dict['to_world']
  return {prim_id: shape_dict}


def _convert_cube(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    motion_interval: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
  """Handles a cube prim and returns the Mitsuba shape dictionary."""
  bsdf, emitter, displacement, interior_medium = material.convert_material(prim)

  if displacement is not None:
    raise ValueError(
        'Displacement is not supported for cubes. Please use a mesh.'
    )

  size = UsdGeom.Cube(prim).GetSizeAttr().Get(time)
  mi_cube: dict[str, Any] = {
      'type': 'cube',
      'to_world': mi.ScalarTransform4f().scale(0.5 * size),
  }
  if bsdf is not None:
    mi_cube['bsdf'] = bsdf
  if interior_medium is not None:
    mi_cube['interior'] = interior_medium
  if emitter is not None:
    mi_cube['emitter'] = emitter

  return _wrap_shape_if_animated(prim, mi_cube, time, motion_interval)


def _convert_plane(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    motion_interval: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
  """Handles a plane prim and returns the Mitsuba shape dictionary."""
  bsdf, emitter, displacement, _ = material.convert_material(prim)
  if displacement is not None:
    raise ValueError(
        'Displacement is not supported for planes. Please use a mesh.'
    )

  plane_prim = UsdGeom.Plane(prim)
  width = plane_prim.GetWidthAttr().Get(time)
  height = plane_prim.GetLengthAttr().Get(time)
  mi_plane = {
      'type': 'rectangle',
      'to_world': mi.ScalarTransform4f().scale(
          mi.ScalarVector3f(width * 0.5, height * 0.5, 1)
      ),
  }
  if bsdf is not None:
    mi_plane['bsdf'] = bsdf
  if emitter is not None:
    mi_plane['emitter'] = emitter
  return _wrap_shape_if_animated(prim, mi_plane, time, motion_interval)


def _supports_transform(
    part: dict[str, Any], to_world: mi.ScalarTransform4f, eps: float = 1e-4
) -> bool:
  """Returns whether `part` supports the object-to-world transform `to_world`."""
  cols = np.array((to_world @ part['to_world']).matrix, dtype=np.float64)[
      :3, :3
  ].T
  lengths = np.linalg.norm(cols, axis=1)
  axes = lengths if part['type'] == 'sphere' else lengths[:2]
  return bool(
      np.all(np.abs(axes - lengths[0]) <= eps * lengths[0])
      and all(
          abs(np.dot(cols[i], cols[j])) <= eps * lengths[i] * lengths[j]
          for i, j in ((0, 1), (0, 2), (1, 2))
      )
  )


def _convert_implicit_surface(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    sensor_bindings: Mapping[Sdf.Path, Usd.Prim],
    motion_interval: tuple[float, float],
) -> dict[str, Any]:
  """Converts a sphere or cylinder prim to native Mitsuba shapes."""
  if prim.IsA(UsdGeom.Sphere):
    radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get(time)
    parts: dict[str, dict[str, Any]] = {
        '': {'type': 'sphere', 'to_world': mi.ScalarTransform4f().scale(radius)}
    }
  else:
    if prim.IsA(UsdGeom.Cylinder_1):
      cylinder = UsdGeom.Cylinder_1(prim)
      radius = cylinder.GetRadiusBottomAttr().Get(time)
      if cylinder.GetRadiusTopAttr().Get(time) != radius:
        Tf.Warn(
            f'Cylinder {prim.GetPath()} has differing top and bottom radii,'
            ' which is not supported. Rendering it with its bottom radius.'
        )
    else:
      cylinder = UsdGeom.Cylinder(prim)
      radius = cylinder.GetRadiusAttr().Get(time)
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
    parts = {
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

  bsdf, emitter, displacement, interior_medium = material.convert_material(prim)
  if displacement is not None:
    raise ValueError(
        'Displacement is not supported for implicit surfaces. Please use a'
        ' mesh.'
    )
  for i, part in enumerate(parts.values()):
    if bsdf is not None:
      part['bsdf'] = bsdf
    if interior_medium is not None:
      part['interior'] = interior_medium
    if emitter is not None:
      part['emitter'] = (
          emitter if i == 0 else material.convert_material(prim)[1]
      )

  first_part = next(iter(parts.values()))
  if sensor_prim := sensor_bindings.get(prim.GetPath()):
    if len(parts) > 1:
      Tf.Warn(
          f'The surface sensor of {prim.GetPath()} only measures its side, not'
          ' its caps.'
      )
    first_part['sensor'] = camera.usd_to_mitsuba(
        UsdGeom.Camera(sensor_prim), time=time
    )

  prim_id = util.get_mitsuba_id(prim)
  if emitter is not None or sensor_prim is not None:
    to_world = util.get_world_transform(prim, time)
    if not _supports_transform(first_part, to_world):
      Tf.Warn(
          f'{prim.GetPath()} has a transform with non-uniform scaling or'
          ' shearing, which requires a Mitsuba instance, but also an emitter'
          ' or surface sensor, which Mitsuba does not support on instances. It'
          ' renders distorted.'
      )
  else:
    to_world = motion.sample_world_transform(
        prim, time, motion_interval, uniform=True
    )
    if isinstance(to_world, mi.AnimatedTransform4f) or not _supports_transform(
        first_part, to_world
    ):
      return motion.wrap_in_instance(prim_id, parts.values(), to_world)
  return {
      (f'{prim_id}_{s}' if s else prim_id): {
          **part,
          'to_world': to_world @ part['to_world'],
      }
      for s, part in parts.items()
  }


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


# M_bspline^-1 @ M_catmullRom (tension 0.5).
_CATMULL_ROM_TO_BSPLINE = (
    dr.scalar.Matrix4f(
        [[7, -4, 5, -2], [-2, 11, -4, 1], [1, -4, 11, -2], [-2, 5, -4, 7]]
    )
    / 6.0
)


def _catmull_rom_to_bspline(
    points: mi.Float, counts: np.ndarray, pinned: bool
) -> mi.Float:
  """Converts Catmull-Rom strands into disjoint 4-point B-spline segments."""
  first = -1 if pinned else 0
  counts_dr = mi.Int32(counts)
  segment_counts = dr.maximum(0, counts_dr - 3 - 2 * first)
  strand = dr.repeat(dr.arange(mi.Int32, len(counts_dr)), segment_counts)
  if len(strand) == 0:
    return mi.Float()
  local = (
      dr.arange(mi.Int32, len(strand))
      - dr.gather(mi.Int32, dr.prefix_sum(segment_counts), strand)
      + first
      + mi.Array4i(0, 1, 2, 3)
  )
  clamped = dr.clip(local, 0, dr.gather(mi.Int32, counts_dr, strand) - 1)
  anchor = dr.gather(mi.Int32, dr.prefix_sum(counts_dr), strand) + clamped
  w = [dr.gather(mi.Vector4f, points, anchor[k]) for k in range(4)]
  if pinned:
    for k in (0, 3):
      nb = dr.gather(mi.Vector4f, points, anchor[k] + clamped[k] - local[k])
      w[k] = 2.0 * w[k] - mi.Vector4f(nb.x, nb.y, nb.z, w[k].w)
  return dr.ravel((_CATMULL_ROM_TO_BSPLINE @ mi.Matrix4f(*w)).T)


def _convert_curves(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> mi.Object:
  """Handles a curves prim and returns the Mitsuba curve object."""
  curve_prim = UsdGeom.Curves(prim)
  widths = curve_prim.GetWidthsAttr().Get(time)
  if widths is None or len(widths) == 0:
    raise ValueError(
        f"Curves prim '{prim.GetPath()}' is missing widths attribute."
    )

  curve_points = np.array(
      curve_prim.GetPointsAttr().Get(time), dtype=np.float32
  )
  world_mat = np.array(
      curve_prim.ComputeLocalToWorldTransform(time), dtype=np.float32
  )
  ones = np.ones((curve_points.shape[0], 1), dtype=np.float32)
  pts_homog = np.hstack([curve_points, ones])
  pts_world_homog = pts_homog @ world_mat
  curve_points = pts_world_homog[:, :3] / pts_world_homog[:, 3:4]
  bsdf, _, _, _ = material.convert_material(prim)

  world_scale = float(np.mean(np.linalg.norm(world_mat[:3, :3], axis=1)))
  widths = np.array(widths, dtype=np.float32) * world_scale
  if widths.ndim == 0 or widths.size == 1:
    radius = np.full((curve_points.shape[0], 1), widths.item() * 0.5)
  elif widths.size == curve_points.shape[0]:
    radius = widths.reshape(-1, 1) * 0.5
  else:
    # Fallback to mean for other interpolations (e.g., uniform per curve)
    radius = np.full((curve_points.shape[0], 1), np.mean(widths) * 0.5)
  curve_points = np.hstack([curve_points, radius])

  curve_point_counts = np.asarray(
      curve_prim.GetCurveVertexCountsAttr().Get(time), dtype=np.int32
  )
  if (
      np.any(curve_point_counts < 0)
      or curve_point_counts.sum() != curve_points.shape[0]
  ):
    raise ValueError(
        f"Curves prim '{prim.GetPath()}': curveVertexCounts sum to "
        f'{curve_point_counts.sum()} (min {curve_point_counts.min(initial=0)})'
        f' but there are {curve_points.shape[0]} points. Fix the curve '
        'topology in the USD asset.'
    )

  curve_points = mi.Float(curve_points.ravel())
  curve_type = 'linearcurve'
  if prim.IsA(UsdGeom.BasisCurves):
    basis_curves = UsdGeom.BasisCurves(prim)
    if basis_curves.GetTypeAttr().Get(time) == UsdGeom.Tokens.cubic:
      basis = basis_curves.GetBasisAttr().Get(time)
      if basis in (UsdGeom.Tokens.bspline, UsdGeom.Tokens.catmullRom):
        curve_type = 'bsplinecurve'
      if basis == UsdGeom.Tokens.catmullRom:
        curve_points = _catmull_rom_to_bspline(
            curve_points,
            curve_point_counts,
            basis_curves.GetWrapAttr().Get(time) == UsdGeom.Tokens.pinned,
        )
        curve_point_counts = np.full(len(curve_points) // 16, 4)
  elif prim.IsA(UsdGeom.NurbsCurves):
    curve_type = 'bsplinecurve'
  degree = 3 if curve_type == 'bsplinecurve' else 1
  # Mitsuba expects the segment_indices to be an array of point indices,
  # skipping the last "degree" points for each segment.
  repeats = np.maximum(0, curve_point_counts - degree)
  if repeats.sum() == 0:
    raise ValueError(
        f"Curves prim '{prim.GetPath()}' has no valid segments for degree"
        f' {degree}.'
    )
  skipped = curve_point_counts - repeats
  curve_indices = np.arange(repeats.sum()) + np.repeat(
      np.cumsum(skipped) - skipped, repeats
  )

  curve_obj = _create_empty_mitsuba_curve(curve_type, bsdf)
  params = mi.traverse(curve_obj)
  params['control_point_count'] = len(curve_points) // 4
  params['segment_indices'] = mi.Int32(curve_indices)
  params['control_points'] = curve_points
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
  if not integrator_dict.setdefault('type', 'path'):
    raise ValueError('mitsuba:integrator:type cannot be empty, if specified.')

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
  sensor_bindings = camera.get_surface_sensor_bindings(stage)
  # Shapes and lights are sampled over the union of all camera shutters.
  motion_interval = motion.get_motion_interval(stage, time)
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
          prim,
          subdivision_level,
          time,
          mi_scene_dict,
          sensor_bindings,
          motion_interval,
      )
    elif prim.IsA(UsdGeom.Mesh):
      mi_scene_dict.update(
          mesh.convert_mesh(
              prim,
              subdivision_level,
              time,
              sensor_bindings,
              motion_interval=motion_interval,
          )
      )
    elif prim.IsA(UsdGeom.Cube):
      mi_scene_dict.update(_convert_cube(prim, time, motion_interval))
    elif prim.IsA(UsdGeom.Plane):
      mi_scene_dict.update(_convert_plane(prim, time, motion_interval))
    elif (
        prim.IsA(UsdGeom.Sphere)
        or prim.IsA(UsdGeom.Cylinder)
        or prim.IsA(UsdGeom.Cylinder_1)
    ):
      mi_scene_dict.update(
          _convert_implicit_surface(prim, time, sensor_bindings, motion_interval)
      )
    elif prim.IsA(UsdGeom.Camera):
      # A sensor bound to a shape is emitted as part of that shape.
      if camera.get_target_shape_path(prim) is None:
        mi_scene_dict[mi_id] = camera.usd_to_mitsuba(
            UsdGeom.Camera(prim), time=time)
    elif prim.IsA(UsdLux.NonboundableLightBase) or prim.IsA(
        UsdLux.BoundableLightBase
    ):
      mi_scene_dict[mi_id] = light.convert_light(prim, time, motion_interval)
    elif prim.IsA(UsdGeom.Curves):
      mi_scene_dict[mi_id] = _convert_curves(prim, time)
    elif prim.IsA(UsdVol.ParticleField3DGaussianSplat):
      mi_scene_dict.update(gaussian_splatting.convert_gaussian_splats(prim, time))
  return mi_scene_dict
