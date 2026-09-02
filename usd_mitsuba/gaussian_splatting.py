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

"""3D Gaussian Splat processing and conversion for Mitsuba rendering."""

from __future__ import annotations

import logging
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Usd
from pxr import UsdVol

from usd_mitsuba import util


def convert_gaussian_splats(
    prim: Usd.Prim,
    time: Usd.TimeCode,
) -> dict[str, Any]:
  """Converts a Gaussian splat prim and returns a dictionary for Mitsuba.

  Args:
    prim: The USD prim, which should be a ParticleField3DGaussianSplat.
    time: The time code.

  Returns:
    A dictionary containing the Mitsuba ellipsoidsmesh shape.
  """
  usd_splats = UsdVol.ParticleField3DGaussianSplat(prim)
  if not usd_splats:
    raise ValueError(
        f'Prim {prim.GetPath()} is not a ParticleField3DGaussianSplat.'
    )

  points = usd_splats.GetPositionsAttr().Get(time)
  scales = usd_splats.GetScalesAttr().Get(time)
  orientations = usd_splats.GetOrientationsAttr().Get(time)
  opacities = usd_splats.GetOpacitiesAttr().Get(time)

  if points is None or len(points) == 0:
    return {}

  num_splats = len(points)

  for name, attr in (
      ('scales', scales),
      ('orientations', orientations),
      ('opacities', opacities),
  ):
    if attr is None or len(attr) != num_splats:
      logging.warning(
          'Gaussian splats %s have missing or mismatched %s (%s vs %d'
          ' positions), skipping.',
          prim.GetPath(),
          name,
          'none' if attr is None else len(attr),
          num_splats,
      )
      return {}

  # The 'ellipsoids' plugin only accepts 'to_world' when it loads a PLY file,
  # so the prim transform has to be baked into the particles here. Split it the
  # same way Mitsuba does on its own PLY path: centers take the full matrix,
  # while per-splat scales and orientations can only carry a uniform scale and
  # a rigid rotation.
  world_transform = util.get_world_transform(prim, time)
  scale_matrix, rotation, _ = dr.transform_decompose(world_transform.matrix, 25)
  scale_diag = dr.diag(scale_matrix)
  scale_factor = scale_diag[0]

  scale_eps = 1e-3 * abs(scale_factor)
  if (
      abs(scale_diag[1] - scale_factor) > scale_eps
      or abs(scale_diag[2] - scale_factor) > scale_eps
  ):
    logging.warning(
        'Non-uniform transform scale (%f, %f, %f) on Gaussian splats %s is not'
        ' supported; applying x-axis scale uniformly.',
        scale_diag[0],
        scale_diag[1],
        scale_diag[2],
        prim.GetPath(),
    )

  # Vt arrays expose their storage through the buffer protocol, so np.asarray
  # is a view and dr.unravel takes the interleaved data as-is - one copy into
  # Dr.Jit, no host-side temporaries.
  to_world = mi.AffineTransform4f(world_transform.matrix)
  centers = to_world @ dr.unravel(
      mi.Point3f, mi.Float(np.asarray(points, dtype=np.float32).ravel())
  )

  # Vt.QuatfArray marshals to (N, 4) laid out as (x, y, z, w), which is both
  # Mitsuba's Quaternion4f layout and the one the plugin expects. USD does not
  # guarantee unit-length orientations and the plugin only normalizes on its
  # PLY path, so normalize after composing with the prim rotation.
  local_quats = dr.unravel(
      mi.Quaternion4f,
      mi.Float(np.asarray(orientations, dtype=np.float32).ravel()),
  )
  world_quats = dr.normalize(mi.Quaternion4f(rotation) * local_quats)

  # Per-splat scales only carry the uniform part of the prim transform.
  scales = mi.TensorXf(np.asarray(scales, dtype=np.float32)) * scale_factor

  # Shape dict for Mitsuba ellipsoidsmesh primitive
  mi_splats = {
      'type': 'ellipsoidsmesh',
      'centers': mi.TensorXf(dr.ravel(centers), shape=(num_splats, 3)),
      'scales': scales,
      'quaternions': mi.TensorXf(dr.ravel(world_quats), shape=(num_splats, 4)),
      'opacities': mi.TensorXf(
          np.asarray(opacities, dtype=np.float32).reshape((num_splats, 1))
      ),
  }

  # SH coeffs (optional, uniform white degree-0 fallback if missing or invalid)
  has_sh = False
  sh_coeffs_attr = usd_splats.GetRadianceSphericalHarmonicsCoefficientsAttr()
  sh_degree_attr = usd_splats.GetRadianceSphericalHarmonicsDegreeAttr()
  if sh_coeffs_attr.IsValid() and sh_degree_attr.IsValid():
    sh_coeffs = sh_coeffs_attr.Get(time)
    sh_degree = sh_degree_attr.Get(time)
    if sh_coeffs is not None and sh_degree is not None:
      num_coeffs = (sh_degree + 1) * (sh_degree + 1)
      if len(sh_coeffs) == num_splats * num_coeffs:
        sh_np = np.asarray(sh_coeffs, dtype=np.float32).reshape(
            (num_splats, num_coeffs * 3)
        )
        mi_splats['sh_coeffs'] = mi.TensorXf(sh_np)
        has_sh = True
      else:
        logging.warning(
            'SH coefficients size mismatch: %d vs expected %d. Defaulting to'
            ' uniform white color.',
            len(sh_coeffs),
            num_splats * num_coeffs,
        )

  if not has_sh:
    mi_splats['sh_coeffs'] = dr.full(mi.TensorXf, 1.0, (num_splats, 3))

  return {util.get_mitsuba_id(prim): mi_splats}
