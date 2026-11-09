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

"""Tests for the mesh geometry pipeline shared by Hydra and usd_mitsuba."""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from hdmitsuba import geometry_ext as geom_lib
from usd_mitsuba import mesh as mesh_lib

# Two unit quads in the z = 0 plane sharing the edge between points 1 and 4.
_POINTS = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]
_COUNTS = [4, 4]
_INDICES = [0, 1, 4, 3, 1, 2, 5, 4]


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _uniform(value: float) -> tuple[Any, bool, bool]:
  return mi.load_dict({'type': 'uniform', 'value': value}), False, False


def _vector(value: tuple[float, float, float]) -> tuple[Any, bool, bool]:
  return mi.load_dict({'type': 'rgb', 'value': value}), False, True


def _process(
    displacement: tuple[Any, bool, bool] | None,
    transform: Gf.Matrix4d | None = None,
    requested: set[str] | None = None,
    points: list[tuple[float, float, float]] | None = None,
    normals: list[tuple[float, float, float]] | None = None,
    normals_interpolation: str = UsdGeom.Tokens.faceVarying,
    st: list[tuple[float, float]] | None = None,
) -> Any:
  """Runs the geometry pipeline on the two quads and returns the sub-mesh."""
  stage = Usd.Stage.CreateInMemory()
  mesh = UsdGeom.Mesh.Define(stage, '/Mesh')
  mesh.CreatePointsAttr(points or _POINTS)
  mesh.CreateFaceVertexCountsAttr(_COUNTS)
  mesh.CreateFaceVertexIndicesAttr(_INDICES)
  mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
  if normals is not None:
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(normals_interpolation)
  if st is not None:
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(st)
  mesh_data = geom_lib.extract_mesh_data(
      stage, mesh.GetPath(), Usd.TimeCode.Default()
  )
  (sub,) = geom_lib.process_mesh(
      mesh_data,
      0,
      transform or Gf.Matrix4d(1.0),
      [(None, True, False) if displacement is None else displacement],
      requested or set(),
  )
  return sub


def _get(sub: Any, name: str) -> np.ndarray:
  return np.array(sub.primvars[name])


def test_scalar_displacement_is_applied_in_object_space():
  transform = Gf.Matrix4d().SetScale(Gf.Vec3d(1, 1, 2)) * Gf.Matrix4d(
  ).SetTranslate(Gf.Vec3d(0, 0, 1))
  sub = _process(
      _uniform(0.5),
      transform,
      {'__p_object', '__n_disp_object', '__n_disp_world'},
  )
  points = _get(sub, 'points')
  # The object-space offset of 0.5 along z is scaled by the transform.
  np.testing.assert_allclose(points[:, 2], 2.0, atol=1e-5)
  p_object = _get(sub, '__p_object')
  np.testing.assert_allclose(p_object[:, 2], 0.0, atol=1e-5)
  np.testing.assert_allclose(p_object[:, :2], points[:, :2], atol=1e-5)
  np.testing.assert_allclose(_get(sub, '__n_disp_object'), [[0, 0, 1]] * 6)
  # The normal-to-world transform scales z normals by 1/2, so the encoded
  # world-space normal is (0, 0, 1) / (1/2).
  np.testing.assert_allclose(_get(sub, '__n_disp_world'), [[0, 0, 2]] * 6)
  np.testing.assert_allclose(
      _get(sub, 'normals'), [[0, 0, 1]] * 6, atol=1e-5
  )


@pytest.mark.parametrize('displaced', [False, True])
def test_mirrored_transform_keeps_outward_winding(displaced: bool):
  sub = _process(
      _uniform(1.0) if displaced else None,
      Gf.Matrix4d().SetScale(Gf.Vec3d(-1, 1, 1)),
  )
  points = _get(sub, 'points')
  tri = points[np.array(sub.triangles).reshape(-1, 3)]
  normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  assert np.all(normals[:, 2] > 0)
  if displaced:
    np.testing.assert_allclose(
        _get(sub, 'normals'), [[0, 0, 1]] * len(points), atol=1e-5
    )


def test_authored_face_varying_normals_are_preserved():
  # Fold the second quad up by 90 degrees around the shared edge, with a hard
  # edge: each face uses its own face normal.
  n0, n1 = (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)
  sub = _process(
      _uniform(0.1),
      requested={'__n_object'},
      points=[(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1)],
      normals=[n0] * 4 + [n1] * 4,
  )
  points = _get(sub, 'points')
  normals = _get(sub, 'normals')
  # The shared edge vertices are split by the face-varying normals.
  assert len(points) == 8
  # Shared vertices are displaced along the averaged geometric normal, so both
  # copies stay at the same position.
  on_edge = (points[:, 0] > 0.9) & (points[:, 2] < 0.1)
  offset = 0.1 * np.sqrt(0.5)
  np.testing.assert_allclose(points[on_edge, 0], 1.0 - offset, atol=1e-5)
  np.testing.assert_allclose(points[on_edge, 2], offset, atol=1e-5)
  # The authored hard-edge normals are only slightly rotated along with the
  # geometric normals.
  close_to = lambda n: np.all(np.abs(normals - n) < 0.05, axis=1)
  assert np.all(close_to(n0) | close_to(n1))
  np.testing.assert_allclose(_get(sub, '__n_object'), normals, atol=1e-5)


def test_authored_vertex_normals_define_displacement_direction():
  n = (0.0, 0.6, 0.8)
  sub = _process(
      _uniform(1.0),
      requested={'__p_object'},
      normals=[n] * len(_POINTS),
      normals_interpolation=UsdGeom.Tokens.vertex,
  )
  offsets = _get(sub, 'points') - _get(sub, '__p_object')
  np.testing.assert_allclose(offsets, [n] * len(_POINTS), atol=1e-5)
  # The planar mesh is only translated, so the normals remain unchanged.
  np.testing.assert_allclose(_get(sub, 'normals'), [n] * len(_POINTS), atol=1e-5)


def test_vector_displacement_uses_uv_tangent_frame_across_seams():
  # The second quad uses a separate UV island, so the shared edge is a seam.
  sub = _process(
      _vector((0.25, 0.5, 1.0)),
      Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
      st=[(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (3, 0), (3, 1), (2, 1)],
  )
  # On both sides of the seam, the object-space tangent frame is (+x, +y, +z),
  # which the rotation maps to (+y, -x, +z). The object-space points (x, y, 0)
  # with integer x and y are thus displaced to (-y - 0.5, x + 0.25, 1).
  points = _get(sub, 'points')
  np.testing.assert_allclose(
      points[:, 0] + 0.5, np.round(points[:, 0] + 0.5), atol=1e-4
  )
  np.testing.assert_allclose(
      points[:, 1] - 0.25, np.round(points[:, 1] - 0.25), atol=1e-4
  )
  np.testing.assert_allclose(points[:, 2], 1.0, atol=1e-4)


def test_vector_displacement_without_uvs_uses_normal():
  sub = _process(_vector((0.0, 0.0, 0.5)))
  np.testing.assert_allclose(_get(sub, 'points')[:, 2], 0.5, atol=1e-5)


def test_flat_shaded_mesh_provides_object_normals():
  sub = _process(
      None,
      Gf.Matrix4d().SetScale(Gf.Vec3d(1, 1, 3)),
      {'__p_object', '__n_object'},
  )
  assert 'normals' not in sub.primvars
  assert len(_get(sub, '__p_object')) == len(_get(sub, 'points'))
  n_object = _get(sub, '__n_object')
  np.testing.assert_allclose(n_object, [[0, 0, 1]] * len(n_object), atol=1e-5)


def test_mesh_attributes_use_usd_texture_coordinates():
  # The texture coordinates equal the xy coordinates of the points.
  sub = _process(
      None,
      st=[(0, 0), (1, 0), (1, 1), (0, 1), (1, 0), (2, 0), (2, 1), (1, 1)],
  )
  mi_mesh = mesh_lib._to_mitsuba_mesh(sub, mi.Properties(), [('st', 2)])
  attribute = np.array(mi_mesh.attribute('vertex_st_2')).reshape(-1, 2)
  np.testing.assert_allclose(attribute, _get(sub, 'points')[:, :2])


def test_points_shared_by_displaced_subsets_are_displaced_once():
  stage = Usd.Stage.CreateInMemory()
  mesh = UsdGeom.Mesh.Define(stage, '/Mesh')
  mesh.CreatePointsAttr(_POINTS)
  mesh.CreateFaceVertexCountsAttr(_COUNTS)
  mesh.CreateFaceVertexIndicesAttr(_INDICES)
  mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
  for i in range(2):
    subset = UsdGeom.Subset.Define(stage, f'/Mesh/subset{i}')
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr('materialBind')
    subset.CreateIndicesAttr([i])
  UsdGeom.Subset.SetFamilyType(
      mesh, 'materialBind', UsdGeom.Tokens.nonOverlapping
  )
  mesh_data = geom_lib.extract_mesh_data(
      stage, mesh.GetPath(), Usd.TimeCode.Default()
  )
  sub_meshes = geom_lib.process_mesh(
      mesh_data,
      0,
      Gf.Matrix4d(1.0),
      [_uniform(1.0), _uniform(2.0)],
      set(),
  )
  # The first subset displaces the points on the shared edge (x = 1).
  for sub, offset in zip(sub_meshes, [1.0, 2.0]):
    points = _get(sub, 'points')
    on_edge = np.isclose(points[:, 0], 1.0)
    np.testing.assert_allclose(points[on_edge, 2], 1.0)
    np.testing.assert_allclose(points[~on_edge, 2], offset)
