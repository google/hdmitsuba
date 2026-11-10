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

"""End-to-end AOV tests for the mesh geometry and primvar pipeline.

Each test builds a small quad grid over `[0, 1]^2` in the `z = 0` plane,
renders geometric AOVs (`position`, `geo_normal`, `sh_normal`, `uv`, `albedo`,
`depth`) at 1 sample per pixel with a box filter, and verifies that:
  1. Hydra (`hdMitsuba`) and offline translation (`usd_mitsuba`) match.
  2. The rendered AOVs satisfy the expected geometric or shading property.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mitsuba as mi
import numpy as np
import numpy.typing as npt
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdShade
import usd_render
from hdmitsuba.tests import test_helpers


_IMAGE_SIZE = 32
_HALF = _IMAGE_SIZE // 2
_AOVS = ('position', 'geo_normal', 'sh_normal', 'uv', 'albedo', 'depth')
_HYDRA_ENGINES: list[tuple[Usd.Stage, usd_render.RenderEngine]] = []


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  yield
  _HYDRA_ENGINES.clear()


# -----------------------------------------------------------------------------
# Scene and material helpers
# -----------------------------------------------------------------------------


def _create_stage(view_size: float = 0.9) -> Usd.Stage:
  """Creates a Z-up stage with a top-down camera centered on `(0.5, 0.5, 0)`.

  At `z = 0`, the camera sees a `view_size x view_size` square around
  `(0.5, 0.5)`. The default `0.9` frames `[0.05, 0.95]^2` strictly inside the
  unit square; pass `view_size > 1.0` to include the mesh boundary.
  """
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

  camera = UsdGeom.Camera.Define(stage, '/Camera')
  # At distance 10 with focal length 100, visible width is aperture * 10 / 100.
  aperture = view_size * 10.0
  camera.CreateHorizontalApertureAttr(aperture)
  camera.CreateVerticalApertureAttr(aperture)
  camera.CreateFocalLengthAttr(100.0)
  camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 100.0))
  camera.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.5, 10.0))
  # Keep each sample inside its own pixel.
  camera.GetPrim().CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('box')

  test_helpers.create_aov_render_settings(
      stage, _AOVS, resolution=(_IMAGE_SIZE, _IMAGE_SIZE)
  )
  return stage


def _create_grid(
    stage: Usd.Stage,
    resolution: int = 2,
    subdivision_level: int = 0,
    bumpy: bool = False,
) -> UsdGeom.Mesh:
  """Defines `/Mesh` as `resolution x resolution` CCW quads over `[0, 1]^2`.

  For `resolution=2`, the four faces in row-major order are:
    - face 0: bottom-left  `[0.0, 0.5] x [0.0, 0.5]` -> `image[_HALF:, :_HALF]`
    - face 1: bottom-right `[0.5, 1.0] x [0.0, 0.5]` -> `image[_HALF:, _HALF:]`
    - face 2: top-left     `[0.0, 0.5] x [0.5, 1.0]` -> `image[:_HALF, :_HALF]`
    - face 3: top-right    `[0.5, 1.0] x [0.5, 1.0]` -> `image[:_HALF, _HALF:]`
  """
  coords = np.linspace(0.0, 1.0, resolution + 1)
  x, y = np.meshgrid(coords, coords)
  z = (
      0.1 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
      if bumpy
      else np.zeros_like(x)
  )
  vertex_grid = np.arange((resolution + 1) ** 2).reshape(
      resolution + 1, resolution + 1
  )
  quads = np.stack(
      [
          vertex_grid[:-1, :-1],
          vertex_grid[:-1, 1:],
          vertex_grid[1:, 1:],
          vertex_grid[1:, :-1],
      ],
      axis=-1,
  )

  mesh = UsdGeom.Mesh.Define(stage, '/Mesh')
  mesh.CreatePointsAttr(np.stack([x, y, z], axis=-1).reshape(-1, 3).tolist())
  mesh.CreateFaceVertexCountsAttr([4] * (resolution**2))
  mesh.CreateFaceVertexIndicesAttr(quads.ravel().tolist())
  _set_subdivision_level(mesh, subdivision_level)
  return mesh


def _set_subdivision_level(mesh: UsdGeom.Mesh, level: int) -> None:
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      'mitsuba:subdivision_level',
      Sdf.ValueTypeNames.Int,
      UsdGeom.Tokens.constant,
  ).Set(level)


def _set_primvar(
    mesh: UsdGeom.Mesh,
    name: str,
    type_name: Sdf.ValueTypeName,
    interpolation: str,
    values: Any,
    indices: Sequence[int] | None = None,
) -> UsdGeom.Primvar:
  primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      name, type_name, interpolation
  )
  primvar.Set(list(values) if type_name.isArray else values)
  if indices is not None:
    primvar.SetIndices(list(indices))
  return primvar


def _set_vertex_uvs(mesh: UsdGeom.Mesh) -> None:
  """Authors vertex `st = (x, y)` matching each point's XY coordinates."""
  points = mesh.GetPointsAttr().Get()
  _set_primvar(
      mesh,
      'st',
      Sdf.ValueTypeNames.TexCoord2fArray,
      UsdGeom.Tokens.vertex,
      [(p[0], p[1]) for p in points],
  )


def _create_diffuse_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> UsdShade.Material:
  material = UsdShade.Material.Define(stage, path)
  diffuse = UsdShade.Shader.Define(stage, f'{path}/Diffuse')
  diffuse.CreateIdAttr('mitsuba_diffuse')
  diffuse.CreateInput('reflectance', Sdf.ValueTypeNames.Color3f).Set(
      Gf.Vec3f(*color)
  )
  material.CreateSurfaceOutput().ConnectToSource(
      diffuse.ConnectableAPI(), 'surface'
  )
  return material


def _create_primvar_material(
    stage: Usd.Stage,
    path: str,
    primvar: str,
    reader_type: str = 'float3',
) -> UsdShade.Material:
  """Creates a diffuse material whose reflectance reads `primvar`."""
  material = UsdShade.Material.Define(stage, path)
  diffuse = UsdShade.Shader.Define(stage, f'{path}/Diffuse')
  diffuse.CreateIdAttr('mitsuba_diffuse')
  reader = UsdShade.Shader.Define(stage, f'{path}/Reader')
  reader.CreateIdAttr(f'UsdPrimvarReader_{reader_type}')
  reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set(primvar)
  diffuse.CreateInput(
      'reflectance', Sdf.ValueTypeNames.Color3f
  ).ConnectToSource(reader.ConnectableAPI(), 'result')
  material.CreateSurfaceOutput().ConnectToSource(
      diffuse.ConnectableAPI(), 'surface'
  )
  return material


def _bind_material(target: Any, material: UsdShade.Material) -> None:
  UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(material)


def _create_subset(
    mesh: UsdGeom.Mesh,
    name: str,
    face_indices: Sequence[int],
    material: UsdShade.Material | None = None,
) -> UsdGeom.Subset:
  stage = mesh.GetPrim().GetStage()
  subset = UsdGeom.Subset.Define(stage, f'{mesh.GetPath()}/{name}')
  subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
  subset.CreateFamilyNameAttr('materialBind')
  subset.CreateIndicesAttr(list(face_indices))
  UsdGeom.Subset.SetFamilyType(
      mesh, 'materialBind', UsdGeom.Tokens.nonOverlapping
  )
  if material is not None:
    _bind_material(subset, material)
  return subset


# -----------------------------------------------------------------------------
# Rendering and comparison helpers
# -----------------------------------------------------------------------------


def _render_aovs(stage: Usd.Stage) -> dict[str, np.ndarray]:
  """Renders `stage` with Hydra and `usd_mitsuba` and asserts they match.

  Reuses a single Hydra `RenderEngine` across repeated calls on the same `stage`
  so multi-step tests also exercise incremental Hydra change tracking.
  """
  engine = next((e for s, e in _HYDRA_ENGINES if s == stage), None)
  if engine is None:
    engine = usd_render.RenderEngine(stage)
    engine.configure(
        hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
    )
    _HYDRA_ENGINES.append((stage, engine))
  hydra = test_helpers.render_aovs(stage, 'hydra', '/Camera', engine=engine)
  offline = test_helpers.render_aovs(stage, 'offline', '/Camera')
  _assert_all_aovs_close(hydra, offline, atol=1e-5)
  return hydra


def _assert_close(
    actual: np.ndarray, expected: npt.ArrayLike, atol: float = 1e-4
) -> None:
  np.testing.assert_allclose(
      actual, np.broadcast_to(expected, actual.shape), atol=atol
  )


def _assert_all_aovs_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    atol: float = 1e-4,
) -> None:
  for name in _AOVS:
    np.testing.assert_allclose(
        actual[name], expected[name], atol=atol, err_msg=name
    )


# -----------------------------------------------------------------------------
# Topology, orientation, transforms, GeomSubsets, and holes
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_left_handed_orientation(subdivision_level: int):
  stage = _create_stage()
  mesh = _create_grid(
      stage, resolution=4, subdivision_level=subdivision_level, bumpy=True
  )
  _set_vertex_uvs(mesh)
  right_handed = _render_aovs(stage)

  mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
  # Reverse each quad `[v0, v1, v2, v3]` to `[v0, v3, v2, v1]`: keeping `v0`
  # first preserves the fan-triangulation diagonal `v0-v2` on non-planar quads.
  quads = np.reshape(mesh.GetFaceVertexIndicesAttr().Get(), (-1, 4))
  mesh.GetFaceVertexIndicesAttr().Set(quads[:, [0, 3, 2, 1]].ravel().tolist())
  left_handed = _render_aovs(stage)

  _assert_all_aovs_close(left_handed, right_handed)
  assert np.all(left_handed['geo_normal'][..., 2] > 0)


_TRANSFORMS = {
    'rigid': Gf.Matrix4d().SetTransform(
        Gf.Rotation(Gf.Vec3d(0, 0, 1), 30), Gf.Vec3d(0.1, -0.05, 0.2)
    ),
    'nonuniform_scale': Gf.Matrix4d().SetScale(Gf.Vec3d(0.8, 0.6, 2.5)),
    'mirror': Gf.Matrix4d().SetScale(Gf.Vec3d(-1, 1, 1)),
}


@pytest.mark.parametrize(
    'transform_name, subdivision_level, authored_normals',
    [
        ('rigid', 0, False),
        ('rigid', 0, True),
        ('rigid', 2, False),
        # Subdivision computes smooth normals in object space, which only
        # commutes with rigid and mirroring transforms (not non-uniform scale).
        ('nonuniform_scale', 0, False),
        ('nonuniform_scale', 0, True),
        ('mirror', 0, False),
        ('mirror', 0, True),
        ('mirror', 2, False),
    ],
)
def test_transform_matches_baked_geometry(
    transform_name: str, subdivision_level: int, authored_normals: bool
):
  # Apply the transform around `(0.5, 0.5, 0)` so the grid stays in view.
  to_origin = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-0.5, -0.5, 0.0))
  transform = to_origin * _TRANSFORMS[transform_name] * to_origin.GetInverse()

  # 1. Render with a scene-graph transform on the mesh prim.
  stage = _create_stage()
  mesh = _create_grid(
      stage, resolution=4, subdivision_level=subdivision_level, bumpy=True
  )
  _set_vertex_uvs(mesh)
  points = list(mesh.GetPointsAttr().Get())
  normals = [Gf.Vec3f(p[0] - 0.5, p[1] - 0.5, 1.0) for p in points]
  if authored_normals:
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  mesh.AddTransformOp().Set(transform)
  transformed = _render_aovs(stage)

  # 2. Bake the transform directly into `points`, `normals`, and `orientation`.
  mesh.ClearXformOpOrder()
  mesh.GetPointsAttr().Set([transform.Transform(p) for p in points])
  if authored_normals:
    normal_matrix = transform.GetInverse().GetTranspose()
    mesh.GetNormalsAttr().Set([normal_matrix.TransformDir(n) for n in normals])
  if transform.GetDeterminant() < 0.0:
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
  baked = _render_aovs(stage)

  _assert_all_aovs_close(baked, transformed)


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_geom_subset_material_assignment(subdivision_level: int):
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2, subdivision_level=subdivision_level)

  red, green, blue = (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.2, 0.8)
  _bind_material(mesh, _create_diffuse_material(stage, '/Materials/Base', blue))
  _create_subset(
      mesh, 'Red', [0], _create_diffuse_material(stage, '/Materials/Red', red)
  )
  _create_subset(
      mesh,
      'Green',
      [1],
      _create_diffuse_material(stage, '/Materials/Green', green),
  )
  # Face 2 is in a subset with no material, and face 3 is not in any subset;
  # both fall back to the mesh's own material (`blue`).
  _create_subset(mesh, 'Unbound', [2])

  albedo = _render_aovs(stage)['albedo']
  _assert_close(albedo[_HALF:, :_HALF], red)  # face 0: bottom-left
  _assert_close(albedo[_HALF:, _HALF:], green)  # face 1: bottom-right
  _assert_close(albedo[:_HALF, :_HALF], blue)  # face 2: top-left
  _assert_close(albedo[:_HALF, _HALF:], blue)  # face 3: top-right


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_geom_subsets_preserve_surface_geometry(subdivision_level: int):
  stage = _create_stage()
  mesh = _create_grid(
      stage, resolution=4, subdivision_level=subdivision_level, bumpy=True
  )
  _set_vertex_uvs(mesh)
  _bind_material(mesh, _create_diffuse_material(stage, '/Materials/Base'))
  whole_mesh = _render_aovs(stage)

  # Split faces across subsets bound to equivalent materials (leaving the last
  # row assigned to the parent mesh's material).
  _create_subset(
      mesh, 'S0', [0, 1, 4, 5, 6], _create_diffuse_material(stage, '/Mat0')
  )
  _create_subset(
      mesh, 'S1', [2, 3, 7], _create_diffuse_material(stage, '/Mat1')
  )
  _create_subset(
      mesh, 'S2', [8, 9, 10, 11], _create_diffuse_material(stage, '/Mat2')
  )
  split_mesh = _render_aovs(stage)

  _assert_all_aovs_close(split_mesh, whole_mesh)


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_hole_indices(subdivision_level: int):
  """Hole faces are omitted, and surviving faces keep materials and primvars."""
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2, subdivision_level=subdivision_level)

  # Mark the left half (faces 0 and 2) as holes.
  mesh.CreateHoleIndicesAttr([0, 2])

  # Color face 1 (bottom-right) via a GeomSubset material, and assign each
  # face a distinct `uniform` `st` primvar. Under subdivision, both rely on
  # mapping surviving patches back to coarse faces while skipping holes.
  red, green = (0.8, 0.1, 0.1), (0.1, 0.8, 0.2)
  _set_primvar(
      mesh,
      'st',
      Sdf.ValueTypeNames.TexCoord2fArray,
      UsdGeom.Tokens.uniform,
      [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.25, 0.75)],
  )
  _bind_material(
      mesh, _create_diffuse_material(stage, '/Materials/Mesh', green)
  )
  _create_subset(
      mesh, 'Red', [1], _create_diffuse_material(stage, '/Materials/Red', red)
  )
  # Displace hole face 0 to verify it does not displace shared seam vertices.
  hole_mat = _create_diffuse_material(stage, '/Materials/Hole')
  disp = UsdShade.Shader.Define(stage, '/Materials/Hole/Displacement')
  disp.CreateIdAttr('mitsuba_uniform')
  disp.CreateInput('value', Sdf.ValueTypeNames.Float).Set(1.0)
  hole_mat.CreateDisplacementOutput('mitsuba').ConnectToSource(
      disp.ConnectableAPI(), 'out'
  )
  _create_subset(mesh, 'DisplacedHole', [0], hole_mat)

  aovs = _render_aovs(stage)
  # Left half (faces 0 and 2) shows the empty background.
  _assert_close(aovs['depth'][:, :_HALF], 0.0)
  _assert_close(aovs['albedo'][:, :_HALF], (0.0, 0.0, 0.0))
  # Right half hits face 1 (bottom-right) and face 3 (top-right), undisplaced.
  assert np.all(aovs['depth'][:, _HALF:] > 0.0)
  _assert_close(aovs['position'][:, _HALF:, 2], 0.0)
  _assert_close(aovs['albedo'][_HALF:, _HALF:], red)
  _assert_close(aovs['albedo'][:_HALF, _HALF:], green)
  _assert_close(aovs['uv'][_HALF:, _HALF:], (0.2, 0.8))
  _assert_close(aovs['uv'][:_HALF, _HALF:], (0.25, 0.25))

  # Marking all faces as holes produces an empty image in both backends.
  mesh.GetHoleIndicesAttr().Set([0, 1, 2, 3])
  _assert_close(_render_aovs(stage)['depth'], 0.0)


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_invalid_topology(subdivision_level: int):
  """Meshes with inconsistent topologies are skipped, and shown once fixed."""
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2, subdivision_level=subdivision_level)
  points = mesh.GetPointsAttr().Get()

  # Face vertex indices exceeding the points (a points-only change).
  mesh.GetPointsAttr().Set(points[:4])
  assert _render_aovs(stage)['depth'][_HALF, _HALF] == 0.0
  mesh.GetPointsAttr().Set(points)
  assert _render_aovs(stage)['depth'][_HALF, _HALF] > 0.0
  # Face vertex counts not summing to the number of face vertex indices.
  mesh.GetFaceVertexCountsAttr().Set([4, 4, 4])
  assert _render_aovs(stage)['depth'][_HALF, _HALF] == 0.0


# -----------------------------------------------------------------------------
# Subdivision and displacement
# -----------------------------------------------------------------------------


def test_catmull_clark_subdivision():
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=4, subdivision_level=0, bumpy=True)
  coarse = _render_aovs(stage)

  _set_subdivision_level(mesh, 2)
  refined = _render_aovs(stage)

  # Subdividing a curved cage changes surface positions.
  assert np.max(np.abs(coarse['position'] - refined['position'])) > 0.01
  # Unsubdivided meshes without authored normals are flat-shaded, whereas
  # subdivided meshes get smooth shading normals.
  _assert_close(coarse['sh_normal'], coarse['geo_normal'], atol=1e-5)
  assert np.max(np.abs(refined['sh_normal'] - refined['geo_normal'])) > 1e-3


def test_subdivision_boundary_interpolation():
  stage = _create_stage(view_size=1.2)
  mesh = _create_grid(stage, resolution=4, subdivision_level=0)
  _set_vertex_uvs(mesh)
  coarse = _render_aovs(stage)

  # Default `edgeAndCorner` boundary interpolation preserves `[0, 1]^2`.
  _set_subdivision_level(mesh, 3)
  pinned = _render_aovs(stage)
  _assert_all_aovs_close(pinned, coarse)

  # `none` leaves boundary faces unpinned / omitted near the open border.
  mesh.CreateInterpolateBoundaryAttr(UsdGeom.Tokens.none)
  unpinned = _render_aovs(stage)
  assert np.sum(unpinned['depth'] > 0.0) < np.sum(pinned['depth'] > 0.0)


@pytest.mark.parametrize('tag_kind', ['crease', 'corner'])
def test_subdivision_creases_and_corners(tag_kind: str):
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=4, subdivision_level=3, bumpy=True)
  smooth = _render_aovs(stage)

  if tag_kind == 'crease':
    # Sharp crease along the ridge `x = 0.25`.
    mesh.CreateCreaseIndicesAttr([1, 6, 11, 16, 21])
    mesh.CreateCreaseLengthsAttr([5])
    mesh.CreateCreaseSharpnessesAttr([10.0])
  else:
    # Sharp corner at vertex `(0.25, 0.5)`.
    mesh.CreateCornerIndicesAttr([11])
    mesh.CreateCornerSharpnessesAttr([10.0])

  tagged = _render_aovs(stage)
  assert not np.allclose(smooth['position'], tagged['position'], atol=0.01)


def test_triangle_subdivision_rule():
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=4, subdivision_level=3, bumpy=True)
  quads = np.reshape(mesh.GetFaceVertexIndicesAttr().Get(), (-1, 4))
  tris = quads[:, [0, 1, 2, 0, 2, 3]].ravel().tolist()
  mesh.CreateFaceVertexCountsAttr([3] * (len(tris) // 3))
  mesh.CreateFaceVertexIndicesAttr(tris)
  catmull_clark = _render_aovs(stage)

  mesh.CreateTriangleSubdivisionRuleAttr(UsdGeom.Tokens.smooth)
  smooth = _render_aovs(stage)
  assert not np.allclose(
      catmull_clark['position'], smooth['position'], atol=1e-3
  )


@pytest.mark.parametrize('subdivision_level', [0, 2])
def test_scalar_displacement(subdivision_level: int):
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2, subdivision_level=subdivision_level)
  mesh.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 2.0))
  _set_vertex_uvs(mesh)

  material = _create_diffuse_material(stage, '/Material')
  disp = UsdShade.Shader.Define(stage, '/Material/Displacement')
  disp.CreateIdAttr('mitsuba_uniform')
  disp.CreateInput('value', Sdf.ValueTypeNames.Float).Set(0.25)
  material.CreateDisplacementOutput('mitsuba').ConnectToSource(
      disp.ConnectableAPI(), 'out'
  )
  _bind_material(mesh, material)

  aovs = _render_aovs(stage)
  x, y = aovs['position'][..., 0], aovs['position'][..., 1]
  # Object-space offset +0.25 scaled by 2.0 along Z gives z = 0.5.
  _assert_close(aovs['position'][..., 2], 0.5)
  _assert_close(aovs['geo_normal'], (0.0, 0.0, 1.0))
  _assert_close(aovs['sh_normal'], (0.0, 0.0, 1.0))
  _assert_close(aovs['uv'], np.stack([x, 1.0 - y], axis=-1))


def test_varying_normals():
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  _set_primvar(mesh, 'st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex', [])
  _render_aovs(stage)

  normal = tuple(Gf.Vec3f(0.3, -0.2, 1.0).GetNormalized())
  mesh.CreateNormalsAttr([normal] * len(mesh.GetPointsAttr().Get()))
  mesh.SetNormalsInterpolation(UsdGeom.Tokens.varying)
  _set_vertex_uvs(mesh)

  aovs = _render_aovs(stage)
  x, y = aovs['position'][..., 0], aovs['position'][..., 1]
  _assert_close(aovs['sh_normal'], normal)
  _assert_close(aovs['uv'], np.stack([x, 1.0 - y], axis=-1))


# -----------------------------------------------------------------------------
# Primvar interpolation, indexing, UV convention, and UsdPrimvarReader
# -----------------------------------------------------------------------------


def test_uv_v_flip_only_applies_to_mesh_texcoords():
  """Mitsuba mesh UVs flip `v`, while `UsdPrimvarReader` reads authored `st`."""
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  _set_vertex_uvs(mesh)
  _bind_material(
      mesh,
      _create_primvar_material(stage, '/Material', 'st', reader_type='float2'),
  )

  aovs = _render_aovs(stage)
  x, y = aovs['position'][..., 0], aovs['position'][..., 1]
  _assert_close(aovs['uv'], np.stack([x, 1.0 - y], axis=-1))
  _assert_close(aovs['albedo'], np.stack([x, y, np.zeros_like(x)], axis=-1))


# Under subdivision, indexed and flattened `faceVarying` primvars only match
# with linear `'all'` interpolation, because smooth `faceVarying` subdivision
# uses the index buffer to detect shared seam topology.
@pytest.mark.parametrize(
    'subdivision_level, fvar_interpolation',
    [(0, 'cornersPlus1'), (2, 'all')],
)
def test_indexed_primvars_match_unindexed(
    subdivision_level: int, fvar_interpolation: str
):
  stage = _create_stage()
  mesh = _create_grid(
      stage, resolution=2, subdivision_level=subdivision_level, bumpy=True
  )
  mesh.CreateFaceVaryingLinearInterpolationAttr(fvar_interpolation)

  points = mesh.GetPointsAttr().Get()
  face_vertex_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
  vertex_uvs = [(p[0], p[1]) for p in points]

  # Left faces (0, 2) and right faces (1, 3) use distinct palette entries.
  color_palette = [(0.8, 0.2, 0.1), (0.1, 0.5, 0.8)]
  normal_palette = [(-0.3, 0.0, 1.0), (0.3, 0.0, 1.0)]
  half_indices = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
  _bind_material(mesh, _create_primvar_material(stage, '/Material', 'color'))
  _assert_close(_render_aovs(stage)['albedo'], (0.0, 0.0, 0.0))

  # 1. Unindexed (one value per face corner).
  _set_primvar(
      mesh,
      'st',
      Sdf.ValueTypeNames.TexCoord2fArray,
      UsdGeom.Tokens.faceVarying,
      [vertex_uvs[i] for i in face_vertex_indices],
  )
  _set_primvar(
      mesh,
      'color',
      Sdf.ValueTypeNames.Color3fArray,
      UsdGeom.Tokens.faceVarying,
      [color_palette[i] for i in half_indices],
  )
  _set_primvar(
      mesh,
      'normals',
      Sdf.ValueTypeNames.Normal3fArray,
      UsdGeom.Tokens.faceVarying,
      [normal_palette[i] for i in half_indices],
  )
  unindexed = _render_aovs(stage)

  # 2. Indexed (compact value table + explicit per-corner indices).
  _set_primvar(
      mesh,
      'st',
      Sdf.ValueTypeNames.TexCoord2fArray,
      UsdGeom.Tokens.faceVarying,
      vertex_uvs,
      indices=face_vertex_indices,
  )
  _set_primvar(
      mesh,
      'color',
      Sdf.ValueTypeNames.Color3fArray,
      UsdGeom.Tokens.faceVarying,
      color_palette,
      indices=half_indices,
  )
  _set_primvar(
      mesh,
      'normals',
      Sdf.ValueTypeNames.Normal3fArray,
      UsdGeom.Tokens.faceVarying,
      normal_palette,
      indices=half_indices,
  )
  _assert_all_aovs_close(_render_aovs(stage), unindexed)


@pytest.mark.parametrize('subdivision_level', [0, 2])
@pytest.mark.parametrize(
    'interpolation',
    ['constant', 'uniform', 'vertex', 'varying', 'faceVarying'],
)
def test_primvar_interpolation(interpolation: str, subdivision_level: int):
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2, subdivision_level=subdivision_level)
  mesh.CreateFaceVaryingLinearInterpolationAttr('all')
  _bind_material(mesh, _create_primvar_material(stage, '/Material', 'color'))

  points = mesh.GetPointsAttr().Get()
  face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
  quadrant_colors = [
      (0.8, 0.1, 0.1),  # face 0: bottom-left
      (0.1, 0.8, 0.1),  # face 1: bottom-right
      (0.1, 0.2, 0.8),  # face 2: top-left
      (0.8, 0.8, 0.1),  # face 3: top-right
  ]
  values_by_interpolation = {
      'constant': [(0.25, 0.5, 0.75)],
      'uniform': quadrant_colors,
      'vertex': [(p[0], p[1], 0.5) for p in points],
      'varying': [(p[0], p[1], 0.5) for p in points],
      'faceVarying': [
          (points[i][0], points[i][1], 0.5) for i in face_vertex_indices
      ],
  }
  _set_primvar(
      mesh,
      'color',
      Sdf.ValueTypeNames.Color3fArray,
      interpolation,
      values_by_interpolation[interpolation],
  )

  aovs = _render_aovs(stage)
  albedo = aovs['albedo']
  if interpolation == 'constant':
    _assert_close(albedo, (0.25, 0.5, 0.75))
  elif interpolation == 'uniform':
    _assert_close(albedo[_HALF:, :_HALF], quadrant_colors[0])
    _assert_close(albedo[_HALF:, _HALF:], quadrant_colors[1])
    _assert_close(albedo[:_HALF, :_HALF], quadrant_colors[2])
    _assert_close(albedo[:_HALF, _HALF:], quadrant_colors[3])
  else:
    x, y = aovs['position'][..., 0], aovs['position'][..., 1]
    _assert_close(albedo, np.stack([x, y, np.full_like(x, 0.5)], axis=-1))


def test_scalar_constant_st():
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  _render_aovs(stage)

  _set_primvar(
      mesh,
      'st',
      Sdf.ValueTypeNames.TexCoord2f,
      UsdGeom.Tokens.constant,
      Gf.Vec2f(0.25, 0.75),
  )
  _assert_close(_render_aovs(stage)['uv'], (0.25, 0.25))


@pytest.mark.parametrize(
    'reader_type, type_name, authored, expected_rgb',
    [
        # Scalars (both array and non-array constant primvars) broadcast to RGB.
        ('float', 'FloatArray', 0.25, (0.25, 0.25, 0.25)),
        ('float', 'Float', 0.25, (0.25, 0.25, 0.25)),
        ('float', 'BoolArray', True, (1.0, 1.0, 1.0)),
        ('float', 'Bool', True, (1.0, 1.0, 1.0)),
        ('int', 'IntArray', 1, (1.0, 1.0, 1.0)),
        ('int', 'Int', 1, (1.0, 1.0, 1.0)),
        # 2D values pad the third channel with zero; 4D values drop the fourth.
        ('float2', 'Float2Array', (0.25, 0.75), (0.25, 0.75, 0.0)),
        ('float2', 'Float2', (0.25, 0.75), (0.25, 0.75, 0.0)),
        ('float3', 'Color3fArray', (0.25, 0.5, 0.75), (0.25, 0.5, 0.75)),
        ('float3', 'Color3f', (0.25, 0.5, 0.75), (0.25, 0.5, 0.75)),
        ('float4', 'Float4Array', (0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75)),
        ('float4', 'Float4', (0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75)),
        ('normal', 'Normal3fArray', (0.25, 0.5, 0.75), (0.25, 0.5, 0.75)),
        ('point', 'Point3fArray', (0.25, 0.5, 0.75), (0.25, 0.5, 0.75)),
        ('vector', 'Vector3fArray', (0.25, 0.5, 0.75), (0.25, 0.5, 0.75)),
    ],
)
def test_primvar_reader_types(
    reader_type: str,
    type_name: str,
    authored: object,
    expected_rgb: tuple[float, float, float],
):
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  sdf_type = getattr(Sdf.ValueTypeNames, type_name)
  _set_primvar(
      mesh,
      'pv',
      sdf_type,
      UsdGeom.Tokens.constant,
      [authored] if sdf_type.isArray else authored,
  )
  _bind_material(
      mesh, _create_primvar_material(stage, '/Material', 'pv', reader_type)
  )
  _assert_close(_render_aovs(stage)['albedo'], expected_rgb)


@pytest.mark.parametrize(
    'reader_type, type_name',
    [
        ('point', 'Point3fArray'),
        ('normal', 'Normal3fArray'),
        ('vector', 'Vector3fArray'),
    ],
)
def test_primvar_reader_returns_object_space_values(
    reader_type: str, type_name: str
):
  """User primvars with spatial roles keep their authored object-space values."""
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  # Rotate in XY around `(0.5, 0.5)` and scale Z so any world-space point,
  # normal, or vector transform would alter `(0.25, 0.5, 0.75)`.
  to_origin = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-0.5, -0.5, 0.0))
  rot_scale = Gf.Matrix4d().SetScale(
      Gf.Vec3d(1.0, 1.0, 2.0)
  ) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
  mesh.AddTransformOp().Set(to_origin * rot_scale * to_origin.GetInverse())

  authored = (0.25, 0.5, 0.75)
  _set_primvar(
      mesh,
      'pv',
      getattr(Sdf.ValueTypeNames, type_name),
      UsdGeom.Tokens.constant,
      [authored],
  )
  _bind_material(
      mesh, _create_primvar_material(stage, '/Material', 'pv', reader_type)
  )
  _assert_close(_render_aovs(stage)['albedo'], authored)


def test_primvar_reader_with_preview_surface():
  """`UsdPreviewSurface` reads primvars (including interface-wired `varname`)."""
  stage = _create_stage()
  mesh = _create_grid(stage, resolution=2)
  authored = (0.2, 0.4, 0.6)
  _set_primvar(
      mesh,
      'coat_tint',
      Sdf.ValueTypeNames.Color3fArray,
      UsdGeom.Tokens.constant,
      [authored],
  )

  material = UsdShade.Material.Define(stage, '/Material')
  varname_input = material.CreateInput('primvarName', Sdf.ValueTypeNames.Token)
  varname_input.Set('coat_tint')

  surface = UsdShade.Shader.Define(stage, '/Material/PreviewSurface')
  surface.CreateIdAttr('UsdPreviewSurface')
  surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(1.0)
  reader = UsdShade.Shader.Define(stage, '/Material/Reader')
  reader.CreateIdAttr('UsdPrimvarReader_float3')
  reader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(
      varname_input
  )
  surface.CreateInput(
      'diffuseColor', Sdf.ValueTypeNames.Color3f
  ).ConnectToSource(reader.ConnectableAPI(), 'result')
  material.CreateSurfaceOutput().ConnectToSource(
      surface.ConnectableAPI(), 'surface'
  )
  _bind_material(mesh, material)

  _assert_close(_render_aovs(stage)['albedo'], authored)

  # Pointing `varname` at a missing primvar evaluates to zero in both backends.
  varname_input.Set('missing_primvar')
  _assert_close(_render_aovs(stage)['albedo'], (0.0, 0.0, 0.0))

  # `UsdPrimvarReader_normal` connected to `inputs:normal` is re-encoded from
  # `[-1, 1]` tangent space to `[0, 1]`.
  _set_vertex_uvs(mesh)
  n_tangent = (-0.6, 0.0, 0.8)
  _set_primvar(
      mesh,
      't_normal',
      Sdf.ValueTypeNames.Normal3fArray,
      UsdGeom.Tokens.constant,
      [n_tangent],
  )
  normal_reader = UsdShade.Shader.Define(stage, '/Material/NormalReader')
  normal_reader.CreateIdAttr('UsdPrimvarReader_normal')
  normal_reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('t_normal')
  surface.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(
      normal_reader.ConnectableAPI(), 'result'
  )
  _assert_close(_render_aovs(stage)['sh_normal'], n_tangent)
