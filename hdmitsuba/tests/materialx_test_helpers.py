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

"""Shared helpers for MaterialX tests in hdMitsuba."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdShade
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import material as usd_mitsuba_material
from usd_mitsuba import translator as usd_mitsuba

ALBEDO_PATH = f'{test_helpers.TEST_ASSETS_PATH}/materials/textures/albedo.png'
NORMAL_PATH = (
    f'{test_helpers.TEST_ASSETS_PATH}/materials/textures/normalmap.png'
)

_SDF_TYPES = {
    Sdf.AssetPath: Sdf.ValueTypeNames.Asset,
    bool: Sdf.ValueTypeNames.Bool,
    int: Sdf.ValueTypeNames.Int,
    str: Sdf.ValueTypeNames.String,
    Gf.Vec2f: Sdf.ValueTypeNames.Float2,
    Gf.Vec3f: Sdf.ValueTypeNames.Vector3f,
    Gf.Matrix3d: Sdf.ValueTypeNames.Matrix3d,
    Gf.Matrix4d: Sdf.ValueTypeNames.Matrix4d,
}
_TUPLE_SDF_TYPES = {
    2: Sdf.ValueTypeNames.Float2,
    3: Sdf.ValueTypeNames.Color3f,
    4: Sdf.ValueTypeNames.Color4f,
    9: Sdf.ValueTypeNames.Matrix3d,
    16: Sdf.ValueTypeNames.Matrix4d,
}


def create_base_scene(
    with_lights: bool = True,
    quad_half_size: float | None = None,
    resolution: tuple[int, int] = (64, 64),
    spp: int = 32,
    mesh_path: str = '/mesh',
) -> tuple[Usd.Stage, UsdGeom.Mesh]:
  """Creates an in-memory USD stage with a UV-mapped mesh, camera, and lights.

  Args:
    with_lights: Whether to create default dome and sphere lights.
    quad_half_size: If set, the mesh is a quad of this half size facing the
      camera instead of a sphere.
    resolution: Output image `(width, height)` in pixels.
    spp: Samples per pixel in render settings.
    mesh_path: USD prim path for the created mesh.

  Returns:
    Tuple of (Usd.Stage, UsdGeom.Mesh) for the main test mesh.
  """
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)

  mesh = UsdGeom.Mesh.Define(stage, mesh_path)
  if quad_half_size is None:
    # Create a quad sphere mesh with explicit UVs and normals
    n_lat, n_lon = 12, 24
    uu, vv = np.meshgrid(
        np.linspace(0.0, 1.0, n_lon + 1, dtype=np.float32),
        np.linspace(0.0, 1.0, n_lat + 1, dtype=np.float32),
    )
    theta, phi = vv * np.pi, uu * (2.0 * np.pi)
    xyz = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.cos(theta),
            np.sin(theta) * np.sin(phi),
        ],
        axis=-1,
    ).reshape(-1, 3)
    normals = xyz
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing='ij')
    p0 = (ii * (n_lon + 1) + jj).ravel()
    quads = np.stack([p0, p0 + 1, p0 + n_lon + 2, p0 + n_lon + 1], axis=-1)
  else:
    uvs = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    xyz = np.pad((2.0 * uvs - 1.0) * quad_half_size, ((0, 0), (0, 1)))
    normals = [(0.0, 0.0, 1.0)] * 4
    quads = np.array([[0, 1, 2, 3]])

  mesh.CreatePointsAttr(xyz)
  mesh.CreateNormalsAttr(normals)
  mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  mesh.CreateFaceVertexCountsAttr([4] * len(quads))
  mesh.CreateFaceVertexIndicesAttr(quads.ravel())
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
  ).Set(uvs)

  UsdGeom.Camera.Define(stage, '/camera').AddTranslateOp().Set((0.0, 0.0, 2.8))

  if with_lights:
    key_light = UsdLux.SphereLight.Define(stage, '/keyLight')
    key_light.CreateIntensityAttr(15.0)
    key_light.CreateColorAttr((1.0, 1.0, 1.0))
    key_light.CreateRadiusAttr(0.5)
    key_light.AddTranslateOp().Set((2.0, 2.0, 2.5))

    dome_light = UsdLux.DomeLight.Define(stage, '/domeLight')
    dome_light.CreateIntensityAttr(0.4)
    dome_light.CreateColorAttr((1.0, 1.0, 1.0))

  test_helpers.create_render_settings(stage, resolution=resolution, spp=spp)
  return stage, mesh


def create_engine(stage: Usd.Stage) -> usd_render.RenderEngine:
  """Creates and configures an HdMitsuba RenderEngine for `stage`.

  Args:
    stage: The USD stage to render.

  Returns:
    A configured `usd_render.RenderEngine` instance.
  """
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
  )
  return engine


def render_center(
    target: Usd.Stage | usd_render.RenderEngine | np.ndarray, half_size: int = 8
) -> np.ndarray:
  """Renders `target` (or takes a rendered image) and returns the mean central RGB.

  Args:
    target: A USD stage, configured `RenderEngine`, or rendered image array.
    half_size: Half-width in pixels of the center crop averaged over.

  Returns:
    A 3-element numpy array containing the mean RGB color of the center patch.
  """
  if isinstance(target, np.ndarray):
    img = target[..., :3]
  else:
    engine = create_engine(target) if isinstance(target, Usd.Stage) else target
    img = np.array(engine.render()['color'])[..., :3]
  assert np.all(np.isfinite(img)), 'Rendered image contains NaN or Inf'
  c = img.shape[0] // 2
  half_size = min(half_size, max(c // 2, 1))
  return np.mean(
      img[c - half_size : c + half_size, c - half_size : c + half_size],
      axis=(0, 1),
  )


def _is_shader_spec(val: Any) -> bool:
  return (
      isinstance(val, tuple)
      and len(val) == 2
      and isinstance(val[0], str)
      and isinstance(val[1], dict)
  )


def define_shader(
    stage: Usd.Stage,
    path: str,
    shader_id: str,
    **inputs: Any,
) -> UsdShade.Shader:
  """Defines a MaterialX UsdShade.Shader and sets or connects its inputs.

  Args:
    stage: The USD stage in which to define the shader.
    path: Absolute prim path or path relative to `/mesh/Material`.
    shader_id: MaterialX NodeDef identifier (e.g.
      `ND_standard_surface_surfaceshader`).
    **inputs: Input parameter values, connected `UsdShade.Shader`/`Output`
      objects, or nested `(node_id, input_dict)` tuples.

  Returns:
    The newly defined `UsdShade.Shader` prim.
  """
  full_path = path if path.startswith('/') else f'/mesh/Material/{path}'
  if stage.GetPrimAtPath(full_path):
    stage.RemovePrim(full_path)
  shader = UsdShade.Shader.Define(stage, full_path)
  shader.CreateIdAttr(shader_id)
  for name, val in inputs.items():
    name = name.removesuffix('_')
    if _is_shader_spec(val):
      val = define_shader(stage, f'{full_path}/{name}', val[0], **val[1])
    if isinstance(val, UsdShade.Shader):
      val = val.CreateOutput('out', Sdf.ValueTypeNames.Token)
    if isinstance(val, UsdShade.Output):
      shader.CreateInput(name, val.GetTypeName()).ConnectToSource(val)
      continue
    if isinstance(val, list):
      val = tuple(val)
    if isinstance(val, tuple):
      sdf_type = _TUPLE_SDF_TYPES[len(val)]
      if len(val) == 9:
        val = Gf.Matrix3d(*val)
      elif len(val) == 16:
        val = Gf.Matrix4d(*val)
    else:
      sdf_type = _SDF_TYPES.get(type(val), Sdf.ValueTypeNames.Float)
    shader.CreateInput(name, sdf_type).Set(val)
  return shader


def _load_stage_hit(
    stage: Usd.Stage,
    p_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ray_origin: tuple[float, float, float] = (0.0, 0.0, 2.0),
) -> tuple[mi.Scene, mi.Shape, mi.SurfaceInteraction3f]:
  """Imports `stage` into Mitsuba and intersects a ray toward `p_target`."""
  scene = mi.load_dict(
      usd_mitsuba.convert_to_mitsuba(stage, subdivision_level=0)
  )
  origin = mi.Point3f(*ray_origin)
  si = scene.ray_intersect(
      mi.Ray3f(origin, dr.normalize(mi.Point3f(*p_target) - origin))
  )
  assert dr.all(si.is_valid()), f'Ray missed mesh toward {p_target}'
  return scene, scene.shapes()[0], si


def eval_unlit(
    stage: Usd.Stage,
    p_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ray_origin: tuple[float, float, float] = (0.0, 0.0, 2.0),
) -> np.ndarray:
  """Evaluates the mesh area emitter directly at the ray hit (zero MC noise).

  Args:
    stage: The USD stage containing the emissive mesh.
    p_target: Target 3D point toward which the ray is cast.
    ray_origin: World-space origin of the test ray.

  Returns:
    A 1D float64 numpy array with the evaluated emitter radiance.
  """
  _, shape, si = _load_stage_hit(
      stage, p_target=p_target, ray_origin=ray_origin
  )
  return np.asarray(shape.emitter().eval(si), dtype=np.float64).ravel()


def author_mtlx_material(
    stage: Usd.Stage,
    mat_path: str,
    shader_id: str,
    **inputs: Any,
) -> tuple[UsdShade.Material, UsdShade.Shader]:
  """Authors a MaterialX surface material at `mat_path`.

  Args:
    stage: The USD stage in which to define the material.
    mat_path: Prim path of the `UsdShade.Material`.
    shader_id: Surface shader NodeDef identifier.
    **inputs: Surface shader inputs.

  Returns:
    Tuple of `(UsdShade.Material, UsdShade.Shader)`.
  """
  material = UsdShade.Material.Define(stage, mat_path)
  shader = define_shader(stage, f'{mat_path}/Shader', shader_id, **inputs)
  material.CreateSurfaceOutput('mtlx').ConnectToSource(
      shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
  )
  return material, shader


def bind_mtlx_shader(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    shader_id: str,
    **inputs: Any,
) -> UsdShade.Shader:
  """Creates a MaterialX shader with the given inputs and binds it to `mesh`.

  Args:
    stage: The USD stage containing `mesh`.
    mesh: The target `UsdGeom.Mesh` prim.
    shader_id: Surface shader NodeDef identifier.
    **inputs: Surface shader inputs.

  Returns:
    The bound surface `UsdShade.Shader`.
  """
  material, shader = author_mtlx_material(
      stage,
      f'{mesh.GetPath()}/Material',
      shader_id,
      **inputs,
  )
  UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
  return shader


def compile_mtlx_material(
    shader_id: str,
    *,
    stage: Usd.Stage | None = None,
    **inputs: Any,
) -> tuple[mi.BSDF | None, mi.Emitter | None]:
  """Compiles a MaterialX surface shader in USD into a `(bsdf, emitter)` pair.

  Args:
    shader_id: Surface shader NodeDef identifier.
    stage: Optional USD stage to reuse.
    **inputs: Surface shader inputs.

  Returns:
    Tuple of `(mi.BSDF | None, mi.Emitter | None)`.
  """
  stage = stage or Usd.Stage.CreateInMemory()
  material, _ = author_mtlx_material(
      stage, '/mesh/Material', shader_id, **inputs
  )
  return usd_mitsuba_material.convert_material(material.GetPrim())[:2]


def eval_emission_expr(
    emission_color: Any,
    *,
    stage: Usd.Stage | None = None,
    si: mi.SurfaceInteraction3f | None = None,
    channels: int = 3,
) -> np.ndarray:
  """Evaluates a MaterialX expression plugged into an area light's emission.

  Args:
    emission_color: Value, `UsdShade.Shader`/`Output`, or `(node_id, kwargs)`
      tuple to plug into `ND_surface_unlit.emission_color`.
    stage: Optional USD stage to reuse.
    si: Optional `mi.SurfaceInteraction3f` at which to evaluate emission.
    channels: Number of output channels to return (`1`..`4`).

  Returns:
    A 1D float64 numpy array of length `channels`.
  """
  stage = stage or Usd.Stage.CreateInMemory()
  if _is_shader_spec(emission_color):
    emission_color = define_shader(
        stage, 'ExprNode', emission_color[0], **emission_color[1]
    )
  if si is None:
    si = make_test_si(wi=(0.0, 0.0, 1.0))

  def _eval(color: Any) -> np.ndarray:
    _, emitter = compile_mtlx_material(
        'ND_surface_unlit', stage=stage, emission_color=color
    )
    return (
        np.zeros(3, dtype=np.float64)
        if emitter is None
        else np.asarray(emitter.eval(si), dtype=np.float64).ravel()
    )

  rgb = _eval(emission_color)
  if channels < 4:
    return rgb[:channels]
  alpha = define_shader(
      stage, 'AlphaExtract', 'ND_extract_color4', in_=emission_color, index=3
  )
  return np.append(rgb, _eval(alpha)[0])


def make_test_si(
    wi: tuple[float, float, float] = (0.3, -0.4, 0.8660254),
) -> mi.SurfaceInteraction3f:
  """Creates a SurfaceInteraction3f with a canonical Z-up shading frame.

  Args:
    wi: Incident direction in the local shading frame (will be normalized).

  Returns:
    A `mi.SurfaceInteraction3f` initialized with `wi` and a Z-up frame.
  """
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.n = mi.Normal3f(0.0, 0.0, 1.0)
  si.sh_frame = mi.Frame3f(
      mi.Vector3f(1.0, 0.0, 0.0),
      mi.Vector3f(0.0, 1.0, 0.0),
      mi.Normal3f(0.0, 0.0, 1.0),
  )
  si.dp_du = mi.Vector3f(1.0, 0.0, 0.0)
  si.dp_dv = mi.Vector3f(0.0, 1.0, 0.0)
  si.wi = dr.normalize(mi.Vector3f(*wi))
  if mi.is_spectral:
    si.wavelengths = mi.Spectrum(450.0, 530.0, 610.0, 680.0)
  return si


def _is_noise_nodedef(nodedef: str) -> bool:
  return any(k in nodedef for k in ('noise', 'worley', 'fractal'))


def _parse_osl_input(val: str) -> Any:
  if val in ('true', 'false'):
    return val == 'true'
  if ',' in val:
    return tuple(float(x) for x in val.split(','))
  return float(val)


def check_osl_reference_nodes(include_noise: bool) -> None:
  """Verifies MaterialX library nodes against `osl_reference.json` via area-light emission.

  Args:
    include_noise: If True, tests procedural noise nodes; otherwise tests
      arithmetic, math, color, channel, and non-noise pattern nodes.
  """
  ref_path = (
      pathlib.Path(__file__).parent / 'materialx_reference/osl_reference.json'
  )
  reference = json.loads(ref_path.read_text())
  si = dr.zeros(mi.SurfaceInteraction3f)
  si.wi = mi.Vector3f(0.0, 0.0, 1.0)

  stage = Usd.Stage.CreateInMemory()
  failures = []
  for entry in reference['nodedefs']:
    nodedef = entry['nodedef']
    if _is_noise_nodedef(nodedef) != include_noise:
      continue
    tol = 5e-3 if nodedef == 'ND_blackbody' else 1e-4
    first_inputs = {
        k: _parse_osl_input(v) for k, v in entry['samples'][0]['inputs'].items()
    }
    node = define_shader(stage, 'Node', nodedef, **first_inputs)
    for sample in entry['samples']:
      for k, v in sample['inputs'].items():
        node.GetInput(k).Set(_parse_osl_input(v))
      for out_name, expected in sample['expected'].items():
        expected_arr = np.asarray(expected, dtype=np.float64)
        out = node.CreateOutput(out_name, Sdf.ValueTypeNames.Token)
        actual = eval_emission_expr(
            out, stage=stage, si=si, channels=len(expected_arr)
        )
        if not np.allclose(actual, expected_arr, rtol=tol, atol=tol):
          failures.append(
              f'{nodedef}.{out_name}({sample["inputs"]}): '
              f'expected {expected_arr}, got {actual}'
          )
          break
  assert not failures, 'OSL reference mismatches:\n' + '\n'.join(failures)
