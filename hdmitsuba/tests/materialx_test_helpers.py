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
import pytest

from hdmitsuba.tests import test_helpers
from usd_mitsuba import material as usd_mitsuba_material
import usd_render

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
}


def define_quad(
    stage: Usd.Stage, path: str, half_size: float = 0.8
) -> UsdGeom.Mesh:
  """Defines a Z-facing UV-mapped quad mesh at `path`.

  Args:
    stage: The USD stage in which to define the mesh.
    path: USD prim path for the quad mesh.
    half_size: Half-width and half-height of the quad in XY.

  Returns:
    The defined `UsdGeom.Mesh` prim.
  """
  uvs = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
  mesh = UsdGeom.Mesh.Define(stage, path)
  mesh.CreatePointsAttr(np.pad((2.0 * uvs - 1.0) * half_size, ((0, 0), (0, 1))))
  mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
  mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  mesh.CreateFaceVertexCountsAttr([4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
  ).Set(uvs)
  return mesh


def create_base_scene(
    with_lights: bool = True,
    quad_half_size: float = 0.8,
    resolution: tuple[int, int] = (64, 64),
    spp: int = 32,
    mesh_path: str = '/mesh',
) -> tuple[Usd.Stage, UsdGeom.Mesh]:
  """Creates an in-memory USD stage with a UV-mapped quad mesh, camera, and lights.

  Args:
    with_lights: Whether to create default dome and sphere lights.
    quad_half_size: Half-width and half-height of the quad facing the camera.
    resolution: Output image `(width, height)` in pixels.
    spp: Samples per pixel in render settings.
    mesh_path: USD prim path for the created mesh.

  Returns:
    Tuple of (Usd.Stage, UsdGeom.Mesh) for the main test mesh.
  """
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  mesh = define_quad(stage, mesh_path, quad_half_size)
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
    target: usd_render.RenderEngine | np.ndarray, half_size: int = 4
) -> np.ndarray:
  """Renders `target` (or takes a rendered image) and returns the mean central RGB.

  Args:
    target: A configured `RenderEngine` or rendered image array.
    half_size: Half-width in pixels of the center crop averaged over.

  Returns:
    A 3-element numpy array containing the mean RGB color of the center patch.
  """
  img = (
      target[..., :3]
      if isinstance(target, np.ndarray)
      else np.array(target.render()['color'])[..., :3]
  )
  cy, cx = img.shape[0] // 2, img.shape[1] // 2
  return np.mean(
      img[cy - half_size : cy + half_size, cx - half_size : cx + half_size],
      axis=(0, 1),
  )


def _is_shader_spec(val: Any) -> bool:
  return (
      isinstance(val, tuple)
      and len(val) in (2, 3)
      and isinstance(val[0], str)
      and isinstance(val[1], dict)
      and (len(val) == 2 or isinstance(val[2], str))
  )


def _resolve_shader_spec(
    stage: Usd.Stage, path: str, val: Any
) -> UsdShade.Output:
  """Defines a shader from a `(id, inputs[, output])` tuple and returns its output."""
  return define_shader(stage, path, val[0], **val[1]).CreateOutput(
      val[2] if len(val) == 3 else 'out', Sdf.ValueTypeNames.Token
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
      objects, or nested `(node_id, input_dict[, output_name])` tuples.

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
      val = _resolve_shader_spec(stage, f'{full_path}/{name}', val)
    if isinstance(val, UsdShade.Shader):
      val = val.CreateOutput('out', Sdf.ValueTypeNames.Token)
    if isinstance(val, UsdShade.Output):
      shader.CreateInput(name, val.GetTypeName()).ConnectToSource(val)
      continue
    if isinstance(val, tuple):
      sdf_type = _TUPLE_SDF_TYPES[len(val)]
    else:
      sdf_type = _SDF_TYPES.get(type(val), Sdf.ValueTypeNames.Float)
    shader.CreateInput(name, sdf_type).Set(val)
  return shader


def _author_mtlx_material(
    stage: Usd.Stage,
    mat_path: str,
    shader_id: str,
    **inputs: Any,
) -> tuple[UsdShade.Material, UsdShade.Shader]:
  """Authors a MaterialX surface material at `mat_path`."""
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
  material, shader = _author_mtlx_material(
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
  material, _ = _author_mtlx_material(
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
    emission_color: Value, `UsdShade.Shader`/`Output`, or
      `(node_id, input_dict[, output_name])` tuple to plug into
      `ND_surface_unlit.emission_color`.
    stage: Optional USD stage to reuse.
    si: Optional `mi.SurfaceInteraction3f` (1-point or `N`-lane batch).
    channels: Number of output channels to return (`1`..`4`).

  Returns:
    Float64 numpy array of shape `(channels,)` for 1-point `si`, or `(N,)` /
    `(N, channels)` for an `N`-lane batched `si`.
  """
  stage = stage or Usd.Stage.CreateInMemory()
  if _is_shader_spec(emission_color):
    emission_color = _resolve_shader_spec(stage, 'ExprNode', emission_color)
  if si is None:
    si = make_test_si(wi=(0.0, 0.0, 1.0))

  n_lanes = max(dr.width(si.p), dr.width(si.uv), dr.width(si.wi), 1)

  def _eval(color: Any) -> np.ndarray:
    _, emitter = compile_mtlx_material(
        'ND_surface_unlit', stage=stage, emission_color=color
    )
    if emitter is None:
      return np.zeros((n_lanes, 3))
    c = emitter.eval(si)
    return np.column_stack([
        np.broadcast_to(np.asarray(c[i], dtype=np.float64), (n_lanes,))
        for i in range(3)
    ])

  rgb = _eval(emission_color)
  if channels == 4:
    alpha = define_shader(
        stage, 'AlphaExtract', 'ND_extract_color4', in_=emission_color, index=3
    )
    out = np.column_stack([rgb, _eval(alpha)[:, 0]])
  else:
    out = rgb[:, :channels]
  if n_lanes == 1:
    return out[0]
  return out[:, 0] if channels == 1 else out


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


def enable_materialx_and_default_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Enables MaterialX translation and sets the default variant.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
  """
  # TODO(vicini): Remove `HDMITSUBA_ENABLE_MATERIALX` once MaterialX support is
  # complete.
  monkeypatch.setenv('HDMITSUBA_ENABLE_MATERIALX', '1')
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def set_variant_or_skip(*variants: str) -> None:
  """Sets the first available Mitsuba variant from `variants`, or skips the test.

  Args:
    *variants: Candidate Mitsuba variant names in priority order.
  """
  try:
    mi.set_variant(*variants)
  except ImportError as error:
    pytest.skip(f'Variants {variants} not available: {error}')
