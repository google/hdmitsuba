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

"""Helper functions for testing."""

from __future__ import annotations

import os
from typing import Any

import mitsuba as mi
import numpy as np

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdRender

import usd_render
from usd_mitsuba import translator as usd_mitsuba

_RENDER_SETTINGS_PATH = '/Render/PrimarySettings'

TEST_ASSETS_PATH = 'hdmitsuba/test_assets'
OUTPUTS_DIR = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', 'test_outputs')


def create_render_settings(
    stage: Usd.Stage,
    resolution: tuple[int, int] = (200, 140),
    spp: int = 128,
    color_data_type: str | None = None,
) -> UsdRender.Settings:
  """Creates a render settings prim in the USD stage.

  Args:
    stage: The USD stage to create the render settings in.
    resolution: The resolution of the render settings.
    spp: The sample count (SPP) override.
    color_data_type: If set, adds a color RenderVar with this data type (e.g.,
      "color4f" to request an alpha channel).

  Returns:
    The render settings prim.
  """
  render_settings = UsdRender.Settings.Define(stage, _RENDER_SETTINGS_PATH)
  render_settings_prim = render_settings.GetPrim()
  render_settings_prim.CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())
  render_settings_prim.CreateAttribute(
      'mitsuba:sample_count', Sdf.ValueTypeNames.Int
  ).Set(spp)
  render_settings_prim.CreateAttribute(
      UsdRender.Tokens.resolution, Sdf.ValueTypeNames.Int2
  ).Set(Gf.Vec2i(*resolution))
  if color_data_type:
    product = UsdRender.Product.Define(stage, '/Render/Product')
    color_var = UsdRender.Var.Define(stage, '/Render/Vars/color')
    color_var.CreateDataTypeAttr(color_data_type)
    color_var.CreateSourceNameAttr('color')
    product.CreateOrderedVarsRel().SetTargets([color_var.GetPath()])
    render_settings.CreateProductsRel().SetTargets([product.GetPath()])

  # Set the stage-level settings metadata. For this, temporarily set the edit
  # target to the root layer and restore the previous target afterwards.
  previous_target = stage.GetEditTarget()
  stage.SetEditTarget(stage.GetRootLayer())
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, _RENDER_SETTINGS_PATH
  )
  stage.SetEditTarget(previous_target)
  return render_settings


def robust_assert_close(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.99,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> None:
  """Robust array assert allclose check. Ignores outliers according to quantile.

  Args:
    x: The array to check.
    y: The array to check against.
    quantile: The quantile to use to reject outliers.
    atol: The absolute tolerance.
    rtol: The relative tolerance.
  """
  error_map = np.abs(x - y)
  x_copy = np.array(x)
  mask = error_map > np.quantile(error_map, quantile)
  x_copy[mask] = y[mask]
  np.testing.assert_allclose(x_copy, y, atol=atol, rtol=rtol)


def write_image(image: np.ndarray, filename: str) -> None:
  bmp = mi.Bitmap(image)
  if filename.endswith('.png'):
    bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
  os.makedirs(OUTPUTS_DIR, exist_ok=True)
  bmp.write(os.path.join(OUTPUTS_DIR, filename))


def assert_hydra_equal_to_offline(
    stage: Usd.Stage,
    camera_path: str | None = None,
    output_prefix: str = 'test',
    spp: int = 128,
    atol: float = 0.05,
    rtol: float = 1e-3,
    engine: Any = None,
    subdivision_level: int = 1,
    refine_level_fallback: int | None = None,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> tuple[np.ndarray, np.ndarray]:
  """Asserts that rendering with Hydra delegate matches offline translation.

  This runs the HdMitsuba delegate render, translates the stage using the
  offline translator, renders it via Mitsuba directly, and compares the
  results. It also writes both images to the output directory.

  Args:
    stage: The USD stage to render.
    camera_path: The camera path. If None, it will use the default camera.
    output_prefix: The prefix for written images.
    spp: The samples per pixel for the offline render.
    atol: Absolute tolerance for comparison.
    rtol: Relative tolerance for comparison.
    engine: Optional existing engine to reuse.
    subdivision_level: Subdivision level for offline translator.
    refine_level_fallback: Refine level fallback for Hydra delegate.
    time: The time code to render at.

  Returns:
    A tuple containing (image_hd, image_usd).
  """

  if engine is None:
    engine = usd_render.RenderEngine(stage)
    configure_args: dict[str, Any] = {
        'hydra_delegate_id': 'HdMitsubaRendererPlugin',
    }
    if camera_path is not None:
      configure_args['camera_path'] = camera_path
    if refine_level_fallback is not None:
      configure_args['refine_level_fallback'] = refine_level_fallback
    engine.configure(**configure_args)

  outputs = engine.render(time_code=time)
  image_hd = outputs['color']

  scene_dict = usd_mitsuba.convert_to_mitsuba(
      stage, time=time, subdivision_level=subdivision_level
  )
  scene = mi.load_dict(scene_dict)
  image_usd = np.array(mi.render(scene, spp=spp))

  write_image(image_hd, f'{output_prefix}_hd.png')
  write_image(image_usd, f'{output_prefix}_usd.png')

  robust_assert_close(image_hd[..., :3], image_usd, atol=atol, rtol=rtol)
  return image_hd, image_usd


# USD render var data types of the AOVs rendered by `render_aovs`.
_AOV_DATA_TYPES = {
    'albedo': 'color3f',
    'depth': 'float',
    'dp_du': 'vector3f',
    'dp_dv': 'vector3f',
    'geo_normal': 'normal3f',
    'position': 'point3f',
    'prim_index': 'int',
    'sh_normal': 'normal3f',
    'shape_index': 'int',
    'uv': 'float2',
}


def create_aov_render_settings(
    stage: Usd.Stage,
    aovs: tuple[str, ...],
    resolution: tuple[int, int] = (32, 32),
) -> UsdRender.Settings:
  """Creates render settings with a single render product holding `aovs`.

  Each AOV is rendered into a render var of the same name. Each pixel holds a
  single sample, so that the AOVs are point samples of the visible surface.

  Args:
    stage: The USD stage to create the render settings in.
    aovs: The Mitsuba AOVs to render, e.g. 'position' or 'uv'.
    resolution: The image resolution.

  Returns:
    The render settings prim.
  """
  render_settings = create_render_settings(stage, resolution, spp=1)
  product = UsdRender.Product.Define(stage, '/Render/AovProduct')
  var_paths = []
  for aov in aovs:
    var = UsdRender.Var.Define(stage, f'/Render/Vars/{aov}')
    var.CreateDataTypeAttr(_AOV_DATA_TYPES[aov])
    var.CreateSourceNameAttr(aov)
    var_paths.append(var.GetPath())
  product.CreateOrderedVarsRel().SetTargets(var_paths)
  render_settings.CreateProductsRel().SetTargets([product.GetPath()])
  return render_settings


def render_aovs(
    stage: Usd.Stage,
    backend: str,
    camera_path: str,
    engine: usd_render.RenderEngine | None = None,
) -> dict[str, np.ndarray]:
  """Renders the AOVs of the render settings of `stage`.

  The render settings are expected to be created by
  `create_aov_render_settings`.

  Args:
    stage: The USD stage to render.
    backend: 'hydra' to render with the Hydra delegate, or 'offline' to render
      the scene translated by `usd_mitsuba`.
    camera_path: The path of the camera to render.
    engine: Optional configured Hydra `RenderEngine` to reuse across stage edits.

  Returns:
    The AOV images, as (height, width, channels) arrays keyed by AOV name.
  """
  if backend == 'hydra':
    if engine is None:
      engine = usd_render.RenderEngine(stage)
      engine.configure(
          hydra_delegate_id='HdMitsubaRendererPlugin', camera_path=camera_path
      )
    return {
        name: np.atleast_3d(np.array(image, dtype=np.float32))
        for name, image in engine.render().items()
    }

  assert backend == 'offline', backend
  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  mi.render(scene)
  # Skips the '<root>' bitmap holding the color channels, like Hydra.
  return {
      name: np.atleast_3d(np.array(bitmap, dtype=np.float32))
      for name, bitmap in scene.sensors()[0].film().bitmap().split()
      if name != '<root>'
  }
