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

"""Parsing of USD render settings for Mitsuba."""

from __future__ import annotations

from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdRender

_SUPPORTED_AOVS = frozenset([
    'color',
    'albedo',
    'depth',
    'position',
    'uv',
    'geo_normal',
    'sh_normal',
    'dp_du',
    'dp_dv',
    'duv_dx',
    'duv_dy',
    'prim_index',
    'shape_index',
])
_AOV_ALIASES = {'normal': 'sh_normal', 'raw': 'color'}


def _parse_aov_var(
    var_path: Sdf.Path, stage: Usd.Stage
) -> tuple[bool, str] | None:
  """Parses a single USD render var to a Mitsuba AOV."""
  var_prim = stage.GetPrimAtPath(var_path)
  render_var = UsdRender.Var(var_prim)
  if not render_var:
    return None

  var_name = var_prim.GetName()
  data_type = render_var.GetDataTypeAttr().Get()
  source_name = render_var.GetSourceNameAttr().Get()
  source_name = _AOV_ALIASES.get(source_name, source_name)
  if source_name not in _SUPPORTED_AOVS:
    Tf.Warn(
        f"  [Var] Name: {var_name} | Type: {data_type} |"
        f" Source: {source_name} (unsupported, skipping)"
    )
    return None

  if source_name == 'color':
    return True, ''

  return False, f'{var_name}:{source_name}'


def parse_aovs(
    stage: Usd.Stage,
    render_settings: UsdRender.RenderSettings,
    render_product_path: str | None = None,
) -> tuple[bool, list[str]]:
  """Parses the AOV buffers from the USD stage.

  Args:
    stage: The USD stage to parse.
    render_settings: The render settings to parse.
    render_product_path: Optional path to the render product to use. If None,
      uses the first valid render product.

  Returns:
    A tuple of (render_color, aovs), where render_color is a boolean
    indicating whether the color buffer should be rendered, and aovs is a list
    of AOV buffers to render. The aovs are in the format 'name:source'.
  """
  products_rel = render_settings.GetProductsRel()
  product_targets = products_rel.GetTargets()
  if not product_targets:
    return True, []

  # If a render product is specified, validate it and use it exclusively.
  if render_product_path is not None:
    target_product_path = next(
        (p for p in product_targets if p.pathString == render_product_path),
        None,
    )
    if target_product_path is None:
      raise ValueError(
          f'Invalid render product path {render_product_path} not found in'
          ' render settings.'
      )
    product_targets = [target_product_path]
  elif len(product_targets) > 1:
    Tf.Warn(
        "Multiple render products are not yet supported. Only the first one"
        " will be used."
    )

  render_color = False
  aovs = []

  for product_path in product_targets:
    render_product = UsdRender.Product(stage.GetPrimAtPath(product_path))
    if not render_product:
      continue
    var_targets = render_product.GetOrderedVarsRel().GetTargets()
    if not var_targets:
      continue

    for var_path in var_targets:
      result = _parse_aov_var(var_path, stage)
      if result is None:
        continue
      is_color, aov_name = result
      if is_color:
        render_color = True
      else:
        aovs.append(aov_name)
    return render_color, aovs

  return True, []


def get_render_settings(stage: Usd.Stage) -> UsdRender.Settings | None:
  """Returns the render settings prim from the stage.

  This function attempts to find the render settings prim as follows:
  1. Use the official USD API to check the stage's renderSettingsPrimPath.
  2. Fall back to scanning the standard /Render scope.
  3. Fall back to a global stage traversal if conventions weren't followed.

  Args:
      stage: The USD stage to get the render settings prim from.

  Returns:
      The UsdRender.Settings schema instance if found, otherwise None.
  """

  # 1. Primary check using the native USD schema API.
  render_settings = UsdRender.Settings.GetStageRenderSettings(stage)
  if render_settings:
    return render_settings

  # 2. Optimized Fallback: Check the standard pipeline /Render scope first.
  render_root = stage.GetPrimAtPath("/Render")
  if render_root:
    for prim in Usd.PrimRange(render_root):
      if prim.IsA(UsdRender.Settings):
        return UsdRender.Settings(prim)

  # 3. Last Resort Fallback: Expensive full stage traversal.
  for prim in stage.Traverse():
    if prim.IsA(UsdRender.Settings):
      return UsdRender.Settings(prim)
  return None
