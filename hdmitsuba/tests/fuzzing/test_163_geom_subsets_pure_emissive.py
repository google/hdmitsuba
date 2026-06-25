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

"""Fuzz testing: intensive emissive material rebinding modification on geometry subsets in USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Gf  # type: ignore
from pxr import Sdf  # type: ignore
from pxr import Usd  # type: ignore
from pxr import UsdShade  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_geom_subsets_pure_emissive():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/geom_subsets.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_geom_subsets_pure_emissive_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_geom_subsets_pure_emissive_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Define emissive material and bind to red subset
  em_mat = UsdShade.Material.Define(stage, '/root/_materials/EmissiveSubset')
  em_shd = UsdShade.Shader.Define(
      stage, '/root/_materials/EmissiveSubset/Shd'
  )
  em_shd.CreateIdAttr('UsdPreviewSurface')
  em_shd.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
      Gf.Vec3f(2.0, 0.2, 0.2)
  )
  em_mat.CreateSurfaceOutput().ConnectToSource(
      em_shd.CreateOutput('surface', Sdf.ValueTypeNames.Token)
  )

  subset_red = stage.GetPrimAtPath('/root/Cube/Cube/red')
  UsdShade.MaterialBindingAPI(subset_red).Bind(em_mat)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_geom_subsets_pure_emissive_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_geom_subsets_pure_emissive_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
