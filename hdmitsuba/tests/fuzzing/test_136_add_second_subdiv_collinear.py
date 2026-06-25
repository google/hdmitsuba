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

"""Fuzz testing: dynamic instantiating of secondary subdivision mesh on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdShade  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_add_second_subdiv_collinear():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/subdiv_cube.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', refine_level_fallback=2
  )

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(
      usd_mitsuba.convert_to_mitsuba(stage, subdivision_level=2)
  )
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_add_second_subdiv_collinear_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_add_second_subdiv_collinear_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Clone subdiv mesh
  orig = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  cloned = UsdGeom.Mesh.Define(stage, '/root/CubeSubdivClone')
  cloned.GetPointsAttr().Set(orig.GetPointsAttr().Get())
  cloned.GetFaceVertexCountsAttr().Set(orig.GetFaceVertexCountsAttr().Get())
  cloned.GetFaceVertexIndicesAttr().Set(orig.GetFaceVertexIndicesAttr().Get())
  cloned.GetExtentAttr().Set(orig.GetExtentAttr().Get())
  cloned.GetSubdivisionSchemeAttr().Set('catmullClark')
  cloned.AddTranslateOp().Set((2.0, 2.0, 0.0))

  material = UsdShade.Material.Get(stage, '/root/_materials/Material')
  UsdShade.MaterialBindingAPI.Apply(cloned.GetPrim()).Bind(material)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(
      usd_mitsuba.convert_to_mitsuba(stage, subdivision_level=2)
  )
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_add_second_subdiv_collinear_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_add_second_subdiv_collinear_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
