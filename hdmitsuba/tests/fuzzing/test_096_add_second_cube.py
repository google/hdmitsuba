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

"""Fuzz testing: dynamic instantiating of secondary mesh on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
from pxr import UsdShade  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_add_second_cube():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_add_second_cube_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_add_second_cube_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Add a secondary cube mesh with extent, st, normals interpolation, and subdivisionScheme set
  orig_cube = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  new_cube = UsdGeom.Mesh.Define(stage, '/root/Cube2')
  new_cube.GetPointsAttr().Set(orig_cube.GetPointsAttr().Get())
  new_cube.GetFaceVertexCountsAttr().Set(
      orig_cube.GetFaceVertexCountsAttr().Get()
  )
  new_cube.GetFaceVertexIndicesAttr().Set(
      orig_cube.GetFaceVertexIndicesAttr().Get()
  )
  new_cube.GetNormalsAttr().Set(orig_cube.GetNormalsAttr().Get())
  new_cube.SetNormalsInterpolation(orig_cube.GetNormalsInterpolation())
  new_cube.GetExtentAttr().Set(orig_cube.GetExtentAttr().Get())
  new_cube.GetSubdivisionSchemeAttr().Set('none')

  st_attr = orig_cube.GetPrim().GetAttribute('primvars:st')
  if st_attr.IsValid():
    new_cube.GetPrim().CreateAttribute(
        'primvars:st', st_attr.GetTypeName()
    ).Set(st_attr.Get())

  new_cube.AddTranslateOp().Set((2.5, 0.0, 0.0))

  material = UsdShade.Material.Get(stage, '/root/_materials/Material')
  UsdShade.MaterialBindingAPI.Apply(new_cube.GetPrim()).Bind(material)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_add_second_cube_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_add_second_cube_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
