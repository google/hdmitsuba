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

"""Fuzz testing: overlapping translation collision modification across separate meshes on USD stage."""

from __future__ import annotations

import mitsuba as mi
import numpy as np

from pxr import Usd  # type: ignore
from pxr import UsdGeom  # type: ignore
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


def test_multimesh_interlocked_collision():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/nodegraphs.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_multimesh_interlocked_collision_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial,
      'test_multimesh_interlocked_collision_initial_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: Translate Mesh1 and Mesh2 into interlocked colliding space
  m1_xform = UsdGeom.Xformable(stage.GetPrimAtPath('/root/Meshes/Mesh1'))
  m1_xform.AddTranslateOp(opSuffix='collide1').Set((0.5, 0.0, 0.0))

  m2_xform = UsdGeom.Xformable(stage.GetPrimAtPath('/root/Meshes/Mesh2'))
  m2_xform.AddTranslateOp(opSuffix='collide2').Set((-0.5, 0.0, 0.0))

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified,
      'test_multimesh_interlocked_collision_modified_hd.png',
  )
  test_helpers.write_image(
      image_usd_modified,
      'test_multimesh_interlocked_collision_modified_usd.png',
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )
