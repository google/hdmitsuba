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

"""Fuzz testing: to_uv with a translation on a mitsuba_bitmap texture.

Unlike the checkerboard to_uv tests, which only use diagonal (scale) matrices,
this test uses an asymmetric matrix with a non-zero UV offset. This verifies
that the bitmap texture cache key embeds the 3x3 UV transform in the layout
Mitsuba expects (translation in the last column after expansion to 4x4) and
that the Hydra delegate and the Python translator agree on the USD row-vector
matrix convention.
"""

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


def test_bitmap_to_uv_translation():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/materials/mitsuba_bitmap.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(128, 128))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')

  image_hd_initial = engine.render()['color']
  scene_initial = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_initial = np.array(mi.render(scene_initial, spp=128))
  test_helpers.write_image(
      image_hd_initial, 'test_bitmap_to_uv_translation_initial_hd.png'
  )
  test_helpers.write_image(
      image_usd_initial, 'test_bitmap_to_uv_translation_initial_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_initial[..., :3], image_usd_initial, atol=0.2
  )

  # Modification: scale UVs by 4 and shift them by (0.25, 0.4). USD matrices
  # use the row-vector convention, so the translation lives in the last row.
  shader = UsdShade.Shader.Get(
      stage, '/root/_materials/ground/myshader/base_color'
  )
  mat = Gf.Matrix3d(4.0, 0, 0, 0, 4.0, 0, 0.25, 0.4, 1.0)
  shader.CreateInput('to_uv', Sdf.ValueTypeNames.Matrix3d).Set(mat)

  image_hd_modified = engine.render()['color']
  scene_modified = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_modified = np.array(mi.render(scene_modified, spp=128))
  test_helpers.write_image(
      image_hd_modified, 'test_bitmap_to_uv_translation_modified_hd.png'
  )
  test_helpers.write_image(
      image_usd_modified, 'test_bitmap_to_uv_translation_modified_usd.png'
  )
  test_helpers.robust_assert_close(
      image_hd_modified[..., :3], image_usd_modified, atol=0.2
  )

  # The offset must visibly move the texture; otherwise both code paths could
  # agree simply by ignoring the translation.
  shader.GetInput('to_uv').Set(Gf.Matrix3d(4.0, 0, 0, 0, 4.0, 0, 0, 0, 1.0))
  scene_scale_only = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))
  image_usd_scale_only = np.array(mi.render(scene_scale_only, spp=128))
  assert not np.allclose(image_usd_modified, image_usd_scale_only, atol=0.05)
  image_hd_scale_only = engine.render()['color']
  assert not np.allclose(
      image_hd_modified[..., :3], image_hd_scale_only[..., :3], atol=0.05
  )
