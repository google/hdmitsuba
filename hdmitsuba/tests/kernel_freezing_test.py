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

"""Tests for Dr.Jit kernel freezing in hdMitsuba."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdRender
from pxr import UsdLux
from hdmitsuba.tests import test_helpers
import usd_render


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _define_light(stage: Usd.Stage, path: str = '/light') -> None:
  light = UsdLux.DomeLight.Define(stage, path)
  light.CreateIntensityAttr(1.3)
  light.CreateColorAttr((1.0, 1.0, 1.0))


def _setup_stage(stage: Usd.Stage, freeze_enabled: bool) -> None:
  geometry = UsdGeom.Mesh.Define(stage, '/mesh')
  points = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]
  face_vertex_counts = [4]
  face_vertex_indices = [3, 2, 1, 0]
  geometry.GetPointsAttr().Set(points)
  geometry.GetFaceVertexCountsAttr().Set(face_vertex_counts)
  geometry.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
  camera = UsdGeom.Camera.Define(stage, '/camera')
  camera.AddTranslateOp().Set((0, 2, 5))  # Look down at the plane
  camera.AddRotateXOp().Set(-20)
  _define_light(stage)
  render_settings_path = '/Render/PrimarySettings'
  render_settings = UsdRender.Settings.Define(stage, render_settings_path)
  render_settings.GetPrim().CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())

  render_settings.GetPrim().CreateAttribute(
      'mitsuba:use_kernel_freezing', Sdf.ValueTypeNames.Bool
  ).Set(freeze_enabled)


def _create_engine(freeze_enabled: bool, overrides: dict | None = None) -> tuple[Usd.Stage, usd_render.RenderEngine]:
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage, freeze_enabled=freeze_enabled)
  test_helpers.create_render_settings(stage)
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera', width=256, overrides=overrides)
  return stage, engine


def test_kernel_freezing_camera_motion():
  # Compare renderings using modified camera with and without freezing.
  stage_frozen, engine_frozen = _create_engine(freeze_enabled=True)
  img_frozen_1 = engine_frozen.render()['color']
  translate_op_frozen = UsdGeom.Xformable(UsdGeom.Camera.Get(stage_frozen, '/camera')).GetOrderedXformOps()[0]
  translate_op_frozen.Set((0.5, 2.0, 5.0))
  img_frozen_2 = engine_frozen.render()['color']
  translate_op_frozen.Set((1.0, 2.0, 5.0))
  img_frozen_3 = engine_frozen.render()['color']

  # Render the same images with freezing disabled.
  stage_ref, engine_ref = _create_engine(freeze_enabled=False)
  img_ref_1 = engine_ref.render()['color']
  translate_op_ref = UsdGeom.Xformable(UsdGeom.Camera.Get(stage_ref, '/camera')).GetOrderedXformOps()[0]
  translate_op_ref.Set((0.5, 2.0, 5.0))
  img_ref_2 = engine_ref.render()['color']
  translate_op_ref.Set((1.0, 2.0, 5.0))
  img_ref_3 = engine_ref.render()['color']

  # Validate close match of all renderings at matching camera positions.
  assert not np.allclose(img_frozen_1, img_frozen_2, atol=1e-3)
  assert not np.allclose(img_frozen_2, img_frozen_3, atol=1e-3)
  np.testing.assert_allclose(img_frozen_1, img_ref_1, atol=1e-4, rtol=1e-4)
  np.testing.assert_allclose(img_frozen_2, img_ref_2, atol=1e-4, rtol=1e-4)
  np.testing.assert_allclose(img_frozen_3, img_ref_3, atol=1e-4, rtol=1e-4)


def test_kernel_freezing_interactive_rendering():
  # Similar to the test above, but with interactive (progressive) rendering enabled.
  stage_frozen, engine_frozen = _create_engine(freeze_enabled=True, overrides={'mitsuba:sample_count': 4, 'enableInteractive': True})
  img_frozen_1 = engine_frozen.render()['color']

  translate_op_frozen = UsdGeom.Xformable(UsdGeom.Camera.Get(stage_frozen, '/camera')).GetOrderedXformOps()[0]
  translate_op_frozen.Set((0.5, 2.0, 5.0))
  img_frozen_2 = engine_frozen.render()['color']
  translate_op_frozen.Set((1.0, 2.0, 5.0))
  img_frozen_3 = engine_frozen.render()['color']

  stage_ref, engine_ref = _create_engine(freeze_enabled=False, overrides={'mitsuba:sample_count': 4, 'enableInteractive': True})
  img_ref_1 = engine_ref.render()['color']
  translate_op_ref = UsdGeom.Xformable(UsdGeom.Camera.Get(stage_ref, '/camera')).GetOrderedXformOps()[0]
  translate_op_ref.Set((0.5, 2.0, 5.0))
  img_ref_2 = engine_ref.render()['color']
  translate_op_ref.Set((1.0, 2.0, 5.0))
  img_ref_3 = engine_ref.render()['color']

  assert not np.allclose(img_frozen_1, img_frozen_2, atol=1e-3)
  assert not np.allclose(img_frozen_2, img_frozen_3, atol=1e-3)
  np.testing.assert_allclose(img_frozen_1, img_ref_1, atol=1e-4, rtol=1e-4)
  np.testing.assert_allclose(img_frozen_2, img_ref_2, atol=1e-4, rtol=1e-4)
  np.testing.assert_allclose(img_frozen_3, img_ref_3, atol=1e-4, rtol=1e-4)


def test_kernel_freezing_structural_change_invalidation():
  # Verify that structural changes correctly invalidate the frozen cache.
  stage, engine = _create_engine(freeze_enabled=True)
  img_before = engine.render()['color']
  # Dynamically add a new geometry
  new_mesh = UsdGeom.Mesh.Define(stage, '/new_mesh')
  new_mesh.GetPointsAttr().Set([(-0.5, 1, 1), (0.5, 1, 1), (0.5, 1, 2), (-0.5, 1, 2)])
  new_mesh.GetFaceVertexCountsAttr().Set([4])
  new_mesh.GetFaceVertexIndicesAttr().Set([3, 2, 1, 0])
  # Verify that the new geometry is visible and the cache was invalidated
  assert not np.allclose(img_before, engine.render()['color'], atol=1e-2)


def test_kernel_freezing_dynamic_toggle():
  # Verify that dynamically toggling freezing ON/OFF/ON mid-render works perfectly.
  stage, engine = _create_engine(freeze_enabled=True)
  engine.render()
  img_frozen = engine.render()['color']
  # Dynamically turn freezing OFF via settings prim!
  settings_prim = stage.GetPrimAtPath('/Render/PrimarySettings')
  settings_prim.GetAttribute('mitsuba:use_kernel_freezing').Set(False)
  img_disabled = engine.render()['color']
  np.testing.assert_allclose(img_frozen, img_disabled, atol=1e-4, rtol=1e-4)
  # Dynamically turn freezing ON again!
  settings_prim.GetAttribute('mitsuba:use_kernel_freezing').Set(True)
  np.testing.assert_allclose(img_disabled, engine.render()['color'], atol=1e-4, rtol=1e-4)


def _render_instancing_frames(freeze_enabled: bool) -> list[np.ndarray]:
  stage, engine = _create_engine(freeze_enabled=freeze_enabled)
  instancer = UsdGeom.PointInstancer.Define(stage, '/instancer')
  instancer.CreatePrototypesRel().SetTargets(['/mesh'])
  proto_indices = instancer.CreateProtoIndicesAttr([0, 0])
  positions = instancer.CreatePositionsAttr([(-1.5, 0.5, 0), (1.5, 0.5, 0)])
  camera = UsdGeom.Xformable(stage.GetPrimAtPath('/camera')).GetOrderedXformOps()[0]
  points = UsdGeom.Mesh.Get(stage, '/mesh').GetPointsAttr()

  def render(expect_replay: bool = False) -> np.ndarray:
    with dr.kernel_history() as kh:
      img = engine.render()['color']
    if freeze_enabled and expect_replay:
      modes = {entry.recording_mode for entry in kh}
      assert dr.KernelRecordingMode.Replayed in modes
      assert dr.KernelRecordingMode.Recorded not in modes
    return img

  frames = [render()]  # Warm-up.
  camera.Set((0.5, 2.0, 5.0))
  frames.append(render())  # Records the frozen kernel.
  camera.Set((1.0, 2.0, 5.0))
  frames.append(render(expect_replay=True))
  positions.Set([(-1.2, 0.8, 0.3), (1.5, 0.5, -0.2)])
  frames.append(render(expect_replay=True))
  points.Set([p * 1.4 for p in points.Get()])
  frames.append(render(expect_replay=True))
  # Changing the instance count is structural and must invalidate the kernel.
  proto_indices.Set([0, 0, 0])
  positions.Set([(-1.2, 0.8, 0.3), (1.5, 0.5, -0.2), (0.0, 1.0, 1.5)])
  frames.append(render())
  return frames


def test_kernel_freezing_instancing():
  """Tests frozen kernel replay and invalidation with a PointInstancer."""
  # Render sequentially: a live second engine would invalidate the recording.
  frozen_frames = _render_instancing_frames(freeze_enabled=True)
  ref_frames = _render_instancing_frames(freeze_enabled=False)
  for prev, curr in zip(frozen_frames, frozen_frames[1:]):
    assert not np.allclose(prev, curr, atol=1e-3)
  for frozen, ref in zip(frozen_frames, ref_frames):
    np.testing.assert_allclose(frozen, ref, atol=1e-4, rtol=1e-4)
