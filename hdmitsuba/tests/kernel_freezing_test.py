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
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender
import pytest

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


def _create_engine(
    freeze_enabled: bool, overrides: dict | None = None
) -> tuple[Usd.Stage, usd_render.RenderEngine]:
  stage = Usd.Stage.CreateInMemory()
  _setup_stage(stage, freeze_enabled=freeze_enabled)
  test_helpers.create_render_settings(stage)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      camera_path='/camera',
      width=256,
      overrides=overrides,
  )
  return stage, engine


def test_kernel_freezing_camera_motion():
  # Compare renderings using modified camera with and without freezing.
  stage_frozen, engine_frozen = _create_engine(freeze_enabled=True)
  img_frozen_1 = engine_frozen.render()['color']
  translate_op_frozen = UsdGeom.Xformable(
      UsdGeom.Camera.Get(stage_frozen, '/camera')
  ).GetOrderedXformOps()[0]
  translate_op_frozen.Set((0.5, 2.0, 5.0))
  img_frozen_2 = engine_frozen.render()['color']
  translate_op_frozen.Set((1.0, 2.0, 5.0))
  img_frozen_3 = engine_frozen.render()['color']

  # Render the same images with freezing disabled.
  stage_ref, engine_ref = _create_engine(freeze_enabled=False)
  img_ref_1 = engine_ref.render()['color']
  translate_op_ref = UsdGeom.Xformable(
      UsdGeom.Camera.Get(stage_ref, '/camera')
  ).GetOrderedXformOps()[0]
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
  stage_frozen, engine_frozen = _create_engine(
      freeze_enabled=True,
      overrides={'mitsuba:sampler:sample_count': 4, 'enableInteractive': True},
  )
  img_frozen_1 = engine_frozen.render()['color']

  translate_op_frozen = UsdGeom.Xformable(
      UsdGeom.Camera.Get(stage_frozen, '/camera')
  ).GetOrderedXformOps()[0]
  translate_op_frozen.Set((0.5, 2.0, 5.0))
  img_frozen_2 = engine_frozen.render()['color']
  translate_op_frozen.Set((1.0, 2.0, 5.0))
  img_frozen_3 = engine_frozen.render()['color']

  stage_ref, engine_ref = _create_engine(
      freeze_enabled=False,
      overrides={'mitsuba:sampler:sample_count': 4, 'enableInteractive': True},
  )
  img_ref_1 = engine_ref.render()['color']
  translate_op_ref = UsdGeom.Xformable(
      UsdGeom.Camera.Get(stage_ref, '/camera')
  ).GetOrderedXformOps()[0]
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
  new_mesh.GetPointsAttr().Set(
      [(-0.5, 1, 1), (0.5, 1, 1), (0.5, 1, 2), (-0.5, 1, 2)]
  )
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
  np.testing.assert_allclose(
      img_disabled, engine.render()['color'], atol=1e-4, rtol=1e-4
  )


def _render_instancing_frames(freeze_enabled: bool) -> list[np.ndarray]:
  stage, engine = _create_engine(freeze_enabled=freeze_enabled)
  instancer = UsdGeom.PointInstancer.Define(stage, '/instancer')
  instancer.CreatePrototypesRel().SetTargets(['/mesh'])
  proto_indices = instancer.CreateProtoIndicesAttr([0, 0])
  positions = instancer.CreatePositionsAttr([(-1.5, 0.5, 0), (1.5, 0.5, 0)])
  camera = UsdGeom.Xformable(
      stage.GetPrimAtPath('/camera')
  ).GetOrderedXformOps()[0]
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


@pytest.mark.parametrize('kind', ['dome', 'point', 'rect', 'distant'])
def test_kernel_freezing_light_emission_updates(kind: str) -> None:
  """Tests that in-place light emission updates replay without re-recording."""
  stage, engine = _create_engine(freeze_enabled=True)
  if kind == 'dome':
    intensity = UsdLux.DomeLight.Get(stage, '/light').GetIntensityAttr()
  else:
    stage.RemovePrim('/light')
    light = {
        'point': UsdLux.SphereLight,
        'rect': UsdLux.RectLight,
        'distant': UsdLux.DistantLight,
    }[kind].Define(stage, '/test_light')
    light.AddTranslateOp().Set((0, 2, 0))
    light.AddRotateXOp().Set(-90)
    if kind == 'point':
      light.CreateTreatAsPointAttr(True)
    intensity = light.CreateIntensityAttr(2.0)
  engine.render()  # Warm-up.
  intensity.Set(2.0 * intensity.Get())
  assert np.max(engine.render()['color'][..., :3]) > 0.0  # Records the kernel.
  intensity.Set(0.0)
  with dr.kernel_history() as kh:
    img = engine.render()['color'][..., :3]
  modes = {entry.recording_mode for entry in kh}
  assert dr.KernelRecordingMode.Replayed in modes
  assert dr.KernelRecordingMode.Recorded not in modes
  assert np.max(img) == 0.0


def _render_area_light_frames(
    kind: str, freeze_enabled: bool
) -> list[np.ndarray]:
  stage, engine = _create_engine(freeze_enabled=freeze_enabled)
  stage.RemovePrim('/light')
  if kind == 'rect':
    light = UsdLux.RectLight.Define(stage, '/test_light')
    width = light.CreateWidthAttr(1.0)
    height = light.CreateHeightAttr(1.0)
    resize = lambda: (width.Set(2.0), height.Set(1.5))
  elif kind == 'disk':
    light = UsdLux.DiskLight.Define(stage, '/test_light')
    radius = light.CreateRadiusAttr(0.5)
    resize = lambda: radius.Set(1.0)
  elif kind == 'sphere':
    light = UsdLux.SphereLight.Define(stage, '/test_light')
    radius = light.CreateRadiusAttr(0.5)
    resize = lambda: radius.Set(0.8)
  else:
    light = UsdLux.CylinderLight.Define(stage, '/test_light')
    radius = light.CreateRadiusAttr(0.3)
    length = light.CreateLengthAttr(1.0)
    resize = lambda: (radius.Set(0.6), length.Set(2.0))

  translate = light.AddTranslateOp()
  translate.Set((0.0, 2.0, 0.0))
  light.AddRotateXOp().Set(-90)
  light.CreateIntensityAttr(4.0)

  def render(expect_replay: bool = False) -> np.ndarray:
    with dr.kernel_history() as kh:
      img = engine.render()['color']
    if freeze_enabled and expect_replay:
      modes = {entry.recording_mode for entry in kh}
      assert dr.KernelRecordingMode.Replayed in modes
      assert dr.KernelRecordingMode.Recorded not in modes
    return img

  frames = [render()]  # Warm-up.
  translate.Set((-0.8, 2.0, 0.3))
  frames.append(render())  # Records the frozen kernel.
  translate.Set((0.8, 1.5, -0.3))
  frames.append(render(expect_replay=True))
  resize()
  frames.append(render(expect_replay=True))
  return frames


@pytest.mark.parametrize('kind', ['rect', 'disk', 'sphere', 'cylinder'])
def test_kernel_freezing_area_light_updates(kind: str) -> None:
  """Tests that area light moves and resizes replay without re-recording."""
  # Render sequentially: a live second engine would invalidate the recording.
  frozen_frames = _render_area_light_frames(kind, freeze_enabled=True)
  ref_frames = _render_area_light_frames(kind, freeze_enabled=False)
  for prev, curr in zip(frozen_frames, frozen_frames[1:]):
    assert not np.allclose(prev, curr, atol=1e-3)
  for frozen, ref in zip(frozen_frames, ref_frames):
    np.testing.assert_allclose(frozen, ref, atol=1e-4, rtol=1e-4)


def _render_outputs(
    engine: usd_render.RenderEngine, expect_replay: bool
) -> dict[str, np.ndarray]:
  if not expect_replay:
    return engine.render()
  with dr.kernel_history() as kh:
    outputs = engine.render()
  modes = {entry.recording_mode for entry in kh}
  assert dr.KernelRecordingMode.Replayed in modes
  assert dr.KernelRecordingMode.Recorded not in modes
  return outputs


def test_kernel_freezing_progressive_pass_ramping() -> None:
  def run_sequence(freeze_enabled: bool) -> list[np.ndarray]:
    stage, engine = _create_engine(freeze_enabled=freeze_enabled)
    translate_op = UsdGeom.Camera.Get(stage, '/camera').GetOrderedXformOps()[0]

    def configure(step_spp: int) -> None:
      engine.configure(
          hydra_delegate_id='HdMitsubaRendererPlugin',
          camera_path='/camera',
          width=32,
          overrides={
              'mitsuba:sampler:sample_count': 8,
              'mitsuba:interactive_samples_per_pass': step_spp,
              'enableInteractive': True,
          },
      )

    # Warm up once so the initial literal camera transform does not record
    # throwaway entries before the pose loop.
    configure(1)
    engine.render()

    outputs = []
    for pose_index, x_offset in enumerate((0.5, 1.0)):
      translate_op.Set((x_offset, 2.0, 5.0))
      for step_spp, completed in ((1, 1), (1, 2), (2, 4), (4, 8)):
        configure(step_spp)
        outputs.append(
            _render_outputs(
                engine, expect_replay=freeze_enabled and pose_index == 1
            )['color']
        )
        assert engine.get_render_stats()['numCompletedSamples'] == completed
      assert engine.is_converged()
    return outputs

  frozen_outputs = run_sequence(True)
  ref_outputs = run_sequence(False)
  assert not np.allclose(frozen_outputs[3], frozen_outputs[7], atol=1e-3)
  np.testing.assert_allclose(frozen_outputs, ref_outputs, atol=1e-4, rtol=1e-4)


def test_kernel_freezing_dynamic_resolution_switching() -> None:
  def run_sequence(freeze_enabled: bool) -> list[dict[str, np.ndarray]]:
    stage, engine = _create_engine(freeze_enabled=freeze_enabled)
    # Attach a depth AOV to verify per-pass aov_integrator preservation across
    # dynamic resolution switches.
    render_settings = UsdRender.Settings.Get(stage, '/Render/PrimarySettings')
    product = UsdRender.Product.Define(stage, '/Render/Product')
    color_var = UsdRender.Var.Define(stage, '/Render/Vars/color')
    color_var.CreateDataTypeAttr('color3f')
    color_var.CreateSourceNameAttr('color')
    depth_var = UsdRender.Var.Define(stage, '/Render/Vars/depth')
    depth_var.CreateDataTypeAttr('float')
    depth_var.CreateSourceNameAttr('depth')
    product.CreateOrderedVarsRel().SetTargets(
        [color_var.GetPath(), depth_var.GetPath()]
    )
    render_settings.CreateProductsRel().SetTargets([product.GetPath()])

    translate_op = UsdGeom.Camera.Get(stage, '/camera').GetOrderedXformOps()[0]
    outputs = []
    for step_index, width in enumerate((32, 16, 32, 24, 32, 16, 24)):
      translate_op.Set((0.15 * step_index, 2.0, 5.0))
      engine.configure(
          hydra_delegate_id='HdMitsubaRendererPlugin',
          camera_path='/camera',
          width=width,
          overrides={
              'mitsuba:sampler:sample_count': 4,
              'enableInteractive': True,
          },
      )
      frame = _render_outputs(
          engine, expect_replay=freeze_enabled and step_index >= 4
      )
      assert frame['color'].shape[1] == width
      assert frame['depth'].shape[1] == width
      outputs.append(frame)
    return outputs

  frozen_outputs = run_sequence(True)
  ref_outputs = run_sequence(False)
  for prev_index, curr_index in ((2, 4), (1, 5), (3, 6)):
    assert not np.allclose(
        frozen_outputs[prev_index]['color'],
        frozen_outputs[curr_index]['color'],
        atol=1e-3,
    )
  for frozen, reference in zip(frozen_outputs, ref_outputs):
    np.testing.assert_allclose(
        frozen['color'], reference['color'], atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        frozen['depth'], reference['depth'], atol=1e-4, rtol=1e-4
    )
