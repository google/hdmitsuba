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

"""Tests for interactive progressive rendering in hdMitsuba."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdRender
import usd_render


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_stage() -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
  mesh = UsdGeom.Mesh.Define(stage, '/World/Quad')
  mesh.CreatePointsAttr([
      Gf.Vec3f(-1, -1, 0),
      Gf.Vec3f(1, -1, 0),
      Gf.Vec3f(1, 1, 0),
      Gf.Vec3f(-1, 1, 0),
  ])
  mesh.CreateFaceVertexCountsAttr([4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  camera = UsdGeom.Camera.Define(stage, '/World/Camera')
  camera.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))
  camera.GetPrim().CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('box')
  light = UsdLux.DomeLight.Define(stage, '/World/Light')
  light.CreateIntensityAttr(1.0)
  settings = UsdRender.Settings.Define(stage, '/Render/PrimarySettings')
  settings.GetPrim().CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())
  return stage


@pytest.mark.parametrize('use_kernel_freezing', [False, True])
def test_progressive_rendering_accumulates(use_kernel_freezing: bool):
  stage = _create_stage()
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': True,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )

  first = engine.render()['color'].astype(np.float64)
  second = engine.render()['color'].astype(np.float64)
  early_delta = np.mean(np.abs(second - first))

  for _ in range(64):
    engine.render()
  late_a = engine.render()['color'].astype(np.float64)
  late_b = engine.render()['color'].astype(np.float64)
  late_delta = np.mean(np.abs(late_b - late_a))

  assert early_delta > 0.0
  assert late_delta < early_delta


@pytest.mark.parametrize('use_kernel_freezing', [False, True])
def test_interactive_samples_per_pass_updates_dynamically(
    use_kernel_freezing: bool,
):
  stage = _create_stage()
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': True,
          'mitsuba:sample_count': 8,
          'mitsuba:interactive_samples_per_pass': 1,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )
  engine.render()
  assert engine.get_render_stats()['numCompletedSamples'] == 1
  assert not engine.is_converged()

  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': True,
          'mitsuba:sample_count': 8,
          'mitsuba:interactive_samples_per_pass': 4,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )
  engine.render()
  assert engine.get_render_stats()['numCompletedSamples'] == 4
  assert not engine.is_converged()

  engine.render()
  stats = engine.get_render_stats()
  assert stats['numCompletedSamples'] == 8
  assert stats['totalSamples'] == 8
  assert engine.is_converged()

  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': True,
          'mitsuba:sample_count': 12,
          'mitsuba:interactive_samples_per_pass': 4,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )
  engine.render()
  assert engine.get_render_stats()['numCompletedSamples'] == 4
  assert not engine.is_converged()


@pytest.mark.parametrize('use_kernel_freezing', [False, True])
def test_interactive_matches_non_interactive_odd_sample_count(
    use_kernel_freezing: bool,
):
  stage = _create_stage()
  batch_engine = usd_render.RenderEngine(stage)
  batch_engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': False,
          'mitsuba:sample_count': 7,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )
  batch_color = batch_engine.render()['color'].astype(np.float64)
  assert batch_engine.is_converged()

  interactive_engine = usd_render.RenderEngine(stage)
  interactive_engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      render_settings_path='/Render/PrimarySettings',
      width=64,
      overrides={
          'enableInteractive': True,
          'mitsuba:sample_count': 7,
          'mitsuba:interactive_samples_per_pass': 2,
          'mitsuba:use_kernel_freezing': use_kernel_freezing,
      },
  )

  expected_completed = [2, 4, 6, 7]
  interactive_color = None
  for expected in expected_completed:
    interactive_color = interactive_engine.render()['color'].astype(np.float64)
    assert (
        interactive_engine.get_render_stats()['numCompletedSamples'] == expected
    )
    assert interactive_engine.is_converged() == (expected == 7)

  assert interactive_color is not None
  assert interactive_engine.get_render_stats()['totalSamples'] == 7
  # Unoccluded dome-light pixels have zero Monte Carlo variance and verify exact
  # sample weighting across the (2, 2, 2, 1) passes (sum of weights = 7).
  dome_mask = batch_color[:, :, 0] == 1.0
  np.testing.assert_allclose(
      interactive_color[dome_mask], batch_color[dome_mask], rtol=1e-5, atol=1e-5
  )
  # Across the full frame (including diffuse Monte Carlo shading with different
  # per-pass TEA seeds), the 7-sample interactive mean matches the 7-sample
  # batch mean.
  np.testing.assert_allclose(
      np.mean(interactive_color, axis=(0, 1)),
      np.mean(batch_color, axis=(0, 1)),
      rtol=1e-2,
      atol=1e-2,
  )
