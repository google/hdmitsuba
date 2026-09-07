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

"""Tests for rigid camera and object motion blur support in hdMitsuba."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from hdmitsuba.tests import test_helpers
import usd_render


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def _create_test_mesh(stage: Usd.Stage, path: str = '/mesh') -> UsdGeom.Mesh:
  mesh = UsdGeom.Mesh.Define(stage, path)
  mesh.GetPointsAttr().Set([
      (-0.5, -0.5, 0.0),
      (0.5, -0.5, 0.0),
      (0.5, 0.5, 0.0),
      (-0.5, 0.5, 0.0),
  ])
  mesh.GetFaceVertexCountsAttr().Set([4])
  mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  return mesh


def _create_test_camera(
    stage: Usd.Stage,
    path: str = '/camera',
    shutter_open: float = -0.5,
    shutter_close: float = 0.5,
) -> UsdGeom.Camera:
  cam = UsdGeom.Camera.Define(stage, path)
  cam.CreateFocalLengthAttr(50.0)
  cam.CreateHorizontalApertureAttr(36.0)
  cam.CreateVerticalApertureAttr(36.0)
  cam.CreateShutterOpenAttr(shutter_open)
  cam.CreateShutterCloseAttr(shutter_close)
  return cam


def _create_test_light(stage: Usd.Stage, path: str = '/light') -> UsdLux.DomeLight:
  light = UsdLux.DomeLight.Define(stage, path)
  light.CreateIntensityAttr(1.0)
  return light


def test_camera_translation_motion_blur():
  """Tests that rigid camera translation produces motion blur matching offline render."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  t_op = xform_cam.AddTranslateOp()
  t_op.Set(Gf.Vec3d(-0.5, 0.0, 5.0), time=-0.5)
  t_op.Set(Gf.Vec3d(0.5, 0.0, 5.0), time=0.5)

  _create_test_mesh(stage)
  _create_test_light(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_camera_translation_mb',
      spp=128,
      atol=0.05,
      time=Usd.TimeCode(0.0),
  )


def test_camera_rotation_motion_blur():
  """Tests that rigid camera rotation produces motion blur matching offline render."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  xform_cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
  rot_op = xform_cam.AddRotateZOp()
  rot_op.Set(-15.0, time=-0.5)
  rot_op.Set(15.0, time=0.5)

  _create_test_mesh(stage)
  _create_test_light(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_camera_rotation_mb',
      spp=128,
      atol=0.05,
      time=Usd.TimeCode(0.0),
  )


def test_mesh_translation_motion_blur():
  """Tests that rigid mesh translation produces motion blur matching offline render."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  xform_cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

  mesh = _create_test_mesh(stage)
  xform_mesh = UsdGeom.Xformable(mesh.GetPrim())
  t_op = xform_mesh.AddTranslateOp()
  t_op.Set(Gf.Vec3d(-0.5, 0.0, 0.0), time=-0.5)
  t_op.Set(Gf.Vec3d(0.5, 0.0, 0.0), time=0.5)

  _create_test_light(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_mesh_translation_mb',
      spp=128,
      atol=0.05,
      time=Usd.TimeCode(0.0),
  )


def test_mesh_rotation_motion_blur():
  """Tests that rigid mesh rotation produces motion blur matching offline render."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  xform_cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

  mesh = _create_test_mesh(stage)
  xform_mesh = UsdGeom.Xformable(mesh.GetPrim())
  rot_op = xform_mesh.AddRotateZOp()
  rot_op.Set(0.0, time=-0.5)
  rot_op.Set(45.0, time=0.5)

  _create_test_light(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_mesh_rotation_mb',
      spp=128,
      atol=0.05,
      time=Usd.TimeCode(0.0),
  )


def test_combined_camera_and_mesh_motion_blur():
  """Tests that simultaneous camera and mesh motion blur matches offline render."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  t_cam = xform_cam.AddTranslateOp()
  t_cam.Set(Gf.Vec3d(0.0, -0.5, 5.0), time=-0.5)
  t_cam.Set(Gf.Vec3d(0.0, 0.5, 5.0), time=0.5)

  mesh = _create_test_mesh(stage)
  xform_mesh = UsdGeom.Xformable(mesh.GetPrim())
  t_mesh = xform_mesh.AddTranslateOp()
  t_mesh.Set(Gf.Vec3d(-0.5, 0.0, 0.0), time=-0.5)
  t_mesh.Set(Gf.Vec3d(0.5, 0.0, 0.0), time=0.5)

  _create_test_light(stage)

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/camera',
      output_prefix='test_combined_mb',
      spp=128,
      atol=0.05,
      time=Usd.TimeCode(0.0),
  )


def test_interactive_camera_motion_blur_update():
  """Tests that updating a static camera to animated in Hydra reflects motion blur."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  t_cam = xform_cam.AddTranslateOp()
  t_cam.Set(Gf.Vec3d(0.0, 0.0, 5.0))

  _create_test_mesh(stage)
  _create_test_light(stage)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera')
  img_static = np.array(engine.render(time_code=Usd.TimeCode(0.0))['color'])

  # Update camera with animated translation
  t_cam.Set(Gf.Vec3d(-0.5, 0.0, 5.0), time=-0.5)
  t_cam.Set(Gf.Vec3d(0.5, 0.0, 5.0), time=0.5)

  img_blurred = np.array(engine.render(time_code=Usd.TimeCode(0.0))['color'])
  diff = np.max(np.abs(img_static - img_blurred))
  assert diff > 0.1, 'Interactive camera update should have altered render output'


def test_interactive_mesh_motion_blur_update():
  """Tests that updating a static mesh to animated in Hydra reflects motion blur."""
  stage = Usd.Stage.CreateInMemory()
  stage.SetStartTimeCode(-1.0)
  stage.SetEndTimeCode(1.0)
  test_helpers.create_render_settings(stage, resolution=(512, 512), spp=128)

  cam = _create_test_camera(stage)
  xform_cam = UsdGeom.Xformable(cam.GetPrim())
  xform_cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

  mesh = _create_test_mesh(stage)
  xform_mesh = UsdGeom.Xformable(mesh.GetPrim())
  t_op = xform_mesh.AddTranslateOp()
  t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

  _create_test_light(stage)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera')
  img_static = np.array(engine.render(time_code=Usd.TimeCode(0.0))['color'])

  # Update mesh with animated translation
  t_op.Set(Gf.Vec3d(-0.5, 0.0, 0.0), time=-0.5)
  t_op.Set(Gf.Vec3d(0.5, 0.0, 0.0), time=0.5)

  img_blurred = np.array(engine.render(time_code=Usd.TimeCode(0.0))['color'])
  diff = np.max(np.abs(img_static - img_blurred))
  assert diff > 0.1, 'Interactive mesh update should have altered render output'
