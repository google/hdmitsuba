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

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdRender
from usd_mitsuba import camera

_APERTURE_SIZE = (36.0, 18.0)


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant("llvm_ad_rgb")


def _create_camera_stage() -> tuple[Usd.Stage, Gf.Matrix4d]:
  """Creates a USD in-memory stage with a configured perspective camera.

  Returns:
      Usd.Stage: The newly created stage with a camera.
      Gf.Matrix4d: The default transform matrix of the camera.
  """

  stage = Usd.Stage.CreateInMemory()
  camera_path = "/World/MyCamera"
  usd_camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
  xform = UsdGeom.Xformable(usd_camera.GetPrim())

  # Set default transform.
  transform_matrix = Gf.Matrix4d()
  transform_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -50))
  transform_matrix = transform_matrix * Gf.Matrix4d().SetTranslate(
      Gf.Vec3d(0, 10, 20)
  )
  transform_op = xform.AddTransformOp()
  transform_op.Set(transform_matrix)

  # Add animation at time=0.0 and time=1.0
  transform_matrix_0 = Gf.Matrix4d()
  transform_matrix_0.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -50))
  transform_matrix_0 = transform_matrix_0 * Gf.Matrix4d().SetTranslate(
      Gf.Vec3d(0, 5, 10)
  )
  transform_op.Set(transform_matrix_0, time=0.0)

  transform_matrix_1 = Gf.Matrix4d()
  transform_matrix_1.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -70))
  transform_matrix_1 = transform_matrix_1 * Gf.Matrix4d().SetTranslate(
      Gf.Vec3d(10, 20, 30)
  )
  transform_op.Set(transform_matrix_1, time=1.0)

  usd_camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
  usd_camera.GetFocalLengthAttr().Set(50.0)
  usd_camera.GetHorizontalApertureAttr().Set(_APERTURE_SIZE[0])
  usd_camera.GetVerticalApertureAttr().Set(_APERTURE_SIZE[1])
  usd_camera.GetHorizontalApertureOffsetAttr().Set(2.3)
  usd_camera.GetVerticalApertureOffsetAttr().Set(-1.5)
  usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.2, 12000.0))
  render_settings_path = "/Render/Settings"
  render_settings = UsdRender.Settings.Define(stage, render_settings_path)
  render_settings.GetResolutionAttr().Set(Gf.Vec2i(200, 100))
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, render_settings_path
  )
  return stage, transform_matrix


def test_convert_to_mitsuba():
  stage, transform_matrix = _create_camera_stage()
  camera_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/MyCamera"))
  mitsuba_camera = camera.usd_to_mitsuba(camera_prim)

  # Transform to Mitsuba's coordinate system.
  transform_matrix = (
      Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 180))
      * transform_matrix
  )
  assert mi.load_dict(mitsuba_camera) is not None
  assert mitsuba_camera["type"] == "perspective"
  assert abs(mitsuba_camera["fov"] - 39.598) < 1e-3
  np.testing.assert_allclose(mitsuba_camera["near_clip"], 0.2)
  np.testing.assert_allclose(mitsuba_camera["far_clip"], 12000.0)
  mi_matrix = mitsuba_camera["to_world"].matrix

  assert abs(mitsuba_camera["principal_point_offset_x"] -
             2.3 / _APERTURE_SIZE[0]) < 1e-6
  assert abs(-mitsuba_camera["principal_point_offset_y"] - -
             1.5 / _APERTURE_SIZE[1]) < 1e-6

  np.testing.assert_allclose(
      mi_matrix, np.array(transform_matrix).T, atol=1e-4
  )
  assert mitsuba_camera["film"]["type"] == "hdrfilm"
  assert mitsuba_camera["film"]["width"] == 200
  assert mitsuba_camera["film"]["height"] == 100


@pytest.mark.parametrize(
    "time, expected_translation, expected_x_rotation",
    [
        (Usd.TimeCode(0), Gf.Vec3d(0, 5, 10), -50),
        (Usd.TimeCode(1), Gf.Vec3d(10, 20, 30), -70),
    ],
    ids=["transformation_at_time_0", "transformation_at_time_1"]
)
def test_convert_to_mitsuba_with_time(
    time, expected_translation, expected_x_rotation
):
  stage, _ = _create_camera_stage()
  camera_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/MyCamera"))
  mitsuba_camera = camera.usd_to_mitsuba(camera_prim, time=time)

  transform_matrix = Gf.Matrix4d()
  transform_matrix.SetRotate(
      Gf.Rotation(Gf.Vec3d(1, 0, 0), expected_x_rotation)
  )
  transform_matrix = transform_matrix * Gf.Matrix4d().SetTranslate(
      expected_translation
  )
  expected_transform = (
      Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 180))
      * transform_matrix
  )
  np.testing.assert_allclose(
      mitsuba_camera["to_world"].matrix,
      np.array(expected_transform).T,
      atol=1e-4,
  )


def test_convert_to_mitsuba_with_crop():
  stage, _ = _create_camera_stage()
  camera_prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/MyCamera"))

  # Set crop window on render settings
  render_settings = UsdRender.Settings(stage.GetPrimAtPath("/Render/Settings"))
  # dataWindowNDC: (0.25, 0.1, 0.75, 0.4)
  # resolution is (200, 100)
  render_settings.CreateDataWindowNDCAttr().Set((0.25, 0.1, 0.75, 0.4))

  mitsuba_camera = camera.usd_to_mitsuba(camera_prim)

  # Expected crop values:
  # width = 200, height = 100
  # xmin=0.25, ymin=0.1, xmax=0.75, ymax=0.4
  # crop_offset_x = round(0.25 * 200) = 50
  # crop_offset_y = round((1.0 - 0.4) * 100) = 60
  # crop_width = round(0.75 * 200) - 50 = 150 - 50 = 100
  # crop_height = round((1.0 - 0.1) * 100) - 60 = 90 - 60 = 30

  film = mitsuba_camera["film"]
  assert film["crop_offset_x"] == 50
  assert film["crop_offset_y"] == 60
  assert film["crop_width"] == 100
  assert film["crop_height"] == 30


def test_convert_to_mitsuba_with_sensor_overrides():
  stage, _ = _create_camera_stage()
  camera_prim = stage.GetPrimAtPath("/World/MyCamera")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:type", Sdf.ValueTypeNames.String
  ).Set("thinlens")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:aperture_radius", Sdf.ValueTypeNames.Float
  ).Set(0.35)
  camera_prim.CreateAttribute(
      "mitsuba:sensor:focus_distance", Sdf.ValueTypeNames.Float
  ).Set(7.5)
  camera_prim.CreateAttribute(
      "mitsuba:sensor:film:pixel_filter:type", Sdf.ValueTypeNames.String
  ).Set("gaussian")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:film:pixel_filter:stddev", Sdf.ValueTypeNames.Float
  ).Set(1.25)

  mitsuba_camera = camera.usd_to_mitsuba(UsdGeom.Camera(camera_prim))
  assert mitsuba_camera["type"] == "thinlens"
  assert abs(mitsuba_camera["fov"] - 39.598) < 1e-3
  np.testing.assert_allclose(mitsuba_camera["aperture_radius"], 0.35)
  np.testing.assert_allclose(mitsuba_camera["focus_distance"], 7.5)
  assert mitsuba_camera["film"]["type"] == "hdrfilm"
  assert mitsuba_camera["film"]["width"] == 200
  assert mitsuba_camera["film"]["height"] == 100
  assert mitsuba_camera["film"]["pixel_filter"] == {
      "type": "gaussian",
      "stddev": 1.25,
  }
  assert mi.load_dict(mitsuba_camera) is not None


def test_thinlens_default_aperture_and_srf_spectrum():
  stage, _ = _create_camera_stage()
  camera_prim = stage.GetPrimAtPath("/World/MyCamera")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:type", Sdf.ValueTypeNames.String
  ).Set("thinlens")

  mitsuba_camera = camera.usd_to_mitsuba(UsdGeom.Camera(camera_prim))
  assert mitsuba_camera["type"] == "thinlens"
  assert mitsuba_camera["aperture_radius"] == 0.0
  assert mi.load_dict(mitsuba_camera) is not None

  camera_prim.CreateAttribute(
      "mitsuba:sensor:srf:type", Sdf.ValueTypeNames.String
  ).Set("spectrum")
  camera_prim.CreateAttribute(
      "mitsuba:sensor:srf:value", Sdf.ValueTypeNames.String
  ).Set("400:0.5, 500:1.0, 600:0.8")
  mitsuba_camera = camera.usd_to_mitsuba(UsdGeom.Camera(camera_prim))
  assert mitsuba_camera["srf"] == {
      "type": "spectrum",
      "value": [(400.0, 0.5), (500.0, 1.0), (600.0, 0.8)],
  }
  assert mi.load_dict(mitsuba_camera["srf"]) is not None


def test_distant_sensor_and_invalid_target_shape():
  stage = Usd.Stage.CreateInMemory()
  cam = UsdGeom.Camera.Define(stage, "/World/DistantSensor")
  prim = cam.GetPrim()
  prim.CreateAttribute(
      "mitsuba:sensor:type", Sdf.ValueTypeNames.String
  ).Set("distant")
  prim.CreateAttribute(
      "mitsuba:sensor:direction", Sdf.ValueTypeNames.Vector3f
  ).Set(Gf.Vec3f(0.0, 0.0, -1.0))
  prim.CreateAttribute(
      "mitsuba:sensor:shape", Sdf.ValueTypeNames.String
  ).Set("not a valid sdf path")

  assert camera.get_target_shape_path(prim) is None
  sensor_dict = camera.usd_to_mitsuba(cam)
  assert "to_world" not in sensor_dict
  assert sensor_dict["film"]["width"] == 1
  assert sensor_dict["film"]["height"] == 1
  assert mi.load_dict(sensor_dict) is not None


def test_standalone_sensor_sampler_override_and_film_crop_clamping():
  stage, _ = _create_camera_stage()
  settings_prim = stage.GetPrimAtPath("/Render/Settings")
  settings_prim.CreateAttribute(
      "mitsuba:sampler:type", Sdf.ValueTypeNames.String
  ).Set("stratified")
  settings_prim.CreateAttribute(
      "mitsuba:sampler:jitter", Sdf.ValueTypeNames.Bool
  ).Set(False)
  settings_prim.CreateAttribute(
      "mitsuba:sample_count", Sdf.ValueTypeNames.Int
  ).Set(32)

  cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/MyCamera"))
  cam_prim = cam.GetPrim()
  assert camera.usd_to_mitsuba(cam)["sampler"] == {
      "type": "stratified",
      "jitter": False,
      "sample_count": 32,
  }

  # Authored `mitsuba:sensor:sampler:*` is a standalone per-sensor override and
  # must not inherit `type`, `jitter`, or `sample_count` from RenderSettings.
  cam_prim.CreateAttribute(
      "mitsuba:sensor:sampler:sample_count", Sdf.ValueTypeNames.Int
  ).Set(6)
  spp_only_dict = camera.usd_to_mitsuba(cam)
  assert spp_only_dict["sampler"] == {"type": "independent", "sample_count": 6}
  assert mi.load_dict(spp_only_dict).sampler().sample_count() == 6

  cam_prim.RemoveProperty("mitsuba:sensor:sampler:sample_count")
  cam_prim.CreateAttribute(
      "mitsuba:sensor:sampler:type", Sdf.ValueTypeNames.String
  ).Set("multijitter")
  type_only_dict = camera.usd_to_mitsuba(cam)
  assert type_only_dict["sampler"] == {"type": "multijitter"}
  assert mi.load_dict(type_only_dict).sampler().sample_count() == 4

  # Authored `mitsuba:sensor:film:crop_offset_x` without `crop_width` clamps
  # cleanly to the remaining film width so `mi.load_dict` succeeds.
  cam_prim.CreateAttribute(
      "mitsuba:sensor:film:crop_offset_x", Sdf.ValueTypeNames.Int
  ).Set(150)
  cropped_dict = camera.usd_to_mitsuba(cam)
  assert cropped_dict["film"]["crop_offset_x"] == 150
  assert cropped_dict["film"]["crop_offset_y"] == 0
  assert cropped_dict["film"]["crop_width"] == 50
  assert cropped_dict["film"]["crop_height"] == 100
  assert mi.load_dict(cropped_dict) is not None
