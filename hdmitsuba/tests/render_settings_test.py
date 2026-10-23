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

"""Tests for the hdMitsuba render settings."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Sdf
from pxr import Tf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdRender
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


_SPP = 4

@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")


def test_basic_render_settings():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")

  # Add an additional ground plane to better distinguish between direct
  # and path integrators.
  plane = UsdGeom.Plane.Define(stage, "/World/Plane")
  plane.GetWidthAttr().Set(10)
  plane.GetLengthAttr().Set(10)

  render_settings = test_helpers.create_render_settings(stage)
  settings_prim = render_settings.GetPrim()

  settings_prim.CreateAttribute(
      "mitsuba:sample_count", Sdf.ValueTypeNames.Int
  ).Set(4)
  settings_prim.CreateAttribute(
      "mitsuba:integrator:type", Sdf.ValueTypeNames.String
  ).Set("direct")

  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert scene_dict["integrator"]["type"] == "direct"

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix="test_basic_render_settings",
      atol=0.1,
  )


def test_aovs():
  stage = Usd.Stage.Open(
      f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs.usda"
  )

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results_hd = engine.render()
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)

  assert scene_dict["integrator"]["type"] == "aov"
  assert scene_dict["integrator"]["aovs"] == "albedo:albedo,normal:sh_normal,depth:depth"
  assert scene_dict["integrator"]["integrator"]["type"] == "path"

  assert set(results_hd.keys()) == {"color", "albedo", "normal", "depth"}

  image_mi = mi.render(mi.load_dict(scene_dict), spp=_SPP)
  image_mi_np = np.array(image_mi)

  for aov_name, channel_slice, atol in [
      ("color", slice(0, 3), 0.05),
      ("albedo", slice(3, 6), 0.05),
      ("normal", slice(6, 9), 0.05),
      ("depth", slice(9, 10), 0.5),
  ]:
    hd_img = results_hd[aov_name]
    hd_img_compare = hd_img[..., :3] if aov_name == "color" else hd_img
    mi_img = image_mi_np[:, :, channel_slice]
    suffix = "" if aov_name == "color" else f"_{aov_name}"
    test_helpers.write_image(hd_img, f"test_aovs{suffix}_hd.png")
    test_helpers.write_image(mi_img, f"test_aovs{suffix}_usd.png")
    test_helpers.robust_assert_close(hd_img_compare, mi_img, atol=atol)


def test_aovs_multiple_products():
  stage = Usd.Stage.Open(
      f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs_multiple_products.usda"
  )

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results_hd = engine.render()
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert scene_dict["integrator"]["type"] == "path"

  scene_dict_aov = usd_mitsuba.convert_to_mitsuba(
      stage, render_product_path="/root/Render/Products/AovOutput"
  )
  assert scene_dict_aov["integrator"]["type"] == "aov"
  assert scene_dict_aov["integrator"]["aovs"] == "albedo:albedo,normal:sh_normal,depth:depth"

  assert set(results_hd.keys()) == {"color", "albedo", "normal", "depth"}

  image_mi = mi.render(mi.load_dict(scene_dict), spp=_SPP)
  image_mi_np = np.array(image_mi)

  color_hd = results_hd["color"]
  color_mi = image_mi_np[:, :, :3]
  test_helpers.write_image(color_hd, "test_aovs_multiple_products_hd.png")
  test_helpers.write_image(color_mi, "test_aovs_multiple_products_usd.png")
  test_helpers.robust_assert_close(color_hd[..., :3], color_mi, atol=0.05)

  image_mi_aov = mi.render(mi.load_dict(scene_dict_aov), spp=_SPP)
  image_mi_aov_np = np.array(image_mi_aov)

  for aov_name, channel_slice, atol in [
      ("albedo", slice(3, 6), 0.05),
      ("normal", slice(6, 9), 0.05),
      ("depth", slice(9, 10), 0.5),
  ]:
    hd_img = results_hd[aov_name]
    mi_img = image_mi_aov_np[:, :, channel_slice]
    test_helpers.write_image(
        hd_img, f"test_aovs_multiple_products_{aov_name}_hd.png"
    )
    test_helpers.write_image(
        mi_img, f"test_aovs_multiple_products_{aov_name}_usd.png"
    )
    test_helpers.robust_assert_close(hd_img, mi_img, atol=atol)


def test_aovs_dynamic_update():
  stage = Usd.Stage.Open(
      f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs.usda"
  )
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results = engine.render()
  assert set(results.keys()) == {"color", "albedo", "normal", "depth"}

  product_prim = stage.GetPrimAtPath("/root/Render/Products/MainOutput")
  render_product = UsdRender.Product(product_prim)
  render_product.GetOrderedVarsRel().SetTargets([
      Sdf.Path("/root/Render/Vars/color"),
      Sdf.Path("/root/Render/Vars/normal"),
  ])

  results_updated = engine.render()
  assert set(results_updated.keys()) == {"color", "normal"}
  test_helpers.robust_assert_close(
      results["color"], results_updated["color"], atol=0.05
  )
  test_helpers.robust_assert_close(
      results["normal"], results_updated["normal"], atol=0.05
  )

  stage.GetPrimAtPath("/root/Render/Vars/normal").SetActive(False)
  assert set(engine.render().keys()) == {"color"}
  assert usd_mitsuba.convert_to_mitsuba(stage)["integrator"]["type"] == "path"


def test_aovs_camera_from_configure():
  original_path = f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs.usda"
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(original_path)
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, "/root/Render/Settings"
  )
  UsdRender.Product(
      stage.GetPrimAtPath("/root/Render/Products/MainOutput")
  ).GetCameraRel().SetTargets([])

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id="HdMitsubaRendererPlugin",
      camera_path="/root/Camera/Camera",
  )
  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert (
      scene_dict["integrator"]["aovs"]
      == "albedo:albedo,normal:sh_normal,depth:depth"
  )
  assert set(engine.render().keys()) == {"color", "albedo", "normal", "depth"}


def test_aovs_duplicate_resolving():
  original_path = f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs.usda"
  # We sublayer the loaded USD since we modify its content. This
  # otherwise could pollute the stage cache and break other tests.
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(original_path)
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, "/root/Render/Settings"
  )

  # Internally, both AOVs will request the `shape_index` from Mitsuba. This
  # test ensures that this is handled correctly.
  prim_id_var = UsdRender.Var.Define(stage, "/root/Render/Vars/primId")
  prim_id_var.GetDataTypeAttr().Set("int")
  prim_id_var.GetSourceNameAttr().Set("primId")
  instance_id_var = UsdRender.Var.Define(
      stage, "/root/Render/Vars/instanceId"
  )
  instance_id_var.GetDataTypeAttr().Set("int")
  instance_id_var.GetSourceNameAttr().Set("instanceId")

  product_prim = stage.GetPrimAtPath("/root/Render/Products/MainOutput")
  render_product = UsdRender.Product(product_prim)
  render_product.GetOrderedVarsRel().SetTargets([
      Sdf.Path("/root/Render/Vars/primId"),
      Sdf.Path("/root/Render/Vars/instanceId"),
  ])

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results = engine.render()

  assert "primId" in results
  assert "instanceId" in results
  np.testing.assert_array_equal(results["primId"], results["instanceId"])


def test_crop_rendering():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")
  render_settings = test_helpers.create_render_settings(
      stage, resolution=(512, 512)
  )
  settings_prim = render_settings.GetPrim()

  settings_prim.CreateAttribute(
      "mitsuba:sample_count", Sdf.ValueTypeNames.Int
  ).Set(4)

  # Render full image first
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  image_full = engine.render()["color"]

  render_settings.CreateDataWindowNDCAttr().Set((0.25, 0.1, 0.75, 0.4))

  # Re-render with crop
  engine_crop = usd_render.RenderEngine(stage)
  engine_crop.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  image_crop = engine_crop.render()["color"]

  test_helpers.write_image(image_full, "test_crop_rendering_full.png")
  test_helpers.write_image(image_crop, "test_crop_rendering_crop.png")

  # Verify content in cropped region (rows [307, 460], cols [128, 384] inclusive, so [307:461, 128:384])
  crop_region_full = image_full[307:461, 128:384, :3]
  crop_region_crop = image_crop[307:461, 128:384, :3]
  test_helpers.robust_assert_close(
      crop_region_crop, crop_region_full, atol=0.02
  )

  # Verify everything else is black
  mask = np.ones((512, 512), dtype=bool)
  mask[307:461, 128:384] = False
  np.testing.assert_array_equal(image_crop[..., :3][mask], 0.0)


def test_variant_change_dynamic():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")
  render_settings = test_helpers.create_render_settings(stage)
  settings_prim = render_settings.GetPrim()

  settings_prim.CreateAttribute(
      "mitsuba:variant", Sdf.ValueTypeNames.String
  ).Set("scalar_rgb")

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results_scalar = engine.render()

  # Change the variant setting dynamically on the same stage and render engine
  settings_prim.GetAttribute("mitsuba:variant").Set("llvm_ad_rgb")
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  results_llvm = engine.render()

  test_helpers.robust_assert_close(
      results_scalar["color"], results_llvm["color"], atol=0.05
  )


def test_seed_render_setting():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")
  render_settings = test_helpers.create_render_settings(
      stage, resolution=(64, 64), spp=1
  )
  settings_prim = render_settings.GetPrim()
  seed_attr = settings_prim.CreateAttribute(
      "mitsuba:seed", Sdf.ValueTypeNames.Int
  )
  seed_attr.Set(0)

  scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(stage))

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  hd_seed_0 = engine.render()["color"][..., :3]
  mi_seed_0 = np.array(mi.render(scene, spp=1, seed=0))
  np.testing.assert_allclose(hd_seed_0, mi_seed_0, rtol=1e-4, atol=1e-4)

  seed_attr.Set(42)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  hd_seed_42 = engine.render()["color"][..., :3]
  mi_seed_42 = np.array(mi.render(scene, spp=1, seed=42))
  np.testing.assert_allclose(hd_seed_42, mi_seed_42, rtol=1e-4, atol=1e-4)
  assert not np.allclose(hd_seed_0, hd_seed_42, atol=1e-4)


def test_integrator_depth_render_settings():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")
  plane = UsdGeom.Plane.Define(stage, "/World/Plane")
  plane.GetWidthAttr().Set(10)
  plane.GetLengthAttr().Set(10)

  settings_prim = test_helpers.create_render_settings(
      stage, resolution=(64, 64), spp=4
  ).GetPrim()
  settings_prim.CreateAttribute(
      "mitsuba:integrator:max_depth", Sdf.ValueTypeNames.Int
  ).Set(2)
  settings_prim.CreateAttribute(
      "mitsuba:integrator:rr_depth", Sdf.ValueTypeNames.Int
  ).Set(2)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  test_helpers.assert_hydra_equal_to_offline(
      stage, output_prefix="test_integrator_depth_2", spp=4, engine=engine
  )

  settings_prim.GetAttribute("mitsuba:integrator:max_depth").Set(6)
  settings_prim.GetAttribute("mitsuba:integrator:rr_depth").Set(4)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  test_helpers.assert_hydra_equal_to_offline(
      stage, output_prefix="test_integrator_depth_6", spp=4, engine=engine
  )


def test_unsupported_integrator_setting_is_ignored():
  stage = Usd.Stage.Open(f"{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda")
  settings_prim = test_helpers.create_render_settings(
      stage, resolution=(64, 64), spp=4
  ).GetPrim()
  # The `direct` integrator does not support `max_depth`.
  settings_prim.CreateAttribute(
      "mitsuba:integrator:type", Sdf.ValueTypeNames.String
  ).Set("direct")
  settings_prim.CreateAttribute(
      "mitsuba:integrator:max_depth", Sdf.ValueTypeNames.Int
  ).Set(2)

  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  hd_image = engine.render()["color"][..., :3]

  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  scene_dict["integrator"] = {"type": "direct"}
  mi_image = np.array(mi.render(mi.load_dict(scene_dict), spp=4))
  test_helpers.robust_assert_close(hd_image, mi_image, atol=0.05)


@pytest.mark.parametrize("variant", ["llvm_ad_rgb", "cuda_ad_rgb"])
def test_denoise_render_settings(variant: str):
  try:
    mi.set_variant(variant)
  except ImportError as error:
    pytest.skip(f"Variant {variant} is not available: {error}")

  original_path = f"{test_helpers.TEST_ASSETS_PATH}/render_settings/aovs.usda"
  stage = Usd.Stage.CreateInMemory()
  stage.GetRootLayer().subLayerPaths.append(original_path)
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, "/root/Render/Settings"
  )
  settings_prim = stage.GetPrimAtPath("/root/Render/Settings")
  settings_prim.CreateAttribute(
      "mitsuba:sample_count", Sdf.ValueTypeNames.Int
  ).Set(1)

  engine = usd_render.RenderEngine(stage)

  def render(denoise: bool, width: int | None = None):
    engine.configure(
        hydra_delegate_id="HdMitsubaRendererPlugin",
        width=width,
        overrides={
            "mitsuba:variant": variant,
            "mitsuba:denoise": denoise,
        },
    )
    return engine.render()

  # Without an explicit pixel filter on the camera, batch mode defaults to
  # Gaussian while enabling denoising rebuilds the sensor with a box filter.
  default_results = render(denoise=False)
  error_mark = Tf.Error.Mark()
  denoised_results = render(denoise=True)
  assert not np.allclose(denoised_results["depth"], default_results["depth"])

  # An explicitly authored camera pixel filter must be respected even when
  # denoising is enabled.
  pixel_filter_attr = stage.GetPrimAtPath(
      "/root/Camera/Camera"
  ).CreateAttribute(
      "mitsuba:sensor:film:pixel_filter:type", Sdf.ValueTypeNames.String
  )
  pixel_filter_attr.Set("gaussian")
  np.testing.assert_allclose(
      render(denoise=True)["depth"], default_results["depth"], atol=1e-5
  )
  if any(
      "HDMITSUBA_ENABLE_OIDN" in error.commentary
      for error in error_mark.GetErrors()
  ):
    error_mark.Clear()
    pytest.skip("OpenImageDenoise is not enabled in this build")

  # Author the box filter explicitly for an apple-to-apples noisy reference.
  pixel_filter_attr.Set("box")
  noisy_results = render(denoise=False)

  noisy_rgb = noisy_results["color"][..., :3]
  denoised_rgb = denoised_results["color"][..., :3]

  # Denoising must modify the noisy 1-spp RGB beauty image and reduce spatial
  # high-frequency gradient energy while preserving mean luminance and non-color
  # channels.
  assert not np.allclose(noisy_rgb, denoised_rgb)
  noisy_gradient = np.mean(np.square(np.diff(noisy_rgb, axis=1)))
  denoised_gradient = np.mean(np.square(np.diff(denoised_rgb, axis=1)))
  assert denoised_gradient < noisy_gradient
  np.testing.assert_allclose(
      np.mean(denoised_rgb), np.mean(noisy_rgb), atol=0.05
  )
  np.testing.assert_array_equal(
      denoised_results["color"][..., 3], noisy_results["color"][..., 3]
  )
  for aov_name in ("albedo", "normal", "depth"):
    np.testing.assert_allclose(
        denoised_results[aov_name], noisy_results[aov_name], atol=1e-5
    )

  # Alternating between a downsampled resolution and full resolution (plus a
  # third resolution) exercises both slots and eviction in the denoiser cache.
  downsampled_rgb = render(denoise=True, width=64)["color"][..., :3]
  assert downsampled_rgb.shape == (64, 64, 3)
  np.testing.assert_allclose(
      render(denoise=True, width=64)["color"][..., :3],
      downsampled_rgb,
      rtol=1e-5,
      atol=1e-5,
  )
  np.testing.assert_allclose(
      render(denoise=True)["color"][..., :3],
      denoised_rgb,
      rtol=1e-5,
      atol=1e-5,
  )
  assert render(denoise=True, width=32)["color"].shape == (32, 32, 4)

  # Requesting normals without albedo falls back to unguided color denoising
  # while preserving the output normal AOV.
  UsdRender.Product(
      stage.GetPrimAtPath("/root/Render/Products/MainOutput")
  ).GetOrderedVarsRel().SetTargets([
      Sdf.Path("/root/Render/Vars/color"),
      Sdf.Path("/root/Render/Vars/normal"),
  ])
  unguided_results = render(denoise=True)
  unguided_rgb = unguided_results["color"][..., :3]
  assert not np.allclose(unguided_rgb, noisy_rgb)
  assert not np.allclose(denoised_rgb, unguided_rgb)
  unguided_gradient = np.mean(np.square(np.diff(unguided_rgb, axis=1)))
  assert unguided_gradient < noisy_gradient
  np.testing.assert_allclose(
      unguided_results["normal"], noisy_results["normal"], atol=1e-5
  )
