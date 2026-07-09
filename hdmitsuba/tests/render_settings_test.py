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
      ("albedo", slice(0, 3), 0.05),
      ("normal", slice(3, 6), 0.05),
      ("depth", slice(6, 7), 0.5),
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
  settings_prim.CreateAttribute(
      "mitsuba:integrator:type", Sdf.ValueTypeNames.String
  ).Set("path")

  # Render full image first
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id="HdMitsubaRendererPlugin")
  image_full = engine.render()["color"]

  # Configure asymmetric crop rendering (x: [0.25, 0.75], y: [0.1, 0.4] in Y-up NDC)
  # y range 0.1 to 0.4 maps to rows [51, 204] in Y-up pixel coordinates.
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
      crop_region_crop, crop_region_full, atol=0.1
  )

  # Verify everything else is black
  mask = np.ones((512, 512, 3), dtype=bool)
  mask[307:461, 128:384, :] = False
  mismatched_coords = np.argwhere((image_crop[..., :3] != 0.0) & mask)

  non_zero_coords = np.argwhere(image_crop[..., :3] != 0.0)
  if len(non_zero_coords) > 0:
    min_y, min_x, _ = non_zero_coords.min(axis=0)
    max_y, max_x, _ = non_zero_coords.max(axis=0)
    print(
        f"Asymmetric crop non-zero bounding box: y in [{min_y}, {max_y}], x"
        f" in [{min_x}, {max_x}]"
    )
  assert len(mismatched_coords) == 0


