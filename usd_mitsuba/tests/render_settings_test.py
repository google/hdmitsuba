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

"""Tests for the Mitsuba render settings parsing."""

from __future__ import annotations

import pytest
from pxr import Usd
from usd_mitsuba import render_settings

_TEST_ASSET_PATH = "hdmitsuba/test_assets/render_settings"


@pytest.mark.parametrize(
    "filename, render_product_path, expected_color, expected_aovs",
    [
        (
            "aovs",
            None,
            True,
            ["albedo:albedo", "normal:sh_normal", "depth:depth"],
        ),
        (
            "aovs_multiple_products",
            None,
            True,
            [],
        ),
        (
            "aovs_multiple_products",
            "/root/Render/Products/AovOutput",
            False,
            ["albedo:albedo", "normal:sh_normal", "depth:depth"],
        ),
    ],
)
def test_parse_aovs(
    filename, render_product_path, expected_color, expected_aovs
):
  stage = Usd.Stage.Open(f"{_TEST_ASSET_PATH}/{filename}.usda")
  render_settings_prim = render_settings.get_render_settings(stage)
  assert render_settings_prim is not None
  render_color, aovs = render_settings.parse_aovs(
      stage, render_settings_prim, render_product_path
  )
  assert render_color == expected_color
  assert list(aovs) == expected_aovs


def test_parse_aovs_invalid_render_product_path():
  stage = Usd.Stage.Open(f"{_TEST_ASSET_PATH}/aovs_multiple_products.usda")
  render_settings_prim = render_settings.get_render_settings(stage)
  assert render_settings_prim is not None
  with pytest.raises(ValueError, match="Invalid render product"):
    render_settings.parse_aovs(
        stage, render_settings_prim, "/root/Render/Products/InvalidOutput"
    )
