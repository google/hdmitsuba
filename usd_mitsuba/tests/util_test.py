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

"""Tests for utility functions."""

from __future__ import annotations

from pxr import Sdf
from pxr import Usd
from usd_mitsuba import util


def test_get_mitsuba_id():
  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/MyObj")
  assert util.get_mitsuba_id(prim) == "_World_MyObj"


def test_extract_nested_dict():
  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/Config")
  prim.CreateAttribute("custom:nested:val1", Sdf.ValueTypeNames.Int).Set(42)
  prim.CreateAttribute(
      "custom:nested:sub:val2", Sdf.ValueTypeNames.String
  ).Set("hello")
  prim.CreateAttribute("ignored:attribute", Sdf.ValueTypeNames.Int).Set(10)

  extracted = util.extract_nested_dict(prim, "custom:nested:")
  assert extracted == {
      "val1": 42,
      "sub": {"val2": "hello"},
  }
