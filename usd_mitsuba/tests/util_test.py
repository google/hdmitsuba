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

import mitsuba as mi
import numpy as np

from usd_mitsuba import util
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import Vt


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
  prim.CreateAttribute(
      "custom:nested:blocked", Sdf.ValueTypeNames.Float
  ).Set(Sdf.ValueBlock())
  time_only = prim.CreateAttribute(
      "custom:nested:time_only", Sdf.ValueTypeNames.Float
  )
  time_only.Set(3.5, Usd.TimeCode(5.0))
  # Non-string / empty `:type` and `:id` must not create empty child dicts.
  prim.CreateAttribute(
      "custom:nested:bad_child:type", Sdf.ValueTypeNames.Int
  ).Set(7)
  prim.CreateAttribute(
      "custom:nested:empty_child:id", Sdf.ValueTypeNames.String
  ).Set("")

  extracted_default = util.extract_nested_dict(prim, "custom:nested:")
  assert extracted_default == {
      "val1": 42,
      "sub": {"val2": "hello"},
  }

  extracted_frame5 = util.extract_nested_dict(
      prim, "custom:nested:", time=Usd.TimeCode(5.0)
  )
  assert extracted_frame5["time_only"] == 3.5


def test_extract_nested_dict_gf_roles_spectra_and_transforms():
  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/Plugin")
  prim.CreateAttribute(
      "mitsuba:emitter:radiance:type", Sdf.ValueTypeNames.String
  ).Set("rgb")
  prim.CreateAttribute(
      "mitsuba:emitter:radiance:value", Sdf.ValueTypeNames.Color3f
  ).Set(Gf.Vec3f(1.0, 2.0, 3.0))
  # Role-tagged spatial attributes map directly to 3-tuples (Mitsuba
  # Vector/Point) without requiring a `:type = "vector"` child block, while
  # Color3f maps to ScalarColor3f.
  prim.CreateAttribute(
      "mitsuba:emitter:origin", Sdf.ValueTypeNames.Point3f
  ).Set(Gf.Vec3f(4.0, 5.0, 6.0))
  prim.CreateAttribute(
      "mitsuba:emitter:direction", Sdf.ValueTypeNames.Vector3f
  ).Set(Gf.Vec3f(0.0, 0.0, -1.0))
  prim.CreateAttribute(
      "mitsuba:emitter:tint", Sdf.ValueTypeNames.Color3f
  ).Set(Gf.Vec3f(0.2, 0.4, 0.6))
  # Explicit `:type = "vector"` also normalizes to a 3-tuple, whereas
  # `:type = "point"` stays a Mitsuba plugin dictionary (e.g. point light).
  prim.CreateAttribute(
      "mitsuba:emitter:up:type", Sdf.ValueTypeNames.String
  ).Set("vector")
  prim.CreateAttribute(
      "mitsuba:emitter:up:value", Sdf.ValueTypeNames.Float3
  ).Set(Gf.Vec3f(0.0, 1.0, 0.0))
  prim.CreateAttribute(
      "mitsuba:emitter:sub_light:type", Sdf.ValueTypeNames.String
  ).Set("point")
  prim.CreateAttribute(
      "mitsuba:emitter:spec_pairs:type", Sdf.ValueTypeNames.String
  ).Set("spectrum")
  prim.CreateAttribute(
      "mitsuba:emitter:spec_pairs:value", Sdf.ValueTypeNames.Float2Array
  ).Set(Vt.Vec2fArray([Gf.Vec2f(400.0, 0.5), Gf.Vec2f(700.0, 1.5)]))
  prim.CreateAttribute(
      "mitsuba:emitter:weights", Sdf.ValueTypeNames.FloatArray
  ).Set(Vt.FloatArray([0.25, 0.75]))

  mat = Gf.Matrix4d(1.0)
  mat.SetTranslate(Gf.Vec3d(1.0, 2.0, 3.0))
  prim.CreateAttribute(
      "mitsuba:emitter:to_world", Sdf.ValueTypeNames.Matrix4d
  ).Set(mat)

  extracted = util.extract_nested_dict(prim, "mitsuba:emitter:")
  mi_mod = mi if mi.variant() is not None else mi.scalar_rgb
  assert extracted["radiance"]["type"] == "rgb"
  assert isinstance(extracted["radiance"]["value"], mi_mod.ScalarColor3f)
  np.testing.assert_allclose(extracted["radiance"]["value"], [1.0, 2.0, 3.0])
  assert extracted["origin"] == (4.0, 5.0, 6.0)
  assert extracted["direction"] == (0.0, 0.0, -1.0)
  assert isinstance(extracted["tint"], mi_mod.ScalarColor3f)
  np.testing.assert_allclose(extracted["tint"], [0.2, 0.4, 0.6])
  assert extracted["up"] == (0.0, 1.0, 0.0)
  assert extracted["sub_light"] == {"type": "point"}
  assert extracted["spec_pairs"] == {
      "type": "spectrum",
      "value": [(400.0, 0.5), (700.0, 1.5)],
  }
  assert extracted["weights"] == "0.25,0.75"
  assert isinstance(extracted["to_world"], mi_mod.ScalarAffineTransform4f)
  # USD row-vector translation (row 3) transposes to column 3 in Mitsuba.
  expected_matrix = np.eye(4, dtype=np.float32)
  expected_matrix[:3, 3] = [1.0, 2.0, 3.0]
  np.testing.assert_allclose(
      np.array(extracted["to_world"].matrix), expected_matrix
  )


def test_extract_nested_dict_resolves_spectrum_filename(tmp_path):
  if mi.variant() is None:
    mi.set_variant("scalar_rgb")
  spd_file = tmp_path / "test_spd.spd"
  spd_file.write_text("400 0.5\n700 1.0\n")

  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/Emitter")
  prim.CreateAttribute(
      "mitsuba:emitter:spec_file:type", Sdf.ValueTypeNames.String
  ).Set("spectrum")
  prim.CreateAttribute(
      "mitsuba:emitter:spec_file:filename", Sdf.ValueTypeNames.String
  ).Set("test_spd.spd")

  old_resolver = mi.file_resolver()
  scoped_resolver = mi.FileResolver(old_resolver)
  scoped_resolver.append(str(tmp_path))
  mi.set_file_resolver(scoped_resolver)
  try:
    extracted = util.extract_nested_dict(prim, "mitsuba:emitter:")
  finally:
    mi.set_file_resolver(old_resolver)

  assert extracted["spec_file"] == {
      "type": "spectrum",
      "filename": str(spd_file),
  }
  assert mi.load_dict(extracted["spec_file"]) is not None
