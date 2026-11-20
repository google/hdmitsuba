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
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import Vt

from usd_mitsuba import util


def test_extract_nested_dict():
  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/TestPrim")

  for name, type_name, value in (
      ("custom:nested:val1", Sdf.ValueTypeNames.Int, 42),
      ("custom:nested:sub:val2", Sdf.ValueTypeNames.String, "hello"),
      ("ignored:attribute", Sdf.ValueTypeNames.Int, 10),
      ("custom:nested:blocked", Sdf.ValueTypeNames.Float, Sdf.ValueBlock()),
      # Non-string / empty `:type` and `:id` must not create empty child dicts.
      ("custom:nested:bad_child:type", Sdf.ValueTypeNames.Int, 7),
      ("custom:nested:empty_child:id", Sdf.ValueTypeNames.String, ""),
  ):
    prim.CreateAttribute(name, type_name).Set(value)

  time_only = prim.CreateAttribute(
      "custom:nested:time_only", Sdf.ValueTypeNames.Float
  )
  time_only.Set(3.5, Usd.TimeCode(5.0))

  assert util.extract_nested_dict(prim, "custom:nested:") == {
      "val1": 42,
      "sub": {"val2": "hello"},
  }
  extracted_frame5 = util.extract_nested_dict(
      prim, "custom:nested:", time=Usd.TimeCode(5.0)
  )
  assert extracted_frame5["time_only"] == 3.5


def test_extract_nested_dict_gf_roles_spectra_and_transforms(tmp_path):
  spd_file = tmp_path / "test_spd.spd"
  spd_file.write_text("400 0.5\n700 1.0\n")

  stage = Usd.Stage.CreateInMemory()
  prim = stage.DefinePrim("/World/Plugin")
  mat = Gf.Matrix4d(1.0)
  mat.SetTranslate(Gf.Vec3d(1.0, 2.0, 3.0))

  for name, type_name, value in (
      ("mitsuba:emitter:radiance:type", Sdf.ValueTypeNames.String, "rgb"),
      (
          "mitsuba:emitter:radiance:value",
          Sdf.ValueTypeNames.Color3f,
          Gf.Vec3f(1.0, 2.0, 3.0),
      ),
      (
          "mitsuba:emitter:origin",
          Sdf.ValueTypeNames.Point3f,
          Gf.Vec3f(4.0, 5.0, 6.0),
      ),
      (
          "mitsuba:emitter:direction",
          Sdf.ValueTypeNames.Vector3f,
          Gf.Vec3f(0.0, 0.0, -1.0),
      ),
      (
          "mitsuba:emitter:tint",
          Sdf.ValueTypeNames.Color3f,
          Gf.Vec3f(0.2, 0.4, 0.6),
      ),
      ("mitsuba:emitter:up:type", Sdf.ValueTypeNames.String, "vector"),
      (
          "mitsuba:emitter:up:value",
          Sdf.ValueTypeNames.Float3,
          Gf.Vec3f(0.0, 1.0, 0.0),
      ),
      ("mitsuba:emitter:sub_light:type", Sdf.ValueTypeNames.String, "point"),
      (
          "mitsuba:emitter:spec_pairs:type",
          Sdf.ValueTypeNames.String,
          "spectrum",
      ),
      (
          "mitsuba:emitter:spec_pairs:value",
          Sdf.ValueTypeNames.Float2Array,
          Vt.Vec2fArray([Gf.Vec2f(400.0, 0.5), Gf.Vec2f(700.0, 1.5)]),
      ),
      ("mitsuba:emitter:spec_str:type", Sdf.ValueTypeNames.String, "spectrum"),
      (
          "mitsuba:emitter:spec_str:value",
          Sdf.ValueTypeNames.String,
          "400:0.5 700:1.5",
      ),
      ("mitsuba:emitter:spec_file:type", Sdf.ValueTypeNames.String, "spectrum"),
      (
          "mitsuba:emitter:spec_file:filename",
          Sdf.ValueTypeNames.Asset,
          Sdf.AssetPath(str(spd_file)),
      ),
      (
          "mitsuba:emitter:weights",
          Sdf.ValueTypeNames.FloatArray,
          Vt.FloatArray([0.25, 0.75]),
      ),
      ("mitsuba:emitter:to_world", Sdf.ValueTypeNames.Matrix4d, mat),
  ):
    prim.CreateAttribute(name, type_name).Set(value)

  extracted = util.extract_nested_dict(prim, "mitsuba:emitter:")
  assert extracted["radiance"]["type"] == "rgb"
  assert isinstance(extracted["radiance"]["value"], mi.scalar_rgb.ScalarColor3f)
  np.testing.assert_allclose(extracted["radiance"]["value"], [1.0, 2.0, 3.0])
  assert extracted["origin"] == (4.0, 5.0, 6.0)
  assert extracted["direction"] == (0.0, 0.0, -1.0)
  assert isinstance(extracted["tint"], mi.scalar_rgb.ScalarColor3f)
  np.testing.assert_allclose(extracted["tint"], [0.2, 0.4, 0.6])
  assert extracted["up"] == (0.0, 1.0, 0.0)
  assert extracted["sub_light"] == {"type": "point"}
  expected_pairs = {"type": "spectrum", "value": [(400.0, 0.5), (700.0, 1.5)]}
  assert extracted["spec_pairs"] == expected_pairs
  assert extracted["spec_str"] == expected_pairs
  assert extracted["spec_file"] == {
      "type": "spectrum",
      "filename": str(spd_file),
  }
  assert extracted["weights"] == "0.25,0.75"
  assert isinstance(
      extracted["to_world"], mi.scalar_rgb.ScalarAffineTransform4f
  )
  expected_matrix = np.eye(4, dtype=np.float32)
  expected_matrix[:3, 3] = [1.0, 2.0, 3.0]
  np.testing.assert_allclose(
      np.array(extracted["to_world"].matrix), expected_matrix
  )
  if mi.variant() is None:
    mi.set_variant("scalar_rgb")
  assert mi.load_dict(extracted["spec_file"]) is not None


def test_deep_merge_dict():
  base = {
      "type": "hdrfilm",
      "width": 200,
      "rfilter": {"type": "gaussian", "stddev": 1.0},
  }
  util.deep_merge_dict(
      base,
      {
          "crop_width": 50,
          "rfilter": {"type": "box"},
      },
  )
  assert base == {
      "type": "hdrfilm",
      "width": 200,
      "crop_width": 50,
      "rfilter": {"type": "box"},
  }
