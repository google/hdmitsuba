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

"""Tests for the motion blur utilities."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from usd_mitsuba import motion


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant("scalar_rgb")


def test_sample_world_transform():
  stage = Usd.Stage.CreateInMemory()
  op = UsdGeom.Xform.Define(stage, "/World").AddTranslateOp()
  for time, x in [(0.0, 0.0), (0.1, 1.0), (1.0, 1.0), (2.0, 1.0)]:
    op.Set((x, 0, 0), time)
  child = UsdGeom.Xform.Define(stage, "/World/Child")
  child.AddTranslateOp().Set((0, 1, 0))
  prim = child.GetPrim()

  # The child inherits the motion and interior keyframes of its parent.
  transform = motion.sample_world_transform(
      prim, Usd.TimeCode(0.5), (-0.5, 0.5)
  )
  assert isinstance(transform, mi.AnimatedTransform4f)
  np.testing.assert_allclose(
      mi.traverse(transform)["times"], [-0.5, -0.4, 0.5]
  )
  assert list(transform.eval(-0.5).translation()) == [0, 1, 0]
  assert list(transform.eval(0.5).translation()) == [1, 1, 0]

  # With `uniform=True`, non-uniform interior keyframes are resampled onto
  # `mitsuba:motion_samples` (default 32) uniform keyframes.
  uniform_transform = motion.sample_world_transform(
      prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=True
  )
  assert isinstance(uniform_transform, mi.AnimatedTransform4f)
  np.testing.assert_allclose(
      mi.traverse(uniform_transform)["times"], np.linspace(-0.5, 0.5, 32)
  )

  # `primvars:mitsuba:motion_samples` is inherited from ancestors and controls
  # the sample count (1 disables motion blur on the prim).
  motion_samples_pv = UsdGeom.PrimvarsAPI(
      stage.GetPrimAtPath("/World")
  ).CreatePrimvar("mitsuba:motion_samples", Sdf.ValueTypeNames.Int)
  motion_samples_pv.Set(11)
  uniform_11 = motion.sample_world_transform(
      prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=True
  )
  np.testing.assert_allclose(
      mi.traverse(uniform_11)["times"], np.linspace(-0.5, 0.5, 11)
  )
  np.testing.assert_allclose(uniform_11.eval(-0.4).translation(), [1, 1, 0])

  motion_samples_pv.Set(1)
  assert isinstance(
      motion.sample_world_transform(
          prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=True
      ),
      mi.ScalarTransform4f,
  )
  motion_samples_pv.GetAttr().Clear()

  # Linear motion with an off-grid interior keyframe only needs the endpoints
  # under `uniform=True`.
  op.Set((0.4, 0, 0), 0.4)
  op.Set((0.1, 0, 0), 0.1)
  np.testing.assert_allclose(
      mi.traverse(
          motion.sample_world_transform(
              prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=True
          )
      )["times"],
      [-0.5, 0.5],
  )
  op.Set((1.0, 0, 0), 0.1)
  op.GetAttr().ClearAtTime(0.4)

  # Static cases: constant transform_fn, empty interval, default time, or
  # interval over which the value is constant.
  assert isinstance(
      motion.sample_world_transform(
          prim, Usd.TimeCode(0.5), (-0.5, 0.5), lambda _: mi.ScalarTransform4f()
      ),
      mi.ScalarTransform4f,
  )
  for t, interval in [
      (Usd.TimeCode(0.5), (0.0, 0.0)),
      (Usd.TimeCode.Default(), (-0.5, 0.5)),
      (Usd.TimeCode(1.5), (-0.25, 0.25)),
  ]:
    assert isinstance(
        motion.sample_world_transform(prim, t, interval), mi.ScalarTransform4f
    )

  # Resetting the transform stack ignores the motion of the parent.
  child.SetResetXformStack(True)
  transform = motion.sample_world_transform(
      prim, Usd.TimeCode(0.5), (-0.5, 0.5)
  )
  assert isinstance(transform, mi.ScalarTransform4f)
  assert list(transform.translation()) == [0, 1, 0]


def test_sample_world_transform_subdivides_curved_motion():
  stage = Usd.Stage.CreateInMemory()
  rotate_op = UsdGeom.Xform.Define(stage, "/Pivot").AddRotateYOp()
  rotate_op.Set(0.0, 0.0)
  rotate_op.Set(360.0, 1.0)
  child = UsdGeom.Xform.Define(stage, "/Pivot/Child")
  child.AddTranslateOp().Set((2, 0, 0))
  prim = child.GetPrim()

  # A 360-degree orbit requires intermediate keyframes and tracks the circle
  # for both adaptive (`uniform=False`) and lockstep (`uniform=True`) sampling.
  # The chord error of the 31 segments of the default 32-sample budget is
  # ~1.03e-2, hence the tolerance.
  for uniform in (False, True):
    transform = motion.sample_world_transform(
        prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=uniform
    )
    assert isinstance(transform, mi.AnimatedTransform4f)
    np.testing.assert_allclose(
        transform.eval(-0.25).translation(), [0, 0, -2], atol=2e-2
    )
    np.testing.assert_allclose(
        transform.eval(0.0).translation(), [-2, 0, 0], atol=2e-2
    )

  # Capping `primvars:mitsuba:motion_samples` limits adaptive subdivision.
  UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
      "mitsuba:motion_samples", Sdf.ValueTypeNames.Int
  ).Set(5)
  for uniform in (False, True):
    capped = motion.sample_world_transform(
        prim, Usd.TimeCode(0.5), (-0.5, 0.5), uniform=uniform
    )
    np.testing.assert_allclose(
        mi.traverse(capped)["times"], np.linspace(-0.5, 0.5, 5)
    )


def test_sample_world_transform_snaps_uniform_keyframes():
  # An interior keyframe slightly off the uniform grid (within the seeding
  # tolerance) must not leak into the keyframe times: Mitsuba's `instance`
  # plugin verifies uniform spacing with a tighter tolerance.
  stage = Usd.Stage.CreateInMemory()
  op = UsdGeom.Xform.Define(stage, "/Orbit").AddTranslateOp()
  op.Set((-1, 0, 0), 0.0)
  op.Set((0, 1, 0), 0.50005)
  op.Set((1, 0, 0), 1.0)
  to_world = motion.sample_world_transform(
      stage.GetPrimAtPath("/Orbit"), Usd.TimeCode(0.5), (-0.5, 0.5),
      uniform=True,
  )
  times = mi.traverse(to_world)["times"]
  assert len(times) > 2
  np.testing.assert_allclose(times, np.linspace(-0.5, 0.5, len(times)))
  mi.load_dict({
      "type": "instance",
      "shapegroup": {"type": "shapegroup", "shape": {"type": "sphere"}},
      "to_world": to_world,
  })


def test_get_motion_interval():
  stage = Usd.Stage.CreateInMemory()
  assert motion.get_motion_interval(stage, Usd.TimeCode(0.0)) == (0.0, 0.0)
  # (1.0, -1.0) has an inverted shutter and does not contribute.
  for i, (open_t, close_t) in enumerate(
      [(-0.25, 0.0), (0.0, 0.5), (1.0, -1.0)]
  ):
    cam = UsdGeom.Camera.Define(stage, f"/Camera{i}")
    cam.GetShutterOpenAttr().Set(open_t)
    cam.GetShutterCloseAttr().Set(close_t)
  assert motion.get_motion_interval(stage, Usd.TimeCode(0.0)) == (-0.25, 0.5)

