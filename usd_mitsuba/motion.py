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

"""Utilities for transformation motion blur."""

from __future__ import annotations

from collections.abc import Callable

import mitsuba as mi

from pxr import Gf
from pxr import Usd
from pxr import UsdGeom

from hdmitsuba import geometry_ext as geom_lib
from usd_mitsuba import util


def get_shutter_interval(
    camera: UsdGeom.Camera, time: Usd.TimeCode
) -> tuple[float, float]:
  """Returns the camera shutter interval, or (0, 0) if motion blur is off."""
  open_time = camera.GetShutterOpenAttr().Get(time)
  close_time = camera.GetShutterCloseAttr().Get(time)
  return (open_time, close_time) if close_time > open_time else (0.0, 0.0)


def get_motion_interval(
    stage: Usd.Stage, time: Usd.TimeCode
) -> tuple[float, float]:
  """Returns the union of all active camera shutter intervals, or (0, 0)."""
  shutters = [
      s
      for prim in stage.Traverse(Usd.TraverseInstanceProxies())
      if prim.IsA(UsdGeom.Camera)
      and (s := get_shutter_interval(UsdGeom.Camera(prim), time)) != (0.0, 0.0)
  ]
  if not shutters:
    return (0.0, 0.0)
  opens, closes = zip(*shutters)
  return (min(opens), max(closes))


def sample_world_transform(
    prim: Usd.Prim,
    time: Usd.TimeCode,
    interval: tuple[float, float],
    transform_fn: Callable[
        [Gf.Matrix4d], mi.ScalarTransform4f
    ] = util.to_mitsuba_transform,
) -> mi.ScalarTransform4f | mi.AnimatedTransform4f:
  """Samples `prim`'s world transform over `interval` as a static or animated transform."""
  keyframes = [
      (t, transform_fn(m))
      for t, m in geom_lib.sample_world_transform(prim, time, interval)
  ]
  if all(m == keyframes[0][1] for _, m in keyframes[1:]):
    return keyframes[0][1]
  return mi.AnimatedTransform4f(keyframes)
