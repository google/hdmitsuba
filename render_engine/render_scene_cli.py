#!/usr/bin/env python3
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

"""Simple CLI that renders a fixed USD scene using Mitsuba."""

from __future__ import annotations

import argparse
import os

import drjit as dr
import mitsuba as mi
import numpy as np

from pxr import Sdf
from pxr import Tf
from pxr import Trace
from pxr import Usd
from pxr import UsdRender
import usd_render

_RENDER_SETTINGS_PATH = '/renderSettings'


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Simple CLI that renders a fixed scene."
  )
  parser.add_argument(
      '--scene',
      required=True,
      help='Scene path (e.g. .usd file).',
  )
  parser.add_argument(
      '--renderer',
      default='HdMitsubaRendererPlugin',
      help='Renderer plugin name.',
  )
  parser.add_argument(
      '--extension',
      default='exr',
      help='Image extension.',
  )
  parser.add_argument(
      '--camera_path',
      default=None,
      help='Camera path.',
  )
  parser.add_argument(
      '--variant',
      default='cuda_ad_rgb',
      help='Mitsuba variant.',
  )
  parser.add_argument(
      '--output',
      default=None,
      help=(
          'Output image path. If not set, writes to'
          ' /tmp/<scene_name>_<renderer>.<extension>.'
      ),
  )
  parser.add_argument(
      '--spp',
      type=int,
      default=None,
      help='Samples per pixel. If not set, uses default.',
  )
  parser.add_argument(
      '--profile',
      action='store_true',
      help='Enable USD tracing and output to <output_path>.trace.json.',
  )
  parser.add_argument(
      '--drjit_log',
      default=None,
      help='Dr.Jit log level (disable, error, warn, info, debug, trace).',
  )
  parser.add_argument(
      '--mitsuba_log',
      default=None,
      help='Mitsuba log level (disable, error, warn, info, debug, trace).',
  )

  args = parser.parse_args()

  if args.drjit_log:
    log_level = getattr(dr.LogLevel, args.drjit_log.capitalize(), None)
    if log_level is not None:
      dr.set_log_level(log_level)
    else:
      Tf.Warn(f"Unknown Dr.Jit log level: {args.drjit_log}")

  if args.mitsuba_log:
    log_level = getattr(mi.LogLevel, args.mitsuba_log.capitalize(), None)
    if log_level is not None:
      mi.set_log_level(log_level)
    else:
      Tf.Warn(f"Unknown Mitsuba log level: {args.mitsuba_log}")

  if args.profile:
    Trace.Collector().enabled = True
    Trace.Collector().pythonTracingEnabled = True

  mi.set_variant(args.variant)

  renderer = args.renderer
  extension = args.extension

  scene_path = args.scene
  scene_name = os.path.splitext(os.path.basename(scene_path))[0]
  stage = Usd.Stage.Open(scene_path)
  camera_path = args.camera_path
  render_settings = UsdRender.Settings.Define(stage, _RENDER_SETTINGS_PATH)
  render_settings.GetPrim().CreateAttribute(
      'mitsuba:variant', Sdf.ValueTypeNames.String
  ).Set(mi.variant())
  stage.SetMetadata(
      UsdRender.Tokens.renderSettingsPrimPath, _RENDER_SETTINGS_PATH
  )

  overrides = {}
  if args.spp is not None:
    overrides['mitsuba:sample_count'] = args.spp

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id=renderer,
      width=512,
      camera_path=camera_path,
      overrides=overrides,
  )
  outputs = engine.render()

  dr.sync_thread()
  image = outputs['color']
  print(image.shape)
  print(np.max(image))
  output_path = args.output
  if not output_path:
    output_path = f'/tmp/{scene_name}_{renderer}.{extension}'

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  ext = os.path.splitext(output_path)[1].lower().lstrip('.')
  bmp = mi.Bitmap(image)
  if ext in ['png', 'jpg', 'jpeg']:
    bmp = bmp.convert(component_format=mi.Bitmap.UInt8, srgb_gamma=True)
  bmp.write(output_path)

  if args.profile:
    Trace.Collector().enabled = False
    reporter = Trace.Reporter.globalReporter
    reporter.UpdateTraceTrees()
    trace_path = output_path + '.trace.json'
    reporter.ReportChromeTracingToFile(trace_path)
    print(f"USD Trace written to {trace_path}")


if __name__ == '__main__':
  main()
