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

r"""Generates OSL reference values for MaterialX pattern nodes.

For each targeted MaterialX library node definition, this script instantiates
the node with a few deterministic input samples, generates an OSL shader with
MaterialXGenOsl, and evaluates it with OSL's `testshade`. The results are
written to `osl_reference.json`, which `materialx_arithmetic_test.py` compares
against hdMitsuba's MaterialX expression compiler and evaluator.

This script only needs to be rerun when changing the sampled node definitions
or upgrading MaterialX. Example usage:

  pip3 install --target ~/.local/mtlxref materialx==1.39.5
  PYTHONPATH=~/.local/mtlxref:$PYTHONPATH python3 generate_osl_reference.py \
      --osl_bin ~/.local/osl/bin --output osl_reference.json
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import subprocess
import tempfile
from typing import Any
import zlib

from MaterialX import PyMaterialXGenOsl as mx_gen_osl
from MaterialX import PyMaterialXGenShader as mx_gen_shader
import MaterialX as mx
import numpy as np

_NUM_RANDOM_SAMPLES = 3

_OUTPUT_COMPONENTS = {
    'boolean': ('o',),
    'float': ('o',),
    'vector2': ('o.x', 'o.y'),
    'color3': ('o[0]', 'o[1]', 'o[2]'),
    'vector3': ('o[0]', 'o[1]', 'o[2]'),
    'color4': ('o.rgb[0]', 'o.rgb[1]', 'o.rgb[2]', 'o.a'),
    'vector4': ('o.x', 'o.y', 'o.z', 'o.w'),
}

# Non-standard, procedural, or MaterialX-specific NodeDefs where OSL parity
# testing complements the unit tests in materialx_arithmetic_test.py.
_TARGETED_NODEDEFS = frozenset({
    # 2D procedural shapes & ramps (stdlib procedural graphs).
    'ND_ramplr_color3',
    'ND_ramptb_color3',
    'ND_splitlr_color3',
    'ND_splittb_color3',
    'ND_ramp4_color3',
    'ND_ramp',
    'ND_checkerboard_color3',
    'ND_circle_float',
    'ND_cloverleaf_float',
    'ND_crosshatch_color3',
    'ND_grid_color3',
    'ND_hexagon_float',
    'ND_line_float',
    'ND_trianglewave_float',
    # Non-trivial color science, PBR helpers, & compositing graphs.
    'ND_hsvadjust_color3',
    'ND_saturate_color3',
    'ND_colorcorrect_color3',
    'ND_contrast_color3',
    'ND_range_color3',
    'ND_luminance_color3',
    'ND_luminance_color4',
    'ND_blackbody',
    'ND_artistic_ior',
    'ND_roughness_anisotropy',
    'ND_roughness_dual',
    'ND_glossiness_anisotropy',
    'ND_open_pbr_anisotropy',
    'ND_srgb_texture_to_lin_rec709_color3',
    'ND_acescg_to_lin_rec709_color3',
    'ND_over_color4',
    'ND_in_color4',
    'ND_out_color4',
    'ND_mask_color4',
    'ND_matte_color4',
    'ND_disjointover_color4',
    'ND_burn_color3',
    'ND_dodge_color3',
    'ND_screen_color3',
    'ND_overlay_color3',
    'ND_difference_color3',
    'ND_plus_color3',
    'ND_minus_color3',
    # Coordinate transforms, normal mapping, & shader-model translation graphs.
    'ND_rotate2d_vector2',
    'ND_rotate3d_vector3',
    'ND_place2d_vector2',
    'ND_UsdTransform2d',
    'ND_normalmap_float',
    'ND_normalmap_vector2',
    'ND_reflect_vector3',
    'ND_refract_vector3',
    'ND_open_pbr_surface_to_standard_surface',
    'ND_standard_surface_to_open_pbr_surface',
    'ND_standard_surface_to_UsdPreviewSurface',
})

_DEFAULT_RANGE = (-1.0, 2.0)

# Sampling ranges of inputs by name, which apply to all node categories.
_INPUT_RANGES = {
    'amount': (-1.0, 1.0),
    'anisotropy': (0.0, 1.0),
    'edge_color': (0.05, 0.95),
    'gamma': (0.3, 3.0),
    'mix': (0.0, 1.0),
    'radius': (0.05, 1.0),
    'reflectivity': (0.05, 0.95),
    'roughness': (0.0, 1.0),
    'temperature': (1500.0, 12000.0),
    'texcoord': (-2.0, 2.0),
}

# Per-category overrides restricting inputs to valid, non-degenerate domains.
# The key '*' applies to all inputs of the category.
_CATEGORY_INPUT_RANGES = {
    'circle': {
        'texcoord': (0.0, 1.0),
        'center': (0.2, 0.8),
        'radius': (0.1, 0.5),
    },
    'cloverleaf': {
        'texcoord': (0.0, 1.0),
        'center': (0.2, 0.8),
        'radius': (0.2, 0.6),
    },
    'disjointover': {'fg': (0.3, 1.0), 'bg': (0.4, 1.0)},
    'hexagon': {
        'texcoord': (0.0, 1.0),
        'center': (0.2, 0.8),
        'radius': (0.2, 0.6),
    },
    'hsvadjust': {'in': (0.0, 1.0), 'amount': (0.0, 1.0)},
    'line': {
        'texcoord': (0.2, 0.8),
        'center': (-0.1, 0.1),
        'point1': (0.2, 0.6),
        'point2': (0.4, 0.8),
        'radius': (0.2, 0.5),
    },
    'open_pbr_surface_to_standard_surface': {'*': (0.0, 1.0)},
    'place2d': {'scale': (0.25, 2.0), 'rotate': (-180.0, 180.0)},
    'ramp': {'texcoord': (0.0, 1.0)},
    'ramp4': {'texcoord': (0.0, 1.0)},
    'ramplr': {'texcoord': (0.0, 1.0)},
    'ramptb': {'texcoord': (0.0, 1.0)},
    'refract': {'ior': (1.0, 2.0)},
    'rotate2d': {'amount': (-180.0, 180.0)},
    'rotate3d': {'amount': (-180.0, 180.0)},
    'splitlr': {'texcoord': (0.0, 1.0), 'center': (0.2, 0.8)},
    'splittb': {'texcoord': (0.0, 1.0), 'center': (0.2, 0.8)},
    'standard_surface_to_open_pbr_surface': {'*': (0.0, 1.0)},
    'standard_surface_to_UsdPreviewSurface': {'*': (0.0, 1.0)},
}

# Groups of inputs sorted per channel after sampling so ordering preconditions
# hold across MaterialX backends.
_SORTED_INPUTS = {
    'range': [('outlow', 'outhigh')],
    'ramp': [tuple(f'interval{i}' for i in range(1, 11))],
}

# Inputs normalized after sampling (directions and normals).
_UNIT_VECTOR_INPUTS = frozenset({'axis', 'bitangent', 'normal', 'tangent'})


def _input_range(
    category: str,
    inp: mx.Input,
    channel: int,
    default: tuple[float, float] = _DEFAULT_RANGE,
) -> tuple[float, float]:
  """Returns the sampling range of a channel of `inp`."""
  name = inp.getName()
  overrides = _CATEGORY_INPUT_RANGES.get(category, {})
  if (override := overrides.get(name, overrides.get('*'))) is not None:
    return override
  lo = inp.getAttribute('uisoftmin') or inp.getAttribute('uimin')
  hi = inp.getAttribute('uisoftmax') or inp.getAttribute('uimax')
  default_lo, default_hi = _INPUT_RANGES.get(name, default)
  if not lo and not hi:
    return default_lo, default_hi

  def _val(s: str, fallback: float) -> float:
    return float(s.split(',')[min(channel, s.count(','))]) if s else fallback

  lo_val = _val(lo, default_lo)
  return lo_val, max(_val(hi, default_hi), lo_val + 1.0)


def _sample_input(rng: random.Random, category: str, inp: mx.Input) -> str:
  """Returns a random MaterialX value string for the input `inp`."""
  name, typ = inp.getName(), inp.getType()
  if typ == 'boolean':
    return rng.choice(['true', 'false'])
  if typ == 'integer':
    if inp.hasAttribute('enumvalues'):
      return rng.choice(inp.getAttribute('enumvalues').split(',')).strip()
    lo, hi = _input_range(category, inp, 0, default=(0.0, 3.0))
    return str(rng.randint(int(lo), int(hi)))
  size = len(_OUTPUT_COMPONENTS[typ])
  values = np.array(
      [rng.uniform(*_input_range(category, inp, c)) for c in range(size)],
      dtype=np.float64,
  )
  if name in _UNIT_VECTOR_INPUTS and size >= 2:
    values /= np.linalg.norm(values)
  return ', '.join(repr(round(float(v), 3)) for v in values)


def _sort_inputs(category: str, inputs: dict[str, str]) -> None:
  """Sorts the channels of the input groups in `_SORTED_INPUTS` in place."""
  for group in _SORTED_INPUTS.get(category, []):
    names = [n for n in group if n in inputs]
    channels = zip(*([float(v) for v in inputs[n].split(',')] for n in names))
    for n, values in zip(names, zip(*(sorted(c) for c in channels))):
      inputs[n] = ', '.join(repr(round(v, 3)) for v in values)


def _orthonormalize_frame(category: str, inputs: dict[str, str]) -> None:
  """Makes the `normalmap` tangent frame inputs orthonormal."""
  if category != 'normalmap':
    return
  n, t = (
      np.array([float(v) for v in inputs[name].split(',')], dtype=np.float64)
      for name in ('normal', 'tangent')
  )
  n /= np.linalg.norm(n)
  t = t - np.dot(t, n) * n
  t /= np.linalg.norm(t)
  # Formats the frame with the precision of GenOsl (see `_generate_osl`).
  for name, v in (('normal', n), ('tangent', t), ('bitangent', np.cross(n, t))):
    inputs[name] = ', '.join(repr(round(float(x), 6)) for x in v)


def _make_samples(nd: mx.NodeDef) -> list[dict[str, str]]:
  """Returns the input samples of `nd`, starting with its default values."""
  rng = random.Random(zlib.crc32(nd.getName().encode()))
  category = nd.getNodeString()
  inputs = nd.getActiveInputs()
  defaults = {
      inp.getName(): (
          inp.getValueString()
          if not inp.hasDefaultGeomPropString() and inp.hasValueString()
          else _sample_input(rng, category, inp)
      )
      for inp in inputs
  }
  samples = [defaults]
  for _ in range(_NUM_RANDOM_SAMPLES):
    sample = {
        inp.getName(): _sample_input(rng, category, inp) for inp in inputs
    }
    _sort_inputs(category, sample)
    samples.append(sample)
  for sample in samples:
    _orthonormalize_frame(category, sample)
  return samples


def _generate_osl(
    stdlib: mx.Document,
    root: str,
    nd: mx.NodeDef,
    output: mx.Output,
    inputs: dict[str, str],
) -> str:
  """Generates an OSL shader evaluating `output` of `nd` for `inputs`."""
  doc = mx.createDocument()
  doc.importLibrary(stdlib)
  node = doc.addNodeInstance(doc.getNodeDef(nd.getName()), 'n')
  for inp in nd.getActiveInputs():
    name, typ = inp.getName(), inp.getType()
    if inp.hasDefaultGeomPropString():
      # GenOsl ignores the values of inputs with a default geometric property
      # (e.g. texture coordinates), so they are connected to constants.
      constant = doc.addNode('constant', f'c_{name}', typ)
      constant.addInput('value', typ).setValueString(inputs[name])
      node.addInput(name, typ).setConnectedNode(constant)
    else:
      node.addInput(name, typ).setValueString(inputs[name])
  out = doc.addOutput('o', output.getType())
  out.setConnectedNode(node)
  if len(nd.getActiveOutputs()) > 1:
    out.setOutputString(output.getName())

  gen = mx_gen_osl.OslShaderGenerator.create()
  ctx = mx_gen_shader.GenContext(gen)
  ctx.registerSourceCodeSearchPath(mx.FilePath(root))
  # Bakes the input values into the shader, formatted with 6 decimals, which
  # represents the sampled values exactly.
  ctx.getOptions().shaderInterfaceType = (
      mx_gen_shader.ShaderInterfaceType.SHADER_INTERFACE_REDUCED
  )
  src = gen.generate('mxref', out, ctx).getSourceCode(mx_gen_shader.PIXEL_STAGE)

  if output.getType() == 'boolean':
    stmt = 'printf("MXREF %d\\n", o);'
  else:
    comps = _OUTPUT_COMPONENTS[output.getType()]
    fmt = ' '.join(['%.9g'] * len(comps))
    stmt = f'printf("MXREF {fmt}\\n", {", ".join(comps)});'
  end = src.rindex('}')
  return f'{src[:end]}    {stmt}\n{src[end:]}'


def _run_osl(osl_bin: str, include_dir: str, src: str) -> list[float]:
  """Compiles and evaluates an OSL shader, returning its printed output."""
  with tempfile.TemporaryDirectory() as workdir:
    osl_path = os.path.join(workdir, 'mxref.osl')
    oso_path = os.path.join(workdir, 'mxref.oso')
    with open(osl_path, 'w', encoding='utf-8') as f:
      f.write(src)
    subprocess.run(
        [
            os.path.join(osl_bin, 'oslc'),
            '-q',
            f'-I{include_dir}',
            '-o',
            oso_path,
            osl_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = subprocess.run(
        [
            os.path.join(osl_bin, 'testshade'),
            '-t',
            '1',
            '-g',
            '1',
            '1',
            oso_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
  line = next(
      line_str
      for line_str in res.stdout.splitlines()
      if line_str.startswith('MXREF ')
  )
  return [float(v) for v in line.split()[1:]]


def _osl_version(osl_bin: str) -> str:
  res = subprocess.run(
      [os.path.join(osl_bin, 'oslc')],
      check=False,
      capture_output=True,
      text=True,
  )
  return (res.stdout + res.stderr).splitlines()[0].split()[-1]


def _write_json(
    path: str, header: dict[str, str], cases: list[dict[str, Any]]
) -> None:
  """Writes the reference data with one sample per line, to ease reviews."""
  with open(path, 'w', encoding='utf-8') as f:
    f.write('{\n')
    for key, value in header.items():
      f.write(f'  {json.dumps(key)}: {json.dumps(value)},\n')
    f.write('  "nodedefs": [\n')
    for i, case in enumerate(cases):
      f.write(f'    {{"nodedef": {json.dumps(case["nodedef"])}, "samples": [\n')
      f.write(',\n'.join('      ' + json.dumps(s) for s in case['samples']))
      f.write('\n    ]}' + (',' if i + 1 < len(cases) else '') + '\n')
    f.write('  ]\n}\n')


def main() -> None:
  """Generates OSL reference output JSON for MaterialX stdlib nodes."""
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument(
      '--osl_bin',
      required=True,
      help='Directory containing oslc and testshade.',
  )
  parser.add_argument('--output', required=True, help='Output JSON file.')
  parser.add_argument(
      '--filter',
      default='',
      help='Only process node definitions matching the regex.',
  )
  parser.add_argument(
      '--materialx_root',
      default=None,
      help='Optional MaterialX root directory containing libraries/.',
  )
  parser.add_argument(
      '--jobs',
      type=int,
      default=os.cpu_count(),
      help='Number of OSL shaders to compile and evaluate in parallel.',
  )
  args = parser.parse_args()

  root = args.materialx_root or os.path.dirname(mx.__file__)
  stdlib = mx.createDocument()
  mx.loadLibraries(['libraries'], mx.FileSearchPath(root), stdlib)
  include_dir = os.path.join(root, 'libraries', 'stdlib', 'genosl', 'include')

  # (nodedef, sample index, output, OSL source) of each OSL evaluation.
  jobs = []
  samples = {}
  for nd in sorted(stdlib.getNodeDefs(), key=lambda nd: nd.getName()):
    if nd.getName() not in _TARGETED_NODEDEFS or not re.search(
        args.filter, nd.getName()
    ):
      continue
    samples[nd.getName()] = _make_samples(nd)
    for output in nd.getActiveOutputs():
      for i, inputs in enumerate(samples[nd.getName()]):
        src = _generate_osl(stdlib, root, nd, output, inputs)
        jobs.append((nd.getName(), i, output.getName(), src))

  with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
    results = list(
        pool.map(lambda job: _run_osl(args.osl_bin, include_dir, job[3]), jobs)
    )

  expected = collections.defaultdict(dict)
  for (nodedef, i, output, _), values in zip(jobs, results):
    if not np.all(np.isfinite(values)):
      raise ValueError(f'{nodedef}.{output}[{i}] produced non-finite {values}')
    expected[(nodedef, output)][i] = [float(str(np.float32(v))) for v in values]

  cases = []
  for nd_name in sorted(samples):
    outputs = sorted(o for (n, o) in expected if n == nd_name)
    cases.append({
        'nodedef': nd_name,
        'samples': [
            {
                'inputs': inputs,
                'expected': {o: expected[(nd_name, o)][i] for o in outputs},
            }
            for i, inputs in enumerate(samples[nd_name])
        ],
    })

  header = {
      'materialx_version': mx.getVersionString(),
      'osl_version': _osl_version(args.osl_bin),
      'generator': 'generate_osl_reference.py',
  }
  _write_json(args.output, header, cases)
  print(f'Wrote {len(cases)} node definitions to {args.output}')


if __name__ == '__main__':
  main()
