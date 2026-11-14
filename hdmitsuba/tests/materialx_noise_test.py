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

"""Tests for MaterialX 2D/3D procedural noise nodes, primvars, and coordinate spaces."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import UsdGeom
from hdmitsuba.tests import materialx_test_helpers as mtlx_helpers
from hdmitsuba.tests import test_helpers
from usd_mitsuba import translator as usd_mitsuba


@pytest.fixture(autouse=True)
def _set_mitsuba_variant() -> None:
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def test_osl_noise_nodes() -> None:
  """Compares MaterialX 2D/3D procedural noise nodes against OSL reference values."""
  mtlx_helpers.check_osl_reference_nodes(include_noise=True)


def test_procedural_pattern_emission_graph() -> None:
  """Tests 3D procedural noise patterns on a UV-less mesh via emission."""
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  UsdGeom.PrimvarsAPI(mesh).RemovePrimvar('st')

  pos = mtlx_helpers.define_shader(stage, 'Position', 'ND_position_vector3')
  fractal = mtlx_helpers.define_shader(
      stage,
      'Fractal',
      'ND_fractal3d_color3',
      position=pos,
      amplitude=Gf.Vec3f(0.5, 0.5, 0.5),
  )
  noise_f = mtlx_helpers.define_shader(
      stage, 'NoiseWeight', 'ND_noise3d_float', position=pos, amplitude=0.4, pivot=1.0
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission=noise_f,
      emission_color=('ND_add_color3', {'in1': fractal, 'in2': (0.1, 0.7, 0.2)}),
  )

  engine = mtlx_helpers.create_engine(stage)
  image, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, engine=engine, spp=4, subdivision_level=0
  )
  image = image[..., :3]
  center = mtlx_helpers.render_center(image)
  assert center[1] > center[0] * 1.3
  # Disabling the scalar noise modulation changes the emitted radiance.
  noise_f.GetInput('amplitude').Set(0.0)
  unmodulated = np.array(engine.render()['color'])[..., :3]
  assert np.mean(np.abs(unmodulated - image)) > 1e-3
  # Disabling the fractal noise changes the emission color.
  fractal.GetInput('amplitude').Set((0.0, 0.0, 0.0))
  no_fractal = np.array(engine.render()['color'])[..., :3]
  assert (
      np.mean(
          np.abs(
              mtlx_helpers.render_center(no_fractal, half_size=4)
              - mtlx_helpers.render_center(unmodulated, half_size=4)
          )
      )
      > 0.01
  )


def test_primvars_and_3d_noise_emission() -> None:
  """Tests custom vertex primvars and `displayColor` combined with 3D noise via emission."""
  stage, mesh = mtlx_helpers.create_base_scene(
      with_lights=False, resolution=(16, 16), spp=4
  )
  num_verts = len(mesh.GetPointsAttr().Get())
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      'tint', Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
  ).Set([(0.1, 0.6, 0.2)] * num_verts)
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
      'displayColor', Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
  ).Set([(0.05, 0.1, 0.05)] * num_verts)

  mtlx_helpers.bind_mtlx_shader(
      stage,
      mesh,
      'ND_surface_unlit',
      emission_color=(
          'ND_add_color3',
          {
              'in1': (
                  'ND_noise3d_color3',
                  {'pivot': 0.4, 'amplitude': Gf.Vec3f(0.4, 0.4, 0.4)},
              ),
              'in2': (
                  'ND_add_color3',
                  {
                      'in1': (
                          'ND_geompropvalue_color3',
                          {'geomprop': 'tint', 'default': (0.9, 0.0, 0.0)},
                      ),
                      'in2': ('ND_geomcolor_color3', {}),
                  },
              ),
          },
      ),
  )

  image_hd, _ = test_helpers.assert_hydra_equal_to_offline(
      stage, spp=4, subdivision_level=0
  )
  center = mtlx_helpers.render_center(image_hd, half_size=4)
  assert center[1] > center[0] * 1.4
  assert center[1] > center[2] * 1.4


def test_object_vs_world_space_position_and_normal() -> None:
  """Verifies object vs world position and normal on a transformed emissive mesh."""
  stage, quad = mtlx_helpers.create_base_scene(
      with_lights=False,
      quad_half_size=0.8,
      resolution=(16, 16),
      spp=4,
  )
  # Transform the quad: translate by (0.4, 0.0, 0.0), rotate 180 deg around Z,
  # and scale by 0.5.
  quad.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.0))
  quad.AddRotateZOp().Set(180.0)
  quad.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))

  pos_node = mtlx_helpers.define_shader(
      stage, 'Pos', 'ND_position_vector3', space='object'
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      quad,
      'ND_surface_unlit',
      emission_color=(
          'ND_convert_vector3_color3',
          {'in_': ('ND_add_vector3', {'in1': pos_node, 'in2': (0.5, 0.5, 0.5)})},
      ),
  )

  # Hit at world position (0.5, 0.1, 0.0) corresponds to object-space position
  # (-0.2, -0.2, 0.0), so `p_obj + 0.5 = (0.3, 0.3, 0.5)`.
  p_hit = (0.5, 0.1, 0.0)
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage, p_target=p_hit, ray_origin=(0.5, 0.1, 2.0)),
      (0.3, 0.3, 0.5),
      atol=1e-4,
  )
  test_helpers.assert_hydra_equal_to_offline(stage, spp=4, subdivision_level=0)

  # Switching to world space yields `p_world + 0.5 = (1.0, 0.6, 0.5)`.
  pos_node.GetInput('space').Set('world')
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage, p_target=p_hit, ray_origin=(0.5, 0.1, 2.0)),
      (1.0, 0.6, 0.5),
      atol=1e-4,
  )
  test_helpers.assert_hydra_equal_to_offline(stage, spp=4, subdivision_level=0)

  # Author tilted object-space normals (0.6, 0.0, 0.8); under 180-deg Z
  # rotation, world-space normal is (-0.6, 0.0, 0.8).
  quad.GetNormalsAttr().Set([(0.6, 0.0, 0.8)] * 4)
  norm_node = mtlx_helpers.define_shader(
      stage, 'Norm', 'ND_normal_vector3', space='object'
  )
  mtlx_helpers.bind_mtlx_shader(
      stage,
      quad,
      'ND_surface_unlit',
      emission_color=(
          'ND_convert_vector3_color3',
          {'in_': ('ND_add_vector3', {'in1': norm_node, 'in2': (1.0, 1.0, 0.0)})},
      ),
  )
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage, p_target=p_hit, ray_origin=(0.5, 0.1, 2.0)),
      (1.6, 1.0, 0.8),
      atol=1e-4,
  )
  norm_node.GetInput('space').Set('world')
  np.testing.assert_allclose(
      mtlx_helpers.eval_unlit(stage, p_target=p_hit, ray_origin=(0.5, 0.1, 2.0)),
      (0.4, 1.0, 0.8),
      atol=1e-4,
  )
