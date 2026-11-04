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

"""Tests for the hdMitsuba render delegate shapes/meshes."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
import pytest

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdLux
from pxr import UsdShade
from pxr import UsdVol
from pxr import Vt
import usd_render
from hdmitsuba.tests import test_helpers
from usd_mitsuba import mesh as usd_mesh
from usd_mitsuba import translator as usd_mitsuba


@pytest.fixture(autouse=True)
def _set_mitsuba_variant():
  mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


@pytest.mark.parametrize(
    'scene_name, time, expect_non_black',
    [
        ('subdiv_cube', Usd.TimeCode.Default(), True),
        ('subdiv_plane', Usd.TimeCode.Default(), True),
        ('subdiv_two_quads', Usd.TimeCode.Default(), True),
        ('subdiv_indexed_uvs', Usd.TimeCode.Default(), True),
        ('displacement_preview_surface', Usd.TimeCode.Default(), True),
        ('curve', Usd.TimeCode.Default(), True),
        ('curve_catmull_rom', Usd.TimeCode.Default(), True),
        ('curve_catmull_rom_pinned', Usd.TimeCode.Default(), True),
        ('geom_subsets', Usd.TimeCode.Default(), True),
        ('empty_mesh', Usd.TimeCode.Default(), False),
        ('skinned_mesh', Usd.TimeCode(20), True),
        ('gaussian_splats', Usd.TimeCode.Default(), True),
        ('implicit_surfaces', Usd.TimeCode.Default(), True),
    ],
)
def test_render(scene_name: str, time: Usd.TimeCode, expect_non_black: bool):
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/{scene_name}.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(512, 512))
  image_hd, image_usd = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix=f'test_render_{scene_name}',
      atol=0.05,
      subdivision_level=4,
      refine_level_fallback=4,
      time=time,
  )

  # Comparing the two paths against each other passes trivially when both go
  # black, which is exactly what happens if a scene silently stops rendering
  # (e.g. splats losing their radiance-field integrator). Assert the images
  # carry signal for every scene that is supposed to produce some.
  if expect_non_black:
    assert np.max(image_hd[..., :3]) > 0.05
    assert np.max(image_usd) > 0.05


def test_modify_shape_transform():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image = engine.render()['color']
  test_helpers.write_image(image, 'test_modify_shape_transform_original.png')

  cube_xform = UsdGeom.Xformable.Get(stage, '/root/Cube')
  cube_xform.AddTranslateOp(opSuffix='new_translation').Set((0.9, 0.1, -0.1))
  cube_xform.AddScaleOp(opSuffix='new_scale').Set((1.0, 1.0, 0.5))
  cube_xform.AddRotateZOp(opSuffix='new_rotation').Set(-40)
  image_modified = engine.render()['color']
  test_helpers.write_image(
      image_modified, 'test_modify_shape_transform_modified.png'
  )
  assert np.mean(np.abs(image_modified[..., :3] - image[..., :3])) > 0.02

  # Re-render with a new render engine and check that the results match.
  engine_new = usd_render.RenderEngine(stage)
  engine_new.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_modified_2 = engine_new.render()['color']
  test_helpers.write_image(
      image_modified_2, 'test_modify_shape_transform_new.png'
  )
  np.testing.assert_allclose(image_modified, image_modified_2, atol=0.05)


def test_curve_transform():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine_orig = usd_render.RenderEngine(stage)
  engine_orig.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine_orig.render()['color']

  curve_xform = UsdGeom.Xformable.Get(stage, '/root/mycurve')
  curve_xform.AddTranslateOp(opSuffix='new_translation').Set((0.1, -0.1, 0.0))
  curve_xform.AddRotateXYZOp(opSuffix='new_rotation').Set((0.0, 0.0, -15.0))
  curve_xform.AddScaleOp(opSuffix='new_scale').Set((1.1, 0.9, 1.0))
  image_hd_modified, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_curve_transform',
      atol=0.05,
  )
  assert (
      np.max(np.abs(image_hd_modified[..., :3] - image_original[..., :3]))
      > 0.2
  )


def test_modify_points():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image = engine.render()['color']
  test_helpers.write_image(image, 'test_modify_points_original.png')

  mesh = UsdGeom.Mesh.Get(stage, '/root/Cube/Cube')
  points = np.array(mesh.GetPointsAttr().Get())
  points[0] = np.array([3.0, 1.0, 1.0])
  points[7] = np.array([-2.0, -2.0, -3.0])
  points *= 1.3
  mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

  image_modified = engine.render()['color']
  test_helpers.write_image(image_modified, 'test_modify_points_modified.png')
  assert np.mean(np.abs(image_modified[..., :3] - image[..., :3])) > 0.02

  engine_new = usd_render.RenderEngine(stage)
  engine_new.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_modified_2 = engine_new.render()['color']
  test_helpers.write_image(image_modified_2, 'test_modify_points_new.png')
  # np.testing.assert_allclose(image_modified, image_modified_2, atol=0.05)


def test_visibility():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Hide the cube
  cube_prim = stage.GetPrimAtPath('/root/Cube')
  imageable = UsdGeom.Imageable(cube_prim)
  imageable.MakeInvisible()

  image_invisible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_invisible',
      spp=8,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_invisible[..., :3]) < 0.01

  # Make it visible again
  imageable.MakeVisible()
  image_visible, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_visibility_visible',
      atol=0.05,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_visible, atol=0.05)


def test_active():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/cube.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']
  assert np.max(image_original[..., :3]) > 0.1

  # Inactivate the cube
  cube_prim = stage.GetPrimAtPath('/root/Cube')
  cube_prim.SetActive(False)

  image_inactive, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_inactive',
      spp=8,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_inactive[..., :3]) < 0.01

  # Reactivate it
  cube_prim.SetActive(True)
  image_active, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_active_active',
      atol=0.05,
      engine=engine,
  )
  np.testing.assert_allclose(image_original, image_active, atol=0.05)


def test_modify_curve_width():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(64, 64))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  _ = engine.render()['color']

  curve = UsdGeom.BasisCurves.Get(stage, '/root/mycurve/curve')
  widths = np.array(curve.GetWidthsAttr().Get()) * 2.5
  curve.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths))

  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_modify_curve_width',
      atol=0.1,
      engine=engine,
  )


def test_modify_curve_transform_only():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']

  curve_xform = UsdGeom.Xformable.Get(stage, '/root/mycurve')
  curve_xform.AddRotateZOp().Set(45.0)

  image_modified, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_modify_curve_transform_only',
      atol=0.05,
      engine=engine,
  )
  assert (
      np.max(np.abs(image_modified[..., :3] - image_original[..., :3])) > 0.2
  )


def test_curve_bspline_type():
  stage = Usd.Stage.Open(f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda')
  test_helpers.create_render_settings(stage, resolution=(128, 128))
  curve = UsdGeom.BasisCurves.Get(stage, '/root/mycurve/curve')
  curve.GetBasisAttr().Set(UsdGeom.Tokens.bspline)
  curve.GetTypeAttr().Set(UsdGeom.Tokens.cubic)
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_curve_bspline_type',
      atol=0.05,
  )


def test_curve_width_world_scale():
  stage_ref = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda'
  )
  test_helpers.create_render_settings(stage_ref, resolution=(128, 128))
  image_ref_hd, image_ref_usd = test_helpers.assert_hydra_equal_to_offline(
      stage_ref,
      output_prefix='test_curve_width_world_scale_ref',
      atol=0.05,
  )

  stage_scaled = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/curve.usda'
  )
  test_helpers.create_render_settings(stage_scaled, resolution=(128, 128))
  curve_xform = UsdGeom.Xformable.Get(stage_scaled, '/root/mycurve')
  curve_xform.AddScaleOp(opSuffix='cm_scale').Set((0.01, 0.01, 0.01))
  curve = UsdGeom.BasisCurves.Get(stage_scaled, '/root/mycurve/curve')
  curve.GetPointsAttr().Set(
      Vt.Vec3fArray.FromNumpy(np.array(curve.GetPointsAttr().Get()) * 100.0)
  )
  curve.GetWidthsAttr().Set(
      Vt.FloatArray.FromNumpy(np.array(curve.GetWidthsAttr().Get()) * 100.0)
  )

  image_scaled_hd, image_scaled_usd = (
      test_helpers.assert_hydra_equal_to_offline(
          stage_scaled,
          output_prefix='test_curve_width_world_scale_scaled',
          atol=0.05,
      )
  )
  np.testing.assert_allclose(
      image_scaled_hd[..., :3], image_ref_hd[..., :3], atol=0.05
  )
  np.testing.assert_allclose(image_scaled_usd, image_ref_usd, atol=0.05)


def test_subdivision_refinement():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/subdiv_cube.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(64, 64), spp=16)

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      refine_level_fallback=0,
  )
  img_coarse = engine.render()['color']

  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      refine_level_fallback=2,
  )
  img_refined = engine.render()['color']
  assert np.mean(np.abs(img_refined[..., :3] - img_coarse[..., :3])) > 0.005

  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin',
      refine_level_fallback=None,
  )
  np.testing.assert_allclose(engine.render()['color'], img_coarse, atol=1e-4)

  subdiv_primvar = UsdGeom.PrimvarsAPI(
      stage.GetPrimAtPath('/root/Cube/Cube')
  ).CreatePrimvar('mitsuba:subdivision_level', Sdf.ValueTypeNames.Int)
  subdiv_primvar.Set(2)
  np.testing.assert_allclose(engine.render()['color'], img_refined, atol=1e-4)

  subdiv_primvar.Set(0)
  np.testing.assert_allclose(engine.render()['color'], img_coarse, atol=1e-4)


def test_modify_implicit_surfaces():
  stage = Usd.Stage.Open(
      f'{test_helpers.TEST_ASSETS_PATH}/shapes/implicit_surfaces.usda'
  )
  test_helpers.create_render_settings(stage, resolution=(256, 256))
  engine = usd_render.RenderEngine(stage)
  engine.configure(hydra_delegate_id='HdMitsubaRendererPlugin')
  image_original = engine.render()['color']

  UsdGeom.Sphere.Get(stage, '/root/Sphere').GetRadiusAttr().Set(0.5)
  UsdGeom.Xformable.Get(stage, '/root/Sphere').AddScaleOp().Set((1.0, 0.5, 1.5))
  cylinder = UsdGeom.Cylinder.Get(stage, '/root/Cylinder')
  cylinder.GetHeightAttr().Set(0.8)
  cylinder.GetAxisAttr().Set(UsdGeom.Tokens.y)

  image_modified, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      output_prefix='test_modify_implicit_surfaces',
      atol=0.05,
      engine=engine,
  )
  assert (
      np.max(np.abs(image_modified[..., :3] - image_original[..., :3])) > 0.2
  )


def test_modify_gaussian_splat_positions():
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.Camera.Define(stage, '/camera').AddTranslateOp().Set((0, 0, 8))
  # Multiple fields, so that their in-place updates run on worker threads.
  splats = []
  for i in range(4):
    prim = stage.DefinePrim(f'/Splats_{i}')
    prim.GetReferences().AddReference(
        f'{test_helpers.TEST_ASSETS_PATH}/shapes/gaussian_splats.usda',
        '/World/gaussian_splats',
    )
    UsdGeom.Xformable(prim).AddTranslateOp().Set((1.5 * i - 2.25, 0, 0))
    splats.append(UsdVol.ParticleField3DGaussianSplat(prim))
  settings = test_helpers.create_render_settings(stage, resolution=(64, 64))
  settings.GetPrim().CreateAttribute(
      'mitsuba:integrator:type', Sdf.ValueTypeNames.String
  ).Set('volprim_rf_basic')
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/camera'
  )
  image = engine.render()['color']

  for field in splats:
    positions = field.GetPositionsAttr()
    positions.Set([p + Gf.Vec3f(0, 0, -3) for p in positions.Get()])
  assert not np.allclose(image, engine.render()['color'], atol=1e-2)


def test_mitsuba_shape_and_emitter_overrides():
  stage = Usd.Stage.CreateInMemory()
  camera = UsdGeom.Camera.Define(stage, '/Camera')
  camera.AddTranslateOp().Set((0.0, 0.0, 3.0))

  quad = UsdGeom.Mesh.Define(stage, '/Quad')
  quad.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  quad.GetPointsAttr().Set(
      [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]
  )
  quad.GetFaceVertexCountsAttr().Set([4])
  quad.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  quad_prim = quad.GetPrim()
  quad_prim.CreateAttribute(
      'mitsuba:emitter:type', Sdf.ValueTypeNames.String
  ).Set('area')
  quad_prim.CreateAttribute(
      'mitsuba:emitter:radiance:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  quad_prim.CreateAttribute(
      'mitsuba:emitter:radiance:value', Sdf.ValueTypeNames.Color3f
  ).Set((0.8, 0.4, 0.2))

  test_helpers.create_render_settings(stage, resolution=(64, 64), spp=16)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
  )

  image_front, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_emitter_front',
      spp=16,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_front[..., :3]) > 0.5

  # Flipping normals reverses the one-sided area emitter away from the camera.
  flip_attr = quad_prim.CreateAttribute(
      'mitsuba:shape:flip_normals', Sdf.ValueTypeNames.Bool
  )
  flip_attr.Set(True)
  image_flipped, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_emitter_flipped',
      spp=16,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_flipped[..., :3]) < 0.01

  # Updating the USD transform or mesh points while flip_normals is True must
  # keep normals flipped rather than reverting to unflipped normals in-place.
  translate_op = quad.AddTranslateOp()
  translate_op.Set((0.1, 0.0, 0.0))
  assert np.max(engine.render()['color'][..., :3]) < 0.01
  quad.GetPointsAttr().Set([
      (-0.9, -0.9, 0.0),
      (0.9, -0.9, 0.0),
      (0.9, 0.9, 0.0),
      (-0.9, 0.9, 0.0),
  ])
  assert np.max(engine.render()['color'][..., :3]) < 0.01
  translate_op.Set((0.0, 0.0, 0.0))
  quad.GetPointsAttr().Set([
      (-1.0, -1.0, 0.0),
      (1.0, -1.0, 0.0),
      (1.0, 1.0, 0.0),
      (-1.0, 1.0, 0.0),
  ])

  flip_attr.Set(False)
  np.testing.assert_allclose(
      engine.render()['color'][..., :3], image_front[..., :3], atol=0.05
  )

  # Authored mitsuba:shape:to_world shifts the mesh in both hdMitsuba and
  # usd_mitsuba, and remains active across subsequent USD transform updates.
  to_world_attr = quad_prim.CreateAttribute(
      'mitsuba:shape:to_world', Sdf.ValueTypeNames.Matrix4d
  )
  to_world_attr.Set(Gf.Matrix4d(1.0).SetTranslate((0.75, 0.0, 0.0)))
  image_shifted, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_to_world_shifted',
      spp=16,
      atol=0.05,
      engine=engine,
  )
  assert np.mean(np.abs(image_shifted[..., :3] - image_front[..., :3])) > 0.05

  # Updating emitter radiance leaf in-place updates emission without rebuilding.
  quad_prim.GetAttribute('mitsuba:emitter:radiance:value').Set((0.1, 0.2, 0.9))
  image_recolored, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_emitter_recolored',
      spp=16,
      atol=0.05,
      engine=engine,
  )
  assert (
      np.mean(image_recolored[..., 2]) > np.mean(image_recolored[..., 0]) * 4.0
  )


def test_mitsuba_shape_medium_in_place_updates():
  stage = Usd.Stage.CreateInMemory()
  camera = UsdGeom.Camera.Define(stage, '/Camera')
  camera.AddTranslateOp().Set((0.0, 0.0, 4.0))

  backdrop = UsdGeom.Mesh.Define(stage, '/Backdrop')
  backdrop.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  backdrop.GetPointsAttr().Set([
      (-2.0, -2.0, -2.0),
      (2.0, -2.0, -2.0),
      (2.0, 2.0, -2.0),
      (-2.0, 2.0, -2.0),
  ])
  backdrop.GetFaceVertexCountsAttr().Set([4])
  backdrop.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  backdrop_prim = backdrop.GetPrim()
  backdrop_prim.CreateAttribute(
      'mitsuba:emitter:type', Sdf.ValueTypeNames.String
  ).Set('area')
  backdrop_prim.CreateAttribute(
      'mitsuba:emitter:radiance:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  backdrop_prim.CreateAttribute(
      'mitsuba:emitter:radiance:value', Sdf.ValueTypeNames.Color3f
  ).Set((1.0, 1.0, 1.0))

  sphere = UsdGeom.Sphere.Define(stage, '/MediumSphere')
  sphere.GetRadiusAttr().Set(0.8)
  sphere_prim = sphere.GetPrim()
  sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:type', Sdf.ValueTypeNames.String
  ).Set('homogeneous')
  scale_attr = sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:scale', Sdf.ValueTypeNames.Float
  )
  scale_attr.Set(0.2)
  sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:sigma_t:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  sigma_t_attr = sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:sigma_t:value', Sdf.ValueTypeNames.Color3f
  )
  sigma_t_attr.Set((0.8, 0.2, 0.2))
  sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:albedo:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:albedo:value', Sdf.ValueTypeNames.Color3f
  ).Set((0.0, 0.0, 0.0))
  sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:phase:type', Sdf.ValueTypeNames.String
  ).Set('hg')
  g_attr = sphere_prim.CreateAttribute(
      'mitsuba:shape:interior:phase:g', Sdf.ValueTypeNames.Float
  )
  g_attr.Set(0.0)

  settings = test_helpers.create_render_settings(
      stage, resolution=(48, 48), spp=32
  )
  settings.GetPrim().CreateAttribute(
      'mitsuba:integrator:type', Sdf.ValueTypeNames.String
  ).Set('volpath')

  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
  )
  image_thin, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_medium_thin',
      spp=32,
      atol=0.05,
      engine=engine,
  )

  # Mutate medium density, extinction spectrum, and phase g in-place.
  scale_attr.Set(4.0)
  sigma_t_attr.Set((0.9, 0.9, 0.1))
  g_attr.Set(0.5)
  image_dense, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_medium_dense',
      spp=32,
      atol=0.05,
      engine=engine,
  )
  assert np.mean(image_dense[..., 0]) < np.mean(image_thin[..., 0]) * 0.5
  assert np.mean(image_dense[..., 2]) > np.mean(image_dense[..., 0]) * 2.0


def test_mitsuba_shape_to_world_non_commuting_composition():
  stage = Usd.Stage.CreateInMemory()
  camera = UsdGeom.Camera.Define(stage, '/Camera')
  camera.AddTranslateOp().Set((0.0, 0.0, 6.0))

  # Non-commuting object-space override (rotation + translation) combined with
  # non-identity USD xformOps across Mesh, Cube, and X-aligned Cylinder.
  extra_matrix = Gf.Matrix4d(1.0).SetRotate(
      Gf.Rotation(Gf.Vec3d(0, 0, 1), 35.0)
  ) * Gf.Matrix4d(1.0).SetTranslate((0.35, -0.2, 0.0))

  mesh = UsdGeom.Mesh.Define(stage, '/MeshQuad')
  mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  mesh.GetPointsAttr().Set([
      (-0.6, -0.3, 0.0),
      (0.6, -0.3, 0.0),
      (0.6, 0.3, 0.0),
      (-0.6, 0.3, 0.0),
  ])
  mesh.GetFaceVertexCountsAttr().Set([4])
  mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  mesh.AddTranslateOp().Set((-1.2, 0.6, 0.0))
  mesh.AddRotateYOp().Set(20.0)

  cube = UsdGeom.Cube.Define(stage, '/Cube')
  cube.GetSizeAttr().Set(0.8)
  cube.AddTranslateOp().Set((1.1, 0.6, 0.0))
  cube.AddRotateXOp().Set(25.0)

  cylinder = UsdGeom.Cylinder.Define(stage, '/Cylinder')
  cylinder.GetAxisAttr().Set(UsdGeom.Tokens.x)
  cylinder.GetRadiusAttr().Set(0.3)
  cylinder.GetHeightAttr().Set(1.1)
  cyl_translate = cylinder.AddTranslateOp()
  cyl_translate.Set((0.0, -0.9, 0.0))
  cylinder.AddRotateZOp().Set(-20.0)

  for idx, prim in enumerate(
      (mesh.GetPrim(), cube.GetPrim(), cylinder.GetPrim())
  ):
    prim.CreateAttribute(
        'mitsuba:emitter:type', Sdf.ValueTypeNames.String
    ).Set('area')
    prim.CreateAttribute(
        'mitsuba:emitter:radiance:type', Sdf.ValueTypeNames.String
    ).Set('rgb')
    color = [0.2, 0.2, 0.2]
    color[idx] = 0.9
    prim.CreateAttribute(
        'mitsuba:emitter:radiance:value', Sdf.ValueTypeNames.Color3f
    ).Set(tuple(color))
    prim.CreateAttribute(
        'mitsuba:shape:to_world', Sdf.ValueTypeNames.Matrix4d
    ).Set(extra_matrix)

  test_helpers.create_render_settings(stage, resolution=(64, 64), spp=16)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
  )
  image_initial, _ = test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_to_world_non_commuting',
      spp=16,
      atol=0.05,
      engine=engine,
  )
  assert np.max(image_initial[..., :3]) > 0.5

  # Live USD transform update on the X-aligned Cylinder preserves parity.
  cyl_translate.Set((0.4, -0.7, 0.0))
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_to_world_non_commuting_updated',
      spp=16,
      atol=0.05,
      engine=engine,
  )


def test_mitsuba_shape_custom_id_and_multi_object_properties():
  stage = Usd.Stage.CreateInMemory()
  camera = UsdGeom.Camera.Define(stage, '/Camera')
  camera.AddTranslateOp().Set((0.0, 0.0, 4.0))

  mesh = UsdGeom.Mesh.Define(stage, '/Quad')
  mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  mesh.GetPointsAttr().Set([
      (-0.6, -0.6, 0.0),
      (0.6, -0.6, 0.0),
      (0.6, 0.6, 0.0),
      (-0.6, 0.6, 0.0),
  ])
  mesh.GetFaceVertexCountsAttr().Set([4])
  mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  mesh.GetDisplayColorAttr().Set([(0.0, 0.0, 0.0)])
  mesh_translate = mesh.AddTranslateOp()
  mesh_translate.Set((-0.8, 0.0, 0.0))
  mesh_prim = mesh.GetPrim()
  mesh_prim.CreateAttribute('mitsuba:shape:id', Sdf.ValueTypeNames.String).Set(
      'hero_quad'
  )
  mesh_prim.CreateAttribute(
      'mitsuba:emitter:type', Sdf.ValueTypeNames.String
  ).Set('area')
  mesh_prim.CreateAttribute(
      'mitsuba:emitter:radiance:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  mesh_prim.CreateAttribute(
      'mitsuba:emitter:radiance:value', Sdf.ValueTypeNames.Color3f
  ).Set((0.9, 0.2, 0.2))

  sphere = UsdGeom.Sphere.Define(stage, '/Sphere')
  sphere.GetRadiusAttr().Set(0.5)
  sphere.GetDisplayColorAttr().Set([(0.0, 0.0, 0.0)])
  sphere_translate = sphere.AddTranslateOp()
  sphere_translate.Set((0.8, 0.0, 0.0))
  sphere_prim = sphere.GetPrim()
  sphere_prim.CreateAttribute(
      'mitsuba:shape:id', Sdf.ValueTypeNames.String
  ).Set('hero_sphere')
  sphere_prim.CreateAttribute(
      'mitsuba:emitter:type', Sdf.ValueTypeNames.String
  ).Set('area')
  sphere_prim.CreateAttribute(
      'mitsuba:emitter:radiance:type', Sdf.ValueTypeNames.String
  ).Set('rgb')
  sphere_prim.CreateAttribute(
      'mitsuba:emitter:radiance:value', Sdf.ValueTypeNames.Color3f
  ).Set((0.2, 0.8, 0.3))

  scene_dict = usd_mitsuba.convert_to_mitsuba(stage)
  assert 'hero_quad' in scene_dict
  assert scene_dict['hero_quad'].id() == 'hero_quad'
  offline_scene = mi.load_dict(scene_dict)
  assert {s.id() for s in offline_scene.shapes()} == {
      'hero_quad',
      'hero_sphere',
  }

  test_helpers.create_render_settings(stage, resolution=(32, 32), spp=64)
  engine = usd_render.RenderEngine(stage)
  engine.configure(
      hydra_delegate_id='HdMitsubaRendererPlugin', camera_path='/Camera'
  )
  image_initial = engine.render()['color']
  assert np.max(image_initial[..., :3]) > 0.5

  # Transform updates and deactivation on custom-ID shapes must succeed cleanly.
  mesh_translate.Set((-0.5, 0.2, 0.0))
  sphere_translate.Set((0.5, -0.2, 0.0))
  test_helpers.assert_hydra_equal_to_offline(
      stage,
      camera_path='/Camera',
      output_prefix='test_mitsuba_shape_custom_id_moved',
      spp=64,
      atol=0.05,
      engine=engine,
  )
  mesh_prim.SetActive(False)
  sphere_prim.SetActive(False)
  assert np.max(engine.render()['color'][..., :3]) < 1e-4

  # Verify `_set_plugin_properties` handles plugins whose `expand()` returns a
  # multi-element list (e.g. Mitsuba `merge` containing non-mesh shapes).
  props = mi.Properties('shapegroup')
  usd_mesh._set_plugin_properties(
      props,
      {
          'child': {
              'type': 'merge',
              's0': {'type': 'sphere'},
              's1': {'type': 'rectangle'},
          }
      },
  )
  assert 'child_0' in props and 'child_1' in props


def test_surface_sensor_on_plane_and_multi_material_mesh():
  # 1. Surface sensor bound via `mitsuba:sensor:shape` to `UsdGeom.Plane` and
  # `UsdGeom.Cube` under a constant `DomeLight` (zero Monte Carlo variance).
  plane_stage = Usd.Stage.CreateInMemory()
  dome = UsdLux.DomeLight.Define(plane_stage, '/Dome')
  dome.GetIntensityAttr().Set(0.5)

  plane = UsdGeom.Plane.Define(plane_stage, '/SensorPlane')
  plane.GetWidthAttr().Set(2.0)
  plane.GetLengthAttr().Set(2.0)

  cube = UsdGeom.Cube.Define(plane_stage, '/SensorCube')
  cube.GetSizeAttr().Set(1.0)
  cube.AddTranslateOp().Set((0.0, 0.0, -5.0))

  cam = UsdGeom.Camera.Define(plane_stage, '/PlaneMeter')
  cam_prim = cam.GetPrim()
  cam_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('irradiancemeter')
  shape_attr = cam_prim.CreateAttribute(
      'mitsuba:sensor:shape', Sdf.ValueTypeNames.String
  )
  shape_attr.Set('/SensorPlane')
  cam_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('box')

  test_helpers.create_render_settings(plane_stage, resolution=(4, 4), spp=16)
  img_plane, _ = test_helpers.assert_hydra_equal_to_offline(
      plane_stage,
      camera_path='/PlaneMeter',
      output_prefix='test_surface_sensor_plane',
      spp=16,
      atol=0.02,
  )
  np.testing.assert_allclose(img_plane[..., :3], 0.5 * np.pi, atol=0.02)

  shape_attr.Set('/SensorCube')
  cube_scene = mi.load_dict(usd_mitsuba.convert_to_mitsuba(plane_stage))
  assert len(cube_scene.sensors()) == 1
  assert float(np.mean(np.array(mi.render(cube_scene, spp=16)))) > 0.5

  # 2. Surface sensor bound to a multi-material `GeomSubset` Mesh attaches only
  # to the first subset (`+Z` face seeing `/Dome`, analytical `0.5 * pi`)
  # rather than duplicating onto `subset_1` (`-Z` face occluded by `/Blocker`).
  subset_stage = Usd.Stage.CreateInMemory()
  subset_dome = UsdLux.DomeLight.Define(subset_stage, '/Dome')
  subset_dome.GetIntensityAttr().Set(0.5)

  blocker = UsdGeom.Mesh.Define(subset_stage, '/Blocker')
  blocker.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  blocker.GetPointsAttr().Set([
      (-20.0, -20.0, -1.5),
      (20.0, -20.0, -1.5),
      (20.0, 20.0, -1.5),
      (-20.0, 20.0, -1.5),
  ])
  blocker.GetFaceVertexCountsAttr().Set([4])
  blocker.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
  blocker.GetDisplayColorAttr().Set([(0.0, 0.0, 0.0)])

  mesh = UsdGeom.Mesh.Define(subset_stage, '/TwoSubsetMesh')
  mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
  mesh.GetPointsAttr().Set([
      (-1.0, -1.0, 0.0),
      (1.0, -1.0, 0.0),
      (1.0, 1.0, 0.0),
      (-1.0, 1.0, 0.0),
      (-1.0, -1.0, -1.0),
      (-1.0, 1.0, -1.0),
      (1.0, 1.0, -1.0),
      (1.0, -1.0, -1.0),
  ])
  mesh.GetFaceVertexCountsAttr().Set([4, 4])
  mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6, 7])
  UsdGeom.Subset.SetFamilyType(
      mesh, UsdShade.Tokens.materialBind, UsdGeom.Tokens.nonOverlapping
  )

  for idx, color in enumerate(((0.8, 0.2, 0.2), (0.2, 0.2, 0.8))):
    mat = UsdShade.Material.Define(subset_stage, f'/Mat{idx}')
    shader = UsdShade.Shader.Define(subset_stage, f'/Mat{idx}/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), 'surface'
    )
    subset = UsdGeom.Subset.CreateGeomSubset(
        mesh,
        f'subset_{idx}',
        UsdGeom.Tokens.face,
        Vt.IntArray([idx]),
        UsdShade.Tokens.materialBind,
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)

  meter = UsdGeom.Camera.Define(subset_stage, '/SubsetMeter')
  meter_prim = meter.GetPrim()
  meter_prim.CreateAttribute(
      'mitsuba:sensor:type', Sdf.ValueTypeNames.String
  ).Set('irradiancemeter')
  meter_prim.CreateAttribute(
      'mitsuba:sensor:shape', Sdf.ValueTypeNames.String
  ).Set('/TwoSubsetMesh')
  meter_prim.CreateAttribute(
      'mitsuba:sensor:film:pixel_filter:type', Sdf.ValueTypeNames.String
  ).Set('box')

  test_helpers.create_render_settings(subset_stage, resolution=(4, 4), spp=16)
  offline_subset_scene = mi.load_dict(
      usd_mitsuba.convert_to_mitsuba(subset_stage)
  )
  assert len(offline_subset_scene.shapes()) == 3
  assert len(offline_subset_scene.sensors()) == 1

  img_subset, _ = test_helpers.assert_hydra_equal_to_offline(
      subset_stage,
      camera_path='/SubsetMeter',
      output_prefix='test_surface_sensor_multi_material_mesh',
      spp=16,
      atol=0.02,
  )
  np.testing.assert_allclose(img_subset[..., :3], 0.5 * np.pi, atol=0.02)
