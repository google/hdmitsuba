// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hdmitsuba/light.h"

#include <cmath>
#include <utility>

#include <drjit/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

std::pair<GfMatrix4d, GfVec3f> ComputeLightTransformAndEmission(
    const TfToken& type_id, const GfMatrix4d& transform,
    const GfVec3f& base_emission, float radius, float width, float height,
    bool normalize, bool treat_as_point, float shaping_cone_angle) {
  GfMatrix4d align_rotation;
  align_rotation.SetRotate(GfRotation(GfVec3d(1, 0, 0), 180.0));
  GfMatrix4d to_world = transform;
  GfVec3f emission = base_emission;
  if (type_id == HdPrimTypeTokens->rectLight) {
    GfMatrix4d scale;
    scale.SetScale(GfVec3d(0.5 * width, 0.5 * height, 1.0));
    to_world = scale * align_rotation * transform;
    if (normalize) {
      GfVec3d w_vec = transform.TransformDir(GfVec3d(width, 0, 0));
      GfVec3d h_vec = transform.TransformDir(GfVec3d(0, height, 0));
      double area = GfCross(w_vec, h_vec).GetLength();
      if (area > 0.0) emission /= area;
    }
  } else if (type_id == HdPrimTypeTokens->diskLight) {
    GfMatrix4d scale;
    scale.SetScale(GfVec3d(radius, radius, 1.0));
    to_world = scale * align_rotation * transform;
    if (normalize) {
      GfVec3d r_vec = transform.TransformDir(GfVec3d(radius, 0, 0));
      double area = drjit::Pi<double> * r_vec.GetLengthSq();
      if (area > 0.0) emission /= area;
    }
  } else if (type_id == HdPrimTypeTokens->sphereLight) {
    if (treat_as_point) {
      if (shaping_cone_angle != 0.0f) {
        to_world = align_rotation * transform;
      }
      if (normalize) {
        emission *= 0.25f;  // Match Mitsuba's point light normalization (1/4)
      }
    } else {
      GfMatrix4d scale;
      scale.SetScale(GfVec3d(radius, radius, radius));
      to_world = scale * transform;
      if (normalize) {
        GfVec3d r_vec = transform.TransformDir(GfVec3d(radius, 0, 0));
        double area = 4.0 * drjit::Pi<double> * r_vec.GetLengthSq();
        if (area > 0.0) emission /= area;
      }
    }
  } else if (type_id == HdPrimTypeTokens->distantLight) {
    to_world = align_rotation * transform;
  }

  return {to_world, emission};
}

}  // namespace

HdMitsubaLight::HdMitsubaLight(const SdfPath& id, const TfToken& typeId)
    : HdLight(id) {
  type_id_ = typeId;
}

void HdMitsubaLight::Sync(HdSceneDelegate* sceneDelegate,
                          HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
  static const HdDataSourceLocator transform_locator(
      HdXformSchema::GetSchemaToken(), HdXformSchemaTokens->matrix);
  static const HdDataSourceLocator visibility_locator(
      HdVisibilitySchema::GetSchemaToken(),
      HdVisibilitySchemaTokens->visibility);

  const SdfPath& id = GetId();
  SceneManager* scene =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  HdSceneIndexBaseRefPtr scene_index =
      sceneDelegate->GetRenderIndex().GetTerminalSceneIndex();

  if (!TF_VERIFY(scene_index)) {
    return;
  }
  HdContainerDataSourceHandle data_source = scene_index->GetPrim(id).dataSource;

  // 1. Visibility
  const bool visible = GetParam<bool>(data_source, visibility_locator, true);

  if (!visible) {
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 2. Transform
  const GfMatrix4d transform =
      GetParam<GfMatrix4d>(data_source, transform_locator, GfMatrix4d(1.0));

  // 3. Light Parameters
  HdLightSchema light_schema = HdLightSchema::GetFromParent(data_source);
  HdContainerDataSourceHandle light_container =
      light_schema.IsDefined() ? light_schema.GetContainer() : nullptr;
  if (!light_container) {
    TF_WARN("Light %s has no light parameters. Removing from scene.",
            id.GetText());
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  GfVec3f color =
      GetParam<GfVec3f>(light_container, HdLightTokens->color, GfVec3f(1.0f));
  float intensity =
      GetParam<float>(light_container, HdLightTokens->intensity, 1.0f);
  float exposure =
      GetParam<float>(light_container, HdLightTokens->exposure, 0.0f);
  float radius = GetParam<float>(light_container, HdLightTokens->radius, 1.0f);
  float width = GetParam<float>(light_container, HdLightTokens->width, 1.0f);
  float height = GetParam<float>(light_container, HdLightTokens->height, 1.0f);
  bool normalize =
      GetParam<bool>(light_container, HdLightTokens->normalize, false);

  static const TfToken treat_as_point_token("treatAsPoint");
  bool treat_as_point =
      GetParam<bool>(light_container, treat_as_point_token, radius == 0.0f);

  float shaping_cone_angle =
      GetParam<float>(light_container, HdLightTokens->shapingConeAngle, 0.0f);
  float shaping_cone_softness = GetParam<float>(
      light_container, HdLightTokens->shapingConeSoftness, 1.0f);

  std::string texture_file_path =
      GetParam<SdfAssetPath>(light_container, HdLightTokens->textureFile)
          .GetResolvedPath();

  float shaping_cone_beam_width =
      shaping_cone_angle * (1.0f - shaping_cone_softness);

  // 2. Calculate base emission (color * intensity * exp2(exposure))
  GfVec3f base_emission = color * intensity * std::exp2(exposure);

  // 3. Calculate final to_world transform and normalized emission
  auto [to_world, emission] = ComputeLightTransformAndEmission(
      type_id_, transform, base_emission, radius, width, height, normalize,
      treat_as_point, shaping_cone_angle);

  // 4. Populate LightSpec
  LightSpec spec;
  spec.id = id;
  spec.prim_type = type_id_;
  spec.transform = UsdToMitsubaTransform(to_world);
  spec.emission = emission;
  spec.shaping_cone_angle = shaping_cone_angle;
  spec.shaping_cone_beam_width = shaping_cone_beam_width;
  spec.treat_as_point = treat_as_point;
  spec.texture_file_path = texture_file_path;

  // 5. Determine if rebuild is needed using a clean declarative check
  bool needs_rebuild =
      !is_instantiated_ || (treat_as_point != treat_as_point_) ||
      ((shaping_cone_angle_ != 0.0f) != (shaping_cone_angle != 0.0f)) ||
      (texture_file_path != texture_file_path_);

  is_instantiated_ = true;
  treat_as_point_ = treat_as_point;
  shaping_cone_angle_ = shaping_cone_angle;
  texture_file_path_ = texture_file_path;

  spec.needs_rebuild = needs_rebuild;
  spec.dirty_bits = *dirtyBits;

  scene->SyncLight(std::move(spec));
  *dirtyBits = HdChangeTracker::Clean;
}

HdDirtyBits HdMitsubaLight::GetInitialDirtyBitsMask() const {
  return HdLight::AllDirty;
}

void HdMitsubaLight::Finalize(HdRenderParam* renderParam) {
  RemoveFromScene(static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene());
}

void HdMitsubaLight::RemoveFromScene(SceneManager* scene) {
  scene->RemoveLight(GetId());
  is_instantiated_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
