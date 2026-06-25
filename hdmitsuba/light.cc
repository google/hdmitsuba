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

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"
#include <drjit/math.h>

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
  const SdfPath& id = GetId();
  bool visible = sceneDelegate->GetVisible(id);
  SceneManager* scene =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  if (!visible) {
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 1. Extract raw USD parameters
  GfMatrix4d transform = sceneDelegate->GetTransform(id);
  GfVec3f color = sceneDelegate->GetLightParamValue(id, HdLightTokens->color)
                      .GetWithDefault<GfVec3f>(GfVec3f(1.0f, 1.0f, 1.0f));
  float intensity =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->intensity)
          .GetWithDefault<float>(1.0f);
  float exposure =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->exposure)
          .GetWithDefault<float>(0.0f);
  float radius = sceneDelegate->GetLightParamValue(id, HdLightTokens->radius)
                     .GetWithDefault<float>(1.0f);
  float width = sceneDelegate->GetLightParamValue(id, HdLightTokens->width)
                    .GetWithDefault<float>(1.0f);
  float height = sceneDelegate->GetLightParamValue(id, HdLightTokens->height)
                     .GetWithDefault<float>(1.0f);
  bool normalize =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->normalize)
          .GetWithDefault<bool>(false);
  bool treat_as_point =
      sceneDelegate->GetLightParamValue(id, TfToken("treatAsPoint"))
          .GetWithDefault<bool>(radius == 0.0f);
  float shaping_cone_angle =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->shapingConeAngle)
          .GetWithDefault<float>(0.0f);
  float shaping_cone_softness =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->shapingConeSoftness)
          .GetWithDefault<float>(1.0f);
  float shaping_cone_beam_width =
      shaping_cone_angle * (1.0f - shaping_cone_softness);

  std::string texture_file_path;
  VtValue texture_file =
      sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);
  if (texture_file.IsHolding<SdfAssetPath>()) {
    texture_file_path = texture_file.Get<SdfAssetPath>().GetResolvedPath();
  }

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
