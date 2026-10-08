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
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include "hdmitsuba/motion.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

struct LightParams {
  GfVec3f color{1.0f};
  float intensity = 1.0f;
  float exposure = 0.0f;
  float radius = 1.0f;
  float width = 1.0f;
  float height = 1.0f;
  bool normalize = false;
  bool treat_as_point = false;
  float angle = 0.0f;
  float shaping_cone_angle = 0.0f;
  float shaping_cone_softness = 1.0f;
  std::string texture_file_path;

  GfVec3f BaseEmission() const {
    return color * intensity * std::exp2(exposure);
  }
  float ShapingConeBeamWidth() const {
    return shaping_cone_angle * (1.0f - shaping_cone_softness);
  }
};

LightParams ReadLightParams(const HdContainerDataSourceHandle& container) {
  static const TfToken treat_as_point_token("treatAsPoint");
  LightParams p;
  p.color = GetParam<GfVec3f>(container, HdLightTokens->color, GfVec3f(1.0f));
  p.intensity = GetParam<float>(container, HdLightTokens->intensity, 1.0f);
  p.exposure = GetParam<float>(container, HdLightTokens->exposure, 0.0f);
  p.radius = GetParam<float>(container, HdLightTokens->radius, 1.0f);
  p.width = GetParam<float>(container, HdLightTokens->width, 1.0f);
  p.height = GetParam<float>(container, HdLightTokens->height, 1.0f);
  p.normalize = GetParam<bool>(container, HdLightTokens->normalize, false);
  p.treat_as_point =
      GetParam<bool>(container, treat_as_point_token, p.radius == 0.0f);
  p.angle = GetParam<float>(container, HdLightTokens->angle, 0.0f);
  p.shaping_cone_angle =
      GetParam<float>(container, HdLightTokens->shapingConeAngle, 0.0f);
  p.shaping_cone_softness =
      GetParam<float>(container, HdLightTokens->shapingConeSoftness, 1.0f);
  p.texture_file_path =
      GetParam<SdfAssetPath>(container, HdLightTokens->textureFile)
          .GetResolvedPath();
  return p;
}

ScalarAffineTransform4f ComputeLightTransform(const TfToken& type_id,
                                              const GfMatrix4d& transform,
                                              const LightParams& params) {
  GfMatrix4d align_rotation;
  align_rotation.SetRotate(GfRotation(GfVec3d(1, 0, 0), 180.0));
  if (type_id == HdPrimTypeTokens->rectLight) {
    GfMatrix4d scale;
    scale.SetScale(GfVec3d(0.5 * params.width, 0.5 * params.height, 1.0));
    return UsdToMitsubaTransform(scale * align_rotation * transform);
  }
  if (type_id == HdPrimTypeTokens->diskLight) {
    GfMatrix4d scale;
    scale.SetScale(GfVec3d(params.radius, params.radius, 1.0));
    return UsdToMitsubaTransform(scale * align_rotation * transform);
  }
  if (type_id == HdPrimTypeTokens->sphereLight && !params.treat_as_point) {
    GfMatrix4d scale;
    scale.SetScale(GfVec3d(params.radius, params.radius, params.radius));
    return UsdToMitsubaTransform(scale * transform);
  }
  // Non-area emitters (point, spot, directional, envmap) do not support scale
  // or shear, and AnimatedTransform rejects keyframe matrices with shear.
  ScalarAffineTransform4f to_world =
      RemoveScaleFromTransform(UsdToMitsubaTransform(transform));
  if ((type_id == HdPrimTypeTokens->sphereLight &&
       params.shaping_cone_angle != 0.0f) ||
      type_id == HdPrimTypeTokens->distantLight) {
    using ScalarVector3f = mitsuba::Vector<float, 3>;
    to_world = to_world *
               ScalarAffineTransform4f::rotate(ScalarVector3f(1, 0, 0), 180);
  }
  return to_world;
}

GfVec3f ComputeLightEmission(const TfToken& type_id,
                             const GfMatrix4d& transform,
                             const LightParams& params) {
  GfVec3f emission = params.BaseEmission();
  if (!params.normalize) {
    return emission;
  }
  if (type_id == HdPrimTypeTokens->rectLight) {
    GfVec3d w_vec = transform.TransformDir(GfVec3d(params.width, 0, 0));
    GfVec3d h_vec = transform.TransformDir(GfVec3d(0, params.height, 0));
    double area = GfCross(w_vec, h_vec).GetLength();
    if (area > 0.0) emission /= area;
  } else if (type_id == HdPrimTypeTokens->diskLight) {
    GfVec3d r_vec = transform.TransformDir(GfVec3d(params.radius, 0, 0));
    double area = drjit::Pi<double> * r_vec.GetLengthSq();
    if (area > 0.0) emission /= area;
  } else if (type_id == HdPrimTypeTokens->sphereLight) {
    if (params.treat_as_point) {
      emission *= 0.25f;  // Match Mitsuba's point light normalization (1/4)
    } else {
      GfVec3d r_vec = transform.TransformDir(GfVec3d(params.radius, 0, 0));
      double area = 4.0 * drjit::Pi<double> * r_vec.GetLengthSq();
      if (area > 0.0) emission /= area;
    }
  }
  return emission;
}

}  // namespace

HdMitsubaLight::HdMitsubaLight(const SdfPath& id, const TfToken& typeId)
    : HdLight(id) {
  type_id_ = typeId;
}

void HdMitsubaLight::Sync(HdSceneDelegate* sceneDelegate,
                          HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
  const SdfPath& id = GetId();
  SceneManager* scene =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  HdContainerDataSourceHandle data_source =
      GetPrimDataSource(sceneDelegate, id);

  if (!GetPrimVisible(data_source)) {
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

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

  const LightParams params = ReadLightParams(light_container);

  LightSpec spec;
  spec.id = id;
  spec.prim_type = type_id_;
  spec.angle = params.angle;
  spec.shaping_cone_angle = params.shaping_cone_angle;
  spec.shaping_cone_beam_width = params.ShapingConeBeamWidth();
  spec.treat_as_point = params.treat_as_point;
  spec.texture_file_path = params.texture_file_path;
  spec.dirty_bits = *dirtyBits;

  MotionSamples<GfMatrix4d> world_transform = SampleTransform(
      data_source, spec.IsAreaLight() && warned_unsupported_motion_
                       ? GfVec2f(0.0f)
                       : GetMotionInterval(sceneDelegate));
  if (spec.IsAreaLight() && world_transform.IsAnimated()) {
    TF_WARN("Motion blur is not supported for light %s of type %s.",
            id.GetText(), type_id_.GetText());
    warned_unsupported_motion_ = true;
    world_transform =
        MotionSamples<GfMatrix4d>::Static(GetPrimTransform(data_source));
  }
  spec.transform = world_transform.Map([&](const GfMatrix4d& m) {
    return ComputeLightTransform(type_id_, m, params);
  });
  // Area lights cannot be animated, so using First() is exact.
  spec.emission =
      ComputeLightEmission(type_id_, world_transform.First(), params);

  const RebuildKey rebuild_key{
      params.treat_as_point, params.angle > 0.0f,
      params.shaping_cone_angle != 0.0f, params.texture_file_path,
      spec.transform.IsAnimated()};
  spec.needs_rebuild = rebuild_key_ != rebuild_key;
  rebuild_key_ = rebuild_key;

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
  rebuild_key_.reset();
}

PXR_NAMESPACE_CLOSE_SCOPE
