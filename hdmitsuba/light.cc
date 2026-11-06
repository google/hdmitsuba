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

#include <algorithm>
#include <cmath>
#include <utility>

#include <drjit/math.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/collectionsSchema.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/pathExpression.h>
#include <pxr/usd/usd/collectionMembershipQuery.h>
#include <pxr/usd/usd/tokens.h>
#include <pxr/usd/usdLux/blackbody.h>

#include "hdmitsuba/motion.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

struct LightParams {
  GfVec3f base_emission{1.0f};
  float radius = 1.0f;
  float width = 1.0f;
  float height = 1.0f;
  float length = 1.0f;
  bool normalize = false;
  bool treat_as_point = false;
  float angle = 0.0f;
  float shaping_cone_angle = 0.0f;
  float shaping_cone_softness = 0.0f;
  std::string texture_file_path;
  std::string visibility;

  float ShapingConeBeamWidth() const {
    return shaping_cone_angle * (1.0f - shaping_cone_softness);
  }
};

LightParams ReadLightParams(const HdContainerDataSourceHandle& container) {
  static const TfToken treat_as_point_token("treatAsPoint");
  static const TfToken visibility_token("mitsuba:visibility");
  LightParams p;
  p.base_emission = GetLightBaseEmission(container);
  p.radius = GetParam<float>(container, HdLightTokens->radius, 1.0f);
  p.width = GetParam<float>(container, HdLightTokens->width, 1.0f);
  p.height = GetParam<float>(container, HdLightTokens->height, 1.0f);
  p.length = GetParam<float>(container, HdLightTokens->length, 1.0f);
  p.normalize = GetParam<bool>(container, HdLightTokens->normalize, false);
  // `treatAsPoint` always has a value (its schema fallback is false), so
  // zero-radius spheres must be checked explicitly.
  p.treat_as_point =
      GetParam<bool>(container, treat_as_point_token, false) ||
      p.radius == 0.0f;
  // Mitsuba's `directional` emitter requires angles below 180 deg.
  p.angle = std::clamp(GetParam<float>(container, HdLightTokens->angle, 0.0f),
                       0.0f, 179.99f);
  p.shaping_cone_angle =
      GetParam<float>(container, HdLightTokens->shapingConeAngle, 0.0f);
  p.shaping_cone_softness =
      GetParam<float>(container, HdLightTokens->shapingConeSoftness, 0.0f);
  p.texture_file_path =
      GetParam<SdfAssetPath>(container, HdLightTokens->textureFile)
          .GetResolvedPath();
  if (auto ds = HdSampledDataSource::Cast(container->Get(visibility_token))) {
    p.visibility =
        ds->GetValue(0.0f).Cast<std::string>().GetWithDefault<std::string>();
  }
  return p;
}

// World-space area of a light-space sphere of radius `radius`, an ellipsoid
// under non-uniform scale, approximated as in hdEmbree.
double SphereArea(const GfMatrix4d& transform, float radius) {
  const double a = transform.TransformDir(GfVec3d(radius, 0, 0)).GetLength();
  const double b = transform.TransformDir(GfVec3d(0, radius, 0)).GetLength();
  const double c = transform.TransformDir(GfVec3d(0, 0, radius)).GetLength();
  constexpr double kP = 1.6;
  return 4.0 * drjit::Pi<double> *
         std::pow((std::pow(a * b, kP) + std::pow(a * c, kP) +
                   std::pow(b * c, kP)) /
                      3.0,
                  1.0 / kP);
}

// World-space perimeter of a light-space circle of radius `radius` in the YZ
// plane, an ellipse under non-uniform scale, approximated as in hdEmbree.
double CylinderPerimeter(const GfMatrix4d& transform, float radius) {
  const double a = transform.TransformDir(GfVec3d(0, radius, 0)).GetLength();
  const double b = transform.TransformDir(GfVec3d(0, 0, radius)).GetLength();
  return drjit::Pi<double> *
         (3.0 * (a + b) - std::sqrt((3.0 * a + b) * (a + 3.0 * b)));
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
    // Mitsuba's `sphere` does not support non-uniform scale, so use the radius
    // of the sphere with the same area.
    const float world_radius = static_cast<float>(std::sqrt(
        SphereArea(transform, params.radius) / (4.0 * drjit::Pi<double>)));
    // The rotation leaves the sphere unchanged but aligns its IES frame.
    return RemoveScaleFromTransform(
               UsdToMitsubaTransform(align_rotation * transform)) *
           ScalarAffineTransform4f::scale(
               mitsuba::Vector<float, 3>(world_radius));
  }
  if (type_id == HdPrimTypeTokens->cylinderLight) {
    // Mitsuba's `cylinder` does not support non-uniform scale of its
    // cross-section, so use the radius of the cylinder with the same area.
    const float length = static_cast<float>(
        transform.TransformDir(GfVec3d(params.length, 0, 0)).GetLength());
    const float radius = static_cast<float>(
        CylinderPerimeter(transform, params.radius) / drjit::TwoPi<double>);
    return RemoveScaleFromTransform(
               UsdToMitsubaTransform(align_rotation * transform)) *
           ScalarAffineTransform4f::scale(
               mitsuba::Vector<float, 3>(length, radius, radius));
  }
  // Non-area emitters (point, spot, ies, directional, envmap) do not support
  // scale or shear, and AnimatedTransform rejects keyframe matrices with shear.
  ScalarAffineTransform4f to_world =
      RemoveScaleFromTransform(UsdToMitsubaTransform(transform));
  if (type_id == HdPrimTypeTokens->sphereLight ||
      type_id == HdPrimTypeTokens->distantLight) {
    using ScalarVector3f = mitsuba::Vector<float, 3>;
    to_world = to_world *
               ScalarAffineTransform4f::rotate(ScalarVector3f(1, 0, 0), 180);
  }
  return to_world;
}

// World-space area of a light-space disk of radius `radius`, an ellipse under
// non-uniform scale.
double DiskArea(const GfMatrix4d& transform, float radius) {
  return drjit::Pi<double> *
         GfCross(transform.TransformDir(GfVec3d(radius, 0, 0)),
                 transform.TransformDir(GfVec3d(0, radius, 0)))
             .GetLength();
}

GfVec3f ComputeLightEmission(const TfToken& type_id,
                             const GfMatrix4d& transform,
                             const LightParams& params) {
  GfVec3f emission = params.base_emission;
  if (type_id == HdPrimTypeTokens->distantLight) {
    // USD lights give the radiance of the cone of half-angle theta, while the
    // `directional` emitter takes the irradiance it spreads over the cone,
    // i.e., the radiance times the cone's solid angle 2 pi (1 - cos(theta)).
    // `normalize` divides the radiance by pi sin^2(theta), the irradiance at
    // normal incidence.
    if (params.angle > 0.0f) {
      const double cos_theta = std::cos(drjit::deg_to_rad(0.5 * params.angle));
      emission *= static_cast<float>(
          params.normalize ? 2.0 / (1.0 + cos_theta)
                           : drjit::TwoPi<double> * (1.0 - cos_theta));
    }
    return emission;
  }
  if (!params.normalize) {
    // A point light stands in for its sphere. A sphere of radiance L has
    // radiant intensity L times its projected area, which on average is a
    // quarter of its surface area (pi * r^2 for a sphere of radius r).
    // Zero-radius lights keep `intensity` as radiant intensity.
    if (type_id == HdPrimTypeTokens->sphereLight && params.treat_as_point &&
        params.radius > 0.0f) {
      emission *= 0.25 * SphereArea(transform, params.radius);
    }
    return emission;
  }
  if (type_id == HdPrimTypeTokens->rectLight) {
    GfVec3d w_vec = transform.TransformDir(GfVec3d(params.width, 0, 0));
    GfVec3d h_vec = transform.TransformDir(GfVec3d(0, params.height, 0));
    double area = GfCross(w_vec, h_vec).GetLength();
    if (area > 0.0) emission /= area;
  } else if (type_id == HdPrimTypeTokens->diskLight) {
    double area = DiskArea(transform, params.radius);
    if (area > 0.0) emission /= area;
  } else if (type_id == HdPrimTypeTokens->cylinderLight) {
    double area =
        CylinderPerimeter(transform, params.radius) *
        transform.TransformDir(GfVec3d(params.length, 0, 0)).GetLength();
    if (area > 0.0) emission /= area;
  } else if (type_id == HdPrimTypeTokens->sphereLight) {
    if (params.treat_as_point) {
      emission *= 0.25f;  // Match Mitsuba's point light normalization (1/4)
    } else {
      double area = SphereArea(transform, params.radius);
      if (area > 0.0) emission /= area;
    }
  }
  return emission;
}

}  // namespace

GfVec3f GetLightBaseEmission(
    const HdContainerDataSourceHandle& light_container) {
  GfVec3f color =
      GetParam<GfVec3f>(light_container, HdLightTokens->color, GfVec3f(1.0f));
  if (GetParam<bool>(light_container, HdLightTokens->enableColorTemperature,
                     false)) {
    color = GfCompMult(color, UsdLuxBlackbodyTemperatureAsRgb(GetParam<float>(
                                  light_container,
                                  HdLightTokens->colorTemperature, 6500.0f)));
  }
  // Like hdEmbree, `diffuse` scales all emission: Mitsuba cannot restrict it
  // to diffuse reflection.
  return color * GetParam<float>(light_container, HdLightTokens->diffuse, 1.0f) *
         GetParam<float>(light_container, HdLightTokens->intensity, 1.0f) *
         std::exp2(
             GetParam<float>(light_container, HdLightTokens->exposure, 0.0f));
}

ShapingSpec GetShapingSpec(const HdContainerDataSourceHandle& light_container,
                           const SdfPath& id) {
  ShapingSpec shaping;
  shaping.cone_angle = GetParam<float>(
      light_container, HdLightTokens->shapingConeAngle, shaping.cone_angle);
  shaping.cone_softness = std::clamp(
      GetParam<float>(light_container, HdLightTokens->shapingConeSoftness,
                      shaping.cone_softness),
      0.0f, 1.0f);
  shaping.focus = GetParam<float>(light_container, HdLightTokens->shapingFocus,
                                  shaping.focus);
  shaping.focus_tint = GetParam<GfVec3f>(
      light_container, HdLightTokens->shapingFocusTint, shaping.focus_tint);
  const auto ies_file =
      GetParam<SdfAssetPath>(light_container, HdLightTokens->shapingIesFile);
  if (ies_file.GetAssetPath().empty()) {
    return shaping;
  }
  if (ies_file.GetResolvedPath().empty()) {
    TF_WARN("Could not resolve IES file '%s' of light %s. Ignoring it.",
            ies_file.GetAssetPath().c_str(), id.GetText());
    return shaping;
  }
  shaping.ies_file_path = ies_file.GetResolvedPath();
  shaping.ies_angle_scale = GetParam<float>(
      light_container, HdLightTokens->shapingIesAngleScale, 0.0f);
  shaping.ies_normalize = GetParam<bool>(
      light_container, HdLightTokens->shapingIesNormalize, false);
  return shaping;
}

HdMitsubaLight::HdMitsubaLight(const SdfPath& id, const TfToken& typeId)
    : HdLight(id) {
  type_id_ = typeId;
}

void HdMitsubaLight::Sync(HdSceneDelegate* sceneDelegate,
                          HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
  const SdfPath& id = GetId();
  auto* render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  SceneManager* scene = render_param->GetScene();

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

  ScopedResolverCache resolver_cache(render_param->ResolverCacheData());
  const LightParams params = ReadLightParams(light_container);

  LightSpec spec;
  spec.id = id;
  spec.prim_type = type_id_;
  spec.angle = params.angle;
  spec.shaping_cone_angle = params.shaping_cone_angle;
  spec.shaping_cone_beam_width = params.ShapingConeBeamWidth();
  spec.treat_as_point = params.treat_as_point;
  spec.texture_file_path = params.texture_file_path;
  spec.visibility = params.visibility.empty()
                        ? (spec.IsAreaLight() ? "secondary" : "all")
                        : params.visibility;
  if (auto expr = HdCollectionsSchema::GetFromParent(data_source)
                      .GetCollection(HdTokens->lightLink)
                      .GetMembershipExpression()) {
    // UsdImaging converts a default (includeRoot, no rules) collection to
    // this expression rather than `SdfPathExpression::Everything()`.
    static const SdfPathExpression default_light_link =
        UsdComputePathExpressionFromCollectionMembershipQueryRuleMap(
            {{SdfPath::AbsoluteRootPath(), UsdTokens->expandPrims}});
    spec.light_link_everything =
        expr->GetTypedValue(0.0f) == default_light_link;
  }

  // IES profiles only apply to positional lights, and are ignored otherwise.
  if (spec.IsAreaLight() || type_id_ == HdPrimTypeTokens->sphereLight) {
    spec.shaping = GetShapingSpec(light_container, id);
    // Point lights realize the cone as `spot` emitters.
    if (!spec.IsAreaLight()) {
      spec.shaping.cone_angle = 180.0f;
    }
  }
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
  // Point lights may be animated, so use the transform at frame time for their
  // radius.
  const GfMatrix4d emission_transform = world_transform.IsAnimated()
                                            ? GetPrimTransform(data_source)
                                            : world_transform.First();
  spec.emission = ComputeLightEmission(type_id_, emission_transform, params);

  const RebuildKey rebuild_key{params.treat_as_point,
                               params.angle > 0.0f,
                               params.shaping_cone_angle != 0.0f,
                               params.texture_file_path,
                               spec.shaping,
                               spec.transform.IsAnimated(),
                               spec.visibility,
                               spec.light_link_everything,
                               spec.TexturedDomeTint()};
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
