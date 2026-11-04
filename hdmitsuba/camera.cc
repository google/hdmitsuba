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

#include "hdmitsuba/camera.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include <drjit/math.h>
#include <drjit/sphere.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/plugin_builder.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

ScalarAffineTransform4f UsdToMitsubaSensorTransform(
    const GfMatrix4d& transform) {
  ScalarAffineTransform4f to_world =
      RemoveScaleFromTransform(UsdToMitsubaTransform(transform));
  using ScalarVector3f = mitsuba::Vector<float, 3>;
  return to_world *
         ScalarAffineTransform4f::rotate(ScalarVector3f(0, 1, 0), 180);
}

std::optional<SdfPath> ExtractTargetShapeId(PluginSpecNode* sensor_spec,
                                            const SdfPath& camera_id) {
  auto shape_it = sensor_spec->params.find("shape");
  if (shape_it == sensor_spec->params.end()) {
    return std::nullopt;
  }
  VtValue value = std::move(shape_it->second);
  sensor_spec->params.erase(shape_it);
  std::string path;
  if (value.IsHolding<std::string>()) {
    path = value.UncheckedGet<std::string>();
  } else if (value.IsHolding<TfToken>()) {
    path = value.UncheckedGet<TfToken>().GetString();
  } else if (value.IsHolding<SdfAssetPath>()) {
    path = ResolvePathFromValue(value);
  } else {
    return std::nullopt;
  }
  if (path.empty()) {
    return std::nullopt;
  }
  if (!SdfPath::IsValidPathString(path)) {
    TF_WARN("Camera %s: mitsuba:sensor:shape is not a valid path: %s",
            camera_id.GetText(), path.c_str());
    return std::nullopt;
  }
  return SdfPath(path);
}

}  // namespace

HdMitsubaCamera::HdMitsubaCamera(const SdfPath& id) : HdCamera(id) {}

void HdMitsubaCamera::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
  HdDirtyBits dirty_bits_copy = *dirtyBits;
  TF_DEBUG(HDMITSUBA_SYNC)
      .Msg("HdMitsubaCamera::Sync: %s\n", GetId().GetText());
  HdCamera::Sync(sceneDelegate, renderParam, dirtyBits);  // Clears dirty bits.

  const GfVec2f shutter = GetShutterClose() > GetShutterOpen()
                              ? GfVec2f(GetShutterOpen(), GetShutterClose())
                              : GfVec2f(0.0f);
  const bool need_sensor_spec =
      !is_instantiated_ || (dirty_bits_copy & HdCamera::DirtyParams);
  const bool need_transform = !is_instantiated_ ||
                              (dirty_bits_copy & HdCamera::DirtyTransform) ||
                              shutter != shutter_;
  HdContainerDataSourceHandle prim_ds =
      (need_sensor_spec || need_transform)
          ? GetPrimDataSource(sceneDelegate, GetId())
          : nullptr;

  bool sensor_structure_changed = false;
  bool sensor_leaf_changed = false;
  if (need_sensor_spec) {
    HdContainerDataSourceHandle mitsuba_source = GetMitsubaDataSource(prim_ds);
    PluginSpecNode parsed_spec =
        ParsePluginSpec(mitsuba_source, kMitsubaSensorNamespace, "perspective");
    std::optional<SdfPath> parsed_target_shape_id =
        ExtractTargetShapeId(&parsed_spec, GetId());
    sensor_structure_changed =
        (parsed_target_shape_id != target_shape_id_) ||
        !CanUpdatePluginStructureInPlace(sensor_spec_, parsed_spec) ||
        (parsed_spec != sensor_spec_ &&
         (parsed_spec.plugin_type == "irradiancemeter" ||
          parsed_target_shape_id.has_value()));
    sensor_leaf_changed = (parsed_spec != sensor_spec_);
    sensor_spec_ = std::move(parsed_spec);
    target_shape_id_ = std::move(parsed_target_shape_id);
  }

  const bool was_animated = transform_.IsAnimated();
  if (need_transform) {
    transform_ =
        SampleTransform(prim_ds, shutter).Map(UsdToMitsubaSensorTransform);
  }

  CameraSpec spec;
  spec.id = GetId();
  spec.transform = transform_;
  spec.shutter = shutter;
  spec.sensor_type = sensor_spec_.plugin_type;
  spec.sensor_spec = sensor_spec_;
  spec.fov = GetHorizontalFieldOfView();
  spec.horizontal_aperture_offset = GetHorizontalPrincipalPointOffset();
  spec.vertical_aperture_offset = GetVerticalPrincipalPointOffset();
  spec.near_clip = GetClippingRange().GetMin();
  spec.far_clip = GetClippingRange().GetMax();
  spec.dirty_bits =
      dirty_bits_copy |
      (sensor_leaf_changed ? HdCamera::DirtyParams : HdChangeTracker::Clean);
  spec.target_shape_id = target_shape_id_;

  const auto usd_params = std::make_tuple(
      spec.fov, spec.horizontal_aperture_offset, spec.vertical_aperture_offset,
      spec.near_clip, spec.far_clip);
  const bool usd_params_changed = (usd_params != usd_params_);
  usd_params_ = usd_params;

  const bool supports_usd_params_in_place =
      spec.sensor_type == "perspective" || spec.sensor_type == "thinlens" ||
      spec.sensor_type == "orthographic";
  spec.needs_rebuild = !is_instantiated_ || sensor_structure_changed ||
                       (!supports_usd_params_in_place && usd_params_changed) ||
                       transform_.IsAnimated() != was_animated ||
                       shutter != shutter_;
  is_instantiated_ = true;
  shutter_ = shutter;

  static_cast<HdMitsubaRenderParam*>(renderParam)
      ->GetScene()
      ->SyncCamera(std::move(spec));
}

float HdMitsubaCamera::GetHorizontalFieldOfView() const {
  if (_focalLength <= 0.0f || _horizontalAperture <= 0.0f) return 0.0f;
  return dr::rad_to_deg(2.0 *
                        dr::atan(_horizontalAperture / (2.0 * _focalLength)));
}

float HdMitsubaCamera::GetHorizontalPrincipalPointOffset() const {
  if (_horizontalAperture <= 0.0f) return 0.0f;
  return _horizontalApertureOffset / _horizontalAperture;
}

float HdMitsubaCamera::GetVerticalPrincipalPointOffset() const {
  if (_verticalAperture <= 0.0f) return 0.0f;
  return -_verticalApertureOffset / _verticalAperture;
}

PXR_NAMESPACE_CLOSE_SCOPE
