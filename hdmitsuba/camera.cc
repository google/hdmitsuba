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
#include <cstdint>
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
  std::string path = ExtractStringOrToken(shape_it->second);
  sensor_spec->params.erase(shape_it);
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

std::optional<FilmCrop> ExtractFilmCrop(PluginSpecNode* sensor_spec) {
  auto film_it = sensor_spec->children.find("film");
  if (film_it == sensor_spec->children.end()) {
    return std::nullopt;
  }
  auto take_int = [&](const char* key) -> std::optional<int64_t> {
    auto it = film_it->second.params.find(key);
    if (it == film_it->second.params.end()) {
      return std::nullopt;
    }
    std::optional<int64_t> value = ExtractInt64Value(it->second);
    film_it->second.params.erase(it);
    return value;
  };
  FilmCrop crop;
  crop.offset_x = take_int("crop_offset_x");
  crop.offset_y = take_int("crop_offset_y");
  crop.width = take_int("crop_width");
  crop.height = take_int("crop_height");
  if (film_it->second.Empty()) {
    sensor_spec->children.erase(film_it);
  }
  if (!crop.offset_x.has_value() && !crop.offset_y.has_value() &&
      !crop.width.has_value() && !crop.height.has_value()) {
    return std::nullopt;
  }
  return crop;
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
  const float aperture_radius =
      (_fStop > 0.0f && _focusDistance > 0.0f && _focalLength > 0.0f)
          ? 0.5f * _focalLength / _fStop
          : 0.0f;
  bool need_sensor_spec =
      !is_instantiated_ || (dirty_bits_copy & HdCamera::DirtyParams);
  bool need_transform = !is_instantiated_ ||
                        (dirty_bits_copy & HdCamera::DirtyTransform) ||
                        shutter != shutter_;
  HdContainerDataSourceHandle prim_source =
      (need_sensor_spec || need_transform)
          ? GetPrimDataSource(sceneDelegate, GetId())
          : nullptr;

  bool sensor_spec_changed = false;
  if (need_sensor_spec) {
    PluginSpecNode parsed_spec = ParsePluginSpec(
        GetMitsubaDataSource(prim_source), kMitsubaSensorNamespace,
        aperture_radius > 0.0f ? "thinlens" : "perspective");
    std::optional<SdfPath> parsed_target_shape_id =
        ExtractTargetShapeId(&parsed_spec, GetId());
    film_crop_ = ExtractFilmCrop(&parsed_spec);
    sensor_spec_changed = (parsed_spec != sensor_spec_) ||
                          (parsed_target_shape_id != target_shape_id_);
    sensor_spec_ = std::move(parsed_spec);
    target_shape_id_ = std::move(parsed_target_shape_id);
  }

  const bool was_animated = transform_.IsAnimated();
  if (need_transform) {
    transform_ =
        SampleTransform(prim_source, shutter).Map(UsdToMitsubaSensorTransform);
  }

  CameraSpec spec;
  spec.id = GetId();
  spec.transform = transform_;
  spec.shutter = shutter;
  spec.sensor_spec = sensor_spec_;
  spec.fov = GetHorizontalFieldOfView();
  spec.horizontal_aperture_offset = GetHorizontalPrincipalPointOffset();
  spec.vertical_aperture_offset = GetVerticalPrincipalPointOffset();
  spec.aperture_radius = aperture_radius;
  spec.focus_distance = _focusDistance;
  spec.near_clip = GetClippingRange().GetMin();
  spec.far_clip = GetClippingRange().GetMax();
  spec.dirty_bits = dirty_bits_copy;
  spec.target_shape_id = target_shape_id_;
  spec.film_crop = film_crop_;

  spec.needs_rebuild = !is_instantiated_ || sensor_spec_changed ||
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
