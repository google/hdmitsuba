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

#include <string>
#include <utility>

#include <drjit/math.h>
#include <drjit/sphere.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

ScalarAffineTransform4f RemoveScaleFromTransform(
    const ScalarAffineTransform4f& transform) {
  auto [s, q, t] = dr::transform_decompose(transform.matrix);
  return ScalarAffineTransform4f(dr::transform_compose<dr::Matrix<float, 4>>(
      dr::Matrix<float, 3>(1.0f), q, t));
}

ScalarAffineTransform4f UsdToMitsubaSensorTransform(
    const GfMatrix4d& transform) {
  ScalarAffineTransform4f to_world = UsdToMitsubaTransform(transform);
  if (to_world.has_scale()) {
    to_world = RemoveScaleFromTransform(to_world);
  }
  using ScalarVector3f = mitsuba::Vector<float, 3>;
  to_world =
      to_world * ScalarAffineTransform4f::rotate(ScalarVector3f(0, 1, 0), 180);
  return to_world;
}

}  // namespace

HdMitsubaCamera::HdMitsubaCamera(const SdfPath& id) : HdCamera(id) {}

void HdMitsubaCamera::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
  HdDirtyBits dirty_bits_copy = *dirtyBits;
  TF_DEBUG(HDMITSUBA_SYNC)
      .Msg("HdMitsubaCamera::Sync: %s\n", GetId().GetText());
  HdCamera::Sync(sceneDelegate, renderParam, dirtyBits);  // Clears dirty bits.

  std::string sensor_type =
      sceneDelegate
          ->GetCameraParamValue(GetId(), TfToken("mitsuba:sensor:type"))
          .GetWithDefault<std::string>("perspective");
  if (dirty_bits_copy & HdCamera::DirtyParams) {
    film_pixel_filter_type_ =
        sceneDelegate
            ->GetCameraParamValue(
                GetId(), TfToken("mitsuba:sensor:film:pixel_filter:type"))
            .GetWithDefault<std::string>("");
  }

  CameraSpec spec;
  spec.id = GetId();
  spec.transform =
      UsdToMitsubaSensorTransform(sceneDelegate->GetTransform(GetId()));
  spec.sensor_type = sensor_type;
  spec.fov = GetHorizontalFieldOfView();
  spec.horizontal_aperture_offset = GetHorizontalPrincipalPointOffset();
  spec.vertical_aperture_offset = GetVerticalPrincipalPointOffset();
  spec.pixel_filter_type = film_pixel_filter_type_;
  spec.near_clip = GetClippingRange().GetMin();
  spec.far_clip = GetClippingRange().GetMax();
  spec.dirty_bits = dirty_bits_copy;

  spec.needs_rebuild = !is_instantiated_;
  spec.needs_rebuild |= sensor_type != sensor_type_;
  spec.needs_rebuild |= (sensor_type != "perspective" &&
                         (dirty_bits_copy & HdCamera::DirtyParams));
  is_instantiated_ = true;
  sensor_type_ = sensor_type;

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
