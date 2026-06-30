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

#pragma once

#include <string>

#include <drjit/matrix.h>
#include <mitsuba/core/transform.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/dataSource.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

template <typename T>
T GetParam(const HdContainerDataSourceHandle& container, const TfToken& name,
           const T& default_value = T()) {
  if (!container) return default_value;
  if (auto data_source = HdSampledDataSource::Cast(container->Get(name))) {
    VtValue value = data_source->GetValue(0.0f);
    if (value.IsHolding<T>()) {
      return value.UncheckedGet<T>();
    } else {
      TF_WARN("GetParam type mismatch for '%s': expected %s, got %s (empty=%d)",
              name.GetText(), TfType::Find<T>().GetTypeName().c_str(),
              value.GetTypeName().c_str(), value.IsEmpty());
    }
  }
  return default_value;
}

template <typename T>
T GetParam(const HdContainerDataSourceHandle& container,
           const HdDataSourceLocator& locator, const T& default_value = T()) {
  if (!container) return default_value;
  if (auto data_source = HdSampledDataSource::Cast(
          HdContainerDataSource::Get(container, locator))) {
    VtValue value = data_source->GetValue(0.0f);
    if (value.IsHolding<T>()) {
      return value.UncheckedGet<T>();
    } else {
      TF_WARN(
          "GetParam type mismatch for locator '%s': expected %s, got %s "
          "(empty=%d)",
          locator.GetString().c_str(), TfType::Find<T>().GetTypeName().c_str(),
          value.GetTypeName().c_str(), value.IsEmpty());
    }
  }
  return default_value;
}

using ScalarAffineTransform4f =
    mitsuba::Transform<mitsuba::Point<float, 4>, true>;

inline ScalarAffineTransform4f UsdToMitsubaTransform(const GfMatrix4d& transform) {
  const double* m = transform.GetArray();
  drjit::Matrix<float, 4> to_world_mat(
      static_cast<float>(m[0]), static_cast<float>(m[4]), static_cast<float>(m[8]), static_cast<float>(m[12]),
      static_cast<float>(m[1]), static_cast<float>(m[5]), static_cast<float>(m[9]), static_cast<float>(m[13]),
      static_cast<float>(m[2]), static_cast<float>(m[6]), static_cast<float>(m[10]), static_cast<float>(m[14]),
      static_cast<float>(m[3]), static_cast<float>(m[7]), static_cast<float>(m[11]), static_cast<float>(m[15]));
  return ScalarAffineTransform4f(to_world_mat);
}

PXR_NAMESPACE_CLOSE_SCOPE
