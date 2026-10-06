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

#include <cstdint>
#include <string>

#include <drjit-core/jit.h>
#include <drjit/matrix.h>
#include <mitsuba/core/transform.h>
#include <pxr/base/gf/matrix3d.h>
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

using ScalarAffineTransform3f =
    mitsuba::Transform<mitsuba::Point<float, 3>, true>;
using ScalarAffineTransform4f =
    mitsuba::Transform<mitsuba::Point<float, 4>, true>;

// USD's Gf matrices use the row-vector convention (p' = p * M, translation in
// the last row) while Mitsuba uses column vectors (p' = M * p). Converting is
// therefore a transpose. All USD -> Mitsuba matrix conversions must go through
// these helpers so the convention is applied consistently.
inline ScalarAffineTransform4f UsdToMitsubaTransform(const GfMatrix4d& m) {
  drjit::Matrix<float, 4> t;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) t(i, j) = static_cast<float>(m[j][i]);
  return ScalarAffineTransform4f(t);
}

inline ScalarAffineTransform3f UsdToMitsubaTransform(const GfMatrix3d& m) {
  drjit::Matrix<float, 3> t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) t(i, j) = static_cast<float>(m[j][i]);
  return ScalarAffineTransform3f(t);
}

// Using Dr.Jit in a multithreaded environment requires explicitly creating
// JIT scopes on each thread. This RAII struct should be used in code blocks
// that may be executed on different threads, with dependencies crossing
// thread boundaries (e.g., accessing a texture that was initialized on a
// different thread).
template <typename Float>
struct JitScopeGuard {
  uint32_t backend = 0;
  uint32_t prev_scope = 0;

  JitScopeGuard() {
    if constexpr (drjit::is_cuda_v<Float>) {
      backend = (uint32_t)JitBackend::CUDA;
    } else if constexpr (drjit::is_llvm_v<Float>) {
      backend = (uint32_t)JitBackend::LLVM;
    } else if constexpr (drjit::is_metal_v<Float>) {
      backend = (uint32_t)JitBackend::Metal;
    }
    if (backend) {
      prev_scope = jit_scope((JitBackend)backend);
      jit_new_scope((JitBackend)backend);
    }
  }

  ~JitScopeGuard() {
    if (backend) {
      jit_set_scope((JitBackend)backend, prev_scope);
    }
  }
};

PXR_NAMESPACE_CLOSE_SCOPE
