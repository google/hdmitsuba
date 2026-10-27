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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <drjit-core/jit.h>
#include <drjit/matrix.h>
#include <drjit/transform.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <nanothread/nanothread.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolverScopedCache.h>
#include <pxr/usd/sdf/path.h>

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
           const HdDataSourceLocator& locator, const T& default_value = T(),
           HdSampledDataSource::Time shutter_offset = 0.0f) {
  if (!container) return default_value;
  if (auto data_source = HdSampledDataSource::Cast(
          HdContainerDataSource::Get(container, locator))) {
    VtValue value = data_source->GetValue(shutter_offset);
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

inline HdContainerDataSourceHandle GetPrimDataSource(
    HdSceneDelegate* scene_delegate, const SdfPath& id) {
  HdSceneIndexBaseRefPtr scene_index =
      scene_delegate->GetRenderIndex().GetTerminalSceneIndex();
  if (!TF_VERIFY(scene_index)) return nullptr;
  return scene_index->GetPrim(id).dataSource;
}

inline bool GetPrimVisible(const HdContainerDataSourceHandle& prim_source,
                           bool default_visible = true) {
  if (auto schema = HdVisibilitySchema::GetFromParent(prim_source)) {
    if (auto vis = schema.GetVisibility()) {
      return vis->GetValue(0.0f).Get<bool>();
    }
  }
  return default_visible;
}

inline GfMatrix4d GetPrimTransform(
    const HdContainerDataSourceHandle& prim_source) {
  if (auto schema = HdXformSchema::GetFromParent(prim_source)) {
    if (auto matrix = schema.GetMatrix()) {
      return matrix->GetValue(0.0f).Get<GfMatrix4d>();
    }
  }
  return GfMatrix4d(1.0);
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

// Strips scale and shear from `transform`, keeping only rotation and
// translation.
inline ScalarAffineTransform4f RemoveScaleFromTransform(
    const ScalarAffineTransform4f& transform) {
  if (!transform.has_scale()) {
    return transform;
  }
  auto [s, q, t] = drjit::transform_decompose(transform.matrix);
  return ScalarAffineTransform4f(
      drjit::transform_compose<drjit::Matrix<float, 4>>(
          drjit::Matrix<float, 3>(1.0f), q, t));
}

inline std::optional<ScalarAffineTransform3f> ExtractTransform3f(
    const VtValue& val) {
  if (val.IsHolding<GfMatrix3f>()) {
    return UsdToMitsubaTransform(GfMatrix3d(val.Get<GfMatrix3f>()));
  }
  if (val.IsHolding<GfMatrix3d>()) {
    return UsdToMitsubaTransform(val.Get<GfMatrix3d>());
  }
  return std::nullopt;
}

inline std::optional<ScalarAffineTransform4f> ExtractTransform4f(
    const VtValue& val) {
  if (val.IsHolding<GfMatrix4f>()) {
    return UsdToMitsubaTransform(GfMatrix4d(val.Get<GfMatrix4f>()));
  }
  if (val.IsHolding<GfMatrix4d>()) {
    return UsdToMitsubaTransform(val.Get<GfMatrix4d>());
  }
  return std::nullopt;
}

// Ensures the directory of hdMitsuba's own Mitsuba plugins (e.g. `ies`) is in
// Mitsuba's file resolver, so that the plugin manager loads them by name.
void EnsurePluginDirInFileResolver();

// Returns the resolved path of an `SdfAssetPath` value (or its asset path if
// it is unresolved), or an empty string otherwise.
std::string ResolvePathFromValue(const VtValue& value);

// Creates the Mitsuba object of type `type` described by `props`, and returns
// the first object of its expansion (e.g. the texture that a `bitmap` plugin
// expands to), or the object itself if it does not expand.
mitsuba::ref<mitsuba::Object> CreateExpandedObject(
    const mitsuba::Properties& props, std::string_view variant,
    mitsuba::ObjectType type);

// Enters a fresh JIT scope on the calling thread, see `jit_new_scope()`.
template <typename Float>
void NewJitScope() {
  if constexpr (drjit::is_jit_v<Float>) {
    jit_new_scope(drjit::backend_v<Float>);
  }
}

// Runs `fn(i)` for all `i` in [0, n) on worker threads.
//
// Using Dr.Jit in a multithreaded environment requires a fresh JIT scope
// whenever variables cross thread boundaries. Like Mitsuba's parallel scene
// loading (`parser.cpp`), a scope is pre-allocated for every item on the
// calling thread, which continues in a scope succeeding all item scopes. All
// work queued by `fn` has finished on return. Worker threads share the
// calling thread's ArResolver cache.
template <typename Float, typename Fn>
void JitParallelFor(size_t n, Fn&& fn) {
  std::vector<uint32_t> scopes(n);
  if constexpr (drjit::is_jit_v<Float>) {
    jit_eval();  // Flush any pending side effects before multithreaded work.
    for (uint32_t& scope : scopes) {
      scope = jit_new_scope(drjit::backend_v<Float>);
    }
  }
  ArResolverScopedCache resolver_cache;
  auto run = [&](drjit::blocked_range<size_t> range) {
    ArResolverScopedCache worker_resolver_cache(&resolver_cache);
    for (size_t i = range.begin(); i != range.end(); ++i) {
      if constexpr (drjit::is_jit_v<Float>) {
        jit_set_scope(drjit::backend_v<Float>, scopes[i]);
      }
      fn(i);
    }
    if constexpr (drjit::is_jit_v<Float>) {
      // Side effects (e.g., scatters) queued on this thread can only be
      // evaluated by this thread.
      jit_eval();
      jit_flush_thread();
    }
  };
  drjit::parallel_for(drjit::blocked_range<size_t>(0, n), std::move(run));
  if constexpr (drjit::is_jit_v<Float>) {
    jit_sync_device();
    jit_new_scope(drjit::backend_v<Float>);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
