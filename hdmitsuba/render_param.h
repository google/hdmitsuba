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

#include <atomic>
#include <utility>

#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolver.h>

#include "hdmitsuba/render_delegate.h"
#include "hdmitsuba/scene_manager.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaRenderDelegate;

// Binds `shared_data` onto the current thread's ArResolver cache stack for the
// lifetime of this guard. If `shared_data` is empty, the cache of the enclosing
// cache scope on this thread is reused, or a fresh cache is created if there is
// none.
class ScopedResolverCache {
 public:
  explicit ScopedResolverCache(VtValue shared_data)
      : cache_data_(std::move(shared_data)) {
    ArGetResolver().BeginCacheScope(&cache_data_);
  }

  ~ScopedResolverCache() { ArGetResolver().EndCacheScope(&cache_data_); }

  ScopedResolverCache(const ScopedResolverCache&) = delete;
  ScopedResolverCache& operator=(const ScopedResolverCache&) = delete;

 private:
  VtValue cache_data_;
};

class HdMitsubaRenderParam final : public HdRenderParam {
 public:
  explicit HdMitsubaRenderParam(SceneManager* scene) : scene_(scene) {}

  SceneManager* GetScene() const { return scene_.load(); }
  void SetScene(SceneManager* scene) { scene_.store(scene); }

  // Replaces the resolver cache shared by the Hydra sync and commit stages of
  // a frame (see `ScopedResolverCache`) with a fresh one, or with the cache of
  // the calling thread's enclosing cache scope (e.g., the one of
  // `RenderEngine::Render`), if any.
  void ResetResolverCache() {
    resolver_cache_data_ = VtValue();
    ArGetResolver().BeginCacheScope(&resolver_cache_data_);
    ArGetResolver().EndCacheScope(&resolver_cache_data_);
  }

  const VtValue& ResolverCacheData() const { return resolver_cache_data_; }

  // Releases the resolver cache, so that it (and, e.g., open `.usdz` archives)
  // does not outlive the frame once the last `ScopedResolverCache` ends, unless
  // it is still held by an enclosing cache scope.
  VtValue TakeResolverCache() {
    return std::exchange(resolver_cache_data_, VtValue());
  }

 private:
  std::atomic<SceneManager*> scene_;
  VtValue resolver_cache_data_;
};

PXR_NAMESPACE_CLOSE_SCOPE
