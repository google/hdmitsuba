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

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <drjit-core/jit.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/util.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"

PXR_NAMESPACE_OPEN_SCOPE

using mitsuba::Object;

template <typename Scene>
struct MitsubaStaticState {
  MitsubaStaticState() {
    mitsuba::Thread::static_initialization();
    mitsuba::Logger::static_initialization();
    mitsuba::Bitmap::static_initialization();

    jit_init(static_cast<uint32_t>(JitBackend::LLVM));

    mitsuba::FileResolver* fr = mitsuba::file_resolver();
    mitsuba::fs::path base_path = mitsuba::util::library_path().parent_path();
    if (!fr->contains(base_path)) {
      fr->append(base_path);
    }

    Scene::static_accel_initialization();
  }

  ~MitsubaStaticState() {
    Scene::static_accel_shutdown();
    mitsuba::Bitmap::static_shutdown();
    mitsuba::Logger::static_shutdown();
    mitsuba::Thread::static_shutdown();
    jit_shutdown();
  }
};

template <typename Scene>
std::tuple<std::unique_ptr<HdMitsubaRenderDelegate>,
           std::unique_ptr<HdRenderIndex>, std::unique_ptr<UsdImagingDelegate>,
           SceneManager*, HdMitsubaRenderParam*>
CreateRenderDelegateStateObjects() {
  auto render_delegate = std::make_unique<HdMitsubaRenderDelegate>();
  auto render_index = std::unique_ptr<HdRenderIndex>(
      pxr::HdRenderIndex::New(render_delegate.get(), {}));
  auto scene_delegate =
      std::make_unique<UsdImagingDelegate>(render_index.get(), SdfPath("/"));
  auto* render_param =
      static_cast<HdMitsubaRenderParam*>(render_delegate->GetRenderParam());
  auto* scene_manager = render_param->GetScene();
  return std::make_tuple(std::move(render_delegate), std::move(render_index),
                         std::move(scene_delegate), scene_manager,
                         render_param);
}

PXR_NAMESPACE_CLOSE_SCOPE
