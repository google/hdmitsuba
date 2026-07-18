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

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/pxr.h>

#include "hdmitsuba/render_delegate.h"
#include "hdmitsuba/scene_manager.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaRenderDelegate;

class HdMitsubaRenderParam final : public HdRenderParam {
 public:
  explicit HdMitsubaRenderParam(SceneManager* scene) : scene_(scene) {}

  SceneManager* GetScene() const { return scene_.load(); }
  void SetScene(SceneManager* scene) { scene_.store(scene); }

 private:
  std::atomic<SceneManager*> scene_;
};

PXR_NAMESPACE_CLOSE_SCOPE
