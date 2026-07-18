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

#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/pxr.h>
PXR_NAMESPACE_OPEN_SCOPE

class SceneManager;

class HdMitsubaRenderPass final : public HdRenderPass {
 public:
  HdMitsubaRenderPass(HdRenderIndex* index,
                      const HdRprimCollection& collection);
  ~HdMitsubaRenderPass() override = default;

  bool IsConverged() const override;

 protected:
  void _Execute(const HdRenderPassStateSharedPtr& renderPassState,
                const TfTokenVector& renderTags) override;

 private:
  SceneManager* last_scene_manager_ = nullptr;
  HdRenderPassAovBindingVector aov_bindings_;
};

PXR_NAMESPACE_CLOSE_SCOPE
