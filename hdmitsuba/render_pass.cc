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

#include "hdmitsuba/render_pass.h"

#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <optional>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/pxr.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaRenderPass::HdMitsubaRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection)
    : HdRenderPass(index, collection) {}

void HdMitsubaRenderPass::_Execute(
    const HdRenderPassStateSharedPtr& renderPassState,
    const TfTokenVector& /*renderTags*/) {
  if (!TF_VERIFY(renderPassState != nullptr)) {
    return;
  }
  HdRenderPassAovBindingVector aov_bindings = renderPassState->GetAovBindings();
  if (aov_bindings.empty()) {
    return;
  }

  HdMitsubaRenderParam* render_param = static_cast<HdMitsubaRenderParam*>(
      GetRenderIndex()->GetRenderDelegate()->GetRenderParam());
  SceneManager* scene_manager = render_param->GetScene();
  if (aov_bindings != aov_bindings_ || scene_manager != last_scene_manager_) {
    scene_manager->SetAovBindings(this, aov_bindings);
    aov_bindings_ = aov_bindings;
    last_scene_manager_ = scene_manager;
  }
  std::optional<GfRect2i> crop_window;
  if (renderPassState->GetFraming().IsValid()) {
    crop_window = renderPassState->GetFraming().dataWindow;
  }
  scene_manager->Render(this, renderPassState->GetCamera(), crop_window);
}

bool HdMitsubaRenderPass::IsConverged() const {
  return static_cast<HdMitsubaRenderParam*>(
             GetRenderIndex()->GetRenderDelegate()->GetRenderParam())
      ->GetScene()
      ->IsConverged();
}

PXR_NAMESPACE_CLOSE_SCOPE
