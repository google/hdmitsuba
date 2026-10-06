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

#include "hdmitsuba/renderer_plugin.h"

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/pxr.h>

#include "hdmitsuba/render_delegate.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register this plugin with the TfType system.
TF_REGISTRY_FUNCTION(TfType) {
  HdRendererPluginRegistry::Define<HdMitsubaRendererPlugin>();
}

HdRenderDelegate* HdMitsubaRendererPlugin::CreateRenderDelegate() {
  return new HdMitsubaRenderDelegate();
}

HdRenderDelegate* HdMitsubaRendererPlugin::CreateRenderDelegate(
    const HdRenderSettingsMap& settingsMap) {
  return new HdMitsubaRenderDelegate(settingsMap);
}

void HdMitsubaRendererPlugin::DeleteRenderDelegate(
    HdRenderDelegate* renderDelegate) {
  delete renderDelegate;
}

bool HdMitsubaRendererPlugin::IsSupported(
    const HdRendererCreateArgs& /*rendererCreateArgs*/,
    std::string* /*reasonWhyNot*/) const {
  return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
