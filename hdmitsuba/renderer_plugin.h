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

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaRendererPlugin final : public HdRendererPlugin {
 public:
  HdMitsubaRendererPlugin() = default;
  ~HdMitsubaRendererPlugin() override = default;

  HdRenderDelegate* CreateRenderDelegate() override;

  HdRenderDelegate* CreateRenderDelegate(
      const HdRenderSettingsMap& settingsMap) override;

  void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;

  bool IsSupported(const HdRendererCreateArgs& rendererCreateArgs,
                   std::string* reasonWhyNot = nullptr) const override;

 private:
  // This class is not copyable.
  HdMitsubaRendererPlugin(const HdMitsubaRendererPlugin&) = delete;
  HdMitsubaRendererPlugin& operator=(const HdMitsubaRendererPlugin&) = delete;
};

PXR_NAMESPACE_CLOSE_SCOPE
