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

#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

#define HDMITSUBA_RENDER_SETTINGS_TOKENS \
  ((variant, "mitsuba:variant")) \
  ((sample_count, "mitsuba:sample_count")) \
  ((integrator_type, "mitsuba:integrator:type")) \
  ((progressive, "mitsuba:progressive")) \
  ((use_kernel_freezing, "mitsuba:use_kernel_freezing"))
// clang-format on

TF_DECLARE_PUBLIC_TOKENS(HdMitsubaRenderSettingsTokens,
                         HDMITSUBA_RENDER_SETTINGS_TOKENS);

class HdMitsubaRenderParam;
class SceneManager;

class HdMitsubaRenderDelegate final : public HdRenderDelegate {
 public:
  HdMitsubaRenderDelegate();
  explicit HdMitsubaRenderDelegate(const HdRenderSettingsMap& settingsMap);
  ~HdMitsubaRenderDelegate() override;

  const TfTokenVector& GetSupportedRprimTypes() const override;
  const TfTokenVector& GetSupportedSprimTypes() const override;
  const TfTokenVector& GetSupportedBprimTypes() const override;

  HdRprim* CreateRprim(const TfToken& typeId, const SdfPath& rprimId) override;
  void DestroyRprim(HdRprim* rPrim) override;

  HdSprim* CreateSprim(const TfToken& typeId, const SdfPath& sprimId) override;
  void DestroySprim(HdSprim* sPrim) override;

  HdBprim* CreateBprim(const TfToken& typeId, const SdfPath& bprimId) override;
  void DestroyBprim(HdBprim* bPrim) override;

  HdRenderPassSharedPtr CreateRenderPass(
      HdRenderIndex* index, const HdRprimCollection& collection) override;

  HdInstancer* CreateInstancer(HdSceneDelegate* delegate,
                               const SdfPath& id) override;
  void DestroyInstancer(HdInstancer* instancer) override;

  HdRenderParam* GetRenderParam() const override;

  HdResourceRegistrySharedPtr GetResourceRegistry() const override;

  HdSprim* CreateFallbackSprim(const TfToken& typeId) override;
  HdBprim* CreateFallbackBprim(const TfToken& typeId) override;

  void CommitResources(HdChangeTracker* tracker) override;

  TfToken GetMaterialBindingPurpose() const override;
  TfTokenVector GetMaterialRenderContexts() const override;

  HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;

  HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

 private:
  void Initialize();

  // This class is not copyable.
  HdMitsubaRenderDelegate(const HdMitsubaRenderDelegate&) = delete;
  HdMitsubaRenderDelegate& operator=(const HdMitsubaRenderDelegate&) = delete;

  std::unique_ptr<SceneManager> scene_impl_;
  std::unique_ptr<HdMitsubaRenderParam> render_param_;

  HdRenderSettingDescriptorList setting_descriptors_;
};

PXR_NAMESPACE_CLOSE_SCOPE
