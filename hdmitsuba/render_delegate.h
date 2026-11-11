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

#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

#define HDMITSUBA_RENDER_SETTINGS_TOKENS \
  ((variant, "mitsuba:variant")) \
  ((sample_count, "mitsuba:sample_count")) \
  ((seed, "mitsuba:seed")) \
  ((interactive_samples_per_pass, "mitsuba:interactive_samples_per_pass")) \
  ((integrator_type, "mitsuba:integrator:type")) \
  ((integrator_max_depth, "mitsuba:integrator:max_depth")) \
  ((integrator_rr_depth, "mitsuba:integrator:rr_depth")) \
  ((use_kernel_freezing, "mitsuba:use_kernel_freezing")) \
  ((denoise, "mitsuba:denoise"))

TF_DECLARE_PUBLIC_TOKENS(HdMitsubaRenderSettingsTokens,
                         HDMITSUBA_RENDER_SETTINGS_TOKENS);

struct MitsubaAovSpec {
  std::string mitsuba_aov;  // Empty for primary color/raw buffer
  int channels;
  HdFormat format;
  VtValue clear_value;
};

const MitsubaAovSpec* FindMitsubaAovSpec(const TfToken& name);

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

  VtDictionary GetRenderStats() const override;

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

  void Update() override;
  void CommitResources(HdChangeTracker* tracker) override;

  TfToken GetMaterialBindingPurpose() const override;
  TfTokenVector GetMaterialRenderContexts() const override;
  bool IsParallelSyncEnabled(const TfToken& primType) const override;

  HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;

  HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

  void SetRenderSetting(const TfToken& key, const VtValue& value) override;

  // Stores the settings authored on the render settings prim, which
  // CommitResources() applies.
  void SyncPrimSettings(const VtDictionary& namespaced_settings);

  bool HasPendingSettings() const {
    return settings_dirty_ ||
           GetRenderSettingsVersion() != last_settings_version_;
  }

 private:
  void Initialize();

  // Returns the value of a setting. Settings are either set on the render
  // delegate (at construction or through SetRenderSetting(), e.g., as an
  // override of the host) or authored on the render settings prim, and the
  // most recent change applies. A value first synced from the prim yields to
  // one set on the render delegate before, as does a prim edit synced in the
  // same frame as the render delegate was set (e.g., when a host reapplies its
  // override). Unset settings use the default of their descriptor, if any.
  VtValue ResolveSetting(const TfToken& key) const;

  // This class is not copyable.
  HdMitsubaRenderDelegate(const HdMitsubaRenderDelegate&) = delete;
  HdMitsubaRenderDelegate& operator=(const HdMitsubaRenderDelegate&) = delete;

  std::unique_ptr<SceneManager> scene_impl_;
  std::unique_ptr<HdMitsubaRenderParam> render_param_;
  std::string current_variant_;
  unsigned int last_settings_version_ = 0;
  // The settings last synced from the render settings prim.
  VtDictionary prim_settings_;
  // The settings set on the render delegate that the render settings prim has
  // not changed since.
  TfToken::HashSet delegate_settings_;
  // The settings set on the render delegate since the last CommitResources().
  TfToken::HashSet recent_delegate_settings_;
  // Whether `prim_settings_` or `delegate_settings_` changed since the last
  // CommitResources().
  bool settings_dirty_ = false;

  HdRenderSettingDescriptorList setting_descriptors_;
};

PXR_NAMESPACE_CLOSE_SCOPE
