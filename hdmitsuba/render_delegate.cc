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

#include "hdmitsuba/render_delegate.h"

#include <memory>
#include <string>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/bprim.h>
#include <pxr/imaging/hd/extComputation.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/sprim.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/camera.h"
#include "hdmitsuba/config.h"
#include "hdmitsuba/curves.h"
#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/instancer.h"
#include "hdmitsuba/light.h"
#include "hdmitsuba/material.h"
#include "hdmitsuba/mesh.h"
#include "hdmitsuba/render_buffer.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/render_pass.h"
#include "hdmitsuba/render_settings.h"
#include "hdmitsuba/scene_manager.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(HdMitsubaRenderSettingsTokens,
                        HDMITSUBA_RENDER_SETTINGS_TOKENS);

const TfTokenVector& HdMitsubaRenderDelegate::GetSupportedRprimTypes() const {
  static const TfTokenVector* kSupportedRprimTypes = new TfTokenVector{
      HdPrimTypeTokens->mesh,
      HdPrimTypeTokens->basisCurves,
  };
  return *kSupportedRprimTypes;
}

const TfTokenVector& HdMitsubaRenderDelegate::GetSupportedSprimTypes() const {
  static const TfTokenVector* kSupportedSprimTypes = new TfTokenVector{
      HdPrimTypeTokens->camera,    HdPrimTypeTokens->sphereLight,
      HdPrimTypeTokens->domeLight, HdPrimTypeTokens->distantLight,
      HdPrimTypeTokens->rectLight, HdPrimTypeTokens->diskLight,
      HdPrimTypeTokens->material,  HdPrimTypeTokens->extComputation,
  };
  return *kSupportedSprimTypes;
}

const TfTokenVector& HdMitsubaRenderDelegate::GetSupportedBprimTypes() const {
  static const TfTokenVector* kSupportedBprimTypes = new TfTokenVector{
      HdPrimTypeTokens->renderBuffer,
      HdPrimTypeTokens->renderSettings,
  };
  return *kSupportedBprimTypes;
}

HdMitsubaRenderDelegate::HdMitsubaRenderDelegate() : HdRenderDelegate() {
  Initialize();
}

HdMitsubaRenderDelegate::HdMitsubaRenderDelegate(
    const HdRenderSettingsMap& settingsMap)
    : HdRenderDelegate(settingsMap) {
  Initialize();
}

HdMitsubaRenderDelegate::~HdMitsubaRenderDelegate() = default;

void HdMitsubaRenderDelegate::Initialize() {
  // Initialize the settings and settings descriptors.
  setting_descriptors_.resize(5);
  setting_descriptors_[0] = {"Variant", HdMitsubaRenderSettingsTokens->variant,
                             VtValue(HdMitsubaConfig::GetInstance().variant)};
  setting_descriptors_[1] = {"Samples Per Pixel",
                             HdMitsubaRenderSettingsTokens->sample_count,
                             VtValue(128)};
  setting_descriptors_[2] = {"Integrator Type",
                             HdMitsubaRenderSettingsTokens->integrator_type,
                             VtValue(std::string("path"))};
  setting_descriptors_[3] = {"Enable Interactive Mode",
                             HdRenderSettingsTokens->enableInteractive,
                             VtValue(true)};
  bool default_freezing = false;
  if (HdMitsubaConfig::GetInstance().use_kernel_freezing == 1) {
    default_freezing = true;
  }
  setting_descriptors_[4] = {"Enable Kernel Freezing (Experimental)",
                             HdMitsubaRenderSettingsTokens->use_kernel_freezing,
                             VtValue(default_freezing)};
  _PopulateDefaultSettings(setting_descriptors_);

  std::string variant = GetRenderSetting(HdMitsubaRenderSettingsTokens->variant)
                            .Get<std::string>();
  scene_impl_ =
      std::unique_ptr<SceneManager>(SceneManager::CreateSceneManager(variant));
  render_param_ = std::make_unique<HdMitsubaRenderParam>(scene_impl_.get());
}

HdRprim* HdMitsubaRenderDelegate::CreateRprim(const TfToken& typeId,
                                              const SdfPath& rprimId) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("CreateRprim: %s %s\n", rprimId.GetText(), typeId.GetText());
  if (typeId == HdPrimTypeTokens->mesh) {
    return new HdMitsubaMesh(rprimId);
  } else if (typeId == HdPrimTypeTokens->basisCurves) {
    return new HdMitsubaCurves(rprimId);
  }
  return nullptr;
}

void HdMitsubaRenderDelegate::DestroyRprim(HdRprim* rPrim) { delete rPrim; }

HdSprim* HdMitsubaRenderDelegate::CreateSprim(const TfToken& typeId,
                                              const SdfPath& sprimId) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("CreateSprim: %s %s\n", sprimId.GetText(), typeId.GetText());
  if (typeId == HdPrimTypeTokens->camera) {
    return new HdMitsubaCamera(sprimId);
  } else if (typeId == HdPrimTypeTokens->sphereLight ||
             typeId == HdPrimTypeTokens->domeLight ||
             typeId == HdPrimTypeTokens->distantLight ||
             typeId == HdPrimTypeTokens->rectLight ||
             typeId == HdPrimTypeTokens->diskLight) {
    return new HdMitsubaLight(sprimId, typeId);
  } else if (typeId == HdPrimTypeTokens->material) {
    return new HdMitsubaMaterial(sprimId);
  } else if (typeId == HdPrimTypeTokens->extComputation) {
    return new HdExtComputation(sprimId);
  }
  return nullptr;
}

void HdMitsubaRenderDelegate::DestroySprim(HdSprim* sPrim) { delete sPrim; }

HdBprim* HdMitsubaRenderDelegate::CreateBprim(const TfToken& typeId,
                                              const SdfPath& bprimId) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("CreateBprim: %s %s\n", typeId.GetText(), bprimId.GetText());
  if (typeId == HdPrimTypeTokens->renderBuffer) {
    return new HdMitsubaRenderBuffer(bprimId);
  } else if (typeId == HdPrimTypeTokens->renderSettings) {
    return new HdMitsubaRenderSettings(bprimId);
  }
  return nullptr;
}

void HdMitsubaRenderDelegate::DestroyBprim(HdBprim* bPrim) { delete bPrim; }

HdRenderPassSharedPtr HdMitsubaRenderDelegate::CreateRenderPass(
    HdRenderIndex* index, const HdRprimCollection& collection) {
  return std::make_shared<HdMitsubaRenderPass>(index, collection,
                                               scene_impl_.get());
}

HdInstancer* HdMitsubaRenderDelegate::CreateInstancer(HdSceneDelegate* delegate,
                                                      const SdfPath& id) {
  return new HdMitsubaInstancer(delegate, id);
}

void HdMitsubaRenderDelegate::DestroyInstancer(HdInstancer* instancer) {
  delete instancer;
}

HdRenderParam* HdMitsubaRenderDelegate::GetRenderParam() const {
  return render_param_.get();
}

HdResourceRegistrySharedPtr HdMitsubaRenderDelegate::GetResourceRegistry()
    const {
  return nullptr;
}

HdSprim* HdMitsubaRenderDelegate::CreateFallbackSprim(const TfToken& typeId) {
  return CreateSprim(typeId, SdfPath(typeId.GetString() + "_fallback"));
}

HdBprim* HdMitsubaRenderDelegate::CreateFallbackBprim(const TfToken& typeId) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("CreateFallbackBprim: %s\n", typeId.GetText());
  if (typeId == HdPrimTypeTokens->renderBuffer) {
    return new HdMitsubaRenderBuffer(SdfPath());
  }
  return nullptr;
}

void HdMitsubaRenderDelegate::CommitResources(HdChangeTracker* /*tracker*/) {
  VtDictionary namespaced_settings;
  namespaced_settings[HdMitsubaRenderSettingsTokens->variant.GetString()] =
      GetRenderSetting(HdMitsubaRenderSettingsTokens->variant);
  namespaced_settings[HdMitsubaRenderSettingsTokens->sample_count.GetString()] =
      GetRenderSetting(HdMitsubaRenderSettingsTokens->sample_count);
  namespaced_settings[HdMitsubaRenderSettingsTokens->integrator_type
                          .GetString()] =
      GetRenderSetting(HdMitsubaRenderSettingsTokens->integrator_type);
  namespaced_settings["enableInteractive"] =
      GetRenderSetting(HdRenderSettingsTokens->enableInteractive);
  namespaced_settings[HdMitsubaRenderSettingsTokens->use_kernel_freezing.GetString()] =
      GetRenderSetting(HdMitsubaRenderSettingsTokens->use_kernel_freezing);

  scene_impl_->UpdateNamespacedSettings(namespaced_settings);

  scene_impl_->CommitResources();
}

TfTokenVector HdMitsubaRenderDelegate::GetMaterialRenderContexts() const {
  return {TfToken("mitsuba")};
}

TfToken HdMitsubaRenderDelegate::GetMaterialBindingPurpose() const {
  return HdTokens->full;
}

HdAovDescriptor HdMitsubaRenderDelegate::GetDefaultAovDescriptor(
    const TfToken& name) const {
  if (name == HdAovTokens->color || name == TfToken("raw")) {
    return HdAovDescriptor(HdFormatFloat32Vec4, false,
                           VtValue(GfVec4f(0.0f, 0.0f, 0.0f, 1.0f)));
  }
  if (name == HdAovTokens->depth) {
    return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0f));
  }
  if (name == HdAovTokens->primId || name == HdAovTokens->instanceId ||
      name == HdAovTokens->elementId || name == TfToken("shape_index") ||
      name == TfToken("prim_index")) {
    return HdAovDescriptor(HdFormatInt32, false, VtValue(-1));
  }
  if (name == HdAovTokens->normal || name == TfToken("geo_normal") ||
      name == TfToken("position") || name == TfToken("albedo") ||
      name == TfToken("dp_du") || name == TfToken("dp_dv")) {
    return HdAovDescriptor(HdFormatFloat32Vec3, false,
                           VtValue(GfVec3f(0.0f, 0.0f, 0.0f)));
  }
  if (name == TfToken("uv") || name == TfToken("duv_dx") ||
      name == TfToken("duv_dy")) {
    return HdAovDescriptor(HdFormatFloat32Vec2, false,
                           VtValue(GfVec2f(0.0f, 0.0f)));
  }
  return HdAovDescriptor();
}

HdRenderSettingDescriptorList
HdMitsubaRenderDelegate::GetRenderSettingDescriptors() const {
  return setting_descriptors_;
}

PXR_NAMESPACE_CLOSE_SCOPE
