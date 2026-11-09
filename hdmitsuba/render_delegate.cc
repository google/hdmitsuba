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

#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/bprim.h>
#include <pxr/imaging/hd/extComputation.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/perfLog.h>
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
#include "hdmitsuba/implicit_surface.h"
#include "hdmitsuba/instancer.h"
#include "hdmitsuba/light.h"
#include "hdmitsuba/material.h"
#include "hdmitsuba/mesh.h"
#include "hdmitsuba/render_buffer.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/render_pass.h"
#include "hdmitsuba/render_settings.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/particle_field.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(HdMitsubaRenderSettingsTokens,
                        HDMITSUBA_RENDER_SETTINGS_TOKENS);

const TfTokenVector& HdMitsubaRenderDelegate::GetSupportedRprimTypes() const {
  static const TfTokenVector* kSupportedRprimTypes =
      HdMitsubaConfig::GetInstance().tessellate_implicit_surfaces
          ? new TfTokenVector{HdPrimTypeTokens->mesh,
                              HdPrimTypeTokens->basisCurves,
                              HdPrimTypeTokens->particleField}
          : new TfTokenVector{
                HdPrimTypeTokens->mesh, HdPrimTypeTokens->basisCurves,
                HdPrimTypeTokens->particleField, HdPrimTypeTokens->sphere,
                HdPrimTypeTokens->cylinder};
  return *kSupportedRprimTypes;
}

const TfTokenVector& HdMitsubaRenderDelegate::GetSupportedSprimTypes() const {
  static const TfTokenVector* kSupportedSprimTypes = new TfTokenVector{
      HdPrimTypeTokens->camera,         HdPrimTypeTokens->sphereLight,
      HdPrimTypeTokens->domeLight,      HdPrimTypeTokens->distantLight,
      HdPrimTypeTokens->rectLight,      HdPrimTypeTokens->diskLight,
      HdPrimTypeTokens->cylinderLight,  HdPrimTypeTokens->material,
      HdPrimTypeTokens->extComputation,
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
  for (const auto& [key, value] : settingsMap) {
    if (!value.IsEmpty()) {
      host_setting_overrides_[key] = value;
    }
  }
  Initialize();
}

HdMitsubaRenderDelegate::~HdMitsubaRenderDelegate() = default;

void HdMitsubaRenderDelegate::Initialize() {
  // Initialize the settings and settings descriptors.
  const HdMitsubaConfig& config = HdMitsubaConfig::GetInstance();
  setting_descriptors_ = {
      {"Variant", HdMitsubaRenderSettingsTokens->variant,
       VtValue(config.variant)},
      {"Samples Per Pixel", HdMitsubaRenderSettingsTokens->sample_count,
       VtValue(128)},
      {"Seed", HdMitsubaRenderSettingsTokens->seed, VtValue(0)},
      {"Interactive Samples Per Pass",
       HdMitsubaRenderSettingsTokens->interactive_samples_per_pass, VtValue(1)},
      {"Integrator Type", HdMitsubaRenderSettingsTokens->integrator_type,
       VtValue(std::string("path"))},
      {"Integrator Max Depth",
       HdMitsubaRenderSettingsTokens->integrator_max_depth, VtValue()},
      {"Integrator Russian Roulette Depth",
       HdMitsubaRenderSettingsTokens->integrator_rr_depth, VtValue()},
      {"Enable Interactive Mode", HdRenderSettingsTokens->enableInteractive,
       VtValue(true)},
      {"Enable Kernel Freezing (Experimental)",
       HdMitsubaRenderSettingsTokens->use_kernel_freezing,
       VtValue(config.use_kernel_freezing == 1)},
      {"Enable Denoising (OptiX on CUDA, OIDN on CPU)",
       HdMitsubaRenderSettingsTokens->denoise, VtValue(false)},
  };
  _PopulateDefaultSettings(setting_descriptors_);

  current_variant_ = HdMitsubaConfig::GetInstance().variant;
  const VtValue variant_val =
      GetRenderSetting(HdMitsubaRenderSettingsTokens->variant);
  std::string configured_variant;
  if (variant_val.IsHolding<std::string>()) {
    configured_variant = variant_val.UncheckedGet<std::string>();
  } else if (variant_val.IsHolding<TfToken>()) {
    configured_variant = variant_val.UncheckedGet<TfToken>().GetString();
  }
  if (!configured_variant.empty()) {
    current_variant_ = configured_variant;
  }
  // Defer syncing Python's global `mi.variant()` until `CommitResources()`
  // resolves `UsdRenderSettings` (`mitsuba:variant`), avoiding a spurious
  // `"scalar_rgb"` switch before the scene's actual variant is known.
  scene_impl_ = std::unique_ptr<SceneManager>(SceneManager::CreateSceneManager(
      current_variant_, /*sync_python_variant=*/false));
  render_param_ =
      std::make_unique<HdMitsubaRenderParam>(this, scene_impl_.get());
  settings_dirty_ = true;
}

HdRprim* HdMitsubaRenderDelegate::CreateRprim(const TfToken& typeId,
                                              const SdfPath& rprimId) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("CreateRprim: %s %s\n", rprimId.GetText(), typeId.GetText());
  if (typeId == HdPrimTypeTokens->mesh) {
    return new HdMitsubaMesh(rprimId);
  } else if (typeId == HdPrimTypeTokens->basisCurves) {
    return new HdMitsubaCurves(rprimId);
  } else if (typeId == HdPrimTypeTokens->particleField) {
    return new HdMitsubaParticleField(rprimId);
  } else if (typeId == HdPrimTypeTokens->sphere ||
             typeId == HdPrimTypeTokens->cylinder) {
    return new HdMitsubaImplicitSurface(rprimId, typeId);
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
             typeId == HdPrimTypeTokens->diskLight ||
             typeId == HdPrimTypeTokens->cylinderLight) {
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
  return std::make_shared<HdMitsubaRenderPass>(index, collection);
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

void HdMitsubaRenderDelegate::Update() {
  HdRenderDelegate::Update();
  render_param_->ResetResolverCache();
}

void HdMitsubaRenderDelegate::CommitResources(HdChangeTracker* /*tracker*/) {
  ScopedResolverCache resolver_cache(render_param_->TakeResolverCache());

  if (settings_dirty_) {
    VtDictionary effective_settings;
    for (const HdRenderSettingDescriptor& descriptor : setting_descriptors_) {
      if (!descriptor.defaultValue.IsEmpty()) {
        effective_settings[descriptor.key.GetString()] =
            descriptor.defaultValue;
      }
    }
    for (const auto& [key, value] : scene_render_settings_) {
      if (!value.IsEmpty()) {
        effective_settings[key] = value;
      }
    }
    for (const auto& [key, value] : host_setting_overrides_) {
      if (!value.IsEmpty()) {
        effective_settings[key.GetString()] = value;
      }
    }

    std::string target_variant = HdMitsubaConfig::GetInstance().variant;
    auto variant_it = effective_settings.find(
        HdMitsubaRenderSettingsTokens->variant.GetString());
    if (variant_it != effective_settings.end()) {
      std::string candidate_variant;
      if (variant_it->second.IsHolding<std::string>()) {
        candidate_variant = variant_it->second.UncheckedGet<std::string>();
      } else if (variant_it->second.IsHolding<TfToken>()) {
        candidate_variant =
            variant_it->second.UncheckedGet<TfToken>().GetString();
      }
      if (!candidate_variant.empty()) {
        target_variant = candidate_variant;
      }
    }
    if (!target_variant.empty() && current_variant_ != target_variant) {
      TF_DEBUG(HDMITSUBA_LIFECYCLE)
          .Msg(
              "Mitsuba variant changed from '%s' to '%s'. Recreating "
              "SceneManager.\n",
              current_variant_.c_str(), target_variant.c_str());
      auto new_scene = std::unique_ptr<SceneManager>(
          SceneManager::CreateSceneManager(target_variant));
      if (scene_impl_) {
        new_scene->TransferSpecsFrom(*scene_impl_);
      }
      scene_impl_ = std::move(new_scene);
      render_param_->SetScene(scene_impl_.get());
      current_variant_ = target_variant;
      python_variant_synced_ = true;
    } else if (!python_variant_synced_) {
      SceneManager::SyncPythonVariant(current_variant_);
      python_variant_synced_ = true;
    }

    scene_impl_->UpdateNamespacedSettings(effective_settings);
    settings_dirty_ = false;
  }

  scene_impl_->CommitResources();
}

VtDictionary HdMitsubaRenderDelegate::GetRenderStats() const {
  VtDictionary stats;
  if (scene_impl_) {
    stats[HdPerfTokens->numCompletedSamples.GetString()] =
        scene_impl_->GetCurrentSampleCount();
    stats["totalSamples"] = scene_impl_->GetTargetSampleCount();
  }
  return stats;
}

TfTokenVector HdMitsubaRenderDelegate::GetMaterialRenderContexts() const {
  return {TfToken("mitsuba"), HdMaterialSchemaTokens->universalRenderContext,
          TfToken("glslfx")};
}

TfToken HdMitsubaRenderDelegate::GetMaterialBindingPurpose() const {
  return HdTokens->full;
}

bool HdMitsubaRenderDelegate::IsParallelSyncEnabled(
    const TfToken& primType) const {
  return primType == HdPrimTypeTokens->material ||
         primType == HdPrimTypeTokens->camera ||
         primType == HdPrimTypeTokens->renderBuffer ||
         HdPrimTypeIsLight(primType) ||
         HdRenderDelegate::IsParallelSyncEnabled(primType);
}

TfTokenVector HdMitsubaRenderDelegate::GetRenderSettingsNamespaces() const {
  return {TfToken("mitsuba")};
}

const MitsubaAovSpec* FindMitsubaAovSpec(const TfToken& name) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TfToken, MitsubaAovSpec, TfToken::HashFunctor>>
      kAovSpecs({
          {HdAovTokens->color,
           {"", 3, HdFormatFloat32Vec3, VtValue(GfVec3f(0.0f))}},
          {TfToken("raw"),
           {"", 3, HdFormatFloat32Vec3, VtValue(GfVec3f(0.0f))}},
          {HdAovTokens->depth,
           {"depth:depth", 1, HdFormatFloat32, VtValue(1.0f)}},
          {HdAovTokens->primId,
           {"primId:shape_index", 1, HdFormatInt32, VtValue(-1)}},
          {TfToken("shape_index"),
           {"shape_index:shape_index", 1, HdFormatInt32, VtValue(-1)}},
          {HdAovTokens->elementId,
           {"elementId:prim_index", 1, HdFormatInt32, VtValue(-1)}},
          {TfToken("prim_index"),
           {"prim_index:prim_index", 1, HdFormatInt32, VtValue(-1)}},
          {HdAovTokens->instanceId,
           {"instanceId:shape_index", 1, HdFormatInt32, VtValue(-1)}},
          {HdAovTokens->normal,
           {"sh_normal:sh_normal", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("sh_normal"),
           {"sh_normal:sh_normal", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("geo_normal"),
           {"geo_normal:geo_normal", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("position"),
           {"position:position", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("albedo"),
           {"albedo:albedo", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("diffuse_albedo"),
           {"diffuse_albedo:diffuse_albedo", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("specular_reflectance"),
           {"specular_reflectance:specular_reflectance", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("specular_transmittance"),
           {"specular_transmittance:specular_transmittance", 3,
            HdFormatFloat32Vec3, VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("roughness"),
           {"roughness:roughness", 1, HdFormatFloat32, VtValue(0.0f)}},
          {TfToken("dp_du"),
           {"dp_du:dp_du", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("dp_dv"),
           {"dp_dv:dp_dv", 3, HdFormatFloat32Vec3,
            VtValue(GfVec3f(0.0f, 0.0f, 0.0f))}},
          {TfToken("uv"),
           {"uv:uv", 2, HdFormatFloat32Vec2, VtValue(GfVec2f(0.0f, 0.0f))}},
      });
  auto it = kAovSpecs->find(name);
  return it != kAovSpecs->end() ? &it->second : nullptr;
}

HdAovDescriptor HdMitsubaRenderDelegate::GetDefaultAovDescriptor(
    const TfToken& name) const {
  if (const MitsubaAovSpec* spec = FindMitsubaAovSpec(name)) {
    return HdAovDescriptor(spec->format, false, spec->clear_value);
  }
  return HdAovDescriptor();
}

void HdMitsubaRenderDelegate::SetRenderSetting(const TfToken& key,
                                               const VtValue& value) {
  if (value.IsEmpty()) {
    if (host_setting_overrides_.erase(key) > 0) {
      bool restored_default = false;
      for (const HdRenderSettingDescriptor& descriptor : setting_descriptors_) {
        if (descriptor.key == key) {
          _settingsMap[key] = descriptor.defaultValue;
          restored_default = true;
          break;
        }
      }
      if (!restored_default) {
        _settingsMap.erase(key);
      }
      ++_settingsVersion;
      settings_dirty_ = true;
    }
    return;
  }

  auto it = host_setting_overrides_.find(key);
  if (it == host_setting_overrides_.end() || it->second != value) {
    host_setting_overrides_[key] = value;
    const auto prev_version = _settingsVersion;
    HdRenderDelegate::SetRenderSetting(key, value);
    if (_settingsVersion == prev_version) {
      ++_settingsVersion;
    }
    settings_dirty_ = true;
  }
}

VtValue HdMitsubaRenderDelegate::GetRenderSetting(const TfToken& key) const {
  auto override_it = host_setting_overrides_.find(key);
  if (override_it != host_setting_overrides_.end() &&
      !override_it->second.IsEmpty()) {
    return override_it->second;
  }
  auto scene_it = scene_render_settings_.find(key.GetString());
  if (scene_it != scene_render_settings_.end() && !scene_it->second.IsEmpty()) {
    return scene_it->second;
  }
  return HdRenderDelegate::GetRenderSetting(key);
}

HdRenderSettingDescriptorList
HdMitsubaRenderDelegate::GetRenderSettingDescriptors() const {
  return setting_descriptors_;
}

void HdMitsubaRenderDelegate::UpdateSceneRenderSettings(
    const SdfPath& id, const VtDictionary& settings) {
  if (active_render_settings_id_ != id || scene_render_settings_ != settings) {
    active_render_settings_id_ = id;
    scene_render_settings_ = settings;
    ++_settingsVersion;
    settings_dirty_ = true;
  }
}

void HdMitsubaRenderDelegate::ClearSceneRenderSettings(const SdfPath& id) {
  if (active_render_settings_id_ == id) {
    active_render_settings_id_ = SdfPath::EmptyPath();
    if (!scene_render_settings_.empty()) {
      scene_render_settings_.clear();
      ++_settingsVersion;
      settings_dirty_ = true;
    }
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
