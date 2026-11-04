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

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/pyLock.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/repr.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hdx/renderSetupTask.h>
#include <pxr/imaging/hdx/renderTask.h>
#include <pxr/imaging/hdx/task.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolverScopedCache.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdRender/product.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdRender/spec.h>
#include <pxr/usd/usdRender/var.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>

#include "absl/strings/str_cat.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace hdmitsuba {

TF_DEFINE_PRIVATE_TOKENS(HdRenderEngineDataTypeTokens,
                         (color3f)(color4f)(float3)(float4));
TF_DEFINE_PRIVATE_TOKENS(HdRenderEngineTokens, (renderBufferDescriptor));

class EngineSceneDelegate final : public HdSceneDelegate {
 public:
  EngineSceneDelegate(HdRenderIndex* parent_index, SdfPath const& delegate_id)
      : HdSceneDelegate(parent_index, delegate_id) {}

  template <typename T>
  void SetParameter(SdfPath const& id, TfToken const& key, T const& value) {
    value_cache_[id][key] = value;
  }

  template <typename T>
  T GetParameter(SdfPath const& id, TfToken const& key) {
    VtValue param = Get(id, key);
    if (!param.IsHolding<T>()) {
      TF_FATAL_ERROR(
          "Incorrect parameter type for key %s in %s. Expected %s, holding %s",
          key.GetText(), id.GetText(), typeid(T).name(),
          param.GetTypeName().c_str());
    }
    return param.Get<T>();
  }

  VtValue Get(SdfPath const& id, TfToken const& key) override {
    auto id_it = value_cache_.find(id);
    if (id_it == value_cache_.end()) {
      TF_FATAL_ERROR("Required id missing: %s", id.GetText());
    }
    auto key_it = id_it->second.find(key);
    if (key_it == id_it->second.end()) {
      TF_FATAL_ERROR("Required key missing: %s in %s", key.GetText(),
                     id.GetText());
    }
    return key_it->second;
  }

  HdRenderBufferDescriptor GetRenderBufferDescriptor(
      SdfPath const& id) override {
    return GetParameter<HdRenderBufferDescriptor>(
        id, HdRenderEngineTokens->renderBufferDescriptor);
  }

 private:
  using ValueMap = absl::flat_hash_map<TfToken, VtValue, TfToken::HashFunctor>;
  absl::flat_hash_map<SdfPath, ValueMap, SdfPath::Hash> value_cache_;
};

namespace {

HdRenderSettingsMap SettingsMapFromVtDict(const VtDictionary& dict) {
  HdRenderSettingsMap settings;
  for (const auto& kv : dict) {
    settings[TfToken{kv.first}] = kv.second;
  }
  return settings;
}

std::vector<RenderEngine::AovInfo> GetAovInfos(
    HdRenderDelegate* delegate, std::vector<UsdRenderSpec::RenderVar> vars) {
  std::vector<RenderEngine::AovInfo> ret;
  for (auto& var : vars) {
    std::string source_name = var.sourceName;
    if (source_name.empty()) {
      source_name = var.renderVarPath.GetName();
      TF_WARN(
          "Invalid AOV: Missing source name for %s, attempting to use Var name "
          "instead.",
          var.renderVarPath.GetText());
    }
    auto descriptor = delegate->GetDefaultAovDescriptor(TfToken{source_name});
    if (descriptor.format == HdFormat::HdFormatInvalid) {
      TF_WARN(
          "Render delegate does not support given AOV source. Ignoring aov %s",
          var.sourceName.c_str());
      continue;
    }
    std::string name = TfGetBaseName(var.renderVarPath.GetString());
    if (name.empty()) {
      throw std::invalid_argument("Invalid AOV:RenderVar Basename empty.");
    }
    // Honor the RenderVar's request for 3 vs. 4 channels (e.g., "color3f" vs.
    // "color4f") when the delegate's default is a float RGB(A) format.
    if (descriptor.format == HdFormatFloat32Vec3 ||
        descriptor.format == HdFormatFloat32Vec4) {
      const auto& tokens = HdRenderEngineDataTypeTokens;
      if (var.dataType == tokens->color3f || var.dataType == tokens->float3) {
        descriptor.format = HdFormatFloat32Vec3;
        descriptor.clearValue = VtValue(GfVec3f(0.0f));
      } else if (var.dataType == tokens->color4f ||
                 var.dataType == tokens->float4) {
        descriptor.format = HdFormatFloat32Vec4;
        descriptor.clearValue = VtValue(GfVec4f(0.0f));
      }
    }
    descriptor.aovSettings = SettingsMapFromVtDict(var.namespacedSettings);
    ret.push_back({std::move(var), std::move(name), std::move(descriptor)});
  }
  return ret;
}

// Collects active UsdRenderVars directly instead of calling
// UsdRenderComputeSpec(), which skips any RenderProduct without an authored
// camera relationship (e.g. when `camera_path` is passed to Configure()) and
// does not filter out deactivated prims.
std::vector<UsdRenderSpec::RenderVar> ReadRenderVars(
    const UsdRenderSettings& settings) {
  UsdStageWeakPtr stage = settings.GetPrim().GetStage();
  SdfPathVector product_paths;
  settings.GetProductsRel().GetForwardedTargets(&product_paths);
  std::vector<UsdRenderSpec::RenderVar> vars;
  for (const SdfPath& product_path : product_paths) {
    UsdRenderProduct product(stage->GetPrimAtPath(product_path));
    if (!product || !product.GetPrim().IsActive()) {
      continue;
    }
    SdfPathVector var_paths;
    product.GetOrderedVarsRel().GetForwardedTargets(&var_paths);
    for (const SdfPath& var_path : var_paths) {
      if (std::any_of(vars.begin(), vars.end(),
                      [&var_path](const UsdRenderSpec::RenderVar& existing) {
                        return existing.renderVarPath == var_path;
                      })) {
        continue;
      }
      UsdPrim prim = stage->GetPrimAtPath(var_path);
      if (!prim || !prim.IsActive() || !prim.IsA<UsdRenderVar>()) {
        continue;
      }
      UsdRenderVar var(prim);
      UsdRenderSpec::RenderVar spec_var;
      spec_var.renderVarPath = var_path;
      var.GetDataTypeAttr().Get(&spec_var.dataType);
      var.GetSourceNameAttr().Get(&spec_var.sourceName);
      var.GetSourceTypeAttr().Get(&spec_var.sourceType);
      spec_var.namespacedSettings =
          UsdRenderComputeNamespacedSettings(prim, {});
      vars.push_back(std::move(spec_var));
    }
  }
  return vars;
}

size_t GetImageHeight(const UsdStagePtr& stage, const SdfPath& camera_path,
                      size_t width, UsdTimeCode time_code) {
  UsdGeomCamera usd_camera = UsdGeomCamera::Get(stage, camera_path);
  GfCamera gf_camera = usd_camera.GetCamera(time_code);
  float aspect_ratio = gf_camera.GetAspectRatio();
  if (aspect_ratio <= 0.0f) {
    throw std::runtime_error(
        absl::StrCat("Invalid aspect ratio: ", aspect_ratio,
                     " for camera: ", camera_path.GetText()));
  }
  size_t image_height =
      static_cast<size_t>(std::round(static_cast<float>(width) / aspect_ratio));
  return std::max<size_t>(image_height, 1u);
}

bool AovInfosChanged(const std::vector<RenderEngine::AovInfo>& old_aovs,
                     const std::vector<RenderEngine::AovInfo>& new_aovs) {
  if (old_aovs.size() != new_aovs.size()) {
    return true;
  }
  for (size_t i = 0; i < old_aovs.size(); ++i) {
    if (old_aovs[i].channel_name != new_aovs[i].channel_name ||
        old_aovs[i].render_var.sourceName !=
            new_aovs[i].render_var.sourceName ||
        old_aovs[i].render_var.dataType != new_aovs[i].render_var.dataType ||
        old_aovs[i].render_var.namespacedSettings !=
            new_aovs[i].render_var.namespacedSettings) {
      return true;
    }
  }
  return false;
}

CameraUtilFraming ComputeFraming(const GfRange2f& ndc,
                                 const GfVec2i& resolution) {
  if (ndc.IsEmpty()) {
    return {};
  }
  GfRange2f display_window(GfVec2f(0.0f, 0.0f),
                           GfVec2f(resolution[0], resolution[1]));
  int min_x = std::round(ndc.GetMin()[0] * resolution[0]);
  int min_y = std::round((1.0f - ndc.GetMax()[1]) * resolution[1]);
  int max_x = std::round(ndc.GetMax()[0] * resolution[0]) - 1;
  int max_y = std::round((1.0f - ndc.GetMin()[1]) * resolution[1]) - 1;
  max_x = std::max(min_x, std::min(max_x, resolution[0] - 1));
  max_y = std::max(min_y, std::min(max_y, resolution[1] - 1));
  return CameraUtilFraming(
      display_window, GfRect2i(GfVec2i(min_x, min_y), GfVec2i(max_x, max_y)));
}

GfRange2f ResolveDataWindowNdc(const UsdRenderSettings& render_settings,
                               const HdRenderSettingsMap& overrides) {
  GfRange2f ndc_range(GfVec2f(0.0f, 0.0f), GfVec2f(1.0f, 1.0f));
  auto ndc_it = overrides.find(TfToken("dataWindowNDC"));
  if (ndc_it != overrides.end() && ndc_it->second.IsHolding<GfVec4f>()) {
    const GfVec4f& ndc = ndc_it->second.UncheckedGet<GfVec4f>();
    return GfRange2f(GfVec2f(ndc[0], ndc[1]), GfVec2f(ndc[2], ndc[3]));
  }
  if (render_settings &&
      render_settings.GetDataWindowNDCAttr().HasAuthoredValue()) {
    GfVec4f ndc;
    if (render_settings.GetDataWindowNDCAttr().Get(&ndc)) {
      ndc_range = GfRange2f(GfVec2f(ndc[0], ndc[1]), GfVec2f(ndc[2], ndc[3]));
    }
  }
  return ndc_range;
}

SdfPath ResolveCameraPath(const UsdStageRefPtr& stage,
                          const UsdRenderSettings& render_settings,
                          const std::optional<SdfPath>& explicit_camera,
                          const HdRenderSettingsMap& overrides) {
  SdfPath resolved;
  if (explicit_camera.has_value()) {
    resolved = explicit_camera.value();
  } else if (auto cam_it = overrides.find(HdTokens->camera);
             cam_it != overrides.end() &&
             cam_it->second.IsHolding<std::string>()) {
    resolved = SdfPath(cam_it->second.UncheckedGet<std::string>());
  } else if (render_settings) {
    SdfPathVector targets;
    render_settings.GetCameraRel().GetForwardedTargets(&targets);
    if (!targets.empty()) {
      resolved = targets[0];
    }
  }
  if (resolved.IsEmpty()) {
    for (const pxr::UsdPrim& prim : stage->Traverse()) {
      if (UsdGeomCamera(prim)) {
        resolved = prim.GetPath();
        break;
      }
    }
    if (resolved.IsEmpty()) {
      throw std::runtime_error("No camera found!");
    }
  }
  return resolved;
}

GfVec2i ResolveResolution(const UsdStageRefPtr& stage,
                          const UsdRenderSettings& render_settings,
                          const SdfPath& camera_path,
                          std::optional<int> explicit_width,
                          const HdRenderSettingsMap& overrides) {
  if (explicit_width.has_value()) {
    const int w = explicit_width.value();
    const size_t h = GetImageHeight(stage, camera_path, w, UsdTimeCode(0));
    return GfVec2i(w, h);
  }
  if (auto res_it = overrides.find(TfToken("resolution"));
      res_it != overrides.end() && res_it->second.IsHolding<GfVec2i>()) {
    return res_it->second.UncheckedGet<GfVec2i>();
  }
  GfVec2i resolved;
  if (render_settings &&
      render_settings.GetResolutionAttr().HasAuthoredValue() &&
      render_settings.GetResolutionAttr().Get(&resolved)) {
    return resolved;
  }
  const int default_width = 512;
  const size_t height =
      GetImageHeight(stage, camera_path, default_width, UsdTimeCode(0));
  return GfVec2i(default_width, height);
}

bool SupportsRenderSettingsBprim(const HdRenderDelegate* delegate) {
  if (!delegate) {
    return false;
  }
  const TfTokenVector& types = delegate->GetSupportedBprimTypes();
  return std::find(types.begin(), types.end(),
                   HdPrimTypeTokens->renderSettings) != types.end();
}

HdRenderSettingsMap MergeFallbackSceneRenderSettings(
    const UsdRenderSettings& render_settings,
    const HdRenderSettingsMap& host_overrides) {
  HdRenderSettingsMap merged;
  if (render_settings) {
    const VtDictionary namespaced =
        UsdRenderComputeNamespacedSettings(render_settings.GetPrim(), {});
    for (const auto& [key, val] : namespaced) {
      merged[TfToken(key)] = val;
    }
  }
  for (const auto& [key, val] : host_overrides) {
    merged[key] = val;
  }
  return merged;
}

std::vector<RenderEngine::AovInfo> ResolveAovInfos(
    HdRenderDelegate* render_delegate,
    const UsdRenderSettings& render_settings) {
  std::vector<UsdRenderSpec::RenderVar> aovs;
  if (render_settings) {
    aovs = ReadRenderVars(render_settings);
  }
  if (aovs.empty()) {
    aovs.push_back({.renderVarPath = SdfPath{"/Render/Vars/color"},
                    .dataType = HdRenderEngineDataTypeTokens->color3f,
                    .sourceName = "color",
                    .sourceType = TfToken{},
                    .namespacedSettings = VtDictionary{}});
  }
  std::vector<RenderEngine::AovInfo> aov_infos =
      GetAovInfos(render_delegate, std::move(aovs));
  if (aov_infos.empty()) {
    throw std::runtime_error("No aovs found.");
  }
  return aov_infos;
}

}  // namespace

RenderEngine::RenderEngine(UsdStagePtr stage)
    : stage_(stage),
      data_window_ndc_(GfVec2f(0.0f, 0.0f), GfVec2f(1.0f, 1.0f)) {
  engine_ = std::make_unique<HdEngine>();
}

// Defined here since EngineSceneDelegate is forward-declared in the header.
RenderEngine::~RenderEngine() = default;

void RenderEngine::CreateRenderBuffers() {
  render_buffer_ids_.clear();
  for (auto& aov : aov_ids_) {
    SdfPath render_buffer_id =
        SdfPath("/task_controller").AppendChild(TfToken(aov.channel_name));
    render_buffer_ids_.push_back(render_buffer_id);
    render_index_->InsertBprim(HdPrimTypeTokens->renderBuffer,
                               params_delegate_.get(), render_buffer_id);
    HdRenderBufferDescriptor desc{};
    desc.multiSampled = aov.descriptor.multiSampled;
    desc.format = aov.descriptor.format;
    desc.dimensions = GfVec3i(resolution_[0], resolution_[1], 1);
    params_delegate_->SetParameter(
        render_buffer_id, HdRenderEngineTokens->renderBufferDescriptor, desc);
    render_index_->GetChangeTracker().MarkBprimDirty(
        render_buffer_id, HdRenderBuffer::DirtyDescription);
  }
}

void RenderEngine::ClearRenderBuffers() {
  for (const auto& buffer_id : render_buffer_ids_) {
    render_index_->RemoveBprim(HdPrimTypeTokens->renderBuffer, buffer_id);
  }
  render_buffer_ids_.clear();
}

pxr::HdRenderPassAovBindingVector RenderEngine::CreateAovBindings() {
  HdRenderPassAovBindingVector aov_bindings;
  if (render_buffer_ids_.size() != aov_ids_.size()) {
    TF_FATAL_ERROR(
        "Render buffer ids and aov ids must have the same size: %zu vs %zu",
        render_buffer_ids_.size(), aov_ids_.size());
  }
  for (size_t i = 0; i < render_buffer_ids_.size(); i++) {
    const auto& aov_info = aov_ids_[i];
    auto& binding = (aov_bindings.emplace_back(), aov_bindings.back());
    binding.aovName = TfToken{aov_info.render_var.sourceName};
    binding.renderBufferId = render_buffer_ids_[i];
    binding.renderBuffer = GetRenderBuffer(i);
    binding.clearValue = aov_info.descriptor.clearValue;
    binding.aovSettings = aov_info.descriptor.aovSettings;
  }
  return aov_bindings;
}

UsdRenderSettings RenderEngine::GetUsdRenderSettings(
    const std::optional<SdfPath>& path) const {
  if (path.has_value()) {
    UsdRenderSettings settings = UsdRenderSettings::Get(stage_, path.value());
    if (!settings) {
      throw std::invalid_argument(absl::StrCat(
          "Invalid render settings prim path: ", path.value().GetText()));
    }
    return settings;
  }
  return UsdRenderSettings::GetStageRenderSettings(stage_);
}

void RenderEngine::Configure(
    const TfToken& hydra_delegate_id,
    const std::optional<SdfPath>& render_settings_prim_path,
    std::optional<int> width, const std::optional<SdfPath>& camera_path,
    std::optional<int> refine_level_fallback,
    const HdRenderSettingsMap& overrides) {
  ArResolverScopedCache resolver_cache;
  UsdRenderSettings render_settings =
      GetUsdRenderSettings(render_settings_prim_path);

  HdRenderSettingsMap delegate_overrides = overrides;
  // Unconditionally disable progressive refinement in the batch render engine
  // unless explicitly overridden by the caller.
  delegate_overrides.insert(
      {pxr::HdRenderSettingsTokens->enableInteractive, pxr::VtValue(false)});

  GfRange2f new_data_window_ndc =
      ResolveDataWindowNdc(render_settings, delegate_overrides);
  bool ndc_changed = (new_data_window_ndc != data_window_ndc_);

  bool rebuild_delegate = !render_delegate_ || !stage_scene_index_ ||
                          (hydra_delegate_id != hydra_delegate_id_);

  if (!rebuild_delegate) {
    const HdRenderSettingsMap effective_overrides =
        SupportsRenderSettingsBprim(render_delegate_.Get())
            ? delegate_overrides
            : MergeFallbackSceneRenderSettings(render_settings,
                                               delegate_overrides);
    const HdRenderSettingDescriptorList descriptors =
        render_delegate_->GetRenderSettingDescriptors();
    for (const auto& [key, _] : settings_map_) {
      if (effective_overrides.find(key) == effective_overrides.end()) {
        render_delegate_->SetRenderSetting(key, VtValue());
        if (render_delegate_->GetRenderSetting(key).IsEmpty()) {
          for (const HdRenderSettingDescriptor& desc : descriptors) {
            if (desc.key == key && !desc.defaultValue.IsEmpty()) {
              render_delegate_->SetRenderSetting(key, desc.defaultValue);
              break;
            }
          }
        }
      }
    }
    for (const auto& [key, value] : effective_overrides) {
      auto prev_it = settings_map_.find(key);
      if (prev_it == settings_map_.end() || prev_it->second != value) {
        render_delegate_->SetRenderSetting(key, value);
      }
    }
    settings_map_ = effective_overrides;
  }

  SdfPath resolved_camera_path = ResolveCameraPath(
      stage_, render_settings, camera_path, delegate_overrides);
  GfVec2i resolved_resolution = ResolveResolution(
      stage_, render_settings, resolved_camera_path, width, delegate_overrides);

  bool cache_invalid = rebuild_delegate ||
                       (resolved_resolution != resolution_) ||
                       (resolved_camera_path != camera_path_);

  if (cache_invalid) {
    // Commit validated state
    camera_path_ = resolved_camera_path;
    resolution_ = resolved_resolution;

    if (rebuild_delegate) {
      // Clean up any previous state in reverse order.
      tasks_.clear();
      params_delegate_ = nullptr;
      render_index_.reset();
      display_style_scene_index_ = nullptr;
      scene_globals_scene_index_ = nullptr;
      stage_scene_index_ = nullptr;
      render_delegate_ = nullptr;
      aov_ids_.clear();
      render_buffer_ids_.clear();

      // Create render delegate, scene index chain, and params delegate.
      renderer_plugin_ =
          HdRendererPluginRegistry::GetInstance().GetRendererPlugin(
              hydra_delegate_id);
      if (!renderer_plugin_) {
        throw std::runtime_error(absl::StrCat("Failed to get renderer plugin: ",
                                              hydra_delegate_id.GetText()));
      }
      render_delegate_ = renderer_plugin_->CreateDelegate(delegate_overrides);
      HdRenderSettingsMap effective_overrides = delegate_overrides;
      if (!SupportsRenderSettingsBprim(render_delegate_.Get())) {
        effective_overrides = MergeFallbackSceneRenderSettings(
            render_settings, delegate_overrides);
        for (const auto& [key, value] : effective_overrides) {
          if (delegate_overrides.find(key) == delegate_overrides.end()) {
            render_delegate_->SetRenderSetting(key, value);
          }
        }
      }

      render_index_.reset(
          HdRenderIndex::New(render_delegate_.Get(), HdDriverVector{}));
      if (!render_index_) {
        throw std::runtime_error("Failed to create render index.");
      }
      UsdImagingCreateSceneIndicesInfo create_info;
      create_info.stage = stage_;
      const UsdImagingSceneIndices scene_indices =
          UsdImagingCreateSceneIndices(create_info);
      stage_scene_index_ = scene_indices.stageSceneIndex;
      scene_globals_scene_index_ =
          HdsiSceneGlobalsSceneIndex::New(scene_indices.finalSceneIndex);
      display_style_scene_index_ =
          HdsiLegacyDisplayStyleOverrideSceneIndex::New(
              scene_globals_scene_index_);
      render_index_->InsertSceneIndex(display_style_scene_index_,
                                      SdfPath::AbsoluteRootPath());
      params_delegate_ = std::make_unique<EngineSceneDelegate>(
          render_index_.get(), SdfPath{"/task_controller"});
      hydra_delegate_id_ = hydra_delegate_id;
      settings_map_ = std::move(effective_overrides);
    }
  }

  configured_width_ = width;
  configured_camera_path_ = camera_path;
  render_settings_prim_path_ = render_settings_prim_path;
  scene_globals_scene_index_->SetActiveRenderSettingsPrimPath(
      render_settings ? render_settings.GetPath() : SdfPath::EmptyPath());
  scene_globals_scene_index_->SetPrimaryCameraPrimPath(resolved_camera_path);
  display_style_scene_index_->SetRefineLevelFallback(refine_level_fallback);

  std::vector<AovInfo> new_aov_ids =
      ResolveAovInfos(render_delegate_.Get(), render_settings);
  bool aovs_changed = AovInfosChanged(aov_ids_, new_aov_ids);

  if (cache_invalid || aovs_changed) {
    if (!rebuild_delegate) {
      ClearRenderBuffers();
      if (!render_task_id_.IsEmpty()) {
        render_index_->RemoveTask(render_task_id_);
      }
    }
    aov_ids_ = std::move(new_aov_ids);
    CreateRenderBuffers();

    HdRprimCollection collection(HdTokens->geometry,
                                 HdReprSelector(HdReprTokens->hull));
    collection.SetRootPath(SdfPath::AbsoluteRootPath());

    tasks_.clear();
    render_task_id_ = SdfPath{"/task_controller/render_task"};
    {
      render_index_->InsertTask<HdxRenderTask>(params_delegate_.get(),
                                               render_task_id_);
      HdxRenderTaskParams params{};
      params.aovBindings = CreateAovBindings();
      params.viewport = GfVec4d(0, 0, resolution_[0], resolution_[1]);
      params.framing = ComputeFraming(new_data_window_ndc, resolution_);
      params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
      params_delegate_->SetParameter(render_task_id_, HdTokens->collection,
                                     collection);
      params_delegate_->SetParameter(
          render_task_id_, HdTokens->renderTags,
          TfTokenVector{HdRenderTagTokens->geometry});
      tasks_.push_back(render_index_->GetTask(render_task_id_));
    }
    if (!rebuild_delegate) {
      stage_scene_index_->ApplyPendingUpdates();
    }
    SetCamera(camera_path_);
  } else if (ndc_changed) {
    auto params = params_delegate_->GetParameter<HdxRenderTaskParams>(
        render_task_id_, HdTokens->params);
    params.framing = ComputeFraming(new_data_window_ndc, resolution_);
    params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
    render_index_->GetChangeTracker().MarkTaskDirty(
        render_task_id_, HdChangeTracker::DirtyParams);
  }

  data_window_ndc_ = new_data_window_ndc;
}

void RenderEngine::SetCamera(const SdfPath& camera) {
  HdSprim* camera_prim =
      render_index_->GetSprim(HdPrimTypeTokens->camera, camera);
  if (!camera_prim) {
    throw std::invalid_argument(
        absl::StrCat("Camera not found: ", camera.GetText()));
  }

  auto params = params_delegate_->GetParameter<HdxRenderTaskParams>(
      render_task_id_, HdTokens->params);
  params.camera = camera;
  params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
  render_index_->GetChangeTracker().MarkTaskDirty(render_task_id_,
                                                  HdChangeTracker::DirtyParams);
}

void RenderEngine::UpdateAovsAndBuffers() {
  UsdRenderSettings render_settings =
      GetUsdRenderSettings(render_settings_prim_path_);
  if (scene_globals_scene_index_) {
    scene_globals_scene_index_->SetActiveRenderSettingsPrimPath(
        render_settings ? render_settings.GetPath() : SdfPath::EmptyPath());
  }

  const SdfPath resolved_camera = ResolveCameraPath(
      stage_, render_settings, configured_camera_path_, settings_map_);
  const GfVec2i resolved_resolution =
      ResolveResolution(stage_, render_settings, resolved_camera,
                        configured_width_, settings_map_);
  const GfRange2f resolved_ndc =
      ResolveDataWindowNdc(render_settings, settings_map_);
  std::vector<AovInfo> new_aov_ids =
      ResolveAovInfos(render_delegate_.Get(), render_settings);

  const bool camera_changed = (resolved_camera != camera_path_);
  const bool resolution_changed = (resolved_resolution != resolution_);
  const bool ndc_changed = (resolved_ndc != data_window_ndc_);
  const bool aovs_changed = AovInfosChanged(aov_ids_, new_aov_ids);

  if (camera_changed) {
    camera_path_ = resolved_camera;
    if (scene_globals_scene_index_) {
      scene_globals_scene_index_->SetPrimaryCameraPrimPath(camera_path_);
    }
    stage_scene_index_->ApplyPendingUpdates();
    SetCamera(camera_path_);
  }

  if (resolution_changed || aovs_changed) {
    ClearRenderBuffers();
    resolution_ = resolved_resolution;
    aov_ids_ = std::move(new_aov_ids);
    CreateRenderBuffers();

    auto params = params_delegate_->GetParameter<HdxRenderTaskParams>(
        render_task_id_, HdTokens->params);
    params.aovBindings = CreateAovBindings();
    params.viewport = GfVec4d(0, 0, resolution_[0], resolution_[1]);
    params.framing = ComputeFraming(resolved_ndc, resolution_);
    params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
    render_index_->GetChangeTracker().MarkTaskDirty(
        render_task_id_, HdChangeTracker::DirtyParams);
  } else if (ndc_changed) {
    auto params = params_delegate_->GetParameter<HdxRenderTaskParams>(
        render_task_id_, HdTokens->params);
    params.framing = ComputeFraming(resolved_ndc, resolution_);
    params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
    render_index_->GetChangeTracker().MarkTaskDirty(
        render_task_id_, HdChangeTracker::DirtyParams);
  }

  data_window_ndc_ = resolved_ndc;
}

absl::flat_hash_map<pxr::TfToken, RenderEngine::OutputBuffer,
                    TfToken::HashFunctor>
RenderEngine::Render(UsdTimeCode time_code) {
  ArResolverScopedCache resolver_cache;
  bool enable_interactive = false;
  auto it = settings_map_.find(pxr::HdRenderSettingsTokens->enableInteractive);
  if (it != settings_map_.end()) {
    enable_interactive = it->second.GetWithDefault<bool>(false);
  }
  UpdateAovsAndBuffers();
  if (scene_globals_scene_index_ && time_code.IsNumeric()) {
    scene_globals_scene_index_->SetCurrentFrame(time_code.GetValue());
  }
  stage_scene_index_->ApplyPendingUpdates();
  stage_scene_index_->SetTime(time_code);
  do {
    TF_PY_ALLOW_THREADS_IN_SCOPE();
    engine_->Execute(render_index_.get(), &tasks_);
  } while (!IsConverged() && !enable_interactive);

  for (size_t i = 0; i < render_buffer_ids_.size(); i++) {
    GetRenderBuffer(i)->Resolve();
  }

  absl::flat_hash_map<pxr::TfToken, OutputBuffer, TfToken::HashFunctor> result;
  result.reserve(aov_ids_.size());
  for (size_t i = 0; i < aov_ids_.size(); ++i) {
    const auto& aov = aov_ids_[i];
    HdRenderBuffer* buffer = GetRenderBuffer(i);
    OutputBuffer output_buffer(buffer->GetWidth(), buffer->GetHeight(),
                               buffer->GetFormat(),
                               aov.render_var.dataType.GetString());
    memcpy(output_buffer.data.get(), buffer->Map(), output_buffer.size);
    buffer->Unmap();
    result.insert({TfToken(aov.channel_name), std::move(output_buffer)});
  }

  return result;
}

bool RenderEngine::IsConverged() const {
  return std::all_of(tasks_.begin(), tasks_.end(),
                     [](const pxr::HdTaskSharedPtr& task) {
                       if (auto progressive_task =
                               std::dynamic_pointer_cast<pxr::HdxTask>(task)) {
                         return progressive_task->IsConverged();
                       }
                       return true;
                     });
}

pxr::VtDictionary RenderEngine::GetRenderStats() const {
  if (!render_delegate_) {
    return {};
  }
  return render_delegate_->GetRenderStats();
}
HdRenderBuffer* RenderEngine::GetRenderBuffer(size_t idx) const {
  if (idx >= render_buffer_ids_.size()) {
    TF_FATAL_ERROR("Render buffer index out of range: %zu vs %zu", idx,
                   render_buffer_ids_.size());
  }
  return static_cast<HdRenderBuffer*>(render_index_->GetBprim(
      HdPrimTypeTokens->renderBuffer, render_buffer_ids_[idx]));
}

}  // namespace hdmitsuba
