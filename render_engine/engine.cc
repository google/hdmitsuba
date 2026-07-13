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
#include <tuple>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4d.h>
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
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdRender/spec.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include "absl/strings/str_cat.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace hdmitsuba {

TF_DEFINE_PRIVATE_TOKENS(HdRenderEngineDataTypeTokens, (color4f));
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

bool IsConverged(const HdTaskSharedPtrVector& tasks) {
  return std::all_of(
      tasks.begin(), tasks.end(), [](const HdTaskSharedPtr& task) {
        if (auto progressive_task = std::dynamic_pointer_cast<HdxTask>(task)) {
          return progressive_task->IsConverged();
        }
        return true;
      });
}

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
    descriptor.aovSettings = SettingsMapFromVtDict(var.namespacedSettings);
    ret.push_back({std::move(var), std::move(name), std::move(descriptor)});
  }
  return ret;
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

std::pair<HdRenderSettingsMap, UsdRenderSettings>
RenderEngine::ReadRenderSettings(const std::optional<SdfPath>& path) const {
  HdRenderSettingsMap settings_map;
  UsdRenderSettings settings;
  if (path.has_value()) {
    settings = UsdRenderSettings::Get(stage_, path.value());
    if (!settings) {
      throw std::invalid_argument(absl::StrCat(
          "Invalid render settings prim path: ", path.value().GetText()));
    }
  } else {
    settings = UsdRenderSettings::GetStageRenderSettings(stage_);
    if (!settings) {
      return std::make_pair(settings_map, settings);
    }
  }

  // Convert attributes to Render Settings Map.
  std::vector<UsdAttribute> attributes =
      settings.GetPrim().GetAuthoredAttributes();
  for (const auto& a : attributes) {
    VtValue value;
    a.Get(&value);
    settings_map[a.GetName()] = value;
  }
  UsdRelationship camera_rel = settings.GetCameraRel();
  if (camera_rel) {
    SdfPathVector targets;
    camera_rel.GetTargets(&targets);
    if (!targets.empty()) {
      settings_map[HdTokens->camera] = targets[0].GetString();
    }
  }
  return std::make_pair(settings_map, settings);
}

void RenderEngine::Configure(
    const TfToken& hydra_delegate_id,
    const std::optional<SdfPath>& render_settings_prim_path,
    std::optional<int> width, const std::optional<SdfPath>& camera_path,
    std::optional<int> refine_level_fallback,
    const HdRenderSettingsMap& overrides) {
  // Unconditionally read settings to support robust fallbacks
  HdRenderSettingsMap settings_map;
  UsdRenderSettings render_settings;
  std::tie(settings_map, render_settings) =
      ReadRenderSettings(render_settings_prim_path);
  render_settings_prim_path_ = render_settings_prim_path;

  // Merge overrides
  for (const auto& [key, value] : overrides) {
    settings_map[key] = value;
  }

  GfRange2f new_data_window_ndc(GfVec2f(0.0f, 0.0f), GfVec2f(1.0f, 1.0f));
  TfToken ndc_token("dataWindowNDC");
  auto ndc_it = settings_map.find(ndc_token);
  if (ndc_it != settings_map.end() && ndc_it->second.IsHolding<GfVec4f>()) {
    GfVec4f ndc = ndc_it->second.Get<GfVec4f>();
    new_data_window_ndc =
        GfRange2f(GfVec2f(ndc[0], ndc[1]), GfVec2f(ndc[2], ndc[3]));
  }
  bool ndc_changed = (new_data_window_ndc != data_window_ndc_);

  // Unconditionally disable progressive refinement in the batch render engine
  settings_map[pxr::HdRenderSettingsTokens->enableInteractive] =
      pxr::VtValue(false);

  bool rebuild_delegate =
      (hydra_delegate_id != hydra_delegate_id_) ||
      (render_settings_prim_path != render_settings_prim_path_);

  if (!rebuild_delegate && render_delegate_) {
    // Helper to look up settings with a fallback default value.
    auto get_setting = [](const HdRenderSettingsMap& map, const TfToken& key,
                          const VtValue& default_val) {
      auto it = map.find(key);
      return (it != map.end()) ? it->second : default_val;
    };

    // Check if any creation-time settings changed
    HdRenderSettingDescriptorList descriptors =
        render_delegate_->GetRenderSettingDescriptors();
    for (const auto& desc : descriptors) {
      if (get_setting(settings_map, desc.key, desc.defaultValue) !=
          get_setting(settings_map_, desc.key, desc.defaultValue)) {
        rebuild_delegate = true;
        break;
      }
    }
  }

  // Resolve target camera path upfront
  SdfPath resolved_camera_path;
  if (camera_path.has_value()) {
    resolved_camera_path = camera_path.value();
  } else if (settings_map.find(HdTokens->camera) != settings_map.end()) {
    resolved_camera_path =
        SdfPath(settings_map[HdTokens->camera].Get<std::string>());
  } else {
    for (const pxr::UsdPrim& prim : stage_->Traverse()) {
      UsdGeomCamera camera(prim);
      if (camera) {
        resolved_camera_path = prim.GetPath();
        break;
      }
    }
    if (resolved_camera_path.IsEmpty()) {
      throw std::runtime_error("No camera found!");
    }
  }

  // Resolve target resolution upfront
  GfVec2i resolved_resolution;
  int resolved_width = 0;
  TfToken resolution_token("resolution");
  if (width.has_value()) {
    resolved_width = width.value();
    size_t height = GetImageHeight(stage_, resolved_camera_path, resolved_width,
                                   UsdTimeCode(0));
    resolved_resolution = GfVec2i(resolved_width, height);
  } else if (settings_map.find(resolution_token) != settings_map.end()) {
    resolved_resolution = settings_map[resolution_token].Get<GfVec2i>();
    resolved_width = resolved_resolution[0];
  } else {
    const int default_width = 512;
    size_t height = GetImageHeight(stage_, resolved_camera_path, default_width,
                                   UsdTimeCode(0));
    resolved_resolution = GfVec2i(default_width, height);
    resolved_width = default_width;
  }

  bool cache_invalid = rebuild_delegate ||
                       (resolved_resolution != resolution_) ||
                       (resolved_camera_path != camera_path_);

  if (cache_invalid) {
    // Commit validated state
    camera_path_ = resolved_camera_path;
    resolution_ = resolved_resolution;
    width_ = resolved_width;

    if (rebuild_delegate) {
      hydra_delegate_id_ = hydra_delegate_id;

      // Clean up any previous state in reverse order.
      params_delegate_ = nullptr;
      scene_delegate_ = nullptr;
      render_delegate_ = nullptr;
      aov_ids_.clear();
      render_buffer_ids_.clear();

      // Create render delegate, scene delegate, and params delegate.
      renderer_plugin_ =
          HdRendererPluginRegistry::GetInstance().GetRendererPlugin(
              hydra_delegate_id);
      if (!renderer_plugin_) {
        throw std::runtime_error(absl::StrCat("Failed to get renderer plugin: ",
                                              hydra_delegate_id.GetText()));
      }
      render_delegate_ = renderer_plugin_->CreateDelegate(settings_map);

      render_index_.reset(
          HdRenderIndex::New(render_delegate_.Get(), HdDriverVector{}));
      if (!render_index_) {
        throw std::runtime_error("Failed to create render index.");
      }
      scene_delegate_ = std::make_unique<UsdImagingDelegate>(
          render_index_.get(), SdfPath::AbsoluteRootPath());
      scene_delegate_->Populate(stage_->GetPseudoRoot());
      params_delegate_ = std::make_unique<EngineSceneDelegate>(
          render_index_.get(), SdfPath{"/task_controller"});
      settings_map_ = settings_map;
    }
  }

  if (refine_level_fallback.has_value() &&
      refine_level_fallback.value() !=
          scene_delegate_->GetRefineLevelFallback()) {
    scene_delegate_->SetRefineLevelFallback(refine_level_fallback.value());
  }

  if (cache_invalid) {
    // This update is necessary to correctly update render buffer resolutions.
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
  auto read_res = ReadRenderSettings(render_settings_prim_path_);
  UsdRenderSettings render_settings = read_res.second;
  std::vector<UsdRenderSpec::RenderVar> aovs;
  if (render_settings) {
    UsdRenderSpec spec = UsdRenderComputeSpec(render_settings, {});
    aovs = std::move(spec.renderVars);
  }
  if (aovs.empty()) {
    aovs.push_back({.renderVarPath = SdfPath{"/Render/Vars/color"},
                    .dataType = HdRenderEngineDataTypeTokens->color4f,
                    .sourceName = "color",
                    .sourceType = TfToken{},
                    .namespacedSettings = VtDictionary{}});
  }

  std::vector<AovInfo> new_aov_ids =
      GetAovInfos(render_delegate_.Get(), std::move(aovs));
  if (new_aov_ids.empty()) {
    throw std::runtime_error("No aovs found.");
  }
  if (!AovInfosChanged(aov_ids_, new_aov_ids)) {
    return;
  }

  ClearRenderBuffers();
  aov_ids_ = std::move(new_aov_ids);
  CreateRenderBuffers();

  auto params = params_delegate_->GetParameter<HdxRenderTaskParams>(
      render_task_id_, HdTokens->params);
  params.aovBindings = CreateAovBindings();
  params_delegate_->SetParameter(render_task_id_, HdTokens->params, params);
  render_index_->GetChangeTracker().MarkTaskDirty(render_task_id_,
                                                  HdChangeTracker::DirtyParams);
}

absl::flat_hash_map<pxr::TfToken, RenderEngine::OutputBuffer,
                    TfToken::HashFunctor>
RenderEngine::Render(UsdTimeCode time_code) {
  UpdateAovsAndBuffers();
  scene_delegate_->SetTime(time_code);
  do {
    TF_PY_ALLOW_THREADS_IN_SCOPE();
    engine_->Execute(&scene_delegate_->GetRenderIndex(), &tasks_);
  } while (!IsConverged(tasks_));

  for (size_t i = 0; i < render_buffer_ids_.size(); i++) {
    GetRenderBuffer(i)->Resolve();
  }

  absl::flat_hash_map<pxr::TfToken, OutputBuffer, TfToken::HashFunctor> result;
  for (const auto& aov : aov_ids_) {
    SdfPath buffer_id =
        SdfPath("/task_controller").AppendChild(TfToken(aov.channel_name));
    HdRenderBuffer* buffer = dynamic_cast<HdRenderBuffer*>(
        render_index_->GetBprim(HdPrimTypeTokens->renderBuffer, buffer_id));
    if (!buffer) {
      throw std::runtime_error(
          absl::StrCat("Render buffer not found: ", buffer_id.GetText()));
    }
    OutputBuffer output_buffer(buffer->GetWidth(), buffer->GetHeight(),
                               buffer->GetFormat(),
                               aov.render_var.dataType.GetString());
    memcpy(output_buffer.data.get(), buffer->Map(), output_buffer.size);
    buffer->Unmap();
    result.insert({TfToken(aov.channel_name), std::move(output_buffer)});
  }

  return result;
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
