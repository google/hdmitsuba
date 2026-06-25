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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/tf/token.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/task.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/usd/usdRender/spec.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdEngine;
class HdRenderDelegate;
class HdRenderIndex;
class HdRendererPlugin;

PXR_NAMESPACE_CLOSE_SCOPE

namespace hdmitsuba {

class EngineSceneDelegate;

class RenderEngine {
 public:
  struct OutputBuffer {
    size_t width = 0;
    size_t height = 0;
    pxr::HdFormat format = pxr::HdFormatInvalid;
    std::string type;
    std::unique_ptr<uint8_t[]> data;
    size_t size = 0;

    OutputBuffer() = default;
    OutputBuffer(const OutputBuffer&) = delete;
    OutputBuffer& operator=(const OutputBuffer&) = delete;
    OutputBuffer(OutputBuffer&& other) = default;

    OutputBuffer(size_t width, size_t height, pxr::HdFormat format,
                 std::string type)
        : width(width), height(height), format(format), type(type) {
      size = width * height * pxr::HdDataSizeOfFormat(format);
      data.reset(new uint8_t[size]);
    }
  };
  struct AovInfo {
    pxr::UsdRenderSpec::RenderVar render_var;
    std::string channel_name;
    pxr::HdAovDescriptor descriptor;
  };

  explicit RenderEngine(pxr::UsdStagePtr stage);
  ~RenderEngine();

  void Configure(const pxr::TfToken& hydra_delegate_id,
                 const std::optional<pxr::SdfPath>& render_settings_prim_path,
                 std::optional<int> width,
                 const std::optional<pxr::SdfPath>& camera_path,
                 std::optional<int> refine_level_fallback,
                 const pxr::HdRenderSettingsMap& overrides = {});

  absl::flat_hash_map<pxr::TfToken, OutputBuffer, pxr::TfToken::HashFunctor>
  Render(pxr::UsdTimeCode time_code);

 private:
  void UpdateAovsAndBuffers();
  void CreateRenderBuffers();
  void ClearRenderBuffers();
  pxr::HdRenderPassAovBindingVector CreateAovBindings();
  pxr::HdRenderBuffer* GetRenderBuffer(size_t idx) const;
  void SetCamera(const pxr::SdfPath& camera);

  std::pair<pxr::HdRenderSettingsMap, pxr::UsdRenderSettings>
  ReadRenderSettings(const std::optional<pxr::SdfPath>& path) const;

  pxr::UsdStageRefPtr stage_;
  pxr::HdRendererPlugin* renderer_plugin_ = nullptr;
  pxr::HdPluginRenderDelegateUniqueHandle render_delegate_ = nullptr;
  std::unique_ptr<pxr::HdRenderIndex> render_index_;
  std::unique_ptr<pxr::UsdImagingDelegate> scene_delegate_;
  std::unique_ptr<EngineSceneDelegate> params_delegate_;
  std::unique_ptr<pxr::HdEngine> engine_;
  pxr::GfVec2i resolution_;
  std::vector<AovInfo> aov_ids_;
  std::vector<pxr::SdfPath> render_buffer_ids_;
  pxr::HdTaskSharedPtrVector tasks_;
  pxr::SdfPath render_task_id_;

  std::string hydra_delegate_id_;
  std::optional<pxr::SdfPath> render_settings_prim_path_;
  pxr::SdfPath camera_path_;
  int width_;
  pxr::HdRenderSettingsMap settings_map_;
};

}  // namespace hdmitsuba
