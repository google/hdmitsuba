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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <absl/log/check.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hf/pluginDesc.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/timeCode.h>

#include "engine.h"
#include "nanobind/usd.h"

namespace nb = nanobind;

using Array = nb::ndarray<nb::numpy, nb::ndim<3>>;

namespace {

nb::dlpack::dtype GetDtype(pxr::HdFormat format) {
  switch (format) {
    case pxr::HdFormat::HdFormatUNorm8:
      return nb::dtype<uint8_t>();
    case pxr::HdFormat::HdFormatSNorm8:
      return nb::dtype<int8_t>();
    case pxr::HdFormat::HdFormatFloat16:
      return nb::dtype<float>();
    case pxr::HdFormat::HdFormatFloat32:
      return nb::dtype<float>();
    case pxr::HdFormat::HdFormatInt32:
      return nb::dtype<int32_t>();
    default:
      throw std::runtime_error("Unsupported format: " +
                               std::to_string(static_cast<int>(format)));
  }
}

// Creates a nb::ndarray from a RenderEngine::OutputBuffer.
// The ownership of the data is transferred to the array.
// The array is in the format (height, width, channels).
// This function flips the input buffer to match Numpy's conventions.
Array CreateArray(hdmitsuba::RenderEngine::OutputBuffer& buffer) {
  uint8_t* data = buffer.data.release();
  const size_t channels = pxr::HdGetComponentCount(buffer.format);
  pxr::HdFormat component_format = pxr::HdGetComponentFormat(buffer.format);
  const size_t component_size = pxr::HdDataSizeOfFormat(component_format);
  nb::dlpack::dtype dtype = GetDtype(component_format);
  nb::capsule owner(data, [](void* p) noexcept { delete[] (uint8_t*)p; });
  const int y_stride = buffer.width * channels;
  return Array(data + (buffer.height - 1) * y_stride * component_size,
               {buffer.height, buffer.width, channels}, owner,
               {-y_stride, static_cast<int64_t>(channels), 1}, dtype);
}

std::vector<std::string> GetRegisteredRenderers() {
  pxr::HfPluginDescVector descs;
  pxr::HdRendererPluginRegistry::GetInstance().GetPluginDescs(&descs);
  std::vector<std::string> renderers;
  renderers.reserve(descs.size());
  for (const auto& desc : descs) {
    renderers.push_back(desc.id.GetString());
  }
  return renderers;
}

}  // namespace

namespace {

// Explicitly register the hdMitsuba hydra plugin. While this can also
// be done via the PXR_PLUGINPATH_NAME env. variable, this logic simplifies
// the user's setup for the plain Python usage.
void RegisterHdMitsubaHydraPlugin(nb::module_& m) {
  if (!nb::hasattr(m, "__file__")) {
    return;
  }

  // Safe cast to string first, then construct the path
  std::string module_str = nb::cast<std::string>(m.attr("__file__"));
  std::filesystem::path library_dir =
      std::filesystem::path(module_str).parent_path();

  // 1. Check build layout (plugInfo.json is in library_dir/hdmitsuba/)
  if (std::filesystem::exists(library_dir / "hdmitsuba" / "plugInfo.json")) {
    pxr::PlugRegistry::GetInstance().RegisterPlugins((library_dir / "hdmitsuba").string());
    return;
  }
  // 2. Check installed layout (prefix/lib/python/usd_render.so -> prefix/plugin/usd)
  std::filesystem::path plugin_path =
      library_dir / ".." / ".." / "plugin" / "usd";
  if (std::filesystem::exists(plugin_path / "plugInfo.json")) {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(plugin_path.string());
  }
}

}  // namespace

NB_MODULE(usd_render, m) {
  RegisterHdMitsubaHydraPlugin(m);
  m.def("get_registered_renderers", &GetRegisteredRenderers,
        "Returns a list of registered Hydra renderer plugin IDs.");
  m.doc() = "Python bindings for the RenderEngine.";

  nb::module_::import_("pxr.Usd");

  nb::class_<hdmitsuba::RenderEngine>(m, "RenderEngine")
      .def(nb::init<pxr::UsdStagePtr>(), nb::arg("stage"),
           "Constructs the engine with a Usd.Stage object.")
      .def(
          "configure",
          [](hdmitsuba::RenderEngine& engine,
             const std::string& hydra_delegate_id,
             std::optional<std::string> render_settings_path,
             std::optional<int> width, std::optional<std::string> camera_path,
             std::optional<int> refine_level_fallback,
             std::optional<std::unordered_map<std::string, pxr::VtValue>>
                 overrides) {
            std::optional<pxr::SdfPath> camera_sdf_path;
            if (camera_path.has_value()) {
              camera_sdf_path = pxr::SdfPath(camera_path.value());
            }
            std::optional<pxr::SdfPath> render_settings_sdf_path;
            if (render_settings_path.has_value()) {
              render_settings_sdf_path =
                  pxr::SdfPath(render_settings_path.value());
            }
            pxr::HdRenderSettingsMap overrides_map;
            if (overrides.has_value()) {
              for (const auto& [key, value] : *overrides) {
                overrides_map[pxr::TfToken(key)] = value;
              }
            }
            engine.Configure(pxr::TfToken(hydra_delegate_id),
                             render_settings_sdf_path, width, camera_sdf_path,
                             refine_level_fallback, overrides_map);
          },
          nb::arg("hydra_delegate_id"),
          nb::arg("render_settings_path") = nb::none(),
          nb::arg("width") = nb::none(), nb::arg("camera_path") = nb::none(),
          nb::arg("refine_level_fallback") = nb::none(),
          nb::arg("overrides") = nb::none(),
          "Configures the Hydra rendering pipeline.")
      .def(
          "render",
          [](hdmitsuba::RenderEngine& engine,
             std::optional<std::variant<pxr::UsdTimeCode, int>> time_code)
              -> std::unordered_map<std::string, Array> {
            pxr::UsdTimeCode usd_time_code;
            if (time_code.has_value()) {
              if (std::holds_alternative<pxr::UsdTimeCode>(*time_code)) {
                usd_time_code = std::get<pxr::UsdTimeCode>(*time_code);
              } else {
                usd_time_code = pxr::UsdTimeCode(std::get<int>(*time_code));
              }
            } else {
              usd_time_code = pxr::UsdTimeCode::Default();
            }
            auto result = engine.Render(usd_time_code);
            std::unordered_map<std::string, Array> result_map;
            for (auto& [key, buffer] : result) {
              result_map[key.GetText()] = CreateArray(buffer);
            }
            return result_map;
          },
          nb::arg("time_code") = nb::none(),
          "Renders a single frame and returns a dictionary of AOV buffers.");
}
