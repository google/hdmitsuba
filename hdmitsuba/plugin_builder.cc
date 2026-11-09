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

#include "hdmitsuba/plugin_builder.h"

#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/vector.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/dataSource.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hdmitsuba/utils.h"
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

std::string ExtractStringOrToken(const VtValue& value) {
  if (value.IsHolding<std::string>()) {
    return value.UncheckedGet<std::string>();
  }
  if (value.IsHolding<TfToken>()) {
    return value.UncheckedGet<TfToken>().GetString();
  }
  return ResolvePathFromValue(value);
}

void InsertPropertyPath(PluginSpecNode* root, std::string_view relative_path,
                        const VtValue& value) {
  if (relative_path.empty() || value.IsEmpty()) {
    return;
  }
  std::vector<std::string_view> parts = absl::StrSplit(relative_path, ':');
  for (std::string_view part : parts) {
    if (part.empty()) {
      return;
    }
  }
  std::string_view leaf = parts.back();
  std::string special_string;
  if (leaf == "type" || leaf == "id") {
    special_string = ExtractStringOrToken(value);
    if (special_string.empty()) {
      return;
    }
  }
  PluginSpecNode* current = root;
  for (size_t index = 0; index + 1 < parts.size(); ++index) {
    std::string child_key(parts[index]);
    current->params.erase(child_key);
    current = &current->children[child_key];
  }
  if (leaf == "type") {
    current->plugin_type = std::move(special_string);
  } else if (leaf == "id") {
    current->id = std::move(special_string);
  } else {
    std::string leaf_key(leaf);
    if ((current->children.count(leaf_key) == 0)) {
      current->params[std::move(leaf_key)] = value;
    }
  }
}

bool TrySetSpecialChildProperty(mitsuba::Properties* props,
                                std::string_view child_name,
                                const PluginSpecNode& child) {
  if (child.plugin_type == "rgb") {
    auto value_it = child.params.find("value");
    if (value_it == child.params.end()) {
      TF_RUNTIME_ERROR(
          "Missing 'value' attribute for Mitsuba 'rgb' entry '%s'.",
          std::string(child_name).c_str());
      return true;
    }
    const VtValue& value = value_it->second;
    if (value.IsHolding<GfVec3f>()) {
      const GfVec3f& color = value.UncheckedGet<GfVec3f>();
      props->set(child_name,
                 mitsuba::Color<double, 3>(color[0], color[1], color[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<GfVec4f>()) {
      const GfVec4f& color = value.UncheckedGet<GfVec4f>();
      props->set(child_name,
                 mitsuba::Color<double, 3>(color[0], color[1], color[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<GfVec3d>()) {
      const GfVec3d& color = value.UncheckedGet<GfVec3d>();
      props->set(child_name,
                 mitsuba::Color<double, 3>(color[0], color[1], color[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<GfVec4d>()) {
      const GfVec4d& color = value.UncheckedGet<GfVec4d>();
      props->set(child_name,
                 mitsuba::Color<double, 3>(color[0], color[1], color[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<float>()) {
      props->set(child_name,
                 mitsuba::Color<double, 3>(value.UncheckedGet<float>()),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<double>()) {
      props->set(child_name,
                 mitsuba::Color<double, 3>(value.UncheckedGet<double>()),
                 /*raise_if_exists=*/false);
      return true;
    }
    TF_RUNTIME_ERROR("Unsupported 'value' type for Mitsuba 'rgb' entry '%s'.",
                     std::string(child_name).c_str());
    return true;
  }

  if (child.plugin_type == "vector") {
    auto value_it = child.params.find("value");
    if (value_it == child.params.end()) {
      TF_RUNTIME_ERROR(
          "Missing 'value' attribute for Mitsuba 'vector' entry '%s'.",
          std::string(child_name).c_str());
      return true;
    }
    const VtValue& value = value_it->second;
    if (value.IsHolding<GfVec3f>()) {
      const GfVec3f& vec = value.UncheckedGet<GfVec3f>();
      props->set(child_name, mitsuba::Vector<double, 3>(vec[0], vec[1], vec[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    if (value.IsHolding<GfVec3d>()) {
      const GfVec3d& vec = value.UncheckedGet<GfVec3d>();
      props->set(child_name, mitsuba::Vector<double, 3>(vec[0], vec[1], vec[2]),
                 /*raise_if_exists=*/false);
      return true;
    }
    TF_RUNTIME_ERROR(
        "Unsupported 'value' type for Mitsuba 'vector' entry '%s'.",
        std::string(child_name).c_str());
    return true;
  }

  if (child.plugin_type == "spectrum") {
    if (auto value_it = child.params.find("value");
        value_it != child.params.end()) {
      const VtValue& value = value_it->second;
      if (value.IsHolding<float>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(value.UncheckedGet<float>()),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<double>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(value.UncheckedGet<double>()),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<int>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(
                       static_cast<double>(value.UncheckedGet<int>())),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<int64_t>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(
                       static_cast<double>(value.UncheckedGet<int64_t>())),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<uint32_t>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(
                       static_cast<double>(value.UncheckedGet<uint32_t>())),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<uint64_t>()) {
        props->set(child_name,
                   mitsuba::Properties::Spectrum(
                       static_cast<double>(value.UncheckedGet<uint64_t>())),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<VtArray<GfVec2f>>()) {
        const VtArray<GfVec2f>& pairs =
            value.UncheckedGet<VtArray<GfVec2f>>();
        std::vector<double> wavelengths;
        std::vector<double> values;
        wavelengths.reserve(pairs.size());
        values.reserve(pairs.size());
        for (const GfVec2f& pair : pairs) {
          wavelengths.push_back(pair[0]);
          values.push_back(pair[1]);
        }
        props->set(child_name,
                   mitsuba::Properties::Spectrum(std::move(wavelengths),
                                                 std::move(values)),
                   /*raise_if_exists=*/false);
        return true;
      }
      if (value.IsHolding<VtArray<GfVec2d>>()) {
        const VtArray<GfVec2d>& pairs =
            value.UncheckedGet<VtArray<GfVec2d>>();
        std::vector<double> wavelengths;
        std::vector<double> values;
        wavelengths.reserve(pairs.size());
        values.reserve(pairs.size());
        for (const GfVec2d& pair : pairs) {
          wavelengths.push_back(pair[0]);
          values.push_back(pair[1]);
        }
        props->set(child_name,
                   mitsuba::Properties::Spectrum(std::move(wavelengths),
                                                 std::move(values)),
                   /*raise_if_exists=*/false);
        return true;
      }
      std::string spectrum_str = ExtractStringOrToken(value);
      if (!spectrum_str.empty()) {
        props->set(
            child_name,
            mitsuba::Properties::Spectrum(std::string_view(spectrum_str)),
            /*raise_if_exists=*/false);
        return true;
      }
    }
    if (auto filename_it = child.params.find("filename");
        filename_it != child.params.end()) {
      std::string filename_str = ExtractStringOrToken(filename_it->second);
      if (!filename_str.empty()) {
        mitsuba::fs::path resolved_path(filename_str);
        if (!mitsuba::fs::exists(resolved_path)) {
          if (mitsuba::FileResolver* resolver = mitsuba::file_resolver()) {
            resolved_path = resolver->resolve(resolved_path);
          }
        }
        props->set(child_name, mitsuba::Properties::Spectrum(resolved_path),
                   /*raise_if_exists=*/false);
        return true;
      }
    }
    TF_RUNTIME_ERROR(
        "Missing or invalid 'value'/'filename' attribute for Mitsuba "
        "'spectrum' entry '%s'.",
        std::string(child_name).c_str());
    return true;
  }

  return false;
}

std::vector<mitsuba::ref<mitsuba::Object>> InstantiateAndExpand(
    const PluginSpecNode& spec, std::string_view variant,
    mitsuba::ObjectType expected_type, std::string_view fallback_id) {
  mitsuba::Properties props = BuildPluginProperties(spec, variant);
  if (props.id().empty() && !fallback_id.empty()) {
    props.set_id(fallback_id);
  }
  mitsuba::ref<mitsuba::Object> object =
      mitsuba::PluginManager::instance()->create_object(props, variant,
                                                        expected_type);
  if (auto unqueried = props.unqueried(); !unqueried.empty()) {
    TF_WARN("Mitsuba plugin '%s' ignores unsupported properties: %s",
            spec.plugin_type.c_str(), absl::StrJoin(unqueried, ", ").c_str());
  }
  std::vector<mitsuba::ref<mitsuba::Object>> expanded = object->expand();
  if (expanded.empty()) {
    expanded.push_back(object);
  }
  return expanded;
}

}  // namespace

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const VtValue& value,
                                 bool invert_float, bool raise_if_exists) {
  // Ignore USD's colorSpace metadata properties (e.g.
  // "colorSpace:diffuseColor"). Mitsuba does not support per-property color
  // spaces and passes unused properties down into plugins where they trigger
  // unreferenced property errors.
  if (absl::StartsWith(name, "colorSpace:")) {
    return;
  }

  if (value.IsHolding<GfVec3f>()) {
    const GfVec3f& color = value.UncheckedGet<GfVec3f>();
    props.set(name, mitsuba::Color<float, 3>(color[0], color[1], color[2]),
              raise_if_exists);
  } else if (value.IsHolding<GfVec4f>()) {
    // Mitsuba never uses alpha channels, so we ignore it.
    const GfVec4f& color = value.UncheckedGet<GfVec4f>();
    props.set(name, mitsuba::Color<float, 3>(color[0], color[1], color[2]),
              raise_if_exists);
  } else if (value.IsHolding<GfVec3d>()) {
    const GfVec3d& color = value.UncheckedGet<GfVec3d>();
    props.set(name, mitsuba::Color<double, 3>(color[0], color[1], color[2]),
              raise_if_exists);
  } else if (value.IsHolding<GfVec4d>()) {
    const GfVec4d& color = value.UncheckedGet<GfVec4d>();
    props.set(name, mitsuba::Color<double, 3>(color[0], color[1], color[2]),
              raise_if_exists);
  } else if (value.IsHolding<float>()) {
    float scalar = value.UncheckedGet<float>();
    props.set(name, invert_float ? 1.0f - scalar : scalar, raise_if_exists);
  } else if (value.IsHolding<double>()) {
    double scalar = value.UncheckedGet<double>();
    props.set(name, invert_float ? 1.0 - scalar : scalar, raise_if_exists);
  } else if (value.IsHolding<bool>()) {
    props.set(name, value.UncheckedGet<bool>(), raise_if_exists);
  } else if (value.IsHolding<int>()) {
    props.set(name, value.UncheckedGet<int>(), raise_if_exists);
  } else if (value.IsHolding<int64_t>()) {
    props.set(name, value.UncheckedGet<int64_t>(), raise_if_exists);
  } else if (value.IsHolding<uint32_t>()) {
    props.set(name, static_cast<int64_t>(value.UncheckedGet<uint32_t>()),
              raise_if_exists);
  } else if (value.IsHolding<uint64_t>()) {
    props.set(name, static_cast<int64_t>(value.UncheckedGet<uint64_t>()),
              raise_if_exists);
  } else if (auto transform3 = ExtractTransform3f(value)) {
    props.set(name, *transform3, raise_if_exists);
  } else if (auto transform4 = ExtractTransform4f(value)) {
    props.set(name, *transform4, raise_if_exists);
  } else if (value.IsHolding<VtFloatArray>()) {
    const VtFloatArray& arr = value.UncheckedGet<VtFloatArray>();
    if (!arr.empty()) {
      props.set(name, absl::StrJoin(arr, ","), raise_if_exists);
    }
  } else if (value.IsHolding<VtDoubleArray>()) {
    const VtDoubleArray& arr = value.UncheckedGet<VtDoubleArray>();
    if (!arr.empty()) {
      props.set(name, absl::StrJoin(arr, ","), raise_if_exists);
    }
  } else if (value.IsHolding<std::string>() || value.IsHolding<TfToken>() ||
             value.IsHolding<SdfAssetPath>()) {
    props.set(name, ExtractStringOrToken(value), raise_if_exists);
  }
}

PluginSpecNode ParsePluginSpec(const VtDictionary& settings,
                               std::string_view prefix,
                               std::string_view default_type) {
  PluginSpecNode root;
  for (const auto& [key, value] : settings) {
    if (absl::StartsWith(key, prefix)) {
      InsertPropertyPath(&root, std::string_view(key).substr(prefix.size()),
                         value);
    }
  }
  if (root.plugin_type.empty() && !default_type.empty()) {
    root.plugin_type = std::string(default_type);
  }
  return root;
}

PluginSpecNode ParsePluginSpec(const HdContainerDataSourceHandle& container,
                               std::string_view prefix,
                               std::string_view default_type) {
  PluginSpecNode root;
  if (container) {
    for (const TfToken& token : container->GetNames()) {
      std::string_view key = token.GetString();
      if (!absl::StartsWith(key, prefix)) {
        continue;
      }
      std::string_view rel_path = key.substr(prefix.size());
      HdDataSourceBaseHandle child_ds = container->Get(token);
      if (auto sampled = HdSampledDataSource::Cast(child_ds)) {
        InsertPropertyPath(&root, rel_path, sampled->GetValue(0.0f));
      } else if (auto typed_container = HdContainerDataSource::Cast(child_ds)) {
        auto type_ds =
            HdSampledDataSource::Cast(typed_container->Get(TfToken("type")));
        auto val_ds =
            HdSampledDataSource::Cast(typed_container->Get(TfToken("value")));
        if (type_ds && val_ds) {
          InsertPropertyPath(&root, absl::StrCat(rel_path, ":type"),
                             type_ds->GetValue(0.0f));
          InsertPropertyPath(&root, absl::StrCat(rel_path, ":value"),
                             val_ds->GetValue(0.0f));
        }
      }
    }
  }
  if (root.plugin_type.empty() && !default_type.empty()) {
    root.plugin_type = std::string(default_type);
  }
  return root;
}

void MergePluginProperties(mitsuba::Properties* props,
                           const PluginSpecNode& spec,
                           std::string_view variant) {
  if (!spec.plugin_type.empty()) {
    props->set_plugin_name(spec.plugin_type);
  }
  if (!spec.id.empty()) {
    props->set_id(spec.id);
  }
  for (const auto& [param_name, param_value] : spec.params) {
    if ((spec.children.count(param_name) > 0)) {
      continue;
    }
    SetMitsubaPropertyFromValue(*props, param_name, param_value,
                                /*invert_float=*/false,
                                /*raise_if_exists=*/false);
  }
  for (const auto& [child_name, child_spec] : spec.children) {
    if (child_spec.Empty()) {
      continue;
    }
    if (TrySetSpecialChildProperty(props, child_name, child_spec)) {
      continue;
    }
    if (child_spec.plugin_type.empty()) {
      TF_RUNTIME_ERROR(
          "Missing 'type' attribute for nested Mitsuba plugin '%s'.",
          child_name.c_str());
      continue;
    }
    std::vector<mitsuba::ref<mitsuba::Object>> expanded = InstantiateAndExpand(
        child_spec, variant, mitsuba::ObjectType::Unknown, child_name);
    if (expanded.size() == 1) {
      props->set(child_name, expanded[0], /*raise_if_exists=*/false);
    } else if (expanded.size() > 1) {
      props->remove_property(child_name);
      for (size_t index = 0; index < expanded.size(); ++index) {
        props->set(absl::StrCat(child_name, "_", index), expanded[index],
                   /*raise_if_exists=*/false);
      }
    }
  }
}

mitsuba::Properties BuildPluginProperties(const PluginSpecNode& spec,
                                          std::string_view variant) {
  mitsuba::Properties props(spec.plugin_type);
  MergePluginProperties(&props, spec, variant);
  return props;
}

mitsuba::ref<mitsuba::Object> BuildPluginObject(
    const PluginSpecNode& spec, std::string_view variant,
    mitsuba::ObjectType expected_type) {
  mitsuba::Properties props = BuildPluginProperties(spec, variant);
  mitsuba::ref<mitsuba::Object> obj =
      CreateExpandedObject(props, variant, expected_type);
  if (auto unqueried = props.unqueried(); !unqueried.empty()) {
    TF_WARN("Mitsuba plugin '%s' ignores unsupported properties: %s",
            spec.plugin_type.c_str(), absl::StrJoin(unqueried, ", ").c_str());
  }
  return obj;
}

PXR_NAMESPACE_CLOSE_SCOPE
