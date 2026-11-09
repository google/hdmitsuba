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

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/dataSource.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

PXR_NAMESPACE_OPEN_SCOPE

// Recursive tree representation of colon-delimited `mitsuba:<domain>:*`
// plugin attributes (e.g. `mitsuba:integrator:*`, `mitsuba:sampler:*`,
// `mitsuba:sensor:*`, `mitsuba:emitter:*`, `mitsuba:shape:*`).
struct PluginSpecNode {
  std::string plugin_type;
  std::string id;
  std::map<std::string, VtValue> params;
  std::map<std::string, PluginSpecNode> children;

  bool Empty() const {
    return plugin_type.empty() && id.empty() && params.empty() &&
           children.empty();
  }

  bool HasProperty(const std::string& key) const {
    return params.count(key) > 0 || children.count(key) > 0;
  }

  PluginSpecNode() = default;
  explicit PluginSpecNode(std::string type)
      : plugin_type(std::move(type)) {}

  bool operator==(const PluginSpecNode& other) const {
    return plugin_type == other.plugin_type && id == other.id &&
           params == other.params && children == other.children;
  }

  bool operator!=(const PluginSpecNode& other) const {
    return !(*this == other);
  }
};

// Sets a scalar, vector/color, transform, string, token, or asset path
// property on `props` from a USD `VtValue`.
void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const VtValue& value,
                                 bool invert_float = false,
                                 bool raise_if_exists = true);

// Parses all entries starting with `prefix` (e.g. `"mitsuba:integrator:"`) from
// a `VtDictionary` into a hierarchical `PluginSpecNode`.
PluginSpecNode ParsePluginSpec(const VtDictionary& settings,
                               std::string_view prefix,
                               std::string_view default_type = "");

// Parses all entries starting with `prefix` (e.g. `"mitsuba:shape:"`) from a
// Hydra 2.0 `HdContainerDataSourceHandle` (flat colon-delimited keys and
// role-tagged `{"type", "value"}` value containers) into a `PluginSpecNode`.
PluginSpecNode ParsePluginSpec(const HdContainerDataSourceHandle& container,
                               std::string_view prefix,
                               std::string_view default_type = "");

// Returns whether `new_spec` preserves the plugin hierarchy of `old_spec`
// (`plugin_type`, `id`, nested child plugins, and no removed parameter
// overrides) so leaf parameter diffs can be attempted in place.
bool CanUpdatePluginStructureInPlace(const PluginSpecNode& old_spec,
                                     const PluginSpecNode& new_spec);

// Merges scalar parameters and instantiated child plugins from `spec` onto an
// existing `mitsuba::Properties` object, overriding existing keys when present.
void MergePluginProperties(mitsuba::Properties* props,
                           const PluginSpecNode& spec,
                           std::string_view variant);

// Builds a `mitsuba::Properties` object from `spec`, recursively instantiating
// any nested child plugins under `variant`.
mitsuba::Properties BuildPluginProperties(const PluginSpecNode& spec,
                                          std::string_view variant);

// Instantiates and expands a `mitsuba::Object` from `spec` under `variant`.
mitsuba::ref<mitsuba::Object> BuildPluginObject(
    const PluginSpecNode& spec, std::string_view variant,
    mitsuba::ObjectType expected_type = mitsuba::ObjectType::Unknown);

template <typename T>
mitsuba::ref<T> BuildPluginObject(const PluginSpecNode& spec) {
  mitsuba::ref<mitsuba::Object> obj =
      BuildPluginObject(spec, T::Variant, T::Type);
  T* typed = dynamic_cast<T*>(obj.get());
  if (typed == nullptr) {
    throw std::runtime_error(
        "Mitsuba plugin '" + spec.plugin_type +
        "' has an incompatible type for the requested target.");
  }
  return mitsuba::ref<T>(typed);
}

PXR_NAMESPACE_CLOSE_SCOPE
