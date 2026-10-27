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

#include "hdmitsuba/utils.h"

#include <string>
#include <string_view>
#include <vector>

#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <pxr/base/vt/value.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>

PXR_NAMESPACE_OPEN_SCOPE

std::string ResolvePathFromValue(const VtValue& value) {
  if (value.IsHolding<SdfAssetPath>()) {
    const auto& asset_path = value.Get<SdfAssetPath>();
    const std::string& resolved = asset_path.GetResolvedPath();
    return resolved.empty() ? asset_path.GetAssetPath() : resolved;
  }
  return {};
}

mitsuba::ref<mitsuba::Object> CreateExpandedObject(
    const mitsuba::Properties& props, std::string_view variant,
    mitsuba::ObjectType type) {
  mitsuba::ref<mitsuba::Object> obj =
      mitsuba::PluginManager::instance()->create_object(props, variant, type);
  std::vector<mitsuba::ref<mitsuba::Object>> expanded = obj->expand();
  return expanded.empty() ? obj : expanded[0];
}

PXR_NAMESPACE_CLOSE_SCOPE
