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

#include "hdmitsuba/traversal.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>

#include <absl/strings/str_cat.h>
#include <mitsuba/core/object.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

using mitsuba::Object;

TraversalCallback::TraversalCallback(std::string_view prefix, Object* parent,
                                      bool recurse_objects)
    : prefix_(prefix), recurse_objects_(recurse_objects) {
  if (parent != nullptr) {
    hierarchy_.insert(parent);
  }
}

TraversalCallback::TraversalCallback(
    std::string_view prefix, Object* parent,
    const absl::flat_hash_set<void*>& parent_hierarchy,
    bool recurse_objects)
    : hierarchy_(parent_hierarchy),
      prefix_(prefix),
      recurse_objects_(recurse_objects) {
  if (parent != nullptr) {
    hierarchy_.insert(parent);
  }
}

void TraversalCallback::put_value(std::string_view name, void* value,
                                  uint32_t /*flags*/,
                                  const std::type_info& type) {
  data.insert({absl::StrCat(prefix_, name), {value, type}});
}

/// Actual implementation for Object references [To be provided by subclass]
void TraversalCallback::put_object(std::string_view name, Object* value,
                                   uint32_t /*flags*/) {
  if (!recurse_objects_ || value == nullptr ||
      hierarchy_.find(value) != hierarchy_.end()) {
    return;
  }
  TraversalCallback cb(absl::StrCat(prefix_, name, "."), value, hierarchy_,
                       recurse_objects_);
  value->traverse(&cb);
  for (auto& [name, value] : cb.data) {
    data.insert({std::move(name), std::move(value)});
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
