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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/transform.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class TraversalCallback : public mitsuba::TraversalCallback {
 public:
  explicit TraversalCallback(std::string_view prefix = "",
                             mitsuba::Object* parent = nullptr);
  TraversalCallback(std::string_view prefix, mitsuba::Object* parent,
                    const absl::flat_hash_set<void*>& parent_hierarchy);

  template <typename T>
  T* get(std::string_view name) {
    auto it = data.find(prefix_ + std::string(name));
    if (!TF_VERIFY(it != data.end(), "No value found for %s",
                   std::string(name).c_str())) {
      return nullptr;
    }
    const std::type_info& type = it->second.second;
    if (!TF_VERIFY(type == typeid(T) ||
                       std::string_view(type.name()) == typeid(T).name(),
                   "Value %s is of type %s, but requested %s",
                   std::string(name).c_str(), type.name(), typeid(T).name())) {
      return nullptr;
    }
    return static_cast<T*>(it->second.first);
  }

  template <typename T>
  void set(std::string_view name, const T& value) {
    if (T* ptr = get<T>(name)) {
      *ptr = value;
    }
  }

  absl::flat_hash_map<std::string, std::pair<void*, const std::type_info&>>
      data;

 protected:
  void put_value(std::string_view name, void* value, uint32_t flags,
                 const std::type_info& type) override;
  void put_object(std::string_view name, mitsuba::Object* value,
                  uint32_t flags) override;

 private:
  absl::flat_hash_set<void*> hierarchy_;
  std::string prefix_;
};

PXR_NAMESPACE_CLOSE_SCOPE
