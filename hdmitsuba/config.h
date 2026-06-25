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

#include <string>

#include <pxr/base/tf/singleton.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaConfig {
 public:
  static const HdMitsubaConfig& GetInstance();

  /// The Mitsuba variant to use.
  std::string variant;

  /// Global override for kernel freezing: -1 = use USD settings, 0 = force off, 1 = force on
  int use_kernel_freezing = -1;

 private:
  HdMitsubaConfig();
  ~HdMitsubaConfig() = default;

  HdMitsubaConfig(const HdMitsubaConfig&) = delete;
  HdMitsubaConfig& operator=(const HdMitsubaConfig&) = delete;
  friend class TfSingleton<HdMitsubaConfig>;
};

PXR_NAMESPACE_CLOSE_SCOPE
