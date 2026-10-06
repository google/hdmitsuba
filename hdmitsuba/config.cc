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

#include "hdmitsuba/config.h"

#include <cstdint>
#include <string>
#include <string_view>

#include <absl/strings/match.h>
#include <absl/strings/str_split.h>
#include <drjit-core/jit.h>
#include <mitsuba/core/config.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/envSetting.h>
#include <pxr/base/tf/instantiateSingleton.h>
#include <pxr/base/tf/singleton.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

bool IsVariantCompiled(std::string_view target) {
  for (std::string_view v :
       absl::StrSplit(MI_VARIANTS, '\n', absl::SkipEmpty())) {
    if (v == target) {
      return true;
    }
  }
  return false;
}

bool IsVariantBackendAvailable(std::string_view variant) {
  if (absl::StartsWith(variant, "scalar_")) {
    return true;
  }
  if (absl::StartsWith(variant, "cuda_")) {
    jit_init(1u << static_cast<uint32_t>(JitBackend::CUDA));
    return jit_has_backend(JitBackend::CUDA);
  }
  if (absl::StartsWith(variant, "llvm_")) {
    jit_init(1u << static_cast<uint32_t>(JitBackend::LLVM));
    return jit_has_backend(JitBackend::LLVM);
  }
  if (absl::StartsWith(variant, "metal_")) {
    jit_init(1u << static_cast<uint32_t>(JitBackend::Metal));
    return jit_has_backend(JitBackend::Metal);
  }
  return false;
}

std::string SelectDefaultVariant() {
  for (std::string_view v :
       absl::StrSplit(MI_VARIANTS, '\n', absl::SkipEmpty())) {
    if (IsVariantBackendAvailable(v)) {
      return std::string(v);
    }
  }
  return "scalar_rgb";
}

}  // namespace

TF_INSTANTIATE_SINGLETON(HdMitsubaConfig);

TF_DEFINE_ENV_SETTING(
    HDMITSUBA_VARIANT, "",
    "The used Mitsuba variant (empty selects the best available variant).");
TF_DEFINE_ENV_SETTING(
    HDMITSUBA_PRINT_CONFIGURATION, 0,
    "Should HdMitsuba print configuration on startup? (values > 0 are true)");
TF_DEFINE_ENV_SETTING(
    HDMITSUBA_USE_KERNEL_FREEZING, -1,
    "Force kernel freezing on (1) or off (0), or use USD settings (-1)");

HdMitsubaConfig::HdMitsubaConfig() {
  variant = TfGetEnvSetting(HDMITSUBA_VARIANT);
  if (variant.empty()) {
    variant = SelectDefaultVariant();
  } else if (!IsVariantCompiled(variant)) {
    std::string fallback = SelectDefaultVariant();
    TF_WARN(
        "Requested HDMITSUBA_VARIANT '%s' is not compiled into Mitsuba; "
        "falling back to '%s'.",
        variant.c_str(), fallback.c_str());
    variant = fallback;
  }
  use_kernel_freezing = TfGetEnvSetting(HDMITSUBA_USE_KERNEL_FREEZING);
  if (TfGetEnvSetting(HDMITSUBA_PRINT_CONFIGURATION) > 0) {
    TF_STATUS("HdMitsuba Configuration:\n  variant             = %s\n  use_kernel_freezing = %d",
              variant.c_str(), use_kernel_freezing);
  }
}

const HdMitsubaConfig& HdMitsubaConfig::GetInstance() {
  return TfSingleton<HdMitsubaConfig>::GetInstance();
}

PXR_NAMESPACE_CLOSE_SCOPE
