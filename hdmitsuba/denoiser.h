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

#include <mitsuba/render/fwd.h>
#include <pxr/pxr.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "hdmitsuba/render_buffer.h"

PXR_NAMESPACE_OPEN_SCOPE

struct AovRequest {
  std::string mitsuba_name;
  HdMitsubaRenderBuffer* buffer = nullptr;
  int channel_count = 0;
};

// Denoises the RGB beauty channels of `display_result` in place using OptiX on
// CUDA variants and OpenImageDenoise on CPU/LLVM/Metal variants, optionally
// guided by `albedo` and `sh_normal` AOVs when present in `aov_requests`.
MI_VARIANT
class Denoiser {
 public:
  using Sensor = mitsuba::Sensor<Float, Spectrum>;
  using TensorXf = typename mitsuba::CoreAliases<Float>::TensorXf;

  Denoiser();
  ~Denoiser();
  Denoiser(Denoiser&&) noexcept;
  Denoiser& operator=(Denoiser&&) noexcept;

  void Denoise(TensorXf& display_result, const Sensor* sensor,
               const std::vector<AovRequest>& aov_requests,
               size_t base_channels, size_t aov_start_channel);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

PXR_NAMESPACE_CLOSE_SCOPE
