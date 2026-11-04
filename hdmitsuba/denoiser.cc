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

#include "hdmitsuba/denoiser.h"

#include <drjit-core/jit.h>
#include <drjit/array.h>
#include <drjit/tensor.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/sensor.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/pxr.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HDMITSUBA_ENABLE_OIDN)
#include <OpenImageDenoise/oidn.hpp>
#endif

#if defined(MI_ENABLE_CUDA)
#include <mitsuba/render/optixdenoiser.h>
#endif

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

#if defined(HDMITSUBA_ENABLE_OIDN)
MI_VARIANT
struct OidnDenoiserBackend {
  MI_IMPORT_TYPES(Sensor);

  void Denoise(TensorXf& display_result, const Sensor* /*sensor*/,
               size_t /*base_channels*/,
               const std::optional<size_t>& albedo_offset,
               const std::optional<size_t>& normal_offset) {
    const size_t height = display_result.shape()[0];
    const size_t width = display_result.shape()[1];
    const size_t total_channels = display_result.shape()[2];

    if (!device) {
      device = oidn::newDevice(oidn::DeviceType::CPU);
      if (device) {
        device.commit();
      }
      const char* error_message = nullptr;
      if (!device || device.getError(error_message) != oidn::Error::None) {
        TF_RUNTIME_ERROR(
            "Failed to initialize OpenImageDenoise CPU device: %s",
            error_message ? error_message : "unknown error");
        device.release();
        return;
      }
    }
    if (!filter) {
      filter = device.newFilter("RT");
    }

    if constexpr (dr::is_jit_v<Float>) {
      dr::eval(display_result);
      display_result =
          TensorXf(dr::migrate(display_result.array(), JitBackend::None),
                   {height, width, total_channels});
      dr::sync_thread();
    }
    float* host_data = display_result.array().data();

    const size_t pixel_stride = total_channels * sizeof(float);
    const size_t row_stride = width * pixel_stride;
    filter.setImage("color", host_data, oidn::Format::Float3, width, height,
                    /*byteOffset=*/0, pixel_stride, row_stride);
    if (albedo_offset.has_value()) {
      filter.setImage("albedo", host_data, oidn::Format::Float3, width, height,
                      *albedo_offset * sizeof(float), pixel_stride, row_stride);
    } else {
      filter.unsetImage("albedo");
    }
    if (normal_offset.has_value()) {
      filter.setImage("normal", host_data, oidn::Format::Float3, width, height,
                      *normal_offset * sizeof(float), pixel_stride, row_stride);
    } else {
      filter.unsetImage("normal");
    }
    filter.setImage("output", host_data, oidn::Format::Float3, width, height,
                    /*byteOffset=*/0, pixel_stride, row_stride);
    filter.set("hdr", true);
    filter.commit();
    filter.execute();

    const char* error_message = nullptr;
    if (device.getError(error_message) != oidn::Error::None) {
      TF_RUNTIME_ERROR("OpenImageDenoise filter execution failed: %s",
                       error_message ? error_message : "unknown error");
    }

    if constexpr (dr::is_metal_v<Float>) {
      display_result =
          TensorXf(dr::migrate(display_result.array(), JitBackend::Metal),
                   {height, width, total_channels});
    }
  }

  oidn::DeviceRef device;
  oidn::FilterRef filter;
};

template <typename Float, typename Spectrum>
using CpuDenoiserBackend = OidnDenoiserBackend<Float, Spectrum>;
#else
MI_VARIANT
struct UnavailableOidnDenoiserBackend {
  MI_IMPORT_TYPES(Sensor);

  void Denoise(TensorXf& /*display_result*/, const Sensor* /*sensor*/,
               size_t /*base_channels*/,
               const std::optional<size_t>& /*albedo_offset*/,
               const std::optional<size_t>& /*normal_offset*/) {
    TF_RUNTIME_ERROR(
        "CPU/LLVM/Metal denoising requires hdMitsuba to be built with "
        "OpenImageDenoise support (HDMITSUBA_ENABLE_OIDN).");
  }
};

template <typename Float, typename Spectrum>
using CpuDenoiserBackend = UnavailableOidnDenoiserBackend<Float, Spectrum>;
#endif

#if defined(MI_ENABLE_CUDA)
struct ScopedCudaContext {
  ScopedCudaContext() { jit_cuda_push_context(jit_cuda_context()); }
  ~ScopedCudaContext() {
    try {
      jit_cuda_pop_context();
    } catch (...) {
    }
  }
  ScopedCudaContext(const ScopedCudaContext&) = delete;
  ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;
};

MI_VARIANT
struct OptixDenoiserBackend {
  MI_IMPORT_TYPES(Sensor);

  using ConfigKey = std::tuple<uint32_t, uint32_t, bool, bool>;
  using OptixDenoiserType = mitsuba::OptixDenoiser<Float, Spectrum>;

  struct CacheSlot {
    mitsuba::ref<OptixDenoiserType> denoiser = nullptr;
    ConfigKey config{0, 0, false, false};
  };

  ~OptixDenoiserBackend() {
    if ((slots[0].denoiser || slots[1].denoiser) &&
        jit_has_backend(JitBackend::CUDA)) {
      try {
        ScopedCudaContext cuda_guard;
        slots[0].denoiser = nullptr;
        slots[1].denoiser = nullptr;
      } catch (...) {
      }
    }
  }

  // Returns a cached OptiX denoiser matching the resolution and guide AOVs,
  // keeping the two most recently used configurations warm.
  const OptixDenoiserType& GetOrCreateDenoiser(uint32_t width, uint32_t height,
                                               bool has_albedo,
                                               bool has_normals) {
    const ConfigKey config_key(width, height, has_albedo, has_normals);
    if (slots[0].denoiser && slots[0].config == config_key) {
      return *slots[0].denoiser;
    }
    if (slots[1].denoiser && slots[1].config == config_key) {
      std::swap(slots[0], slots[1]);
      return *slots[0].denoiser;
    }
    // Mitsuba's ~OptixDenoiser calls optixDenoiserDestroy without pushing
    // Dr.Jit's CUDA context, so guard both construction and slot eviction.
    ScopedCudaContext cuda_guard;
    mitsuba::ref<OptixDenoiserType> new_denoiser = new OptixDenoiserType(
        ScalarVector2u(width, height), has_albedo, has_normals);
    slots[1] = std::move(slots[0]);
    slots[0] = CacheSlot{std::move(new_denoiser), config_key};
    return *slots[0].denoiser;
  }

  void Denoise(TensorXf& display_result, const Sensor* sensor,
               size_t base_channels,
               const std::optional<size_t>& albedo_offset,
               const std::optional<size_t>& normal_offset) {
    const size_t height = display_result.shape()[0];
    const size_t width = display_result.shape()[1];
    const size_t total_channels = display_result.shape()[2];
    const bool has_albedo = albedo_offset.has_value();
    const bool has_normals = normal_offset.has_value();
    const OptixDenoiserType& denoiser = GetOrCreateDenoiser(
        static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        has_albedo, has_normals);

    if (total_channels == base_channels &&
        (base_channels == 3 || base_channels == 4)) {
      display_result = denoiser(display_result);
    } else {
      using UInt32 = dr::uint32_array_t<Float>;
      const size_t pixel_count = width * height;
      const UInt32 index = dr::arange<UInt32>(pixel_count * 3);
      const UInt32 base_pixel_index =
          (index / 3) * static_cast<uint32_t>(total_channels) + (index % 3);
      auto extract_rgb_tensor = [&](size_t channel_offset) -> TensorXf {
        UInt32 source_index =
            base_pixel_index + static_cast<uint32_t>(channel_offset);
        Float gathered =
            dr::gather<Float>(display_result.array(), source_index);
        return TensorXf(std::move(gathered), {height, width, 3});
      };

      TensorXf noisy_rgb = extract_rgb_tensor(0);
      TensorXf albedo_tensor =
          has_albedo ? extract_rgb_tensor(*albedo_offset) : TensorXf();
      TensorXf normals_tensor =
          has_normals ? extract_rgb_tensor(*normal_offset) : TensorXf();
      AffineTransform4f to_sensor =
          has_normals
              ? AffineTransform4f(sensor->world_transform_scalar().inverse())
              : AffineTransform4f();

      dr::eval(noisy_rgb, albedo_tensor, normals_tensor);
      TensorXf denoised_rgb =
          denoiser(noisy_rgb, albedo_tensor, normals_tensor, to_sensor);

      dr::scatter(display_result.array(), denoised_rgb.array(),
                  base_pixel_index);
    }
  }

  // Two-slot MRU cache so interactive viewports alternating between a
  // downsampled drag resolution and full resolution keep both warm.
  std::array<CacheSlot, 2> slots;
};
#endif

}  // namespace

MI_VARIANT
struct Denoiser<Float, Spectrum>::Impl {
#if defined(MI_ENABLE_CUDA)
  using DenoiserBackend =
      std::conditional_t<dr::is_cuda_v<Float>,
                         OptixDenoiserBackend<Float, Spectrum>,
                         CpuDenoiserBackend<Float, Spectrum>>;
#else
  using DenoiserBackend = CpuDenoiserBackend<Float, Spectrum>;
#endif

  DenoiserBackend backend;
};

MI_VARIANT Denoiser<Float, Spectrum>::Denoiser() = default;
MI_VARIANT Denoiser<Float, Spectrum>::~Denoiser() = default;
MI_VARIANT Denoiser<Float, Spectrum>::Denoiser(Denoiser&&) noexcept = default;
MI_VARIANT Denoiser<Float, Spectrum>& Denoiser<Float, Spectrum>::operator=(
    Denoiser&&) noexcept = default;

MI_VARIANT void Denoiser<Float, Spectrum>::Denoise(
    TensorXf& display_result, const Sensor* sensor,
    const std::vector<AovRequest>& aov_requests, size_t base_channels,
    size_t aov_start_channel) {
  const size_t height = display_result.shape()[0];
  const size_t width = display_result.shape()[1];
  const size_t total_channels = display_result.shape()[2];
  if (width == 0 || height == 0 || total_channels < base_channels) {
    return;
  }

  std::optional<size_t> albedo_offset;
  std::optional<size_t> normal_offset;
  size_t current_offset = aov_start_channel;
  for (const AovRequest& aov_request : aov_requests) {
    if (current_offset + aov_request.channel_count > total_channels) {
      break;
    }
    if (aov_request.mitsuba_name == "albedo:albedo" &&
        !albedo_offset.has_value()) {
      albedo_offset = current_offset;
    } else if (aov_request.mitsuba_name == "sh_normal:sh_normal" &&
               !normal_offset.has_value()) {
      normal_offset = current_offset;
    }
    current_offset += aov_request.channel_count;
  }
  if (normal_offset.has_value() && !albedo_offset.has_value()) {
    TF_WARN(
        "Denoising with shading normals ('sh_normal' or 'normal' AOV) "
        "requires the 'albedo' AOV; denoising color without guide AOVs.");
    normal_offset.reset();
  }

  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  try {
    impl_->backend.Denoise(display_result, sensor, base_channels, albedo_offset,
                           normal_offset);
  } catch (const std::exception& error) {
    TF_RUNTIME_ERROR("Denoising failed: %s", error.what());
  }
}

using mitsuba::Color;
using mitsuba::MuellerMatrix;
using mitsuba::Spectrum;

MI_INSTANTIATE_CLASS(Denoiser)

PXR_NAMESPACE_CLOSE_SCOPE
