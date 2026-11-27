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

#include <drjit-core/jit.h>
#include <drjit/array.h>
#include <drjit/tensor.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/sensor.h>
#include <pxr/base/tf/debug.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/pxr.h>

#include "hdmitsuba/debug_codes.h"

#if defined(HDMITSUBA_ENABLE_OIDN)
#include <OpenImageDenoise/oidn.hpp>
#endif

#if defined(MI_ENABLE_CUDA)
#include <mitsuba/render/optixdenoiser.h>
#endif

#if defined(MI_ENABLE_METAL)
#include <drjit-core/metal.h>
#endif

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

#if defined(HDMITSUBA_ENABLE_OIDN)
#if defined(__APPLE__) && OIDN_VERSION >= 20200
#define HDMITSUBA_OIDN_METAL 1
#if defined(MI_ENABLE_METAL)
#define HDMITSUBA_OIDN_METAL_INTEROP 1
#endif
#endif

bool CheckOidnError(oidn::DeviceRef& device, const char* context) {
  const char* error_message = nullptr;
  if (device.getError(error_message) == oidn::Error::None) {
    return true;
  }
  TF_RUNTIME_ERROR("%s: %s", context,
                   error_message ? error_message : "unknown error");
  return false;
}

MI_VARIANT
struct OidnDenoiserBackend {
  MI_IMPORT_TYPES(Sensor);

  bool EnsureDevice() {
#if defined(HDMITSUBA_OIDN_METAL_INTEROP)
    if constexpr (dr::is_metal_v<Float>) {
      if (device && metal_queue != nullptr &&
          jit_metal_command_queue() != metal_queue) {
        filter.release();
        device.release();
        metal_queue = nullptr;
      }
    }
#endif
    if (device) {
      return true;
    }

#if defined(HDMITSUBA_OIDN_METAL)
    auto try_metal = [&](oidn::DeviceRef candidate) {
      if (candidate) {
        candidate.commit();
        if (candidate.getError() == oidn::Error::None) {
          device = std::move(candidate);
          return true;
        }
      }
      oidn::getError();
      return false;
    };
    if constexpr (dr::is_metal_v<Float>) {
#if defined(HDMITSUBA_OIDN_METAL_INTEROP)
      if (void* queue = jit_metal_command_queue();
          try_metal(oidn::newMetalDevice(queue))) {
        metal_queue = queue;
      }
#endif
    } else {
      try_metal(oidn::newDevice(oidn::DeviceType::Metal));
    }
#endif

    if (!device) {
      device = oidn::newDevice(oidn::DeviceType::CPU);
      if (device) {
        device.commit();
      }
      if (!CheckOidnError(device, "Failed to initialize OpenImageDenoise") ||
          !device) {
        device.release();
        return false;
      }
    }
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Initialized OpenImageDenoise %s device\n",
             device.get<oidn::DeviceType>("type") == oidn::DeviceType::CPU
                 ? "CPU"
                 : "Metal");
    return true;
  }

  bool RunFilter(const oidn::BufferRef& in_buf, size_t in_offset,
                 const oidn::BufferRef& out_buf, size_t out_offset,
                 const TensorXf& tensor,
                 const std::optional<size_t>& albedo_offset,
                 const std::optional<size_t>& normal_offset) {
    if (!in_buf || !out_buf) {
      CheckOidnError(device, "Failed to prepare OpenImageDenoise buffers");
      return false;
    }
    if (!filter) {
      filter = device.newFilter("RT");
    }
    const size_t height = tensor.shape()[0];
    const size_t width = tensor.shape()[1];
    const size_t pixel_stride = tensor.shape()[2] * sizeof(float);
    const size_t row_stride = width * pixel_stride;
    auto bind = [&](const char* name, std::optional<size_t> ch) {
      if (ch) {
        filter.setImage(name, in_buf, oidn::Format::Float3, width, height,
                        in_offset + *ch * sizeof(float), pixel_stride,
                        row_stride);
      } else {
        filter.unsetImage(name);
      }
    };
    bind("color", 0);
    bind("albedo", albedo_offset);
    bind("normal", normal_offset);
    filter.setImage("output", out_buf, oidn::Format::Float3, width, height,
                    out_offset, pixel_stride, row_stride);
    filter.set("hdr", true);
    filter.commit();
    filter.execute();
    return CheckOidnError(device, "OpenImageDenoise filter execution failed");
  }

  void Denoise(TensorXf& display_result, const Sensor* /*sensor*/,
               size_t /*base_channels*/,
               const std::optional<size_t>& albedo_offset,
               const std::optional<size_t>& normal_offset) {
    if (!EnsureDevice()) {
      return;
    }

#if defined(HDMITSUBA_OIDN_METAL_INTEROP)
    if constexpr (dr::is_metal_v<Float>) {
      if (metal_queue != nullptr) {
        // Write into an unaliased Metal buffer so denoising never mutates a
        // shared progressive accumulation buffer.
        float* in_ptr = display_result.array().data();
        Float out_arr = display_result.shape()[2] == 3
                            ? dr::empty<Float>(display_result.size())
                            : display_result.array().copy();
        float* out_ptr = out_arr.data();
        jit_flush_thread();

        auto wrap = [&](void* ptr, size_t& offset) -> oidn::BufferRef {
          void* mtl = jit_metal_lookup_buffer(ptr, &offset);
          return mtl ? oidnNewSharedBufferFromMetal(device.getHandle(), mtl)
                     : nullptr;
        };
        size_t in_offset = 0;
        size_t out_offset = 0;
        oidn::BufferRef in_buf = wrap(in_ptr, in_offset);
        oidn::BufferRef out_buf = wrap(out_ptr, out_offset);
        if (RunFilter(in_buf, in_offset, out_buf, out_offset, display_result,
                      albedo_offset, normal_offset)) {
          display_result = TensorXf(std::move(out_arr), display_result.shape());
        }
        return;
      }
    }
#endif

    if constexpr (dr::is_jit_v<Float>) {
      dr::eval(display_result);
      display_result =
          TensorXf(dr::migrate(display_result.array(), JitBackend::None),
                   display_result.shape());
      dr::sync_thread();
    }
    float* host_data = display_result.array().data();
    const size_t total_bytes = display_result.size() * sizeof(float);
    if (device.get<bool>("systemMemorySupported")) {
      oidn::BufferRef buf = device.newBuffer(host_data, total_bytes);
      RunFilter(buf, 0, buf, 0, display_result, albedo_offset, normal_offset);
    } else {
      if (!staging_buffer || staging_buffer.getSize() < total_bytes) {
        staging_buffer = device.newBuffer(total_bytes);
      }
      if (staging_buffer) {
        staging_buffer.write(/*byteOffset=*/0, total_bytes, host_data);
      }
      if (RunFilter(staging_buffer, 0, staging_buffer, 0, display_result,
                    albedo_offset, normal_offset)) {
        staging_buffer.read(/*byteOffset=*/0, total_bytes, host_data);
      }
    }

    if constexpr (dr::is_metal_v<Float>) {
      display_result =
          TensorXf(dr::migrate(display_result.array(), JitBackend::Metal),
                   display_result.shape());
    }
  }

  // `filter` and `staging_buffer` must be destroyed before `device`.
  oidn::DeviceRef device;
  oidn::BufferRef staging_buffer;
  oidn::FilterRef filter;
#if defined(HDMITSUBA_OIDN_METAL_INTEROP)
  void* metal_queue = nullptr;
#endif
};

#undef HDMITSUBA_OIDN_METAL
#undef HDMITSUBA_OIDN_METAL_INTEROP

template <typename Float, typename Spectrum>
using NonCudaDenoiserBackend = OidnDenoiserBackend<Float, Spectrum>;
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
using NonCudaDenoiserBackend = UnavailableOidnDenoiserBackend<Float, Spectrum>;
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
               size_t base_channels, const std::optional<size_t>& albedo_offset,
               const std::optional<size_t>& normal_offset) {
    const size_t height = display_result.shape()[0];
    const size_t width = display_result.shape()[1];
    const size_t total_channels = display_result.shape()[2];
    const bool has_albedo = albedo_offset.has_value();
    const bool has_normals = normal_offset.has_value();
    const OptixDenoiserType& denoiser = GetOrCreateDenoiser(
        static_cast<uint32_t>(width), static_cast<uint32_t>(height), has_albedo,
        has_normals);

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
                         NonCudaDenoiserBackend<Float, Spectrum>>;
#else
  using DenoiserBackend = NonCudaDenoiserBackend<Float, Spectrum>;
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
