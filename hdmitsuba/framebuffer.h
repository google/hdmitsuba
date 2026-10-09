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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <drjit/array_traverse.h>
#include <pxr/base/gf/rect2i.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>

#include "hdmitsuba/render_buffer.h"

PXR_NAMESPACE_OPEN_SCOPE

// Copies data from a host float buffer (backed by a Mitsuba Tensor) to a Hydra
// Render Buffer.
template <typename TensorT>
void CopyFromHostToRenderBuffer(
    HdMitsubaRenderBuffer* render_buffer, const float* src_data,
    const TensorT& tensor, int src_offset, int channels, bool is_int,
    const std::optional<GfRect2i>& crop_window = std::nullopt) {
  if (!TF_VERIFY(render_buffer, "Render buffer is null.")) {
    return;
  }
  int dst_channels = HdGetComponentCount(render_buffer->GetFormat());
  if (!TF_VERIFY(
          dst_channels == channels || (dst_channels == 4 && channels == 3),
          "Destination and source channel counts do not match (%d vs %d)",
          dst_channels, channels)) {
    return;
  }

  size_t dst_width = render_buffer->GetWidth();
  size_t dst_height = render_buffer->GetHeight();
  size_t src_channels = tensor.shape()[2];
  void* dst_ptr = render_buffer->Map();

  size_t crop_x = 0;
  size_t crop_y = 0;
  size_t crop_w = dst_width;
  size_t crop_h = dst_height;
  if (crop_window.has_value()) {
    crop_x = crop_window->GetMinX();
    crop_y = crop_window->GetMinY();
    crop_w = crop_window->GetWidth();
    crop_h = crop_window->GetHeight();
  }

  if (crop_w != tensor.shape()[1] || crop_h != tensor.shape()[0]) {
    TF_FATAL_ERROR(
        "Tensor dimensions do not match crop window: %lu x %lu vs %lu x %lu",
        crop_w, crop_h, tensor.shape()[1], tensor.shape()[0]);
  }

  if (crop_x != 0 || crop_y != 0 || crop_w != dst_width ||
      crop_h != dst_height) {
    size_t elem_size = is_int ? sizeof(int32_t) : sizeof(float);
    std::memset(dst_ptr, 0, dst_width * dst_height * dst_channels * elem_size);
  }

  if (is_int) {
    int32_t* dst_int = static_cast<int32_t*>(dst_ptr);
    const int copy_channels =
        channels < dst_channels ? channels : dst_channels;
    for (size_t src_y = 0; src_y < crop_h; ++src_y) {
      size_t dst_y = (dst_height - 1) - (crop_y + src_y);
      const float* src_row =
          src_data + src_y * crop_w * src_channels + src_offset;
      int32_t* dst_row = dst_int + (dst_y * dst_width + crop_x) * dst_channels;
      if (copy_channels == 1 && dst_channels == 1) {
        for (size_t src_x = 0; src_x < crop_w; ++src_x) {
          dst_row[src_x] = static_cast<int32_t>(src_row[src_x * src_channels]);
        }
      } else {
        for (size_t src_x = 0; src_x < crop_w; ++src_x) {
          for (int c = 0; c < copy_channels; ++c) {
            dst_row[src_x * dst_channels + c] =
                static_cast<int32_t>(src_row[src_x * src_channels + c]);
          }
        }
      }
    }
  } else {
    float* dst_float = static_cast<float*>(dst_ptr);
    if (src_channels == static_cast<size_t>(dst_channels) &&
        dst_channels == channels && src_offset == 0) {
      const size_t row_bytes = crop_w * dst_channels * sizeof(float);
      for (size_t src_y = 0; src_y < crop_h; ++src_y) {
        size_t dst_y = (dst_height - 1) - (crop_y + src_y);
        std::memcpy(dst_float + (dst_y * dst_width + crop_x) * dst_channels,
                    src_data + src_y * crop_w * src_channels, row_bytes);
      }
    } else if (dst_channels == 4 && channels == 3) {
      for (size_t src_y = 0; src_y < crop_h; ++src_y) {
        size_t dst_y = (dst_height - 1) - (crop_y + src_y);
        const float* src_row =
            src_data + src_y * crop_w * src_channels + src_offset;
        float* dst_row = dst_float + (dst_y * dst_width + crop_x) * 4;
        for (size_t src_x = 0; src_x < crop_w; ++src_x) {
          const float* sp = src_row + src_x * src_channels;
          float* dp = dst_row + src_x * 4;
          dp[0] = sp[0];
          dp[1] = sp[1];
          dp[2] = sp[2];
          dp[3] = 1.0f;
        }
      }
    } else {
      const int copy_channels =
          channels < dst_channels ? channels : dst_channels;
      for (size_t src_y = 0; src_y < crop_h; ++src_y) {
        size_t dst_y = (dst_height - 1) - (crop_y + src_y);
        const float* src_row =
            src_data + src_y * crop_w * src_channels + src_offset;
        float* dst_row =
            dst_float + (dst_y * dst_width + crop_x) * dst_channels;
        for (size_t src_x = 0; src_x < crop_w; ++src_x) {
          for (int c = 0; c < copy_channels; ++c) {
            dst_row[src_x * dst_channels + c] =
                src_row[src_x * src_channels + c];
          }
        }
      }
    }
  }
  render_buffer->SetConverged(true);
  render_buffer->Unmap();
}

// Copies data from a Mitsuba Tensor to a Hydra Render Buffer (Scalar/CPU path).
template <typename TensorT>
void ScalarCopyToRenderBuffer(
    HdMitsubaRenderBuffer* render_buffer, const TensorT& tensor, int src_offset,
    int channels, bool is_int,
    const std::optional<GfRect2i>& crop_window = std::nullopt) {
  CopyFromHostToRenderBuffer(render_buffer, tensor.array().data(), tensor,
                             src_offset, channels, is_int, crop_window);
}

struct CopyDestination {
  HdMitsubaRenderBuffer* buffer;
  int src_offset;
  int channels;
  bool is_int;
};

// Helper function to batch copy all output buffers / AOVs to hydra's render
// buffers.
template <typename Float, typename TensorT>
void PerformBatchedCopy(
    const TensorT& tensor, const std::vector<CopyDestination>& destinations,
    const std::optional<GfRect2i>& crop_window = std::nullopt) {
  namespace dr = drjit;
  if (destinations.empty()) {
    return;
  }
  if constexpr (dr::is_cuda_v<Float> || dr::is_metal_v<Float>) {
    using Int32 = dr::int32_array_t<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    using MigratedFloat = std::decay_t<decltype(dr::migrate(
        std::declval<Float>(), JitBackend::None))>;
    using MigratedInt = std::decay_t<decltype(dr::migrate(std::declval<Int32>(),
                                                          JitBackend::None))>;

    std::vector<std::variant<Float, Int32>> gathered_vars;
    gathered_vars.reserve(destinations.size());
    for (const auto& dest : destinations) {
      int dst_channels = HdGetComponentCount(dest.buffer->GetFormat());
      size_t dst_width = dest.buffer->GetWidth();
      size_t dst_height = dest.buffer->GetHeight();
      size_t dst_pixel_count = dst_width * dst_height;

      size_t crop_x = 0;
      size_t crop_y = 0;
      size_t crop_w = dst_width;
      size_t crop_h = dst_height;
      if (crop_window.has_value()) {
        crop_x = crop_window->GetMinX();
        crop_y = crop_window->GetMinY();
        crop_w = crop_window->GetWidth();
        crop_h = crop_window->GetHeight();
      }

      UInt32 dst_pixel_idx = dr::arange<UInt32>(dst_pixel_count);
      UInt32 dst_x = dst_pixel_idx % dst_width;
      UInt32 dst_row = dst_pixel_idx / dst_width;
      UInt32 r_topdown = dst_height - 1 - dst_row;

      auto in_crop = (dst_x >= crop_x) && (dst_x < crop_x + crop_w) &&
                     (r_topdown >= crop_y) && (r_topdown < crop_y + crop_h);

      UInt32 src_x = dst_x - crop_x;
      UInt32 src_y_down = r_topdown - crop_y;
      UInt32 src_pixel_idx =
          dr::select(in_crop, src_y_down * crop_w + src_x, 0);

      UInt32 repeated_pixel_idx = dr::repeat(src_pixel_idx, dst_channels);
      UInt32 channel_offsets =
          dr::tile(dr::arange<UInt32>(dst_channels), dst_pixel_count);
      UInt32 final_idx = repeated_pixel_idx * tensor.shape()[2] +
                         dest.src_offset + channel_offsets;

      auto in_crop_channel = dr::repeat(in_crop, dst_channels);
      auto valid_channel = in_crop_channel && (channel_offsets < dest.channels);

      if (dest.is_int) {
        Int32 gathered_int =
            Int32(dr::gather<Float>(tensor.array(), final_idx, valid_channel));
        dr::schedule(gathered_int);
        gathered_vars.push_back(std::move(gathered_int));
      } else {
        Float gathered_float =
            dr::gather<Float>(tensor.array(), final_idx, valid_channel);
        if (dst_channels == 4 && dest.channels == 3) {
          gathered_float = dr::select(
              in_crop_channel && (channel_offsets == 3), 1.0f, gathered_float);
        }
        dr::schedule(gathered_float);
        gathered_vars.push_back(std::move(gathered_float));
      }
    }
    dr::eval();

    // Migrate result back to host memory.
    std::vector<std::variant<MigratedFloat, MigratedInt>> migrated_vars;
    migrated_vars.reserve(destinations.size());
    for (size_t i = 0; i < destinations.size(); ++i) {
      std::visit(
          [&](auto& var) {
            migrated_vars.push_back(dr::migrate(var, JitBackend::None));
          },
          gathered_vars[i]);
    }
    dr::sync_thread();

    for (size_t i = 0; i < destinations.size(); ++i) {
      const auto& dest = destinations[i];
      void* dst_ptr = dest.buffer->Map();
      int dst_channels = HdGetComponentCount(dest.buffer->GetFormat());
      size_t dst_width = dest.buffer->GetWidth();
      size_t dst_height = dest.buffer->GetHeight();
      size_t element_size = dest.is_int ? sizeof(int32_t) : sizeof(float);
      size_t size_bytes = dst_width * dst_height * dst_channels * element_size;

      std::visit(
          [&](auto& host_arr) {
            std::memcpy(dst_ptr, host_arr.data(), size_bytes);
          },
          migrated_vars[i]);
      dest.buffer->SetConverged(true);
      dest.buffer->Unmap();
    }
  } else {
    if constexpr (dr::is_jit_v<Float>) {
      dr::eval(tensor.array());
      dr::sync_thread();
    }
    const float* src_data = tensor.array().data();
    for (const auto& dest : destinations) {
      CopyFromHostToRenderBuffer(dest.buffer, src_data, tensor, dest.src_offset,
                                 dest.channels, dest.is_int, crop_window);
    }
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
