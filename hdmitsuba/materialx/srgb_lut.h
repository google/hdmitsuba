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

// Traversable sRGB-to-spectrum upsampler backed by a weak Float32 LUT cache.
// Include only from `evaluator.cc` so the cache is never duplicated across TUs.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/synchronization/mutex.h>
#include <drjit/traversable_base.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/srgb.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx::detail {

// Device-resident `data/srgb.coeff` lookup tables shared via a weak cache so
// `SrgbUpsampler::traverse_cb()` can expose `data` and `scale` to `@dr.freeze`
// and `FrozenRender`.
template <typename Float32>
struct SrgbLut {
  uint32_t res = 0;
  Float32 data;
  Float32 scale;
};

template <typename Float32>
std::shared_ptr<SrgbLut<Float32>> get_srgb_lut() {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<std::weak_ptr<SrgbLut<Float32>>> cached;
  absl::MutexLock lock(*mu);
  if (auto lut = cached->lock()) return lut;

  fs::path path = file_resolver()->resolve("data/srgb.coeff");
  std::ifstream f(path, std::ios::binary);
  char header[4] = {};
  uint32_t res = 0;
  if (!f.read(header, 4) || std::memcmp(header, "SPEC", 4) != 0 ||
      !f.read(reinterpret_cast<char*>(&res), sizeof(res)) || res < 2) {
    Throw("Could not load sRGB-to-spectrum upsampling model ('%s')",
          path.string());
  }
  std::vector<float> host_scale(res);
  std::vector<float> host_data(size_t(res) * res * res * 9);
  if (!f.read(reinterpret_cast<char*>(host_scale.data()),
              host_scale.size() * sizeof(float)) ||
      !f.read(reinterpret_cast<char*>(host_data.data()),
              host_data.size() * sizeof(float))) {
    Throw("Truncated sRGB-to-spectrum upsampling model ('%s')", path.string());
  }

  auto lut = std::make_shared<SrgbLut<Float32>>();
  lut->res = res;
  lut->scale = dr::load<Float32>(host_scale.data(), host_scale.size());
  lut->data = dr::load<Float32>(host_data.data(), host_data.size());
  *cached = lut;
  return lut;
}

template <typename Float, typename Spectrum>
class SrgbUpsampler {
 public:
  using Color3f = Color<Float, 3>;
  using Wavelength = wavelength_t<Spectrum>;
  using UnpolarizedSpectrum = unpolarized_spectrum_t<Spectrum>;

  void load() {
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      m_lut = get_srgb_lut<Float32>();
    }
  }

  // Spectral upsampling of an RGB color computed at render time. `SRGBModel`
  // upsamples colors in [0, 1], so an unbounded color `c` maps to
  // `2 * max(c) * S(c / (2 * max(c)))`, like `D65Spectrum::normalization` for
  // constant colors: the brightest channel sits at 0.5 inside the smooth
  // interior of the model, and the mapping remains homogeneous of degree 1 for
  // unbounded HDR emission and conductor IOR/extinction.
  UnpolarizedSpectrum upsample(const Color3f& c, const Wavelength& wavelengths,
                               bool unbounded) const {
    if constexpr (!is_spectral_v<Spectrum>) {
      (void)wavelengths;
      (void)unbounded;
      if constexpr (is_monochromatic_v<Spectrum>) {
        return luminance(c);
      } else {
        return c;
      }
    } else {
      if (!unbounded) {
        return srgb_model_eval<UnpolarizedSpectrum>(fetch(c), wavelengths);
      }
      Color3f rgb = dr::maximum(c, 0.0f);
      Float z = dr::max(rgb) * 2.0f;
      auto active = z > 0.0f;
      return srgb_model_eval<UnpolarizedSpectrum>(
                 fetch(dr::select(active, rgb / dr::select(active, z, 1.0f),
                                  Color3f(0.5f))),
                 wavelengths) *
             z;
    }
  }

  void traverse_cb([[maybe_unused]] void* payload,
                   [[maybe_unused]] const drjit::TraverseVisitor& cb) {
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      if (m_lut) {
        DR_TRAVERSE_MEMBER(m_lut->scale)
        DR_TRAVERSE_MEMBER(m_lut->data)
      }
    }
  }

 private:
  using Float32 = dr::float32_array_t<dr::detached_t<Float>>;

  // Mirrors `mitsuba::SRGBModel::fetch_jit()` (`mitsuba/src/render/srgb.cpp`),
  // reading from the traversable `m_lut`.
  dr::Array<Float, 3> fetch(const Color3f& color) const {
    if constexpr (!dr::is_jit_v<Float>) {
      return SRGBModel<Float, Spectrum>::fetch(color);
    } else {
      using Array3f = dr::Array<Float, 3>;
      using UInt32 = dr::uint32_array_t<Float>;
      using Mask = dr::mask_t<Float>;

      uint32_t res = m_lut->res;
      float res_f = static_cast<float>(res);
      Color3f c = dr::clip(color, 0.0f, 1.0f);

      Mask g_ge_r = c.g() >= c.r();
      Float rg = dr::select(g_ge_r, c.g(), c.r());
      Mask b_largest = c.b() >= rg;

      UInt32 i = dr::select(b_largest, 2u, dr::select(g_ge_r, 1u, 0u));
      Float z = dr::select(b_largest, c.b(), rg),
            x = dr::select(b_largest, c.r(), dr::select(g_ge_r, c.b(), c.g())),
            y = dr::select(b_largest, c.g(), dr::select(g_ge_r, c.r(), c.b()));

      Float scale = (res_f - 1.0f) / dr::maximum(z, 1e-9f);
      x *= scale;
      y *= scale;

      auto scale_at = [&](const UInt32& idx) {
        return Float(dr::gather<Float32>(m_lut->scale, dr::detach<false>(idx)));
      };
      UInt32 zi = dr::binary_search<UInt32>(
                      1, res - 1,
                      [&](const UInt32& idx) { return scale_at(idx) <= z; }) -
                  1;
      Float z0 = scale_at(zi), z1 = scale_at(zi + 1);

      UInt32 xi = dr::minimum(UInt32(x), res - 2),
             yi = dr::minimum(UInt32(y), res - 2);
      Float wx = x - Float(xi), wy = y - Float(yi), wz = (z - z0) / (z1 - z0);

      UInt32 index = ((i * res + zi) * res + yi) * res + xi, dy = res,
             dz = res * res;
      auto fetch_coeff = [&](const UInt32& idx) {
        return Array3f(dr::gather<dr::Array<Float32, 3>>(
            m_lut->data, dr::detach<false>(idx)));
      };
      auto lerp_x = [&](const UInt32& idx) {
        return dr::lerp(fetch_coeff(idx), fetch_coeff(idx + 1), wx);
      };
      Array3f result = dr::lerp(
          dr::lerp(lerp_x(index), lerp_x(index + dy), wy),
          dr::lerp(lerp_x(index + dz), lerp_x(index + dz + dy), wy), wz);

      Float v = c.r();
      Mask mono = (c.g() == v) && (c.b() == v);
      Float mono_c2 =
          dr::select(v <= 0.0f, -8192.0f,
                     dr::select(v >= 1.0f, 8192.0f,
                                (v - 0.5f) * dr::rsqrt(v * (1.0f - v))));
      Array3f mono_coeff(0.0f, 0.0f, mono_c2);
      if constexpr (dr::is_diff_v<Float>) {
        mono_coeff = dr::replace_grad(mono_coeff, result);
      }
      return dr::select(mono, mono_coeff, result);
    }
  }

  std::shared_ptr<SrgbLut<Float32>> m_lut;
};

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
