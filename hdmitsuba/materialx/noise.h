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

// Pure Dr.Jit port of the MaterialX noise functions in
// `libraries/stdlib/genglsl/lib/mx_noise.glsl` (Apache-2.0, Contributors to the
// MaterialX Project). Include only from `evaluator.cc`.

#include <cstddef>
#include <tuple>
#include <type_traits>

#include <drjit/while_loop.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx::detail {

template <typename UInt32>
UInt32 mx_rotl32(const UInt32& x, int k) {
  return (x << k) | (x >> (32 - k));
}

template <typename UInt32>
void mx_bjmix(UInt32& a, UInt32& b, UInt32& c) {
  a -= c;
  a ^= mx_rotl32(c, 4);
  c += b;
  b -= a;
  b ^= mx_rotl32(a, 6);
  a += c;
  c -= b;
  c ^= mx_rotl32(b, 8);
  b += a;
  a -= c;
  a ^= mx_rotl32(c, 16);
  c += b;
  b -= a;
  b ^= mx_rotl32(a, 19);
  a += c;
  c -= b;
  c ^= mx_rotl32(b, 4);
  b += a;
}

template <typename UInt32>
UInt32 mx_bjfinal(UInt32 a, UInt32 b, UInt32 c) {
  c ^= b;
  c -= mx_rotl32(b, 14);
  a ^= c;
  a -= mx_rotl32(c, 11);
  b ^= a;
  b -= mx_rotl32(a, 25);
  c ^= b;
  c -= mx_rotl32(b, 16);
  a ^= c;
  a -= mx_rotl32(c, 4);
  b ^= a;
  b -= mx_rotl32(a, 14);
  c ^= b;
  c -= mx_rotl32(b, 24);
  return c;
}

template <typename Float, typename UInt32 = dr::uint32_array_t<Float>>
Float mx_bits_to_01(const UInt32& bits) {
  return Float(bits) * Float(1.0 / 0xffffffffu);
}

template <typename Float>
Float mx_fade(const Float& t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

template <typename Int32, typename... Z>
dr::uint32_array_t<Int32> mx_hash_int(const Int32& x, const Int32& y,
                                      const Z&... z) {
  using UInt32 = dr::uint32_array_t<Int32>;
  UInt32 seed = UInt32(0xdeadbeefu + ((2u + sizeof...(Z)) << 2u) + 13u);
  return mx_bjfinal(seed + UInt32(x), seed + UInt32(y),
                    (seed + ... + UInt32(z)));
}

template <typename Float, typename Int32 = dr::int32_array_t<Float>>
Float mx_floorfrac(const Float& x, Int32& i) {
  Float fl = dr::floor(x);
  i = Int32(fl);
  return x - fl;
}

template <int Dim, typename Float, typename UInt32>
Float mx_gradient_float(const UInt32& hash, const Float& x, const Float& y,
                        const Float& z) {
  UInt32 h;
  Float u, v;
  if constexpr (Dim == 2) {
    h = hash & 7u;
    auto lt4 = h < 4u;
    u = dr::select(lt4, x, y);
    v = 2.0f * dr::select(lt4, y, x);
  } else {
    h = hash & 15u;
    u = dr::select(h < 8u, x, y);
    v = dr::select(h < 4u, y, dr::select((h == 12u) | (h == 14u), x, z));
  }
  return dr::select((h & 1u) != 0u, -u, u) + dr::select((h & 2u) != 0u, -v, v);
}

template <int Dim, typename Out, typename Float,
          typename Int32 = dr::int32_array_t<Float>>
Out mx_cell_noise(const Int32& ix, const Int32& iy, const Int32& iz) {
  if constexpr (std::is_same_v<Out, Float>) {
    if constexpr (Dim == 3) {
      return mx_bits_to_01<Float>(mx_hash_int(ix, iy, iz));
    } else {
      return mx_bits_to_01<Float>(mx_hash_int(ix, iy));
    }
  } else if constexpr (Dim == 2) {
    Out r;
    for (int i = 0; i < int(dr::size_v<Out>); ++i) {
      r[i] = mx_bits_to_01<Float>(mx_hash_int(ix, iy, Int32(i)));
    }
    return r;
  } else {
    using UInt32 = dr::uint32_array_t<Float>;
    UInt32 a = UInt32(0xdeadbeefu + (4u << 2u) + 13u);
    UInt32 b = a + UInt32(iy), c = a + UInt32(iz);
    a += UInt32(ix);
    mx_bjmix(a, b, c);
    return Out(mx_bits_to_01<Float>(mx_bjfinal(a, b, c)),
               mx_bits_to_01<Float>(mx_bjfinal(a + 1u, b, c)),
               mx_bits_to_01<Float>(mx_bjfinal(a + 2u, b, c)));
  }
}

template <int Dim, typename Out, typename Float>
Out mx_cell_noise(const Vector<Float, 3>& p) {
  using Int32 = dr::int32_array_t<Float>;
  Int32 ix(dr::floor(p.x())), iy(dr::floor(p.y())), iz(0);
  if constexpr (Dim == 3) iz = Int32(dr::floor(p.z()));
  return mx_cell_noise<Dim, Out, Float>(ix, iy, iz);
}

template <int Dim, typename Out, typename Float>
Out mx_perlin_noise(const Vector<Float, 3>& p) {
  using Int32 = dr::int32_array_t<Float>;
  Int32 X, Y, Z(0);
  Float fx = mx_floorfrac(p.x(), X), fy = mx_floorfrac(p.y(), Y), fz(0.0f);
  if constexpr (Dim == 3) fz = mx_floorfrac(p.z(), Z);
  Float u = mx_fade(fx), v = mx_fade(fy);
  // Bilinearly interpolated gradients of the cell layer `z` (unused in 2D).
  auto layer = [&](const Int32& z, const Float& gz) {
    auto g = [&](int dx, int dy) {
      Int32 x = dx ? X + 1 : X, y = dy ? Y + 1 : Y;
      dr::uint32_array_t<Float> h;
      if constexpr (Dim == 3) {
        h = mx_hash_int(x, y, z);
      } else {
        h = mx_hash_int(x, y);
      }
      Float gx = dx ? fx - 1.0f : fx, gy = dy ? fy - 1.0f : fy;
      if constexpr (std::is_same_v<Out, Float>) {
        return mx_gradient_float<Dim>(h, gx, gy, gz);
      } else {
        return Out(mx_gradient_float<Dim>(h, gx, gy, gz),
                   mx_gradient_float<Dim>(h >> 8, gx, gy, gz),
                   mx_gradient_float<Dim>(h >> 16, gx, gy, gz));
      }
    };
    return dr::lerp(dr::lerp(g(0, 0), g(1, 0), u),
                    dr::lerp(g(0, 1), g(1, 1), u), v);
  };
  if constexpr (Dim == 2) {
    return 0.6616f * layer(Z, fz);
  } else {
    return 0.9820f *
           dr::lerp(layer(Z, fz), layer(Z + 1, fz - 1.0f), mx_fade(fz));
  }
}

template <int Dim, typename Out, typename Float>
Out mx_fractal_noise(Vector<Float, 3> p, int octaves, const Float& lacunarity,
                     const Float& diminish) {
  Out result(0.0f);
  Float amplitude(1.0f);
  for (int i = 0; i < octaves; ++i) {
    result += amplitude * mx_perlin_noise<Dim, Out>(p);
    amplitude *= diminish;
    p *= lacunarity;
  }
  return result;
}

// Fractal noise with an octave count that varies per sample.
template <int Dim, typename Out, typename Float>
Out mx_fractal_noise(const Vector<Float, 3>& p, const Float& octaves,
                     const Float& lacunarity, const Float& diminish,
                     int max_octaves) {
  using Int32 = dr::int32_array_t<Float>;
  using Vec = Vector<Float, 3>;
  // Widen loop-carried state up front so forward-mode AD never sees a width
  // 1 -> W transition mid-body, and copy invariant scalar parameters so
  // symbolic virtual calls wrap outer AD variables in `Gather` nodes.
  size_t w = dr::width(p, octaves, lacunarity, diminish);
  Int32 max_oct = Int32(dr::round(dr::clip(octaves, 0.0f, Float(max_octaves))));
  Float lac = lacunarity, dim = diminish;
  if constexpr (dr::is_diff_v<Float>) {
    lac = lac.copy();
    dim = dim.copy();
  }
  return std::get<1>(dr::while_loop(
      std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Out>(w),
                      dr::full<Float>(1.0f, w), p + dr::zeros<Float>(w),
                      max_oct, lac, dim),
      [](const Int32& i, const Out&, const Float&, const Vec&,
         const Int32& octaves, const Float&,
         const Float&) { return i < octaves; },
      [](Int32& i, Out& result, Float& amplitude, Vec& p, Int32&,
         Float& lacunarity, Float& diminish) {
        result += amplitude * mx_perlin_noise<Dim, Out>(p);
        amplitude *= diminish;
        p *= lacunarity;
        i += 1;
      },
      "MaterialX fractal noise", max_octaves));
}

// Returns the cell noise of the cell containing the nearest jittered feature
// point where `cell` is set (a scalar for `Out = Float`), and the distances
// to the three nearest feature points elsewhere, matching
// `mx_worley_noise_*()` in `mx_noise.glsl`.
template <int Dim, typename Out, typename Float>
Out mx_worley_noise(const Vector<Float, 3>& p, const Float& jitter,
                    const dr::mask_t<Float>& cell) {
  using Int32 = dr::int32_array_t<Float>;
  using VecD = Vector<Float, Dim>;
  Int32 X, Y, Z(0);
  VecD local_pos, min_off(0.0f);
  local_pos.x() = mx_floorfrac(p.x(), X);
  local_pos.y() = mx_floorfrac(p.y(), Y);
  if constexpr (Dim == 3) local_pos.z() = mx_floorfrac(p.z(), Z);
  Float f1(1e6f);
  [[maybe_unused]] Float f2(1e6f), f3(1e6f);
  constexpr int dz_max = (Dim == 3) ? 1 : 0;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -dz_max; dz <= dz_max; ++dz) {
        VecD rnd = mx_cell_noise<Dim, VecD, Float>(X + dx, Y + dy, Z + dz);
        VecD cell_off = (rnd - 0.5f) * jitter;
        cell_off.x() += dx + 0.5f;
        cell_off.y() += dy + 0.5f;
        if constexpr (Dim == 3) cell_off.z() += dz + 0.5f;
        Float d2 = dr::squared_norm(cell_off - local_pos);
        min_off = dr::select(d2 < f1, cell_off, min_off);
        if constexpr (!std::is_same_v<Out, Float>) {
          f3 = dr::minimum(f3, dr::maximum(d2, f2));
          f2 = dr::minimum(f2, dr::maximum(d2, f1));
        }
        f1 = dr::minimum(f1, d2);
      }
    }
  }
  Int32 cx = X + Int32(dr::floor(min_off.x()));
  Int32 cy = Y + Int32(dr::floor(min_off.y()));
  Int32 cz(0);
  if constexpr (Dim == 3) cz = Z + Int32(dr::floor(min_off.z()));
  Out cell_out = mx_cell_noise<Dim, Out, Float>(cx, cy, cz);
  Out dist;
  if constexpr (std::is_same_v<Out, Float>) {
    dist = dr::safe_sqrt(f1);
  } else {
    dist = dr::safe_sqrt(Out(f1, f2, f3));
  }
  return dr::select(cell, cell_out, dist);
}

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
