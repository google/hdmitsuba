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
  using Vec = Vector<Float, 3>;
  Vector<Int32, 3> cell(0);
  Vec frac(mx_floorfrac(p.x(), cell.x()), mx_floorfrac(p.y(), cell.y()), 0.0f);
  if constexpr (Dim == 3) frac.z() = mx_floorfrac(p.z(), cell.z());
  Vec fade = mx_fade(frac);
  size_t w = dr::width(p);
  Out sum = std::get<1>(dr::while_loop(
      std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Out>(w), frac, fade),
      [](const Int32& k, const Out&, const Vec&, const Vec&) {
        return k < (1 << Dim);
      },
      [cell](Int32& k, Out& acc, Vec& f, Vec& fd) {
        auto bx = (k & 1) != 0, by = (k & 2) != 0;
        dr::uint32_array_t<Float> h;
        Float gx = dr::select(bx, f.x() - 1.0f, f.x());
        Float gy = dr::select(by, f.y() - 1.0f, f.y());
        Float gz(0.0f);
        Float wt = dr::select(bx, fd.x(), 1.0f - fd.x()) *
                   dr::select(by, fd.y(), 1.0f - fd.y());
        if constexpr (Dim == 3) {
          auto bz = (k & 4) != 0;
          h = mx_hash_int(cell.x() + (k & 1), cell.y() + ((k >> 1) & 1),
                          cell.z() + (k >> 2));
          gz = dr::select(bz, f.z() - 1.0f, f.z());
          wt *= dr::select(bz, fd.z(), 1.0f - fd.z());
        } else {
          h = mx_hash_int(cell.x() + (k & 1), cell.y() + ((k >> 1) & 1));
        }
        if constexpr (std::is_same_v<Out, Float>) {
          acc += wt * mx_gradient_float<Dim>(h, gx, gy, gz);
        } else {
          acc += wt * Out(mx_gradient_float<Dim>(h, gx, gy, gz),
                          mx_gradient_float<Dim>(h >> 8, gx, gy, gz),
                          mx_gradient_float<Dim>(h >> 16, gx, gy, gz));
        }
        k += 1;
      },
      "MaterialX Perlin noise", 1 << Dim));
  return (Dim == 3 ? 0.9820f : 0.6616f) * sum;
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

template <int Dim, typename Out, typename Float>
Out mx_fractal_noise(const Vector<Float, 3>& p, int octaves,
                     const Float& lacunarity, const Float& diminish) {
  if (octaves <= 0) return Out(0.0f);
  if (octaves == 1) return mx_perlin_noise<Dim, Out>(p);
  return mx_fractal_noise<Dim, Out>(p, Float(octaves), lacunarity, diminish,
                                    octaves);
}

// Returns the cell noise of the cell containing the nearest jittered feature
// point where `cell` is set (a scalar for `Out = Float`), and the distances
// to the three nearest feature points elsewhere, matching
// `mx_worley_noise_*()` in `mx_noise.glsl`.
template <int Dim, typename Out, typename Float>
Out mx_worley_noise(const Vector<Float, 3>& p, const Float& jitter,
                    const dr::mask_t<Float>& cell) {
  using Int32 = dr::int32_array_t<Float>;
  using Mask = dr::mask_t<Float>;
  using VecD = Vector<Float, Dim>;
  Vector<Int32, 3> base(0);
  VecD local_pos;
  local_pos.x() = mx_floorfrac(p.x(), base.x());
  local_pos.y() = mx_floorfrac(p.y(), base.y());
  if constexpr (Dim == 3) local_pos.z() = mx_floorfrac(p.z(), base.z());

  size_t w = dr::width(p, jitter);
  Float jit = jitter;
  if constexpr (dr::is_diff_v<Float>) jit = jit.copy();
  Vector<Int32, 3> d0(dr::full<Int32>(-1, w), dr::full<Int32>(-1, w),
                      dr::full<Int32>(Dim == 3 ? -1 : 0, w));
  auto [_, f, min_off, _lp, _jit] = dr::while_loop(
      std::make_tuple(d0, Out(dr::full<Float>(1e6f, w)), dr::zeros<VecD>(w),
                      local_pos, jit),
      [](const Vector<Int32, 3>& d, const auto&...) { return d.x() <= 1; },
      [base](Vector<Int32, 3>& d, Out& f_acc, VecD& min_off_acc, VecD& lp,
             Float& jit_in) {
        VecD rnd = mx_cell_noise<Dim, VecD, Float>(
            base.x() + d.x(), base.y() + d.y(), base.z() + d.z());
        VecD cell_off = (rnd - 0.5f) * jit_in + VecD(dr::head<Dim>(d)) + 0.5f;
        Float d2 = dr::squared_norm(cell_off - lp);
        if constexpr (std::is_same_v<Out, Float>) {
          min_off_acc = dr::select(d2 < f_acc, cell_off, min_off_acc);
          f_acc = dr::minimum(f_acc, d2);
        } else {
          min_off_acc = dr::select(d2 < f_acc.x(), cell_off, min_off_acc);
          f_acc = Out(dr::minimum(f_acc.x(), d2),
                      dr::minimum(f_acc.y(), dr::maximum(d2, f_acc.x())),
                      dr::minimum(f_acc.z(), dr::maximum(d2, f_acc.y())));
        }
        Mask z_wrap = (Dim == 3) ? (d.z() == 1) : Mask(true);
        Mask y_wrap = z_wrap & (d.y() == 1);
        if constexpr (Dim == 3)
          d.z() = dr::select(z_wrap, Int32(-1), d.z() + 1);
        d.y() =
            dr::select(y_wrap, Int32(-1), dr::select(z_wrap, d.y() + 1, d.y()));
        d.x() = dr::select(y_wrap, d.x() + 1, d.x());
      },
      "MaterialX Worley noise", (Dim == 3) ? 27 : 9);

  Int32 cx = base.x() + Int32(dr::floor(min_off.x()));
  Int32 cy = base.y() + Int32(dr::floor(min_off.y()));
  Int32 cz(0);
  if constexpr (Dim == 3) cz = base.z() + Int32(dr::floor(min_off.z()));
  return dr::select(cell, mx_cell_noise<Dim, Out, Float>(cx, cy, cz),
                    dr::safe_sqrt(f));
}

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
