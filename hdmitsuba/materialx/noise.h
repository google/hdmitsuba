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
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>

#include "hdmitsuba/materialx/math.h"

namespace mitsuba::mtlx::detail {

constexpr uint32_t MxHashSeed(uint32_t n) {
  return 0xdeadbeefu + (n << 2u) + 13u;
}

template <typename UInt32>
UInt32 MxRotl32(const UInt32& x, int k) {
  return (x << k) | (x >> (32 - k));
}

template <typename UInt32>
void MxBjmix(UInt32& a, UInt32& b, UInt32& c) {
  a -= c;
  a ^= MxRotl32(c, 4);
  c += b;
  b -= a;
  b ^= MxRotl32(a, 6);
  a += c;
  c -= b;
  c ^= MxRotl32(b, 8);
  b += a;
  a -= c;
  a ^= MxRotl32(c, 16);
  c += b;
  b -= a;
  b ^= MxRotl32(a, 19);
  a += c;
  c -= b;
  c ^= MxRotl32(b, 4);
  b += a;
}

template <typename UInt32>
UInt32 MxBjfinal(UInt32 a, UInt32 b, UInt32 c) {
  c ^= b;
  c -= MxRotl32(b, 14);
  a ^= c;
  a -= MxRotl32(c, 11);
  b ^= a;
  b -= MxRotl32(a, 25);
  c ^= b;
  c -= MxRotl32(b, 16);
  a ^= c;
  a -= MxRotl32(c, 4);
  b ^= a;
  b -= MxRotl32(a, 14);
  c ^= b;
  c -= MxRotl32(b, 24);
  return c;
}

template <typename Float, typename UInt32 = dr::uint32_array_t<Float>>
Float MxBitsTo01(const UInt32& bits) {
  return Float(bits) * Float(1.0 / 0xffffffffu);
}

template <typename Float>
Float MxFade(const Float& t) {
  return t * t * t * dr::fmadd(t, dr::fmadd(t, 6.0f, -15.0f), 10.0f);
}

template <typename Int32, typename... Z>
dr::uint32_array_t<Int32> MxHashInt(const Int32& x, const Int32& y,
                                    const Z&... z) {
  using UInt32 = dr::uint32_array_t<Int32>;
  UInt32 seed(MxHashSeed(2u + static_cast<uint32_t>(sizeof...(Z))));
  return MxBjfinal(seed + UInt32(x), seed + UInt32(y),
                   (seed + ... + UInt32(z)));
}

template <typename Float, typename Int32 = dr::int32_array_t<Float>>
Float MxFloorFrac(const Float& x, Int32& i) {
  Float fl = dr::floor(x);
  i = Int32(fl);
  return x - fl;
}

template <int Dim, typename Float, typename UInt32>
Float MxGradientFloat(const UInt32& hash, const Float& x, const Float& y,
                      const Float& z) {
  Float u, v;
  if constexpr (Dim == 2) {
    auto lt4 = (hash & 4u) == 0u;
    u = dr::select(lt4, x, y);
    v = 2.0f * dr::select(lt4, y, x);
  } else {
    u = dr::select((hash & 8u) == 0u, x, y);
    v = dr::select((hash & 12u) == 0u, y,
                   dr::select((hash & 13u) == 12u, x, z));
  }
  return dr::select((hash & 1u) != 0u, -u, u) +
         dr::select((hash & 2u) != 0u, -v, v);
}

template <int Dim, typename Out, typename Float,
          typename Int32 = dr::int32_array_t<Float>>
Out MxCellNoise(const Int32& ix, const Int32& iy, const Int32& iz) {
  if constexpr (std::is_same_v<Out, Float>) {
    if constexpr (Dim == 3) {
      return MxBitsTo01<Float>(MxHashInt(ix, iy, iz));
    } else {
      return MxBitsTo01<Float>(MxHashInt(ix, iy));
    }
  } else if constexpr (Dim == 2) {
    Out r;
    for (int i = 0; i < int(dr::size_v<Out>); ++i) {
      r[i] = MxBitsTo01<Float>(MxHashInt(ix, iy, Int32(i)));
    }
    return r;
  } else {
    using UInt32 = dr::uint32_array_t<Float>;
    UInt32 a(MxHashSeed(4u));
    UInt32 b = a + UInt32(iy), c = a + UInt32(iz);
    a += UInt32(ix);
    MxBjmix(a, b, c);
    return Out(MxBitsTo01<Float>(MxBjfinal(a, b, c)),
               MxBitsTo01<Float>(MxBjfinal(a + 1u, b, c)),
               MxBitsTo01<Float>(MxBjfinal(a + 2u, b, c)));
  }
}

template <int Dim, typename Out, typename Float>
Out MxCellNoise(const Vector<Float, Dim>& p) {
  using Int32 = dr::int32_array_t<Float>;
  Int32 ix(dr::floor(p.x())), iy(dr::floor(p.y())), iz(0);
  if constexpr (Dim == 3) iz = Int32(dr::floor(p.z()));
  return MxCellNoise<Dim, Out, Float>(ix, iy, iz);
}

template <int Dim, typename Out, typename Float>
Out MxPerlinNoise(const Vector<Float, Dim>& p) {
  using Int32 = dr::int32_array_t<Float>;
  using UInt32 = dr::uint32_array_t<Float>;
  using VecD = Vector<Float, Dim>;
  Vector<Int32, Dim> cell;
  VecD frac;
  for (int i = 0; i < Dim; ++i) frac[i] = MxFloorFrac(p[i], cell[i]);
  VecD fade = MxFade(frac);
  constexpr uint32_t kSeed = MxHashSeed(uint32_t(Dim));
  Vector<UInt32, Dim> hbase = Vector<UInt32, Dim>(cell) + kSeed;
  Vector<UInt32, Dim> hbase1 = hbase + 1u;
  size_t w = dr::width(p);
  Out sum = std::get<1>(WhileLoopWithInvariants(
      std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Out>(w)),
      std::make_tuple(frac, fade),
      [](const Int32& k, const auto&...) { return k < (1 << Dim); },
      [hbase, hbase1](Int32& k, Out& acc, const VecD& f, const VecD& fd) {
        auto bx = (k & 1) != 0, by = (k & 2) != 0;
        UInt32 hx = dr::select(bx, hbase1.x(), hbase.x());
        UInt32 hy = dr::select(by, hbase1.y(), hbase.y());
        UInt32 h;
        Float gx = dr::select(bx, f.x() - 1.0f, f.x());
        Float gy = dr::select(by, f.y() - 1.0f, f.y());
        Float gz(0.0f);
        Float wt = dr::select(bx, fd.x(), 1.0f - fd.x()) *
                   dr::select(by, fd.y(), 1.0f - fd.y());
        if constexpr (Dim == 3) {
          auto bz = (k & 4) != 0;
          h = MxBjfinal(hx, hy, dr::select(bz, hbase1.z(), hbase.z()));
          gz = dr::select(bz, f.z() - 1.0f, f.z());
          wt *= dr::select(bz, fd.z(), 1.0f - fd.z());
        } else {
          h = MxBjfinal(hx, hy, UInt32(kSeed));
        }
        if constexpr (std::is_same_v<Out, Float>) {
          acc = dr::fmadd(wt, MxGradientFloat<Dim>(h, gx, gy, gz), acc);
        } else {
          acc = dr::fmadd(wt,
                          Out(MxGradientFloat<Dim>(h, gx, gy, gz),
                              MxGradientFloat<Dim>(h >> 8, gx, gy, gz),
                              MxGradientFloat<Dim>(h >> 16, gx, gy, gz)),
                          acc);
        }
        k += 1;
      },
      "MaterialX Perlin noise", -1));
  return (Dim == 3 ? 0.9820f : 0.6616f) * sum;
}

// Fractal noise with an octave count that varies per sample.
template <int Dim, typename Out, typename Float>
Out MxFractalNoise(const Vector<Float, Dim>& p, const Float& octaves,
                   const Float& lacunarity, const Float& diminish,
                   int max_octaves) {
  using Int32 = dr::int32_array_t<Float>;
  using VecD = Vector<Float, Dim>;
  // Widen loop-carried state up front so forward-mode AD never sees a width
  // 1 -> W transition mid-body, and copy AD-enabled scalar invariants so
  // symbolic virtual calls wrap outer AD variables in `Gather` nodes.
  size_t w = dr::width(p, octaves, lacunarity, diminish);
  Int32 max_oct = Int32(dr::round(dr::clip(octaves, 0.0f, Float(max_octaves))));
  Float lac = lacunarity, dim = diminish;
  if constexpr (dr::is_diff_v<Float>) {
    if (dr::grad_enabled(lac)) lac = lac.copy();
    if (dr::grad_enabled(dim)) dim = dim.copy();
  }
  return std::get<1>(WhileLoopWithInvariants(
      std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Out>(w),
                      dr::full<Float>(1.0f, w), p + dr::zeros<Float>(w)),
      std::make_tuple(lac, dim),
      [max_oct](const Int32& i, const auto&...) { return i < max_oct; },
      [](Int32& i, Out& result, Float& amplitude, VecD& p_cur,
         const Float& lac_in, const Float& dim_in) {
        result = dr::fmadd(amplitude, MxPerlinNoise<Dim, Out>(p_cur), result);
        amplitude *= dim_in;
        p_cur *= lac_in;
        i += 1;
      },
      "MaterialX fractal noise", max_octaves));
}

template <int Dim, typename Out, typename Float>
Out MxFractalNoise(const Vector<Float, Dim>& p, int octaves,
                   const Float& lacunarity, const Float& diminish) {
  if (octaves <= 0) return Out(0.0f);
  if (octaves == 1) return MxPerlinNoise<Dim, Out>(p);
  return MxFractalNoise<Dim, Out>(p, Float(octaves), lacunarity, diminish,
                                  octaves);
}

// Returns the cell noise of the cell containing the nearest jittered feature
// point where `cell` is set (a scalar for `Out = Float`), and the distances
// to the three nearest feature points elsewhere, matching
// `mx_worley_noise_*()` in `mx_noise.glsl`. When `DistanceOnly` is true, skips
// `min_delta` tracking and the cell-color lookup.
template <int Dim, typename Out, bool DistanceOnly, typename Float>
Out MxWorleyNoise(const Vector<Float, Dim>& p, const Float& jitter,
                  const dr::mask_t<Float>& cell) {
  using Int32 = dr::int32_array_t<Float>;
  using Mask = dr::mask_t<Float>;
  using VecD = Vector<Float, Dim>;
  Vector<Int32, Dim> base;
  VecD local_pos;
  for (int i = 0; i < Dim; ++i) local_pos[i] = MxFloorFrac(p[i], base[i]);

  size_t w = dr::width(p, jitter);
  Float jit = jitter;
  if constexpr (dr::is_diff_v<Float>) {
    if (dr::grad_enabled(jit)) jit = jit.copy();
  }
  VecD bias = dr::fmadd(jit, -0.5f, 0.5f) - local_pos;
  Vector<Int32, Dim> d0(dr::full<Int32>(-1, w));
  auto cond = [](const Vector<Int32, Dim>& d, const auto&...) {
    return d.x() <= 1;
  };
  auto step = [base](Vector<Int32, Dim>& d, Out& f_acc, VecD* min_delta_acc,
                     const VecD& b, const Float& j) {
    Int32 dz(0);
    if constexpr (Dim == 3) dz = base.z() + d.z();
    VecD rnd =
        MxCellNoise<Dim, VecD, Float>(base.x() + d.x(), base.y() + d.y(), dz);
    VecD delta = dr::fmadd(rnd, j, VecD(d) + b);
    Float d2 = dr::squared_norm(delta);
    if constexpr (std::is_same_v<Out, Float>) {
      if constexpr (!DistanceOnly) {
        *min_delta_acc =
            dr::select(d2 < f_acc, dr::detach(delta), *min_delta_acc);
      }
      f_acc = dr::minimum(f_acc, d2);
    } else {
      if constexpr (!DistanceOnly) {
        *min_delta_acc =
            dr::select(d2 < f_acc.x(), dr::detach(delta), *min_delta_acc);
      }
      f_acc = Out(dr::minimum(f_acc.x(), d2),
                  dr::minimum(f_acc.y(), dr::maximum(d2, f_acc.x())),
                  dr::minimum(f_acc.z(), dr::maximum(d2, f_acc.y())));
    }
    if constexpr (Dim == 3) {
      Mask z_wrap = d.z() == 1;
      Mask y_wrap = z_wrap & (d.y() == 1);
      d.z() = dr::select(z_wrap, Int32(-1), d.z() + 1);
      d.y() =
          dr::select(y_wrap, Int32(-1), dr::select(z_wrap, d.y() + 1, d.y()));
      d.x() = dr::select(y_wrap, d.x() + 1, d.x());
    } else {
      Mask y_wrap = d.y() == 1;
      d.y() = dr::select(y_wrap, Int32(-1), d.y() + 1);
      d.x() = dr::select(y_wrap, d.x() + 1, d.x());
    }
  };
  constexpr long long kIters = (Dim == 3) ? 27 : 9;
  if constexpr (DistanceOnly) {
    (void)cell;
    auto [_, f] = WhileLoopWithInvariants(
        std::make_tuple(d0, Out(dr::full<Float>(1e6f, w))),
        std::make_tuple(bias, jit), cond,
        [step](Vector<Int32, Dim>& d, Out& f_acc, const VecD& b,
               const Float& j) { step(d, f_acc, nullptr, b, j); },
        "MaterialX Worley noise", kIters);
    return dr::safe_sqrt(f);
  } else {
    auto [_, f, min_delta] = WhileLoopWithInvariants(
        std::make_tuple(d0, Out(dr::full<Float>(1e6f, w)), dr::zeros<VecD>(w)),
        std::make_tuple(bias, jit), cond,
        [step](Vector<Int32, Dim>& d, Out& f_acc, VecD& min_delta_acc,
               const VecD& b,
               const Float& j) { step(d, f_acc, &min_delta_acc, b, j); },
        "MaterialX Worley noise", kIters);
    VecD min_off = min_delta + dr::detach(local_pos);
    Int32 cx = base.x() + Int32(dr::floor(min_off.x()));
    Int32 cy = base.y() + Int32(dr::floor(min_off.y()));
    Int32 cz(0);
    if constexpr (Dim == 3) cz = base.z() + Int32(dr::floor(min_off.z()));
    return dr::select(cell, MxCellNoise<Dim, Out, Float>(cx, cy, cz),
                      dr::safe_sqrt(f));
  }
}

}  // namespace mitsuba::mtlx::detail
