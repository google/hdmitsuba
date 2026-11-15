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

#include "hdmitsuba/materialx/evaluator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/inlined_vector.h>
#include <absl/synchronization/mutex.h>
#include <drjit/color.h>
#include <drjit/matrix.h>
#include <drjit/while_loop.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {
namespace {

template <typename Vec>
MI_INLINE Vec safe_normalize(const Vec& v, const Vec& fallback = Vec(0.0f)) {
  auto len2 = dr::squared_norm(v);
  auto valid = len2 > 0.f;
  return dr::select(valid, v * dr::rsqrt(dr::select(valid, len2, 1.0f)),
                    fallback);
}

template <size_t N, typename Float>
void eval_matrix_op(Op op, uint16_t imm, Vector<Float, 4>*& sp,
                    Vector<Float, 4>* tmps) {
  using Matrix = dr::Matrix<Float, N>;
  using Row = dr::Array<Float, N>;
  auto pop = [&]() {
    sp -= N;
    Matrix m;
    for (size_t i = 0; i < N; ++i) m.entry(i) = Row(dr::head<N>(sp[i]));
    return m;
  };
  Matrix result;
  if (op == Op::MatMul) {
    Matrix b = pop();
    result = pop() * b;
  } else {
    Matrix m = pop();
    if (op == Op::MatDet) {
      *sp++ = Vector<Float, 4>(dr::det(m));
      return;
    }
    if (op == Op::MatTranspose) {
      result = dr::transpose(m);
    } else {
      // Like division, the inverse of a singular matrix is zero.
      auto valid = dr::det(m) != 0.f;
      result = dr::select(
          valid, dr::inverse(dr::select(valid, m, dr::identity<Matrix>())),
          dr::zeros<Matrix>());
    }
  }
  for (size_t i = 0; i < N; ++i) {
    Vector<Float, 4> row(0.f);
    for (size_t j = 0; j < N; ++j) row.entry(j) = result(i, j);
    tmps[imm + i] = row;
  }
}

// Port of the MaterialX noise functions in
// `libraries/stdlib/genglsl/lib/mx_noise.glsl` (Apache-2.0, Contributors to the
// MaterialX Project), vectorized with Dr.Jit.

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
                     const Float& lacunarity, const Float& diminish) {
  using Int32 = dr::int32_array_t<Float>;
  using Vec = Vector<Float, 3>;
  // Widen loop-carried state up front so forward-mode AD never sees a width
  // 1 -> W transition mid-body, and copy invariant scalar parameters so
  // symbolic virtual calls wrap outer AD variables in `Gather` nodes.
  size_t w = dr::width(p, octaves, lacunarity, diminish);
  Int32 max_oct =
      Int32(dr::round(dr::clip(octaves, 0.0f, Float(kMaxFractalOctaves))));
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
      "MaterialX fractal noise", kMaxFractalOctaves));
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

template <int Dim, typename Out, typename Float>
void eval_noise_op(Op op, uint16_t imm, Vector<Float, 4>*& sp) {
  Out res;
  if (op == Op::CellNoise) {
    res = mx_cell_noise<Dim, Out, Float>(dr::head<3>(sp[-1]));
  } else if (op == Op::Noise) {
    res = mx_perlin_noise<Dim, Out, Float>(dr::head<3>(sp[-1]));
  } else if (op == Op::Worley) {
    sp -= 2;
    res = mx_worley_noise<Dim, Out, Float>(dr::head<3>(sp[-1]), sp[0].x(),
                                           dr::round(sp[1].x()) == 1.f);
  } else {
    uint16_t oct = imm >> kFractalOctavesShift;
    sp -= oct == kFractalDynamicOctaves ? 3 : 2;
    res = oct == kFractalDynamicOctaves
              ? mx_fractal_noise<Dim, Out, Float>(
                    dr::head<3>(sp[-1]), sp[2].x(), sp[0].x(), sp[1].x())
              : mx_fractal_noise<Dim, Out, Float>(dr::head<3>(sp[-1]), int(oct),
                                                  sp[0].x(), sp[1].x());
  }
  if constexpr (std::is_same_v<Out, Float>) {
    sp[-1] = Vector<Float, 4>(res);
  } else {
    sp[-1] = Vector<Float, 4>(res.x(), res.y(), res.z(), 0.f);
  }
}

}  // namespace

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::compile(const Properties& props) {
  m_inputs.clear();
  for (size_t i = 0;; ++i) {
    std::string name = "in" + std::to_string(i);
    if (!props.has_property(name)) break;
    m_inputs.push_back(props.get_texture<Texture>(name));
  }

  m_prog = props.get_any<MtlxProgram>("program");
  m_opaques.clear();
  m_opaques.reserve(m_prog.opaques.size());
  for (double v : m_prog.opaques) {
    m_opaques.push_back(dr::opaque<Float>(static_cast<ScalarFloat>(v)));
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse(TraversalCallback* cb) {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    cb->put("in" + std::to_string(i), m_inputs[i], ParamFlags::Differentiable);
  }
  for (size_t i = 0; i < m_opaques.size(); ++i) {
    cb->put("param" + std::to_string(i), m_opaques[i],
            ParamFlags::Differentiable);
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::parameters_changed() {
  if constexpr (dr::is_jit_v<Float>) {
    bool scheduled = false;
    for (Float& p : m_opaques) {
      if (unlikely(p.size() != 1)) {
        Throw("MtlxExprTexture: parameter must have size 1 (got %zu)!",
              p.size());
      }
      scheduled |= dr::detail::schedule_force(p);
    }
    if (scheduled) dr::eval();
  }
}

template <typename Float, typename Spectrum>
std::string MtlxEvaluator<Float, Spectrum>::to_string() const {
  std::ostringstream oss;
  oss << "  code_size = " << m_prog.code.size() << ",\n";
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    oss << "  in" << i << " = " << string::indent(m_inputs[i]) << ",\n";
  }
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::is_spatially_varying() const {
  for (Inst inst : m_prog.code) {
    if (inst.op == Op::SampleInput) {
      if (m_inputs[inst.imm]->is_spatially_varying()) return true;
    } else if (inst.op >= Op::SampleInputUV && inst.op <= Op::GeomIncidentDir) {
      return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::elementwise(Op op, const Val* sp, int c) {
  const Float& a0 = sp[-1][c];
  switch (op) {
    case Op::Neg:
      return -a0;
    case Op::Abs:
      return dr::abs(a0);
    case Op::Sin:
      return dr::sin(a0);
    case Op::Cos:
      return dr::cos(a0);
    case Op::Tan:
      return dr::tan(a0);
    case Op::Exp:
      return dr::exp(a0);
    case Op::Round: {
      // Like OSL, rounds halfway cases away from zero.
      Float t = dr::trunc(a0);
      return dr::select(dr::abs(a0 - t) >= 0.5f, t + dr::copysign(1.f, a0), t);
    }
    case Op::Floor:
      return dr::floor(a0);
    case Op::Ceil:
      return dr::ceil(a0);
    case Op::Sqrt:
      return dr::safe_sqrt(a0);
    case Op::Log: {
      Mask pos = a0 > 0.f;
      return dr::select(pos, dr::log(dr::select(pos, a0, 1.f)), 0.f);
    }
    case Op::Asin:
      return dr::safe_asin(a0);
    case Op::Acos:
      return dr::safe_acos(a0);
    case Op::Fract:
      return a0 - dr::floor(a0);
    case Op::Add:
      return a0 + sp[0][c];
    case Op::Sub:
      return a0 - sp[0][c];
    case Op::Mul:
      return a0 * sp[0][c];
    case Op::Div: {
      const Float& a1 = sp[0][c];
      Mask nz = a1 != 0.f;
      return dr::select(nz, a0 / dr::select(nz, a1, 1.f), 0.f);
    }
    case Op::Lt:
      return dr::select(a0 < sp[0][c], 1.f, 0.f);
    case Op::Le:
      return dr::select(a0 <= sp[0][c], 1.f, 0.f);
    case Op::Eq:
      return dr::select(a0 == sp[0][c], 1.f, 0.f);
    case Op::Min:
      return dr::minimum(a0, sp[0][c]);
    case Op::Max:
      return dr::maximum(a0, sp[0][c]);
    case Op::Pow: {
      // Like OSL's `safe_pow()`, raises negative bases to integer powers only,
      // and returns 0 for other negative bases.
      const Float& y = sp[0][c];
      Float mag = dr::pow(dr::select(a0 != 0.f, dr::abs(a0), 1.f), y);
      Float neg = dr::select(
          dr::floor(y) == y,
          dr::select(dr::floor(0.5f * y) * 2.f != y, -mag, mag), 0.f);
      return dr::select(
          y == 0.f, 1.f,
          dr::select(a0 == 0.f, 0.f, dr::select(a0 > 0.f, mag, neg)));
    }
    case Op::Atan2: {
      Mask nz = (a0 != 0.f) || (sp[0][c] != 0.f);
      return dr::select(nz, dr::atan2(a0, dr::select(nz, sp[0][c], 1.f)), 0.f);
    }
    case Op::Select:
      return dr::select(a0 != 0.f, sp[0][c], sp[1][c]);
    case Op::Mix:
      return dr::lerp(sp[0][c], a0, sp[1][c]);
    case Op::Clip:
      return dr::clip(a0, sp[0][c], sp[1][c]);
    default:
      return 0.f;
  }
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval(const SurfaceInteraction3f& si,
                                     Mask active) const {
  absl::InlinedVector<Val, 64> mem(m_prog.num_tmps + m_prog.stack_size);
  Val* tmps = mem.data();
  Val* sp = tmps + m_prog.num_tmps;
  std::optional<SurfaceInteraction3f> si_uv;

  auto xy = [](const Val& v) -> Vector2f { return dr::head<2>(v); };
  auto xyz = [](const Val& v) -> Vector3f { return dr::head<3>(v); };
  auto val3 = [](const auto& v, const Float& w = 0.f) {
    return Val(v.x(), v.y(), v.z(), w);
  };

  auto eval_input = [&](uint16_t k, bool mono,
                        const SurfaceInteraction3f& si_q) -> Val {
    return mono ? Val(m_inputs[k]->eval_1(si_q, active))
                : val3(m_inputs[k]->eval_3(si_q, active));
  };

  for (Inst inst : m_prog.code) {
    uint16_t imm = inst.imm;
    if (int n = elementwise_arity(inst.op)) {
      sp -= n - 1;
      Val res(0.f);
      for (int c = 0; c < inst.channels; ++c) {
        res[c] = elementwise(inst.op, sp, c);
      }
      sp[-1] = inst.channels == 1 ? Val(res.x()) : res;
      continue;
    }
    switch (inst.op) {
      case Op::Const:
        *sp++ = Val(static_cast<ScalarFloat>(m_prog.consts[imm]));
        break;
      case Op::Opaque:
        *sp++ = Val(m_opaques[imm]);
        break;
      case Op::LoadTmp:
        *sp++ = tmps[imm];
        break;
      case Op::StoreTmp:
        tmps[imm] = *--sp;
        break;
      case Op::SampleInput:
        *sp++ = eval_input(imm, inst.channels == 1, si);
        break;
      case Op::SampleInputUV: {
        --sp;
        if (!si_uv) si_uv.emplace(dr::zeros<SurfaceInteraction3f>());
        si_uv->uv = Point2f(sp[-1].x(), sp[0].x());
        sp[-1] = eval_input(imm, inst.channels == 1, *si_uv);
        break;
      }
      case Op::Attr: {
        if constexpr (!dr::is_jit_v<Float>) {
          if (!active || !si.shape) break;
        }
        const std::string& attr = m_prog.attr_names[imm];
        Mask has = si.shape->has_attribute(attr, active) & active;
        if (dr::none_or<false>(has)) break;
        // A fourth channel is that of the default.
        sp[-1] = dr::select(
            has,
            inst.channels == 1
                ? Val(si.shape->eval_attribute_1(attr, si, has))
                : val3(si.shape->eval_attribute_3(attr, si, has), sp[-1].w()),
            sp[-1]);
        break;
      }

      case Op::GeomUV:
        *sp++ = Val(si.uv.x(), 1.f - si.uv.y(), 0.f, 0.f);
        break;
      case Op::GeomPosWorld:
        *sp++ = val3(si.p);
        break;
      case Op::GeomNormalWorld:
        *sp++ = val3(si.sh_frame.n);
        break;
      case Op::GeomTangentWorld:
      case Op::GeomBitangentWorld: {
        Vector3f t = safe_normalize(
            Vector3f(si.dp_du -
                     si.sh_frame.n * dr::dot(si.sh_frame.n, si.dp_du)),
            si.sh_frame.s);
        if (inst.op == Op::GeomTangentWorld) {
          *sp++ = val3(t);
        } else {
          Vector3f b = dr::cross(si.sh_frame.n, t);
          // Mitsuba's V is flipped relative to MaterialX, so dp/dv_mtlx =
          // -dp_dv.
          *sp++ = val3(dr::select(dr::dot(b, si.dp_dv) > 0.f, -b, b));
        }
        break;
      }
      case Op::GeomIncidentDir:
        *sp++ = val3(-si.to_world(si.wi));
        break;

      case Op::Rgb:
      case Op::Vec2:
        sp -= inst.op == Op::Rgb ? 2 : 1;
        sp[-1] = Val(sp[-1].x(), sp[0].x(),
                     inst.op == Op::Rgb ? sp[1].x() : 0.f, 0.f);
        break;
      case Op::Vec4:
        --sp;
        sp[-1].w() = sp[0].x();
        break;
      case Op::Chan:
        sp[-1] = Val(sp[-1].entry(imm));
        break;
      case Op::Dot:
        --sp;
        sp[-1] = Val(imm == 2   ? dr::dot(xy(sp[-1]), xy(sp[0]))
                     : imm == 4 ? dr::dot(sp[-1], sp[0])
                                : dr::dot(xyz(sp[-1]), xyz(sp[0])));
        break;
      case Op::Cross:
        --sp;
        sp[-1] = val3(dr::cross(xyz(sp[-1]), xyz(sp[0])));
        break;
      case Op::Normalize:
        sp[-1] =
            imm == 2
                ? Val(dr::concat(safe_normalize(xy(sp[-1])), Vector2f(0.f)))
            : imm == 4 ? safe_normalize(sp[-1])
                       : val3(safe_normalize(xyz(sp[-1])));
        break;
      case Op::Rotate2D: {
        --sp;
        Val v = sp[-1];
        auto [s, c] = dr::sincos(dr::deg_to_rad(sp[0].x()));
        sp[-1] = Val(v.x() * c + v.y() * s, v.y() * c - v.x() * s, 0.f, 0.f);
        break;
      }
      case Op::Rotate3D: {
        sp -= 2;
        Vector3f v = xyz(sp[-1]), axis = safe_normalize(xyz(sp[1]));
        auto [s, c] = dr::sincos(-dr::deg_to_rad(sp[0].x()));
        sp[-1] = val3(v * c + dr::cross(axis, v) * s +
                      axis * dr::dot(axis, v) * (1.f - c));
        break;
      }
      case Op::NormalMap: {
        sp -= 4;
        Vector3f val = xyz(sp[-1]), tangent = xyz(sp[2]),
                 N = safe_normalize(xyz(sp[1]), Vector3f(si.sh_frame.n)),
                 T = safe_normalize(tangent - N * dr::dot(N, tangent),
                                    coordinate_system(N).first),
                 B = dr::cross(N, T);
        B = dr::select(dr::dot(xyz(sp[3]), B) < 0.f, -B, B);
        val = dr::select(dr::squared_norm(val) == 0.f, Vector3f(0.f, 0.f, 1.f),
                         val * 2.f - 1.f);
        sp[-1] = val3(safe_normalize(
            T * (val.x() * sp[0].x()) + B * (val.y() * sp[0].y()) + N * val.z(),
            N));
        break;
      }
      // The color conversions keep a fourth (alpha) channel.
      case Op::RgbToHsv:
        sp[-1] = val3(dr::rgb_to_hsv(xyz(sp[-1])), sp[-1].w());
        break;
      case Op::HsvToRgb:
        sp[-1] = val3(dr::hsv_to_rgb(xyz(sp[-1])), sp[-1].w());
        break;
      case Op::Blackbody: {
        // Port of MaterialX `mx_blackbody.glsl` (Apache-2.0, Contributors
        // to the MaterialX Project).
        Float temp = dr::clip(sp[-1].x(), 800.f, 25000.f);
        Float t = 1000.f / temp;
        Float xc = dr::select(
            temp < 4000.f,
            dr::horner(t, 0.179910f, 0.8776956f, -0.2343580f, -0.2661239f),
            dr::horner(t, 0.240390f, 0.2226347f, 2.1070379f, -3.0258469f));
        Float yc =
            dr::select(temp < 2222.f,
                       dr::horner(xc, -0.20219683f, 2.18555832f, -1.34811020f,
                                  -1.1063814f),
                       dr::select(temp < 4000.f,
                                  dr::horner(xc, -0.16748867f, 2.09137015f,
                                             -1.37418593f, -0.9549476f),
                                  dr::horner(xc, -0.37001483f, 3.75112997f,
                                             -5.87338670f, 3.0817580f)));
        Color3f rgb = xyz_to_srgb(Color3f(xc / yc, 1.f, (1.f - xc - yc) / yc));
        sp[-1] = val3(dr::maximum(rgb, 0.f));
        break;
      }
      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal: {
        bool is_3d = imm & kNoise3D;
        if (imm & kNoiseVec3) {
          if (is_3d) {
            eval_noise_op<3, Vector3f>(inst.op, imm, sp);
          } else {
            eval_noise_op<2, Vector3f>(inst.op, imm, sp);
          }
        } else {
          if (is_3d) {
            eval_noise_op<3, Float>(inst.op, imm, sp);
          } else {
            eval_noise_op<2, Float>(inst.op, imm, sp);
          }
        }
        break;
      }
      case Op::MatDet:
      case Op::MatTranspose:
      case Op::MatInverse:
      case Op::MatMul:
        if (inst.channels == 3) {
          eval_matrix_op<3>(inst.op, imm, sp, tmps);
        } else {
          eval_matrix_op<4>(inst.op, imm, sp, tmps);
        }
        break;
      default:  // Element-wise ops are handled above.
        break;
    }
  }

  return sp[-1];
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::eval_1(const SurfaceInteraction3f& si,
                                             Mask active) const {
  Color3f c = eval_3(si, active);
  return m_prog.result_uniform ? c.x() : luminance(c);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Color3f
MtlxEvaluator<Float, Spectrum>::eval_3(const SurfaceInteraction3f& si,
                                       Mask active) const {
  MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
  Val v = eval(si, active);
  return Color3f(v.x(), v.y(), v.z());
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse_cb(
    void* payload, const drjit::TraverseVisitor& cb) {
  DR_TRAVERSE_MEMBER(m_inputs)
  DR_TRAVERSE_MEMBER(m_opaques)
  if (!cb.state) return;
  auto put_span = [&](const auto& vec, const char* name) {
    size_t n = vec.size();
    drjit::detail::traverse_state(payload, cb, &n, sizeof(n), name);
    if (n > 0) {
      drjit::detail::traverse_state(payload, cb, vec.data(), n * sizeof(vec[0]),
                                    name);
    }
  };
  put_span(m_prog.code, "code");
  put_span(m_prog.consts, "consts");
  size_t n_attrs = m_prog.attr_names.size();
  DR_TRAVERSE_STATE(n_attrs)
  for (const std::string& s : m_prog.attr_names) {
    put_span(s, "attr_names");
  }
  DR_TRAVERSE_STATE(m_prog.result_uniform)
  DR_TRAVERSE_STATE(m_prog.unbounded)
}

namespace {

// Device-resident `data/srgb.coeff` lookup tables shared via a weak cache so
// `MtlxExprTexture::traverse_cb()` can expose `data` and `scale` to
// `@dr.freeze` and `FrozenRender` (`mitsuba::SRGBModel` keeps its buffers in a
// translation-unit static).
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

// Mirrors `mitsuba::SRGBModel::fetch_jit()` (`mitsuba/src/render/srgb.cpp`),
// reading from the traversable `lut` of `MtlxExprTexture`.
template <typename Float, typename Spectrum,
          typename Float32 = dr::float32_array_t<dr::detached_t<Float>>>
MI_INLINE dr::Array<Float, 3> fetch_srgb(const Color<Float, 3>& color,
                                         const SrgbLut<Float32>* lut) {
  if constexpr (!dr::is_jit_v<Float>) {
    (void)lut;
    return SRGBModel<Float, Spectrum>::fetch(color);
  } else {
    using Array3f = dr::Array<Float, 3>;
    using UInt32 = dr::uint32_array_t<Float>;
    using Mask = dr::mask_t<Float>;

    uint32_t res = lut->res;
    float res_f = static_cast<float>(res);
    Color<Float, 3> c = dr::clip(color, 0.f, 1.f);

    Mask g_ge_r = c.g() >= c.r();
    Float rg = dr::select(g_ge_r, c.g(), c.r());
    Mask b_largest = c.b() >= rg;

    UInt32 i = dr::select(b_largest, 2u, dr::select(g_ge_r, 1u, 0u));
    Float z = dr::select(b_largest, c.b(), rg),
          x = dr::select(b_largest, c.r(), dr::select(g_ge_r, c.b(), c.g())),
          y = dr::select(b_largest, c.g(), dr::select(g_ge_r, c.r(), c.b()));

    Float scale = (res_f - 1.f) / dr::maximum(z, 1e-9f);
    x *= scale;
    y *= scale;

    auto scale_at = [&](const UInt32& idx) {
      return Float(dr::gather<Float32>(lut->scale, dr::detach<false>(idx)));
    };
    UInt32 zi =
        dr::binary_search<UInt32>(
            1, res - 1, [&](const UInt32& idx) { return scale_at(idx) <= z; }) -
        1;
    Float z0 = scale_at(zi), z1 = scale_at(zi + 1);

    UInt32 xi = dr::minimum(UInt32(x), res - 2),
           yi = dr::minimum(UInt32(y), res - 2);
    Float wx = x - Float(xi), wy = y - Float(yi), wz = (z - z0) / (z1 - z0);

    UInt32 index = ((i * res + zi) * res + yi) * res + xi, dy = res,
           dz = res * res;
    auto fetch = [&](const UInt32& idx) {
      return Array3f(
          dr::gather<dr::Array<Float32, 3>>(lut->data, dr::detach<false>(idx)));
    };
    auto lerp_x = [&](const UInt32& idx) {
      return dr::lerp(fetch(idx), fetch(idx + 1), wx);
    };
    Array3f result =
        dr::lerp(dr::lerp(lerp_x(index), lerp_x(index + dy), wy),
                 dr::lerp(lerp_x(index + dz), lerp_x(index + dz + dy), wy), wz);

    Float v = c.r();
    Mask mono = (c.g() == v) && (c.b() == v);
    Float mono_c2 = dr::select(
        v <= 0.f, -8192.f,
        dr::select(v >= 1.f, 8192.f, (v - .5f) * dr::rsqrt(v * (1.f - v))));
    Array3f mono_coeff(0.f, 0.f, mono_c2);
    if constexpr (dr::is_diff_v<Float>) {
      mono_coeff = dr::replace_grad(mono_coeff, result);
    }
    return dr::select(mono, mono_coeff, result);
  }
}

// Spectral upsampling of an RGB color computed at render time. `SRGBModel`
// upsamples colors in [0, 1], so an unbounded color `c` maps to
// `2 * max(c) * S(c / (2 * max(c)))`, like `D65Spectrum::normalization` for
// constant colors: the brightest channel sits at 0.5 inside the smooth
// interior of the model, and the mapping remains homogeneous of degree 1 for
// unbounded HDR emission.
template <typename Float, typename Spectrum, typename Float32>
MI_INLINE unpolarized_spectrum_t<Spectrum> upsample_rgb(
    const Color<Float, 3>& c, bool unbounded,
    const wavelength_t<Spectrum>& wavelengths, const SrgbLut<Float32>* lut) {
  if (!unbounded) {
    return srgb_model_eval<unpolarized_spectrum_t<Spectrum>>(
        fetch_srgb<Float, Spectrum>(c, lut), wavelengths);
  }
  Color<Float, 3> rgb = dr::maximum(c, 0.f);
  Float z = dr::max(rgb) * 2.f;
  auto active = z > 0.f;
  return srgb_model_eval<unpolarized_spectrum_t<Spectrum>>(
             fetch_srgb<Float, Spectrum>(
                 dr::select(active, rgb / dr::select(active, z, 1.f),
                            Color<Float, 3>(0.5f)),
                 lut),
             wavelengths) *
         z;
}

/**!

.. _texture-materialx_expr:

MaterialX Expression (:monosp:`materialx_expr`)
-------------------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxProgram`` bytecode and constant/opaque
     tables.

 * - in0, in1, ...
   - |texture|
   - Nested input textures referenced by ``Op::SampleInput`` and
     ``Op::SampleInputUV``.
   - |exposed|, |differentiable|

 * - param0, param1, ...
   - |float|
   - Differentiable opaque scalar parameters exposed through traversal.
   - |exposed|, |differentiable|

Evaluates a compiled MaterialX expression DAG over surface interactions,
supporting procedural noises, 2D texture lookups, geometric intrinsics,
and mesh attributes.

*/
template <typename Float, typename Spectrum>
class MtlxExprTexture final : public Texture<Float, Spectrum> {
 public:
  MI_IMPORT_TYPES(Texture)
  using SrgbFloat32 = dr::float32_array_t<dr::detached_t<Float>>;

  explicit MtlxExprTexture(const Properties& props) : Texture(props) {
    m_eval.compile(props);
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      if (!m_eval.program().result_uniform) {
        m_srgb_lut = get_srgb_lut<SrgbFloat32>();
      }
    }
  }

  void traverse(TraversalCallback* cb) override { m_eval.traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    m_eval.parameters_changed();
  }

  UnpolarizedSpectrum eval(const SurfaceInteraction3f& si,
                           Mask active) const override {
    if constexpr (is_monochromatic_v<Spectrum>) {
      return eval_1(si, active);
    } else if constexpr (is_rgb_v<Spectrum>) {
      return eval_3(si, active);
    } else {
      Color3f c = eval_3(si, active);
      const MtlxProgram& prog = m_eval.program();
      if (prog.result_uniform) {
        return UnpolarizedSpectrum(prog.unbounded ? dr::maximum(c.x(), 0.f)
                                                  : dr::clip(c.x(), 0.f, 1.f));
      }
      return upsample_rgb<Float, Spectrum>(c, prog.unbounded, si.wavelengths,
                                           m_srgb_lut.get());
    }
  }

  std::pair<Wavelength, UnpolarizedSpectrum> sample_spectrum(
      const SurfaceInteraction3f& si, [[maybe_unused]] const Wavelength& sample,
      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);
    if constexpr (is_spectral_v<Spectrum>) {
      SurfaceInteraction3f si2(si);
      si2.wavelengths = MI_CIE_MIN + (MI_CIE_MAX - MI_CIE_MIN) * sample;
      return {si2.wavelengths,
              eval(si2, active) * ScalarFloat(MI_CIE_MAX - MI_CIE_MIN)};
    } else {
      return {dr::empty<Wavelength>(), eval(si, active)};
    }
  }

  Wavelength pdf_spectrum(const SurfaceInteraction3f&, Mask) const override {
    if constexpr (is_spectral_v<Spectrum>) {
      return dr::rcp(ScalarFloat(MI_CIE_MAX - MI_CIE_MIN));
    } else {
      return dr::empty<Wavelength>();
    }
  }

  Float eval_1(const SurfaceInteraction3f& si, Mask active) const override {
    return m_eval.eval_1(si, active);
  }

  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const override {
    return m_eval.eval_3(si, active);
  }

  bool is_spatially_varying() const override {
    return m_eval.is_spatially_varying();
  }

  std::string to_string() const override {
    return "MtlxExprTexture[\n" + m_eval.to_string() + "]";
  }

  MI_DECLARE_CLASS(MtlxExprTexture)

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb) override {
    if (cb.role != drjit::TraverseRole::Freeze &&
        cb.role != drjit::TraverseRole::Children) {
      return;
    }
    Texture::traverse_cb(payload, cb);
    m_eval.traverse_cb(payload, cb);
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      if (m_srgb_lut) {
        DR_TRAVERSE_MEMBER(m_srgb_lut->scale)
        DR_TRAVERSE_MEMBER(m_srgb_lut->data)
      }
    }
  }

 private:
  MtlxEvaluator<Float, Spectrum> m_eval;
  std::shared_ptr<SrgbLut<SrgbFloat32>> m_srgb_lut;
};

}  // namespace

MI_INSTANTIATE_CLASS(MtlxEvaluator)

void RegisterMaterialXExprTexture() {
  static const bool registered = [] {
    auto register_fn = [](auto... args) {
      PluginManager::instance()->register_plugin(args..., nullptr, nullptr);
    };
    MI_REGISTER_PLUGIN(register_fn, "materialx_expr", MtlxExprTexture)
    return true;
  }();
  (void)registered;
}

}  // namespace mtlx
NAMESPACE_END(mitsuba)
