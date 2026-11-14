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

#include "hdmitsuba/materialx/vm.h"

#include <drjit/color.h>
#include <drjit/tensor.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {
namespace {

// Port of the MaterialX noise functions in
// `libraries/stdlib/genglsl/lib/mx_noise.glsl` (Apache-2.0, Contributors to the
// MaterialX Project), vectorized with Dr.Jit.

template <typename UInt32>
MI_INLINE UInt32 mx_rotl32(const UInt32& x, int k) {
  return (x << k) | (x >> (32 - k));
}

template <typename UInt32>
MI_INLINE void mx_bjmix(UInt32& a, UInt32& b, UInt32& c) {
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
MI_INLINE UInt32 mx_bjfinal(UInt32 a, UInt32 b, UInt32 c) {
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
MI_INLINE Float mx_bits_to_01(const UInt32& bits) {
  return Float(bits) * Float(1.0 / static_cast<double>(0xffffffffu));
}

template <typename Float>
MI_INLINE Float mx_fade(const Float& t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

template <typename Int32, typename... Z>
MI_INLINE dr::uint32_array_t<Int32> mx_hash_int(const Int32& x, const Int32& y,
                                                const Z&... z) {
  using UInt32 = dr::uint32_array_t<Int32>;
  UInt32 seed = UInt32(0xdeadbeefu + ((2u + sizeof...(Z)) << 2u) + 13u);
  return mx_bjfinal(seed + UInt32(x), seed + UInt32(y),
                    (seed + ... + UInt32(z)));
}

template <typename Float, typename Int32 = dr::int32_array_t<Float>>
MI_INLINE Float mx_floorfrac(const Float& x, Int32& i) {
  Float fl = dr::floor(x);
  i = Int32(fl);
  return x - fl;
}

template <typename Float, typename UInt32 = dr::uint32_array_t<Float>,
          typename... Z>
MI_INLINE Float mx_gradient_float(const UInt32& hash, const Float& x,
                                  const Float& y, const Z&... z) {
  UInt32 h;
  Float u, v;
  if constexpr (sizeof...(Z) == 0) {
    h = hash & 7u;
    auto lt4 = h < 4u;
    u = dr::select(lt4, x, y);
    v = 2.0f * dr::select(lt4, y, x);
  } else {
    h = hash & 15u;
    u = dr::select(h < 8u, x, y);
    v = dr::select(h < 4u, y, dr::select((h == 12u) | (h == 14u), x, (z, ...)));
  }
  return dr::select((h & 1u) != 0u, -u, u) + dr::select((h & 2u) != 0u, -v, v);
}

template <typename T, typename Float>
MI_INLINE T mx_bilerp(const T& v0, const T& v1, const T& v2, const T& v3,
                      const Float& s, const Float& t) {
  return dr::lerp(dr::lerp(v0, v1, s), dr::lerp(v2, v3, s), t);
}

template <int Dim, typename Out, typename Float,
          typename Int32 = dr::int32_array_t<Float>>
MI_INLINE Out mx_cell_noise(const Int32& ix, const Int32& iy, const Int32& iz) {
  if constexpr (std::is_same_v<Out, Float>) {
    return mx_bits_to_01<Float>((Dim == 3) ? mx_hash_int(ix, iy, iz)
                                           : mx_hash_int(ix, iy));
  } else if constexpr (Dim == 2) {
    return Out(mx_bits_to_01<Float>(mx_hash_int(ix, iy, Int32(0))),
               mx_bits_to_01<Float>(mx_hash_int(ix, iy, Int32(1))),
               mx_bits_to_01<Float>(mx_hash_int(ix, iy, Int32(2))));
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
MI_INLINE Out mx_cell_noise(const Vector<Float, 3>& p) {
  Vector<dr::int32_array_t<Float>, 3> i(dr::floor(p));
  return mx_cell_noise<Dim, Out, Float>(i.x(), i.y(), i.z());
}

template <typename Out, typename UInt32, typename Float, typename... Z>
MI_INLINE Out mx_gradient_eval(const UInt32& h, const Float& x, const Float& y,
                               const Z&... z) {
  if constexpr (std::is_same_v<Out, Float>) {
    return mx_gradient_float(h, x, y, z...);
  } else {
    return Out(mx_gradient_float(h & 0xFFu, x, y, z...),
               mx_gradient_float((h >> 8) & 0xFFu, x, y, z...),
               mx_gradient_float((h >> 16) & 0xFFu, x, y, z...));
  }
}

template <int Dim, typename Out, typename Float>
MI_INLINE Out mx_perlin_noise(const Vector<Float, 3>& p) {
  using Int32 = dr::int32_array_t<Float>;
  Int32 X, Y;
  Float fx = mx_floorfrac(p.x(), X), fy = mx_floorfrac(p.y(), Y);
  Float u = mx_fade(fx), v = mx_fade(fy);
  auto layer = [&](const auto&... z) {
    auto g = [&](int dx, int dy) {
      return mx_gradient_eval<Out>(
          mx_hash_int(dx ? X + 1 : X, dy ? Y + 1 : Y, z.first...),
          dx ? fx - 1.0f : fx, dy ? fy - 1.0f : fy, z.second...);
    };
    return mx_bilerp(g(0, 0), g(1, 0), g(0, 1), g(1, 1), u, v);
  };
  if constexpr (Dim == 2) {
    return 0.6616f * layer();
  } else {
    Int32 Z;
    Float fz = mx_floorfrac(p.z(), Z);
    return 0.9820f * dr::lerp(layer(std::pair{Z, fz}),
                              layer(std::pair{Z + 1, fz - 1.0f}), mx_fade(fz));
  }
}

template <int Dim, typename Out, typename Float>
MI_INLINE Out mx_fractal_noise(Vector<Float, 3> p, const Float& octaves_f,
                               const Float& lacunarity, const Float& diminish) {
  Out result(0.0f);
  Float amplitude(1.0f);
  for (int i = 0; i < 16; ++i) {
    auto active_oct = Float(i) < octaves_f;
    if (!mx_may_be_true(active_oct)) break;
    result += dr::select(active_oct, amplitude * mx_perlin_noise<Dim, Out>(p),
                         Out(0.0f));
    amplitude *= diminish;
    p *= lacunarity;
  }
  return result;
}

// Returns the distances to the three nearest feature points (`style` 0), or the
// cell noise of the cell containing the nearest feature point (`style` 1),
// which is a scalar for `Out = Float`, like `mx_worley_noise_float()`.
template <int Dim, typename Out, typename Float>
MI_INLINE Out mx_worley_noise(const Vector<Float, 3>& p, const Float& jitter,
                              const Float& style_f) {
  constexpr bool kScalar = std::is_same_v<Out, Float>;
  using Int32 = dr::int32_array_t<Float>;
  Int32 X, Y, Z(0);
  Float fx = mx_floorfrac(p.x(), X), fy = mx_floorfrac(p.y(), Y);
  Float fz = (Dim == 3) ? mx_floorfrac(p.z(), Z) : Float(0.0f);
  Vector<Float, 3> f_offset = 0.5f - Vector<Float, 3>(fx, fy, fz);
  Float f1(1e6f), f2(1e6f), f3(1e6f);
  using CellVal =
      std::conditional_t<kScalar, Vector<Int32, 3>, Vector<Float, 3>>;
  CellVal cell_val(0);
  auto use_cell = style_f >= 0.5f;
  bool need_cell = mx_may_be_true(use_cell);
  bool need_dist = mx_may_be_true(!use_cell);
  constexpr int dz_max = (Dim == 3) ? 1 : 0;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -dz_max; dz <= dz_max; ++dz) {
        Int32 ix = X + dx, iy = Y + dy, iz = Z + dz;
        Vector<Float, 3> rnd =
            mx_cell_noise<Dim, Vector<Float, 3>, Float>(ix, iy, iz);
        Vector<Float, 3> d = Vector<Float, 3>(Float(dx), Float(dy), Float(dz)) +
                             (rnd - 0.5f) * jitter + f_offset;
        Float d2 = dr::squared_norm(dr::head<Dim>(d));
        if (need_cell) {
          if constexpr (kScalar) {
            cell_val = dr::select(d2 < f1, CellVal(ix, iy, iz), cell_val);
          } else {
            cell_val = dr::select(d2 < f1, rnd, cell_val);
          }
        }
        if constexpr (!kScalar) {
          if (need_dist) {
            f3 = dr::minimum(f3, dr::maximum(d2, f2));
            f2 = dr::minimum(f2, dr::maximum(d2, f1));
          }
        }
        f1 = dr::minimum(f1, d2);
      }
    }
  }
  if constexpr (kScalar) {
    return dr::select(use_cell,
                      need_cell ? mx_cell_noise<Dim, Float, Float>(
                                      cell_val.x(), cell_val.y(), cell_val.z())
                                : Float(0.0f),
                      need_dist ? dr::safe_sqrt(f1) : Float(0.0f));
  } else {
    return dr::select(use_cell, cell_val,
                      need_dist ? dr::safe_sqrt(Out(f1, f2, f3)) : Out(0.0f));
  }
}

}  // namespace

template <typename Float, typename Spectrum>
void MtlxVM<Float, Spectrum>::compile(const Properties& props) {
  m_inputs.clear();
  for (size_t i = 0;; ++i) {
    std::string name = "in" + std::to_string(i);
    if (!props.has_property(name)) break;
    m_inputs.push_back(props.get_texture<Texture>(name));
  }

  m_prog = props.get_any<MtlxProgram>("program");
  m_params.clear();
  m_params.reserve(m_prog.params.size());
  for (double v : m_prog.params) {
    m_params.push_back(static_cast<ScalarFloat>(v));
  }
  parameters_changed();
}

template <typename Float, typename Spectrum>
void MtlxVM<Float, Spectrum>::traverse(TraversalCallback* cb) {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    cb->put("in" + std::to_string(i), m_inputs[i], ParamFlags::Differentiable);
  }
  for (size_t i = 0; i < m_params.size(); ++i) {
    cb->put("param" + std::to_string(i), m_params[i],
            ParamFlags::Differentiable);
  }
}

template <typename Float, typename Spectrum>
void MtlxVM<Float, Spectrum>::parameters_changed() {
  m_input_mono.resize(m_inputs.size());
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    m_input_mono[i] = is_monochromatic_v<Spectrum> ||
                      is_single_channel_bitmap(m_inputs[i].get());
  }
  for (Float& p : m_params) dr::make_opaque(p);
}

template <typename Float, typename Spectrum>
std::string MtlxVM<Float, Spectrum>::to_string() const {
  std::ostringstream oss;
  oss << "  code_size = " << m_prog.code.size() << ",\n";
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    oss << "  in" << i << " = " << string::indent(m_inputs[i]) << ",\n";
  }
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxVM<Float, Spectrum>::is_spatially_varying() const {
  for (Inst inst : m_prog.code) {
    if (inst.op == Op::SampleInput) {
      if (m_inputs[inst.imm]->is_spatially_varying()) return true;
    } else if (inst.op >= Op::SampleInputUV && inst.op <= Op::Fractal) {
      return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
template <typename T>
T MtlxVM<Float, Spectrum>::elementwise(Op op, const T* a) {
  auto safe_div = [](const T& x, const T& y) {
    auto nz = dr::abs(y) > 1e-20f;
    return dr::select(nz, x / dr::select(nz, y, T(1.0f)), T(0.0f));
  };
  auto to_float = [](const auto& mask) {
    return dr::select(mask, T(1.0f), T(0.0f));
  };
  switch (op) {
    case Op::Add:
      return a[0] + a[1];
    case Op::Sub:
      return a[0] - a[1];
    case Op::Mul:
      return a[0] * a[1];
    case Op::Div:
      return safe_div(a[0], a[1]);
    case Op::Neg:
      return -a[0];
    case Op::Lt:
      return to_float(a[0] < a[1]);
    case Op::Le:
      return to_float(a[0] <= a[1]);
    case Op::Eq:
      return to_float(a[0] == a[1]);
    case Op::Ne:
      return to_float(a[0] != a[1]);
    case Op::Select:
      return dr::select(a[0] != T(0.0f), a[1], a[2]);
    case Op::Sqrt:
      return dr::safe_sqrt(a[0]);
    case Op::Log:
      return dr::log(dr::maximum(a[0], T(1e-20f)));
    case Op::Asin:
      return dr::safe_asin(a[0]);
    case Op::Acos:
      return dr::safe_acos(a[0]);
    case Op::Fract:
      return a[0] - dr::floor(a[0]);
    case Op::Sign:  // Unlike `dr::sign()`, returns 0 for 0.
      return dr::select(a[0] == 0.0f, T(0.0f), dr::sign(a[0]));
    case Op::Min:
      return dr::minimum(a[0], a[1]);
    case Op::Max:
      return dr::maximum(a[0], a[1]);
    case Op::Pow: {
      // Like OSL's `safe_pow()`, raises negative bases to integer powers
      // only, and returns 0 for other negative bases.
      const T& x = a[0];
      const T& y = a[1];
      T mag = dr::pow(dr::select(x != 0.0f, dr::abs(x), T(1.0f)), y);
      T neg = dr::select(dr::floor(y) == y,
                         dr::select(dr::floor(0.5f * y) * 2.0f != y, -mag, mag),
                         T(0.0f));
      return dr::select(
          y == 0.0f, T(1.0f),
          dr::select(x == 0.0f, T(0.0f), dr::select(x > 0.0f, mag, neg)));
    }
    case Op::Atan2:
      return mx_safe_atan2(a[0], a[1]);
    case Op::Modulo:
      return a[0] - a[1] * dr::floor(safe_div(a[0], a[1]));
    case Op::Mix:
      return dr::lerp(a[1], a[0], a[2]);
    case Op::Clip:
      return dr::clip(a[0], a[1], a[2]);
    case Op::Smoothstep: {
      // Like OSL, a step function at `a[0]` if `a[0] >= a[1]`.
      T t = safe_div(a[2] - a[0], a[1] - a[0]);
      return dr::select(
          a[2] < a[0], T(0.0f),
          dr::select(a[2] >= a[1], T(1.0f), t * t * (T(3.0f) - T(2.0f) * t)));
    }
    case Op::Remap:
      return a[3] + safe_div(a[0] - a[1], a[2] - a[1]) * (a[4] - a[3]);
    case Op::ArtisticIorN: {
      T r = dr::clip(a[0], T(0.0f), T(0.99f)), r_sqrt = dr::safe_sqrt(r);
      return dr::lerp((T(1.0f) + r_sqrt) / (T(1.0f) - r_sqrt),
                      (T(1.0f) - r) / (T(1.0f) + r),
                      dr::clip(a[1], T(0.0f), T(1.0f)));
    }
    case Op::ArtisticIorK: {
      T r = dr::clip(a[0], T(0.0f), T(0.99f));
      return dr::safe_sqrt(
          (dr::square(a[1] + T(1.0f)) * r - dr::square(a[1] - T(1.0f))) /
          (T(1.0f) - r));
    }
    case Op::Abs:
      return dr::abs(a[0]);
    case Op::Sin:
      return dr::sin(a[0]);
    case Op::Cos:
      return dr::cos(a[0]);
    case Op::Tan:
      return dr::tan(a[0]);
    case Op::Exp:
      return dr::exp(a[0]);
    case Op::Round:
      return dr::round(a[0]);
    case Op::Floor:
      return dr::floor(a[0]);
    case Op::Ceil:
      return dr::ceil(a[0]);
    default:
      return T(0.0f);
  }
}

template <typename Float, typename Spectrum>
typename MtlxVM<Float, Spectrum>::Val MtlxVM<Float, Spectrum>::eval(
    const SurfaceInteraction3f& si, Mask active) const {
  uint32_t n_inputs = static_cast<uint32_t>(m_inputs.size());
  uint32_t total_slots = n_inputs + m_prog.num_tmps + m_prog.stack_size;
  ScopedArray<Val> mem(alloca(total_slots * sizeof(Val)), total_slots);
  ScopedArray<bool> input_loaded(
      alloca(std::max<uint32_t>(n_inputs, 1) * sizeof(bool)), n_inputs, false);

  Val* tmps = mem + n_inputs;
  Val* sp = mem + n_inputs + m_prog.num_tmps;

  auto eval_input_color3 = [&](uint16_t k,
                               const SurfaceInteraction3f& si_q) -> Val {
    if (m_input_mono[k]) return Val(m_inputs[k]->eval_1(si_q, active));
    if constexpr (is_rgb_v<Spectrum>) {
      return Val(m_inputs[k]->eval(si_q, active));
    } else {
      return Val(m_inputs[k]->eval_3(si_q, active));
    }
  };

  Val tangent_world;
  bool tangent_loaded = false;
  auto get_tangent_world = [&]() -> const Val& {
    if (!tangent_loaded) {
      tangent_world = safe_normalize(
          si.dp_du - si.sh_frame.n * dr::dot(si.sh_frame.n, si.dp_du),
          si.sh_frame.s, 1e-12f);
      tangent_loaded = true;
    }
    return tangent_world;
  };

  for (uint32_t pc = 0; pc < m_prog.code.size(); ++pc) {
    Inst inst = m_prog.code[pc];
    uint16_t imm = inst.imm;
    if (int n = elementwise_arity(inst.op)) {
      sp -= n - 1;
      if (inst.channels == 1) {
        Float args[kMaxElementwiseArity];
        for (int i = 0; i < n; ++i) args[i] = sp[i - 1].x();
        sp[-1] = Val(elementwise(inst.op, args));
      } else if (inst.channels == 2) {
        Vector2f args[kMaxElementwiseArity];
        for (int i = 0; i < n; ++i) args[i] = dr::head<2>(sp[i - 1]);
        Vector2f r = elementwise(inst.op, args);
        sp[-1] = Val(r.x(), r.y(), 0.0f);
      } else {
        sp[-1] = elementwise(inst.op, sp - 1);
      }
      continue;
    }
    switch (inst.op) {
      case Op::Const:
        *sp++ = Val(static_cast<ScalarFloat>(m_prog.consts[imm]));
        break;
      case Op::Param:
        *sp++ = Val(m_params[imm]);
        break;
      case Op::LoadTmp:
        *sp++ = tmps[imm];
        break;
      case Op::StoreTmp:
        tmps[imm] = *--sp;
        break;
      case Op::SampleInput: {
        if (!input_loaded[imm]) {
          mem[imm] = eval_input_color3(imm, si);
          input_loaded[imm] = true;
        }
        *sp++ = inst.channels == 1 ? Val(mem[imm].x()) : mem[imm];
        break;
      }
      case Op::SampleInputUV: {
        --sp;
        SurfaceInteraction3f si_uv = si;
        si_uv.uv = Point2f(sp[-1].x(), sp[0].x());
        Val v = eval_input_color3(imm, si_uv);
        sp[-1] = inst.channels == 1 ? Val(v.x()) : v;
        break;
      }
      case Op::Attr1:
      case Op::Attr3: {
        if constexpr (dr::is_jit_v<Float>) {
          if (!si.shape.index() || jit_var_is_zero_literal(si.shape.index()))
            break;
        } else {
          if (!si.shape) break;
        }
        const std::string& attr = m_prog.attr_names[imm];
        Mask has = inst.channels == 1
                       ? active
                       : si.shape->has_attribute(attr, active) & active;
        Val value = inst.op == Op::Attr1
                        ? Val(si.shape->eval_attribute_1(attr, si, has))
                        : Val(si.shape->eval_attribute_3(attr, si, has));
        sp[-1] = dr::select(has, value, sp[-1]);
        break;
      }

      case Op::GeomUV:
        *sp++ = Val(si.uv.x(), si.uv.y(), 0.0f);
        break;
      case Op::GeomPosWorld:
        *sp++ = Val(si.p);
        break;
      case Op::GeomNormalWorld:
        *sp++ = si.sh_frame.n;
        break;
      case Op::GeomTangentWorld:
      case Op::GeomBitangentWorld: {
        const Val& t = get_tangent_world();
        *sp++ = inst.op == Op::GeomTangentWorld
                    ? t
                    : Val(dr::cross(si.sh_frame.n, t));
        break;
      }
      case Op::GeomIncidentDir:
        *sp++ = -si.to_world(si.wi);
        break;

      case Op::Rgb:
      case Op::Vec2:
        sp -= inst.op == Op::Rgb ? 2 : 1;
        sp[-1] =
            Val(sp[-1].x(), sp[0].x(), inst.op == Op::Rgb ? sp[1].x() : 0.0f);
        break;
      case Op::Chan:
        sp[-1] = Val(sp[-1].entry(imm));
        break;
      case Op::Dot:
        --sp;
        sp[-1] =
            Val(imm == 2 ? dr::dot(dr::head<2>(sp[-1]), dr::head<2>(sp[0]))
                         : dr::dot(sp[-1], sp[0]));
        break;
      case Op::Cross:
        --sp;
        sp[-1] = dr::cross(sp[-1], sp[0]);
        break;
      case Op::Length:
        sp[-1] =
            Val(dr::safe_sqrt(imm == 2 ? dr::squared_norm(dr::head<2>(sp[-1]))
                                       : dr::squared_norm(sp[-1])));
        break;
      case Op::Normalize:
        sp[-1] = safe_normalize(imm == 2 ? Val(sp[-1].x(), sp[-1].y(), 0.0f)
                                         : sp[-1]);
        break;
      case Op::Rotate2D: {
        --sp;
        Val v = sp[-1];
        auto [s, c] = dr::sincos(-dr::deg_to_rad(sp[0].x()));
        sp[-1] = Val(v.x() * c - v.y() * s, v.x() * s + v.y() * c, 0.0f);
        break;
      }
      case Op::Rotate3D: {
        sp -= 2;
        Val v = sp[-1], axis = safe_normalize(sp[1]);
        auto [s, c] = dr::sincos(-dr::deg_to_rad(sp[0].x()));
        sp[-1] = v * c + dr::cross(axis, v) * s +
                 axis * dr::dot(axis, v) * (1.0f - c);
        break;
      }
      case Op::NormalMap: {
        sp -= imm ? 4 : 3;
        Val val = sp[-1], scale = sp[0], N = sp[1],
            T = safe_normalize(sp[2] - N * dr::dot(N, sp[2]), sp[2]);
        Val B = imm ? safe_normalize(sp[3] - N * dr::dot(N, sp[3]) -
                                         T * dr::dot(T, sp[3]),
                                     sp[3])
                    : safe_normalize(dr::cross(N, T));
        val = dr::select(dr::squared_norm(val) == 0.0f, Val(0.0f, 0.0f, 1.0f),
                         val * 2.0f - 1.0f);
        sp[-1] = safe_normalize(
            T * (val.x() * scale.x()) + B * (val.y() * scale.y()) + N * val.z(),
            N);
        break;
      }
      case Op::RgbToHsv:
        sp[-1] = dr::rgb_to_hsv(sp[-1]);
        break;
      case Op::HsvToRgb:
        sp[-1] = dr::hsv_to_rgb(sp[-1]);
        break;
      case Op::Blackbody: {
        // Port of MaterialX `mx_blackbody.glsl` (Apache-2.0, Contributors
        // to the MaterialX Project).
        Float temp = dr::clip(sp[-1].x(), 800.0f, 25000.0f);
        Float t = 1000.0f / temp;
        Float xc = dr::select(
            temp < 4000.0f,
            dr::horner(t, 0.179910f, 0.8776956f, -0.2343580f, -0.2661239f),
            dr::horner(t, 0.240390f, 0.2226347f, 2.1070379f, -3.0258469f));
        Float yc =
            dr::select(temp < 2222.0f,
                       dr::horner(xc, -0.20219683f, 2.18555832f, -1.34811020f,
                                  -1.1063814f),
                       dr::select(temp < 4000.0f,
                                  dr::horner(xc, -0.16748867f, 2.09137015f,
                                             -1.37418593f, -0.9549476f),
                                  dr::horner(xc, -0.37001483f, 3.75112997f,
                                             -5.87338670f, 3.0817580f)));
        Float yc_safe = dr::maximum(yc, 1e-6f);
        Color3f rgb = xyz_to_srgb(
            Color3f(xc / yc_safe, 1.0f, (1.0f - xc - yc) / yc_safe));
        sp[-1] =
            dr::select(yc <= 0.0f, Val(1.0f), dr::maximum(Val(rgb), Val(0.0f)));
        break;
      }
      case Op::RoughnessAnisotropy: {
        --sp;
        Float r2 = dr::clip(dr::square(sp[-1].x()), 1e-5f, 1.0f);
        Float a = sp[0].x();
        Float aspect = dr::sqrt(1.0f - dr::clip(a, 0.0f, 0.98f));
        sp[-1] = dr::select(
            a > 0.0f, Val(dr::minimum(r2 / aspect, 1.0f), r2 * aspect, 0.0f),
            Val(r2, r2, 0.0f));
        break;
      }
      case Op::RoughnessDual: {
        Val r = sp[-1];
        Float ry = dr::select(r.y() < 0.0f, r.x(), r.y());
        sp[-1] = Val(dr::clip(r.x() * r.x(), 1e-5f, 1.0f),
                     dr::clip(ry * ry, 1e-5f, 1.0f), 0.0f);
        break;
      }

      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal: {
        auto run = [&](auto dim_tag, auto out_tag) -> Val {
          constexpr int Dim = decltype(dim_tag)::value;
          using Out = decltype(out_tag);
          if (inst.op == Op::CellNoise) {
            return Val(mx_cell_noise<Dim, Out>(sp[-1]));
          }
          if (inst.op == Op::Noise) {
            sp -= 2;
            if constexpr (std::is_same_v<Out, Val>) {
              return sp[0] * mx_perlin_noise<Dim, Val>(sp[-1]) + sp[1];
            } else {
              return Val(sp[0].x() * mx_perlin_noise<Dim, Float>(sp[-1]) +
                         sp[1].x());
            }
          }
          if (inst.op == Op::Worley) {
            sp -= 2;
            return Val(
                mx_worley_noise<Dim, Out>(sp[-1], sp[0].x(), sp[1].x()));
          }
          sp -= 3;
          return Val(mx_fractal_noise<Dim, Out>(sp[-1], sp[0].x(), sp[1].x(),
                                                sp[2].x()));
        };
        std::integral_constant<int, 2> d2;
        std::integral_constant<int, 3> d3;
        bool is_3d = imm & kNoise3D;
        sp[-1] = (imm & kNoiseVec3)
                     ? (is_3d ? run(d3, Val()) : run(d2, Val()))
                     : (is_3d ? run(d3, Float()) : run(d2, Float()));
        break;
      }

      default:  // Element-wise ops are handled above.
        break;
    }
  }

  return m_prog.code.empty() ? Val(0.0f) : sp[-1];
}

template <typename Float, typename Spectrum>
Float MtlxVM<Float, Spectrum>::eval_1(const SurfaceInteraction3f& si,
                                      Mask active) const {
  Color3f c = eval_3(si, active);
  return m_prog.result_uniform ? c.x() : luminance(c);
}

template <typename Float, typename Spectrum>
typename MtlxVM<Float, Spectrum>::Color3f MtlxVM<Float, Spectrum>::eval_3(
    const SurfaceInteraction3f& si, Mask active) const {
  MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
  return Color3f(eval(si, active));
}

template <typename Float, typename Spectrum>
MaterialXCopy MtlxVM<Float, Spectrum>::copy_parameters(Object* dst,
                                                       Object* src) {
  if (dst == src) return MaterialXCopy::kUnchanged;
  if (!dst || !src || dst->class_name() != src->class_name()) {
    return MaterialXCopy::kIncompatible;
  }
  auto get_vm = [](Object* obj) -> MtlxVM<Float, Spectrum>* {
    if (obj->class_name() == "D65Spectrum") {
      struct NestedFinder final : TraversalCallback {
        Object* nested = nullptr;
        void put_value(std::string_view, void*, uint32_t,
                       const std::type_info&) override {}
        void put_object(std::string_view name, Object* child,
                        uint32_t) override {
          if (name == "nested_texture") nested = child;
        }
      } cb;
      obj->traverse(&cb);
      obj = cb.nested;
    }
    auto* mtlx_obj = dynamic_cast<MtlxObject<Float, Spectrum>*>(obj);
    return mtlx_obj ? &mtlx_obj->vm() : nullptr;
  };
  MtlxVM<Float, Spectrum>* dst_vm = get_vm(dst);
  MtlxVM<Float, Spectrum>* src_vm = get_vm(src);
  if (!dst_vm || !src_vm || dst_vm->inputs() != src_vm->inputs() ||
      !dst_vm->program().same_structure(src_vm->program())) {
    return MaterialXCopy::kIncompatible;
  }
  if (dst_vm->program().params == src_vm->program().params) {
    return MaterialXCopy::kUnchanged;
  }
  dst_vm->program().params = src_vm->program().params;
  dst_vm->params() = src_vm->params();
  return MaterialXCopy::kCopied;
}

template <typename Float, typename Spectrum>
bool MtlxVM<Float, Spectrum>::is_single_channel_bitmap(Texture* texture) {
  struct Callback final : TraversalCallback {
    bool mono = false;
    void put_value(std::string_view name, void* value, uint32_t,
                   const std::type_info& type) override {
      auto check = [&](auto tag) {
        using S = decltype(tag);
        using Tensor = dr::replace_scalar_t<TensorXf, S>;
        if (name == "data" && type == typeid(Tensor)) {
          mono = static_cast<Tensor*>(value)->shape(2) == 1;
        }
      };
      check(ScalarFloat{});
      check(dr::half{});
      check(uint8_t{});
    }
    void put_object(std::string_view, Object*, uint32_t) override {}
  } cb;
  texture->traverse(&cb);
  return cb.mono;
}

MI_INSTANTIATE_CLASS(MtlxVM)

namespace {

/**!

.. _texture-materialx_expr:

MaterialX Expression (:monosp:`materialx_expr`)
-------------------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxProgram`` bytecode and constant/parameter
     tables.

 * - in0, in1, ...
   - |texture|
   - Nested input textures referenced by ``Op::SampleInput`` and
     ``Op::SampleInputUV``.

 * - param0, param1, ...
   - |float|
   - Differentiable opaque scalar parameters exposed through traversal for
     kernel-reusing parameter edits.
   - |exposed|, |differentiable|

Evaluates a compiled MaterialX expression DAG over surface interactions,
supporting procedural noises, 2D texture lookups, geometric intrinsics,
and mesh attributes.

*/
template <typename Float, typename Spectrum>
class MtlxExprTexture final : public Texture<Float, Spectrum>,
                              public MtlxObject<Float, Spectrum> {
 public:
  MI_IMPORT_TYPES(Texture)
  using MtlxObject<Float, Spectrum>::m_vm;

  explicit MtlxExprTexture(const Properties& props) : Texture(props) {
    m_vm.compile(props);
  }

  void traverse(TraversalCallback* cb) override { m_vm.traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    m_vm.parameters_changed();
  }

  UnpolarizedSpectrum eval(const SurfaceInteraction3f& si,
                           Mask active) const override {
    if (m_vm.program().result_uniform) {
      Float v = eval_1(si, active);
      if constexpr (is_spectral_v<Spectrum>) return dr::maximum(v, 0.f);
      return v;
    }
    Color3f c = eval_3(si, active);
    if constexpr (is_monochromatic_v<Spectrum>) {
      return luminance(c);
    } else if constexpr (is_rgb_v<Spectrum>) {
      return c;
    } else {
      return upsample_rgb<Float, Spectrum>(c, si.wavelengths);
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

  ScalarFloat max() const override {
    if (is_spatially_varying()) return 1.0f;
    Color3f c = dr::maximum(eval_3(dr::zeros<SurfaceInteraction3f>(), true),
                            Color3f(0.0f));
    if constexpr (is_monochromatic_v<Spectrum>) {
      return dr::max_nested(luminance(c));
    }
    return dr::max_nested(c);
  }

  Float eval_1(const SurfaceInteraction3f& si, Mask active) const override {
    return m_vm.eval_1(si, active);
  }

  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const override {
    return m_vm.eval_3(si, active);
  }

  bool is_spatially_varying() const override {
    return m_vm.is_spatially_varying();
  }

  std::string to_string() const override {
    return "MtlxExprTexture[\n" + m_vm.to_string() + "]";
  }

  MI_DECLARE_CLASS(MtlxExprTexture)
  MI_TRAVERSE_CB(Texture, m_vm.inputs(), m_vm.params())
};

}  // namespace

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
