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
#include <type_traits>

#include <drjit/color.h>
#include <drjit/matrix.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>

#include "hdmitsuba/materialx/program.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx::detail {

template <typename Vec>
Vec safe_normalize(const Vec& v, const Vec& fallback = Vec(0.0f),
                   dr::scalar_t<Vec> min_len2 = 0.0f) {
  auto len2 = dr::squared_norm(v);
  auto valid = len2 > min_len2;
  return dr::select(valid, v * dr::rsqrt(dr::select(valid, len2, 1.0f)),
                    fallback);
}

template <typename SurfaceInteraction>
auto geom_tangent_world(const SurfaceInteraction& si) {
  using Vec3 = std::decay_t<decltype(si.sh_frame.s)>;
  return safe_normalize(dr::fnmadd(Vec3(si.sh_frame.n),
                                   dr::dot(si.sh_frame.n, si.dp_du), si.dp_du),
                        si.sh_frame.s, dr::scalar_t<Vec3>(1e-20f));
}

template <typename SurfaceInteraction, typename Vec3>
Vec3 geom_bitangent_world(const SurfaceInteraction& si, const Vec3& t) {
  Vec3 b = dr::cross(si.sh_frame.n, t);
  // Mitsuba's V is flipped relative to MaterialX, so dp/dv_mtlx = -dp_dv.
  return dr::select(dr::dot(b, si.dp_dv) > 0.0f, -b, b);
}

// Like OSL, rounds halfway cases away from zero.
template <typename Float>
Float mx_round(const Float& x) {
  Float t = dr::trunc(x);
  return dr::select(dr::abs(x - t) >= 0.5f, t + dr::copysign(1.0f, x), t);
}

template <typename Float>
Float mx_safe_log(const Float& x) {
  auto pos = x > 0.0f;
  return dr::select(pos, dr::log(dr::select(pos, x, 1.0f)), 0.0f);
}

template <typename T>
T mx_safe_div(const T& a, const T& b) {
  auto nz = b != 0.0f;
  return dr::select(nz, a / dr::select(nz, b, T(1.0f)), T(0.0f));
}

// Like OSL's `safe_pow()`, raises negative bases to integer powers only, and
// returns 0 for other negative bases.
template <typename Float>
Float mx_safe_pow(const Float& x, const Float& y) {
  Float mag = dr::pow(dr::select(x != 0.0f, dr::abs(x), 1.0f), y);
  Float neg =
      dr::select(dr::floor(y) == y,
                 dr::select(dr::floor(0.5f * y) * 2.0f != y, -mag, mag), 0.0f);
  return dr::select(
      y == 0.0f, 1.0f,
      dr::select(x == 0.0f, 0.0f, dr::select(x > 0.0f, mag, neg)));
}

template <typename T>
T mx_safe_atan2(const T& y, const T& x) {
  auto nz = (y != 0.0f) | (x != 0.0f);
  return dr::select(nz, dr::atan2(y, dr::select(nz, x, T(1.0f))), T(0.0f));
}

// Like division, the inverse of a singular matrix is zero.
template <typename Float, size_t N>
dr::Matrix<Float, N> mx_safe_inverse(const dr::Matrix<Float, N>& m) {
  using Matrix = dr::Matrix<Float, N>;
  auto valid = dr::det(m) != 0.0f;
  return dr::select(valid,
                    dr::inverse(dr::select(valid, m, dr::identity<Matrix>())),
                    dr::zeros<Matrix>());
}

template <typename Float>
Vector<Float, 2> mx_rotate2d(const Vector<Float, 2>& v,
                             const Float& amount_deg) {
  auto [s, c] = dr::sincos(dr::deg_to_rad(amount_deg));
  return Vector<Float, 2>(v.x() * c + v.y() * s, v.y() * c - v.x() * s);
}

template <typename Float>
Vector<Float, 3> mx_rotate3d(const Vector<Float, 3>& v, const Float& amount_deg,
                             const Vector<Float, 3>& axis_in) {
  Vector<Float, 3> axis = safe_normalize(axis_in);
  auto [s, c] = dr::sincos(-dr::deg_to_rad(amount_deg));
  return v * c + dr::cross(axis, v) * s + axis * dr::dot(axis, v) * (1.0f - c);
}

// Evaluates tangent-space normal mapping against an orthonormal `frame`
// `(tangent, bitangent, normal)`.
template <typename Float>
Vector<Float, 3> mx_normal_map_frame(const Vector<Float, 3>& val,
                                     const Vector<Float, 2>& scale,
                                     const Frame<Float>& frame) {
  using Vector3f = Vector<Float, 3>;
  Vector3f v =
      dr::select(dr::squared_norm(val) == 0.0f, Vector3f(0.0f, 0.0f, 1.0f),
                 dr::fmadd(val, 2.0f, -1.0f));
  return safe_normalize(
      frame.to_world(Vector3f(v.x() * scale.x(), v.y() * scale.y(), v.z())),
      Vector3f(frame.n));
}

template <typename Float>
Vector<Float, 3> mx_normal_map(const Vector<Float, 3>& val,
                               const Vector<Float, 2>& scale,
                               const Vector<Float, 3>& n,
                               const Vector<Float, 3>& tangent,
                               const Vector<Float, 3>& bitangent,
                               const Vector<Float, 3>& fallback_n) {
  using Vector3f = Vector<Float, 3>;
  Vector3f N = safe_normalize(n, fallback_n);
  Vector3f T = safe_normalize(dr::fnmadd(N, dr::dot(N, tangent), tangent),
                              coordinate_system(N).first);
  Vector3f B = dr::cross(N, T);
  B = dr::select(dr::dot(bitangent, B) < 0.0f, -B, B);
  return mx_normal_map_frame(val, scale,
                             Frame<Float>(T, B, Normal<Float, 3>(N)));
}

// Port of MaterialX `mx_blackbody.glsl` (Apache-2.0, Contributors to the
// MaterialX Project).
template <typename Float>
Color<Float, 3> mx_blackbody(const Float& temp_k) {
  Float temp = dr::clip(temp_k, 800.0f, 25000.0f);
  Float t = 1000.0f / temp;
  Float xc =
      dr::select(temp < 4000.0f,
                 dr::horner(t, 0.179910f, 0.8776956f, -0.2343580f, -0.2661239f),
                 dr::horner(t, 0.240390f, 0.2226347f, 2.1070379f, -3.0258469f));
  Float yc = dr::select(
      temp < 2222.0f,
      dr::horner(xc, -0.20219683f, 2.18555832f, -1.34811020f, -1.1063814f),
      dr::select(
          temp < 4000.0f,
          dr::horner(xc, -0.16748867f, 2.09137015f, -1.37418593f, -0.9549476f),
          dr::horner(xc, -0.37001483f, 3.75112997f, -5.87338670f, 3.0817580f)));
  Color<Float, 3> rgb =
      xyz_to_srgb(Color<Float, 3>(xc / yc, 1.0f, (1.0f - xc - yc) / yc));
  return dr::maximum(rgb, 0.0f);
}

template <typename Float>
Vector<Float, 4> eval_elementwise(Op op, int channels,
                                  const Vector<Float, 4>* a) {
  auto map = [&](auto fn) {
    if (channels == 1) return Vector<Float, 4>(fn(0));
    Vector<Float, 4> out(0.0f);
    for (int c = 0; c < channels; ++c) out[c] = fn(c);
    return out;
  };
  auto unary = [&](auto fn) { return map([&](int c) { return fn(a[0][c]); }); };
  auto binary = [&](auto fn) {
    return map([&](int c) { return fn(a[0][c], a[1][c]); });
  };
  auto ternary = [&](auto fn) {
    return map([&](int c) { return fn(a[0][c], a[1][c], a[2][c]); });
  };
  switch (op) {
    case Op::Neg:
      return unary([](const Float& x) { return -x; });
    case Op::Abs:
      return unary([](const Float& x) { return dr::abs(x); });
    case Op::Sin:
      return unary([](const Float& x) { return dr::sin(x); });
    case Op::Cos:
      return unary([](const Float& x) { return dr::cos(x); });
    case Op::Tan:
      return unary([](const Float& x) { return dr::tan(x); });
    case Op::Exp:
      return unary([](const Float& x) { return dr::exp(x); });
    case Op::Round:
      return unary([](const Float& x) { return mx_round(x); });
    case Op::Floor:
      return unary([](const Float& x) { return dr::floor(x); });
    case Op::Ceil:
      return unary([](const Float& x) { return dr::ceil(x); });
    case Op::Sqrt:
      return unary([](const Float& x) { return dr::safe_sqrt(x); });
    case Op::Log:
      return unary([](const Float& x) { return mx_safe_log(x); });
    case Op::Asin:
      return unary([](const Float& x) { return dr::safe_asin(x); });
    case Op::Acos:
      return unary([](const Float& x) { return dr::safe_acos(x); });
    case Op::Fract:
      return unary([](const Float& x) { return x - dr::floor(x); });
    case Op::Add:
      return binary([](const Float& x, const Float& y) { return x + y; });
    case Op::Sub:
      return binary([](const Float& x, const Float& y) { return x - y; });
    case Op::Mul:
      return binary([](const Float& x, const Float& y) { return x * y; });
    case Op::Div:
      return binary(
          [](const Float& x, const Float& y) { return mx_safe_div(x, y); });
    case Op::Lt:
      return binary([](const Float& x, const Float& y) {
        return dr::select(x < y, Float(1.0f), Float(0.0f));
      });
    case Op::Le:
      return binary([](const Float& x, const Float& y) {
        return dr::select(x <= y, Float(1.0f), Float(0.0f));
      });
    case Op::Eq:
      return binary([](const Float& x, const Float& y) {
        return dr::select(x == y, Float(1.0f), Float(0.0f));
      });
    case Op::Min:
      return binary(
          [](const Float& x, const Float& y) { return dr::minimum(x, y); });
    case Op::Max:
      return binary(
          [](const Float& x, const Float& y) { return dr::maximum(x, y); });
    case Op::Pow:
      return binary(
          [](const Float& x, const Float& y) { return mx_safe_pow(x, y); });
    case Op::Atan2:
      return binary(
          [](const Float& y, const Float& x) { return mx_safe_atan2(y, x); });
    case Op::Select:
      return ternary([](const Float& c, const Float& t, const Float& f) {
        return dr::select(c != 0.0f, t, f);
      });
    case Op::Mix:
      return ternary([](const Float& fg, const Float& bg, const Float& m) {
        return dr::lerp(bg, fg, m);
      });
    case Op::Clip:
      return ternary([](const Float& x, const Float& lo, const Float& hi) {
        return dr::clip(x, lo, hi);
      });
    default:
      return Vector<Float, 4>(0.0f);
  }
}

template <typename Float>
bool eval_pure_op(Inst inst, Vector<Float, 4>*& sp) {
  using Val = Vector<Float, 4>;
  auto pop = [&](int n) {
    sp -= n;
    return sp;
  };
  auto xy = [](const Val& v) -> Vector<Float, 2> { return dr::head<2>(v); };
  auto xyz = [](const Val& v) -> Vector<Float, 3> { return dr::head<3>(v); };
  auto val2 = [](const auto& v) { return Val(v.x(), v.y(), 0.0f, 0.0f); };
  auto val3 = [](const auto& v, const Float& w = 0.0f) {
    return Val(v.x(), v.y(), v.z(), w);
  };

  if (int n = elementwise_arity(inst.op)) {
    Val* a = pop(n);
    *sp++ = eval_elementwise<Float>(inst.op, inst.channels, a);
    return true;
  }
  uint16_t imm = inst.imm;
  switch (inst.op) {
    case Op::Rgb: {
      Val* a = pop(3);
      *sp++ = Val(a[0].x(), a[1].x(), a[2].x(), 0.0f);
      return true;
    }
    case Op::Vec2: {
      Val* a = pop(2);
      *sp++ = Val(a[0].x(), a[1].x(), 0.0f, 0.0f);
      return true;
    }
    case Op::Vec4: {
      Val* a = pop(2);
      *sp++ = val3(a[0], a[1].x());
      return true;
    }
    case Op::Chan:
      sp[-1] = Val(sp[-1].entry(imm));
      return true;
    case Op::Dot: {
      Val* a = pop(2);
      *sp++ = Val(imm == 2   ? dr::dot(xy(a[0]), xy(a[1]))
                  : imm == 4 ? dr::dot(a[0], a[1])
                             : dr::dot(xyz(a[0]), xyz(a[1])));
      return true;
    }
    case Op::Cross: {
      Val* a = pop(2);
      *sp++ = val3(dr::cross(xyz(a[0]), xyz(a[1])));
      return true;
    }
    case Op::Normalize:
      sp[-1] = imm == 2   ? val2(safe_normalize(xy(sp[-1])))
               : imm == 4 ? safe_normalize(sp[-1])
                          : val3(safe_normalize(xyz(sp[-1])));
      return true;
    case Op::Rotate2D: {
      Val* a = pop(2);
      *sp++ = val2(mx_rotate2d(xy(a[0]), a[1].x()));
      return true;
    }
    case Op::Rotate3D: {
      Val* a = pop(3);
      *sp++ = val3(mx_rotate3d(xyz(a[0]), a[1].x(), xyz(a[2])));
      return true;
    }
    // The color conversions preserve the fourth (alpha) channel natively.
    case Op::RgbToHsv:
      sp[-1] = dr::rgb_to_hsv(sp[-1]);
      return true;
    case Op::HsvToRgb:
      sp[-1] = dr::hsv_to_rgb(sp[-1]);
      return true;
    case Op::Blackbody:
      sp[-1] = val3(mx_blackbody(sp[-1].x()));
      return true;
    default:
      return false;
  }
}

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
