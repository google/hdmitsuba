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
#include <type_traits>

#include <drjit/matrix.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>

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
  return safe_normalize(
      Vec3(si.dp_du - si.sh_frame.n * dr::dot(si.sh_frame.n, si.dp_du)),
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

template <typename Float>
Float mx_safe_div(const Float& a, const Float& b) {
  auto nz = b != 0.0f;
  return dr::select(nz, a / dr::select(nz, b, 1.0f), 0.0f);
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

template <typename Float>
Float mx_safe_atan2(const Float& y, const Float& x) {
  auto nz = (y != 0.0f) || (x != 0.0f);
  return dr::select(nz, dr::atan2(y, dr::select(nz, x, 1.0f)), 0.0f);
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

template <typename Float>
Vector<Float, 3> mx_normal_map(const Vector<Float, 3>& val,
                               const Vector<Float, 2>& scale,
                               const Vector<Float, 3>& n,
                               const Vector<Float, 3>& tangent,
                               const Vector<Float, 3>& bitangent,
                               const Vector<Float, 3>& fallback_n) {
  using Vector3f = Vector<Float, 3>;
  Vector3f N = safe_normalize(n, fallback_n);
  Vector3f T = safe_normalize(tangent - N * dr::dot(N, tangent),
                              coordinate_system(N).first);
  Vector3f B = dr::cross(N, T);
  B = dr::select(dr::dot(bitangent, B) < 0.0f, -B, B);
  Vector3f v = dr::select(dr::squared_norm(val) == 0.0f,
                          Vector3f(0.0f, 0.0f, 1.0f), val * 2.0f - 1.0f);
  return safe_normalize(
      T * (v.x() * scale.x()) + B * (v.y() * scale.y()) + N * v.z(), N);
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

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
