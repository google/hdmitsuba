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

#include <utility>

#include <mitsuba/core/math.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx::detail {

template <typename Float>
Float oren_nayar_diffuse(const Float& n_dot_v, const Float& n_dot_l,
                         const Float& l_dot_v, const Float& roughness) {
  Float s = dr::fnmadd(n_dot_l, n_dot_v, l_dot_v);
  Float stinv = dr::select(
      s > 0.0f,
      s / dr::maximum(dr::maximum(n_dot_l, n_dot_v), dr::Epsilon<Float>), 0.0f);
  Float sigma2 = dr::square(roughness);
  return dr::fmadd(0.45f * sigma2 / (sigma2 + 0.09f), stinv,
                   1.0f - 0.5f * sigma2 / (sigma2 + 0.33f));
}

template <typename Float>
Float fujii_dir_albedo(const Float& cos_theta, const Float& roughness,
                       const Float& a) {
  Float ct = dr::clip(cos_theta, 0.0f, 1.0f);
  Float sin_theta_2 = dr::fnmadd(ct, ct, 1.0f);
  Float sin_theta = dr::safe_sqrt(sin_theta_2);
  // Factored form of `(sin / cos) * (1 - sin^3) = sin * cos * (1 + sin^2 / (1 +
  // sin))` to avoid the 0/0 singularity at grazing angles.
  Float g =
      sin_theta *
      (dr::safe_acos(ct) - sin_theta * ct +
       (2.0f / 3.0f) * (ct * (1.0f + sin_theta_2 / (1.0f + sin_theta)) - 1.0f));
  return dr::clip(a * dr::fmadd(roughness * dr::InvPi<Float>, g, 1.0f), 0.0f,
                  1.0f);
}

template <typename Float, typename Spectrum>
Spectrum oren_nayar_compensated(const Float& n_dot_v, const Float& n_dot_l,
                                const Float& l_dot_v, const Float& roughness,
                                const Spectrum& color) {
  using Scalar = dr::scalar_t<Float>;
  constexpr Scalar kFujii1 = 0.5 - (2.0 / 3.0) * dr::InvPi<Scalar>;
  constexpr Scalar kFujii2 = (2.0 / 3.0) - (28.0 / 15.0) * dr::InvPi<Scalar>;
  Float s = dr::fnmadd(n_dot_l, n_dot_v, l_dot_v);
  Float stinv = dr::select(
      s > 0.0f,
      s / dr::maximum(dr::maximum(n_dot_l, n_dot_v), dr::Epsilon<Float>), s);
  Float a = dr::rcp(dr::fmadd(kFujii1, roughness, 1.0f));
  Float avg_a = a * dr::fmadd(kFujii2, roughness, 1.0f);
  Float one_minus_avg_a = dr::maximum(1.0f - avg_a, 0.0f);
  Spectrum col_ms =
      dr::square(color) * avg_a /
      dr::maximum(1.0f - color * one_minus_avg_a, dr::Epsilon<Float>);
  Float ms_weight = (1.0f - fujii_dir_albedo(n_dot_v, roughness, a)) *
                    (1.0f - fujii_dir_albedo(n_dot_l, roughness, a)) /
                    dr::maximum(one_minus_avg_a, dr::Epsilon<Float>);
  return dr::maximum(
      color * (a * dr::fmadd(roughness, stinv, 1.0f)) + col_ms * ms_weight,
      0.0f);
}

// Rational quadratic fit to GGX directional albedo (MaterialX
// `mx_ggx_dir_albedo_analytic`). Returns `(A, B)` such that the directional
// albedo is `F0 * A + F90 * B`.
template <typename Float>
std::pair<Float, Float> ggx_dir_albedo_ab(const Float& n_dot_v,
                                          const Float& alpha) {
  using Vec4 = dr::Array<Float, 4>;
  Float x = dr::clip(n_dot_v, 0.0f, 1.0f), y = dr::clip(alpha, 0.0f, 1.0f);
  Float x2 = dr::square(x), y2 = dr::square(y);
  Vec4 r = Vec4(0.1003f, 0.9345f, 1.0f, 1.0f) +
           Vec4(-0.6303f, -2.323f, -1.765f, 0.2281f) * x +
           Vec4(9.748f, 2.229f, 8.263f, 15.94f) * y +
           Vec4(-2.038f, -3.748f, 11.53f, -55.83f) * (x * y) +
           Vec4(29.34f, 1.424f, 28.96f, 13.08f) * x2 +
           Vec4(-8.245f, -0.7684f, -7.507f, 41.26f) * y2 +
           Vec4(-26.44f, 1.436f, -36.11f, 54.9f) * (x2 * y) +
           Vec4(19.99f, 0.2913f, 15.86f, 300.2f) * (x * y2) +
           Vec4(-5.448f, 0.6286f, 33.37f, -285.1f) * (x2 * y2);
  return {dr::clip(r.x() / r.z(), 0.0f, 1.0f),
          dr::clip(r.y() / r.w(), 0.0f, 1.0f)};
}

// Turquin (2019) GGX multi-scattering energy compensation factor from
// single-scattering directional albedo `ess = A + B`.
template <typename Float, typename Spectrum>
Spectrum ggx_energy_compensation(const Float& ess, const Spectrum& fss) {
  return dr::fmadd(fss, (1.0f - ess) / ess, 1.0f);
}

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
