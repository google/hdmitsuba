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

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
