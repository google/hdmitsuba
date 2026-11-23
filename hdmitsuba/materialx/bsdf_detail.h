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

// Pure local-space MaterialX BSDF optics and lobe physics. Include only from
// `bsdf.cc`.

#include <tuple>
#include <utility>

#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/shape.h>

#include "hdmitsuba/materialx/math.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx::detail {

template <typename ScalarFloat>
inline constexpr ScalarFloat kMinLen2 =
    dr::Epsilon<ScalarFloat> * dr::Epsilon<ScalarFloat>;

template <typename Float>
Float pow5(const Float& x) {
  return dr::square(dr::square(x)) * x;
}

template <typename Spectrum, typename UnpolSpec>
auto selection_weight(const UnpolSpec& c) {
  if constexpr (is_rgb_v<Spectrum>) {
    return luminance(c);
  } else {
    return dr::mean(c);
  }
}

// Curve tangent along the fiber axis (`dp_dv` / `sh_frame.t` on Mitsuba
// `BSplineCurve` / `LinearCurve`, where `u` is circumferential), falling back
// to `geom_tangent_world` (`dp_du`) on meshes and other shapes.
template <typename SurfaceInteraction>
auto geom_curve_tangent_world(const SurfaceInteraction& si) {
  using Vec3 = std::decay_t<decltype(si.sh_frame.s)>;
  if constexpr (!dr::is_jit_v<Vec3>) {
    if (!si.shape) return geom_tangent_world(si);
  }
  auto is_curve = (si.shape->shape_type() &
                   (+ShapeType::BSplineCurve | +ShapeType::LinearCurve)) != 0u;
  Vec3 raw = dr::select(is_curve, si.dp_dv, si.dp_du);
  Vec3 fallback = dr::select(is_curve, si.sh_frame.t, si.sh_frame.s);
  return safe_normalize(Vec3(raw - si.sh_frame.n * dr::dot(si.sh_frame.n, raw)),
                        fallback, dr::scalar_t<Vec3>(1e-20f));
}

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
  Float ess_c = dr::clip(ess, dr::Epsilon<Float>, 1.0f);
  return dr::fmadd(fss, (1.0f - ess_c) / ess_c, 1.0f);
}

// Energy-compensated GGX directional albedo from `(F0, F(n_dot_v), F90)`.
template <typename Float, typename Spectrum>
Spectrum ggx_refl_albedo(const Float& n_dot_v, const Float& alpha,
                         const Spectrum& f0, const Spectrum& f_nv,
                         const Spectrum& f90) {
  auto [a, b] = ggx_dir_albedo_ab(n_dot_v, alpha);
  return dr::minimum(
      ggx_energy_compensation(a + b, f_nv) * dr::fmadd(f0, a, f90 * b), 1.0f);
}

template <typename Float>
Float f0_to_ior(const Float& f0) {
  Float s = dr::sqrt(dr::clip(f0, 0.01f, 0.99f));
  return (1.0f + s) / (1.0f - s);
}

// Hoffman-Schlick generalized Fresnel (MaterialX `mx_fresnel_hoffman_schlick`).
template <typename Float, typename Spectrum>
Spectrum fresnel_hoffman_schlick(const Float& cos_theta, const Spectrum& f0,
                                 const Spectrum& f82, const Spectrum& f90,
                                 const Float& exponent) {
  using Scalar = dr::scalar_t<Float>;
  // `1 / ((1/7) * (6/7)^6) = 7^7 / 6^6`
  constexpr Scalar kFactor = Scalar(823543.0 / 46656.0);
  Float x = dr::clip(cos_theta, 0.0f, 1.0f);
  Float exp_c = dr::maximum(exponent, 0.0f);
  Float omx = 1.0f - x;
  auto pos = omx > 0.0f;
  Float t = dr::select(pos, dr::pow(dr::select(pos, omx, 1.0f), exp_c),
                       dr::select(exp_c > 0.0f, 0.0f, 1.0f));
  Spectrum a = dr::lerp(f0, f90, dr::pow(Float(Scalar(6.0 / 7.0)), exp_c)) *
               (1.0f - f82) * kFactor;
  return dr::clip(
      dr::lerp(f0, f90, t) - a * (x * dr::square(omx * dr::square(omx))), 0.0f,
      1.0f);
}

// Extracts Belcour-Barla `(r_p, r_s, phi_p, phi_s)` from Mitsuba's polarized
// Fresnel reflection amplitudes `(a_s, a_p)` (`<mitsuba/render/fresnel.h>`).
template <typename Val>
std::tuple<Val, Val, Val, Val> polarized_fresnel_r_phi(
    const dr::Complex<Val>& a_s, const dr::Complex<Val>& a_p) {
  return {dr::squared_norm(a_p), dr::squared_norm(a_s),
          mx_safe_atan2(-dr::imag(a_p), dr::real(a_p)),
          mx_safe_atan2(dr::imag(a_s), -dr::real(a_s))};
}

// Belcour & Barla (2017) cosine/sine sensitivity basis in CIE-RGB (or
// monochromatic CIE-Y) for thin-film interference (`mx_eval_sensitivity`),
// decoupled from per-channel phase shifts so `XYZ_TO_RGB` never mixes R/G/B
// phases.
template <typename Float, typename Spectrum>
std::pair<unpolarized_spectrum_t<Spectrum>, unpolarized_spectrum_t<Spectrum>>
eval_sensitivity_cs(const Float& opd) {
  using Scalar = dr::scalar_t<Float>;
  using UnpolSpec = unpolarized_spectrum_t<Spectrum>;
  constexpr Scalar kNorm = Scalar(1.0 / 1.0685e-7);
  Float phase = dr::TwoPi<Float> * opd, phase2 = dr::square(phase);
  if constexpr (is_rgb_v<Spectrum>) {
    using Color3f = Color<Float, 3>;
    using ScalarColor3 = Color<Scalar, 3>;
    ScalarColor3 kVar(Scalar(4.3278e+09), Scalar(9.3046e+09),
                      Scalar(6.6121e+09));
    ScalarColor3 kAmp = ScalarColor3(Scalar(5.4856e-13), Scalar(4.4201e-13),
                                     Scalar(5.2481e-13)) *
                        dr::sqrt(dr::TwoPi<Scalar> * kVar) * kNorm;
    ScalarColor3 kPos(Scalar(1.6810e+06), Scalar(1.7953e+06),
                      Scalar(2.2084e+06));
    constexpr Scalar kVarX2 = Scalar(4.5282e+09);
    Scalar kAmpX2 =
        Scalar(9.7470e-14) * dr::sqrt(dr::TwoPi<Scalar> * kVarX2) * kNorm;
    constexpr Scalar kPosX2 = Scalar(2.2399e+06);
    dr::Matrix<Scalar, 3> kXyzToRgb(
        Scalar(2.3706743), Scalar(-0.9000405), Scalar(-0.4706338),
        Scalar(-0.5138850), Scalar(1.4253036), Scalar(0.0885814),
        Scalar(0.0052982), Scalar(-0.0146949), Scalar(1.0093968));

    Color3f env = Color3f(kAmp) * dr::exp(Color3f(-kVar) * phase2);
    auto [s_xyz, c_xyz] = dr::sincos(Color3f(kPos) * phase);
    Color3f xyz_c = env * c_xyz, xyz_s = env * s_xyz;
    Float env_x2 = kAmpX2 * dr::exp(-kVarX2 * phase2);
    auto [s_x2, c_x2] = dr::sincos(kPosX2 * phase);
    xyz_c.x() += env_x2 * c_x2;
    xyz_s.x() += env_x2 * s_x2;
    return {kXyzToRgb * xyz_c, kXyzToRgb * xyz_s};
  } else {
    constexpr Scalar kVarY = Scalar(9.3046e+09);
    Scalar kAmpY =
        Scalar(4.4201e-13) * dr::sqrt(dr::TwoPi<Scalar> * kVarY) * kNorm;
    constexpr Scalar kPosY = Scalar(1.7953e+06);
    Float env = kAmpY * dr::exp(-kVarY * phase2);
    auto [sy, cy] = dr::sincos(kPosY * phase);
    return {UnpolSpec(env * cy), UnpolSpec(env * sy)};
  }
}

// Belcour & Barla (2017) Airy thin-film reflectance given air-to-film polarized
// amplitudes `(a12_s, a12_p)`, optical path difference `opd` in nanometers
// (`2 * eta2 * ct2 * thickness_nm`), and substrate polarized
// reflectances/phases.
template <typename Float, typename Spectrum>
unpolarized_spectrum_t<Spectrum> airy_thinfilm_reflectance(
    const dr::Complex<Float>& a12_s, const dr::Complex<Float>& a12_p,
    const Float& opd, const unpolarized_spectrum_t<Spectrum>& r23_p,
    const unpolarized_spectrum_t<Spectrum>& r23_s,
    const unpolarized_spectrum_t<Spectrum>& phi23_p,
    const unpolarized_spectrum_t<Spectrum>& phi23_s,
    const wavelength_t<Spectrum>& wavelengths) {
  using Scalar = dr::scalar_t<Float>;
  using UnpolSpec = unpolarized_spectrum_t<Spectrum>;
  Float r12_s = dr::squared_norm(a12_s), r12_p = dr::squared_norm(a12_p);
  Float phi21_p =
      dr::select(dr::real(a12_p) < 0.0f, Float(0.0f), Float(dr::Pi<Float>));

  if constexpr (is_spectral_v<Spectrum>) {
    UnpolSpec phase =
        dr::TwoPi<Float> * opd /
        dr::select(wavelengths > 0.0f, wavelengths, UnpolSpec(550.0f));
    auto eval_pol = [&](const Float& r12, const UnpolSpec& r23,
                        const Float& phi21, const UnpolSpec& phi23) {
      UnpolSpec r123 = dr::safe_sqrt(r12 * r23);
      UnpolSpec s = dr::sin(0.5f * (phase + phi23 + phi21));
      UnpolSpec denom = dr::square(1.0f - r123) + 4.0f * r123 * dr::square(s);
      return 1.0f - mx_safe_div((1.0f - r12) * (1.0f - r23), denom);
    };
    return dr::clip(
        0.5f * (eval_pol(r12_p, r23_p, phi21_p, phi23_p) +
                eval_pol(r12_s, r23_s, Float(dr::Pi<Float>), phi23_s)),
        0.0f, 1.0f);
  } else {
    std::pair<UnpolSpec, UnpolSpec> sens[3];
    for (int m = 1; m <= 3; ++m) {
      sens[m - 1] =
          eval_sensitivity_cs<Float, Spectrum>(Scalar(m * 1e-9) * opd);
    }
    auto eval_pol = [&](const Float& r12, const UnpolSpec& r23,
                        const Float& phi21, const UnpolSpec& phi23) {
      UnpolSpec r123 = dr::safe_sqrt(r12 * r23), shift = phi23 + phi21;
      Float t121 = 1.0f - r12;
      UnpolSpec rs = (dr::square(t121) * r23) /
                     dr::maximum(1.0f - r12 * r23, dr::Epsilon<Float>);
      UnpolSpec out = r12 + rs, cm = 2.0f * (rs - t121);
      for (int m = 1; m <= 3; ++m) {
        cm *= r123;
        auto [ss, cs] = dr::sincos(Float(m) * shift);
        out += cm * (sens[m - 1].first * cs - sens[m - 1].second * ss);
      }
      return out;
    };
    return dr::clip(
        0.5f * (eval_pol(r12_p, r23_p, phi21_p, phi23_p) +
                eval_pol(r12_s, r23_s, Float(dr::Pi<Float>), phi23_s)),
        0.0f, 1.0f);
  }
}

// Height-correlated Smith masking-shadowing ratio `G2 / G1(wi)`.
template <typename Float>
Float ggx_g2_over_g1(const Float& g1_i, const Float& g1_o) {
  return g1_o /
         dr::maximum(dr::fmadd(g1_o, 1.0f - g1_i, g1_i), dr::Epsilon<Float>);
}

// Clamps IOR positive, and away from 1 for solid transmission to avoid the
// `wi + eta * wo == 0` half-vector singularity.
template <typename Float>
Float safe_dielectric_ior(const Float& ior_raw, bool solid_trans) {
  using ScalarFloat = dr::scalar_t<Float>;
  constexpr ScalarFloat kIorEps = 1e-4f;
  Float ior = dr::maximum(ior_raw, kIorEps);
  if (!solid_trans) return ior;
  return dr::select(
      dr::abs(ior - 1.0f) < kIorEps,
      dr::select(ior < 1.0f, Float(1.0f - kIorEps), Float(1.0f + kIorEps)),
      ior);
}

// Wraps `mitsuba::fresnel`, handling two-sided dielectrics (`thin_walled` or
// reflection-only) and clamping `F` to exact `1` under total internal
// reflection (`cos_theta_t == 0`), where `dr::rcp` in `mitsuba::fresnel` can
// round `F` just below `1` (e.g. `0.99999988f` on AVX/CUDA).
template <typename Float>
std::tuple<Float, Float, Float, Float> exact_fresnel(const Float& cos_theta_i,
                                                     const Float& eta,
                                                     bool two_sided) {
  auto [f, cos_theta_t, eta_it, eta_ti] =
      fresnel(two_sided ? dr::abs(cos_theta_i) : cos_theta_i, eta);
  return {dr::select((cos_theta_t == 0.0f) && (eta_it < 1.0f), Float(1.0f), f),
          cos_theta_t, eta_it, eta_ti};
}

// Clamps conductor `ior` away from zero to avoid `0 / 0` in
// `fresnel_conductor` at normal incidence when `ior == extinction == 0`.
template <typename Float, typename Spectrum>
dr::Complex<Spectrum> safe_conductor_eta(const Spectrum& ior,
                                         const Spectrum& extinction) {
  return dr::Complex<Spectrum>(dr::maximum(ior, dr::Epsilon<Float>),
                               extinction);
}

template <typename Float, typename Spectrum>
Frame<Float> build_shading_frame(const SurfaceInteraction<Float, Spectrum>& si,
                                 const Vector<Float, 3>& n_geom,
                                 Vector<Float, 3> nl,
                                 const Vector<Float, 3>& t_world) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Vector3f = Vector<Float, 3>;
  Float n_dot_g = dr::dot(n_geom, nl);
  nl = dr::select(n_dot_g < 0.0f, nl - 2.0f * n_dot_g * n_geom, nl);
  n_dot_g = dr::abs(n_dot_g);
  // Keep the shading normal on the geometric side of `si.wi` (Schuessler et
  // al. 2017; reduces to `(-nl.x, -nl.y, nl.z)` when `n_geom == (0, 0, 1)`).
  Vector3f n_loc = dr::select(
      dr::mulsign(Float(1.0f), dr::dot(n_geom, si.wi)) * dr::dot(si.wi, nl) <=
          0.0f,
      dr::select(n_dot_g > 0.0f, 2.0f * n_dot_g * n_geom - nl, n_geom), nl);
  Vector3f t_raw = si.to_local(t_world);
  Vector3f s_loc =
      safe_normalize(dr::fnmadd(n_loc, dr::dot(n_loc, t_raw), t_raw),
                     coordinate_system(n_loc).first, kMinLen2<ScalarFloat>);
  return Frame<Float>(s_loc, dr::cross(n_loc, s_loc), n_loc);
}

struct MicrofacetOpts {
  bool refl_on, trans_on, thin_walled, two_sided, retroreflective, trans_only,
      need_ior_fresnel;
};

template <typename Float>
struct MicrofacetSample {
  Vector<Float, 3> wo_l;
  dr::mask_t<Float> is_refl;
  Float eta;
  dr::mask_t<Float> valid;
};

// Follows `roughdielectric`: sample visible normals for `wi` flipped into the
// upper hemisphere, then reflect/refract the unflipped `wi`. Handles
// `Dielectric`, `Conductor`, and `GeneralizedSchlick`.
template <typename Float, typename ProbFn>
MicrofacetSample<Float> ggx_microfacet_sample(
    const Vector<Float, 3>& wi_l, const Vector<Float, 2>& alpha,
    const Float& ior, const MicrofacetOpts& opts, const Float& u1,
    const Point<Float, 2>& u2, ProbFn&& refl_prob_fn) {
  using Mask = dr::mask_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Normal3f = Normal<Float, 3>;
  using Frame3f = Frame<Float>;
  Float cos_i = Frame3f::cos_theta(wi_l);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Normal3f m = distr.sample(dr::mulsign(wi_l, cos_i), u2).first;
  Vector3f wo_refl = reflect(wi_l, m);
  Vector3f wo_l = opts.retroreflective ? reflect(wo_refl) : wo_refl;
  Mask do_reflect(opts.refl_on);
  Mask trans_ok(false);
  Float out_eta(1.0f);
  if (opts.trans_on) {
    Float cos_im = dr::dot(wi_l, m);
    Float fres(0.0f), cos_theta_t(0.0f), eta_it(1.0f), eta_ti(1.0f);
    if (opts.need_ior_fresnel) {
      std::tie(fres, cos_theta_t, eta_it, eta_ti) =
          exact_fresnel(cos_im, ior, opts.two_sided);
    }
    trans_ok = opts.thin_walled ? Mask(true) : (fres < 1.0f);
    if (opts.refl_on) {
      do_reflect = !trans_ok || (u1 < refl_prob_fn(cos_im, fres, cos_theta_t));
    }
    Vector3f wo_trans = opts.thin_walled
                            ? Vector3f(wo_refl.x(), wo_refl.y(), -wo_refl.z())
                            : refract(wi_l, m, cos_theta_t, eta_ti);
    wo_l = dr::select(do_reflect, wo_l, wo_trans);
    if (!opts.thin_walled)
      out_eta = dr::select(do_reflect, Float(1.0f), eta_it);
  }
  Float cos_o = Frame3f::cos_theta(wo_l);
  Mask valid = dr::select(do_reflect, cos_i * cos_o > 0.0f,
                          trans_ok && (cos_i * cos_o < 0.0f));
  return {wo_l, do_reflect, out_eta, valid};
}

// Follows `roughdielectric::eval()` / `pdf()`, using height-correlated Smith G2
// for reflection to match MaterialX `pbrlib`. Handles `Dielectric`,
// `Conductor`, and `GeneralizedSchlick`.
template <typename Float, typename Spectrum, typename FresnelFn,
          typename ProbFn>
std::pair<Spectrum, Float> ggx_microfacet_eval_pdf(
    const Vector<Float, 3>& wi_l, const Vector<Float, 3>& wo_l,
    const Vector<Float, 2>& alpha, const Float& ior, const MicrofacetOpts& opts,
    TransportMode transport_mode, FresnelFn&& fresnel_fn,
    ProbFn&& refl_prob_fn) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Mask = dr::mask_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Frame3f = Frame<Float>;
  Float cos_i = Frame3f::cos_theta(wi_l), cos_o = Frame3f::cos_theta(wo_l);
  Mask is_refl = cos_i * cos_o > 0.0f;
  if (opts.refl_on != opts.trans_on &&
      dr::none_or<false>(opts.refl_on ? is_refl : !is_refl)) {
    return {0.0f, 0.0f};
  }
  bool solid_trans = opts.trans_on && !opts.thin_walled;
  Float n_dot_v = dr::abs(cos_i);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Vector3f wo_m =
      (opts.trans_on && opts.thin_walled)
          ? Vector3f(wo_l.x(), wo_l.y(), dr::mulsign(dr::abs(cos_o), cos_i))
          : wo_l;
  if (opts.retroreflective) wo_m = dr::select(is_refl, reflect(wo_m), wo_m);
  Float eta =
      solid_trans ? dr::select(cos_i > 0.0f, ior, dr::rcp(ior)) : Float(1.0f);
  Vector3f m = safe_normalize(
      wi_l +
          (solid_trans ? wo_m * dr::select(is_refl, Float(1.0f), eta) : wo_m),
      Vector3f(0.0f, 0.0f, 1.0f), kMinLen2<ScalarFloat>);
  m = dr::mulsign(m, Frame3f::cos_theta(m));
  Float cos_im = dr::dot(wi_l, m), cos_om = dr::dot(wo_m, m);
  Mask valid =
      (cos_im * cos_i > 0.0f) && (cos_om * Frame3f::cos_theta(wo_m) > 0.0f);

  Float f_d(0.0f), cos_theta_t(0.0f);
  if (opts.need_ior_fresnel) {
    std::tie(f_d, cos_theta_t, std::ignore, std::ignore) =
        exact_fresnel(cos_im, ior, opts.two_sided);
  }
  Spectrum f_c =
      opts.trans_only ? Spectrum(0.0f) : fresnel_fn(cos_im, f_d, cos_theta_t);
  Mask trans_ok = opts.thin_walled ? Mask(true) : (f_d < 1.0f);
  Float p_r = (opts.refl_on && opts.trans_on)
                  ? dr::select(trans_ok, refl_prob_fn(f_d, f_c), 1.0f)
                  : Float(opts.refl_on ? 1.0f : 0.0f);
  Float g1_i = distr.smith_g1(wi_l, m), g1_o = distr.smith_g1(wo_m, m);
  Float d_g1_over_cos = distr.eval(m) * g1_i / dr::select(valid, n_dot_v, 1.0f);
  Float refl_pdf = 0.25f * d_g1_over_cos;
  Spectrum val(0.0f);
  Float pdf(0.0f);
  if (opts.refl_on) {
    Mask use_refl = valid && is_refl;
    if (dr::any_or<true>(use_refl)) {
      auto [a, b] = ggx_dir_albedo_ab(n_dot_v, dr::sqrt(dr::prod(alpha)));
      val = dr::select(use_refl,
                       f_c * ggx_energy_compensation(a + b, f_c) *
                           (refl_pdf * ggx_g2_over_g1(g1_i, g1_o)),
                       val);
      pdf = dr::select(use_refl, p_r * refl_pdf, pdf);
    }
  }
  if (opts.trans_on) {
    Mask use_trans = valid && !is_refl && trans_ok;
    if (dr::any_or<true>(use_trans)) {
      Float trans_val, trans_pdf;
      if (solid_trans) {
        Float denom = dr::square(cos_im + eta * cos_om);
        use_trans = use_trans && (denom > kMinLen2<ScalarFloat>);
        Float base = d_g1_over_cos * dr::abs(cos_im * cos_om) /
                     dr::select(use_trans, denom, 1.0f);
        trans_pdf = base * dr::square(eta);
        trans_val =
            (transport_mode == TransportMode::Radiance ? base : trans_pdf) *
            g1_o;
      } else {
        trans_val = refl_pdf * g1_o;
        trans_pdf = refl_pdf;
      }
      val = dr::select(
          use_trans,
          opts.trans_only ? Spectrum(trans_val) : (1.0f - f_c) * trans_val,
          val);
      pdf = dr::select(use_trans, (1.0f - p_r) * trans_pdf, pdf);
    }
  }
  return {val, pdf};
}

template <typename Float>
std::pair<Vector<Float, 3>, dr::mask_t<Float>> oren_nayar_sample(
    const Vector<Float, 3>& wi_l, const Point<Float, 2>& u2) {
  Float cos_i = Frame<Float>::cos_theta(wi_l);
  Vector<Float, 3> wo_l = warp::square_to_cosine_hemisphere(u2);
  wo_l.z() = dr::mulsign(wo_l.z(), cos_i);
  return {wo_l, cos_i != 0.0f};
}

template <typename Float, typename Spectrum>
std::pair<Spectrum, Float> oren_nayar_eval_pdf(const Vector<Float, 3>& wi_l,
                                               const Vector<Float, 3>& wo_l,
                                               const Spectrum& color,
                                               const Float& roughness,
                                               bool energy_comp) {
  Float cos_i = Frame<Float>::cos_theta(wi_l),
        cos_o = Frame<Float>::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i), n_dot_l = dr::mulsign(cos_o, cos_i);
  auto valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f);
  Float pdf_val = n_dot_l * dr::InvPi<Float>;
  Float l_dot_v = dr::dot(wi_l, wo_l);
  Spectrum val =
      energy_comp ? oren_nayar_compensated(n_dot_v, n_dot_l, l_dot_v, roughness,
                                           color) *
                        pdf_val
                  : color * (pdf_val * oren_nayar_diffuse(n_dot_v, n_dot_l,
                                                          l_dot_v, roughness));
  return {dr::select(valid, val, 0.0f), dr::select(valid, pdf_val, 0.0f)};
}

template <typename Float, typename Spectrum>
std::pair<Spectrum, Float> burley_diffuse_eval_pdf(const Vector<Float, 3>& wi_l,
                                                   const Vector<Float, 3>& wo_l,
                                                   const Spectrum& color,
                                                   const Float& roughness) {
  Float cos_i = Frame<Float>::cos_theta(wi_l),
        cos_o = Frame<Float>::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i), n_dot_l = dr::mulsign(cos_o, cos_i);
  auto valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f);
  Float pdf_val = n_dot_l * dr::InvPi<Float>;
  // For unit vectors, `2 * (wo_l . h)^2 = 1 + wi_l . wo_l`.
  Float f90_m1 = dr::fmadd(roughness, 1.0f + dr::dot(wi_l, wo_l), -0.5f);
  Float burley = dr::fmadd(f90_m1, pow5(1.0f - n_dot_l), 1.0f) *
                 dr::fmadd(f90_m1, pow5(1.0f - n_dot_v), 1.0f);
  return {dr::select(valid, color * (pdf_val * burley), 0.0f),
          dr::select(valid, pdf_val, 0.0f)};
}

// Sheen directional albedo: Zeltner et al. (2022) LTC
// (`mx_zeltner_sheen_dir_albedo`) or Imageworks / Conty-Kulla rational
// quadratic fit
// (`mx_imageworks_sheen_dir_albedo_analytic`).
template <typename Float>
Float sheen_dir_albedo(const Float& n_dot_v, const Float& roughness,
                       bool zeltner) {
  Float x = dr::clip(n_dot_v, 0.0f, 1.0f);
  if (zeltner) {
    Float y = dr::clip(roughness, 0.01f, 1.0f);
    Float s = y * dr::horner(y, 0.0206607f, 1.58491f) /
              dr::horner(y, 0.0379424f, 1.32227f, 1.0f);
    Float m = y *
              dr::horner(y, -0.193854f, -1.14885f, 1.7932f, 0.0f, -0.95943f) /
              (0.046391f + y);
    Float o = y * dr::horner(y, 0.000654023f, -0.0207818f, 0.119681f) /
              dr::horner(y, 1.26264f, -1.92021f, 1.0f);
    return dr::clip(
        dr::exp(-0.5f * dr::square((x - m) / s)) * dr::InvSqrtTwoPi<Float> / s +
            o,
        0.0f, 1.0f);
  }
  Float y = dr::clip(roughness, 0.0f, 1.0f);
  Float x2 = dr::square(x), y2 = dr::square(y), xy = x * y;
  Float rx = 13.673f - 68.78018f * x + 799.08825f * y - 905.00061f * xy +
             60.28956f * x2 + 1086.96473f * y2;
  Float ry = 1.0f + 61.57746f * x + 442.78211f * y + 2597.49308f * xy +
             121.81241f * x2 + 3045.55075f * y2;
  return dr::clip(rx / ry, 0.0f, 1.0f);
}

// Returns `(a_inv, b_inv, tx, ty)` for Zeltner LTC sheen oriented by `wi_pos`
// (local frame with `wi_pos.z() >= 0`).
template <typename Float>
std::tuple<Float, Float, Float, Float> zeltner_sheen_ltc(
    const Vector<Float, 3>& wi_pos, const Float& roughness) {
  const Float& x = wi_pos.z();
  Float y = dr::clip(roughness, 0.01f, 1.0f);
  Float a_inv = (2.58126f * x + 0.813703f * y) * y /
                (1.0f + 0.310327f * dr::square(x) + 2.60994f * x * y);
  Float b_inv = dr::safe_sqrt(1.0f - x) * (y - 1.0f) * y * dr::square(y) /
                (0.0000254053f + 1.71228f * x - 1.71506f * x * y +
                 1.34174f * dr::square(y));
  auto t = safe_normalize(dr::head<2>(wi_pos), dr::Array<Float, 2>(1.0f, 0.0f),
                          kMinLen2<dr::scalar_t<Float>>);
  return {a_inv, b_inv, t.x(), t.y()};
}

template <typename Float>
std::pair<Vector<Float, 3>, dr::mask_t<Float>> sheen_sample(
    const Vector<Float, 3>& wi_l, const Float& roughness, bool zeltner,
    const Point<Float, 2>& u2) {
  if (!zeltner) return oren_nayar_sample(wi_l, u2);
  using Vector3f = Vector<Float, 3>;
  Float cos_i = Frame<Float>::cos_theta(wi_l), n_dot_v = dr::abs(cos_i);
  auto [a_inv, b_inv, tx, ty] =
      zeltner_sheen_ltc(Vector3f(wi_l.x(), wi_l.y(), n_dot_v), roughness);
  Vector3f wo = warp::square_to_cosine_hemisphere(u2);
  Vector3f w =
      dr::normalize(Vector3f(wo.x() - b_inv * wo.z(), wo.y(), a_inv * wo.z()));
  Vector3f wo_l(tx * w.x() - ty * w.y(), ty * w.x() + tx * w.y(),
                dr::mulsign(w.z(), cos_i));
  return {wo_l, (n_dot_v > 0.0f) && (w.z() > 0.0f)};
}

template <typename Float, typename Spectrum>
std::pair<Spectrum, Float> sheen_eval_pdf(const Vector<Float, 3>& wi_l,
                                          const Vector<Float, 3>& wo_l,
                                          const Spectrum& color,
                                          const Float& roughness,
                                          bool zeltner) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Vector3f = Vector<Float, 3>;
  Float cos_i = Frame<Float>::cos_theta(wi_l),
        cos_o = Frame<Float>::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i), n_dot_l = dr::mulsign(cos_o, cos_i);
  auto valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f);
  Vector3f wi_pos(wi_l.x(), wi_l.y(), n_dot_v),
      wo_pos(wo_l.x(), wo_l.y(), n_dot_l);
  if (zeltner) {
    auto [a_inv, b_inv, tx, ty] = zeltner_sheen_ltc(wi_pos, roughness);
    Float x = tx * wo_pos.x() + ty * wo_pos.y(),
          y = tx * wo_pos.y() - ty * wo_pos.x();
    Vector3f w(a_inv * x + b_inv * n_dot_l, a_inv * y, n_dot_l);
    Float density = dr::select(
        valid,
        n_dot_l * dr::InvPi<Float> *
            dr::square(a_inv / dr::select(valid, dr::squared_norm(w), 1.0f)),
        0.0f);
    return {color * (sheen_dir_albedo(n_dot_v, roughness, /*zeltner=*/true) *
                     density),
            density};
  }
  Float pdf_val = n_dot_l * dr::InvPi<Float>;
  Float inv_r = dr::rcp(dr::clip(roughness, 0.005f, 1.0f));
  Vector3f h = safe_normalize(wi_pos + wo_pos, Vector3f(0.0f, 0.0f, 1.0f),
                              kMinLen2<ScalarFloat>);
  Float sin2 = Frame<Float>::sin_theta_2(h);
  auto pos_sin = sin2 > 0.0f;
  Float d_pi = dr::select(
      pos_sin,
      (2.0f + inv_r) * dr::pow(dr::select(pos_sin, sin2, 1.0f), 0.5f * inv_r) *
          0.5f,
      0.0f);
  Float denom = 4.0f * (n_dot_l + n_dot_v - n_dot_l * n_dot_v);
  return {dr::select(
              valid,
              color * (pdf_val * d_pi / dr::maximum(denom, dr::Epsilon<Float>)),
              0.0f),
          dr::select(valid, pdf_val, 0.0f)};
}

template <typename Float>
std::pair<Vector<Float, 3>, dr::mask_t<Float>> translucent_sample(
    const Vector<Float, 3>& wi_l, const Point<Float, 2>& u2) {
  Float cos_i = Frame<Float>::cos_theta(wi_l);
  Vector<Float, 3> wo_l = warp::square_to_cosine_hemisphere(u2);
  wo_l.z() = dr::mulsign(wo_l.z(), -cos_i);
  return {wo_l, cos_i != 0.0f};
}

template <typename Float, typename Spectrum>
std::pair<Spectrum, Float> translucent_eval_pdf(const Vector<Float, 3>& wi_l,
                                                const Vector<Float, 3>& wo_l,
                                                const Spectrum& color) {
  Float cos_i = Frame<Float>::cos_theta(wi_l),
        cos_o = Frame<Float>::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i), n_dot_l = dr::mulsign(cos_o, -cos_i);
  auto valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f);
  Float pdf_val = n_dot_l * dr::InvPi<Float>;
  return {dr::select(valid, color * pdf_val, 0.0f),
          dr::select(valid, pdf_val, 0.0f)};
}

// Longitudinal (`Mp`) and trimmed-logistic azimuthal (`Np`) scattering
// functions for Chiang et al. (2016) hair (`mx_chiang_hair_bsdf.glsl`).
template <typename Float>
Float hair_mp(const Float& sin_theta_i, const Float& cos_theta_i,
              const Float& sin_theta_o, const Float& cos_theta_o,
              const Float& v) {
  Float inv_v = dr::rcp(v);
  Float a = cos_theta_i * cos_theta_o * inv_v;
  Float st = sin_theta_i * sin_theta_o;
  Float i0_lo = warp::detail::i0(dr::minimum(a, 12.0f));
  Float a_hi = dr::maximum(a, 12.0f);
  Float log_i0 = dr::select(
      a > 12.0f,
      a + 0.5f * (-dr::log(dr::TwoPi<Float> * a_hi) + dr::rcp(8.0f * a_hi)),
      dr::log(i0_lo));
  // Bound `inv_v_lo` so eager SIMD/JIT evaluation of the `v >= 0.1` branch
  // never overflows `sinh` in float32 when `v` is small.
  Float inv_v_lo = dr::minimum(inv_v, 20.0f);
  return dr::select(
      v < 0.1f, dr::exp(log_i0 - st * inv_v - inv_v + dr::log(inv_v)),
      dr::exp(-st * inv_v_lo) * i0_lo * inv_v_lo / (2.0f * dr::sinh(inv_v_lo)));
}

template <typename Float>
Float hair_trimmed_logistic(const Float& x, const Float& s) {
  Float e = dr::exp(-dr::abs(x) / s);
  Float e_pi = dr::exp(-dr::Pi<Float> / s);
  return e / (s * dr::square(1.0f + e)) * (1.0f + e_pi) /
         dr::maximum(1.0f - e_pi, 1e-7f);
}

template <typename Float>
Float hair_trimmed_logistic_sample(const Float& u, const Float& s) {
  Float e_pi = dr::exp(-dr::Pi<Float> / s);
  return dr::clip(-s * dr::log(dr::lerp(Float(1.0f), e_pi, u) /
                               dr::lerp(e_pi, Float(1.0f), u)),
                  -dr::Pi<Float>, dr::Pi<Float>);
}

template <typename Float, typename UnpolSpec>
struct ChiangHairParams {
  UnpolSpec ap[4];
  Float ap_pdf[4], sin_vp[4], cos_vp[4], v[4], s_az[3];
  Float gamma_t, phi_i;
};

template <typename Float>
Vector<Float, 3> to_hair_local(const Vector<Float, 3>& v, const Float& cos_i) {
  return Vector<Float, 3>(v.x(), v.y(), dr::mulsign(v.z(), cos_i));
}

template <typename Float, typename UnpolSpec>
Float hair_phi(const Float& p, const ChiangHairParams<Float, UnpolSpec>& hp) {
  return p * (2.0f * hp.gamma_t + dr::Pi<Float>)-2.0f * hp.phi_i +
         dr::Pi<Float>;
}

template <typename Float, typename Spectrum, typename UnpolSpec>
ChiangHairParams<Float, UnpolSpec> build_chiang_hair_params(
    const Vector<Float, 3>& wi_l, const UnpolSpec& tint_r,
    const UnpolSpec& tint_tt, const UnpolSpec& tint_trt,
    const Vector<Float, 2>& r_r, const Vector<Float, 2>& r_tt,
    const Vector<Float, 2>& r_trt, const Float& ior_raw,
    const UnpolSpec& sigma_a, const Float& cuticle_angle) {
  using Scalar = dr::scalar_t<Float>;
  Vector<Float, 3> wi_h = to_hair_local(wi_l, Frame<Float>::cos_theta(wi_l));
  Float ior = dr::maximum(ior_raw, 1.0001f);

  Float sin_theta_v = dr::clip(wi_h.x(), -1.0f, 1.0f);
  Float cos_theta_v = warp::circ(sin_theta_v);

  Float inv_eta_p = dr::safe_rsqrt(dr::square(ior) - dr::square(sin_theta_v));
  Float sin_gamma_t = dr::clip(-wi_h.y() * inv_eta_p, -1.0f, 1.0f);
  Float cos_gamma_t = warp::circ(sin_gamma_t);

  Float f = std::get<0>(fresnel(dr::clip(wi_h.z(), 0.0f, 1.0f), ior));
  UnpolSpec tr = dr::exp(-sigma_a * (2.0f * cos_gamma_t * ior * inv_eta_p));
  UnpolSpec a1 = dr::square(1.0f - f) * tr;
  UnpolSpec tr_f = tr * f;

  ChiangHairParams<Float, UnpolSpec> hp;
  hp.ap[0] = tint_r * f;
  hp.ap[1] = tint_tt * a1;
  hp.ap[2] = tint_trt * (a1 * tr_f);
  hp.ap[3] = hp.ap[2] * tr_f / dr::maximum(1.0f - tr_f, 1e-5f);

  Float w[4] = {selection_weight<Spectrum>(hp.ap[0]),
                selection_weight<Spectrum>(hp.ap[1]),
                selection_weight<Spectrum>(hp.ap[2]),
                selection_weight<Spectrum>(hp.ap[3])};
  Float w_sum = w[0] + w[1] + w[2] + w[3];
  auto ok = w_sum > 0.0f;
  Float inv_w = dr::rcp(dr::select(ok, w_sum, 1.0f));
  for (int p = 0; p < 4; ++p) {
    hp.ap_pdf[p] = dr::detach(dr::select(ok, w[p] * inv_w, 0.25f));
  }

  auto [sa, ca] = dr::sincos((cuticle_angle - 0.5f) * dr::Pi<Float>);
  Float s2a = 2.0f * sa * ca, c2a = dr::square(ca) - dr::square(sa);
  Float s4a = 2.0f * s2a * c2a, c4a = dr::square(c2a) - dr::square(s2a);
  hp.sin_vp[0] = sin_theta_v * c2a + cos_theta_v * s2a;
  hp.cos_vp[0] = dr::abs(cos_theta_v * c2a - sin_theta_v * s2a);
  hp.sin_vp[1] = sin_theta_v * ca - cos_theta_v * sa;
  hp.cos_vp[1] = dr::abs(cos_theta_v * ca + sin_theta_v * sa);
  hp.sin_vp[2] = sin_theta_v * c4a - cos_theta_v * s4a;
  hp.cos_vp[2] = dr::abs(cos_theta_v * c4a + sin_theta_v * s4a);
  hp.sin_vp[3] = sin_theta_v;
  hp.cos_vp[3] = cos_theta_v;

  hp.v[0] = r_r.x();
  hp.v[1] = r_tt.x();
  hp.v[2] = hp.v[3] = r_trt.x();
  constexpr Scalar kSqrtPiOver8 = Scalar(0.25) * dr::SqrtTwoPi<Scalar>;
  hp.s_az[0] = r_r.y() * kSqrtPiOver8;
  hp.s_az[1] = r_tt.y() * kSqrtPiOver8;
  hp.s_az[2] = r_trt.y() * kSqrtPiOver8;
  hp.gamma_t = dr::safe_asin(sin_gamma_t);
  hp.phi_i = mx_safe_atan2(wi_h.z(), wi_h.y());
  return hp;
}

template <typename Float, typename UnpolSpec>
std::tuple<Vector<Float, 3>, dr::uint32_array_t<Float>, dr::mask_t<Float>>
chiang_hair_sample(const ChiangHairParams<Float, UnpolSpec>& hp,
                   const Vector<Float, 3>& wi_l, bool refl_on, bool trans_on,
                   const Float& sample1, const Point<Float, 2>& sample2) {
  using Mask = dr::mask_t<Float>;
  using UInt32 = dr::uint32_array_t<Float>;
  using Vector3f = Vector<Float, 3>;
  Float u0 = dr::clip(sample1, 0.0f, dr::OneMinusEpsilon<Float>);
  Float u_rem = u0, p_pdf = hp.ap_pdf[0];
  Mask rem[3] = {(hp.ap_pdf[1] + hp.ap_pdf[2] + hp.ap_pdf[3]) > 0.0f,
                 (hp.ap_pdf[2] + hp.ap_pdf[3]) > 0.0f, hp.ap_pdf[3] > 0.0f};
  UInt32 p(0);
  Float v_p = hp.v[0], sin_vp = hp.sin_vp[0], cos_vp = hp.cos_vp[0],
        s_p = hp.s_az[0];
  for (uint32_t j = 0; j < 3; ++j) {
    Mask next = (u0 >= hp.ap_pdf[j]) && rem[j];
    u0 -= hp.ap_pdf[j];
    p = dr::select(next, UInt32(j + 1), p);
    u_rem = dr::select(next, u0, u_rem);
    p_pdf = dr::select(next, hp.ap_pdf[j + 1], p_pdf);
    v_p = dr::select(next, hp.v[j + 1], v_p);
    sin_vp = dr::select(next, hp.sin_vp[j + 1], sin_vp);
    cos_vp = dr::select(next, hp.cos_vp[j + 1], cos_vp);
    if (j < 2) s_p = dr::select(next, hp.s_az[j + 1], s_p);
  }
  Float u_az = dr::clip(u_rem / dr::select(p_pdf > 0.0f, p_pdf, 1.0f), 1e-6f,
                        1.0f - 1e-6f);

  Vector3f vmf = warp::square_to_von_mises_fisher(sample2, dr::rcp(v_p));
  Float sin_theta_o =
      dr::clip(vmf.x() * cos_vp - vmf.z() * sin_vp, -1.0f, 1.0f);
  Float cos_theta_o = warp::circ(sin_theta_o);

  Float delta_phi = dr::select(
      p < 3u, hair_phi(Float(p), hp) + hair_trimmed_logistic_sample(u_az, s_p),
      dr::TwoPi<Float> * u_az);
  auto [sin_phi_o, cos_phi_o] = dr::sincos(hp.phi_i + delta_phi);
  Vector3f wo_h(sin_theta_o, cos_theta_o * cos_phi_o, cos_theta_o * sin_phi_o);
  Vector3f wo_l = to_hair_local(wo_h, Frame<Float>::cos_theta(wi_l));
  Mask is_refl = wo_h.z() > 0.0f;
  UInt32 sampled_type =
      dr::select(is_refl, UInt32(+BSDFFlags::GlossyReflection),
                 UInt32(+BSDFFlags::GlossyTransmission));
  Mask dir_ok = dr::select(is_refl, Mask(refl_on), Mask(trans_on));
  return {wo_l, sampled_type,
          dir_ok && (wi_l.z() != 0.0f) && (wo_h.z() != 0.0f)};
}

template <typename Float, typename UnpolSpec>
std::pair<UnpolSpec, Float> chiang_hair_eval_pdf(
    const ChiangHairParams<Float, UnpolSpec>& hp, const Vector<Float, 3>& wi_l,
    const Vector<Float, 3>& wo_l, bool refl_on, bool trans_on) {
  using Mask = dr::mask_t<Float>;
  Vector<Float, 3> wo_h = to_hair_local(wo_l, Frame<Float>::cos_theta(wi_l));
  Mask dir_ok = dr::select(wo_h.z() > 0.0f, Mask(refl_on), Mask(trans_on));
  Mask valid = (wi_l.z() != 0.0f) && (wo_h.z() != 0.0f) && dir_ok;

  Float sin_theta_o = dr::clip(wo_h.x(), -1.0f, 1.0f);
  Float cos_theta_o = warp::circ(sin_theta_o);
  Float phi = mx_safe_atan2(wo_h.z(), wo_h.y()) - hp.phi_i;

  UnpolSpec val(0.0f);
  Float pdf(0.0f);
  for (int p = 0; p < 3; ++p) {
    Float mp =
        hair_mp(hp.sin_vp[p], hp.cos_vp[p], sin_theta_o, cos_theta_o, hp.v[p]);
    Float dphi = phi - hair_phi(Float(p), hp);
    dphi -= dr::TwoPi<Float> * dr::round(dphi * dr::InvTwoPi<Float>);
    Float lobe = mp * hair_trimmed_logistic(dphi, hp.s_az[p]);
    val = dr::fmadd(hp.ap[p], lobe, val);
    pdf = dr::fmadd(hp.ap_pdf[p], lobe, pdf);
  }
  Float mp3 =
      hair_mp(hp.sin_vp[3], hp.cos_vp[3], sin_theta_o, cos_theta_o, hp.v[3]) *
      dr::InvTwoPi<Float>;
  val = dr::fmadd(hp.ap[3], mp3, val);
  pdf = dr::fmadd(hp.ap_pdf[3], mp3, pdf);
  Mask ok = valid && dr::isfinite(pdf);
  return {dr::select(ok, val, 0.0f), dr::select(ok, pdf, 0.0f)};
}

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
