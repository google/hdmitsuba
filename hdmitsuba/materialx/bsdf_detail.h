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

}  // namespace mtlx::detail
NAMESPACE_END(mitsuba)
