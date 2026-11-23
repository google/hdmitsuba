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

template <typename ScalarFloat>
inline constexpr ScalarFloat kMinCos = ScalarFloat(1e-6f);

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

// Energy-compensated GGX directional albedo from `(F0, F(n_dot_v))`.
template <typename Float, typename Spectrum>
Spectrum ggx_refl_albedo(const Float& n_dot_v, const Float& alpha,
                         const Spectrum& f0, const Spectrum& f_nv) {
  auto [a, b] = ggx_dir_albedo_ab(n_dot_v, alpha);
  return dr::minimum(ggx_energy_compensation(a + b, f_nv) * dr::fmadd(f0, a, b),
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

template <typename Float>
struct DielectricSample {
  Vector<Float, 3> wo_l;
  dr::mask_t<Float> is_refl;
  Float eta;
  dr::mask_t<Float> valid;
};

// Follows `roughdielectric`: sample visible normals for `wi` flipped into the
// upper hemisphere, then reflect/refract the unflipped `wi`.
template <typename Float>
DielectricSample<Float> ggx_dielectric_sample(const Vector<Float, 3>& wi_l,
                                              const Vector<Float, 2>& alpha,
                                              const Float& ior, bool refl_on,
                                              bool trans_on, bool thin_walled,
                                              bool two_sided, const Float& u1,
                                              const Point<Float, 2>& u2) {
  using Mask = dr::mask_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Normal3f = Normal<Float, 3>;
  using Frame3f = Frame<Float>;
  Float cos_i = Frame3f::cos_theta(wi_l);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Normal3f m = distr.sample(dr::mulsign(wi_l, cos_i), u2).first;
  Vector3f wo_l = reflect(wi_l, m);
  Mask do_reflect(refl_on);
  Mask trans_ok(false);
  Float out_eta(1.0f);
  if (trans_on) {
    auto [fres, cos_theta_t, eta_it, eta_ti] =
        exact_fresnel(dr::dot(wi_l, m), ior, two_sided);
    trans_ok = fres < 1.0f;
    if (refl_on) do_reflect = !trans_ok || (u1 < fres);
    Vector3f wo_trans = thin_walled ? Vector3f(wo_l.x(), wo_l.y(), -wo_l.z())
                                    : refract(wi_l, m, cos_theta_t, eta_ti);
    wo_l = dr::select(do_reflect, wo_l, wo_trans);
    if (!thin_walled) out_eta = dr::select(do_reflect, Float(1.0f), eta_it);
  }
  Float cos_o = Frame3f::cos_theta(wo_l);
  Mask valid = dr::select(do_reflect, cos_i * cos_o > 0.0f,
                          trans_ok && (cos_i * cos_o < 0.0f));
  return {wo_l, do_reflect, out_eta, valid};
}

// Follows `roughdielectric::eval()` / `pdf()`, using height-correlated Smith G2
// for reflection to match MaterialX `pbrlib`.
template <typename Float>
std::pair<Float, Float> ggx_dielectric_eval_pdf(
    const Vector<Float, 3>& wi_l, const Vector<Float, 3>& wo_l,
    const Vector<Float, 2>& alpha, const Float& ior, bool trans_only,
    bool refl_on, bool trans_on, bool thin_walled, bool two_sided,
    TransportMode transport_mode) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Mask = dr::mask_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Frame3f = Frame<Float>;
  bool solid_trans = trans_on && !thin_walled;
  Float cos_i = Frame3f::cos_theta(wi_l), cos_o = Frame3f::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Mask is_refl = cos_i * cos_o > 0.0f;
  Vector3f wo_m =
      (trans_on && thin_walled)
          ? Vector3f(wo_l.x(), wo_l.y(), dr::mulsign(dr::abs(cos_o), cos_i))
          : wo_l;
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

  Float f_d = std::get<0>(exact_fresnel(cos_im, ior, two_sided));
  Float g1_i = distr.smith_g1(wi_l, m), g1_o = distr.smith_g1(wo_m, m);
  Float d_g1_over_cos = distr.eval(m) * g1_i / dr::select(valid, n_dot_v, 1.0f);
  Float refl_pdf = 0.25f * d_g1_over_cos;
  Float val(0.0f), pdf(0.0f);
  if (refl_on) {
    Mask use_refl = valid && is_refl;
    if (dr::any_or<true>(use_refl)) {
      auto [a, b] = ggx_dir_albedo_ab(n_dot_v, dr::sqrt(dr::prod(alpha)));
      val = dr::select(use_refl,
                       f_d * ggx_energy_compensation(a + b, f_d) *
                           (refl_pdf * ggx_g2_over_g1(g1_i, g1_o)),
                       val);
      pdf = dr::select(use_refl, trans_on ? f_d * refl_pdf : refl_pdf, pdf);
    }
  }
  if (trans_on) {
    Mask use_trans = valid && !is_refl && (f_d < 1.0f);
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
      val = dr::select(use_trans,
                       trans_only ? trans_val : (1.0f - f_d) * trans_val, val);
      pdf = dr::select(use_trans,
                       refl_on ? (1.0f - f_d) * trans_pdf : trans_pdf, pdf);
    }
  }
  return {val, pdf};
}

template <typename Float>
Float ggx_dielectric_refl_albedo(const Float& cos_i,
                                 const Vector<Float, 2>& alpha,
                                 const Float& ior, bool two_sided) {
  Float n_dot_v = dr::abs(cos_i);
  auto [f_smooth, cos_t, eta_it, eta_ti] = exact_fresnel(cos_i, ior, two_sided);
  Float f0 = dr::select(eta_it < 1.0f, f_smooth,
                        dr::square((ior - 1.0f) / (ior + 1.0f)));
  return dr::select(
      ior == 1.0f, Float(0.0f),
      ggx_refl_albedo(n_dot_v, dr::sqrt(dr::prod(alpha)), f0, f_smooth));
}

template <typename Float>
std::pair<Vector<Float, 3>, dr::mask_t<Float>> ggx_conductor_sample(
    const Vector<Float, 3>& wi_l, const Vector<Float, 2>& alpha,
    const Point<Float, 2>& u2) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Frame3f = Frame<Float>;
  Float cos_i = Frame3f::cos_theta(wi_l);
  Float n_dot_v = dr::abs(cos_i);
  Vector3f wi_pos(wi_l.x(), wi_l.y(), n_dot_v);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Vector3f wo_l = reflect(wi_pos, distr.sample(wi_pos, u2).first);
  auto valid = (n_dot_v > kMinCos<ScalarFloat>) &&
               (Frame3f::cos_theta(wo_l) > kMinCos<ScalarFloat>);
  wo_l.z() = dr::mulsign(wo_l.z(), cos_i);
  return {wo_l, valid};
}

template <typename Float, typename Spectrum>
std::pair<Spectrum, Float> ggx_conductor_eval_pdf(
    const Vector<Float, 3>& wi_l, const Vector<Float, 3>& wo_l,
    const Vector<Float, 2>& alpha, const dr::Complex<Spectrum>& eta) {
  using ScalarFloat = dr::scalar_t<Float>;
  using Vector3f = Vector<Float, 3>;
  using Frame3f = Frame<Float>;
  Float cos_i = Frame3f::cos_theta(wi_l), cos_o = Frame3f::cos_theta(wo_l);
  Float n_dot_v = dr::abs(cos_i), n_dot_l = dr::mulsign(cos_o, cos_i);
  Vector3f wi_pos(wi_l.x(), wi_l.y(), n_dot_v),
      wo_pos(wo_l.x(), wo_l.y(), n_dot_l);
  auto valid =
      (n_dot_v > kMinCos<ScalarFloat>) && (n_dot_l > kMinCos<ScalarFloat>);
  MicrofacetDistribution<Float, Color<Float, 3>> distr(MicrofacetType::GGX,
                                                       alpha.x(), alpha.y());
  Vector3f h = safe_normalize(wi_pos + wo_pos, Vector3f(0.0f, 0.0f, 1.0f),
                              kMinLen2<ScalarFloat>);
  Float g1_i = distr.smith_g1(wi_pos, h), g1_o = distr.smith_g1(wo_pos, h);
  Float d_g1_over_4cos =
      distr.eval(h) * g1_i / (4.0f * dr::select(valid, n_dot_v, 1.0f));
  Spectrum fres = fresnel_conductor(
      Spectrum(dr::maximum(dr::dot(wi_pos, h), kMinCos<ScalarFloat>)), eta);
  auto [a, b] = ggx_dir_albedo_ab(n_dot_v, dr::sqrt(dr::prod(alpha)));
  return {dr::select(valid,
                     fres * ggx_energy_compensation(a + b, fres) *
                         (d_g1_over_4cos * ggx_g2_over_g1(g1_i, g1_o)),
                     0.0f),
          dr::select(valid, d_g1_over_4cos, 0.0f)};
}

template <typename Float, typename Spectrum>
Spectrum ggx_conductor_albedo(const Float& n_dot_v,
                              const Vector<Float, 2>& alpha,
                              const dr::Complex<Spectrum>& eta) {
  Float ndv = dr::maximum(n_dot_v, dr::Epsilon<Float>);
  return ggx_refl_albedo(ndv, dr::sqrt(dr::prod(alpha)),
                         fresnel_conductor(Spectrum(1.0f), eta),
                         fresnel_conductor(Spectrum(ndv), eta));
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
