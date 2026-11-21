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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/microfacet.h>

#include "hdmitsuba/materialx/bsdf_detail.h"
#include "hdmitsuba/materialx/evaluator.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {
namespace {

/**!

.. _bsdf-materialx_bsdf:

MaterialX BSDF (:monosp:`materialx_bsdf`)
-----------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxProgram`` bytecode, closure table, and
     constant/opaque tables.

 * - in0, in1, ...
   - |texture|
   - Nested input textures referenced by ``Op::SampleInput`` and
     ``Op::SampleInputUV``.
   - |exposed|, |differentiable|

 * - param0, param1, ...
   - |float|
   - Differentiable opaque scalar parameters exposed through traversal.
   - |exposed|, |differentiable|

Evaluates and importance-samples a compiled MaterialX closure over surface
interactions.

*/
template <typename Float, typename Spectrum>
class MtlxBsdf final : public BSDF<Float, Spectrum> {
 public:
  MI_IMPORT_BASE(BSDF, m_flags, m_components)
  MI_IMPORT_TYPES(MicrofacetDistribution)
  using Val = typename MtlxEvaluator<Float, Spectrum>::Val;

  explicit MtlxBsdf(const Properties& props) : Base(props) {
    m_eval.compile(props);
    const std::vector<ClosureNode>& closures = prog().closures;

    bool need_lut = false;
    uint32_t surf_flags = 0u;
    for (size_t i = 0; i < closures.size(); ++i) {
      const ClosureNode& c = closures[i];
      if (is_leaf(c.op)) {
        std::pair<int, int> key(c.args[leaf_arg::kNormal],
                                c.args[leaf_arg::kTangent]);
        auto it =
            std::find(m_unique_frames.begin(), m_unique_frames.end(), key);
        size_t f_idx = static_cast<size_t>(it - m_unique_frames.begin());
        if (it == m_unique_frames.end()) m_unique_frames.push_back(key);
        m_leaves.push_back({i, f_idx});
        m_surface_types |= leaf_type(c);
        if (c.op == ClosureOp::Conductor || c.op == ClosureOp::Dielectric) {
          surf_flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
        }
        if (key.first >= 0) surf_flags |= +BSDFFlags::NormalMapped;
      } else if (c.op == ClosureOp::Layer) {
        m_has_layers = true;
        surf_flags |= +BSDFFlags::NonSymmetric;
      }
      if constexpr (is_spectral_v<Spectrum>) {
        int c_slot = is_leaf(c.op) ? c.args[leaf_arg::kColor]
                                   : c.args[combinator_arg::kWeight];
        need_lut |= c_slot >= 0 && !prog().is_tmp_uniform(c_slot);
        if (c.op == ClosureOp::Conductor) {
          need_lut |= !prog().is_tmp_uniform(c.args[leaf_arg::kExtinction]);
        }
      }
    }
    if (need_lut) m_eval.load_srgb_lut();

    uint32_t sv =
        m_eval.is_spatially_varying() ? +BSDFFlags::SpatiallyVarying : 0u;
    if (m_surface_types != 0u) {
      m_components.push_back(m_surface_types | surf_flags |
                             BSDFFlags::FrontSide | BSDFFlags::BackSide | sv);
    }
    if (prog().opacity_slot >= 0) {
      m_null_component = static_cast<uint32_t>(m_components.size());
      m_components.push_back(BSDFFlags::Null | BSDFFlags::FrontSide |
                             BSDFFlags::BackSide | sv);
    }
    m_flags = +BSDFFlags::Empty;
    for (uint32_t f : m_components) m_flags |= f;
  }

  void traverse(TraversalCallback* cb) override { m_eval.traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    m_eval.parameters_changed();
  }

  std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext& ctx,
                                           const SurfaceInteraction3f& si,
                                           Float sample1,
                                           const Point2f& sample2,
                                           Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);
    if (unlikely(dr::none_or<false>(active) ||
                 (!has_surface_enabled(ctx) && !has_cutout(ctx)))) {
      return {dr::zeros<BSDFSample3f>(), 0.0f};
    }
    return sample_impl(ctx, si, eval_state(ctx, si, active), sample1, sample2,
                       active);
  }

  Spectrum eval(const BSDFContext& ctx, const SurfaceInteraction3f& si,
                const Vector3f& wo, Mask active) const override {
    return eval_pdf(ctx, si, wo, active).first;
  }

  Float pdf(const BSDFContext& ctx, const SurfaceInteraction3f& si,
            const Vector3f& wo, Mask active) const override {
    return eval_pdf(ctx, si, wo, active).second;
  }

  std::pair<Spectrum, Float> eval_pdf(const BSDFContext& ctx,
                                      const SurfaceInteraction3f& si,
                                      const Vector3f& wo,
                                      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (unlikely(dr::none_or<false>(active) || !has_surface_enabled(ctx))) {
      return {0.0f, 0.0f};
    }
    return eval_pdf_impl(ctx, eval_state(ctx, si, active), wo);
  }

  std::tuple<Spectrum, Float, BSDFSample3f, Spectrum> eval_pdf_sample(
      const BSDFContext& ctx, const SurfaceInteraction3f& si,
      const Vector3f& wo, Float sample1, const Point2f& sample2,
      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (unlikely(dr::none_or<false>(active) ||
                 (!has_surface_enabled(ctx) && !has_cutout(ctx)))) {
      return {0.0f, 0.0f, dr::zeros<BSDFSample3f>(), 0.0f};
    }
    State s = eval_state(ctx, si, active);
    return std::tuple_cat(eval_pdf_impl(ctx, s, wo),
                          sample_impl(ctx, si, s, sample1, sample2, active));
  }

  Spectrum eval_null(const SurfaceInteraction3f& si,
                     Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    int op_slot = prog().opacity_slot;
    if (op_slot < 0 || unlikely(dr::none_or<false>(active))) return 0.0f;
    return Spectrum(1.0f -
                    clip_opacity(m_eval.eval(si, active, nullptr, op_slot))) &
           active;
  }

  BSDFFeatures3f eval_features(const SurfaceInteraction3f& si,
                               Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (unlikely(dr::none_or<false>(active))) {
      return dr::zeros<BSDFFeatures3f>();
    }
    State s = eval_state(BSDFContext(), si, active);
    Float transparency = 1.0f - s.opacity;
    Mask has_trans = active && (transparency > 0.0f);
    UnpolarizedSpectrum diffuse_albedo(0.0f), specular_reflectance(0.0f),
        specular_transmittance = UnpolarizedSpectrum(transparency) & active;
    Float spec_roughness = dr::select(has_trans, 0.0f, 1.0f);
    Float best_spec_w = dr::select(has_trans, transparency, 0.0f);
    Float best_wt_w = best_spec_w;
    Vector3f wt = dr::select(has_trans, -si.to_world(si.wi), Vector3f(0.0f));
    Frame3f sh_frame = si.sh_frame;
    if (s.has_surface) {
      UnpolarizedSpectrum surf_transmittance(0.0f);
      for (size_t k = 0; k < m_leaves.size(); ++k) {
        auto [i, f_idx] = m_leaves[k];
        const ClosureNode& c = prog().closures[i];
        UnpolarizedSpectrum contrib = s.leaf_weight[k] * leaf_albedo(k, s);
        if (c.op == ClosureOp::Conductor || c.op == ClosureOp::Dielectric) {
          Float sw;
          if (c.op == ClosureOp::Conductor || scatter_mode(c) == kScatterR) {
            specular_reflectance += contrib;
            sw = selection_weight(contrib);
          } else {
            bool is_rt = scatter_mode(c) == kScatterRT;
            auto [f_smooth, cos_t, eta_it, eta_ti] =
                dielectric_fresnel(c, Frame3f::cos_theta(s.wi_local[f_idx]),
                                   dielectric_ior(c, s.tmps.data()));
            if (is_rt) specular_reflectance += contrib * f_smooth;
            UnpolarizedSpectrum trans = is_rt ? contrib * (1.0f - f_smooth)
                                              : (contrib & (f_smooth < 1.0f));
            surf_transmittance += trans;
            Float tw = selection_weight(trans);
            sw = is_rt ? selection_weight(contrib) : tw;
            Mask pick_wt = s.surf_active && (tw > best_wt_w);
            wt = dr::select(pick_wt,
                            is_thin_walled(c)
                                ? -si.to_world(si.wi)
                                : si.to_world(s.frames[f_idx].to_world(refract(
                                      s.wi_local[f_idx], cos_t, eta_ti))),
                            wt);
            best_wt_w = dr::select(pick_wt, tw, best_wt_w);
          }
          spec_roughness = dr::select(sw > best_spec_w,
                                      dr::max(get_roughness(c, s.tmps.data())),
                                      spec_roughness);
          best_spec_w = dr::maximum(best_spec_w, sw);
        } else {
          diffuse_albedo += contrib;
        }
      }
      diffuse_albedo = dr::minimum(diffuse_albedo, 1.0f) & s.surf_active;
      specular_reflectance =
          dr::minimum(specular_reflectance, 1.0f) & s.surf_active;
      specular_transmittance = dr::minimum(
          specular_transmittance + (surf_transmittance & s.surf_active), 1.0f);
      Frame3f f = s.frames[0];
      if (m_unique_frames.size() > 1) {
        Float best_w(0.0f);
        for (size_t k = 0; k < m_leaves.size(); ++k) {
          Float sw = selection_weight(s.leaf_weight[k]);
          f = dr::select(sw > best_w, s.frames[m_leaves[k].frame], f);
          best_w = dr::maximum(best_w, sw);
        }
      }
      sh_frame = Frame3f(si.sh_frame.to_world(f.s), si.sh_frame.to_world(f.t),
                         si.sh_frame.to_world(f.n));
    }
    return {diffuse_albedo, specular_reflectance, specular_transmittance,
            sh_frame,       spec_roughness,       wt};
  }

  std::string to_string() const override {
    std::ostringstream oss;
    oss << "MtlxBsdf[\n"
        << "  closures = " << prog().closures.size() << ",\n"
        << "  opacity_slot = " << prog().opacity_slot << ",\n"
        << m_eval.to_string() << "]";
    return oss.str();
  }

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb) override {
    if (cb.role != drjit::TraverseRole::Freeze &&
        cb.role != drjit::TraverseRole::Children) {
      return;
    }
    Base::traverse_cb(payload, cb);
    m_eval.traverse_cb(payload, cb);
  }

  MI_DECLARE_CLASS(MtlxBsdf)

 private:
  struct Leaf {
    size_t node, frame;
  };

  using LeafSpectra = absl::InlinedVector<UnpolarizedSpectrum, 8>;

  // Per-query material state shared by all BSDF entry points.
  struct State {
    absl::InlinedVector<Val, 32> tmps;
    // Deduplicated local shading frames and incident directions (relative to
    // `si.sh_frame`).
    absl::InlinedVector<Frame3f, 4> frames;
    absl::InlinedVector<Vector3f, 4> wi_local;
    // Per-leaf primary color (or conductor `ior`), conductor `extinction`,
    // combined linear weight (combinators * leaf weight), and detached sampling
    // probability (summing to 1 over active leaves).
    LeafSpectra leaf_color;
    LeafSpectra leaf_extinction;
    LeafSpectra leaf_weight;
    absl::InlinedVector<Float, 8> leaf_prob;
    Vector3f n_geom;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  static constexpr ScalarFloat kMinLen2 =
      dr::Epsilon<ScalarFloat> * dr::Epsilon<ScalarFloat>;
  static constexpr ScalarFloat kMinCos = 1e-6f;

  const MtlxProgram& prog() const { return m_eval.program(); }
  static Vector3f xyz(const Val& v) { return dr::head<3>(v); }
  static int scatter_mode(const ClosureNode& c) {
    return c.args[leaf_arg::kMode] & kScatterModeMask;
  }
  static bool is_thin_walled(const ClosureNode& c) {
    return (c.args[leaf_arg::kMode] & kScatterThinWalled) != 0;
  }

  static uint32_t leaf_type(const ClosureNode& c) {
    switch (c.op) {
      case ClosureOp::Translucent:
        return +BSDFFlags::DiffuseTransmission;
      case ClosureOp::Conductor:
        return +BSDFFlags::GlossyReflection;
      case ClosureOp::Dielectric: {
        int sm = scatter_mode(c);
        return (sm != kScatterT ? +BSDFFlags::GlossyReflection : 0u) |
               (sm != kScatterR ? +BSDFFlags::GlossyTransmission : 0u);
      }
      default:
        return +BSDFFlags::DiffuseReflection;
    }
  }

  static bool leaf_enabled(const BSDFContext& ctx, const ClosureNode& c) {
    return (leaf_type(c) & ctx.type_mask) != 0u;
  }

  static Vector2f get_roughness(const ClosureNode& c, const Val* tmps) {
    return dr::clip(dr::head<2>(tmps[c.args[leaf_arg::kRoughness]]), 1e-3f,
                    1.0f);
  }

  // Clamps IOR positive, and away from 1 for solid transmission to avoid the
  // `wi + eta * wo == 0` half-vector singularity.
  static Float dielectric_ior(const ClosureNode& c, const Val* tmps) {
    constexpr ScalarFloat kIorEps = 1e-4f;
    Float ior = dr::maximum(tmps[c.args[leaf_arg::kIor]].x(), kIorEps);
    if (is_thin_walled(c) || scatter_mode(c) == kScatterR) return ior;
    return dr::select(
        dr::abs(ior - 1.0f) < kIorEps,
        dr::select(ior < 1.0f, Float(1.0f - kIorEps), Float(1.0f + kIorEps)),
        ior);
  }

  // Wraps `mitsuba::fresnel`, handling two-sided dielectrics (`thin_walled` or
  // reflection-only) and clamping `F` to exact `1` under total internal
  // reflection (`cos_theta_t == 0`), where `dr::rcp` in `mitsuba::fresnel` can
  // round `F` just below `1` (e.g. `0.99999988f` on AVX/CUDA).
  static std::tuple<Float, Float, Float, Float> dielectric_fresnel(
      const ClosureNode& c, const Float& cos_theta_i, const Float& eta) {
    bool two_sided = (c.args[leaf_arg::kMode] & kScatterTwoSided) != 0;
    auto [f, cos_theta_t, eta_it, eta_ti] =
        fresnel(two_sided ? dr::abs(cos_theta_i) : cos_theta_i, eta);
    return {
        dr::select((cos_theta_t == 0.0f) && (eta_it < 1.0f), Float(1.0f), f),
        cos_theta_t, eta_it, eta_ti};
  }

  static Float dielectric_refl_albedo(const ClosureNode& c,
                                      const Vector3f& wi_l, const Val* tmps) {
    Float cos_i = Frame3f::cos_theta(wi_l), n_dot_v = dr::abs(cos_i),
          ior = dielectric_ior(c, tmps);
    auto [f_smooth, cos_t, eta_it, eta_ti] = dielectric_fresnel(c, cos_i, ior);
    Float f0 = dr::select(eta_it < 1.0f, f_smooth,
                          dr::square((ior - 1.0f) / (ior + 1.0f)));
    return dr::select(
        ior == 1.0f, Float(0.0f),
        detail::ggx_refl_albedo(
            n_dot_v, dr::sqrt(dr::prod(get_roughness(c, tmps))), f0, f_smooth));
  }

  // Clamps conductor `ior` away from zero to avoid `0 / 0` in
  // `fresnel_conductor` at normal incidence when `ior == extinction == 0`.
  static dr::Complex<UnpolarizedSpectrum> conductor_eta(size_t k,
                                                        const State& s) {
    return dr::Complex<UnpolarizedSpectrum>(
        dr::maximum(s.leaf_color[k], dr::Epsilon<Float>), s.leaf_extinction[k]);
  }

  UnpolarizedSpectrum leaf_albedo(size_t k, const State& s) const {
    auto [i, f_idx] = m_leaves[k];
    const ClosureNode& c = prog().closures[i];
    if (c.op == ClosureOp::Dielectric) {
      return scatter_mode(c) != kScatterR
                 ? s.leaf_color[k]
                 : s.leaf_color[k] * dielectric_refl_albedo(
                                         c, s.wi_local[f_idx], s.tmps.data());
    }
    if (c.op != ClosureOp::Conductor) return s.leaf_color[k];
    Float n_dot_v = dr::maximum(dr::abs(Frame3f::cos_theta(s.wi_local[f_idx])),
                                dr::Epsilon<Float>);
    dr::Complex<UnpolarizedSpectrum> eta_c = conductor_eta(k, s);
    return detail::ggx_refl_albedo(
        n_dot_v, dr::sqrt(dr::prod(get_roughness(c, s.tmps.data()))),
        fresnel_conductor(UnpolarizedSpectrum(1.0f), eta_c),
        fresnel_conductor(UnpolarizedSpectrum(n_dot_v), eta_c));
  }

  UnpolarizedSpectrum eval_color(int slot, const SurfaceInteraction3f& si,
                                 const Val* tmps,
                                 bool unbounded = false) const {
    const Val& v = tmps[slot];
    if (prog().is_tmp_uniform(slot)) {
      return UnpolarizedSpectrum(unbounded ? dr::maximum(v.x(), 0.0f)
                                           : dr::clip(v.x(), 0.0f, 1.0f));
    }
    Color3f c = unbounded ? dr::maximum(Color3f(xyz(v)), 0.0f)
                          : dr::clip(Color3f(xyz(v)), 0.0f, 1.0f);
    return m_eval.upsample_rgb(c, si.wavelengths, unbounded);
  }

  // Flattens the closure DAG into per-leaf colors, conductor extinctions, and
  // effective linear weights (combinators * leaf weight), computing layer
  // throughputs when needed.
  std::tuple<LeafSpectra, LeafSpectra, LeafSpectra> flatten_closures(
      const SurfaceInteraction3f& si, const Val* tmps,
      const Vector3f* wi_local) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();

    absl::InlinedVector<UnpolarizedSpectrum, 16> tp;
    if (m_has_layers) {
      tp.resize(n - 1);
      size_t leaf_k = 0;
      for (size_t i = 0; i + 1 < n; ++i) {
        const ClosureNode& c = closures[i];
        if (c.op == ClosureOp::Null) {
          tp[i] = 1.0f;
        } else if (is_leaf(c.op)) {
          size_t k = leaf_k++;
          if (c.op == ClosureOp::Dielectric && scatter_mode(c) == kScatterR) {
            Float albedo =
                dielectric_refl_albedo(c, wi_local[m_leaves[k].frame], tmps);
            Float w_leaf =
                dr::maximum(tmps[c.args[leaf_arg::kWeight]].x(), 0.0f);
            tp[i] = dr::maximum(0.0f, 1.0f - w_leaf * albedo);
          } else {
            tp[i] = 0.0f;
          }
        } else {
          int c0 = c.args[combinator_arg::kFirst],
              c1 = c.args[combinator_arg::kSecond];
          if (c.op == ClosureOp::Layer) {
            tp[i] = tp[c0] * tp[c1];
          } else if (c.op == ClosureOp::Mix) {
            tp[i] =
                dr::lerp(tp[c1], tp[c0],
                         eval_color(c.args[combinator_arg::kWeight], si, tmps));
          } else if (c.op == ClosureOp::Add) {
            tp[i] = dr::maximum(0.0f, tp[c0] + tp[c1] - 1.0f);
          } else {  // Multiply
            tp[i] = tp[c0];
          }
        }
      }
    }

    absl::InlinedVector<UnpolarizedSpectrum, 16> w(n, 0.0f);
    w[n - 1] = 1.0f;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      const ClosureNode& c = closures[i];
      if (!is_combinator(c.op)) continue;
      int c0 = c.args[combinator_arg::kFirst],
          c1 = c.args[combinator_arg::kSecond];
      if (c.op == ClosureOp::Layer) {
        w[c0] += w[i];
        w[c1] = dr::fmadd(w[i], tp[c0], w[c1]);
      } else if (c.op == ClosureOp::Mix) {
        UnpolarizedSpectrum m =
            eval_color(c.args[combinator_arg::kWeight], si, tmps);
        w[c0] = dr::fmadd(w[i], m, w[c0]);
        w[c1] = dr::fmadd(w[i], 1.0f - m, w[c1]);
      } else if (c.op == ClosureOp::Add) {
        w[c0] += w[i];
        w[c1] += w[i];
      } else {  // Multiply
        w[c0] = dr::fmadd(
            w[i], eval_color(c.args[combinator_arg::kWeight], si, tmps), w[c0]);
      }
    }

    size_t num_leaves = m_leaves.size();
    LeafSpectra leaf_color(num_leaves);
    LeafSpectra leaf_extinction;
    LeafSpectra leaf_weight(num_leaves);
    for (size_t k = 0; k < num_leaves; ++k) {
      const ClosureNode& c = closures[m_leaves[k].node];
      bool is_cond = c.op == ClosureOp::Conductor;
      leaf_color[k] =
          eval_color(c.args[leaf_arg::kColor], si, tmps, /*unbounded=*/is_cond);
      if (is_cond) {
        if (leaf_extinction.empty()) leaf_extinction.resize(num_leaves);
        leaf_extinction[k] = eval_color(c.args[leaf_arg::kExtinction], si, tmps,
                                        /*unbounded=*/true);
      }
      leaf_weight[k] = w[m_leaves[k].node] *
                       dr::maximum(tmps[c.args[leaf_arg::kWeight]].x(), 0.0f);
    }
    return {std::move(leaf_color), std::move(leaf_extinction),
            std::move(leaf_weight)};
  }

  State eval_state(const BSDFContext& ctx, const SurfaceInteraction3f& si,
                   Mask active) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    int op_slot = prog().opacity_slot;

    State s;
    s.tmps.resize(prog().tmp_channels.size());
    m_eval.eval(si, active, s.tmps.data());
    s.opacity = op_slot < 0 ? Float(1.0f) : clip_opacity(s.tmps[op_slot]);
    s.surf_active = (op_slot < 0 || dr::grad_enabled(s.opacity))
                        ? active
                        : (active && (s.opacity > 0.0f));
    s.has_surface = has_surface_enabled(ctx) && dr::any_or<true>(s.surf_active);
    if (!s.has_surface) return s;

    s.n_geom =
        safe_normalize(si.to_local(si.n), Vector3f(0.0f, 0.0f, 1.0f), kMinLen2);

    s.frames.resize(m_unique_frames.size());
    s.wi_local.resize(m_unique_frames.size());
    for (size_t f = 0; f < m_unique_frames.size(); ++f) {
      s.frames[f] =
          build_local_frame(m_unique_frames[f].first, m_unique_frames[f].second,
                            si, s.n_geom, s.tmps.data());
      s.wi_local[f] = s.frames[f].to_local(si.wi);
    }

    std::tie(s.leaf_color, s.leaf_extinction, s.leaf_weight) =
        flatten_closures(si, s.tmps.data(), s.wi_local.data());

    size_t num_leaves = m_leaves.size();
    s.leaf_prob.resize(num_leaves);
    if (num_leaves == 1) {
      s.leaf_prob[0] = 1.0f;
    } else {
      Float w_sum(0.0f);
      for (size_t k = 0; k < num_leaves; ++k) {
        if (!leaf_enabled(ctx, closures[m_leaves[k].node])) {
          s.leaf_prob[k] = 0.0f;
        } else {
          Float sw = selection_weight(s.leaf_weight[k] * leaf_albedo(k, s));
          s.leaf_prob[k] = sw;
          w_sum += sw;
        }
      }
      Mask ok = w_sum > 0.0f;
      Float inv_sum =
          dr::select(ok, dr::rcp(dr::select(ok, w_sum, 1.0f)), 0.0f);
      for (Float& p : s.leaf_prob) p = dr::detach(p * inv_sum);
    }
    return s;
  }

  static Float clip_opacity(const Val& op) {
    return dr::clip(op.x(), 0.0f, 1.0f);
  }

  static Float selection_weight(const UnpolarizedSpectrum& c) {
    if constexpr (is_rgb_v<Spectrum>) {
      return luminance(c);
    } else {
      return dr::mean(c);
    }
  }

  bool has_cutout(const BSDFContext& ctx) const {
    return m_null_component != uint32_t(-1) &&
           ctx.is_enabled(BSDFFlags::Null, m_null_component);
  }

  bool has_surface_enabled(const BSDFContext& ctx) const {
    return (ctx.component == uint32_t(-1) || ctx.component == 0) &&
           (m_surface_types & ctx.type_mask) != 0u;
  }

  std::pair<Spectrum, Float> eval_pdf_impl(const BSDFContext& ctx,
                                           const State& s,
                                           const Vector3f& wo) const {
    if (!s.has_surface) return {0.0f, 0.0f};
    auto [f, pdf] = eval_and_pdf_leaves(ctx, s, wo);
    return {depolarizer<Spectrum>(f) & s.surf_active,
            dr::detach(dr::select(
                s.surf_active, has_cutout(ctx) ? pdf * s.opacity : pdf, 0.0f))};
  }

  std::pair<BSDFSample3f, Spectrum> sample_impl(const BSDFContext& ctx,
                                                const SurfaceInteraction3f& si,
                                                const State& s, Float sample1,
                                                const Point2f& sample2,
                                                Mask active) const {
    bool cutout = has_cutout(ctx);
    Float prob_surface = cutout ? s.opacity : Float(1.0f);
    Mask pass_through =
        cutout ? (active && (prob_surface < 1.0f) && (sample1 >= prob_surface))
               : Mask(false);

    Vector3f wo(0.0f, 0.0f, 1.0f);
    UInt32 sampled_type(0);
    Float eta(1.0f);
    Mask leaf_valid(false);
    if (s.has_surface) {
      Mask remaining = s.surf_active && !pass_through;
      if (cutout) {
        Float ps = dr::detach(prob_surface);
        Mask ok = remaining && (ps > 0.0f);
        sample1 = dr::select(ok, sample1 / dr::select(ok, ps, 1.0f), 0.0f);
      }

      Float total_p(0.0f);
      for (const Float& p : s.leaf_prob) total_p += p;
      Float xi = dr::clip(sample1, 0.0f, dr::OneMinusEpsilon<Float>) * total_p;
      Float cdf(0.0f);
      for (size_t k = 0; k < m_leaves.size(); ++k) {
        if (!leaf_enabled(ctx, prog().closures[m_leaves[k].node])) continue;
        Float cdf_prev = cdf;
        cdf += s.leaf_prob[k];
        Mask pick = remaining && (xi < cdf);
        remaining = remaining && !pick;
        if (dr::none_or<false>(pick)) continue;
        Float leaf_u1 =
            m_leaves.size() > 1
                ? (xi - cdf_prev) / dr::select(pick, s.leaf_prob[k], 1.0f)
                : xi;
        auto [leaf_wo, leaf_sampled_type, leaf_eta, valid] =
            sample_leaf(ctx, k, leaf_u1, sample2, s);
        Mask hit = pick && valid;
        wo = dr::select(hit, leaf_wo, wo);
        sampled_type = dr::select(hit, leaf_sampled_type, sampled_type);
        eta = dr::select(hit, leaf_eta, eta);
        leaf_valid = leaf_valid || hit;
      }
    }

    // Detached one-sample MIS over the lobe mixture: weight = f_tot / pdf_tot.
    wo = dr::detach(wo);
    auto [f_tot, pdf_tot] =
        (s.has_surface && dr::any_or<true>(leaf_valid))
            ? eval_and_pdf_leaves(ctx, s, wo)
            : std::pair<UnpolarizedSpectrum, Float>(0.0f, 0.0f);
    Float surf_pdf = dr::select(leaf_valid, pdf_tot * prob_surface, 0.0f);
    Mask surf_valid = leaf_valid && (surf_pdf > 0.0f);

    BSDFSample3f bs = dr::zeros<BSDFSample3f>();
    bs.wo = dr::select(pass_through, -si.wi, wo);
    bs.pdf =
        dr::detach(dr::select(pass_through, 1.0f - prob_surface, surf_pdf));
    bs.eta = eta;
    bs.sampled_type =
        dr::select(pass_through, UInt32(+BSDFFlags::Null), sampled_type);
    bs.sampled_component =
        dr::select(pass_through, UInt32(m_null_component), UInt32(0));

    Float pass_weight(1.0f);
    if constexpr (dr::is_diff_v<Float>) {
      if (cutout && dr::grad_enabled(s.opacity)) {
        pass_weight = dr::replace_grad(
            pass_weight, (1.0f - s.opacity) /
                             dr::select(pass_through,
                                        dr::detach(1.0f - prob_surface), 1.0f));
      }
    }
    UnpolarizedSpectrum surf_weight =
        f_tot / dr::select(surf_valid, dr::detach(surf_pdf), 1.0f);
    return {bs, dr::select(pass_through, Spectrum(pass_weight),
                           depolarizer<Spectrum>(surf_weight) & surf_valid)};
  }

  static Frame3f build_local_frame(int normal_slot, int tangent_slot,
                                   const SurfaceInteraction3f& si,
                                   const Vector3f& n_geom, const Val* tmps) {
    Vector3f nl =
        normal_slot >= 0
            ? si.to_local(safe_normalize(xyz(tmps[normal_slot]),
                                         Vector3f(si.sh_frame.n), kMinLen2))
            : Vector3f(0.0f, 0.0f, 1.0f);
    Float n_dot_g = dr::dot(n_geom, nl);
    nl = dr::select(n_dot_g < 0.0f, nl - 2.0f * n_dot_g * n_geom, nl);
    n_dot_g = dr::abs(n_dot_g);
    // Keep the shading normal on the geometric side of `si.wi` (Schuessler et
    // al. 2017; reduces to `(-nl.x, -nl.y, nl.z)` when `n_geom == (0, 0, 1)`).
    Vector3f n_loc = dr::select(
        dr::mulsign(Float(1.0f), dr::dot(n_geom, si.wi)) * dr::dot(si.wi, nl) <=
            0.0f,
        dr::select(n_dot_g > 0.0f, 2.0f * n_dot_g * n_geom - nl, n_geom), nl);
    Vector3f t_raw =
        si.to_local(tangent_slot >= 0 ? safe_normalize(xyz(tmps[tangent_slot]),
                                                       si.sh_frame.s, kMinLen2)
                                      : geom_tangent_world(si));
    Vector3f s_loc =
        safe_normalize(dr::fnmadd(n_loc, dr::dot(n_loc, t_raw), t_raw),
                       coordinate_system(n_loc).first, kMinLen2);
    return Frame3f(s_loc, dr::cross(n_loc, s_loc), n_loc);
  }

  // Samples a direction from leaf `k`. Returns the direction (in `si.sh_frame`
  // coordinates), the sampled lobe type, the relative IOR along the sampled
  // direction, and whether the sample is valid.
  std::tuple<Vector3f, UInt32, Float, Mask> sample_leaf(const BSDFContext& ctx,
                                                        size_t k,
                                                        const Float& sample1,
                                                        const Point2f& sample2,
                                                        const State& s) const {
    auto [i, f] = m_leaves[k];
    const ClosureNode& c = prog().closures[i];
    const Vector3f& wi_l = s.wi_local[f];
    Float cos_i = Frame3f::cos_theta(wi_l);
    Vector3f wo_l;
    UInt32 sampled_type(leaf_type(c));
    Float out_eta(1.0f);
    Mask valid;
    if (c.op == ClosureOp::Dielectric) {
      // Follows `roughdielectric`: sample visible normals for `wi` flipped
      // into the upper hemisphere, then reflect/refract the unflipped `wi`.
      int sm = scatter_mode(c);
      bool refl_on =
          sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection);
      bool trans_on =
          sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission);
      bool thin = is_thin_walled(c);
      Vector2f alpha = get_roughness(c, s.tmps.data());
      MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y());
      Normal3f m = distr.sample(dr::mulsign(wi_l, cos_i), sample2).first;
      wo_l = reflect(wi_l, m);
      Mask do_reflect(refl_on);
      Mask trans_ok(false);
      if (trans_on) {
        auto [fres, cos_theta_t, eta_it, eta_ti] = dielectric_fresnel(
            c, dr::dot(wi_l, m), dielectric_ior(c, s.tmps.data()));
        trans_ok = fres < 1.0f;
        if (refl_on) do_reflect = !trans_ok || (sample1 < fres);
        Vector3f wo_trans = thin ? Vector3f(wo_l.x(), wo_l.y(), -wo_l.z())
                                 : refract(wi_l, m, cos_theta_t, eta_ti);
        wo_l = dr::select(do_reflect, wo_l, wo_trans);
        if (!thin) out_eta = dr::select(do_reflect, Float(1.0f), eta_it);
      }
      Float cos_o = Frame3f::cos_theta(wo_l);
      valid = dr::select(do_reflect, cos_i * cos_o > 0.0f,
                         trans_ok && (cos_i * cos_o < 0.0f));
      sampled_type =
          dr::select(do_reflect, UInt32(+BSDFFlags::GlossyReflection),
                     UInt32(+BSDFFlags::GlossyTransmission));
    } else if (c.op == ClosureOp::Conductor) {
      Float n_dot_v = dr::abs(cos_i);
      Vector3f wi_pos(wi_l.x(), wi_l.y(), n_dot_v);
      Vector2f alpha = get_roughness(c, s.tmps.data());
      MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y());
      wo_l = reflect(wi_pos, distr.sample(wi_pos, sample2).first);
      valid = (n_dot_v > kMinCos) && (Frame3f::cos_theta(wo_l) > kMinCos);
      wo_l.z() = dr::mulsign(wo_l.z(), cos_i);
    } else {
      bool transmit = c.op == ClosureOp::Translucent;
      wo_l = warp::square_to_cosine_hemisphere(sample2);
      wo_l.z() = dr::mulsign(wo_l.z(), transmit ? -cos_i : cos_i);
      valid = cos_i != 0.0f;
    }
    Vector3f wo = s.frames[f].to_world(wo_l);
    return {wo, sampled_type, out_eta,
            valid && (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> eval_and_pdf_leaf(
      const BSDFContext& ctx, size_t k, const Vector3f& wo,
      const State& s) const {
    auto [i, f] = m_leaves[k];
    const ClosureNode& c = prog().closures[i];
    const Vector3f& wi_l = s.wi_local[f];
    Vector3f wo_l = s.frames[f].to_local(wo);
    Float cos_i = Frame3f::cos_theta(wi_l), cos_o = Frame3f::cos_theta(wo_l);
    Float n_dot_v = dr::abs(cos_i);
    Mask same_side = dr::dot(s.n_geom, wo) * cos_o > 0.0f;
    if (c.op == ClosureOp::Dielectric) {
      // Follows `roughdielectric::eval()` / `pdf()`, using height-correlated
      // Smith G2 for reflection to match MaterialX `pbrlib`.
      int sm = scatter_mode(c);
      bool refl_on =
          sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection);
      bool trans_on =
          sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission);
      bool thin = is_thin_walled(c);
      bool solid_trans = trans_on && !thin;
      Float ior = dielectric_ior(c, s.tmps.data());
      Vector2f alpha = get_roughness(c, s.tmps.data());
      MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y());
      Mask is_refl = cos_i * cos_o > 0.0f;
      Vector3f wo_m =
          (trans_on && thin)
              ? Vector3f(wo_l.x(), wo_l.y(), dr::mulsign(dr::abs(cos_o), cos_i))
              : wo_l;
      Float eta = solid_trans ? dr::select(cos_i > 0.0f, ior, dr::rcp(ior))
                              : Float(1.0f);
      Vector3f m = safe_normalize(
          wi_l + (solid_trans ? wo_m * dr::select(is_refl, Float(1.0f), eta)
                              : wo_m),
          Vector3f(0.0f, 0.0f, 1.0f), kMinLen2);
      m = dr::mulsign(m, Frame3f::cos_theta(m));
      Float cos_im = dr::dot(wi_l, m), cos_om = dr::dot(wo_m, m);
      Mask valid = (cos_im * cos_i > 0.0f) &&
                   (cos_om * Frame3f::cos_theta(wo_m) > 0.0f) && same_side;

      Float f_d = std::get<0>(dielectric_fresnel(c, cos_im, ior));
      Float g1_i = distr.smith_g1(wi_l, m), g1_o = distr.smith_g1(wo_m, m);
      Float d_g1_over_cos =
          distr.eval(m) * g1_i / dr::select(valid, n_dot_v, 1.0f);
      Float refl_pdf = 0.25f * d_g1_over_cos;
      Float val(0.0f), pdf(0.0f);
      if (refl_on) {
        Mask use_refl = valid && is_refl;
        if (dr::any_or<true>(use_refl)) {
          auto [a, b] =
              detail::ggx_dir_albedo_ab(n_dot_v, dr::sqrt(dr::prod(alpha)));
          val = dr::select(use_refl,
                           f_d * detail::ggx_energy_compensation(a + b, f_d) *
                               (refl_pdf * detail::ggx_g2_over_g1(g1_i, g1_o)),
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
            use_trans = use_trans && (denom > kMinLen2);
            Float base = d_g1_over_cos * dr::abs(cos_im * cos_om) /
                         dr::select(use_trans, denom, 1.0f);
            trans_pdf = base * dr::square(eta);
            trans_val =
                (ctx.mode == TransportMode::Radiance ? base : trans_pdf) * g1_o;
          } else {
            trans_val = refl_pdf * g1_o;
            trans_pdf = refl_pdf;
          }
          val = dr::select(
              use_trans, sm == kScatterT ? trans_val : (1.0f - f_d) * trans_val,
              val);
          pdf = dr::select(use_trans,
                           refl_on ? (1.0f - f_d) * trans_pdf : trans_pdf, pdf);
        }
      }
      return {s.leaf_color[k] * val, pdf};
    }
    if (c.op == ClosureOp::Conductor) {
      Float n_dot_l = dr::mulsign(cos_o, cos_i);
      Vector3f wi_pos(wi_l.x(), wi_l.y(), n_dot_v),
          wo_pos(wo_l.x(), wo_l.y(), n_dot_l);
      Mask valid = (n_dot_v > kMinCos) && (n_dot_l > kMinCos) && same_side;
      Vector2f alpha = get_roughness(c, s.tmps.data());
      MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y());
      Vector3f h =
          safe_normalize(wi_pos + wo_pos, Vector3f(0.0f, 0.0f, 1.0f), kMinLen2);
      Float g1_i = distr.smith_g1(wi_pos, h), g1_o = distr.smith_g1(wo_pos, h);
      Float d_g1_over_4cos =
          distr.eval(h) * g1_i / (4.0f * dr::select(valid, n_dot_v, 1.0f));
      UnpolarizedSpectrum fres = fresnel_conductor(
          UnpolarizedSpectrum(dr::maximum(dr::dot(wi_pos, h), kMinCos)),
          conductor_eta(k, s));
      auto [a, b] =
          detail::ggx_dir_albedo_ab(n_dot_v, dr::sqrt(dr::prod(alpha)));
      return {
          dr::select(valid,
                     fres * detail::ggx_energy_compensation(a + b, fres) *
                         (d_g1_over_4cos * detail::ggx_g2_over_g1(g1_i, g1_o)),
                     0.0f),
          dr::select(valid, d_g1_over_4cos, 0.0f)};
    }
    bool transmit = c.op == ClosureOp::Translucent;
    Float n_dot_l = dr::mulsign(cos_o, transmit ? -cos_i : cos_i);
    Mask valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f) && same_side;
    Float pdf_val = n_dot_l * dr::InvPi<Float>;
    const UnpolarizedSpectrum& col = s.leaf_color[k];
    UnpolarizedSpectrum val;
    if (transmit) {
      val = col * pdf_val;
    } else {
      Float r = dr::clip(s.tmps[c.args[leaf_arg::kRoughness]].x(), 0.0f, 1.0f);
      Float l_dot_v = dr::dot(wi_l, wo_l);
      val = c.args[leaf_arg::kMode] == 1
                ? detail::oren_nayar_compensated(n_dot_v, n_dot_l, l_dot_v, r,
                                                 col) *
                      pdf_val
                : col * (pdf_val * detail::oren_nayar_diffuse(n_dot_v, n_dot_l,
                                                              l_dot_v, r));
    }
    return {dr::select(valid, val, 0.0f), dr::select(valid, pdf_val, 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> eval_and_pdf_leaves(
      const BSDFContext& ctx, const State& s, const Vector3f& wo) const {
    UnpolarizedSpectrum f(0.0f);
    Float pdf(0.0f);
    for (size_t k = 0; k < m_leaves.size(); ++k) {
      if (!leaf_enabled(ctx, prog().closures[m_leaves[k].node])) continue;
      auto [f_k, pdf_k] = eval_and_pdf_leaf(ctx, k, wo, s);
      f = dr::fmadd(s.leaf_weight[k], f_k, f);
      pdf = dr::fmadd(s.leaf_prob[k], pdf_k, pdf);
    }
    return {f, pdf};
  }

  MtlxEvaluator<Float, Spectrum> m_eval;
  bool m_has_layers = false;
  uint32_t m_surface_types = 0u;
  uint32_t m_null_component = uint32_t(-1);
  // Leaf closures paired with their deduplicated frame index.
  std::vector<Leaf> m_leaves;
  // Deduplicated `(normal_slot, tangent_slot)` pairs.
  std::vector<std::pair<int, int>> m_unique_frames;
};

}  // namespace

void RegisterMaterialXBsdf() {
  static const bool registered = [] {
    auto register_fn = [](auto... args) {
      PluginManager::instance()->register_plugin(args..., nullptr, nullptr);
    };
    MI_REGISTER_PLUGIN(register_fn, "materialx_bsdf", MtlxBsdf)
    return true;
  }();
  (void)registered;
}

}  // namespace mtlx
NAMESPACE_END(mitsuba)
