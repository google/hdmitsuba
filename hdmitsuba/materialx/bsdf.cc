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
#include <mitsuba/render/interaction.h>

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
  MI_IMPORT_TYPES()
  using Val = typename MtlxEvaluator<Float, Spectrum>::Val;

  explicit MtlxBsdf(const Properties& props) : Base(props) {
    m_eval.compile(props);
    const std::vector<ClosureNode>& closures = prog().closures;

    bool need_lut = false;
    uint32_t surf_flags = 0u;
    for (size_t i = 0; i < closures.size(); ++i) {
      const ClosureNode& c = closures[i];
      if (is_leaf(c.op)) {
        int n_slot = c.args[leaf_arg::kNormal];
        auto it =
            std::find(m_unique_normals.begin(), m_unique_normals.end(), n_slot);
        size_t f_idx = static_cast<size_t>(it - m_unique_normals.begin());
        if (it == m_unique_normals.end()) m_unique_normals.push_back(n_slot);
        m_leaves.push_back({i, f_idx});
        m_surface_types |= +leaf_type(c.op);
        if (n_slot >= 0) surf_flags |= +BSDFFlags::NormalMapped;
      } else if (c.op == ClosureOp::Layer) {
        m_has_layers = true;
        surf_flags |= +BSDFFlags::NonSymmetric;
      }
      if constexpr (is_spectral_v<Spectrum>) {
        int c_slot = is_leaf(c.op) ? c.args[leaf_arg::kColor]
                                   : c.args[combinator_arg::kWeight];
        need_lut |= c_slot >= 0 && !prog().is_tmp_uniform(c_slot);
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
    UnpolarizedSpectrum diffuse_albedo(0.0f);
    Frame3f sh_frame = si.sh_frame;
    if (s.has_surface) {
      for (size_t k = 0; k < m_leaves.size(); ++k) {
        diffuse_albedo =
            dr::fmadd(s.leaf_weight[k], s.leaf_color[k], diffuse_albedo);
      }
      diffuse_albedo = dr::minimum(diffuse_albedo, 1.0f) & s.surf_active;
      Frame3f f = s.frames[0];
      if (m_unique_normals.size() > 1) {
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
    Mask has_trans = active && (transparency > 0.0f);
    return {diffuse_albedo,
            UnpolarizedSpectrum(0.0f),
            UnpolarizedSpectrum(transparency) & active,
            sh_frame,
            dr::select(has_trans, 0.0f, 1.0f),
            dr::select(has_trans, -si.to_world(si.wi), Vector3f(0.0f))};
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
    // Per-leaf color, combined linear weight (combinators * leaf weight), and
    // detached sampling probability (summing to 1 over active leaves).
    LeafSpectra leaf_color;
    LeafSpectra leaf_weight;
    absl::InlinedVector<Float, 8> leaf_prob;
    Vector3f n_geom;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  static constexpr ScalarFloat kMinLen2 =
      dr::Epsilon<ScalarFloat> * dr::Epsilon<ScalarFloat>;

  const MtlxProgram& prog() const { return m_eval.program(); }
  static Vector3f xyz(const Val& v) { return dr::head<3>(v); }
  static BSDFFlags leaf_type(ClosureOp op) {
    return op == ClosureOp::Translucent ? BSDFFlags::DiffuseTransmission
                                        : BSDFFlags::DiffuseReflection;
  }

  UnpolarizedSpectrum eval_color(int slot, const SurfaceInteraction3f& si,
                                 const Val* tmps) const {
    const Val& v = tmps[slot];
    return prog().is_tmp_uniform(slot)
               ? UnpolarizedSpectrum(dr::clip(v.x(), 0.0f, 1.0f))
               : m_eval.upsample_rgb(dr::clip(Color3f(xyz(v)), 0.0f, 1.0f),
                                     si.wavelengths, /*unbounded=*/false);
  }

  // Flattens the closure DAG into per-leaf colors and effective linear weights
  // (combinators * leaf weight), computing layer throughputs when needed.
  std::pair<LeafSpectra, LeafSpectra> flatten_closures(
      const SurfaceInteraction3f& si, const Val* tmps) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();

    absl::InlinedVector<UnpolarizedSpectrum, 16> tp;
    if (m_has_layers) {
      tp.resize(n - 1);
      for (size_t i = 0; i + 1 < n; ++i) {
        const ClosureNode& c = closures[i];
        if (c.op == ClosureOp::Null) {
          tp[i] = 1.0f;
        } else if (is_leaf(c.op)) {
          tp[i] = 0.0f;
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
    LeafSpectra leaf_weight(num_leaves);
    for (size_t k = 0; k < num_leaves; ++k) {
      const ClosureNode& c = closures[m_leaves[k].node];
      leaf_color[k] = eval_color(c.args[leaf_arg::kColor], si, tmps);
      leaf_weight[k] = w[m_leaves[k].node] *
                       dr::maximum(tmps[c.args[leaf_arg::kWeight]].x(), 0.0f);
    }
    return {std::move(leaf_color), std::move(leaf_weight)};
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

    s.frames.resize(m_unique_normals.size());
    s.wi_local.resize(m_unique_normals.size());
    for (size_t f = 0; f < m_unique_normals.size(); ++f) {
      s.frames[f] =
          build_local_frame(m_unique_normals[f], si, s.n_geom, s.tmps.data());
      s.wi_local[f] = s.frames[f].to_local(si.wi);
    }

    std::tie(s.leaf_color, s.leaf_weight) = flatten_closures(si, s.tmps.data());

    size_t num_leaves = m_leaves.size();
    s.leaf_prob.resize(num_leaves);
    if (num_leaves == 1) {
      s.leaf_prob[0] = 1.0f;
    } else {
      Float w_sum(0.0f);
      for (size_t k = 0; k < num_leaves; ++k) {
        if (!ctx.is_enabled(leaf_type(closures[m_leaves[k].node].op), 0)) {
          s.leaf_prob[k] = 0.0f;
        } else {
          Float sw = selection_weight(s.leaf_weight[k] * s.leaf_color[k]);
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
    return {
        depolarizer<Spectrum>(f) & s.surf_active,
        dr::select(s.surf_active,
                   has_cutout(ctx) ? pdf * dr::detach(s.opacity) : pdf, 0.0f)};
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
    Mask leaf_valid(false);
    if (s.has_surface) {
      Mask remaining = s.surf_active && !pass_through;
      if (cutout && m_leaves.size() > 1) {
        Float ps = dr::detach(prob_surface);
        Mask ok = remaining && (ps > 0.0f);
        sample1 = dr::select(ok, sample1 / dr::select(ok, ps, 1.0f), 0.0f);
      }

      Float total_p(0.0f);
      for (const Float& p : s.leaf_prob) total_p += p;
      Float xi = dr::clip(sample1, 0.0f, dr::OneMinusEpsilon<Float>) * total_p;
      Float cdf(0.0f);
      for (size_t k = 0; k < m_leaves.size(); ++k) {
        if (!ctx.is_enabled(leaf_type(prog().closures[m_leaves[k].node].op),
                            0)) {
          continue;
        }
        cdf += s.leaf_prob[k];
        Mask pick = remaining && (xi < cdf);
        remaining = remaining && !pick;
        if (dr::none_or<false>(pick)) continue;
        auto [leaf_wo, leaf_sampled_type, valid] = sample_leaf(k, sample2, s);
        Mask hit = pick && valid;
        wo = dr::select(hit, leaf_wo, wo);
        sampled_type = dr::select(hit, leaf_sampled_type, sampled_type);
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
    bs.eta = 1.0f;
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

  static Frame3f build_local_frame(int normal_slot,
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
    Vector3f t_raw = si.to_local(geom_tangent_world(si));
    Vector3f s_loc =
        safe_normalize(dr::fnmadd(n_loc, dr::dot(n_loc, t_raw), t_raw),
                       coordinate_system(n_loc).first, kMinLen2);
    return Frame3f(s_loc, dr::cross(n_loc, s_loc), n_loc);
  }

  // Samples a direction from leaf `k`. Returns the direction (in `si.sh_frame`
  // coordinates), the sampled lobe type, and whether the sample is valid.
  std::tuple<Vector3f, UInt32, Mask> sample_leaf(size_t k,
                                                 const Point2f& sample2,
                                                 const State& s) const {
    auto [i, f] = m_leaves[k];
    ClosureOp op = prog().closures[i].op;
    bool transmit = op == ClosureOp::Translucent;
    Float cos_i = Frame3f::cos_theta(s.wi_local[f]);
    Vector3f wo_l = warp::square_to_cosine_hemisphere(sample2);
    wo_l.z() = dr::mulsign(wo_l.z(), transmit ? -cos_i : cos_i);
    Vector3f wo = s.frames[f].to_world(wo_l);
    return {wo, UInt32(+leaf_type(op)),
            (cos_i != 0.0f) &&
                (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> eval_and_pdf_leaf(
      size_t k, const Vector3f& wo, const State& s) const {
    auto [i, f] = m_leaves[k];
    const ClosureNode& c = prog().closures[i];
    bool transmit = c.op == ClosureOp::Translucent;
    Vector3f wo_l = s.frames[f].to_local(wo);
    Float cos_i = Frame3f::cos_theta(s.wi_local[f]),
          cos_o = Frame3f::cos_theta(wo_l);
    Float n_dot_v = dr::abs(cos_i),
          n_dot_l = dr::mulsign(cos_o, transmit ? -cos_i : cos_i);
    Mask valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f) &&
                 (dr::dot(s.n_geom, wo) * cos_o > 0.0f);
    Float pdf_val = n_dot_l * dr::InvPi<Float>;
    const UnpolarizedSpectrum& col = s.leaf_color[k];
    UnpolarizedSpectrum val;
    if (transmit) {
      val = col * pdf_val;
    } else {
      Float r = dr::clip(s.tmps[c.args[leaf_arg::kRoughness]].x(), 0.0f, 1.0f);
      Float l_dot_v = dr::dot(s.wi_local[f], wo_l);
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
      if (!ctx.is_enabled(leaf_type(prog().closures[m_leaves[k].node].op), 0)) {
        continue;
      }
      auto [f_k, pdf_k] = eval_and_pdf_leaf(k, wo, s);
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
  std::vector<int> m_unique_normals;
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
