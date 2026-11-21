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
    const ClosureNode& c = closure();

    if constexpr (is_spectral_v<Spectrum>) {
      if (c.op != ClosureOp::Null &&
          !prog().is_tmp_uniform(c.args[leaf_arg::kColor])) {
        m_eval.load_srgb_lut();
      }
    }

    uint32_t sv =
        m_eval.is_spatially_varying() ? +BSDFFlags::SpatiallyVarying : 0u;
    if (c.op != ClosureOp::Null) {
      uint32_t surf_flags = BSDFFlags::FrontSide | BSDFFlags::BackSide | sv;
      surf_flags |= c.op == ClosureOp::Translucent
                        ? +BSDFFlags::DiffuseTransmission
                        : +BSDFFlags::DiffuseReflection;
      if (c.args[leaf_arg::kNormal] >= 0) {
        surf_flags |= +BSDFFlags::NormalMapped;
      }
      m_components.push_back(surf_flags);
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
    return sample_impl(ctx, si, eval_state(si, active), sample1, sample2,
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
    return eval_pdf_impl(ctx, eval_state(si, active), wo);
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
    State s = eval_state(si, active);
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
    State s = eval_state(si, active);
    Float transparency = 1.0f - s.opacity;
    UnpolarizedSpectrum diffuse_albedo(0.0f);
    Frame3f sh_frame = si.sh_frame;
    if (s.has_surface) {
      Float w =
          dr::maximum(s.tmps[closure().args[leaf_arg::kWeight]].x(), 0.0f);
      diffuse_albedo =
          (dr::minimum(s.color * w, 1.0f) * s.opacity) & s.surf_active;
      sh_frame = Frame3f(si.sh_frame.to_world(s.frame.s),
                         si.sh_frame.to_world(s.frame.t),
                         si.sh_frame.to_world(s.frame.n));
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
  // Per-query material state shared by all BSDF entry points.
  struct State {
    absl::InlinedVector<Val, 32> tmps;
    UnpolarizedSpectrum color;
    // Local shading frame and incident direction (relative to `si.sh_frame`).
    Frame3f frame;
    Vector3f wi_local;
    Vector3f n_geom;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  static constexpr ScalarFloat kMinLen2 =
      dr::Epsilon<ScalarFloat> * dr::Epsilon<ScalarFloat>;

  const MtlxProgram& prog() const { return m_eval.program(); }
  const ClosureNode& closure() const { return prog().closures[0]; }
  static Vector3f xyz(const Val& v) { return dr::head<3>(v); }

  State eval_state(const SurfaceInteraction3f& si, Mask active) const {
    const ClosureNode& c = closure();
    int op_slot = prog().opacity_slot;

    State s;
    s.tmps.resize(prog().tmp_channels.size());
    m_eval.eval(si, active, s.tmps.data());
    s.opacity = op_slot < 0 ? Float(1.0f) : clip_opacity(s.tmps[op_slot]);
    s.surf_active = (op_slot < 0 || dr::grad_enabled(s.opacity))
                        ? active
                        : (active && (s.opacity > 0.0f));
    s.has_surface = c.op != ClosureOp::Null && dr::any_or<true>(s.surf_active);
    if (!s.has_surface) return s;

    s.n_geom =
        safe_normalize(si.to_local(si.n), Vector3f(0.0f, 0.0f, 1.0f), kMinLen2);

    int col_slot = c.args[leaf_arg::kColor];
    const Val& col_val = s.tmps[col_slot];
    if (is_spectral_v<Spectrum> && prog().is_tmp_uniform(col_slot)) {
      s.color = UnpolarizedSpectrum(dr::clip(col_val.x(), 0.0f, 1.0f));
    } else {
      s.color = m_eval.upsample_rgb(dr::clip(Color3f(xyz(col_val)), 0.0f, 1.0f),
                                    si.wavelengths, /*unbounded=*/false);
    }

    s.frame = build_local_frame(c.args[leaf_arg::kNormal], si, s.n_geom,
                                s.tmps.data());
    s.wi_local = s.frame.to_local(si.wi);
    return s;
  }

  static Float clip_opacity(const Val& op) {
    return dr::clip(op.x(), 0.0f, 1.0f);
  }

  bool has_cutout(const BSDFContext& ctx) const {
    return m_null_component != uint32_t(-1) &&
           ctx.is_enabled(BSDFFlags::Null, m_null_component);
  }

  bool has_surface_enabled(const BSDFContext& ctx) const {
    const ClosureNode& c = closure();
    return c.op != ClosureOp::Null &&
           ctx.is_enabled(c.op == ClosureOp::Translucent
                              ? BSDFFlags::DiffuseTransmission
                              : BSDFFlags::DiffuseReflection,
                          0);
  }

  std::pair<Spectrum, Float> eval_pdf_impl(const BSDFContext& ctx,
                                           const State& s,
                                           const Vector3f& wo) const {
    if (!s.has_surface || !has_surface_enabled(ctx)) return {0.0f, 0.0f};
    auto [f, pdf] = eval_and_pdf_leaf(wo, s);
    return {depolarizer<Spectrum>(f * s.opacity) & s.surf_active,
            dr::select(s.surf_active, has_cutout(ctx) ? pdf * s.opacity : pdf,
                       0.0f)};
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
    bool eval_surface = s.has_surface && has_surface_enabled(ctx);
    if (eval_surface) {
      Mask sel = s.surf_active && !pass_through;
      if (dr::any_or<true>(sel)) {
        auto [leaf_wo, leaf_type, valid] = sample_leaf(sample2, s);
        leaf_valid = sel && valid;
        wo = dr::select(leaf_valid, leaf_wo, wo);
        sampled_type = dr::select(leaf_valid, leaf_type, sampled_type);
      } else {
        eval_surface = false;
      }
    }

    // Detached sampling: `wo` and `surf_pdf` are detached so `f / pdf` yields
    // an unbiased gradient across closure mixtures and stochastic opacity.
    wo = dr::detach(wo);
    auto [f_tot, pdf_tot] =
        (eval_surface && dr::any_or<true>(leaf_valid))
            ? eval_and_pdf_leaf(wo, s)
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
        f_tot *
        (s.opacity / dr::select(surf_valid, dr::detach(surf_pdf), 1.0f));
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

  // Samples a direction from the leaf closure. Returns the direction (in
  // `si.sh_frame` coordinates), the sampled lobe type, and whether the sample
  // is valid.
  std::tuple<Vector3f, UInt32, Mask> sample_leaf(const Point2f& sample2,
                                                 const State& s) const {
    bool transmit = closure().op == ClosureOp::Translucent;
    BSDFFlags type = transmit ? BSDFFlags::DiffuseTransmission
                              : BSDFFlags::DiffuseReflection;
    Float cos_i = Frame3f::cos_theta(s.wi_local);
    Vector3f wo_l = warp::square_to_cosine_hemisphere(sample2);
    wo_l.z() = dr::mulsign(wo_l.z(), transmit ? -cos_i : cos_i);
    Vector3f wo = s.frame.to_world(wo_l);
    return {wo, UInt32(+type),
            (cos_i != 0.0f) &&
                (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> eval_and_pdf_leaf(
      const Vector3f& wo, const State& s) const {
    const ClosureNode& c = closure();
    bool transmit = c.op == ClosureOp::Translucent;
    Vector3f wo_l = s.frame.to_local(wo);
    Float cos_i = Frame3f::cos_theta(s.wi_local),
          cos_o = Frame3f::cos_theta(wo_l);
    Float n_dot_v = dr::abs(cos_i),
          n_dot_l = dr::mulsign(cos_o, transmit ? -cos_i : cos_i);
    Mask valid = (n_dot_v > 0.0f) && (n_dot_l > 0.0f) &&
                 (dr::dot(s.n_geom, wo) * cos_o > 0.0f);
    Float pdf_val = n_dot_l * dr::InvPi<Float>;
    Float scale =
        dr::maximum(s.tmps[c.args[leaf_arg::kWeight]].x(), 0.0f) * pdf_val;
    UnpolarizedSpectrum val;
    if (transmit) {
      val = s.color * scale;
    } else {
      Float r = dr::clip(s.tmps[c.args[leaf_arg::kRoughness]].x(), 0.0f, 1.0f);
      Float l_dot_v = dr::dot(s.wi_local, wo_l);
      val = c.args[leaf_arg::kMode] == 1
                ? detail::oren_nayar_compensated(n_dot_v, n_dot_l, l_dot_v, r,
                                                 s.color) *
                      scale
                : s.color * (scale * detail::oren_nayar_diffuse(
                                         n_dot_v, n_dot_l, l_dot_v, r));
    }
    return {dr::select(valid, val, 0.0f), dr::select(valid, pdf_val, 0.0f)};
  }

  MtlxEvaluator<Float, Spectrum> m_eval;
  uint32_t m_null_component = uint32_t(-1);
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
