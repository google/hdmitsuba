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
#include <cstddef>
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
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/interaction.h>

#include "hdmitsuba/materialx/bsdf_detail.h"
#include "hdmitsuba/materialx/evaluator.h"
#include "hdmitsuba/materialx/math.h"

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
    m_comb_spec.assign(closures.size(), -1);

    bool need_lut = false;
    auto add_spec = [&](int tmp_slot, bool unbounded = false) -> int {
      SpecSlot key{tmp_slot, unbounded};
      auto it = std::find(m_spec_slots.begin(), m_spec_slots.end(), key);
      if (it != m_spec_slots.end()) {
        return static_cast<int>(it - m_spec_slots.begin());
      }
      m_spec_slots.push_back(key);
      if constexpr (is_spectral_v<Spectrum>) {
        need_lut |= !prog().is_tmp_uniform(tmp_slot);
      }
      return static_cast<int>(m_spec_slots.size() - 1);
    };

    uint32_t surf_flags = 0u;
    for (size_t i = 0; i < closures.size(); ++i) {
      const ClosureNode& c = closures[i];
      if (is_leaf(c.op)) {
        std::pair<int, int> key = leaf_frame_slots(c);
        auto it =
            std::find(m_unique_frames.begin(), m_unique_frames.end(), key);
        size_t f_idx = static_cast<size_t>(it - m_unique_frames.begin());
        if (it == m_unique_frames.end()) m_unique_frames.push_back(key);
        Leaf leaf{i, f_idx};
        switch (c.op) {
          case ClosureOp::OrenNayarDiffuse:
          case ClosureOp::BurleyDiffuse:
          case ClosureOp::Sheen:
            leaf.spec[0] = add_spec(c.diffuse.color);
            if (c.op == ClosureOp::Sheen && c.diffuse.mode == kSheenZeltner) {
              surf_flags |= +BSDFFlags::NonSymmetric;
            }
            break;
          case ClosureOp::Translucent:
            leaf.spec[0] = add_spec(c.translucent.color);
            break;
          case ClosureOp::Dielectric:
            leaf.spec[0] = add_spec(c.microfacet.p[0]);
            surf_flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
            break;
          case ClosureOp::Conductor:
            leaf.spec[0] = add_spec(c.microfacet.p[0], /*unbounded=*/true);
            leaf.spec[1] = add_spec(c.microfacet.p[1], /*unbounded=*/true);
            surf_flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
            break;
          case ClosureOp::GeneralizedSchlick:
            leaf.spec[0] = add_spec(c.microfacet.p[0]);
            leaf.spec[1] = add_spec(c.microfacet.p[1]);
            leaf.spec[2] = add_spec(c.microfacet.p[2]);
            surf_flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
            break;
          default:
            break;
        }
        m_leaves.push_back(leaf);
        m_surface_types |= leaf_type(c);
        if (key.first >= 0) surf_flags |= +BSDFFlags::NormalMapped;
      } else if (c.op == ClosureOp::Layer) {
        m_has_layers = true;
        surf_flags |= +BSDFFlags::NonSymmetric;
      } else if (c.op == ClosureOp::Mix || c.op == ClosureOp::Multiply) {
        m_comb_spec[i] = add_spec(c.comb.weight);
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
    State s = eval_state(ctx, si, active);
    if (s.has_surface) compute_leaf_probs(ctx, s);
    return sample_impl(ctx, si, s, sample1, sample2, active);
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
    State s = eval_state(ctx, si, active);
    if (s.has_surface) compute_leaf_probs(ctx, s);
    return eval_pdf_impl(ctx, s, wo);
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
    if (s.has_surface) compute_leaf_probs(ctx, s);
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
        const Leaf& leaf = m_leaves[k];
        const ClosureNode& c = prog().closures[leaf.node];
        UnpolarizedSpectrum contrib = s.leaf_weight[k] * leaf_albedo(k, s);
        if (c.op == ClosureOp::Conductor || c.op == ClosureOp::Dielectric ||
            c.op == ClosureOp::GeneralizedSchlick) {
          Float sw;
          Vector2f alpha = get_ggx_alpha(c, s.tmps.data());
          if (scatter_mode(c) == kScatterR) {
            specular_reflectance += contrib;
            sw = selection_weight(contrib);
          } else {
            bool is_rt = scatter_mode(c) == kScatterRT;
            bool thin_w = is_thin_walled(c);
            Float cos_i = Frame3f::cos_theta(s.wi_local[leaf.frame]);
            Float f_d(0.0f), cos_t(0.0f), eta_ti(1.0f);
            if (!thin_w || (c.op == ClosureOp::Dielectric && is_rt)) {
              std::tie(f_d, cos_t, std::ignore, eta_ti) = detail::exact_fresnel(
                  cos_i, specular_ior(c, s), is_two_sided(c));
            }
            UnpolarizedSpectrum trans;
            if (is_rt) {
              UnpolarizedSpectrum f_c =
                  specular_fresnel(c, leaf, cos_i, f_d, cos_t, s);
              specular_reflectance += contrib * f_c;
              trans = contrib * (1.0f - f_c);
            } else {
              trans = thin_w ? contrib : (contrib & (f_d < 1.0f));
            }
            surf_transmittance += trans;
            Float tw = selection_weight(trans);
            sw = is_rt ? selection_weight(contrib) : tw;
            Mask pick_wt = s.surf_active && (tw > best_wt_w);
            wt = dr::select(
                pick_wt,
                thin_w ? -si.to_world(si.wi)
                       : si.to_world(s.frames[leaf.frame].to_world(
                             refract(s.wi_local[leaf.frame], cos_t, eta_ti))),
                wt);
            best_wt_w = dr::select(pick_wt, tw, best_wt_w);
          }
          spec_roughness =
              dr::select(sw > best_spec_w, dr::max(alpha), spec_roughness);
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
  struct SpecSlot {
    int tmp_slot;
    bool unbounded;
    bool operator==(const SpecSlot& o) const {
      return tmp_slot == o.tmp_slot && unbounded == o.unbounded;
    }
  };

  struct Leaf {
    size_t node, frame;
    int spec[4] = {-1, -1, -1, -1};
  };

  using SpecVec = absl::InlinedVector<UnpolarizedSpectrum, 8>;

  // Per-query material state shared by all BSDF entry points.
  struct State {
    absl::InlinedVector<Val, 32> tmps;
    // Deduplicated local shading frames and incident directions (relative to
    // `si.sh_frame`).
    absl::InlinedVector<Frame3f, 4> frames;
    absl::InlinedVector<Vector3f, 4> wi_local;
    // Deduplicated upsampled spectra (`m_spec_slots`), per-leaf combined linear
    // weights (combinators * leaf weight), optional cached transmissive layer
    // directional albedos (when layered), and detached sampling probabilities.
    SpecVec spectra;
    SpecVec leaf_weight;
    SpecVec layer_albedo;
    absl::InlinedVector<Float, 8> leaf_prob;
    Vector3f n_geom;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  struct SurfaceLobeSample {
    Vector3f wo = Vector3f(0.0f, 0.0f, 1.0f);
    UInt32 sampled_type = UInt32(0);
    Float eta = Float(1.0f);
    Mask valid = Mask(false);
  };

  const MtlxProgram& prog() const { return m_eval.program(); }
  static Vector3f xyz(const Val& v) { return dr::head<3>(v); }
  static int scatter_mode(const ClosureNode& c) {
    return c.microfacet.mode & kScatterModeMask;
  }
  static bool is_thin_walled(const ClosureNode& c) {
    return (c.microfacet.mode & kScatterThinWalled) != 0;
  }
  static bool is_two_sided(const ClosureNode& c) {
    return (c.microfacet.mode & kScatterTwoSided) != 0;
  }
  static bool is_retroreflective(const ClosureNode& c) {
    return (c.microfacet.mode & kScatterRetroreflective) != 0;
  }

  static std::pair<int, int> leaf_frame_slots(const ClosureNode& c) {
    switch (c.op) {
      case ClosureOp::OrenNayarDiffuse:
      case ClosureOp::BurleyDiffuse:
      case ClosureOp::Sheen:
        return {c.diffuse.normal, -1};
      case ClosureOp::Translucent:
        return {c.translucent.normal, -1};
      case ClosureOp::Dielectric:
      case ClosureOp::Conductor:
      case ClosureOp::GeneralizedSchlick:
        return {c.microfacet.normal, c.microfacet.tangent};
      default:
        return {-1, -1};
    }
  }

  static int leaf_weight_slot(const ClosureNode& c) {
    switch (c.op) {
      case ClosureOp::OrenNayarDiffuse:
      case ClosureOp::BurleyDiffuse:
      case ClosureOp::Sheen:
        return c.diffuse.weight;
      case ClosureOp::Translucent:
        return c.translucent.weight;
      case ClosureOp::Dielectric:
      case ClosureOp::Conductor:
      case ClosureOp::GeneralizedSchlick:
        return c.microfacet.weight;
      default:
        return -1;
    }
  }

  static uint32_t leaf_type(const ClosureNode& c) {
    switch (c.op) {
      case ClosureOp::Translucent:
        return +BSDFFlags::DiffuseTransmission;
      case ClosureOp::Conductor:
      case ClosureOp::Dielectric:
      case ClosureOp::GeneralizedSchlick: {
        int sm = scatter_mode(c);
        return (sm != kScatterT ? +BSDFFlags::GlossyReflection : 0u) |
               (sm != kScatterR ? +BSDFFlags::GlossyTransmission : 0u);
      }
      default:
        return +BSDFFlags::DiffuseReflection;
    }
  }

  static detail::MicrofacetOpts microfacet_opts(const ClosureNode& c,
                                                const BSDFContext& ctx) {
    int sm = scatter_mode(c);
    bool tw = is_thin_walled(c);
    bool need_ior =
        (c.op == ClosureOp::Dielectric && (!tw || sm != kScatterT)) ||
        (c.op == ClosureOp::GeneralizedSchlick && !is_two_sided(c));
    return {sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection),
            sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission),
            tw,
            is_two_sided(c),
            is_retroreflective(c),
            sm == kScatterT,
            need_ior};
  }

  static bool leaf_enabled(const BSDFContext& ctx, const ClosureNode& c) {
    return (leaf_type(c) & ctx.type_mask) != 0u;
  }

  static Vector2f get_ggx_alpha(const ClosureNode& c, const Val* tmps) {
    return dr::clip(dr::head<2>(tmps[c.microfacet.roughness]), 1e-3f, 1.0f);
  }

  Float specular_ior(const ClosureNode& c, const State& s) const {
    if (c.op == ClosureOp::Dielectric) {
      return detail::safe_dielectric_ior(
          s.tmps[c.microfacet.p[1]].x(),
          !is_thin_walled(c) && scatter_mode(c) != kScatterR);
    }
    int slot = c.microfacet.p[0];
    return detail::f0_to_ior(
        prog().is_tmp_uniform(slot)
            ? s.tmps[slot].x()
            : dr::mean(dr::clip(xyz(s.tmps[slot]), 0.0f, 1.0f)));
  }

  static UnpolarizedSpectrum specular_fresnel(
      const ClosureNode& c, const Leaf& leaf, const Float& cos_im,
      const Float& f_d, const Float& cos_theta_t, const State& s) {
    if (c.op == ClosureOp::Dielectric) return UnpolarizedSpectrum(f_d);
    if (c.op == ClosureOp::Conductor) {
      return fresnel_conductor(
          UnpolarizedSpectrum(dr::maximum(dr::abs(cos_im), dr::Epsilon<Float>)),
          detail::safe_conductor_eta<Float>(s.spectra[leaf.spec[0]],
                                            s.spectra[leaf.spec[1]]));
    }
    bool two_sided = is_two_sided(c);
    Float cos_tf =
        two_sided ? dr::abs(cos_im)
                  : dr::select(cos_im < 0.0f, dr::abs(cos_theta_t), cos_im);
    UnpolarizedSpectrum f_schlick = detail::fresnel_hoffman_schlick(
        cos_tf, s.spectra[leaf.spec[0]], s.spectra[leaf.spec[1]],
        s.spectra[leaf.spec[2]], s.tmps[c.microfacet.p[3]].x());
    if (two_sided) return f_schlick;
    Mask is_tir = (cos_im < 0.0f) && (cos_theta_t == 0.0f);
    return dr::select(is_tir, UnpolarizedSpectrum(1.0f), f_schlick);
  }

  static Float specular_refl_prob(const ClosureNode& c, const Float& f_d,
                                  const UnpolarizedSpectrum& f_c) {
    return c.op == ClosureOp::Dielectric ? f_d : selection_weight(f_c);
  }

  UnpolarizedSpectrum specular_refl_albedo(const ClosureNode& c,
                                           const Leaf& leaf,
                                           const State& s) const {
    Float cos_i = Frame3f::cos_theta(s.wi_local[leaf.frame]);
    Float ndv = dr::abs(cos_i);
    Float alpha_eff = dr::sqrt(dr::prod(get_ggx_alpha(c, s.tmps.data())));
    UnpolarizedSpectrum f_nv, f0, f90(1.0f);
    Mask index_matched(false);
    if (c.op == ClosureOp::Conductor) {
      f_nv = specular_fresnel(c, leaf, ndv, Float(0.0f), Float(0.0f), s);
      f0 = specular_fresnel(c, leaf, Float(1.0f), Float(0.0f), Float(0.0f), s);
    } else if (c.op == ClosureOp::GeneralizedSchlick) {
      f_nv = specular_fresnel(c, leaf, ndv, Float(0.0f), Float(0.0f), s);
      f0 = s.spectra[leaf.spec[0]];
      f90 = s.spectra[leaf.spec[2]];
    } else {
      Float ior = specular_ior(c, s);
      auto [f_d, cos_t, eta_it, eta_ti] =
          detail::exact_fresnel(cos_i, ior, is_two_sided(c));
      f_nv = specular_fresnel(c, leaf, cos_i, f_d, cos_t, s);
      f0 = dr::select(
          eta_it < 1.0f, f_nv,
          UnpolarizedSpectrum(dr::square((ior - 1.0f) / (ior + 1.0f))));
      index_matched = ior == 1.0f;
    }
    UnpolarizedSpectrum res =
        detail::ggx_refl_albedo(ndv, alpha_eff, f0, f_nv, f90);
    return c.op == ClosureOp::Dielectric ? dr::select(index_matched, 0.0f, res)
                                         : res;
  }

  static Float sheen_dir_albedo(const ClosureNode& c, const Vector3f& wi_l,
                                const Val* tmps) {
    return detail::sheen_dir_albedo(dr::abs(Frame3f::cos_theta(wi_l)),
                                    tmps[c.diffuse.roughness].x(),
                                    c.diffuse.mode == kSheenZeltner);
  }

  UnpolarizedSpectrum leaf_albedo(size_t k, const State& s) const {
    const Leaf& leaf = m_leaves[k];
    const ClosureNode& c = prog().closures[leaf.node];
    switch (c.op) {
      case ClosureOp::Dielectric: {
        const UnpolarizedSpectrum& tint = s.spectra[leaf.spec[0]];
        if (scatter_mode(c) != kScatterR) return tint;
        return tint * (!s.layer_albedo.empty()
                           ? s.layer_albedo[k]
                           : specular_refl_albedo(c, leaf, s));
      }
      case ClosureOp::GeneralizedSchlick:
        if (scatter_mode(c) != kScatterR) return UnpolarizedSpectrum(1.0f);
        return !s.layer_albedo.empty() ? s.layer_albedo[k]
                                       : specular_refl_albedo(c, leaf, s);
      case ClosureOp::Sheen:
        return s.spectra[leaf.spec[0]] *
               (!s.layer_albedo.empty()
                    ? s.layer_albedo[k]
                    : UnpolarizedSpectrum(sheen_dir_albedo(
                          c, s.wi_local[leaf.frame], s.tmps.data())));
      case ClosureOp::Conductor:
        return specular_refl_albedo(c, leaf, s);
      default:
        return s.spectra[leaf.spec[0]];
    }
  }

  UnpolarizedSpectrum eval_color(int slot, const SurfaceInteraction3f& si,
                                 const Val* tmps, bool unbounded) const {
    const Val& v = tmps[slot];
    if (prog().is_tmp_uniform(slot)) {
      return UnpolarizedSpectrum(unbounded ? dr::maximum(v.x(), 0.0f)
                                           : dr::clip(v.x(), 0.0f, 1.0f));
    }
    Color3f c = unbounded ? dr::maximum(Color3f(xyz(v)), 0.0f)
                          : dr::clip(Color3f(xyz(v)), 0.0f, 1.0f);
    return m_eval.upsample_rgb(c, si.wavelengths, unbounded);
  }

  // Flattens the closure DAG into per-leaf effective linear weights
  // (combinators * leaf weight), computing layer throughputs and caching
  // transmissive layer albedos when layered.
  void flatten_closures(State& s) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();
    const Val* tmps = s.tmps.data();

    absl::InlinedVector<UnpolarizedSpectrum, 16> tp;
    if (m_has_layers) {
      tp.resize(n - 1);
      s.layer_albedo.resize(m_leaves.size(), 0.0f);
      size_t leaf_k = 0;
      for (size_t i = 0; i + 1 < n; ++i) {
        const ClosureNode& c = closures[i];
        if (c.op == ClosureOp::Null) {
          tp[i] = 1.0f;
        } else if (is_leaf(c.op)) {
          size_t k = leaf_k++;
          if (((c.op == ClosureOp::Dielectric ||
                c.op == ClosureOp::GeneralizedSchlick) &&
               scatter_mode(c) == kScatterR) ||
              c.op == ClosureOp::Sheen) {
            UnpolarizedSpectrum albedo =
                c.op == ClosureOp::Sheen
                    ? UnpolarizedSpectrum(sheen_dir_albedo(
                          c, s.wi_local[m_leaves[k].frame], tmps))
                    : specular_refl_albedo(c, m_leaves[k], s);
            s.layer_albedo[k] = albedo;
            Float w_leaf = dr::maximum(tmps[leaf_weight_slot(c)].x(), 0.0f);
            tp[i] = dr::maximum(0.0f, 1.0f - w_leaf * albedo);
          } else {
            tp[i] = 0.0f;
          }
        } else {
          int c0 = c.comb.first, c1 = c.comb.second;
          if (c.op == ClosureOp::Layer) {
            tp[i] = tp[c0] * tp[c1];
          } else if (c.op == ClosureOp::Mix) {
            tp[i] = dr::lerp(tp[c1], tp[c0], s.spectra[m_comb_spec[i]]);
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
      int c0 = c.comb.first, c1 = c.comb.second;
      if (c.op == ClosureOp::Layer) {
        w[c0] += w[i];
        w[c1] = dr::fmadd(w[i], tp[c0], w[c1]);
      } else if (c.op == ClosureOp::Mix) {
        const UnpolarizedSpectrum& m = s.spectra[m_comb_spec[i]];
        w[c0] = dr::fmadd(w[i], m, w[c0]);
        w[c1] = dr::fmadd(w[i], 1.0f - m, w[c1]);
      } else if (c.op == ClosureOp::Add) {
        w[c0] += w[i];
        w[c1] += w[i];
      } else {  // Multiply
        w[c0] = dr::fmadd(w[i], s.spectra[m_comb_spec[i]], w[c0]);
      }
    }

    s.leaf_weight.resize(m_leaves.size());
    for (size_t k = 0; k < m_leaves.size(); ++k) {
      const ClosureNode& c = closures[m_leaves[k].node];
      s.leaf_weight[k] = w[m_leaves[k].node] *
                         dr::maximum(tmps[leaf_weight_slot(c)].x(), 0.0f);
    }
  }

  State eval_state(const BSDFContext& ctx, const SurfaceInteraction3f& si,
                   Mask active) const {
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
        detail::safe_normalize(si.to_local(si.n), Vector3f(0.0f, 0.0f, 1.0f),
                               detail::kMinLen2<ScalarFloat>);

    s.frames.resize(m_unique_frames.size());
    s.wi_local.resize(m_unique_frames.size());
    for (size_t f = 0; f < m_unique_frames.size(); ++f) {
      auto [normal_slot, tangent_slot] = m_unique_frames[f];
      Vector3f nl = normal_slot >= 0
                        ? si.to_local(detail::safe_normalize(
                              xyz(s.tmps[normal_slot]), Vector3f(si.sh_frame.n),
                              detail::kMinLen2<ScalarFloat>))
                        : Vector3f(0.0f, 0.0f, 1.0f);
      Vector3f t_world =
          tangent_slot >= 0
              ? detail::safe_normalize(xyz(s.tmps[tangent_slot]), si.sh_frame.s,
                                       detail::kMinLen2<ScalarFloat>)
              : detail::geom_tangent_world(si);
      s.frames[f] = detail::build_shading_frame(si, s.n_geom, nl, t_world);
      s.wi_local[f] = s.frames[f].to_local(si.wi);
    }

    s.spectra.resize(m_spec_slots.size());
    for (size_t j = 0; j < m_spec_slots.size(); ++j) {
      s.spectra[j] = eval_color(m_spec_slots[j].tmp_slot, si, s.tmps.data(),
                                m_spec_slots[j].unbounded);
    }
    flatten_closures(s);
    return s;
  }

  void compute_leaf_probs(const BSDFContext& ctx, State& s) const {
    size_t num_leaves = m_leaves.size();
    s.leaf_prob.resize(num_leaves);
    if (num_leaves == 1) {
      s.leaf_prob[0] = 1.0f;
      return;
    }
    Float w_sum(0.0f);
    for (size_t k = 0; k < num_leaves; ++k) {
      const ClosureNode& c = prog().closures[m_leaves[k].node];
      if (!leaf_enabled(ctx, c)) {
        s.leaf_prob[k] = 0.0f;
      } else {
        UnpolarizedSpectrum alb = leaf_albedo(k, s);
        if (c.op == ClosureOp::Sheen && c.diffuse.mode != kSheenZeltner) {
          alb = dr::maximum(alb, s.spectra[m_leaves[k].spec[0]] * 1e-3f);
        }
        Float sw = selection_weight(s.leaf_weight[k] * alb);
        s.leaf_prob[k] = sw;
        w_sum += sw;
      }
    }
    Mask ok = w_sum > 0.0f;
    Float inv_sum = dr::select(ok, dr::rcp(dr::select(ok, w_sum, 1.0f)), 0.0f);
    for (Float& p : s.leaf_prob) p = dr::detach(p * inv_sum);
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

  SurfaceLobeSample sample_surface_lobe(const BSDFContext& ctx, const State& s,
                                        Float sample1, const Point2f& sample2,
                                        Mask active) const {
    SurfaceLobeSample out;
    if (!s.has_surface) return out;
    Mask remaining = active;
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
      out.wo = dr::select(hit, leaf_wo, out.wo);
      out.sampled_type = dr::select(hit, leaf_sampled_type, out.sampled_type);
      out.eta = dr::select(hit, leaf_eta, out.eta);
      out.valid = out.valid || hit;
    }
    return out;
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
    Mask surf_mask = s.surf_active && !pass_through;
    if (cutout && s.has_surface) {
      Float ps = dr::detach(prob_surface);
      Mask ok = surf_mask && (ps > 0.0f);
      sample1 = dr::select(ok, sample1 / dr::select(ok, ps, 1.0f), 0.0f);
    }

    SurfaceLobeSample lobe =
        sample_surface_lobe(ctx, s, sample1, sample2, surf_mask);

    // Detached one-sample MIS over the lobe mixture: weight = f_tot / pdf_tot.
    Vector3f wo = dr::detach(lobe.wo);
    auto [f_tot, pdf_tot] =
        (s.has_surface && dr::any_or<true>(lobe.valid))
            ? eval_and_pdf_leaves(ctx, s, wo)
            : std::pair<UnpolarizedSpectrum, Float>(0.0f, 0.0f);
    Float surf_pdf = dr::select(lobe.valid, pdf_tot * prob_surface, 0.0f);
    Mask surf_valid = lobe.valid && (surf_pdf > 0.0f);

    BSDFSample3f bs = dr::zeros<BSDFSample3f>();
    bs.wo = dr::select(pass_through, -si.wi, wo);
    bs.pdf =
        dr::detach(dr::select(pass_through, 1.0f - prob_surface, surf_pdf));
    bs.eta = lobe.eta;
    bs.sampled_type =
        dr::select(pass_through, UInt32(+BSDFFlags::Null), lobe.sampled_type);
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

  // Samples a direction from leaf `k`. Returns the direction (in `si.sh_frame`
  // coordinates), the sampled lobe type, the relative IOR along the sampled
  // direction, and whether the sample is valid.
  std::tuple<Vector3f, UInt32, Float, Mask> sample_leaf(const BSDFContext& ctx,
                                                        size_t k,
                                                        const Float& sample1,
                                                        const Point2f& sample2,
                                                        const State& s) const {
    const Leaf& leaf = m_leaves[k];
    const ClosureNode& c = prog().closures[leaf.node];
    const Vector3f& wi_l = s.wi_local[leaf.frame];
    Vector3f wo_l(0.0f, 0.0f, 1.0f);
    UInt32 sampled_type(leaf_type(c));
    Float out_eta(1.0f);
    Mask valid(false);
    switch (c.op) {
      case ClosureOp::Dielectric:
      case ClosureOp::Conductor:
      case ClosureOp::GeneralizedSchlick: {
        detail::MicrofacetOpts opts = microfacet_opts(c, ctx);
        Float ior = (opts.trans_on && opts.need_ior_fresnel)
                        ? specular_ior(c, s)
                        : Float(1.0f);
        auto ds = detail::ggx_microfacet_sample(
            wi_l, get_ggx_alpha(c, s.tmps.data()), ior, opts, sample1, sample2,
            [&](const Float& cos_im, const Float& f_d, const Float& cos_t) {
              return specular_refl_prob(
                  c, f_d, specular_fresnel(c, leaf, cos_im, f_d, cos_t, s));
            });
        wo_l = ds.wo_l;
        sampled_type =
            dr::select(ds.is_refl, UInt32(+BSDFFlags::GlossyReflection),
                       UInt32(+BSDFFlags::GlossyTransmission));
        out_eta = ds.eta;
        valid = ds.valid;
        break;
      }
      case ClosureOp::Sheen:
        std::tie(wo_l, valid) =
            detail::sheen_sample(wi_l, s.tmps[c.diffuse.roughness].x(),
                                 c.diffuse.mode == kSheenZeltner, sample2);
        break;
      case ClosureOp::Translucent:
        std::tie(wo_l, valid) = detail::translucent_sample(wi_l, sample2);
        break;
      default:
        std::tie(wo_l, valid) = detail::oren_nayar_sample(wi_l, sample2);
        break;
    }
    Vector3f wo = s.frames[leaf.frame].to_world(wo_l);
    return {wo, sampled_type, out_eta,
            valid && (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> eval_and_pdf_leaf(
      const BSDFContext& ctx, size_t k, const Vector3f& wo,
      const State& s) const {
    const Leaf& leaf = m_leaves[k];
    const ClosureNode& c = prog().closures[leaf.node];
    const Vector3f& wi_l = s.wi_local[leaf.frame];
    Vector3f wo_l = s.frames[leaf.frame].to_local(wo);
    Mask same_side = dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f;
    if (dr::none_or<false>(same_side)) return {0.0f, 0.0f};
    UnpolarizedSpectrum f(0.0f);
    Float pdf(0.0f);
    switch (c.op) {
      case ClosureOp::Dielectric:
      case ClosureOp::Conductor:
      case ClosureOp::GeneralizedSchlick: {
        detail::MicrofacetOpts opts = microfacet_opts(c, ctx);
        Float ior = opts.need_ior_fresnel ? specular_ior(c, s) : Float(1.0f);
        std::tie(f, pdf) =
            detail::ggx_microfacet_eval_pdf<Float, UnpolarizedSpectrum>(
                wi_l, wo_l, get_ggx_alpha(c, s.tmps.data()), ior, opts,
                ctx.mode,
                [&](const Float& cos_im, const Float& f_d, const Float& cos_t) {
                  return specular_fresnel(c, leaf, cos_im, f_d, cos_t, s);
                },
                [&](const Float& f_d, const UnpolarizedSpectrum& f_c) {
                  return specular_refl_prob(c, f_d, f_c);
                });
        if (c.op == ClosureOp::Dielectric) f *= s.spectra[leaf.spec[0]];
        break;
      }
      case ClosureOp::Translucent:
        std::tie(f, pdf) =
            detail::translucent_eval_pdf(wi_l, wo_l, s.spectra[leaf.spec[0]]);
        break;
      case ClosureOp::Sheen:
        std::tie(f, pdf) = detail::sheen_eval_pdf(
            wi_l, wo_l, s.spectra[leaf.spec[0]],
            s.tmps[c.diffuse.roughness].x(), c.diffuse.mode == kSheenZeltner);
        break;
      case ClosureOp::BurleyDiffuse:
        std::tie(f, pdf) = detail::burley_diffuse_eval_pdf(
            wi_l, wo_l, s.spectra[leaf.spec[0]],
            dr::clip(s.tmps[c.diffuse.roughness].x(), 0.0f, 1.0f));
        break;
      default:
        std::tie(f, pdf) = detail::oren_nayar_eval_pdf(
            wi_l, wo_l, s.spectra[leaf.spec[0]],
            dr::clip(s.tmps[c.diffuse.roughness].x(), 0.0f, 1.0f),
            c.diffuse.mode != 0);
        break;
    }
    return {dr::select(same_side, f, 0.0f), dr::select(same_side, pdf, 0.0f)};
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
  // Deduplicated spectral slots evaluated once per query.
  std::vector<SpecSlot> m_spec_slots;
  // Per-closure combinator spectral slot index (-1 for non-Mix/Multiply).
  std::vector<int> m_comb_spec;
  // Leaf closures paired with their deduplicated frame and spectral slots.
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
