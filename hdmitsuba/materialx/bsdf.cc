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
#include <type_traits>
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

#include "hdmitsuba/materialx/evaluator.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {
namespace {

template <typename Mask>
bool may_be_true(const Mask& m) {
  if constexpr (dr::is_jit_v<Mask>) {
    return m.index() && m.size() > 0 &&
           (m.state() != VarState::Literal || m.entry(0));
  } else {
    return dr::any(m);
  }
}

// The `mx_*` functions below port the corresponding functions in MaterialX
// `libraries/pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl` and
// `mx_microfacet_specular.glsl` (Apache-2.0, Contributors to the MaterialX
// Project).

template <typename Float>
Float mx_oren_nayar_diffuse(const Float& NdotV, const Float& NdotL,
                            const Float& LdotV, const Float& roughness) {
  Float s = LdotV - NdotL * NdotV;
  Float stinv = dr::select(
      s > 0.0f, s / dr::maximum(dr::maximum(NdotL, NdotV), 1e-6f), 0.0f);
  Float sigma2 = dr::square(roughness);
  return (1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f))) +
         (0.45f * sigma2 / (sigma2 + 0.09f)) * stinv;
}

template <typename Float>
Float mx_oren_nayar_dir_albedo(const Float& NdotV, const Float& roughness) {
  Float rx = 1.0f - 0.4297f * roughness - 0.7632f * NdotV * roughness +
             1.4385f * dr::square(roughness);
  Float ry = 1.0f - 0.6076f * roughness - 0.4993f * NdotV * roughness +
             2.0315f * dr::square(roughness);
  return dr::clip(rx / dr::maximum(ry, 1e-6f), 0.0f, 1.0f);
}

constexpr float kFujii1 = 0.5f - 2.0f / (3.0f * dr::Pi<float>);
constexpr float kFujii2 = 2.0f / 3.0f - 28.0f / (15.0f * dr::Pi<float>);

template <typename Float>
Float mx_fujii_dir_albedo(const Float& cos_theta, const Float& roughness,
                          const Float& A) {
  Float ct = dr::clip(cos_theta, 1e-5f, 1.0f);
  Float Si2 = 1.0f - dr::square(ct);
  Float Si = dr::safe_sqrt(Si2);
  Float G = Si * (dr::safe_acos(ct) - Si * ct) +
            2.0f * ((Si / ct) * (1.0f - Si2 * Si) - Si) / 3.0f;
  return dr::clip(A * (1.0f + roughness * G * dr::InvPi<Float>), 0.0f, 1.0f);
}

template <typename Float, typename Col>
std::tuple<Float, Float, Col> mx_fujii_ms_albedo(const Float& roughness,
                                                 const Col& color) {
  Float A = 1.0f / (1.0f + kFujii1 * roughness);
  Float avg_a = A * (1.0f + kFujii2 * roughness);
  return {A, avg_a,
          dr::square(color) * avg_a /
              dr::maximum(Col(1.0f) - color * dr::maximum(0.0f, 1.0f - avg_a),
                          Col(1e-5f))};
}

template <typename Float, typename Col>
Col mx_oren_nayar_compensated(const Float& NdotV, const Float& NdotL,
                              const Float& LdotV, const Float& roughness,
                              const Col& color) {
  Float s = LdotV - NdotL * NdotV;
  Float stinv = dr::select(
      s > 0.0f, s / dr::maximum(dr::maximum(NdotL, NdotV), 1e-6f), s);
  auto [A, avg_a, col_ms] = mx_fujii_ms_albedo(roughness, color);
  Col multi_s =
      col_ms *
      dr::maximum(1e-5f, 1.0f - mx_fujii_dir_albedo(NdotV, roughness, A)) *
      dr::maximum(1e-5f, 1.0f - mx_fujii_dir_albedo(NdotL, roughness, A)) /
      dr::maximum(1e-5f, 1.0f - avg_a);
  return dr::maximum(color * A * (1.0f + roughness * stinv) + multi_s,
                     Col(0.0f));
}

template <typename Float, typename Col>
Col mx_oren_nayar_compensated_dir_albedo(const Float& NdotV,
                                         const Float& roughness,
                                         const Col& color) {
  auto [A, avg_a, col_ms] = mx_fujii_ms_albedo(roughness, color);
  return dr::lerp(col_ms, color, mx_fujii_dir_albedo(NdotV, roughness, A));
}

template <typename Float>
std::pair<Float, Float> mx_ggx_dir_albedo_ab(const Float& NdotV,
                                             const Float& alpha) {
  Float x = dr::clip(NdotV, 0.0f, 1.0f);
  Float y = dr::clip(alpha, 0.0f, 1.0f);
  Float x2 = x * x, y2 = y * y;
  Float rx = 0.1003f - 0.6303f * x + 9.748f * y - 2.038f * x * y + 29.34f * x2 -
             8.245f * y2 - 26.44f * x2 * y + 19.99f * x * y2 - 5.448f * x2 * y2;
  Float ry = 0.9345f - 2.323f * x + 2.229f * y - 3.748f * x * y + 1.424f * x2 -
             0.7684f * y2 + 1.436f * x2 * y + 0.2913f * x * y2 +
             0.6286f * x2 * y2;
  Float rz = 1.0f - 1.765f * x + 8.263f * y + 11.53f * x * y + 28.96f * x2 -
             7.507f * y2 - 36.11f * x2 * y + 15.86f * x * y2 + 33.37f * x2 * y2;
  Float rw = 1.0f + 0.2281f * x + 15.94f * y - 55.83f * x * y + 13.08f * x2 +
             41.26f * y2 + 54.9f * x2 * y + 300.2f * x * y2 - 285.1f * x2 * y2;
  return {dr::clip(rx / dr::maximum(rz, 1e-5f), 0.0f, 1.0f),
          dr::clip(ry / dr::maximum(rw, 1e-5f), 0.0f, 1.0f)};
}

template <typename Float, typename T>
T mx_ggx_energy_compensation_from_ess(const Float& Ess, const T& Fss) {
  return T(1.0f) + Fss * ((1.0f - Ess) / dr::maximum(Ess, 1e-4f));
}

template <typename Float, typename T>
T mx_ggx_energy_compensation(const Float& NdotV, const Float& alpha,
                             const T& Fss) {
  auto [ax, ay] = mx_ggx_dir_albedo_ab(NdotV, alpha);
  return mx_ggx_energy_compensation_from_ess(ax + ay, Fss);
}

/**!

.. _bsdf-materialx_bsdf:

MaterialX BSDF (:monosp:`materialx_bsdf`)
-----------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxProgram`` bytecode, closure DAG, and
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

Evaluates and importance-samples a compiled MaterialX closure DAG over surface
interactions.

*/
template <typename Float, typename Spectrum>
class MtlxBsdf final : public BSDF<Float, Spectrum> {
 public:
  MI_IMPORT_BASE(BSDF, m_flags, m_components)
  MI_IMPORT_TYPES(MicrofacetDistribution)
  using Val = typename MtlxEvaluator<Float, Spectrum>::Val;
  // Color type of closure inputs and values. Spectral variants evaluate the
  // closures per wavelength on upsampled inputs; other variants use RGB.
  using Col =
      std::conditional_t<is_spectral_v<Spectrum>, UnpolarizedSpectrum, Color3f>;

  explicit MtlxBsdf(const Properties& props) : Base(props) {
    m_eval.compile(props);
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();

    m_frame_idx.resize(n);
    for (size_t i = 0; i < n; ++i) {
      if (!is_leaf(closures[i].op)) continue;
      std::pair<int, int> key(closures[i].args[leaf_arg::kNormal],
                              closures[i].args[leaf_arg::kTangent]);
      auto it = std::find(m_unique_frames.begin(), m_unique_frames.end(), key);
      m_frame_idx[i] = static_cast<size_t>(it - m_unique_frames.begin());
      if (it == m_unique_frames.end()) m_unique_frames.push_back(key);
    }

    if constexpr (is_spectral_v<Spectrum>) {
      m_color_idx.resize(n);
      auto add_color = [&](int slot, bool unbounded) -> int {
        if (slot < 0) return -1;
        for (size_t k = 0; k < m_color_slots.size(); ++k) {
          if (m_color_slots[k].tmp_slot == slot &&
              m_color_slots[k].unbounded == unbounded) {
            return static_cast<int>(k);
          }
        }
        m_color_slots.push_back({slot, unbounded});
        return static_cast<int>(m_color_slots.size()) - 1;
      };
      for (size_t i = 0; i < n; ++i) {
        const ClosureNode& c = closures[i];
        if (is_combinator(c.op)) {
          m_color_idx[i].primary =
              add_color(c.args[combinator_arg::kWeight], false);
        } else if (is_leaf(c.op)) {
          m_color_idx[i].primary =
              add_color(c.args[leaf_arg::kColor], c.op == ClosureOp::Conductor);
          if (c.op == ClosureOp::Conductor) {
            m_color_idx[i].secondary =
                add_color(c.args[leaf_arg::kExtinction], true);
          }
        }
      }
      if (std::any_of(m_color_slots.begin(), m_color_slots.end(),
                      [&](const ColorSlot& cs) {
                        return !prog().is_tmp_uniform(cs.tmp_slot);
                      })) {
        m_eval.load_srgb_lut();
      }
    }

    m_needs_throughput.assign(n, 0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      const ClosureNode& c = closures[i];
      if (is_binary(c.op)) m_has_binary = true;
      if (c.op == ClosureOp::Layer) {
        m_needs_throughput[c.args[combinator_arg::kFirst]] = 1;
        m_has_layers = true;
      }
      if (m_needs_throughput[i] && is_combinator(c.op)) {
        m_needs_throughput[c.args[combinator_arg::kFirst]] = 1;
        if (c.op != ClosureOp::Multiply) {
          m_needs_throughput[c.args[combinator_arg::kSecond]] = 1;
        }
      }
    }

    uint32_t flags = BSDFFlags::FrontSide | BSDFFlags::BackSide;
    if (m_eval.is_spatially_varying()) flags |= +BSDFFlags::SpatiallyVarying;
    if (prog().opacity_slot >= 0) flags |= +BSDFFlags::Null;
    for (const ClosureNode& c : closures) {
      switch (c.op) {
        case ClosureOp::OrenNayarDiffuse:
          flags |= +BSDFFlags::DiffuseReflection;
          break;
        case ClosureOp::Translucent:
          flags |= +BSDFFlags::DiffuseTransmission;
          break;
        case ClosureOp::Conductor:
          flags |= BSDFFlags::GlossyReflection | BSDFFlags::Anisotropic |
                   BSDFFlags::NonSymmetric;
          break;
        case ClosureOp::Dielectric: {
          int sm = scatter_mode(c);
          if (sm != kScatterT) flags |= +BSDFFlags::GlossyReflection;
          if (sm != kScatterR) flags |= +BSDFFlags::GlossyTransmission;
          flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
          break;
        }
        case ClosureOp::Layer:
          flags |= +BSDFFlags::NonSymmetric;
          break;
        default:
          break;
      }
      if (is_leaf(c.op) &&
          (c.args[leaf_arg::kNormal] >= 0 || c.args[leaf_arg::kTangent] >= 0)) {
        flags |= +BSDFFlags::NormalMapped;
      }
    }
    m_components.push_back(flags);
    m_flags = flags;
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
    return eval_pdf_impl(ctx, eval_state(si, active), wo);
  }

  std::tuple<Spectrum, Float, BSDFSample3f, Spectrum> eval_pdf_sample(
      const BSDFContext& ctx, const SurfaceInteraction3f& si,
      const Vector3f& wo, Float sample1, const Point2f& sample2,
      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    State s = eval_state(si, active);
    return std::tuple_cat(eval_pdf_impl(ctx, s, wo),
                          sample_impl(ctx, si, s, sample1, sample2, active));
  }

  Spectrum eval_null(const SurfaceInteraction3f& si,
                     Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    int op_slot = prog().opacity_slot;
    if (op_slot < 0) return 0.0f;
    return Spectrum(UnpolarizedSpectrum(
               1.0f -
               clip_opacity(m_eval.eval(si, active, nullptr, op_slot)))) &
           active;
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
  struct ColorSlot {
    int tmp_slot;
    bool unbounded;
  };
  struct ColorRef {
    int primary = -1;
    int secondary = -1;
  };

  // Per-query material state shared by all BSDF entry points.
  struct State {
    absl::InlinedVector<Val, 32> tmps;
    // Compact upsampled color temporaries (spectral variants only).
    absl::InlinedVector<Col, is_spectral_v<Spectrum> ? 16 : 1> colors;
    // Deduplicated local shading frames and incident directions (relative to
    // `si.sh_frame`).
    absl::InlinedVector<Frame3f, 4> frames;
    absl::InlinedVector<Vector3f, 4> wi_local;
    // Layer throughput (`top.throughput` in MaterialX `genglsl`) of each
    // closure.
    absl::InlinedVector<Col, 16> throughput;
    // Probability of selecting the first child of each binary combinator.
    absl::InlinedVector<Float, 16> child_prob;
    Vector3f n_geom;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  const MtlxProgram& prog() const { return m_eval.program(); }
  static Vector3f xyz(const Val& v) { return dr::head<3>(v); }
  static bool is_leaf(ClosureOp op) {
    return op > ClosureOp::Null && op < ClosureOp::Layer;
  }
  static bool is_combinator(ClosureOp op) { return op >= ClosureOp::Layer; }
  static bool is_binary(ClosureOp op) {
    return is_combinator(op) && op != ClosureOp::Multiply;
  }
  static int scatter_mode(const ClosureNode& c) {
    return c.args[leaf_arg::kMode] & kScatterModeMask;
  }
  static bool is_thin_walled(const ClosureNode& c) {
    return (c.args[leaf_arg::kMode] & kScatterThinWalled) != 0;
  }
  static bool is_two_sided(const ClosureNode& c) {
    return (c.args[leaf_arg::kMode] & kScatterTwoSided) != 0;
  }

  // Evaluates the VM, upsampled colors, unique shading frames, and per-closure
  // directional albedos / throughputs.
  State eval_state(const SurfaceInteraction3f& si, Mask active) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();
    int op_slot = prog().opacity_slot;

    State s;
    s.tmps.resize(prog().tmp_channels.size());
    m_eval.eval(si, active, s.tmps.data());
    s.opacity = op_slot < 0 ? Float(1.0f) : clip_opacity(s.tmps[op_slot]);
    s.surf_active = (op_slot < 0 || dr::grad_enabled(s.opacity))
                        ? active
                        : (active && (s.opacity > 0.0f));
    s.has_surface = !(n == 1 && closures[0].op == ClosureOp::Null) &&
                    may_be_true(s.surf_active);
    if (!s.has_surface) return s;

    s.n_geom = safe_normalize(si.to_local(si.n), Vector3f(0.0f, 0.0f, 1.0f),
                              ScalarFloat(1e-20f));
    Float wi_side = dr::mulsign(Float(1.0f), dr::dot(s.n_geom, si.wi));

    if constexpr (is_spectral_v<Spectrum>) {
      s.colors.resize(m_color_slots.size());
      for (size_t k = 0; k < m_color_slots.size(); ++k) {
        const ColorSlot& cs = m_color_slots[k];
        const Val& v = s.tmps[cs.tmp_slot];
        s.colors[k] = prog().is_tmp_uniform(cs.tmp_slot)
                          ? Col(cs.unbounded ? dr::maximum(v.x(), 0.f)
                                             : dr::clip(v.x(), 0.f, 1.f))
                          : m_eval.upsample_rgb(Color3f(xyz(v)), si.wavelengths,
                                                cs.unbounded);
      }
    }

    s.frames.resize(m_unique_frames.size());
    s.wi_local.resize(m_unique_frames.size());
    for (size_t f = 0; f < m_unique_frames.size(); ++f) {
      s.frames[f] =
          build_local_frame(m_unique_frames[f].first, m_unique_frames[f].second,
                            si, s.n_geom, wi_side, s.tmps.data());
      s.wi_local[f] = s.frames[f].to_local(si.wi);
    }

    if (m_has_binary) {
      absl::InlinedVector<Col, 16> albedo(n - 1);
      if (m_has_layers) s.throughput.resize(n);
      s.child_prob.resize(n);
      for (size_t i = 0; i < n; ++i) {
        if (i + 1 < n) eval_node_albedo_and_throughput(i, s, albedo.data());
        if (is_binary(closures[i].op)) {
          s.child_prob[i] = first_child_prob(i, s, albedo.data());
        }
      }
    }
    return s;
  }

  static Float clip_opacity(const Val& op) {
    return dr::clip(op.x(), 0.0f, 1.0f);
  }

  bool has_cutout(const BSDFContext& ctx) const {
    return prog().opacity_slot >= 0 && ctx.is_enabled(BSDFFlags::Null);
  }

  std::pair<Spectrum, Float> eval_pdf_impl(const BSDFContext& ctx,
                                           const State& s,
                                           const Vector3f& wo) const {
    if (!s.has_surface) return {0.0f, 0.0f};
    auto [f, pdf] = eval_and_pdf_tree(ctx, s, wo);
    return {depolarizer<Spectrum>(color_to_spectrum(f)) & s.surf_active,
            dr::select(s.surf_active, has_cutout(ctx) ? pdf * s.opacity : pdf,
                       0.0f)};
  }

  std::pair<BSDFSample3f, Spectrum> sample_impl(const BSDFContext& ctx,
                                                const SurfaceInteraction3f& si,
                                                const State& s, Float sample1,
                                                const Point2f& sample2,
                                                Mask active) const {
    // Stochastic cutout opacity.
    bool cutout = has_cutout(ctx);
    Float prob_surface = cutout ? s.opacity : Float(1.0f);
    Mask pass_through =
        cutout ? (active && (prob_surface < 1.0f) && (sample1 >= prob_surface))
               : Mask(false);

    Vector3f wo(0.0f, 0.0f, 1.0f);
    UInt32 sampled_type(0);
    Float sampled_eta(1.0f);
    Mask leaf_valid(false);
    if (s.has_surface) {
      if (cutout) {
        Mask has_prob = prob_surface > 0.0f;
        sample1 = dr::select(
            has_prob,
            dr::minimum(sample1 / dr::select(has_prob, prob_surface, 1.0f),
                        dr::OneMinusEpsilon<Float>),
            0.0f);
      }

      // Top-down selection and sampling of a leaf closure using `sample1`.
      const std::vector<ClosureNode>& closures = prog().closures;
      size_t n = closures.size();
      absl::InlinedVector<Mask, 16> node_sel(n, Mask(false));
      absl::InlinedVector<Float, 16> node_xi(n, Float(0.0f));
      node_sel[n - 1] = s.surf_active && !pass_through;
      node_xi[n - 1] = sample1;
      auto route = [&](int child, const Mask& pick, const Float& xi) {
        node_sel[child] = node_sel[child] || pick;
        node_xi[child] =
            dr::select(pick, dr::clip(xi, 0.0f, dr::OneMinusEpsilon<Float>),
                       node_xi[child]);
      };

      for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        const Mask& sel = node_sel[i];
        if (!may_be_true(sel)) continue;
        const ClosureNode& c = closures[i];
        const Float& xi = node_xi[i];
        if (is_leaf(c.op)) {
          auto [leaf_wo, leaf_type, leaf_eta, valid] =
              sample_leaf(ctx, i, xi, sample2, s);
          Mask hit = sel && valid;
          wo = dr::select(hit, leaf_wo, wo);
          sampled_type = dr::select(hit, leaf_type, sampled_type);
          sampled_eta = dr::select(hit, leaf_eta, sampled_eta);
          leaf_valid = leaf_valid || hit;
          continue;
        }
        int c0 = c.args[combinator_arg::kFirst];
        if (c.op == ClosureOp::Multiply) {
          route(c0, sel, xi);
        } else {
          const Float& p = s.child_prob[i];
          Mask pick_first = sel && (xi < p);
          route(c0, pick_first, xi / p);
          route(c.args[combinator_arg::kSecond], sel && !pick_first,
                (xi - p) / (1.0f - p));
        }
      }
    }

    // Detached one-sample MIS over the lobe mixture: weight = f_tot / pdf_tot.
    wo = dr::detach(wo);
    auto [f_tot, pdf_tot] = may_be_true(leaf_valid)
                                ? eval_and_pdf_tree(ctx, s, wo)
                                : std::pair<Col, Float>(Col(0.0f), 0.0f);
    Float surf_pdf = dr::select(leaf_valid, pdf_tot * prob_surface, 0.0f);
    Mask surf_valid = leaf_valid && (surf_pdf > 0.0f);

    BSDFSample3f bs = dr::zeros<BSDFSample3f>();
    bs.wo = dr::select(pass_through, -si.wi, wo);
    bs.pdf =
        dr::detach(dr::select(pass_through, 1.0f - prob_surface, surf_pdf));
    bs.eta = dr::select(pass_through, Float(1.0f), sampled_eta);
    bs.sampled_type =
        dr::select(pass_through, UInt32(+BSDFFlags::Null), sampled_type);

    Float pass_weight(1.0f);
    if constexpr (dr::is_diff_v<Float>) {
      if (cutout && dr::grad_enabled(s.opacity)) {
        pass_weight = dr::replace_grad(
            pass_weight, (1.0f - s.opacity) /
                             dr::select(pass_through,
                                        dr::detach(1.0f - prob_surface), 1.0f));
      }
    }
    Col surf_weight =
        f_tot / dr::select(surf_valid, dr::detach(surf_pdf), 1.0f);
    return {bs,
            dr::select(pass_through, Spectrum(UnpolarizedSpectrum(pass_weight)),
                       depolarizer<Spectrum>(color_to_spectrum(surf_weight)) &
                           surf_valid)};
  }

  static UnpolarizedSpectrum color_to_spectrum(const Col& c) {
    if constexpr (is_monochromatic_v<Spectrum>) {
      return luminance(c);
    } else {
      return c;
    }
  }

  // Scalar weight of a closure color, used for lobe selection probabilities.
  static Float selection_weight(const Col& c) {
    if constexpr (is_spectral_v<Spectrum>) {
      return dr::mean(c);
    } else {
      return luminance(c);
    }
  }

  static Float get_weight(const ClosureNode& c, const State& s) {
    return dr::maximum(s.tmps[c.args[leaf_arg::kWeight]].x(), 0.0f);
  }

  static Vector2f get_roughness(const ClosureNode& c, const State& s) {
    return dr::clip(dr::head<2>(s.tmps[c.args[leaf_arg::kRoughness]]),
                    Vector2f(1e-3f), Vector2f(1.0f));
  }

  static Float dielectric_ior(const ClosureNode& c, const State& s) {
    Float ior = dr::maximum(s.tmps[c.args[leaf_arg::kIor]].x(), 1e-4f);
    return dr::select(
        dr::abs(ior - 1.0f) < 1e-4f,
        dr::select(ior < 1.0f, Float(1.0f - 1e-4f), Float(1.0f + 1e-4f)), ior);
  }

  // Returns non-negative color input of closure `i` (upsampled in spectral
  // variants).
  Col get_color(size_t i, const State& s, bool secondary = false) const {
    if constexpr (is_spectral_v<Spectrum>) {
      return s.colors[secondary ? m_color_idx[i].secondary
                                : m_color_idx[i].primary];
    } else {
      const ClosureNode& c = prog().closures[i];
      int slot = secondary ? c.args[leaf_arg::kExtinction]
                           : c.args[is_leaf(c.op) ? leaf_arg::kColor
                                                  : combinator_arg::kWeight];
      return dr::maximum(Color3f(xyz(s.tmps[slot])), Color3f(0.0f));
    }
  }

  // Clamps conductor `ior` away from zero to avoid `0 / 0` in
  // `fresnel_conductor` at normal incidence when `ior == extinction == 0`.
  dr::Complex<Col> conductor_eta(size_t i, const State& s) const {
    return dr::Complex<Col>(dr::maximum(get_color(i, s), Col(1e-6f)),
                            get_color(i, s, /*secondary=*/true));
  }

  // Wraps `mitsuba::fresnel`, clamping `F` to exact `1` under total internal
  // reflection (`cos_theta_t == 0`), where `dr::rcp` in `mitsuba::fresnel` can
  // round `F` just below `1` (e.g. `0.99999988f` on AVX/CUDA).
  static auto exact_fresnel(const Float& cos_theta_i, const Float& eta) {
    auto [F, cos_theta_t, eta_it, eta_ti] = fresnel(cos_theta_i, eta);
    F = dr::select(cos_theta_t == 0.0f, Float(1.0f), F);
    return std::make_tuple(F, cos_theta_t, eta_it, eta_ti);
  }

  template <typename T>
  static T ggx_refl_dir_albedo(const ClosureNode& c, const Float& NdotV,
                               const T& F, const T& F0, const State& s) {
    Vector2f alpha = get_roughness(c, s);
    auto [ax, ay] =
        mx_ggx_dir_albedo_ab(NdotV, dr::sqrt(alpha.x() * alpha.y()));
    return mx_ggx_energy_compensation_from_ess(ax + ay, F) * (F0 * ax + ay);
  }

  static Frame3f build_local_frame(int normal_slot, int tangent_slot,
                                   const SurfaceInteraction3f& si,
                                   const Vector3f& n_geom, const Float& wi_side,
                                   const Val* tmps) {
    Vector3f nl = normal_slot >= 0
                      ? si.to_local(safe_normalize(xyz(tmps[normal_slot]),
                                                   Vector3f(si.sh_frame.n),
                                                   ScalarFloat(1e-20f)))
                      : Vector3f(0.0f, 0.0f, 1.0f);
    Float ndotg = dr::dot(n_geom, nl);
    nl = dr::select(ndotg < 0.0f, nl - 2.0f * ndotg * n_geom, nl);
    ndotg = dr::abs(ndotg);
    // Keep the shading normal on the geometric side of `si.wi` (Schuessler et
    // al. 2017; reduces to `(-nl.x, -nl.y, nl.z)` when `n_geom == (0, 0, 1)`).
    Vector3f n_loc = dr::select(
        wi_side * dr::dot(si.wi, nl) <= 0.0f,
        dr::select(ndotg > 0.0f, 2.0f * ndotg * n_geom - nl, n_geom), nl);
    Vector3f t_raw = si.to_local(
        tangent_slot >= 0 ? safe_normalize(xyz(tmps[tangent_slot]),
                                           si.sh_frame.s, ScalarFloat(1e-20f))
                          : geom_tangent_world(si));
    Vector3f s_loc =
        safe_normalize(t_raw - n_loc * dr::dot(n_loc, t_raw),
                       coordinate_system(n_loc).first, ScalarFloat(1e-20f));
    return Frame3f(s_loc, dr::cross(n_loc, s_loc), n_loc);
  }

  // Probability of selecting the first child of a binary combinator
  // (`layer`, `mix`, `add`) when sampling. The tree pdf in
  // `eval_and_pdf_tree()` uses the same probabilities.
  Float first_child_prob(size_t i, const State& s, const Col* albedo) const {
    const ClosureNode& c = prog().closures[i];
    int c0 = c.args[combinator_arg::kFirst],
        c1 = c.args[combinator_arg::kSecond];
    const Col &a0 = albedo[c0], &a1 = albedo[c1];
    Float w0, w1, fallback(0.5f);
    if (c.op == ClosureOp::Layer) {
      w0 = selection_weight(a0);
      w1 = selection_weight(s.throughput[c0] * a1);
    } else if (c.op == ClosureOp::Mix) {
      Col m = get_color(i, s);
      w0 = selection_weight(m * a0);
      w1 = selection_weight((Col(1.0f) - m) * a1);
      fallback = selection_weight(m);
    } else {
      w0 = selection_weight(a0);
      w1 = selection_weight(a1);
    }
    Float w_sum = w0 + w1;
    Mask ok = w_sum > 1e-6f;
    return dr::clip(dr::select(ok, w0 / dr::select(ok, w_sum, 1.0f), fallback),
                    0.05f, 0.95f);
  }

  void eval_node_albedo_and_throughput(size_t i, State& s, Col* albedo) const {
    const ClosureNode& c = prog().closures[i];
    bool need_tp = m_needs_throughput[i] != 0;
    switch (c.op) {
      case ClosureOp::Null:
        albedo[i] = Col(0.0f);
        if (need_tp) s.throughput[i] = Col(1.0f);
        return;
      case ClosureOp::OrenNayarDiffuse: {
        Col col = get_color(i, s);
        Float r =
            dr::clip(s.tmps[c.args[leaf_arg::kRoughness]].x(), 0.0f, 1.0f);
        Float NdotV = dr::abs(Frame3f::cos_theta(s.wi_local[m_frame_idx[i]]));
        albedo[i] = get_weight(c, s) *
                    (c.args[leaf_arg::kMode] == 1
                         ? mx_oren_nayar_compensated_dir_albedo(NdotV, r, col)
                         : col * mx_oren_nayar_dir_albedo(NdotV, r));
        if (need_tp) s.throughput[i] = Col(0.0f);
        return;
      }
      case ClosureOp::Translucent:
        albedo[i] = get_weight(c, s) * get_color(i, s);
        if (need_tp) s.throughput[i] = Col(0.0f);
        return;
      case ClosureOp::Dielectric: {
        int sm = scatter_mode(c);
        Float w = get_weight(c, s);
        Col tint = get_color(i, s);
        if (sm == kScatterR) {
          Float cos_i = Frame3f::cos_theta(s.wi_local[m_frame_idx[i]]),
                NdotV = dr::abs(cos_i), ior = dielectric_ior(c, s);
          auto [F_smooth, cos_t, eta_it, eta_ti] =
              exact_fresnel(is_two_sided(c) ? NdotV : cos_i, ior);
          Float F0 = dr::select(eta_it < 1.0f, F_smooth,
                                dr::square((ior - 1.0f) / (ior + 1.0f)));
          Float refl_albedo =
              w * ggx_refl_dir_albedo(c, NdotV, F_smooth, F0, s);
          albedo[i] = refl_albedo * tint;
          if (need_tp) {
            s.throughput[i] = Col(dr::maximum(0.0f, 1.0f - refl_albedo));
          }
          return;
        }
        if (need_tp) s.throughput[i] = Col(0.0f);
        if (sm == kScatterRT) {
          albedo[i] = w * tint;
        } else {
          Float cos_i = Frame3f::cos_theta(s.wi_local[m_frame_idx[i]]);
          Float F_smooth = std::get<0>(exact_fresnel(
              is_two_sided(c) ? dr::abs(cos_i) : cos_i, dielectric_ior(c, s)));
          albedo[i] = dr::select(F_smooth >= 1.0f, Col(0.0f), w * tint);
        }
        return;
      }
      case ClosureOp::Conductor: {
        Float NdotV = dr::maximum(
            dr::abs(Frame3f::cos_theta(s.wi_local[m_frame_idx[i]])), 1e-6f);
        dr::Complex<Col> eta_c = conductor_eta(i, s);
        albedo[i] =
            get_weight(c, s) *
            ggx_refl_dir_albedo(c, NdotV, fresnel_conductor(Col(NdotV), eta_c),
                                fresnel_conductor(Col(1.0f), eta_c), s);
        if (need_tp) s.throughput[i] = Col(0.0f);
        return;
      }
      case ClosureOp::Layer:
      case ClosureOp::Mix:
      case ClosureOp::Add:
      case ClosureOp::Multiply: {
        albedo[i] = eval_combinator(i, s, albedo);
        if (!need_tp) return;
        int c0 = c.args[combinator_arg::kFirst],
            c1 = c.args[combinator_arg::kSecond];
        if (c.op == ClosureOp::Layer) {
          s.throughput[i] = s.throughput[c0] * s.throughput[c1];
        } else if (c.op == ClosureOp::Mix) {
          s.throughput[i] =
              dr::lerp(s.throughput[c1], s.throughput[c0], get_color(i, s));
        } else if (c.op == ClosureOp::Add) {
          s.throughput[i] = dr::maximum(
              Col(0.0f), s.throughput[c0] + s.throughput[c1] - Col(1.0f));
        } else {  // Multiply
          s.throughput[i] =
              dr::lerp(Col(1.0f), s.throughput[c0], get_color(i, s));
        }
        return;
      }
    }
  }

  Col eval_combinator(size_t i, const State& s, const Col* vals) const {
    const ClosureNode& c = prog().closures[i];
    int c0 = c.args[combinator_arg::kFirst],
        c1 = c.args[combinator_arg::kSecond];
    switch (c.op) {
      case ClosureOp::Layer:
        return vals[c0] + s.throughput[c0] * vals[c1];
      case ClosureOp::Mix:
        return dr::lerp(vals[c1], vals[c0], get_color(i, s));
      case ClosureOp::Add:
        return vals[c0] + vals[c1];
      default:  // Multiply
        return get_color(i, s) * vals[c0];
    }
  }

  // Samples a direction from leaf closure `i`. Returns the direction (in
  // `si.sh_frame` coordinates), the sampled lobe type, the relative IOR along
  // the sampled direction, and whether the sample is valid.
  std::tuple<Vector3f, UInt32, Float, Mask> sample_leaf(const BSDFContext& ctx,
                                                        size_t i,
                                                        const Float& sample1,
                                                        const Point2f& sample2,
                                                        const State& s) const {
    const ClosureNode& c = prog().closures[i];
    size_t f_idx = m_frame_idx[i];
    const Frame3f& frame = s.frames[f_idx];
    const Vector3f& wi_l = s.wi_local[f_idx];
    Float cos_i = Frame3f::cos_theta(wi_l);
    switch (c.op) {
      case ClosureOp::OrenNayarDiffuse:
      case ClosureOp::Translucent: {
        bool transmit = c.op == ClosureOp::Translucent;
        BSDFFlags type = transmit ? BSDFFlags::DiffuseTransmission
                                  : BSDFFlags::DiffuseReflection;
        if (!ctx.is_enabled(type)) break;
        Vector3f wo_l = warp::square_to_cosine_hemisphere(sample2);
        wo_l.z() = dr::mulsign(wo_l.z(), transmit ? -cos_i : cos_i);
        Vector3f wo = frame.to_world(wo_l);
        return {wo, UInt32(+type), 1.0f,
                (cos_i != 0.0f) &&
                    (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
      }
      case ClosureOp::Conductor: {
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection)) break;
        Vector3f wi_pos(wi_l.x(), wi_l.y(), dr::abs(cos_i));
        Vector2f alpha = get_roughness(c, s);
        MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y(),
                                     true);
        Vector3f wo_pos = reflect(wi_pos, distr.sample(wi_pos, sample2).first);
        Vector3f wo_l(wo_pos.x(), wo_pos.y(),
                      dr::mulsign(Frame3f::cos_theta(wo_pos), cos_i));
        Vector3f wo = frame.to_world(wo_l);
        return {wo, UInt32(+BSDFFlags::GlossyReflection), 1.0f,
                (dr::abs(cos_i) > 1e-6f) &&
                    (Frame3f::cos_theta(wo_pos) > 1e-6f) &&
                    (dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f)};
      }
      case ClosureOp::Dielectric: {
        // Follows `roughdielectric`: sample visible normals for `wi` flipped
        // into the upper hemisphere, then reflect/refract the unflipped `wi`.
        int sm = scatter_mode(c);
        bool refl_on =
            sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection);
        bool trans_on =
            sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission);
        if (!refl_on && !trans_on) break;
        bool thin = is_thin_walled(c);
        Vector2f alpha = get_roughness(c, s);
        MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y(),
                                     true);
        Normal3f m = distr.sample(dr::mulsign(wi_l, cos_i), sample2).first;
        Vector3f wo_l = reflect(wi_l, m);
        Mask do_reflect(refl_on);
        Mask trans_ok(false);
        Float out_eta(1.0f);
        if (trans_on) {
          Float ior = dielectric_ior(c, s);
          Float cos_im = dr::dot(wi_l, m);
          auto [F, cos_theta_t, eta_it, eta_ti] =
              exact_fresnel(is_two_sided(c) ? dr::abs(cos_im) : cos_im, ior);
          trans_ok = F < 1.0f;
          if (refl_on) do_reflect = sample1 < F;
          Vector3f wo_trans = thin ? Vector3f(wo_l.x(), wo_l.y(), -wo_l.z())
                                   : refract(wi_l, m, cos_theta_t, eta_ti);
          wo_l = dr::select(do_reflect, wo_l, wo_trans);
          if (!thin) out_eta = dr::select(do_reflect, Float(1.0f), eta_it);
        }
        Vector3f wo = frame.to_world(wo_l);
        Float cos_o = Frame3f::cos_theta(wo_l);
        Mask valid = dr::select(do_reflect, cos_i * cos_o > 0.0f,
                                trans_ok && (cos_i * cos_o < 0.0f)) &&
                     (dr::dot(s.n_geom, wo) * cos_o > 0.0f);
        return {wo,
                dr::select(do_reflect, UInt32(+BSDFFlags::GlossyReflection),
                           UInt32(+BSDFFlags::GlossyTransmission)),
                out_eta, valid};
      }
      default:
        break;
    }
    return {Vector3f(0.0f, 0.0f, 1.0f), UInt32(0), 1.0f, Mask(false)};
  }

  std::pair<Col, Float> eval_and_pdf_leaf(const BSDFContext& ctx, size_t i,
                                          const Vector3f& wo,
                                          const State& s) const {
    const ClosureNode& c = prog().closures[i];
    size_t f_idx = m_frame_idx[i];
    const Vector3f& wi_l = s.wi_local[f_idx];
    Vector3f wo_l = s.frames[f_idx].to_local(wo);
    Float cos_i = Frame3f::cos_theta(wi_l), cos_o = Frame3f::cos_theta(wo_l);
    Mask same_side = dr::dot(s.n_geom, wo) * cos_o > 0.0f;
    switch (c.op) {
      case ClosureOp::OrenNayarDiffuse:
      case ClosureOp::Translucent: {
        bool transmit = c.op == ClosureOp::Translucent;
        if (!ctx.is_enabled(transmit ? BSDFFlags::DiffuseTransmission
                                     : BSDFFlags::DiffuseReflection)) {
          break;
        }
        Col col = get_color(i, s);
        Float NdotV = dr::abs(cos_i),
              NdotL = dr::mulsign(cos_o, transmit ? -cos_i : cos_i);
        Mask valid = (NdotV > 0.0f) && (NdotL > 0.0f) && same_side;
        Float pdf_val = NdotL * dr::InvPi<Float>;
        if (!transmit) {
          Float r =
              dr::clip(s.tmps[c.args[leaf_arg::kRoughness]].x(), 0.0f, 1.0f);
          Float LdotV = dr::dot(wi_l, wo_l);
          col = c.args[leaf_arg::kMode] == 1
                    ? mx_oren_nayar_compensated(NdotV, NdotL, LdotV, r, col)
                    : col * mx_oren_nayar_diffuse(NdotV, NdotL, LdotV, r);
        }
        return {dr::select(valid, get_weight(c, s) * col * pdf_val, Col(0.0f)),
                dr::select(valid, pdf_val, 0.0f)};
      }
      case ClosureOp::Conductor: {
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection)) break;
        Vector3f wi_pos(wi_l.x(), wi_l.y(), dr::abs(cos_i));
        Vector3f wo_pos(wo_l.x(), wo_l.y(), dr::mulsign(cos_o, cos_i));
        Float NdotV = Frame3f::cos_theta(wi_pos);
        Mask valid = (NdotV > 1e-6f) && (Frame3f::cos_theta(wo_pos) > 1e-6f) &&
                     same_side;
        Vector2f alpha = get_roughness(c, s);
        MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y(),
                                     true);
        Vector3f H = safe_normalize(wi_pos + wo_pos, Vector3f(0.0f, 0.0f, 1.0f),
                                    ScalarFloat(1e-20f));
        Float d_g1_over_4cos = distr.eval(H) * distr.smith_g1(wi_pos, H) /
                               (4.0f * dr::select(valid, NdotV, 1.0f));
        Col F = fresnel_conductor(Col(dr::maximum(dr::dot(wi_pos, H), 1e-6f)),
                                  conductor_eta(i, s));
        Col ms_comp = mx_ggx_energy_compensation(
            NdotV, dr::sqrt(alpha.x() * alpha.y()), F);
        return {dr::select(valid,
                           get_weight(c, s) * F * ms_comp *
                               (d_g1_over_4cos * distr.smith_g1(wo_pos, H)),
                           Col(0.0f)),
                dr::select(valid, d_g1_over_4cos, 0.0f)};
      }
      case ClosureOp::Dielectric: {
        // Follows `roughdielectric::eval()` / `pdf()`.
        int sm = scatter_mode(c);
        bool refl_on =
            sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection);
        bool trans_on =
            sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission);
        if (!refl_on && !trans_on) break;
        bool thin = is_thin_walled(c);
        bool solid_trans = trans_on && !thin;
        Float ior = dielectric_ior(c, s);
        Vector2f alpha = get_roughness(c, s);
        MicrofacetDistribution distr(MicrofacetType::GGX, alpha.x(), alpha.y(),
                                     true);
        Mask is_refl = cos_i * cos_o > 0.0f;
        Vector3f wo_m = thin ? Vector3f(wo_l.x(), wo_l.y(),
                                        dr::mulsign(dr::abs(cos_o), cos_i))
                             : wo_l;
        Float eta = solid_trans ? dr::select(cos_i > 0.0f, ior, dr::rcp(ior))
                                : Float(1.0f);
        Vector3f m = safe_normalize(
            wi_l + (solid_trans ? wo_m * dr::select(is_refl, Float(1.0f), eta)
                                : wo_m),
            Vector3f(0.0f, 0.0f, 1.0f), ScalarFloat(1e-20f));
        m = dr::mulsign(m, Frame3f::cos_theta(m));
        Float cos_im = dr::dot(wi_l, m), cos_om = dr::dot(wo_m, m);
        Mask valid = (cos_i != 0.0f) && (cos_o != 0.0f) &&
                     (cos_im * cos_i > 0.0f) &&
                     (cos_om * Frame3f::cos_theta(wo_m) > 0.0f) && same_side;

        Float F_d = std::get<0>(
            exact_fresnel(is_two_sided(c) ? dr::abs(cos_im) : cos_im, ior));
        Float P_r = !refl_on ? Float(0.0f) : !trans_on ? Float(1.0f) : F_d;
        Float abs_cos_i = dr::abs(cos_i);
        Float d_g1_over_cos = distr.eval(m) * distr.smith_g1(wi_l, m) /
                              dr::select(valid, abs_cos_i, 1.0f);
        Float G1_o = distr.smith_g1(wo_m, m);
        Float refl_pdf = 0.25f * d_g1_over_cos, refl_value = refl_pdf * G1_o;
        Float f(0.0f), pdf(0.0f);
        if (refl_on) {
          Mask use_refl = valid && is_refl;
          Float ms = mx_ggx_energy_compensation(
              abs_cos_i, dr::sqrt(alpha.x() * alpha.y()), F_d);
          f = dr::select(use_refl, F_d * ms * refl_value, f);
          pdf = dr::select(use_refl, P_r * refl_pdf, pdf);
        }
        if (trans_on) {
          Mask use_trans = valid && !is_refl && (F_d < 1.0f);
          Float trans_val = refl_value, trans_pdf = (1.0f - P_r) * refl_pdf;
          if (!thin) {
            Float denom = dr::square(cos_im + eta * cos_om);
            use_trans = use_trans && (denom > 1e-20f);
            Float jacobian = dr::abs(dr::square(eta) * cos_om /
                                     dr::select(use_trans, denom, 1.0f));
            Float pdf_m = d_g1_over_cos * dr::abs(cos_im);
            Float scale = ctx.mode == TransportMode::Radiance
                              ? dr::square(dr::rcp(eta))
                              : Float(1.0f);
            trans_val = scale * pdf_m * G1_o * jacobian;
            trans_pdf = (1.0f - P_r) * pdf_m * jacobian;
          }
          f = dr::select(
              use_trans,
              (sm == kScatterT ? Float(1.0f) : 1.0f - F_d) * trans_val, f);
          pdf = dr::select(use_trans, trans_pdf, pdf);
        }
        return {get_color(i, s) * (get_weight(c, s) * f), pdf};
      }
      default:
        break;
    }
    return {Col(0.0f), 0.0f};
  }

  std::pair<Col, Float> eval_and_pdf_tree(const BSDFContext& ctx,
                                          const State& s,
                                          const Vector3f& wo) const {
    const std::vector<ClosureNode>& closures = prog().closures;
    size_t n = closures.size();
    absl::InlinedVector<Col, 16> f_val(n);
    absl::InlinedVector<Float, 16> pdf_val(n);

    for (size_t i = 0; i < n; ++i) {
      const ClosureNode& c = closures[i];
      if (!is_combinator(c.op)) {
        std::tie(f_val[i], pdf_val[i]) = eval_and_pdf_leaf(ctx, i, wo, s);
        continue;
      }
      int c0 = c.args[combinator_arg::kFirst];
      int c1 = c.args[combinator_arg::kSecond];
      f_val[i] = eval_combinator(i, s, f_val.data());
      pdf_val[i] = c.op == ClosureOp::Multiply
                       ? pdf_val[c0]
                       : dr::lerp(pdf_val[c1], pdf_val[c0], s.child_prob[i]);
    }
    return {f_val[n - 1], pdf_val[n - 1]};
  }

  MtlxEvaluator<Float, Spectrum> m_eval;
  bool m_has_binary = false;
  bool m_has_layers = false;
  // Deduplicated `(normal_slot, tangent_slot)` pairs and per-leaf index into
  // `m_unique_frames`.
  std::vector<std::pair<int, int>> m_unique_frames;
  std::vector<size_t> m_frame_idx;
  // Whether each closure's throughput is read by an ancestor `Layer`.
  std::vector<uint8_t> m_needs_throughput;
  // Unique color slots upsampled once per query and per-closure indices into
  // `m_color_slots` (spectral variants only).
  std::vector<ColorSlot> m_color_slots;
  std::vector<ColorRef> m_color_idx;
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
