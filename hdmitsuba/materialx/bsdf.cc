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
#include <mitsuba/render/interaction.h>

#include "hdmitsuba/materialx/bsdf_detail.h"
#include "hdmitsuba/materialx/evaluator.h"
#include "hdmitsuba/materialx/math.h"

namespace mitsuba::mtlx {
namespace {

/**!

.. _bsdf-materialx_bsdf:

MaterialX BSDF (:monosp:`materialx_bsdf`)
-----------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxBsdfProgram`` SSA value program, closure
     DAG, and constant/opaque tables.

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

  using EvalState = typename MtlxEvaluator<Float, Spectrum>::EvalState;

  explicit MtlxBsdf(const Properties& props) : Base(props) {
    MtlxBsdfProgram bsdf_prog = props.get_any<MtlxBsdfProgram>("program");
    eval_.Compile(props, std::move(bsdf_prog.values));
    closures_ = std::move(bsdf_prog.closures);
    opacity_root_ = bsdf_prog.opacity_root;
    comb_spec_.assign(closures_.size(), -1);

    uint32_t surf_flags = 0u;
    for (size_t i = 0; i < closures_.size(); ++i) {
      const ClosureNode& c = closures_[i];
      if (IsLeaf(c.op)) {
        FrameKey key = LeafFrameKey(c);
        auto it = std::find_if(unique_frames_.begin(), unique_frames_.end(),
                               [&](const FrameKey& f) {
                                 return f.normal == key.normal &&
                                        f.tangent == key.tangent &&
                                        f.is_curve == key.is_curve;
                               });
        size_t f_idx = static_cast<size_t>(it - unique_frames_.begin());
        if (it == unique_frames_.end()) {
          unique_frames_.push_back(key);
        } else {
          it->need_tangent |= key.need_tangent;
        }
        Leaf leaf{i, f_idx};
        switch (c.op) {
          case ClosureOp::OrenNayarDiffuse:
          case ClosureOp::BurleyDiffuse:
          case ClosureOp::Sheen:
          case ClosureOp::Translucent:
            leaf.spec[0] = eval_.AddSpectrum(c.diffuse.color);
            if (c.op == ClosureOp::Sheen && c.diffuse.mode == kSheenZeltner) {
              surf_flags |= +BSDFFlags::NonSymmetric;
            }
            break;
          case ClosureOp::Dielectric:
          case ClosureOp::Conductor:
          case ClosureOp::GeneralizedSchlick:
            if (c.op == ClosureOp::Dielectric) {
              leaf.spec[0] = eval_.AddSpectrum(c.microfacet.p[0]);
              if (c.microfacet.p[2] != kInvalidValue) {
                leaf.spec[1] = eval_.AddSpectrum(c.microfacet.p[2]);
              }
            } else if (c.op == ClosureOp::Conductor) {
              leaf.spec[0] =
                  eval_.AddSpectrum(c.microfacet.p[0], /*unbounded=*/true);
              leaf.spec[1] =
                  eval_.AddSpectrum(c.microfacet.p[1], /*unbounded=*/true);
            } else {
              leaf.spec[0] = eval_.AddSpectrum(c.microfacet.p[0]);
              leaf.spec[1] = eval_.AddSpectrum(c.microfacet.p[1]);
              leaf.spec[2] = eval_.AddSpectrum(c.microfacet.p[2]);
            }
            leaf.mf_idx = num_microfacet_++;
            if (key.need_tangent) surf_flags |= +BSDFFlags::Anisotropic;
            surf_flags |= +BSDFFlags::NonSymmetric;
            break;
          case ClosureOp::ChiangHair:
            leaf.spec[0] = eval_.AddSpectrum(c.hair.tint_r);
            leaf.spec[1] = eval_.AddSpectrum(c.hair.tint_tt);
            leaf.spec[2] = eval_.AddSpectrum(c.hair.tint_trt);
            leaf.spec[3] =
                eval_.AddSpectrum(c.hair.absorption, /*unbounded=*/true);
            leaf.hair_idx = num_hair_++;
            surf_flags |= BSDFFlags::Anisotropic | BSDFFlags::NonSymmetric;
            break;
          default:
            break;
        }
        leaves_.push_back(leaf);
        surface_types_ |= LeafType(c);
        if (key.normal != kInvalidValue) surf_flags |= +BSDFFlags::NormalMapped;
      } else if (c.op == ClosureOp::Layer) {
        has_layers_ = true;
        surf_flags |= +BSDFFlags::NonSymmetric;
      } else if (c.op == ClosureOp::Mix || c.op == ClosureOp::Multiply) {
        comb_spec_[i] = eval_.AddSpectrum(c.comb.weight);
      }
    }
    eval_.FinalizeUniforms();

    uint32_t sv =
        eval_.IsSpatiallyVarying() ? +BSDFFlags::SpatiallyVarying : 0u;
    if (surface_types_ != 0u) {
      m_components.push_back(surface_types_ | surf_flags |
                             BSDFFlags::FrontSide | BSDFFlags::BackSide | sv);
    }
    if (opacity_root_ != kInvalidValue) {
      null_component_ = static_cast<int>(m_components.size());
      m_components.push_back(BSDFFlags::Null | BSDFFlags::FrontSide |
                             BSDFFlags::BackSide | sv);
    }
    m_flags = +BSDFFlags::Empty;
    for (uint32_t f : m_components) m_flags |= f;
  }

  void traverse(TraversalCallback* cb) override { eval_.Traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    eval_.ParametersChanged();
  }

  std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext& ctx,
                                           const SurfaceInteraction3f& si,
                                           Float sample1,
                                           const Point2f& sample2,
                                           Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);
    if (unlikely(dr::none_or<false>(active) ||
                 (!HasSurfaceEnabled(ctx) && !HasCutout(ctx)))) {
      return {dr::zeros<BSDFSample3f>(), 0.0f};
    }
    State s = EvaluateState(ctx, si, active);
    if (s.has_surface) ComputeLeafProbs(ctx, s);
    return SampleImpl(ctx, si, s, sample1, sample2, active);
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
    if (unlikely(dr::none_or<false>(active) || !HasSurfaceEnabled(ctx))) {
      return {0.0f, 0.0f};
    }
    State s = EvaluateState(ctx, si, active);
    if (s.has_surface) ComputeLeafProbs(ctx, s);
    return EvalPdfImpl(ctx, s, wo);
  }

  std::tuple<Spectrum, Float, BSDFSample3f, Spectrum> eval_pdf_sample(
      const BSDFContext& ctx, const SurfaceInteraction3f& si,
      const Vector3f& wo, Float sample1, const Point2f& sample2,
      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (unlikely(dr::none_or<false>(active) ||
                 (!HasSurfaceEnabled(ctx) && !HasCutout(ctx)))) {
      return {0.0f, 0.0f, dr::zeros<BSDFSample3f>(), 0.0f};
    }
    State s = EvaluateState(ctx, si, active);
    if (s.has_surface) ComputeLeafProbs(ctx, s);
    return std::tuple_cat(EvalPdfImpl(ctx, s, wo),
                          SampleImpl(ctx, si, s, sample1, sample2, active));
  }

  Spectrum eval_null(const SurfaceInteraction3f& si,
                     Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (opacity_root_ == kInvalidValue || unlikely(dr::none_or<false>(active)))
      return 0.0f;
    EvalState state;
    return Spectrum(1.0f - ClipOpacity(eval_.EvalOpacity(opacity_root_, si,
                                                         active, state))) &
           active;
  }

  BSDFFeatures3f eval_features(const SurfaceInteraction3f& si,
                               Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
    if (unlikely(dr::none_or<false>(active))) {
      return dr::zeros<BSDFFeatures3f>();
    }
    State s = EvaluateState(BSDFContext(), si, active);
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
      auto update_trans = [&](const UnpolarizedSpectrum& trans, const Float& tw,
                              const Vector3f& wt_cand) {
        surf_transmittance += trans;
        Mask pick_wt = s.surf_active && (tw > best_wt_w);
        wt = dr::select(pick_wt, wt_cand, wt);
        best_wt_w = dr::select(pick_wt, tw, best_wt_w);
      };
      auto update_rough = [&](const Float& sw, const Float& rough) {
        spec_roughness = dr::select(sw > best_spec_w, rough, spec_roughness);
        best_spec_w = dr::maximum(best_spec_w, sw);
      };
      for (size_t k = 0; k < leaves_.size(); ++k) {
        const Leaf& leaf = leaves_[k];
        const ClosureNode& c = closures_[leaf.node];
        UnpolarizedSpectrum contrib = s.leaf_weight[k] * LeafAlbedo(k, s);
        if (leaf.mf_idx >= 0) {
          const auto& m = s.microfacet[leaf.mf_idx];
          auto mf = detail::MicrofacetEvalFeatures(
              m, s.wi_local[leaf.frame],
              m.has_trans_tint ? s.leaf_weight[k] : contrib);
          specular_reflectance += mf.refl;
          if (mf.has_trans) {
            update_trans(
                mf.trans, mf.trans_w,
                m.thin_walled
                    ? -si.to_world(si.wi)
                    : si.to_world(s.frames[leaf.frame].to_world(mf.wt_local)));
          }
          update_rough(mf.spec_w, mf.roughness);
        } else if (leaf.hair_idx >= 0) {
          const auto& hp = s.hair[leaf.hair_idx];
          specular_reflectance +=
              s.leaf_weight[k] * (hp.ap[0] + hp.ap[2] + 0.5f * hp.ap[3]);
          UnpolarizedSpectrum trans =
              s.leaf_weight[k] * (hp.ap[1] + 0.5f * hp.ap[3]);
          update_trans(trans, SelectionWeight(trans), -si.to_world(si.wi));
          update_rough(
              SelectionWeight(contrib),
              dr::clip(dr::max(dr::head<2>(ReadVal(c.hair.roughness_r, s))),
                       1e-3f, 1.0f));
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
      if (unique_frames_.size() > 1) {
        Float best_w(0.0f);
        for (size_t k = 0; k < leaves_.size(); ++k) {
          Float sw = SelectionWeight(s.leaf_weight[k]);
          f = dr::select(sw > best_w, s.frames[leaves_[k].frame], f);
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
        << "  closures = " << closures_.size() << ",\n"
        << "  opacity_root = "
        << (opacity_root_ == kInvalidValue ? -1
                                           : static_cast<int>(opacity_root_))
        << ",\n"
        << eval_.ToString() << "]";
    return oss.str();
  }

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb) override {
    if (cb.role != drjit::TraverseRole::Freeze &&
        cb.role != drjit::TraverseRole::Children) {
      return;
    }
    Base::traverse_cb(payload, cb);
    eval_.TraverseCb(payload, cb);
    if (cb.state) {
      size_t n = closures_.size();
      drjit::detail::traverse_state(payload, cb, &n, sizeof(n), "closures");
      if (n > 0) {
        drjit::detail::traverse_state(payload, cb, closures_.data(),
                                      n * sizeof(closures_[0]), "closures");
      }
      DR_TRAVERSE_STATE(opacity_root_)
    }
  }

  MI_DECLARE_CLASS(MtlxBsdf)

 private:
  struct FrameKey {
    ValueId normal;
    ValueId tangent;
    bool need_tangent;
    bool is_curve;
  };

  struct Leaf {
    size_t node, frame;
    int spec[4] = {-1, -1, -1, -1};
    int mf_idx = -1;
    int hair_idx = -1;
  };

  using SpecVec = absl::InlinedVector<UnpolarizedSpectrum, 8>;

  // Per-query material state shared by all BSDF entry points.
  struct State {
    EvalState eval;
    // Deduplicated local shading frames and incident directions (relative to
    // `si.sh_frame`).
    absl::InlinedVector<Frame3f, 4> frames;
    absl::InlinedVector<Vector3f, 4> wi_local;
    // Deduplicated upsampled spectra, per-leaf combined linear weights
    // (combinators * leaf weight), optional cached transmissive layer
    // directional albedos (when layered), and detached sampling probabilities.
    SpecVec spectra;
    SpecVec leaf_weight;
    SpecVec layer_albedo;
    absl::InlinedVector<detail::MicrofacetParams<Float, Spectrum>, 2>
        microfacet;
    absl::InlinedVector<detail::ChiangHairParams<Float, UnpolarizedSpectrum>, 1>
        hair;
    absl::InlinedVector<Float, 8> leaf_prob;
    Vector3f n_geom;
    Wavelength wavelengths;
    Float opacity;
    Mask surf_active;
    bool has_surface = false;
  };

  const MtlxValueProgram& Values() const { return eval_.Program(); }
  Val ReadVal(ValueId id, const State& s) const {
    return eval_.ReadVal(id, s.eval);
  }
  static Vector3f Xyz(const Val& v) { return dr::head<3>(v); }
  static uint16_t ScatterMode(const ClosureNode& c) {
    return c.microfacet.mode & kScatterModeMask;
  }

  bool IsAnisotropicMicrofacet(const ClosureNode& c) const {
    return !Values().IsValueScalar(c.microfacet.roughness) ||
           c.microfacet.tangent != kInvalidValue;
  }

  FrameKey LeafFrameKey(const ClosureNode& c) const {
    switch (c.op) {
      case ClosureOp::OrenNayarDiffuse:
      case ClosureOp::BurleyDiffuse:
      case ClosureOp::Sheen:
      case ClosureOp::Translucent:
        return {c.diffuse.normal, kInvalidValue, false, false};
      case ClosureOp::Dielectric:
      case ClosureOp::Conductor:
      case ClosureOp::GeneralizedSchlick:
        return {c.microfacet.normal, c.microfacet.tangent,
                IsAnisotropicMicrofacet(c), false};
      case ClosureOp::ChiangHair:
        return {c.hair.normal, c.hair.curve_direction, true,
                c.hair.curve_direction == kInvalidValue};
      default:
        return {kInvalidValue, kInvalidValue, false, false};
    }
  }

  static uint32_t LeafType(const ClosureNode& c) {
    switch (c.op) {
      case ClosureOp::Translucent:
        return +BSDFFlags::DiffuseTransmission;
      case ClosureOp::Conductor:
      case ClosureOp::Dielectric:
      case ClosureOp::GeneralizedSchlick: {
        int sm = ScatterMode(c);
        return (sm != kScatterT ? +BSDFFlags::GlossyReflection : 0u) |
               (sm != kScatterR ? +BSDFFlags::GlossyTransmission : 0u);
      }
      case ClosureOp::ChiangHair:
        return +BSDFFlags::GlossyReflection | +BSDFFlags::GlossyTransmission;
      default:
        return +BSDFFlags::DiffuseReflection;
    }
  }

  static bool LeafEnabled(const BSDFContext& ctx, const ClosureNode& c) {
    return (LeafType(c) & ctx.type_mask) != 0u;
  }

  detail::MicrofacetParams<Float, Spectrum> BuildMicrofacetParams(
      const BSDFContext& ctx, const ClosureNode& c, const Leaf& leaf,
      const State& s) const {
    using Distr = MicrofacetDistribution<Float, Color3f>;
    detail::MicrofacetParams<Float, Spectrum> m;
    int sm = ScatterMode(c);
    m.kind = c.op == ClosureOp::Dielectric ? detail::MicrofacetKind::Dielectric
             : c.op == ClosureOp::Conductor
                 ? detail::MicrofacetKind::Conductor
                 : detail::MicrofacetKind::GeneralizedSchlick;
    m.refl_on = sm != kScatterT && ctx.is_enabled(BSDFFlags::GlossyReflection);
    m.trans_on =
        sm != kScatterR && ctx.is_enabled(BSDFFlags::GlossyTransmission);
    m.thin_walled = (c.microfacet.mode & kScatterThinWalled) != 0;
    m.two_sided = (c.microfacet.mode & kScatterTwoSided) != 0;
    m.retroreflective = (c.microfacet.mode & kScatterRetroreflective) != 0;
    m.trans_only = sm == kScatterT;
    m.refl_only = sm == kScatterR;
    m.alpha =
        dr::clip(dr::head<2>(ReadVal(c.microfacet.roughness, s)), 1e-3f, 1.0f);
    m.distr = Values().IsValueScalar(c.microfacet.roughness)
                  ? Distr(MicrofacetType::GGX, m.alpha.x())
                  : Distr(MicrofacetType::GGX, m.alpha.x(), m.alpha.y());
    if (c.op == ClosureOp::Dielectric) {
      m.ior = detail::SafeDielectricIor(ReadVal(c.microfacet.p[1], s).x(),
                                        !m.thin_walled && !m.refl_only);
      m.spec0 = s.spectra[leaf.spec[0]];
      if (leaf.spec[1] >= 0) {
        m.has_trans_tint = true;
        m.spec1 = s.spectra[leaf.spec[1]];
      }
    } else if (c.op == ClosureOp::Conductor) {
      m.spec0 = s.spectra[leaf.spec[0]];
      m.spec1 = s.spectra[leaf.spec[1]];
    } else {
      m.spec0 = s.spectra[leaf.spec[0]];
      m.spec1 = s.spectra[leaf.spec[1]];
      m.spec2 = s.spectra[leaf.spec[2]];
      m.exponent = ReadVal(c.microfacet.p[3], s).x();
      if (detail::MicrofacetNeedIorFresnel(m)) {
        ValueId id = c.microfacet.p[0];
        Val v = ReadVal(id, s);
        m.ior =
            detail::F0ToIor(Values().IsValueScalar(id)
                                ? v.x()
                                : dr::mean(Values().IsValueClipped01(id)
                                               ? Xyz(v)
                                               : dr::clip(Xyz(v), 0.0f, 1.0f)));
      }
    }
    if (c.microfacet.tf_thickness != kInvalidValue) {
      m.has_thinfilm = true;
      m.tf_thickness = ReadVal(c.microfacet.tf_thickness, s).x();
      m.tf_ior = ReadVal(c.microfacet.tf_ior, s).x();
    }
    m.wavelengths = s.wavelengths;
    return m;
  }

  Float SheenDirAlbedo(const ClosureNode& c, const Vector3f& wi_l,
                       const State& s) const {
    return detail::SheenDirAlbedo(dr::abs(Frame3f::cos_theta(wi_l)),
                                  ReadVal(c.diffuse.roughness, s).x(),
                                  c.diffuse.mode == kSheenZeltner);
  }

  UnpolarizedSpectrum LeafAlbedo(size_t k, const State& s) const {
    const Leaf& leaf = leaves_[k];
    const ClosureNode& c = closures_[leaf.node];
    bool cached_tp = (c.flags & kClosureNeedsTp) != 0;
    if (leaf.mf_idx >= 0) {
      const auto& m = s.microfacet[leaf.mf_idx];
      UnpolarizedSpectrum a =
          !m.refl_only
              ? UnpolarizedSpectrum(1.0f)
              : (cached_tp
                     ? s.layer_albedo[k]
                     : detail::MicrofacetReflAlbedo(m, s.wi_local[leaf.frame]));
      if (m.kind == detail::MicrofacetKind::Dielectric) {
        return (m.has_trans_tint ? dr::maximum(m.spec0, m.spec1) : m.spec0) * a;
      }
      return a;
    }
    if (c.op == ClosureOp::Sheen) {
      return s.spectra[leaf.spec[0]] *
             (cached_tp ? s.layer_albedo[k]
                        : UnpolarizedSpectrum(
                              SheenDirAlbedo(c, s.wi_local[leaf.frame], s)));
    }
    if (leaf.hair_idx >= 0) {
      const auto& hp = s.hair[leaf.hair_idx];
      return hp.ap[0] + hp.ap[1] + hp.ap[2] + hp.ap[3];
    }
    return s.spectra[leaf.spec[0]];
  }

  // Flattens the closure DAG into per-leaf effective linear weights
  // (combinators * leaf weight), computing layer throughputs and caching
  // transmissive layer albedos only for closures marked `kClosureNeedsTp`.
  void FlattenClosures(State& s) const {
    size_t n = closures_.size();

    absl::InlinedVector<UnpolarizedSpectrum, 16> tp;
    if (has_layers_) {
      tp.resize(n - 1);
      s.layer_albedo.resize(leaves_.size(), 0.0f);
      size_t leaf_k = 0;
      for (size_t i = 0; i + 1 < n; ++i) {
        const ClosureNode& c = closures_[i];
        if (!(c.flags & kClosureNeedsTp)) {
          if (IsLeaf(c.op)) ++leaf_k;
          continue;
        }
        if (c.op == ClosureOp::Null) {
          tp[i] = 1.0f;
        } else if (c.op == ClosureOp::OpaqueBlocker) {
          tp[i] = 0.0f;
        } else if (IsLeaf(c.op)) {
          size_t k = leaf_k++;
          if (LeafHasLayerThroughput(c)) {
            const Leaf& leaf = leaves_[k];
            UnpolarizedSpectrum albedo =
                c.op == ClosureOp::Sheen
                    ? UnpolarizedSpectrum(
                          SheenDirAlbedo(c, s.wi_local[leaf.frame], s))
                    : detail::MicrofacetReflAlbedo(s.microfacet[leaf.mf_idx],
                                                   s.wi_local[leaf.frame]);
            s.layer_albedo[k] = albedo;
            Float w_leaf = dr::maximum(ReadVal(LeafWeightId(c), s).x(), 0.0f);
            tp[i] = dr::maximum(0.0f, 1.0f - w_leaf * albedo);
          } else {
            tp[i] = 0.0f;
          }
        } else {
          int c0 = c.comb.first, c1 = c.comb.second;
          if (c.op == ClosureOp::Layer) {
            tp[i] = tp[c0] * tp[c1];
          } else if (c.op == ClosureOp::Mix) {
            tp[i] = dr::lerp(tp[c1], tp[c0], s.spectra[comb_spec_[i]]);
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
      const ClosureNode& c = closures_[i];
      if (!IsCombinator(c.op)) continue;
      int c0 = c.comb.first, c1 = c.comb.second;
      if (c.op == ClosureOp::Layer) {
        w[c0] += w[i];
        if (closures_[c0].flags & kClosureNeedsTp) {
          w[c1] = dr::fmadd(w[i], tp[c0], w[c1]);
        }
      } else if (c.op == ClosureOp::Mix) {
        const UnpolarizedSpectrum& m = s.spectra[comb_spec_[i]];
        w[c0] = dr::fmadd(w[i], m, w[c0]);
        w[c1] = dr::fmadd(w[i], 1.0f - m, w[c1]);
      } else if (c.op == ClosureOp::Add) {
        w[c0] += w[i];
        w[c1] += w[i];
      } else {  // Multiply
        w[c0] = dr::fmadd(w[i], s.spectra[comb_spec_[i]], w[c0]);
      }
    }

    s.leaf_weight.resize(leaves_.size());
    for (size_t k = 0; k < leaves_.size(); ++k) {
      const ClosureNode& c = closures_[leaves_[k].node];
      ValueId w_id = LeafWeightId(c);
      s.leaf_weight[k] =
          w_id != kInvalidValue
              ? w[leaves_[k].node] * dr::maximum(ReadVal(w_id, s).x(), 0.0f)
              : w[leaves_[k].node];
    }
  }

  State EvaluateState(const BSDFContext& ctx, const SurfaceInteraction3f& si,
                      Mask active) const {
    State s;
    if (opacity_root_ == kInvalidValue) {
      s.opacity = Float(1.0f);
    } else {
      s.opacity =
          ClipOpacity(eval_.EvalOpacity(opacity_root_, si, active, s.eval));
    }
    s.surf_active =
        (opacity_root_ == kInvalidValue || dr::grad_enabled(s.opacity))
            ? active
            : (active && (s.opacity > 0.0f));
    s.has_surface = HasSurfaceEnabled(ctx) && dr::any_or<true>(s.surf_active);
    if (!s.has_surface) return s;

    eval_.EvalSurface(si, active, s.eval);

    s.n_geom =
        detail::SafeNormalize(si.to_local(si.n), Vector3f(0.0f, 0.0f, 1.0f),
                              detail::kMinLen2<ScalarFloat>);
    s.wavelengths = si.wavelengths;

    s.frames.resize(unique_frames_.size());
    s.wi_local.resize(unique_frames_.size());
    for (size_t f = 0; f < unique_frames_.size(); ++f) {
      const FrameKey& key = unique_frames_[f];
      Vector3f nl(0.0f, 0.0f, 1.0f);
      if (key.normal != kInvalidValue) {
        Vector3f nw = Xyz(ReadVal(key.normal, s));
        if (!Values().IsValueUnitNormal(key.normal)) {
          nw = detail::SafeNormalize(nw, Vector3f(si.sh_frame.n),
                                     detail::kMinLen2<ScalarFloat>);
        }
        nl = si.to_local(nw);
      }
      if (!key.need_tangent) {
        s.frames[f] = Frame3f(detail::ClampShadingNormal(si, s.n_geom, nl));
      } else {
        Vector3f t_world;
        if (key.tangent != kInvalidValue) {
          t_world = Xyz(ReadVal(key.tangent, s));
          if (!Values().IsValueUnitNormal(key.tangent)) {
            t_world = detail::SafeNormalize(t_world, si.sh_frame.s,
                                            detail::kMinLen2<ScalarFloat>);
          }
        } else {
          t_world = key.is_curve ? detail::GeomCurveTangentWorld(si)
                                 : detail::GeomTangentWorld(si);
        }
        s.frames[f] = detail::BuildShadingFrame(si, s.n_geom, nl, t_world);
      }
      s.wi_local[f] = s.frames[f].to_local(si.wi);
    }

    s.spectra.resize(eval_.NumSpectra());
    for (size_t j = 0; j < s.spectra.size(); ++j) {
      s.spectra[j] = eval_.EvalSpectrum(j, si.wavelengths, s.eval);
    }
    if (num_microfacet_ > 0) {
      s.microfacet.resize(num_microfacet_);
      for (const Leaf& leaf : leaves_) {
        if (leaf.mf_idx < 0) continue;
        const ClosureNode& c = closures_[leaf.node];
        if (!LeafEnabled(ctx, c) && !(c.flags & kClosureNeedsTp)) continue;
        s.microfacet[leaf.mf_idx] = BuildMicrofacetParams(ctx, c, leaf, s);
      }
    }
    if (num_hair_ > 0 &&
        (ctx.type_mask & (+BSDFFlags::GlossyReflection |
                          +BSDFFlags::GlossyTransmission)) != 0u) {
      s.hair.resize(num_hair_);
      auto hair_r = [&](ValueId id) {
        return dr::maximum(dr::head<2>(ReadVal(id, s)), 1e-3f);
      };
      for (const Leaf& leaf : leaves_) {
        if (leaf.hair_idx < 0) continue;
        const ClosureNode& c = closures_[leaf.node];
        s.hair[leaf.hair_idx] = detail::BuildChiangHairParams<Float, Spectrum>(
            s.wi_local[leaf.frame], s.spectra[leaf.spec[0]],
            s.spectra[leaf.spec[1]], s.spectra[leaf.spec[2]],
            hair_r(c.hair.roughness_r), hair_r(c.hair.roughness_tt),
            hair_r(c.hair.roughness_trt), ReadVal(c.hair.ior, s).x(),
            s.spectra[leaf.spec[3]], ReadVal(c.hair.cuticle_angle, s).x());
      }
    }
    FlattenClosures(s);
    return s;
  }

  void ComputeLeafProbs(const BSDFContext& ctx, State& s) const {
    size_t num_leaves = leaves_.size();
    s.leaf_prob.resize(num_leaves);
    if (num_leaves == 1) {
      s.leaf_prob[0] = 1.0f;
      return;
    }
    Float w_sum(0.0f);
    for (size_t k = 0; k < num_leaves; ++k) {
      const ClosureNode& c = closures_[leaves_[k].node];
      if (!LeafEnabled(ctx, c)) {
        s.leaf_prob[k] = 0.0f;
      } else {
        UnpolarizedSpectrum alb = LeafAlbedo(k, s);
        if (c.op == ClosureOp::Sheen && c.diffuse.mode != kSheenZeltner) {
          alb = dr::maximum(alb, s.spectra[leaves_[k].spec[0]] * 1e-3f);
        }
        Float sw = SelectionWeight(s.leaf_weight[k] * alb);
        s.leaf_prob[k] = sw;
        w_sum += sw;
      }
    }
    Mask ok = w_sum > 0.0f;
    Float inv_sum = dr::select(ok, dr::rcp(dr::select(ok, w_sum, 1.0f)), 0.0f);
    for (Float& p : s.leaf_prob) p = dr::detach(p * inv_sum);
  }

  Float ClipOpacity(const Val& op) const {
    return Values().IsValueClipped01(opacity_root_)
               ? op.x()
               : dr::clip(op.x(), 0.0f, 1.0f);
  }

  Float ReadClipped01(ValueId id, const State& s) const {
    Float v = ReadVal(id, s).x();
    return Values().IsValueClipped01(id) ? v : dr::clip(v, 0.0f, 1.0f);
  }

  static Float SelectionWeight(const UnpolarizedSpectrum& c) {
    return detail::SelectionWeight<Spectrum>(c);
  }

  bool HasCutout(const BSDFContext& ctx) const {
    return null_component_ >= 0 &&
           ctx.is_enabled(BSDFFlags::Null,
                          static_cast<uint32_t>(null_component_));
  }

  bool HasSurfaceEnabled(const BSDFContext& ctx) const {
    return (ctx.component == uint32_t(-1) || ctx.component == 0) &&
           (surface_types_ & ctx.type_mask) != 0u;
  }

  std::pair<Spectrum, Float> EvalPdfImpl(const BSDFContext& ctx, const State& s,
                                         const Vector3f& wo) const {
    if (!s.has_surface) return {0.0f, 0.0f};
    auto [f, pdf] = EvalAndPdfLeaves(ctx, s, wo);
    return {depolarizer<Spectrum>(f) & s.surf_active,
            dr::detach(dr::select(
                s.surf_active, HasCutout(ctx) ? pdf * s.opacity : pdf, 0.0f))};
  }

  detail::LobeSample<Float> SampleSurfaceLobe(const BSDFContext& ctx,
                                              const State& s, Float sample1,
                                              const Point2f& sample2,
                                              Mask active) const {
    detail::LobeSample<Float> out;
    if (!s.has_surface) return out;
    Mask remaining = active;
    Float total_p(0.0f);
    for (const Float& p : s.leaf_prob) total_p += p;
    Float xi = dr::clip(sample1, 0.0f, dr::OneMinusEpsilon<Float>) * total_p;
    Float cdf(0.0f);
    for (size_t k = 0; k < leaves_.size(); ++k) {
      if (!LeafEnabled(ctx, closures_[leaves_[k].node])) continue;
      Float cdf_prev = cdf;
      cdf += s.leaf_prob[k];
      Mask pick = remaining && (xi < cdf);
      remaining = remaining && !pick;
      if (dr::none_or<false>(pick)) continue;
      Float leaf_u1 =
          leaves_.size() > 1
              ? (xi - cdf_prev) / dr::select(pick, s.leaf_prob[k], 1.0f)
              : xi;
      detail::LobeSample<Float> ls = SampleLeaf(ctx, k, leaf_u1, sample2, s);
      Mask hit = pick && ls.valid;
      out.wo = dr::select(hit, ls.wo, out.wo);
      out.sampled_type = dr::select(hit, ls.sampled_type, out.sampled_type);
      out.eta = dr::select(hit, ls.eta, out.eta);
      out.valid = out.valid || hit;
    }
    return out;
  }

  std::pair<BSDFSample3f, Spectrum> SampleImpl(const BSDFContext& ctx,
                                               const SurfaceInteraction3f& si,
                                               const State& s, Float sample1,
                                               const Point2f& sample2,
                                               Mask active) const {
    bool cutout = HasCutout(ctx);
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

    detail::LobeSample<Float> lobe =
        SampleSurfaceLobe(ctx, s, sample1, sample2, surf_mask);

    // Detached one-sample MIS over the lobe mixture: weight = f_tot / pdf_tot.
    Vector3f wo = dr::detach(lobe.wo);
    auto [f_tot, pdf_tot] =
        (s.has_surface && dr::any_or<true>(lobe.valid))
            ? EvalAndPdfLeaves(ctx, s, wo)
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
        cutout ? dr::select(pass_through,
                            UInt32(static_cast<uint32_t>(null_component_)),
                            UInt32(0))
               : UInt32(0);

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

  // Samples a direction from leaf `k` (with `wo` in `si.sh_frame` coordinates).
  detail::LobeSample<Float> SampleLeaf(const BSDFContext& ctx, size_t k,
                                       const Float& sample1,
                                       const Point2f& sample2,
                                       const State& s) const {
    const Leaf& leaf = leaves_[k];
    const ClosureNode& c = closures_[leaf.node];
    const Vector3f& wi_l = s.wi_local[leaf.frame];
    detail::LobeSample<Float> ls;
    if (leaf.mf_idx >= 0) {
      ls = detail::MicrofacetSample(s.microfacet[leaf.mf_idx], wi_l, sample1,
                                    sample2);
    } else if (leaf.hair_idx >= 0) {
      ls = detail::ChiangHairSample(
          s.hair[leaf.hair_idx], wi_l,
          ctx.is_enabled(BSDFFlags::GlossyReflection),
          ctx.is_enabled(BSDFFlags::GlossyTransmission), sample1, sample2);
    } else {
      ls.sampled_type = UInt32(LeafType(c));
      std::tie(ls.wo, ls.valid) =
          c.op == ClosureOp::Sheen
              ? detail::SheenSample(wi_l, ReadVal(c.diffuse.roughness, s).x(),
                                    c.diffuse.mode == kSheenZeltner, sample2)
              : detail::CosineHemisphereSample(
                    wi_l, sample2,
                    /*same_side=*/c.op != ClosureOp::Translucent);
    }
    Float cos_o = Frame3f::cos_theta(ls.wo);
    ls.wo = s.frames[leaf.frame].to_world(ls.wo);
    ls.valid = ls.valid && (dr::dot(s.n_geom, ls.wo) * cos_o > 0.0f);
    return ls;
  }

  std::pair<UnpolarizedSpectrum, Float> EvalAndPdfLeaf(const BSDFContext& ctx,
                                                       size_t k,
                                                       const Vector3f& wo,
                                                       const State& s) const {
    const Leaf& leaf = leaves_[k];
    const ClosureNode& c = closures_[leaf.node];
    const Vector3f& wi_l = s.wi_local[leaf.frame];
    Vector3f wo_l = s.frames[leaf.frame].to_local(wo);
    Mask same_side = dr::dot(s.n_geom, wo) * Frame3f::cos_theta(wo_l) > 0.0f;
    if (dr::none_or<false>(same_side)) return {0.0f, 0.0f};
    UnpolarizedSpectrum f(0.0f);
    Float pdf(0.0f);
    if (leaf.mf_idx >= 0) {
      std::tie(f, pdf) = detail::MicrofacetEvalPdf(s.microfacet[leaf.mf_idx],
                                                   wi_l, wo_l, ctx.mode);
    } else if (leaf.hair_idx >= 0) {
      std::tie(f, pdf) = detail::ChiangHairEvalPdf(
          s.hair[leaf.hair_idx], wi_l, wo_l,
          ctx.is_enabled(BSDFFlags::GlossyReflection),
          ctx.is_enabled(BSDFFlags::GlossyTransmission));
    } else if (c.op == ClosureOp::Translucent) {
      std::tie(f, pdf) =
          detail::TranslucentEvalPdf(wi_l, wo_l, s.spectra[leaf.spec[0]]);
    } else if (c.op == ClosureOp::Sheen) {
      std::tie(f, pdf) = detail::SheenEvalPdf(
          wi_l, wo_l, s.spectra[leaf.spec[0]],
          ReadVal(c.diffuse.roughness, s).x(), c.diffuse.mode == kSheenZeltner);
    } else if (c.op == ClosureOp::BurleyDiffuse) {
      std::tie(f, pdf) =
          detail::BurleyDiffuseEvalPdf(wi_l, wo_l, s.spectra[leaf.spec[0]],
                                       ReadClipped01(c.diffuse.roughness, s));
    } else {
      std::tie(f, pdf) = detail::OrenNayarEvalPdf(
          wi_l, wo_l, s.spectra[leaf.spec[0]],
          ReadClipped01(c.diffuse.roughness, s), c.diffuse.mode != 0);
    }
    return {dr::select(same_side, f, 0.0f), dr::select(same_side, pdf, 0.0f)};
  }

  std::pair<UnpolarizedSpectrum, Float> EvalAndPdfLeaves(
      const BSDFContext& ctx, const State& s, const Vector3f& wo) const {
    UnpolarizedSpectrum f(0.0f);
    Float pdf(0.0f);
    for (size_t k = 0; k < leaves_.size(); ++k) {
      if (!LeafEnabled(ctx, closures_[leaves_[k].node])) continue;
      auto [f_k, pdf_k] = EvalAndPdfLeaf(ctx, k, wo, s);
      f = dr::fmadd(s.leaf_weight[k], f_k, f);
      pdf = dr::fmadd(s.leaf_prob[k], pdf_k, pdf);
    }
    return {f, pdf};
  }

  MtlxEvaluator<Float, Spectrum> eval_;
  std::vector<ClosureNode> closures_;
  ValueId opacity_root_ = kInvalidValue;
  bool has_layers_ = false;
  int num_microfacet_ = 0;
  int num_hair_ = 0;
  uint32_t surface_types_ = 0u;
  int null_component_ = -1;
  // Per-closure combinator spectral slot index (-1 for non-Mix/Multiply).
  std::vector<int> comb_spec_;
  // Leaf closures paired with their deduplicated frame and spectral slots.
  std::vector<Leaf> leaves_;
  // Deduplicated shading frame descriptors.
  std::vector<FrameKey> unique_frames_;
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

}  // namespace mitsuba::mtlx
