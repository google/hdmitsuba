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

#include "hdmitsuba/materialx/evaluator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

#include "hdmitsuba/materialx/math.h"
#include "hdmitsuba/materialx/noise.h"

namespace mitsuba::mtlx {
namespace {

template <int Dim, typename Out, typename Float>
Vector<Float, 4> EvalNoiseOp(const Node& node, const Vector<Float, 4>* a) {
  Out res;
  if (node.op == Op::CellNoise) {
    res = detail::MxCellNoise<Dim, Out, Float>(dr::head<Dim>(a[0]));
  } else if (node.op == Op::Noise) {
    res = detail::MxPerlinNoise<Dim, Out, Float>(dr::head<Dim>(a[0]));
  } else if (node.op == Op::Worley) {
    if (WorleyDistanceOnly(node.imm)) {
      res = detail::MxWorleyNoise<Dim, Out, true, Float>(
          dr::head<Dim>(a[0]), a[1].x(), dr::mask_t<Float>(false));
    } else {
      res = detail::MxWorleyNoise<Dim, Out, false, Float>(
          dr::head<Dim>(a[0]), a[1].x(), dr::round(a[2].x()) == 1.0f);
    }
  } else {
    uint16_t oct = FractalOctaves(node.imm);
    if (oct == kFractalDynamicOctaves) {
      res = detail::MxFractalNoise<Dim, Out, Float>(
          dr::head<Dim>(a[0]), a[3].x(), a[1].x(), a[2].x(),
          kMaxFractalOctaves);
    } else {
      res = detail::MxFractalNoise<Dim, Out, Float>(
          dr::head<Dim>(a[0]), int(oct), a[1].x(), a[2].x());
    }
  }
  if constexpr (std::is_same_v<Out, Float>) {
    return Vector<Float, 4>(res);
  } else {
    return Vector<Float, 4>(res.x(), res.y(), res.z(), 0.0f);
  }
}

template <typename Color3f>
Color3f ClampColor(const Color3f& c, bool unbounded, bool clipped01) {
  if (unbounded) return dr::maximum(c, 0.0f);
  return clipped01 ? c : dr::clip(c, 0.0f, 1.0f);
}

/**!

.. _texture-materialx_expr:

MaterialX Expression (:monosp:`materialx_expr`)
-------------------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxValueProgram`` SSA IR and constant/opaque
     tables.

 * - in0, in1, ...
   - |texture|
   - Nested input textures referenced by ``Op::SampleInput`` and
     ``Op::SampleInputUV``.
   - |exposed|, |differentiable|

 * - param0, param1, ...
   - |float|
   - Differentiable opaque scalar parameters exposed through traversal.
   - |exposed|, |differentiable|

Evaluates a compiled MaterialX expression DAG over surface interactions,
supporting procedural noises, 2D texture lookups, geometric intrinsics,
and mesh attributes.

*/
template <typename Float, typename Spectrum>
class MtlxExprTexture final : public Texture<Float, Spectrum> {
 public:
  MI_IMPORT_TYPES(Texture)

  explicit MtlxExprTexture(const Properties& props) : Texture(props) {
    eval_.Compile(props);
    if constexpr (is_spectral_v<Spectrum>) {
      if (eval_.Program().result_root != kInvalidValue) {
        eval_.AddSpectrum(eval_.Program().result_root, /*unbounded=*/true);
      }
    }
    eval_.FinalizeUniforms();
  }

  void traverse(TraversalCallback* cb) override { eval_.Traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    eval_.ParametersChanged();
  }

  UnpolarizedSpectrum eval(const SurfaceInteraction3f& si,
                           Mask active) const override {
    if constexpr (is_monochromatic_v<Spectrum>) {
      return eval_1(si, active);
    } else if constexpr (is_rgb_v<Spectrum>) {
      return eval_3(si, active);
    } else {
      MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
      if (eval_.Program().result_root == kInvalidValue) return 0.0f;
      typename MtlxEvaluator<Float, Spectrum>::EvalState state;
      eval_.EvalSurface(si, active, state);
      return eval_.EvalSpectrum(0, si.wavelengths, state);
    }
  }

  std::pair<Wavelength, UnpolarizedSpectrum> sample_spectrum(
      const SurfaceInteraction3f& si, [[maybe_unused]] const Wavelength& sample,
      Mask active) const override {
    MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);
    if constexpr (is_spectral_v<Spectrum>) {
      SurfaceInteraction3f si2(si);
      si2.wavelengths = MI_CIE_MIN + (MI_CIE_MAX - MI_CIE_MIN) * sample;
      return {si2.wavelengths,
              eval(si2, active) * ScalarFloat(MI_CIE_MAX - MI_CIE_MIN)};
    } else {
      return {dr::empty<Wavelength>(), eval(si, active)};
    }
  }

  Wavelength pdf_spectrum(const SurfaceInteraction3f&, Mask) const override {
    if constexpr (is_spectral_v<Spectrum>) {
      return dr::rcp(ScalarFloat(MI_CIE_MAX - MI_CIE_MIN));
    } else {
      return dr::empty<Wavelength>();
    }
  }

  Float eval_1(const SurfaceInteraction3f& si, Mask active) const override {
    return eval_.Eval1(si, active);
  }

  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const override {
    return eval_.Eval3(si, active);
  }

  bool is_spatially_varying() const override {
    return eval_.IsSpatiallyVarying();
  }

  std::string to_string() const override {
    return "MtlxExprTexture[\n" + eval_.ToString() + "]";
  }

  MI_DECLARE_CLASS(MtlxExprTexture)

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb) override {
    if (cb.role != drjit::TraverseRole::Freeze &&
        cb.role != drjit::TraverseRole::Children) {
      return;
    }
    Texture::traverse_cb(payload, cb);
    eval_.TraverseCb(payload, cb);
  }

 private:
  MtlxEvaluator<Float, Spectrum> eval_;
};

}  // namespace

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::Compile(const Properties& props) {
  Compile(props, props.get_any<MtlxValueProgram>("program"));
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::Compile(const Properties& props,
                                             MtlxValueProgram prog) {
  inputs_.clear();
  for (size_t i = 0;; ++i) {
    std::string name = "in" + std::to_string(i);
    if (!props.has_property(name)) break;
    inputs_.push_back(props.get_texture<Texture>(name));
  }

  prog_ = std::move(prog);
  spec_slots_.clear();
  uniform_specs_.clear();
  opaques_.clear();
  opaques_.reserve(prog_.opaques.size());
  for (double v : prog_.opaques) {
    opaques_.push_back(dr::opaque<Float>(static_cast<ScalarFloat>(v)));
  }
}

template <typename Float, typename Spectrum>
int MtlxEvaluator<Float, Spectrum>::AddSpectrum(ValueId id, bool unbounded) {
  for (size_t i = 0; i < spec_slots_.size(); ++i) {
    if (spec_slots_[i].value == id && spec_slots_[i].unbounded == unbounded) {
      return static_cast<int>(i);
    }
  }
  int uniform_spec_idx = -1;
  if constexpr (is_spectral_v<Spectrum>) {
    if (!prog_.IsValueScalar(id)) {
      srgb_.Load();
      if (id < prog_.uniform_end) {
        uniform_spec_idx = static_cast<int>(uniform_specs_.size());
        uniform_specs_.emplace_back();
      }
    }
  }
  spec_slots_.push_back({id, unbounded, uniform_spec_idx});
  return static_cast<int>(spec_slots_.size() - 1);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::FinalizeUniforms() {
  UpdateUniformState();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::HasOpaqueGrad() const {
  if constexpr (dr::is_diff_v<Float>) {
    for (const Float& p : opaques_) {
      if (dr::grad_enabled(p)) return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::UpdateUniformState() {
  // Precompute finite-difference UV steps per `HeightToNormal` region.
  region_eps_.resize(prog_.regions.size());
  for (size_t r_idx = 0; r_idx < prog_.regions.size(); ++r_idx) {
    const Region& r = prog_.regions[r_idx];
    ScalarVector2i max_res(1);
    for (uint32_t i = r.begin; i < r.end; ++i) {
      const Node& n = prog_.nodes[i];
      if (n.op == Op::SampleInput || n.op == Op::SampleInputUV) {
        max_res = dr::maximum(max_res, inputs_[n.imm]->resolution());
      }
    }
    // Use an exact multiple of 1/256 texel (`0.125 / res`) when sampling
    // bitmaps so CUDA's 8-bit fixed-point `tex2D` sub-texel interpolation
    // introduces zero quantization error in central differences.
    region_eps_[r_idx] =
        dr::select(max_res > 1,
                   dr::maximum(ScalarFloat(0.125f) / ScalarPoint2f(max_res),
                               ScalarFloat(1e-5f)),
                   ScalarPoint2f(1e-4f));
  }

  uniform_scalars_.clear();
  if (prog_.uniform_scalar_count == 0 && uniform_specs_.empty()) return;

  absl::InlinedVector<Val, 32> uvals;
  EvalUniformNodes(uvals);

  bool scheduled = false;
  if (prog_.uniform_scalar_count > 0) {
    uniform_scalars_.reserve(prog_.uniform_scalar_count);
    for (uint32_t u = 0; u < prog_.uniform_end; ++u) {
      if (prog_.uniform_offsets[u] == kNoUniformScalar) continue;
      uint8_t ch = prog_.nodes[u].channels;
      for (uint8_t c = 0; c < ch; ++c) {
        uniform_scalars_.push_back(dr::detach(uvals[u][c]));
      }
    }
    if constexpr (dr::is_jit_v<Float>) {
      for (Float& s : uniform_scalars_) {
        scheduled |= dr::detail::schedule_force(s);
      }
    }
  }
  if constexpr (is_spectral_v<Spectrum>) {
    for (const SpecSlot& slot : spec_slots_) {
      if (slot.uniform_spec_idx < 0) continue;
      Color3f c =
          ClampColor(Color3f(dr::head<3>(uvals[slot.value])), slot.unbounded,
                     prog_.IsValueClipped01(slot.value));
      auto [coeff, scale] = srgb_.FetchCoeffs(c, slot.unbounded);
      UniformSpec& dst = uniform_specs_[slot.uniform_spec_idx];
      dst.coeff = dr::detach(coeff);
      dst.scale = dr::detach(scale);
      if constexpr (dr::is_jit_v<Float>) {
        scheduled |= dr::detail::schedule_force(dst.coeff);
        if (slot.unbounded) scheduled |= dr::detail::schedule_force(dst.scale);
      }
    }
  }
  if constexpr (dr::is_jit_v<Float>) {
    if (scheduled) dr::eval();
  }
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::LoadUniform(ValueId id) const {
  const Node& n = prog_.nodes[id];
  if (n.op == Op::Const) {
    return Val(static_cast<ScalarFloat>(prog_.consts[n.imm]));
  }
  if (n.op == Op::Opaque) return Val(opaques_[n.imm]);
  uint32_t off = prog_.uniform_offsets[id];
  if (n.channels == 1) return Val(uniform_scalars_[off]);
  Val out(0.0f);
  for (uint8_t c = 0; c < n.channels; ++c) out[c] = uniform_scalars_[off + c];
  return out;
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::EvalUniformNodes(
    absl::InlinedVector<Val, 32>& uvals) const {
  uvals.resize(prog_.uniform_end);
  EvalScratch scratch;
  EvalFrame frame{uvals.data(), uvals.data(), scratch};
  SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
  EvalRange(0, prog_.uniform_end, 0, frame, si, Mask(true));
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::EvalUntil(uint32_t end,
                                               const SurfaceInteraction3f& si,
                                               Mask active,
                                               EvalState& state) const {
  if (!state.uniform_ready) {
    state.uniform_ready = true;
    if (prog_.uniform_scalar_count > 0 && HasOpaqueGrad()) {
      EvalUniformNodes(state.uniform_vals);
    }
  }
  uint32_t start = std::max(state.evaluated_end, prog_.uniform_end);
  if (start >= end) return;

  state.varying_vals.resize(end - prog_.uniform_end);
  const Val* uptr =
      state.uniform_vals.empty() ? nullptr : state.uniform_vals.data();
  EvalFrame frame{state.varying_vals.data(), uptr, state.scratch};
  EvalRange(start, end, prog_.uniform_end, frame, si, active);
  state.evaluated_end = end;
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::Traverse(TraversalCallback* cb) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    cb->put("in" + std::to_string(i), inputs_[i], ParamFlags::Differentiable);
  }
  for (size_t i = 0; i < opaques_.size(); ++i) {
    cb->put("param" + std::to_string(i), opaques_[i],
            ParamFlags::Differentiable);
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::ParametersChanged() {
  if constexpr (dr::is_jit_v<Float>) {
    bool scheduled = false;
    for (Float& p : opaques_) {
      if (unlikely(p.size() != 1)) {
        Throw("MtlxEvaluator: parameter must have size 1 (got %zu)!", p.size());
      }
      scheduled |= dr::detail::schedule_force(p);
    }
    if (scheduled) dr::eval();
  }
  UpdateUniformState();
}

template <typename Float, typename Spectrum>
std::string MtlxEvaluator<Float, Spectrum>::ToString() const {
  std::ostringstream oss;
  oss << "  code_size = " << prog_.nodes.size() << ",\n";
  for (size_t i = 0; i < inputs_.size(); ++i) {
    oss << "  in" << i << " = " << string::indent(inputs_[i]) << ",\n";
  }
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::IsSpatiallyVarying() const {
  for (size_t i = prog_.uniform_end; i < prog_.nodes.size(); ++i) {
    const Node& n = prog_.nodes[i];
    if (n.op == Op::SampleInput) {
      if (inputs_[n.imm]->is_spatially_varying()) return true;
    } else if (IsSpatiallyVaryingOp(n.op)) {
      return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::ReadVal(ValueId id,
                                        const EvalState& state) const {
  if (id < prog_.uniform_end) {
    return state.uniform_vals.empty() ? LoadUniform(id)
                                      : state.uniform_vals[id];
  }
  return state.varying_vals[id - prog_.uniform_end];
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::EvalOpacity(ValueId opacity_root,
                                            const SurfaceInteraction3f& si,
                                            Mask active,
                                            EvalState& state) const {
  EvalUntil(prog_.opacity_end, si, active, state);
  return ReadVal(opacity_root, state);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::EvalSurface(const SurfaceInteraction3f& si,
                                                 Mask active,
                                                 EvalState& state) const {
  EvalUntil(prog_.main_end, si, active, state);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::EvalExpr(const SurfaceInteraction3f& si,
                                         Mask active) const {
  if (prog_.result_root == kInvalidValue) return Val(0.0f);
  EvalState state;
  EvalUntil(prog_.main_end, si, active, state);
  return ReadVal(prog_.result_root, state);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::EvalRange(uint32_t begin, uint32_t end,
                                               uint32_t base_offset,
                                               EvalFrame& frame,
                                               const SurfaceInteraction3f& si,
                                               Mask active) const {
  uint32_t uniform_end = prog_.uniform_end;
  auto get = [&](ValueId a) -> Val {
    if (a < uniform_end) {
      return frame.uniform_vals ? frame.uniform_vals[a] : LoadUniform(a);
    }
    return frame.out_vals[a - base_offset];
  };
  auto xyz = [](const Val& v) -> Vector3f { return dr::head<3>(v); };
  auto val3 = [](const auto& v, const Float& w = 0.0f) {
    return Val(v.x(), v.y(), v.z(), w);
  };
  auto eval_input = [&](uint16_t k, bool mono,
                        const SurfaceInteraction3f& si_q) -> Val {
    return mono ? Val(inputs_[k]->eval_1(si_q, active))
                : val3(inputs_[k]->eval_3(si_q, active));
  };

  Val a[8];
  for (uint32_t i = begin; i < end; ++i) {
    const Node& n = prog_.nodes[i];
    auto arg_ids = NodeArgs(n, prog_.extra_args);
    for (size_t j = 0; j < arg_ids.size(); ++j) a[j] = get(arg_ids[j]);

    Val out(0.0f);
    if (detail::EvalPureOp<Float>(n, a, out)) {
      frame.out_vals[i - base_offset] = out;
      continue;
    }

    uint16_t imm = n.imm;
    bool mono = (n.flags & kNodeReadChannelsMask) == 1;
    switch (n.op) {
      case Op::Const:
        out = Val(static_cast<ScalarFloat>(prog_.consts[imm]));
        break;
      case Op::Opaque:
        out = Val(opaques_[imm]);
        break;
      case Op::SampleInput:
        out = eval_input(imm, mono, si);
        break;
      case Op::SampleInputUV:
        if (!frame.scratch.si_uv) {
          frame.scratch.si_uv.emplace(dr::zeros<SurfaceInteraction3f>());
        }
        frame.scratch.si_uv->uv = Point2f(a[0].x(), a[1].x());
        out = eval_input(imm, mono, *frame.scratch.si_uv);
        break;
      case Op::Attr: {
        out = a[0];
        if constexpr (!dr::is_jit_v<Float>) {
          if (!active || !si.shape) break;
        }
        auto& attr_cache = frame.scratch.attr_cache;
        if (attr_cache.empty()) attr_cache.resize(prog_.attr_names.size());
        std::optional<CachedAttr>& cached = attr_cache[imm];
        if (!cached) {
          const std::string& attr = prog_.attr_names[imm];
          Mask has = si.shape->has_attribute(attr, active) & active;
          Val val =
              dr::none_or<false>(has)
                  ? Val(0.0f)
                  : (mono ? Val(si.shape->eval_attribute_1(attr, si, has))
                          : val3(si.shape->eval_attribute_3(attr, si, has)));
          cached.emplace(CachedAttr{has, val});
        }
        if (dr::none_or<false>(cached->has)) break;
        // A fourth channel is that of the default.
        out =
            dr::select(cached->has,
                       mono ? cached->val : val3(cached->val, a[0].w()), a[0]);
        break;
      }

      case Op::GeomUV:
        out = Val(si.uv.x(), 1.0f - si.uv.y(), 0.0f, 0.0f);
        break;
      case Op::GeomPosWorld:
        out = val3(si.p);
        break;
      case Op::GeomNormalWorld:
        out = val3(si.sh_frame.n);
        break;
      case Op::GeomTangentWorld:
        out = val3(detail::GeomTangentWorld(si));
        break;
      case Op::GeomBitangentWorld:
        out =
            val3(detail::GeomBitangentWorld(si, detail::GeomTangentWorld(si)));
        break;
      case Op::GeomIncidentDir:
        out = val3(-si.to_world(si.wi));
        break;

      case Op::NormalMap:
        if (imm == kNormalMapDefaultFrame) {
          Vector3f t = detail::GeomTangentWorld(si);
          out = val3(detail::MxNormalMapFrame(
              xyz(a[0]), Vector2f(dr::head<2>(a[1])),
              Frame3f(t, detail::GeomBitangentWorld(si, t), si.sh_frame.n)));
        } else {
          out = val3(detail::MxNormalMap(xyz(a[0]), Vector2f(dr::head<2>(a[1])),
                                         xyz(a[2]), xyz(a[3]), xyz(a[4]),
                                         Vector3f(si.sh_frame.n)));
        }
        break;

      case Op::HeightToNormal: {
        Float scale = a[0].x();
        const Region& r = prog_.regions[imm];
        ScalarPoint2f eps = region_eps_[imm];
        // Matches MaterialX 1.39 `mx_heighttonormal_vector3` (`1 / 16` Sobel
        // scale factor and cross-product Jacobian over `(u_mtlx, v_mtlx)`).
        ScalarFloat inv_2eps_u = 0.5f / eps.x();
        ScalarFloat inv_2eps_v = 0.5f / eps.y();
        constexpr ScalarFloat kSobelScale = 1.0f / 16.0f;
        bool ad_uniform = (frame.uniform_vals != nullptr);
        using Int32 = dr::int32_array_t<Float>;
        size_t w = dr::width(si, active);
        auto run_h2n = [&](auto zero_acc) {
          using Acc = decltype(zero_acc);
          auto [_, d_u, d_v] = detail::WhileLoopWithInvariants(
              std::make_tuple(dr::zeros<Int32>(w), zero_acc, zero_acc),
              std::make_tuple(si),
              [](const Int32& k, const auto&...) { return k < 4; },
              [this, r, ad_uniform, active, eps](
                  Int32& k, Acc& du_acc, Acc& dv_acc,
                  const SurfaceInteraction3f& si_in) {
                Mask is_v = (k & 2) != 0;
                Float sgn = dr::select((k & 1) == 0, Float(1.0f), Float(-1.0f));
                // MaterialX's v is `1 - si.uv.y`, so `+v` steps along
                // `-si.uv.y`.
                Float du = dr::select(is_v, Float(0.0f), sgn * eps.x());
                Float dv = dr::select(is_v, -sgn * eps.y(), Float(0.0f));
                SurfaceInteraction3f s = si_in;
                s.uv += Point2f(du, dv);
                s.p += du * si_in.dp_du + dv * si_in.dp_dv;
                absl::InlinedVector<Val, 32> reg_uvals;
                const Val* uptr = nullptr;
                if (ad_uniform && prog_.uniform_end > 0) {
                  EvalUniformNodes(reg_uvals);
                  uptr = reg_uvals.data();
                }
                absl::InlinedVector<Val, 64> reg_vals(r.end - r.begin);
                EvalScratch reg_scratch;
                EvalFrame reg_frame{reg_vals.data(), uptr, reg_scratch};
                EvalRange(r.begin, r.end, r.begin, reg_frame, s, active);
                Val v = reg_vals[r.root - r.begin];
                Acc val;
                if constexpr (std::is_same_v<Acc, Float>) {
                  val = v.x();
                } else {
                  val = Vector3f(v.x(), v.y(), v.z());
                }
                du_acc += dr::select(is_v, Acc(0.0f), sgn * val);
                dv_acc += dr::select(is_v, sgn * val, Acc(0.0f));
                k += 1;
              },
              "MaterialX heighttonormal", -1);
          return std::make_pair(d_u, d_v);
        };
        Vector3f norm;
        if (r.has_texcoord) {
          auto [d_u, d_v] = run_h2n(dr::zeros<Vector3f>(w));
          Float dh_du = d_u.x() * (inv_2eps_u * kSobelScale) * scale;
          Float dh_dv = d_v.x() * (inv_2eps_v * kSobelScale) * scale;
          Vector3f tangent(d_u.y() * inv_2eps_u, d_u.z() * inv_2eps_u, dh_du);
          Vector3f bitangent(d_v.y() * inv_2eps_v, d_v.z() * inv_2eps_v, dh_dv);
          norm = dr::cross(tangent, bitangent);
          norm = detail::SafeNormalize(
              dr::select(norm.z() < 0.0f, -norm, norm),
              Vector3f(0.0f, 0.0f, 1.0f),
              static_cast<ScalarFloat>(kMtlxFloatEps * kMtlxFloatEps));
        } else {
          auto [d_u, d_v] = run_h2n(dr::zeros<Float>(w));
          Float dh_du = d_u * (inv_2eps_u * kSobelScale) * scale;
          Float dh_dv = d_v * (inv_2eps_v * kSobelScale) * scale;
          norm = dr::normalize(Vector3f(-dh_du, -dh_dv, 1.0f));
        }
        out = val3(norm * 0.5f + 0.5f);
        break;
      }

      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal: {
        bool is_3d = NoiseIs3D(imm);
        if (NoiseIsVec3(imm)) {
          out = is_3d ? EvalNoiseOp<3, Vector3f>(n, a)
                      : EvalNoiseOp<2, Vector3f>(n, a);
        } else {
          out =
              is_3d ? EvalNoiseOp<3, Float>(n, a) : EvalNoiseOp<2, Float>(n, a);
        }
        break;
      }

      default:  // Pure ops are handled above.
        break;
    }
    frame.out_vals[i - base_offset] = out;
  }
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::Eval1(const SurfaceInteraction3f& si,
                                            Mask active) const {
  Color3f c = Eval3(si, active);
  return prog_.ResultScalar() ? c.x() : luminance(c);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Color3f
MtlxEvaluator<Float, Spectrum>::Eval3(const SurfaceInteraction3f& si,
                                      Mask active) const {
  MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
  Val v = EvalExpr(si, active);
  return Color3f(v.x(), v.y(), v.z());
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::UnpolarizedSpectrum
MtlxEvaluator<Float, Spectrum>::EvalSpectrum(size_t slot_idx,
                                             const Wavelength& wavelengths,
                                             const EvalState& state) const {
  const SpecSlot& slot = spec_slots_[slot_idx];
  if constexpr (is_spectral_v<Spectrum>) {
    if (slot.uniform_spec_idx >= 0 && !HasOpaqueGrad()) {
      const UniformSpec& u = uniform_specs_[slot.uniform_spec_idx];
      return detail::SrgbUpsampler<Float, Spectrum>::EvalCoeffs(
          u.coeff, u.scale, wavelengths, slot.unbounded);
    }
  }
  Val v = ReadVal(slot.value, state);
  bool clipped01 = prog_.IsValueClipped01(slot.value);
  if (prog_.IsValueScalar(slot.value)) {
    return UnpolarizedSpectrum(
        slot.unbounded ? dr::maximum(v.x(), 0.0f)
                       : (clipped01 ? v.x() : dr::clip(v.x(), 0.0f, 1.0f)));
  }
  return srgb_.Upsample(
      ClampColor(Color3f(dr::head<3>(v)), slot.unbounded, clipped01),
      wavelengths, slot.unbounded);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::TraverseCb(
    void* payload, const drjit::TraverseVisitor& cb) {
  DR_TRAVERSE_MEMBER(inputs_)
  DR_TRAVERSE_MEMBER(opaques_)
  DR_TRAVERSE_MEMBER(uniform_scalars_)
  if constexpr (is_spectral_v<Spectrum>) {
    DR_TRAVERSE_MEMBER(uniform_specs_)
  }
  srgb_.TraverseCb(payload, cb);
  prog_.TraverseState(payload, cb);
}

MI_INSTANTIATE_CLASS(MtlxEvaluator)

void RegisterMaterialXExprTexture() {
  static const bool registered = [] {
    auto register_fn = [](auto... args) {
      PluginManager::instance()->register_plugin(args..., nullptr, nullptr);
    };
    MI_REGISTER_PLUGIN(register_fn, "materialx_expr", MtlxExprTexture)
    return true;
  }();
  (void)registered;
}

}  // namespace mitsuba::mtlx
