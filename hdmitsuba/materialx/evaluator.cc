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
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <drjit/matrix.h>
#include <drjit/while_loop.h>
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
#include "hdmitsuba/materialx/srgb_lut.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {
namespace {

template <size_t N, typename Float>
void eval_matrix_op(Op op, uint16_t imm, Vector<Float, 4>*& sp,
                    Vector<Float, 4>* tmps) {
  using Matrix = dr::Matrix<Float, N>;
  using Row = dr::Array<Float, N>;
  auto pop_mat = [&]() {
    sp -= N;
    Matrix m;
    for (size_t i = 0; i < N; ++i) m.entry(i) = Row(dr::head<N>(sp[i]));
    return m;
  };
  Matrix result;
  if (op == Op::MatMul) {
    Matrix b = pop_mat();
    result = pop_mat() * b;
  } else {
    Matrix m = pop_mat();
    if (op == Op::MatDet) {
      *sp++ = Vector<Float, 4>(dr::det(m));
      return;
    }
    result =
        op == Op::MatTranspose ? dr::transpose(m) : detail::mx_safe_inverse(m);
  }
  for (size_t i = 0; i < N; ++i) {
    Vector<Float, 4> row(0.0f);
    for (size_t j = 0; j < N; ++j) row.entry(j) = result(i, j);
    tmps[imm + i] = row;
  }
}

template <int Dim, typename Out, typename Float>
void eval_noise_op(Op op, uint16_t imm, Vector<Float, 4>*& sp) {
  auto pop = [&](int n) {
    sp -= n;
    return sp;
  };
  Out res;
  if (op == Op::CellNoise) {
    Vector<Float, 4>* a = pop(1);
    res = detail::mx_cell_noise<Dim, Out, Float>(dr::head<3>(a[0]));
  } else if (op == Op::Noise) {
    Vector<Float, 4>* a = pop(1);
    res = detail::mx_perlin_noise<Dim, Out, Float>(dr::head<3>(a[0]));
  } else if (op == Op::Worley) {
    Vector<Float, 4>* a = pop(3);
    res = detail::mx_worley_noise<Dim, Out, Float>(dr::head<3>(a[0]), a[1].x(),
                                                   dr::round(a[2].x()) == 1.0f);
  } else {
    uint16_t oct = fractal_octaves(imm);
    if (oct == kFractalDynamicOctaves) {
      Vector<Float, 4>* a = pop(4);
      res = detail::mx_fractal_noise<Dim, Out, Float>(
          dr::head<3>(a[0]), a[3].x(), a[1].x(), a[2].x(), kMaxFractalOctaves);
    } else {
      Vector<Float, 4>* a = pop(3);
      res = detail::mx_fractal_noise<Dim, Out, Float>(
          dr::head<3>(a[0]), int(oct), a[1].x(), a[2].x());
    }
  }
  if constexpr (std::is_same_v<Out, Float>) {
    *sp++ = Vector<Float, 4>(res);
  } else {
    *sp++ = Vector<Float, 4>(res.x(), res.y(), res.z(), 0.0f);
  }
}

/**!

.. _texture-materialx_expr:

MaterialX Expression (:monosp:`materialx_expr`)
-------------------------------------------------

.. pluginparameters::

 * - program
   - |any|
   - Compiled ``mitsuba::mtlx::MtlxProgram`` bytecode and constant/opaque
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
    m_eval.compile(props);
    if (!m_eval.program().result_uniform) m_eval.load_srgb_lut();
  }

  void traverse(TraversalCallback* cb) override { m_eval.traverse(cb); }

  void parameters_changed(const std::vector<std::string>& = {}) override {
    m_eval.parameters_changed();
  }

  UnpolarizedSpectrum eval(const SurfaceInteraction3f& si,
                           Mask active) const override {
    if constexpr (is_monochromatic_v<Spectrum>) {
      return eval_1(si, active);
    } else if constexpr (is_rgb_v<Spectrum>) {
      return eval_3(si, active);
    } else {
      Color3f c = eval_3(si, active);
      const MtlxProgram& prog = m_eval.program();
      if (prog.result_uniform) {
        return UnpolarizedSpectrum(prog.unbounded
                                       ? dr::maximum(c.x(), 0.0f)
                                       : dr::clip(c.x(), 0.0f, 1.0f));
      }
      return m_eval.upsample_rgb(c, si.wavelengths, prog.unbounded);
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
    return m_eval.eval_1(si, active);
  }

  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const override {
    return m_eval.eval_3(si, active);
  }

  bool is_spatially_varying() const override {
    return m_eval.is_spatially_varying();
  }

  std::string to_string() const override {
    return "MtlxExprTexture[\n" + m_eval.to_string() + "]";
  }

  MI_DECLARE_CLASS(MtlxExprTexture)

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb) override {
    if (cb.role != drjit::TraverseRole::Freeze &&
        cb.role != drjit::TraverseRole::Children) {
      return;
    }
    Texture::traverse_cb(payload, cb);
    m_eval.traverse_cb(payload, cb);
  }

 private:
  MtlxEvaluator<Float, Spectrum> m_eval;
};

}  // namespace

template <typename Float, typename Spectrum>
MtlxEvaluator<Float, Spectrum>::MtlxEvaluator()
    : m_srgb(std::make_unique<detail::SrgbUpsampler<Float, Spectrum>>()) {}

template <typename Float, typename Spectrum>
MtlxEvaluator<Float, Spectrum>::~MtlxEvaluator() = default;

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::compile(const Properties& props) {
  m_inputs.clear();
  for (size_t i = 0;; ++i) {
    std::string name = "in" + std::to_string(i);
    if (!props.has_property(name)) break;
    m_inputs.push_back(props.get_texture<Texture>(name));
  }

  m_prog = props.get_any<MtlxProgram>("program");
  m_opaques.clear();
  m_opaques.reserve(m_prog.opaques.size());
  for (double v : m_prog.opaques) {
    m_opaques.push_back(dr::opaque<Float>(static_cast<ScalarFloat>(v)));
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::load_srgb_lut() {
  m_srgb->load();
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse(TraversalCallback* cb) {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    cb->put("in" + std::to_string(i), m_inputs[i], ParamFlags::Differentiable);
  }
  for (size_t i = 0; i < m_opaques.size(); ++i) {
    cb->put("param" + std::to_string(i), m_opaques[i],
            ParamFlags::Differentiable);
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::parameters_changed() {
  if constexpr (dr::is_jit_v<Float>) {
    bool scheduled = false;
    for (Float& p : m_opaques) {
      if (unlikely(p.size() != 1)) {
        Throw("MtlxEvaluator: parameter must have size 1 (got %zu)!", p.size());
      }
      scheduled |= dr::detail::schedule_force(p);
    }
    if (scheduled) dr::eval();
  }
}

template <typename Float, typename Spectrum>
std::string MtlxEvaluator<Float, Spectrum>::to_string() const {
  std::ostringstream oss;
  oss << "  code_size = " << m_prog.code.size() << ",\n";
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    oss << "  in" << i << " = " << string::indent(m_inputs[i]) << ",\n";
  }
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::is_spatially_varying() const {
  for (Inst inst : m_prog.code) {
    if (inst.op == Op::SampleInput) {
      if (m_inputs[inst.imm]->is_spatially_varying()) return true;
    } else if (is_spatially_varying_op(inst.op)) {
      return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval(const SurfaceInteraction3f& si,
                                     Mask active, Val* out_tmps,
                                     int stop_after_tmp) const {
  return eval_code(m_prog.code, m_prog.tmp_channels.size(), m_prog.stack_size,
                   si, active, out_tmps, stop_after_tmp);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval_code(absl::Span<const Inst> code,
                                          size_t tmp_count, uint32_t stack_size,
                                          const SurfaceInteraction3f& si,
                                          Mask active, Val* out_tmps,
                                          int stop_after_tmp) const {
  size_t local_tmps =
      out_tmps ? 0
               : (stop_after_tmp >= 0 ? static_cast<size_t>(stop_after_tmp) + 1
                                      : tmp_count);
  absl::InlinedVector<Val, 64> mem(local_tmps + stack_size);
  Val* tmps = out_tmps ? out_tmps : mem.data();
  Val* stack_base = mem.data() + local_tmps;
  Val* sp = stack_base;
  std::optional<SurfaceInteraction3f> si_uv;

  auto pop = [&](int n) {
    sp -= n;
    return sp;
  };
  auto xyz = [](const Val& v) -> Vector3f { return dr::head<3>(v); };
  auto val3 = [](const auto& v, const Float& w = 0.0f) {
    return Val(v.x(), v.y(), v.z(), w);
  };

  auto eval_input = [&](uint16_t k, bool mono,
                        const SurfaceInteraction3f& si_q) -> Val {
    return mono ? Val(m_inputs[k]->eval_1(si_q, active))
                : val3(m_inputs[k]->eval_3(si_q, active));
  };

  for (Inst inst : code) {
    if (detail::eval_pure_op<Float>(inst, sp)) continue;
    uint16_t imm = inst.imm;
    switch (inst.op) {
      case Op::Const:
        *sp++ = Val(static_cast<ScalarFloat>(m_prog.consts[imm]));
        break;
      case Op::Opaque:
        *sp++ = Val(m_opaques[imm]);
        break;
      case Op::LoadTmp:
        *sp++ = tmps[imm];
        break;
      case Op::StoreTmp:
        tmps[imm] = *--sp;
        if (static_cast<int>(imm) == stop_after_tmp) return tmps[imm];
        break;
      case Op::SampleInput:
        *sp++ = eval_input(imm, inst.channels == 1, si);
        break;
      case Op::SampleInputUV: {
        Val* a = pop(2);
        if (!si_uv) si_uv.emplace(dr::zeros<SurfaceInteraction3f>());
        si_uv->uv = Point2f(a[0].x(), a[1].x());
        *sp++ = eval_input(imm, inst.channels == 1, *si_uv);
        break;
      }
      case Op::Attr: {
        if constexpr (!dr::is_jit_v<Float>) {
          if (!active || !si.shape) break;
        }
        const std::string& attr = m_prog.attr_names[imm];
        Mask has = si.shape->has_attribute(attr, active) & active;
        if (dr::none_or<false>(has)) break;
        // A fourth channel is that of the default.
        sp[-1] = dr::select(
            has,
            inst.channels == 1
                ? Val(si.shape->eval_attribute_1(attr, si, has))
                : val3(si.shape->eval_attribute_3(attr, si, has), sp[-1].w()),
            sp[-1]);
        break;
      }

      case Op::GeomUV:
        *sp++ = Val(si.uv.x(), 1.0f - si.uv.y(), 0.0f, 0.0f);
        break;
      case Op::GeomPosWorld:
        *sp++ = val3(si.p);
        break;
      case Op::GeomNormalWorld:
        *sp++ = val3(si.sh_frame.n);
        break;
      case Op::GeomTangentWorld:
        *sp++ = val3(detail::geom_tangent_world(si));
        break;
      case Op::GeomBitangentWorld:
        *sp++ = val3(
            detail::geom_bitangent_world(si, detail::geom_tangent_world(si)));
        break;
      case Op::GeomIncidentDir:
        *sp++ = val3(-si.to_world(si.wi));
        break;

      case Op::NormalMap: {
        Val* a = pop(5);
        *sp++ = val3(detail::mx_normal_map(
            xyz(a[0]), Vector2f(dr::head<2>(a[1])), xyz(a[2]), xyz(a[3]),
            xyz(a[4]), Vector3f(si.sh_frame.n)));
        break;
      }
      case Op::HeightToNormal: {
        Float scale = pop(1)[0].x();
        const SubProgram& sub = m_prog.sub_programs[imm];
        absl::Span<const Inst> sub_code(m_prog.sub_code.data() + sub.code_begin,
                                        sub.code_end - sub.code_begin);
        // Matches MaterialX 1.39 `mx_heighttonormal_vector3` (`1 / 16` Sobel
        // scale factor and cross-product Jacobian over `(u_mtlx, v_mtlx)`).
        ScalarVector2i max_res(1);
        for (Inst sub_inst : sub_code) {
          if (sub_inst.op == Op::SampleInput ||
              sub_inst.op == Op::SampleInputUV) {
            max_res =
                dr::maximum(max_res, m_inputs[sub_inst.imm]->resolution());
          }
        }
        // Use an exact multiple of 1/256 texel (`0.125 / res`) when sampling
        // bitmaps so CUDA's 8-bit fixed-point `tex2D` sub-texel interpolation
        // introduces zero quantization error in central differences.
        ScalarPoint2f eps =
            dr::select(max_res > 1,
                       dr::maximum(ScalarFloat(0.125f) / ScalarPoint2f(max_res),
                                   ScalarFloat(1e-5f)),
                       ScalarPoint2f(1e-4f));
        ScalarFloat inv_2eps_u = 0.5f / eps.x();
        ScalarFloat inv_2eps_v = 0.5f / eps.y();
        constexpr ScalarFloat kSobelScale = 1.0f / 16.0f;
        using Int32 = dr::int32_array_t<Float>;
        size_t w = dr::width(si, active);
        auto [_, d_u, d_v, _si] = dr::while_loop(
            std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Vector3f>(w),
                            dr::zeros<Vector3f>(w), si),
            [](const Int32& k, const Vector3f&, const Vector3f&,
               const SurfaceInteraction3f&) { return k < 4; },
            [this, sub_code, sub, active, eps](Int32& k, Vector3f& du_acc,
                                               Vector3f& dv_acc,
                                               SurfaceInteraction3f& si_in) {
              Mask is_v = (k & 2) != 0;
              Float sgn = dr::select((k & 1) == 0, Float(1.0f), Float(-1.0f));
              // MaterialX's v is `1 - si.uv.y`, so `+v` steps along `-si.uv.y`.
              Float du = dr::select(is_v, Float(0.0f), sgn * eps.x());
              Float dv = dr::select(is_v, -sgn * eps.y(), Float(0.0f));
              SurfaceInteraction3f s = si_in;
              s.uv += Point2f(du, dv);
              s.p += du * si_in.dp_du + dv * si_in.dp_dv;
              Val v =
                  eval_code(sub_code, sub.tmp_count, sub.stack_size, s, active);
              Vector3f val = sub.has_texcoord ? Vector3f(v.x(), v.y(), v.z())
                                              : Vector3f(v.x(), 0.0f, 0.0f);
              du_acc += dr::select(is_v, Vector3f(0.0f), sgn * val);
              dv_acc += dr::select(is_v, sgn * val, Vector3f(0.0f));
              k += 1;
            },
            "MaterialX heighttonormal", 4);
        Float dh_du = d_u.x() * (inv_2eps_u * kSobelScale) * scale;
        Float dh_dv = d_v.x() * (inv_2eps_v * kSobelScale) * scale;
        Vector3f n;
        if (sub.has_texcoord) {
          Vector3f tangent(d_u.y() * inv_2eps_u, d_u.z() * inv_2eps_u, dh_du);
          Vector3f bitangent(d_v.y() * inv_2eps_v, d_v.z() * inv_2eps_v, dh_dv);
          n = dr::cross(tangent, bitangent);
          n = detail::safe_normalize(
              dr::select(n.z() < 0.0f, -n, n), Vector3f(0.0f, 0.0f, 1.0f),
              static_cast<ScalarFloat>(kMtlxFloatEps * kMtlxFloatEps));
        } else {
          n = dr::normalize(Vector3f(-dh_du, -dh_dv, 1.0f));
        }
        *sp++ = val3(n * 0.5f + 0.5f);
        break;
      }
      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal: {
        bool is_3d = noise_is_3d(imm);
        if (noise_is_vec3(imm)) {
          if (is_3d) {
            eval_noise_op<3, Vector3f>(inst.op, imm, sp);
          } else {
            eval_noise_op<2, Vector3f>(inst.op, imm, sp);
          }
        } else {
          if (is_3d) {
            eval_noise_op<3, Float>(inst.op, imm, sp);
          } else {
            eval_noise_op<2, Float>(inst.op, imm, sp);
          }
        }
        break;
      }
      case Op::MatDet:
      case Op::MatTranspose:
      case Op::MatInverse:
      case Op::MatMul:
        if (inst.channels == 3) {
          eval_matrix_op<3>(inst.op, imm, sp, tmps);
        } else {
          eval_matrix_op<4>(inst.op, imm, sp, tmps);
        }
        break;
      default:  // Pure ops are handled above.
        break;
    }
  }

  return sp > stack_base ? sp[-1] : Val(0.0f);
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::eval_1(const SurfaceInteraction3f& si,
                                             Mask active) const {
  Color3f c = eval_3(si, active);
  return m_prog.result_uniform ? c.x() : luminance(c);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Color3f
MtlxEvaluator<Float, Spectrum>::eval_3(const SurfaceInteraction3f& si,
                                       Mask active) const {
  MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
  Val v = eval(si, active);
  return Color3f(v.x(), v.y(), v.z());
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::UnpolarizedSpectrum
MtlxEvaluator<Float, Spectrum>::upsample_rgb(const Color3f& c,
                                             const Wavelength& wavelengths,
                                             bool unbounded) const {
  return m_srgb->upsample(c, wavelengths, unbounded);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse_cb(
    void* payload, const drjit::TraverseVisitor& cb) {
  DR_TRAVERSE_MEMBER(m_inputs)
  DR_TRAVERSE_MEMBER(m_opaques)
  m_srgb->traverse_cb(payload, cb);
  m_prog.traverse_state(payload, cb);
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

}  // namespace mtlx
NAMESPACE_END(mitsuba)
