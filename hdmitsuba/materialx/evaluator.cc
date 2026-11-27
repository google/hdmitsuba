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
void eval_matrix_op(const Node& node, const Vector<Float, 4>* a,
                    Vector<Float, 4>& out, Vector<Float, 4>* mat_rows) {
  using Matrix = dr::Matrix<Float, N>;
  using Row = dr::Array<Float, N>;
  auto read_mat = [&](size_t off) {
    Matrix m;
    for (size_t i = 0; i < N; ++i) m.entry(i) = Row(dr::head<N>(a[off + i]));
    return m;
  };
  Matrix result;
  if (node.op == Op::MatMul) {
    result = read_mat(0) * read_mat(N);
  } else {
    Matrix m = read_mat(0);
    if (node.op == Op::MatDet) {
      out = Vector<Float, 4>(dr::det(m));
      return;
    }
    result = node.op == Op::MatTranspose ? dr::transpose(m)
                                         : detail::mx_safe_inverse(m);
  }
  Vector<Float, 4>* dst = mat_rows + static_cast<size_t>(node.imm) * 4;
  for (size_t i = 0; i < N; ++i) {
    Vector<Float, 4> row(0.0f);
    for (size_t j = 0; j < N; ++j) row.entry(j) = result(i, j);
    dst[i] = row;
  }
}

template <int Dim, typename Out, typename Float>
Vector<Float, 4> eval_noise_op(const Node& node, const Vector<Float, 4>* a) {
  Out res;
  if (node.op == Op::CellNoise) {
    res = detail::mx_cell_noise<Dim, Out, Float>(dr::head<Dim>(a[0]));
  } else if (node.op == Op::Noise) {
    res = detail::mx_perlin_noise<Dim, Out, Float>(dr::head<Dim>(a[0]));
  } else if (node.op == Op::Worley) {
    res = detail::mx_worley_noise<Dim, Out, Float>(
        dr::head<Dim>(a[0]), a[1].x(), dr::round(a[2].x()) == 1.0f);
  } else {
    uint16_t oct = fractal_octaves(node.imm);
    if (oct == kFractalDynamicOctaves) {
      res = detail::mx_fractal_noise<Dim, Out, Float>(
          dr::head<Dim>(a[0]), a[3].x(), a[1].x(), a[2].x(),
          kMaxFractalOctaves);
    } else {
      res = detail::mx_fractal_noise<Dim, Out, Float>(
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
Color3f clamp_color(const Color3f& c, bool unbounded) {
  return unbounded ? dr::maximum(c, 0.0f) : dr::clip(c, 0.0f, 1.0f);
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
    m_eval.compile(props);
    if constexpr (is_spectral_v<Spectrum>) {
      if (m_eval.program().result_root != kInvalidValue) {
        m_eval.add_spectrum(m_eval.program().result_root, /*unbounded=*/true);
      }
    }
    m_eval.finalize_uniforms();
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
      MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
      if (m_eval.program().result_root == kInvalidValue) return 0.0f;
      typename MtlxEvaluator<Float, Spectrum>::EvalState state;
      m_eval.eval_surface(si, active, state);
      return m_eval.eval_spectrum(0, si.wavelengths, state);
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
  compile(props, props.get_any<MtlxValueProgram>("program"));
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::compile(const Properties& props,
                                             MtlxValueProgram prog) {
  m_inputs.clear();
  for (size_t i = 0;; ++i) {
    std::string name = "in" + std::to_string(i);
    if (!props.has_property(name)) break;
    m_inputs.push_back(props.get_texture<Texture>(name));
  }

  m_prog = std::move(prog);
  m_spec_slots.clear();
  m_uniform_specs.clear();
  m_opaques.clear();
  m_opaques.reserve(m_prog.opaques.size());
  for (double v : m_prog.opaques) {
    m_opaques.push_back(dr::opaque<Float>(static_cast<ScalarFloat>(v)));
  }
}

template <typename Float, typename Spectrum>
int MtlxEvaluator<Float, Spectrum>::add_spectrum(ValueId id, bool unbounded) {
  for (size_t i = 0; i < m_spec_slots.size(); ++i) {
    if (m_spec_slots[i].value == id && m_spec_slots[i].unbounded == unbounded) {
      return static_cast<int>(i);
    }
  }
  int uniform_spec_idx = -1;
  if constexpr (is_spectral_v<Spectrum>) {
    if (!m_prog.is_value_scalar(id)) {
      m_srgb->load();
      if (id < m_prog.uniform_end) {
        uniform_spec_idx = static_cast<int>(m_uniform_specs.size());
        m_uniform_specs.emplace_back();
      }
    }
  }
  m_spec_slots.push_back({id, unbounded, uniform_spec_idx});
  return static_cast<int>(m_spec_slots.size() - 1);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::finalize_uniforms() {
  update_uniform_state();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::has_opaque_grad() const {
  if constexpr (dr::is_diff_v<Float>) {
    for (const Float& p : m_opaques) {
      if (dr::grad_enabled(p)) return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::update_uniform_state() {
  // Precompute finite-difference UV steps per `HeightToNormal` region.
  m_region_eps.resize(m_prog.regions.size());
  for (size_t r_idx = 0; r_idx < m_prog.regions.size(); ++r_idx) {
    const Region& r = m_prog.regions[r_idx];
    ScalarVector2i max_res(1);
    for (uint32_t i = r.begin; i < r.end; ++i) {
      const Node& n = m_prog.nodes[i];
      if (n.op == Op::SampleInput || n.op == Op::SampleInputUV) {
        max_res = dr::maximum(max_res, m_inputs[n.imm]->resolution());
      }
    }
    // Use an exact multiple of 1/256 texel (`0.125 / res`) when sampling
    // bitmaps so CUDA's 8-bit fixed-point `tex2D` sub-texel interpolation
    // introduces zero quantization error in central differences.
    m_region_eps[r_idx] =
        dr::select(max_res > 1,
                   dr::maximum(ScalarFloat(0.125f) / ScalarPoint2f(max_res),
                               ScalarFloat(1e-5f)),
                   ScalarPoint2f(1e-4f));
  }

  m_uniform_scalars.clear();
  if (m_prog.uniform_scalar_count == 0 && m_uniform_specs.empty()) return;

  absl::InlinedVector<Val, 32> uvals;
  eval_uniform_nodes(uvals);

  bool scheduled = false;
  if (m_prog.uniform_scalar_count > 0) {
    m_uniform_scalars.reserve(m_prog.uniform_scalar_count);
    for (uint32_t u = 0; u < m_prog.uniform_end; ++u) {
      if (m_prog.uniform_offsets[u] == kNoUniformScalar) continue;
      uint8_t ch = m_prog.nodes[u].channels;
      for (uint8_t c = 0; c < ch; ++c) {
        m_uniform_scalars.push_back(dr::detach(uvals[u][c]));
      }
    }
    if constexpr (dr::is_jit_v<Float>) {
      for (Float& s : m_uniform_scalars) {
        scheduled |= dr::detail::schedule_force(s);
      }
    }
  }
  if constexpr (is_spectral_v<Spectrum>) {
    for (const SpecSlot& slot : m_spec_slots) {
      if (slot.uniform_spec_idx < 0) continue;
      Color3f c =
          clamp_color(Color3f(dr::head<3>(uvals[slot.value])), slot.unbounded);
      auto [coeff, scale] = m_srgb->fetch_coeffs(c, slot.unbounded);
      UniformSpec& dst = m_uniform_specs[slot.uniform_spec_idx];
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
MtlxEvaluator<Float, Spectrum>::load_uniform(ValueId id) const {
  const Node& n = m_prog.nodes[id];
  if (n.op == Op::Const) {
    return Val(static_cast<ScalarFloat>(m_prog.consts[n.imm]));
  }
  if (n.op == Op::Opaque) return Val(m_opaques[n.imm]);
  uint32_t off = m_prog.uniform_offsets[id];
  if (n.channels == 1) return Val(m_uniform_scalars[off]);
  Val out(0.0f);
  for (uint8_t c = 0; c < n.channels; ++c) out[c] = m_uniform_scalars[off + c];
  return out;
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::eval_uniform_nodes(
    absl::InlinedVector<Val, 32>& uvals) const {
  uvals.resize(m_prog.uniform_end);
  absl::InlinedVector<Val, 4> mats(m_prog.uniform_mat_count * 4);
  absl::InlinedVector<std::optional<CachedAttr>, 4> attr_cache;
  std::optional<SurfaceInteraction3f> si_uv;
  SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
  eval_range(0, m_prog.uniform_end, 0, uvals.data(), mats.data(), uvals.data(),
             si, Mask(true), attr_cache, si_uv);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::eval_until(uint32_t end,
                                                const SurfaceInteraction3f& si,
                                                Mask active,
                                                EvalState& state) const {
  if (!state.uniform_ready) {
    state.uniform_ready = true;
    if (m_prog.uniform_scalar_count > 0 && has_opaque_grad()) {
      eval_uniform_nodes(state.uniform_vals);
    }
  }
  uint32_t start = std::max(state.evaluated_end, m_prog.uniform_end);
  if (start >= end) return;

  state.varying_vals.resize(end - m_prog.uniform_end);
  if (m_prog.main_mat_count > 0 && state.mat_rows.empty()) {
    state.mat_rows.resize(m_prog.main_mat_count * 4);
  }
  const Val* uptr =
      state.uniform_vals.empty() ? nullptr : state.uniform_vals.data();
  eval_range(start, end, m_prog.uniform_end, state.varying_vals.data(),
             state.mat_rows.data(), uptr, si, active, state.attr_cache,
             state.si_uv);
  state.evaluated_end = end;
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
  update_uniform_state();
}

template <typename Float, typename Spectrum>
std::string MtlxEvaluator<Float, Spectrum>::to_string() const {
  std::ostringstream oss;
  oss << "  code_size = " << m_prog.nodes.size() << ",\n";
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    oss << "  in" << i << " = " << string::indent(m_inputs[i]) << ",\n";
  }
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::is_spatially_varying() const {
  for (size_t i = m_prog.uniform_end; i < m_prog.nodes.size(); ++i) {
    const Node& n = m_prog.nodes[i];
    if (n.op == Op::SampleInput) {
      if (m_inputs[n.imm]->is_spatially_varying()) return true;
    } else if (is_spatially_varying_op(n.op)) {
      return true;
    }
  }
  return false;
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::read_val(ValueId id,
                                         const EvalState& state) const {
  if (id < m_prog.uniform_end) {
    return state.uniform_vals.empty() ? load_uniform(id)
                                      : state.uniform_vals[id];
  }
  return state.varying_vals[id - m_prog.uniform_end];
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval_opacity(ValueId opacity_root,
                                             const SurfaceInteraction3f& si,
                                             Mask active,
                                             EvalState& state) const {
  eval_until(m_prog.opacity_end, si, active, state);
  return read_val(opacity_root, state);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::eval_surface(
    const SurfaceInteraction3f& si, Mask active, EvalState& state) const {
  eval_until(m_prog.main_end, si, active, state);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval_expr(const SurfaceInteraction3f& si,
                                          Mask active) const {
  if (m_prog.result_root == kInvalidValue) return Val(0.0f);
  EvalState state;
  eval_until(m_prog.main_end, si, active, state);
  return read_val(m_prog.result_root, state);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::eval_range(
    uint32_t begin, uint32_t end, uint32_t base_offset, Val* out_vals,
    Val* mat_rows, const Val* uniform_vals, const SurfaceInteraction3f& si,
    Mask active, absl::InlinedVector<std::optional<CachedAttr>, 4>& attr_cache,
    std::optional<SurfaceInteraction3f>& si_uv) const {
  uint32_t uniform_end = m_prog.uniform_end;
  auto get = [&](ValueId a) -> Val {
    if (a < uniform_end) {
      return uniform_vals ? uniform_vals[a] : load_uniform(a);
    }
    return out_vals[a - base_offset];
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

  Val a[8];
  for (uint32_t i = begin; i < end; ++i) {
    const Node& n = m_prog.nodes[i];
    if (n.op == Op::MatRow) {
      uint16_t mat_slot = m_prog.nodes[n.args[0]].imm;
      out_vals[i - base_offset] =
          mat_rows[static_cast<size_t>(mat_slot) * 4 + n.imm];
      continue;
    }
    auto arg_ids = node_args(n, m_prog.extra_args);
    for (size_t j = 0; j < arg_ids.size(); ++j) a[j] = get(arg_ids[j]);

    Val out(0.0f);
    if (detail::eval_pure_op<Float>(n, a, out)) {
      out_vals[i - base_offset] = out;
      continue;
    }

    uint16_t imm = n.imm;
    switch (n.op) {
      case Op::Const:
        out = Val(static_cast<ScalarFloat>(m_prog.consts[imm]));
        break;
      case Op::Opaque:
        out = Val(m_opaques[imm]);
        break;
      case Op::SampleInput:
        out = eval_input(imm, n.flags == 1, si);
        break;
      case Op::SampleInputUV:
        if (!si_uv) si_uv.emplace(dr::zeros<SurfaceInteraction3f>());
        si_uv->uv = Point2f(a[0].x(), a[1].x());
        out = eval_input(imm, n.flags == 1, *si_uv);
        break;
      case Op::Attr: {
        out = a[0];
        if constexpr (!dr::is_jit_v<Float>) {
          if (!active || !si.shape) break;
        }
        if (attr_cache.empty()) attr_cache.resize(m_prog.attr_names.size());
        std::optional<CachedAttr>& cached = attr_cache[imm];
        if (!cached) {
          const std::string& attr = m_prog.attr_names[imm];
          Mask has = si.shape->has_attribute(attr, active) & active;
          Val val =
              dr::none_or<false>(has)
                  ? Val(0.0f)
                  : (n.flags == 1
                         ? Val(si.shape->eval_attribute_1(attr, si, has))
                         : val3(si.shape->eval_attribute_3(attr, si, has)));
          cached.emplace(CachedAttr{has, val});
        }
        if (dr::none_or<false>(cached->has)) break;
        // A fourth channel is that of the default.
        out = dr::select(
            cached->has,
            n.flags == 1 ? cached->val : val3(cached->val, a[0].w()), a[0]);
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
        out = val3(detail::geom_tangent_world(si));
        break;
      case Op::GeomBitangentWorld:
        out = val3(
            detail::geom_bitangent_world(si, detail::geom_tangent_world(si)));
        break;
      case Op::GeomIncidentDir:
        out = val3(-si.to_world(si.wi));
        break;

      case Op::NormalMap:
        if (imm == kNormalMapDefaultFrame) {
          Vector3f t = detail::geom_tangent_world(si);
          out = val3(detail::mx_normal_map_frame(
              xyz(a[0]), Vector2f(dr::head<2>(a[1])),
              Frame3f(t, detail::geom_bitangent_world(si, t), si.sh_frame.n)));
        } else {
          out = val3(detail::mx_normal_map(
              xyz(a[0]), Vector2f(dr::head<2>(a[1])), xyz(a[2]), xyz(a[3]),
              xyz(a[4]), Vector3f(si.sh_frame.n)));
        }
        break;

      case Op::HeightToNormal: {
        Float scale = a[0].x();
        const Region& r = m_prog.regions[imm];
        ScalarPoint2f eps = m_region_eps[imm];
        // Matches MaterialX 1.39 `mx_heighttonormal_vector3` (`1 / 16` Sobel
        // scale factor and cross-product Jacobian over `(u_mtlx, v_mtlx)`).
        ScalarFloat inv_2eps_u = 0.5f / eps.x();
        ScalarFloat inv_2eps_v = 0.5f / eps.y();
        constexpr ScalarFloat kSobelScale = 1.0f / 16.0f;
        bool ad_uniform = (uniform_vals != nullptr);
        using Int32 = dr::int32_array_t<Float>;
        size_t w = dr::width(si, active);
        auto [_, d_u, d_v, _si] = dr::while_loop(
            std::make_tuple(dr::zeros<Int32>(w), dr::zeros<Vector3f>(w),
                            dr::zeros<Vector3f>(w), si),
            [](const Int32& k, const Vector3f&, const Vector3f&,
               const SurfaceInteraction3f&) { return k < 4; },
            [this, r, ad_uniform, active, eps](Int32& k, Vector3f& du_acc,
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
              absl::InlinedVector<Val, 32> reg_uvals;
              const Val* uptr = nullptr;
              if (ad_uniform && m_prog.uniform_end > 0) {
                eval_uniform_nodes(reg_uvals);
                uptr = reg_uvals.data();
              }
              absl::InlinedVector<Val, 64> reg_vals(r.end - r.begin);
              absl::InlinedVector<Val, 4> reg_mats(r.mat_count * 4);
              absl::InlinedVector<std::optional<CachedAttr>, 4> reg_attrs;
              std::optional<SurfaceInteraction3f> reg_si_uv;
              eval_range(r.begin, r.end, r.begin, reg_vals.data(),
                         reg_mats.data(), uptr, s, active, reg_attrs,
                         reg_si_uv);
              Val v = reg_vals[r.root - r.begin];
              Vector3f val = r.has_texcoord ? Vector3f(v.x(), v.y(), v.z())
                                            : Vector3f(v.x(), 0.0f, 0.0f);
              du_acc += dr::select(is_v, Vector3f(0.0f), sgn * val);
              dv_acc += dr::select(is_v, sgn * val, Vector3f(0.0f));
              k += 1;
            },
            "MaterialX heighttonormal", 4);
        Float dh_du = d_u.x() * (inv_2eps_u * kSobelScale) * scale;
        Float dh_dv = d_v.x() * (inv_2eps_v * kSobelScale) * scale;
        Vector3f norm;
        if (r.has_texcoord) {
          Vector3f tangent(d_u.y() * inv_2eps_u, d_u.z() * inv_2eps_u, dh_du);
          Vector3f bitangent(d_v.y() * inv_2eps_v, d_v.z() * inv_2eps_v, dh_dv);
          norm = dr::cross(tangent, bitangent);
          norm = detail::safe_normalize(
              dr::select(norm.z() < 0.0f, -norm, norm),
              Vector3f(0.0f, 0.0f, 1.0f),
              static_cast<ScalarFloat>(kMtlxFloatEps * kMtlxFloatEps));
        } else {
          norm = dr::normalize(Vector3f(-dh_du, -dh_dv, 1.0f));
        }
        out = val3(norm * 0.5f + 0.5f);
        break;
      }

      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal: {
        bool is_3d = noise_is_3d(imm);
        if (noise_is_vec3(imm)) {
          out = is_3d ? eval_noise_op<3, Vector3f>(n, a)
                      : eval_noise_op<2, Vector3f>(n, a);
        } else {
          out = is_3d ? eval_noise_op<3, Float>(n, a)
                      : eval_noise_op<2, Float>(n, a);
        }
        break;
      }

      case Op::MatDet:
      case Op::MatTranspose:
      case Op::MatInverse:
      case Op::MatMul: {
        uint8_t dim =
            n.op == Op::MatDet ? static_cast<uint8_t>(n.arity) : n.channels;
        if (dim == 3) {
          eval_matrix_op<3>(n, a, out, mat_rows);
        } else {
          eval_matrix_op<4>(n, a, out, mat_rows);
        }
        break;
      }

      default:  // Pure ops and MatRow are handled above.
        break;
    }
    out_vals[i - base_offset] = out;
  }
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::eval_1(const SurfaceInteraction3f& si,
                                             Mask active) const {
  Color3f c = eval_3(si, active);
  return m_prog.result_scalar() ? c.x() : luminance(c);
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Color3f
MtlxEvaluator<Float, Spectrum>::eval_3(const SurfaceInteraction3f& si,
                                       Mask active) const {
  MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
  Val v = eval_expr(si, active);
  return Color3f(v.x(), v.y(), v.z());
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::UnpolarizedSpectrum
MtlxEvaluator<Float, Spectrum>::eval_spectrum(size_t slot_idx,
                                              const Wavelength& wavelengths,
                                              const EvalState& state) const {
  const SpecSlot& slot = m_spec_slots[slot_idx];
  if constexpr (is_spectral_v<Spectrum>) {
    if (slot.uniform_spec_idx >= 0 && !has_opaque_grad()) {
      const UniformSpec& u = m_uniform_specs[slot.uniform_spec_idx];
      return detail::SrgbUpsampler<Float, Spectrum>::eval_coeffs(
          u.coeff, u.scale, wavelengths, slot.unbounded);
    }
  }
  Val v = read_val(slot.value, state);
  if (m_prog.is_value_scalar(slot.value)) {
    return UnpolarizedSpectrum(slot.unbounded ? dr::maximum(v.x(), 0.0f)
                                              : dr::clip(v.x(), 0.0f, 1.0f));
  }
  return m_srgb->upsample(clamp_color(Color3f(dr::head<3>(v)), slot.unbounded),
                          wavelengths, slot.unbounded);
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse_cb(
    void* payload, const drjit::TraverseVisitor& cb) {
  DR_TRAVERSE_MEMBER(m_inputs)
  DR_TRAVERSE_MEMBER(m_opaques)
  DR_TRAVERSE_MEMBER(m_uniform_scalars)
  if constexpr (is_spectral_v<Spectrum>) {
    DR_TRAVERSE_MEMBER(m_uniform_specs)
  }
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
