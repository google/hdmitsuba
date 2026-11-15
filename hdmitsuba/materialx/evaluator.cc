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
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/inlined_vector.h>
#include <absl/synchronization/mutex.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::compile(const Properties& props) {
  m_prog = props.get_any<MtlxProgram>("program");
  m_opaques.clear();
  m_opaques.reserve(m_prog.opaques.size());
  for (double v : m_prog.opaques) {
    m_opaques.push_back(dr::opaque<Float>(static_cast<ScalarFloat>(v)));
  }
}

template <typename Float, typename Spectrum>
void MtlxEvaluator<Float, Spectrum>::traverse(TraversalCallback* cb) {
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
        Throw("MtlxExprTexture: parameter must have size 1 (got %zu)!",
              p.size());
      }
      scheduled |= dr::detail::schedule_force(p);
    }
    if (scheduled) dr::eval();
  }
}

template <typename Float, typename Spectrum>
std::string MtlxEvaluator<Float, Spectrum>::to_string() const {
  std::ostringstream oss;
  oss << "  code_size = " << m_prog.code.size() << "\n";
  return oss.str();
}

template <typename Float, typename Spectrum>
bool MtlxEvaluator<Float, Spectrum>::is_spatially_varying() const {
  for (Inst inst : m_prog.code) {
    if (inst.op == Op::GeomUV) return true;
  }
  return false;
}

template <typename Float, typename Spectrum>
Float MtlxEvaluator<Float, Spectrum>::elementwise(Op op, const Val* sp, int c) {
  const Float& a0 = sp[-1][c];
  switch (op) {
    case Op::Neg:
      return -a0;
    case Op::Add:
      return a0 + sp[0][c];
    case Op::Sub:
      return a0 - sp[0][c];
    case Op::Mul:
      return a0 * sp[0][c];
    case Op::Div: {
      const Float& a1 = sp[0][c];
      Mask nz = a1 != 0.f;
      return dr::select(nz, a0 / dr::select(nz, a1, 1.f), 0.f);
    }
    case Op::Lt:
      return dr::select(a0 < sp[0][c], 1.f, 0.f);
    case Op::Le:
      return dr::select(a0 <= sp[0][c], 1.f, 0.f);
    case Op::Eq:
      return dr::select(a0 == sp[0][c], 1.f, 0.f);
    case Op::Max:
      return dr::maximum(a0, sp[0][c]);
    case Op::Select:
      return dr::select(a0 != 0.f, sp[0][c], sp[1][c]);
    case Op::Mix:
      return dr::lerp(sp[0][c], a0, sp[1][c]);
    case Op::Clip:
      return dr::clip(a0, sp[0][c], sp[1][c]);
    default:
      return 0.f;
  }
}

template <typename Float, typename Spectrum>
typename MtlxEvaluator<Float, Spectrum>::Val
MtlxEvaluator<Float, Spectrum>::eval(const SurfaceInteraction3f& si,
                                     Mask /*active*/) const {
  absl::InlinedVector<Val, 64> mem(m_prog.num_tmps + m_prog.stack_size);
  Val* tmps = mem.data();
  Val* sp = tmps + m_prog.num_tmps;

  for (Inst inst : m_prog.code) {
    uint16_t imm = inst.imm;
    if (int n = elementwise_arity(inst.op)) {
      sp -= n - 1;
      Val res(0.f);
      for (int c = 0; c < inst.channels; ++c) {
        res[c] = elementwise(inst.op, sp, c);
      }
      sp[-1] = inst.channels == 1 ? Val(res.x()) : res;
      continue;
    }
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
        break;
      case Op::GeomUV:
        *sp++ = Val(si.uv.x(), si.uv.y(), 0.f, 0.f);
        break;
      case Op::Rgb:
      case Op::Vec2:
        sp -= inst.op == Op::Rgb ? 2 : 1;
        sp[-1] = Val(sp[-1].x(), sp[0].x(),
                     inst.op == Op::Rgb ? sp[1].x() : 0.f, 0.f);
        break;
      case Op::Vec4:
        --sp;
        sp[-1].w() = sp[0].x();
        break;
      case Op::Chan:
        sp[-1] = Val(sp[-1].entry(imm));
        break;
      case Op::Dot:
        --sp;
        sp[-1] = Val(dr::dot(dr::head<3>(sp[-1]), dr::head<3>(sp[0])));
        break;
      default:  // Element-wise ops are handled above.
        break;
    }
  }

  return sp[-1];
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
void MtlxEvaluator<Float, Spectrum>::traverse_cb(
    void* payload, const drjit::TraverseVisitor& cb) {
  DR_TRAVERSE_MEMBER(m_opaques)
  if (!cb.state) return;
  auto put_span = [&](const auto& vec, const char* name) {
    size_t n = vec.size();
    drjit::detail::traverse_state(payload, cb, &n, sizeof(n), name);
    if (n > 0) {
      drjit::detail::traverse_state(payload, cb, vec.data(), n * sizeof(vec[0]),
                                    name);
    }
  };
  put_span(m_prog.code, "code");
  put_span(m_prog.consts, "consts");
  DR_TRAVERSE_STATE(m_prog.result_uniform)
  DR_TRAVERSE_STATE(m_prog.unbounded)
}

namespace {

// Device-resident `data/srgb.coeff` lookup tables shared via a weak cache so
// `MtlxExprTexture::traverse_cb()` can expose `data` and `scale` to
// `@dr.freeze` and `FrozenRender` (`mitsuba::SRGBModel` keeps its buffers in a
// translation-unit static).
template <typename Float32>
struct SrgbLut {
  uint32_t res = 0;
  Float32 data;
  Float32 scale;
};

template <typename Float32>
std::shared_ptr<SrgbLut<Float32>> get_srgb_lut() {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<std::weak_ptr<SrgbLut<Float32>>> cached;
  absl::MutexLock lock(*mu);
  if (auto lut = cached->lock()) return lut;

  fs::path path = file_resolver()->resolve("data/srgb.coeff");
  std::ifstream f(path, std::ios::binary);
  char header[4] = {};
  uint32_t res = 0;
  if (!f.read(header, 4) || std::memcmp(header, "SPEC", 4) != 0 ||
      !f.read(reinterpret_cast<char*>(&res), sizeof(res)) || res < 2) {
    Throw("Could not load sRGB-to-spectrum upsampling model ('%s')",
          path.string());
  }
  std::vector<float> host_scale(res);
  std::vector<float> host_data(size_t(res) * res * res * 9);
  if (!f.read(reinterpret_cast<char*>(host_scale.data()),
              host_scale.size() * sizeof(float)) ||
      !f.read(reinterpret_cast<char*>(host_data.data()),
              host_data.size() * sizeof(float))) {
    Throw("Truncated sRGB-to-spectrum upsampling model ('%s')", path.string());
  }

  auto lut = std::make_shared<SrgbLut<Float32>>();
  lut->res = res;
  lut->scale = dr::load<Float32>(host_scale.data(), host_scale.size());
  lut->data = dr::load<Float32>(host_data.data(), host_data.size());
  *cached = lut;
  return lut;
}

// Mirrors `mitsuba::SRGBModel::fetch_jit()` (`mitsuba/src/render/srgb.cpp`),
// reading from the traversable `lut` of `MtlxExprTexture`.
template <typename Float, typename Spectrum,
          typename Float32 = dr::float32_array_t<dr::detached_t<Float>>>
MI_INLINE dr::Array<Float, 3> fetch_srgb(const Color<Float, 3>& color,
                                         const SrgbLut<Float32>* lut) {
  if constexpr (!dr::is_jit_v<Float>) {
    (void)lut;
    return SRGBModel<Float, Spectrum>::fetch(color);
  } else {
    using Array3f = dr::Array<Float, 3>;
    using UInt32 = dr::uint32_array_t<Float>;
    using Mask = dr::mask_t<Float>;

    uint32_t res = lut->res;
    float res_f = static_cast<float>(res);
    Color<Float, 3> c = dr::clip(color, 0.f, 1.f);

    Mask g_ge_r = c.g() >= c.r();
    Float rg = dr::select(g_ge_r, c.g(), c.r());
    Mask b_largest = c.b() >= rg;

    UInt32 i = dr::select(b_largest, 2u, dr::select(g_ge_r, 1u, 0u));
    Float z = dr::select(b_largest, c.b(), rg),
          x = dr::select(b_largest, c.r(), dr::select(g_ge_r, c.b(), c.g())),
          y = dr::select(b_largest, c.g(), dr::select(g_ge_r, c.r(), c.b()));

    Float scale = (res_f - 1.f) / dr::maximum(z, 1e-9f);
    x *= scale;
    y *= scale;

    auto scale_at = [&](const UInt32& idx) {
      return Float(dr::gather<Float32>(lut->scale, dr::detach<false>(idx)));
    };
    UInt32 zi =
        dr::binary_search<UInt32>(
            1, res - 1, [&](const UInt32& idx) { return scale_at(idx) <= z; }) -
        1;
    Float z0 = scale_at(zi), z1 = scale_at(zi + 1);

    UInt32 xi = dr::minimum(UInt32(x), res - 2),
           yi = dr::minimum(UInt32(y), res - 2);
    Float wx = x - Float(xi), wy = y - Float(yi), wz = (z - z0) / (z1 - z0);

    UInt32 index = ((i * res + zi) * res + yi) * res + xi, dy = res,
           dz = res * res;
    auto fetch = [&](const UInt32& idx) {
      return Array3f(
          dr::gather<dr::Array<Float32, 3>>(lut->data, dr::detach<false>(idx)));
    };
    auto lerp_x = [&](const UInt32& idx) {
      return dr::lerp(fetch(idx), fetch(idx + 1), wx);
    };
    Array3f result =
        dr::lerp(dr::lerp(lerp_x(index), lerp_x(index + dy), wy),
                 dr::lerp(lerp_x(index + dz), lerp_x(index + dz + dy), wy), wz);

    Float v = c.r();
    Mask mono = (c.g() == v) && (c.b() == v);
    Float mono_c2 = dr::select(
        v <= 0.f, -8192.f,
        dr::select(v >= 1.f, 8192.f, (v - .5f) * dr::rsqrt(v * (1.f - v))));
    Array3f mono_coeff(0.f, 0.f, mono_c2);
    if constexpr (dr::is_diff_v<Float>) {
      mono_coeff = dr::replace_grad(mono_coeff, result);
    }
    return dr::select(mono, mono_coeff, result);
  }
}

// Spectral upsampling of an RGB color computed at render time. `SRGBModel`
// upsamples colors in [0, 1], so an unbounded color `c` maps to
// `2 * max(c) * S(c / (2 * max(c)))`, like `D65Spectrum::normalization` for
// constant colors: the brightest channel sits at 0.5 inside the smooth
// interior of the model, and the mapping remains homogeneous of degree 1 for
// unbounded HDR emission.
template <typename Float, typename Spectrum, typename Float32>
MI_INLINE unpolarized_spectrum_t<Spectrum> upsample_rgb(
    const Color<Float, 3>& c, bool unbounded,
    const wavelength_t<Spectrum>& wavelengths, const SrgbLut<Float32>* lut) {
  if (!unbounded) {
    return srgb_model_eval<unpolarized_spectrum_t<Spectrum>>(
        fetch_srgb<Float, Spectrum>(c, lut), wavelengths);
  }
  Color<Float, 3> rgb = dr::maximum(c, 0.f);
  Float z = dr::max(rgb) * 2.f;
  auto active = z > 0.f;
  return srgb_model_eval<unpolarized_spectrum_t<Spectrum>>(
             fetch_srgb<Float, Spectrum>(
                 dr::select(active, rgb / dr::select(active, z, 1.f),
                            Color<Float, 3>(0.5f)),
                 lut),
             wavelengths) *
         z;
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

 * - param0, param1, ...
   - |float|
   - Differentiable opaque scalar parameters exposed through traversal.
   - |exposed|, |differentiable|

Evaluates a compiled MaterialX expression DAG over surface interactions.

*/
template <typename Float, typename Spectrum>
class MtlxExprTexture final : public Texture<Float, Spectrum> {
 public:
  MI_IMPORT_TYPES(Texture)
  using SrgbFloat32 = dr::float32_array_t<dr::detached_t<Float>>;

  explicit MtlxExprTexture(const Properties& props) : Texture(props) {
    m_eval.compile(props);
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      if (!m_eval.program().result_uniform) {
        m_srgb_lut = get_srgb_lut<SrgbFloat32>();
      }
    }
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
        return UnpolarizedSpectrum(prog.unbounded ? dr::maximum(c.x(), 0.f)
                                                  : dr::clip(c.x(), 0.f, 1.f));
      }
      return upsample_rgb<Float, Spectrum>(c, prog.unbounded, si.wavelengths,
                                           m_srgb_lut.get());
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
    if constexpr (is_spectral_v<Spectrum> && dr::is_jit_v<Float>) {
      if (m_srgb_lut) {
        DR_TRAVERSE_MEMBER(m_srgb_lut->scale)
        DR_TRAVERSE_MEMBER(m_srgb_lut->data)
      }
    }
  }

 private:
  MtlxEvaluator<Float, Spectrum> m_eval;
  std::shared_ptr<SrgbLut<SrgbFloat32>> m_srgb_lut;
};

}  // namespace

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
