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

#pragma once

#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <mitsuba/core/math.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/texture.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

inline constexpr size_t kMaxImm = std::numeric_limits<uint16_t>::max();
// Operand count of `Op::Remap`, the element-wise op with the most operands.
inline constexpr int kMaxElementwiseArity = 5;

// Constructs `n` objects in caller-provided storage and destroys them at scope
// exit. The VM allocates arrays whose sizes are only known at trace time with
// `alloca()`, like Dr.Jit's texture code: they run for every query in scalar
// variants, where heap allocations would be comparatively expensive.
template <typename T>
class ScopedArray {
 public:
  ScopedArray(void* ptr, size_t n) : m_ptr(static_cast<T*>(ptr)), m_size(n) {
    std::uninitialized_default_construct_n(m_ptr, m_size);
  }
  ScopedArray(void* ptr, size_t n, const T& init)
      : m_ptr(static_cast<T*>(ptr)), m_size(n) {
    std::uninitialized_fill_n(m_ptr, m_size, init);
  }
  ~ScopedArray() { std::destroy_n(m_ptr, m_size); }
  operator T*() const { return m_ptr; }
  ScopedArray(const ScopedArray&) = delete;
  ScopedArray& operator=(const ScopedArray&) = delete;

 private:
  T* m_ptr;
  size_t m_size;
};

template <typename Float>
MI_INLINE Vector<Float, 3> safe_normalize(
    const Vector<Float, 3>& v,
    const Vector<Float, 3>& fallback = Vector<Float, 3>(0.0f),
    dr::scalar_t<Float> min_len2 = 1e-20f) {
  Float len2 = dr::squared_norm(v);
  auto valid = len2 > min_len2;
  return dr::select(valid, v * dr::rsqrt(dr::select(valid, len2, 1.0f)),
                    fallback);
}

template <typename T>
MI_INLINE T mx_safe_atan2(const T& y, const T& x) {
  auto zero = (y == T(0.0f)) & (x == T(0.0f));
  return dr::select(zero, T(0.0f), dr::atan2(y, dr::select(zero, T(1.0f), x)));
}

template <typename Mask>
MI_INLINE bool mx_may_be_true(const Mask& m) {
  if constexpr (dr::is_jit_v<Mask>) {
    return m.index() && m.size() > 0 &&
           (m.state() != VarState::Literal || m.entry(0));
  } else {
    return dr::any(m);
  }
}

// Bit flags in `Inst::imm` for `CellNoise`, `Noise`, `Worley` and `Fractal`.
inline constexpr uint16_t kNoise3D = 1, kNoiseVec3 = 2;

// Note: `elementwise_arity()`, `result_channels()`, and
// `MtlxVM::is_spatially_varying()` rely on the contiguous enum ranges below.
enum class Op : uint8_t {
  Const,
  // Opaque parameter, which is not baked into compiled kernels.
  Param,
  LoadTmp,
  StoreTmp,
  SampleInput,

  // Spatially varying ops (`SampleInputUV..Fractal`): re-parameterized input
  // sampling, named shape attributes (`Attr1`/`Attr3`, evaluating to a default
  // expression on shapes lacking the attribute), geometric intrinsics (object
  // space is passed via mesh attributes), and procedural noise (`imm` is a
  // bitmask of `kNoise3D` and `kNoiseVec3`).
  SampleInputUV,
  Attr1,
  Attr3,
  GeomUV,
  GeomPosWorld,
  GeomNormalWorld,
  GeomTangentWorld,
  GeomBitangentWorld,
  GeomIncidentDir,
  CellNoise,
  Noise,
  Worley,
  Fractal,

  // 1-ary element-wise ops (`Neg..Sign`).
  Neg,
  Abs,
  Sin,
  Cos,
  Tan,
  Exp,
  Round,
  Floor,
  Ceil,
  Sqrt,
  Log,
  Asin,
  Acos,
  Fract,
  Sign,

  // 2-ary element-wise ops (`Add..ArtisticIorK`).
  Add,
  Sub,
  Mul,
  Div,
  Lt,
  Le,
  Eq,
  Ne,
  Min,
  Max,
  Pow,
  Atan2,
  Modulo,
  ArtisticIorN,
  ArtisticIorK,

  // 3-ary and 5-ary element-wise ops (`Select..Smoothstep`, `Remap`).
  Select,
  Mix,
  Clip,
  Smoothstep,
  Remap,

  // Color & vector ops (`Dot`, `Length` and `Normalize` take `imm = 2` for 2D;
  // `NormalMap` takes `imm = 1` when a bitangent is passed).
  Chan,
  Dot,
  Length,
  Vec2,
  Rotate2D,
  RoughnessAnisotropy,
  RoughnessDual,
  Normalize,
  Rgb,
  Cross,
  Rotate3D,
  NormalMap,
  RgbToHsv,
  HsvToRgb,
  Blackbody
};

struct Inst {
  Op op;
  // For element-wise ops: active channel count (1, 2, or 3). For `SampleInput`
  // and `SampleInputUV`: 1 for single-channel reads, 3 otherwise. For `Attr1`
  // and `Attr3`: 1 if the default is zero. 0 for all other ops.
  uint8_t channels = 0;
  uint16_t imm = 0;

  bool operator==(const Inst& o) const {
    return op == o.op && channels == o.channels && imm == o.imm;
  }
  bool operator!=(const Inst& o) const { return !(*this == o); }

  template <typename H>
  friend H AbslHashValue(H h, Inst i) {
    return H::combine(std::move(h), i.op, i.channels, i.imm);
  }
};
static_assert(sizeof(Inst) == 4);

// Returns the operand count (at most `kMaxElementwiseArity`) of ops that act
// independently on each channel, and 0 for all other ops.
constexpr int elementwise_arity(Op op) {
  if (op >= Op::Neg && op <= Op::Sign) return 1;
  if (op >= Op::Add && op <= Op::ArtisticIorK) return 2;
  if (op >= Op::Select && op <= Op::Smoothstep) return 3;
  return op == Op::Remap ? kMaxElementwiseArity : 0;
}

// Returns the active channel count (1, 2, or 3) of non-elementwise ops.
constexpr uint8_t result_channels(Op op, uint16_t imm = 0) {
  if (op <= Op::Param || (op >= Op::Chan && op <= Op::Length)) return 1;
  if (op >= Op::CellNoise && op <= Op::Fractal) {
    return (imm & kNoiseVec3) ? 3 : 1;
  }
  if (op == Op::GeomUV || (op >= Op::Vec2 && op <= Op::RoughnessDual) ||
      (op == Op::Normalize && imm == 2)) {
    return 2;
  }
  return 3;
}

struct Expr;

// Variant-independent compiled MaterialX program passed via `Properties`.
//
// Programs are built by appending temporaries (`add_tmp()`) followed by the
// result (`finish()`). Temporaries are numbered in order of appearance. A
// program has at most `kMaxImm` constants, parameters and temporaries each, and
// `stack_size` bounds the VM stack depth.
struct MtlxProgram {
  std::vector<Inst> code;
  std::vector<double> consts = {0.0, 1.0};
  std::vector<double> params;
  std::vector<std::string> attr_names;
  std::vector<uint8_t> tmp_channels;
  bool result_uniform = false;
  uint32_t num_tmps = 0;
  uint32_t stack_size = 1;

  Expr c(double value);
  Expr c_rgb(double r, double g, double b);
  Expr c_vec2(double x, double y);
  Expr param(double value);
  Expr attr(bool three, std::string_view name, const Expr& def);
  int add_tmp(const Expr& val);
  Expr tmp(int k) const;
  bool is_tmp_uniform(int k) const { return tmp_channels[k] == 1; }
  MtlxProgram& finish(const Expr& result);

  bool same_structure(const MtlxProgram& o) const {
    return code == o.code && consts == o.consts && attr_names == o.attr_names &&
           result_uniform == o.result_uniform &&
           params.size() == o.params.size();
  }
};

// Result of `CopyMaterialXParameters`.
enum class MaterialXCopy {
  // The objects differ in more than parameter values; nothing was copied.
  kIncompatible,
  // The objects are identical.
  kUnchanged,
  // The parameter values differed and were copied.
  kCopied,
};

// Evaluates an `MtlxProgram` for the variant `Float`/`Spectrum`, given its
// texture inputs.
template <typename Float, typename Spectrum>
class MtlxVM {
 public:
  MI_IMPORT_TYPES(Texture)
  using Val = Vector3f;

  void compile(const Properties& props);

  uint32_t tmp_count() const { return m_prog.num_tmps; }
  const MtlxProgram& program() const { return m_prog; }
  MtlxProgram& program() { return m_prog; }
  std::vector<ref<Texture>>& inputs() { return m_inputs; }

  // Opaque parameters, in order of appearance in `program().params`.
  std::vector<Float>& params() { return m_params; }

  // Reports texture inputs ("in<i>") and opaque parameters ("param<i>").
  void traverse(TraversalCallback* cb);

  void parameters_changed();

  // Returns the indented code size and texture inputs, one per line, for the
  // `to_string()` methods of the plugins.
  std::string to_string() const;

  // Returns true if the program reads geometric intrinsics, evaluates noise,
  // or samples a spatially varying (or re-parameterized) input.
  bool is_spatially_varying() const;

  // Evaluates the program and returns its result.
  Val eval(const SurfaceInteraction3f& si, Mask active) const;

  // Returns the luminance of 3-channel results, like Mitsuba's `bitmap`.
  Float eval_1(const SurfaceInteraction3f& si, Mask active) const;
  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const;

  static MaterialXCopy copy_parameters(Object* dst, Object* src);

 private:
  template <typename T>
  static T elementwise(Op op, const T* a);

  static bool is_single_channel_bitmap(Texture* texture);

  MtlxProgram m_prog;
  std::vector<ref<Texture>> m_inputs;
  // Opaque parameters, which are kernel inputs rather than literals.
  std::vector<Float> m_params;
  // Per input: whether it should be evaluated with `eval_1()` (monochromatic
  // variants or single-channel bitmaps).
  std::vector<bool> m_input_mono;
};

// Base mixin for MaterialX plugins (`MtlxExprTexture`), providing direct access
// to their `MtlxVM`.
template <typename Float, typename Spectrum>
struct MtlxObject {
  MtlxVM<Float, Spectrum>& vm() { return m_vm; }
  const MtlxVM<Float, Spectrum>& vm() const { return m_vm; }

 protected:
  MtlxVM<Float, Spectrum> m_vm;
};

// Copies the opaque parameter values of the MaterialX texture `src` into `dst`
// if both have the same program structure, i.e. differ only in parameter
// values. Keeping `dst` preserves its Dr.Jit registry ID, which is part of the
// compiled kernels, so that parameter edits reuse them. A `d65` texture
// wrapping a MaterialX texture is compared by its nested texture.
template <typename Float, typename Spectrum>
MaterialXCopy CopyMaterialXParameters(Object* dst, Object* src) {
  return MtlxVM<Float, Spectrum>::copy_parameters(dst, src);
}

// Registers the `materialx_expr` texture with Mitsuba's plugin manager.
// Idempotent and thread-safe. Must be called before creating a
// `materialx_expr` texture; deliberately not done in a static initializer to
// avoid depending on the initialization order of Mitsuba's `PluginManager`
// singleton.
void RegisterMaterialXExprTexture();

// Homogeneous RGB-to-spectrum upsampling for colors computed at render time.
//
// A color `c` maps to `max(c) * S(c / max(c))`, where `S` is the smooth sigmoid
// spectrum from Mitsuba's `SRGBModel`. This supports unbounded colors (e.g.
// emission) and makes the conversion commute with scalar factors.
template <typename Float, typename Spectrum>
MI_INLINE unpolarized_spectrum_t<Spectrum> upsample_rgb(
    const Color<Float, 3>& c, const wavelength_t<Spectrum>& wavelengths) {
  Color<Float, 3> rgb = dr::maximum(c, 0.f);
  Float z = dr::max(rgb);
  return srgb_model_eval<unpolarized_spectrum_t<Spectrum>>(
             SRGBModel<Float, Spectrum>::fetch(rgb /
                                               dr::select(z > 0.f, z, 1.f)),
             wavelengths) *
         z;
}

}  // namespace mtlx
NAMESPACE_END(mitsuba)
