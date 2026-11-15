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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

inline constexpr size_t kMaxImm = std::numeric_limits<uint16_t>::max();
// Matches MaterialX's `M_FLOAT_EPS`.
inline constexpr double kMtlxFloatEps = 1e-8;

// Note: `elementwise_arity()`, `result_channels()`, and
// `MtlxEvaluator::is_spatially_varying()` rely on the contiguous enum ranges
// below.
enum class Op : uint8_t {
  Const,
  // Opaque scalar (`dr::opaque`), not baked as a literal into compiled kernels.
  Opaque,
  LoadTmp,
  StoreTmp,
  SampleInput,

  // Spatially varying ops (`SampleInputUV..GeomIncidentDir`): re-parameterized
  // input sampling, named shape attributes (`Attr`, evaluating to a default
  // expression on shapes lacking the attribute), and geometric intrinsics
  // (object space is passed via mesh attributes).
  SampleInputUV,
  Attr,
  GeomUV,
  GeomPosWorld,
  GeomNormalWorld,
  GeomTangentWorld,
  GeomBitangentWorld,
  GeomIncidentDir,

  // 1-ary element-wise ops (`Neg..Fract`).
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

  // 2-ary element-wise ops (`Add..Atan2`).
  Add,
  Sub,
  Mul,
  Div,
  Lt,
  Le,
  Eq,
  Min,
  Max,
  Pow,
  Atan2,

  // 3-ary element-wise ops (`Select..Clip`).
  Select,
  Mix,
  Clip,

  // Color & vector ops (`Dot` and `Normalize` take `imm = 2` for 2D and
  // `imm = 4` for 4D; `Vec4` appends a fourth channel to a 3-channel value).
  Chan,
  Dot,
  Vec2,
  Rotate2D,
  Normalize,
  Rgb,
  Cross,
  Rotate3D,
  NormalMap,
  RgbToHsv,
  HsvToRgb,
  Blackbody,
  Vec4,

  // Matrix ops on 3x3 or 4x4 matrices (`channels` is the size), which are
  // passed as rows. `MatTranspose`, `MatInverse` and `MatMul` store the rows of
  // their result in consecutive temporaries, starting at `imm`.
  MatDet,
  MatTranspose,
  MatInverse,
  MatMul
};

struct Inst {
  Op op;
  // For element-wise ops: active channel count (1 to 4). For `SampleInput`,
  // `SampleInputUV`, and `Attr`: 1 for single-channel reads, 3 otherwise. For
  // matrix ops: the matrix size. 0 for all other ops.
  uint8_t channels = 0;
  uint16_t imm = 0;

  bool operator==(const Inst& o) const {
    return op == o.op && channels == o.channels && imm == o.imm;
  }

  template <typename H>
  friend H AbslHashValue(H h, Inst i) {
    return H::combine(std::move(h), i.op, i.channels, i.imm);
  }
};
static_assert(sizeof(Inst) == 4);
static_assert(std::has_unique_object_representations_v<Inst>);

// Returns the operand count of ops that act independently on each channel, and
// 0 for all other ops.
constexpr int elementwise_arity(Op op) {
  if (op >= Op::Neg && op <= Op::Fract) return 1;
  if (op >= Op::Add && op <= Op::Atan2) return 2;
  if (op >= Op::Select && op <= Op::Clip) return 3;
  return 0;
}

// Returns the active channel count (1 to 4) of non-elementwise ops.
constexpr uint8_t result_channels(Op op, uint16_t imm = 0) {
  if (op <= Op::Opaque || op == Op::Chan || op == Op::Dot || op == Op::MatDet) {
    return 1;
  }
  if (op == Op::Vec4 || (op == Op::Normalize && imm == 4)) return 4;
  if (op == Op::GeomUV || op == Op::Vec2 || op == Op::Rotate2D ||
      (op == Op::Normalize && imm == 2)) {
    return 2;
  }
  return 3;
}

// Variant-independent compiled MaterialX program passed via `Properties`.
//
// A program has at most `kMaxImm` constants, opaque values and temporaries
// each, and `stack_size` bounds the evaluation stack depth.
struct MtlxProgram {
  std::vector<Inst> code;
  std::vector<double> consts = {0.0, 1.0};
  std::vector<double> opaques;
  std::vector<std::string> attr_names;
  bool result_uniform = false;
  bool unbounded = false;
  uint32_t num_tmps = 0;
  uint32_t stack_size = 1;
};

// Evaluates an `MtlxProgram` for the variant `Float`/`Spectrum`, given its
// texture inputs.
template <typename Float, typename Spectrum>
class MtlxEvaluator {
 public:
  MI_IMPORT_TYPES(Texture)
  // Values have up to four channels. A 1-channel value is replicated to all
  // of them, and the unused channels of other values are unspecified.
  using Val = Vector<Float, 4>;

  void compile(const Properties& props);

  const MtlxProgram& program() const { return m_prog; }

  // Reports texture inputs ("in<i>") and opaque parameters ("param<i>").
  void traverse(TraversalCallback* cb);

  void parameters_changed();

  // Returns the indented code size and texture inputs, one per line, for
  // `MtlxExprTexture::to_string()`.
  std::string to_string() const;

  // Returns true if the program reads geometric intrinsics or samples a
  // spatially varying (or re-parameterized) input.
  bool is_spatially_varying() const;

  // Evaluates the program and returns its result.
  Val eval(const SurfaceInteraction3f& si, Mask active) const;

  // Returns the luminance of 3-channel results, like Mitsuba's `bitmap`.
  Float eval_1(const SurfaceInteraction3f& si, Mask active) const;
  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const;

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb);

 private:
  static Float elementwise(Op op, const Val* sp, int c);

  MtlxProgram m_prog;
  std::vector<ref<Texture>> m_inputs;
  // Opaque scalars (`dr::opaque`), passed as kernel inputs rather than
  // literals.
  std::vector<Float> m_opaques;
};

// Registers the `materialx_expr` texture with Mitsuba's plugin manager.
// Idempotent and thread-safe. Must be called before creating a
// `materialx_expr` texture; deliberately not done in a static initializer to
// avoid depending on the initialization order of Mitsuba's `PluginManager`
// singleton.
void RegisterMaterialXExprTexture();

}  // namespace mtlx
NAMESPACE_END(mitsuba)
