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

#include <drjit/traversable_base.h>

namespace mitsuba::mtlx {

inline constexpr size_t kMaxImm = std::numeric_limits<uint16_t>::max();
// Matches MaterialX's `M_FLOAT_EPS`.
inline constexpr double kMtlxFloatEps = 1e-8;

// Bit flags in `Inst::imm` for `CellNoise`, `Noise`, `Worley` and `Fractal`.
// For `Fractal`, the bits from `kFractalOctavesShift` hold the constant octave
// count, or `kFractalDynamicOctaves` when octaves are passed as the last
// operand.
inline constexpr uint16_t kNoise3D = 1, kNoiseVec3 = 2;
inline constexpr int kMaxFractalOctaves = 16;
inline constexpr int kFractalOctavesShift = 2;
inline constexpr uint16_t kFractalDynamicOctaves = kMaxFractalOctaves + 1;

constexpr uint16_t pack_noise_imm(bool is_3d, bool is_vec3,
                                  uint16_t octaves = 0) {
  return static_cast<uint16_t>((is_3d ? kNoise3D : 0) |
                               (is_vec3 ? kNoiseVec3 : 0) |
                               (octaves << kFractalOctavesShift));
}
constexpr bool noise_is_3d(uint16_t imm) { return (imm & kNoise3D) != 0; }
constexpr bool noise_is_vec3(uint16_t imm) { return (imm & kNoiseVec3) != 0; }
constexpr uint16_t fractal_octaves(uint16_t imm) {
  return static_cast<uint16_t>(imm >> kFractalOctavesShift);
}

enum class Op : uint8_t {
  Const,
  // Opaque scalar (`dr::opaque`), not baked as a literal into compiled kernels.
  Opaque,
  LoadTmp,
  StoreTmp,
  SampleInput,

  // Spatially varying ops: re-parameterized input sampling, named shape
  // attributes (`Attr`, evaluating to a default expression on shapes lacking
  // the attribute), and geometric intrinsics (object space is passed via mesh
  // attributes).
  SampleInputUV,
  Attr,
  GeomUV,
  GeomPosWorld,
  GeomNormalWorld,
  GeomTangentWorld,
  GeomBitangentWorld,
  GeomIncidentDir,

  // 1-ary element-wise ops.
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

  // 2-ary element-wise ops.
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

  // 3-ary element-wise ops.
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

  // Procedural noise ops (see `pack_noise_imm`).
  CellNoise,
  Noise,
  Worley,
  Fractal,

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
  switch (op) {
    case Op::Neg:
    case Op::Abs:
    case Op::Sin:
    case Op::Cos:
    case Op::Tan:
    case Op::Exp:
    case Op::Round:
    case Op::Floor:
    case Op::Ceil:
    case Op::Sqrt:
    case Op::Log:
    case Op::Asin:
    case Op::Acos:
    case Op::Fract:
      return 1;
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Lt:
    case Op::Le:
    case Op::Eq:
    case Op::Min:
    case Op::Max:
    case Op::Pow:
    case Op::Atan2:
      return 2;
    case Op::Select:
    case Op::Mix:
    case Op::Clip:
      return 3;
    case Op::Const:
    case Op::Opaque:
    case Op::LoadTmp:
    case Op::StoreTmp:
    case Op::SampleInput:
    case Op::SampleInputUV:
    case Op::Attr:
    case Op::GeomUV:
    case Op::GeomPosWorld:
    case Op::GeomNormalWorld:
    case Op::GeomTangentWorld:
    case Op::GeomBitangentWorld:
    case Op::GeomIncidentDir:
    case Op::Chan:
    case Op::Dot:
    case Op::Vec2:
    case Op::Rotate2D:
    case Op::Normalize:
    case Op::Rgb:
    case Op::Cross:
    case Op::Rotate3D:
    case Op::NormalMap:
    case Op::RgbToHsv:
    case Op::HsvToRgb:
    case Op::Blackbody:
    case Op::Vec4:
    case Op::CellNoise:
    case Op::Noise:
    case Op::Worley:
    case Op::Fractal:
    case Op::MatDet:
    case Op::MatTranspose:
    case Op::MatInverse:
    case Op::MatMul:
      return 0;
  }
  return 0;
}

// Returns the active channel count (1 to 4) of non-elementwise ops.
constexpr uint8_t result_channels(Op op, uint16_t imm = 0) {
  switch (op) {
    case Op::Const:
    case Op::Opaque:
    case Op::Chan:
    case Op::Dot:
    case Op::MatDet:
      return 1;
    case Op::GeomUV:
    case Op::Vec2:
    case Op::Rotate2D:
      return 2;
    case Op::Vec4:
      return 4;
    case Op::Normalize:
      return imm == 4 ? 4 : (imm == 2 ? 2 : 3);
    case Op::CellNoise:
    case Op::Noise:
    case Op::Worley:
    case Op::Fractal:
      return noise_is_vec3(imm) ? 3 : 1;
    case Op::LoadTmp:
    case Op::StoreTmp:
    case Op::SampleInput:
    case Op::SampleInputUV:
    case Op::Attr:
    case Op::GeomPosWorld:
    case Op::GeomNormalWorld:
    case Op::GeomTangentWorld:
    case Op::GeomBitangentWorld:
    case Op::GeomIncidentDir:
    case Op::Neg:
    case Op::Abs:
    case Op::Sin:
    case Op::Cos:
    case Op::Tan:
    case Op::Exp:
    case Op::Round:
    case Op::Floor:
    case Op::Ceil:
    case Op::Sqrt:
    case Op::Log:
    case Op::Asin:
    case Op::Acos:
    case Op::Fract:
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Lt:
    case Op::Le:
    case Op::Eq:
    case Op::Min:
    case Op::Max:
    case Op::Pow:
    case Op::Atan2:
    case Op::Select:
    case Op::Mix:
    case Op::Clip:
    case Op::Rgb:
    case Op::Cross:
    case Op::Rotate3D:
    case Op::NormalMap:
    case Op::RgbToHsv:
    case Op::HsvToRgb:
    case Op::Blackbody:
    case Op::MatTranspose:
    case Op::MatInverse:
    case Op::MatMul:
      return 3;
  }
  return 3;
}

// Returns true if `op` unconditionally depends on the surface interaction.
// (`SampleInput` depends on whether its texture input is spatially varying.)
constexpr bool is_spatially_varying_op(Op op) {
  switch (op) {
    case Op::SampleInputUV:
    case Op::Attr:
    case Op::GeomUV:
    case Op::GeomPosWorld:
    case Op::GeomNormalWorld:
    case Op::GeomTangentWorld:
    case Op::GeomBitangentWorld:
    case Op::GeomIncidentDir:
      return true;
    case Op::Const:
    case Op::Opaque:
    case Op::LoadTmp:
    case Op::StoreTmp:
    case Op::SampleInput:
    case Op::Neg:
    case Op::Abs:
    case Op::Sin:
    case Op::Cos:
    case Op::Tan:
    case Op::Exp:
    case Op::Round:
    case Op::Floor:
    case Op::Ceil:
    case Op::Sqrt:
    case Op::Log:
    case Op::Asin:
    case Op::Acos:
    case Op::Fract:
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Lt:
    case Op::Le:
    case Op::Eq:
    case Op::Min:
    case Op::Max:
    case Op::Pow:
    case Op::Atan2:
    case Op::Select:
    case Op::Mix:
    case Op::Clip:
    case Op::Chan:
    case Op::Dot:
    case Op::Vec2:
    case Op::Rotate2D:
    case Op::Normalize:
    case Op::Rgb:
    case Op::Cross:
    case Op::Rotate3D:
    case Op::NormalMap:
    case Op::RgbToHsv:
    case Op::HsvToRgb:
    case Op::Blackbody:
    case Op::Vec4:
    case Op::CellNoise:
    case Op::Noise:
    case Op::Worley:
    case Op::Fractal:
    case Op::MatDet:
    case Op::MatTranspose:
    case Op::MatInverse:
    case Op::MatMul:
      return false;
  }
  return false;
}

// Dielectric `ClosureNode::Dielectric::mode` scatter mode values
// (`kScatterModeMask`) and flags.
inline constexpr int kScatterR = 0, kScatterT = 1, kScatterRT = 2;
inline constexpr int kScatterModeMask = 3, kScatterThinWalled = 4,
                     kScatterTwoSided = 8;

inline constexpr int kSheenContyKulla = 0, kSheenZeltner = 1;

enum class ClosureOp : int32_t {
  Null,
  OrenNayarDiffuse,
  BurleyDiffuse,
  Sheen,
  Translucent,
  Dielectric,
  Conductor,
  Layer,
  Mix,
  Add,
  Multiply
};

constexpr bool is_leaf(ClosureOp op) {
  return op > ClosureOp::Null && op < ClosureOp::Layer;
}
constexpr bool is_combinator(ClosureOp op) { return op >= ClosureOp::Layer; }

// Tagged POD closure node stored in `MtlxProgram::closures`. Unless noted as a
// literal or closure index, each field is a VM temporary slot (-1 if absent).
struct ClosureNode {
  // Shared by `OrenNayarDiffuse` (`mode = energy_comp`), `BurleyDiffuse`, and
  // `Sheen` (`mode = kSheen*`).
  struct Diffuse {
    int weight, color, roughness, mode, normal;
    int _pad[2] = {};
  };
  struct Translucent {
    int weight, color, normal;
    int _pad[4] = {};
  };
  struct Dielectric {
    int weight, tint, roughness, mode, ior, normal, tangent;
  };
  struct Conductor {
    int weight, ior, roughness, extinction, normal, tangent;
    int _pad[1] = {};
  };
  // Combinator args: `first`/`second` index `MtlxProgram::closures` — Layer
  // (top, base), Mix (fg, bg), Add (in1, in2), Multiply (in1, -1); `weight` is
  // the VM temporary slot of the Mix weight or Multiply factor (-1 otherwise).
  struct Comb {
    int first, second, weight;
    int _pad[4] = {};
  };

  ClosureOp op = ClosureOp::Null;
  union {
    int raw[7] = {};
    Diffuse diffuse;
    Translucent translucent;
    Dielectric dielectric;
    Conductor conductor;
    Comb comb;
  };

  ClosureNode() = default;

  static ClosureNode MakeNull() { return ClosureNode(ClosureOp::Null); }

  static ClosureNode MakeDiffuse(int weight, int color, int roughness,
                                 int energy_comp, int normal) {
    ClosureNode n(ClosureOp::OrenNayarDiffuse);
    n.diffuse = {weight, color, roughness, energy_comp, normal};
    return n;
  }

  static ClosureNode MakeBurleyDiffuse(int weight, int color, int roughness,
                                       int normal) {
    ClosureNode n(ClosureOp::BurleyDiffuse);
    n.diffuse = {weight, color, roughness, 0, normal};
    return n;
  }

  static ClosureNode MakeSheen(int weight, int color, int roughness, int mode,
                               int normal) {
    ClosureNode n(ClosureOp::Sheen);
    n.diffuse = {weight, color, roughness, mode, normal};
    return n;
  }

  static ClosureNode MakeTranslucent(int weight, int color, int normal) {
    ClosureNode n(ClosureOp::Translucent);
    n.translucent = {weight, color, normal};
    return n;
  }

  static ClosureNode MakeDielectric(int weight, int tint, int roughness,
                                    int mode, int ior, int normal,
                                    int tangent) {
    ClosureNode n(ClosureOp::Dielectric);
    n.dielectric = {weight, tint, roughness, mode, ior, normal, tangent};
    return n;
  }

  static ClosureNode MakeConductor(int weight, int ior, int roughness,
                                   int extinction, int normal, int tangent) {
    ClosureNode n(ClosureOp::Conductor);
    n.conductor = {weight, ior, roughness, extinction, normal, tangent};
    return n;
  }

  static ClosureNode MakeLayer(int top, int base) {
    ClosureNode n(ClosureOp::Layer);
    n.comb = {top, base, -1};
    return n;
  }

  static ClosureNode MakeMix(int fg, int bg, int weight) {
    ClosureNode n(ClosureOp::Mix);
    n.comb = {fg, bg, weight};
    return n;
  }

  static ClosureNode MakeAdd(int in1, int in2) {
    ClosureNode n(ClosureOp::Add);
    n.comb = {in1, in2, -1};
    return n;
  }

  static ClosureNode MakeMultiply(int in1, int factor) {
    ClosureNode n(ClosureOp::Multiply);
    n.comb = {in1, -1, factor};
    return n;
  }

 private:
  explicit ClosureNode(ClosureOp op) : op(op) {}
};
static_assert(sizeof(ClosureNode) == 32);
static_assert(std::has_unique_object_representations_v<ClosureNode>);

// Variant-independent compiled MaterialX program passed via `Properties`.
//
// A program has at most `kMaxImm` constants, opaque values and temporaries
// each, and `stack_size` bounds the evaluation stack depth.
struct MtlxProgram {
  std::vector<Inst> code;
  std::vector<double> consts = {0.0, 1.0};
  std::vector<double> opaques;
  std::vector<std::string> attr_names;
  // Topological post-order closure DAG; non-empty with root at `back()`.
  std::vector<ClosureNode> closures;
  std::vector<uint8_t> tmp_channels;
  int opacity_slot = -1;
  bool result_uniform = false;
  bool unbounded = false;
  uint32_t stack_size = 1;

  bool is_tmp_uniform(int k) const { return tmp_channels[k] == 1; }

  void traverse_state(void* payload, const drjit::TraverseVisitor& cb) const {
    if (!cb.state) return;
    auto put_span = [&](const auto& vec, const char* name) {
      size_t n = vec.size();
      drjit::detail::traverse_state(payload, cb, &n, sizeof(n), name);
      if (n > 0) {
        drjit::detail::traverse_state(payload, cb, vec.data(),
                                      n * sizeof(vec[0]), name);
      }
    };
    put_span(code, "code");
    put_span(consts, "consts");
    size_t n_attrs = attr_names.size();
    DR_TRAVERSE_STATE(n_attrs)
    for (const std::string& s : attr_names) {
      put_span(s, "attr_names");
    }
    put_span(closures, "closures");
    put_span(tmp_channels, "tmp_channels");
    DR_TRAVERSE_STATE(opacity_slot)
    DR_TRAVERSE_STATE(result_uniform)
    DR_TRAVERSE_STATE(unbounded)
  }
};

// Register the `materialx_expr` texture and `materialx_bsdf` BSDF with
// Mitsuba's plugin manager. Idempotent and thread-safe. Must be called before
// creating the corresponding plugin; deliberately not done in a static
// initializer to avoid depending on the initialization order of Mitsuba's
// `PluginManager` singleton.
void RegisterMaterialXExprTexture();
void RegisterMaterialXBsdf();

}  // namespace mitsuba::mtlx
