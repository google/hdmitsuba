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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/types/span.h>
#include <drjit/array.h>
#include <drjit/traversable_base.h>

namespace mitsuba::mtlx {

using ValueId = uint16_t;
inline constexpr ValueId kInvalidValue = std::numeric_limits<ValueId>::max();
inline constexpr size_t kMaxImm = std::numeric_limits<uint16_t>::max();
inline constexpr uint32_t kNoUniformScalar =
    std::numeric_limits<uint32_t>::max();

// Matches MaterialX's `M_FLOAT_EPS`.
inline constexpr double kMtlxFloatEps = 1e-8;

// `Node::imm` flag for `Op::NormalMap`: takes 2 operands `(in, scale)` and
// evaluates against the shading frame's default world basis.
inline constexpr uint16_t kNormalMapDefaultFrame = 1;

// Bit flags in `Node::imm` for `CellNoise`, `Noise`, `Worley` and `Fractal`.
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
  SampleInput,

  // Spatially varying ops: re-parameterized input sampling, named shape
  // attributes (`Attr`, taking a default operand used on shapes lacking the
  // attribute; `flags` holds the read channel count 1 or 3), geometric
  // intrinsics, and normal perturbation.
  SampleInputUV,
  Attr,
  GeomUV,
  GeomPosWorld,
  GeomNormalWorld,
  GeomTangentWorld,
  GeomBitangentWorld,
  GeomIncidentDir,
  NormalMap,
  HeightToNormal,

  // 1-ary pure element-wise ops.
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

  // 2-ary commutative pure element-wise ops.
  Add,
  Mul,
  Eq,
  Min,
  Max,

  // 2-ary non-commutative pure element-wise ops.
  Sub,
  Div,
  Lt,
  Le,
  Pow,
  Atan2,

  // 3-ary pure element-wise ops.
  Select,
  Mix,
  Clip,
  Fma,

  // Non-element-wise pure color & vector ops (`Dot` and `Normalize` take
  // `imm = 2` for 2D and `imm = 4` for 4D; `Vec4` appends a fourth channel to a
  // 3-channel value).
  Chan,
  Dot,
  Vec2,
  Rotate2D,
  Normalize,
  Rgb,
  Cross,
  Rotate3D,
  RgbToHsv,
  HsvToRgb,
  Blackbody,
  Vec4,

  // Procedural noise ops (see `pack_noise_imm`).
  CellNoise,
  Noise,
  Worley,
  Fractal,

  // Matrix ops on 3x3 or 4x4 matrices (`channels` is the matrix dimension for
  // producers and `MatRow`; rows are passed as operands). `MatTranspose`,
  // `MatInverse`, and `MatMul` write rows to slot `imm`, read via `MatRow`.
  MatDet,
  MatTranspose,
  MatInverse,
  MatMul,
  MatRow
};

constexpr int elementwise_arity(Op op) {
  if (op >= Op::Neg && op <= Op::Fract) return 1;
  if (op >= Op::Add && op <= Op::Atan2) return 2;
  if (op >= Op::Select && op <= Op::Fma) return 3;
  return 0;
}
constexpr bool is_spatially_varying_op(Op op) {
  return op >= Op::SampleInputUV && op <= Op::HeightToNormal;
}
constexpr bool is_pure_op(Op op) { return op >= Op::Neg && op <= Op::Vec4; }
constexpr bool is_commutative_op(Op op) {
  return op >= Op::Add && op <= Op::Max;
}
constexpr bool is_matrix_producer(Op op) {
  return op >= Op::MatTranspose && op <= Op::MatMul;
}

// Flat SSA IR node stored in `MtlxValueProgram::nodes`.
//
// Operands are stored inline in `args[0..arity-1]` when `arity <= 3`, or as a
// slice `extra_args[args[0] .. args[0] + arity)` when `arity > 3`.
struct Node {
  Op op = Op::Const;
  // Output active channel count (1..4); for matrix producers, matrix dimension.
  uint8_t channels = 1;
  uint8_t arity = 0;
  // Per-op flags: read channel count (1 or 3) for `SampleInput`,
  // `SampleInputUV`, and `Attr`; 0 for all other ops.
  uint8_t flags = 0;
  uint16_t imm = 0;
  uint16_t args[3] = {};
};
static_assert(sizeof(Node) == 12);
static_assert(std::has_unique_object_representations_v<Node>);

inline absl::Span<const ValueId> node_args(
    const Node& n, absl::Span<const ValueId> extra_args) {
  return n.arity <= 3 ? absl::MakeConstSpan(n.args, n.arity)
                      : extra_args.subspan(n.args[0], n.arity);
}

// Self-contained SSA slice in `MtlxValueProgram::nodes[begin .. end)` evaluated
// at perturbed surface interactions by `Op::HeightToNormal`. Nodes in a region
// may reference uniform values (`< uniform_end`) or earlier nodes in the same
// region (`[begin .. i)`). Evaluates `root` to `h` when `has_texcoord == 0`, or
// `(h, tc.x, tc.y)` otherwise.
struct Region {
  uint32_t begin = 0;
  uint32_t end = 0;
  ValueId root = kInvalidValue;
  uint16_t mat_count = 0;
  uint16_t has_texcoord = 0;
  uint16_t _pad = 0;
};
static_assert(sizeof(Region) == 16);
static_assert(std::has_unique_object_representations_v<Region>);

// Pure flat SSA value program owned by `MtlxEvaluator` and used standalone by
// `MtlxExprTexture` (emission and displacement) with no BSDF knowledge.
//
// After `OptimizeValueProgram` / `OptimizeBsdfProgram`, `nodes` is laid out as:
//   [0 .. uniform_end)          : spatially-invariant nodes (main + regions)
//   [uniform_end .. opacity_end): varying nodes for the early opacity prefix
//   [opacity_end .. main_end)   : remaining varying main nodes
//   [r.begin .. r.end)          : varying nodes of each `HeightToNormal` region
struct MtlxValueProgram {
  std::vector<Node> nodes;
  std::vector<ValueId> extra_args;
  std::vector<Region> regions;
  std::vector<double> consts = {0.0, 1.0};
  std::vector<double> opaques;
  std::vector<std::string> attr_names;

  // For each `id < uniform_end`, the offset into
  // `MtlxEvaluator::m_uniform_scalars` if `id` is a computed uniform read
  // outside `[0 .. uniform_end)`, or `kNoUniformScalar` otherwise (`Const`,
  // `Opaque`, or internal-only uniform).
  std::vector<uint32_t> uniform_offsets;
  uint32_t uniform_scalar_count = 0;

  uint32_t uniform_end = 0;
  uint32_t opacity_end = 0;
  uint32_t main_end = 0;
  uint32_t uniform_mat_count = 0;
  uint32_t main_mat_count = 0;
  // Root expression for `MtlxExprTexture`, or `kInvalidValue` for BSDFs.
  ValueId result_root = kInvalidValue;

  bool is_value_scalar(ValueId id) const { return nodes[id].channels == 1; }
  bool result_scalar() const {
    return result_root != kInvalidValue && is_value_scalar(result_root);
  }

  // Appends a self-contained region node buffer (with 0-based local ValueIds)
  // as a contiguous `Region` slice after `main_end`.
  void append_region(absl::Span<const Node> reg_nodes,
                     absl::Span<const ValueId> reg_extra_args, ValueId reg_root,
                     uint16_t has_texcoord) {
    uint32_t base = static_cast<uint32_t>(nodes.size());
    uint32_t extra_base = static_cast<uint32_t>(extra_args.size());
    if (base + reg_nodes.size() >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many SSA nodes");
    }
    if (extra_base + reg_extra_args.size() > kMaxImm) {
      throw std::runtime_error("materialx_expr: too many SSA operands");
    }
    for (ValueId a : reg_extra_args) {
      extra_args.push_back(static_cast<ValueId>(a + base));
    }
    for (Node n : reg_nodes) {
      if (n.arity <= 3) {
        for (uint8_t j = 0; j < n.arity; ++j) {
          n.args[j] = static_cast<ValueId>(n.args[j] + base);
        }
      } else {
        n.args[0] = static_cast<uint16_t>(n.args[0] + extra_base);
      }
      nodes.push_back(n);
    }
    Region r;
    r.begin = base;
    r.end = static_cast<uint32_t>(nodes.size());
    r.root = static_cast<ValueId>(base + reg_root);
    r.has_texcoord = has_texcoord;
    regions.push_back(r);
  }

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
    put_span(nodes, "nodes");
    put_span(extra_args, "extra_args");
    put_span(regions, "regions");
    put_span(consts, "consts");
    put_span(uniform_offsets, "uniform_offsets");
    size_t n_attrs = attr_names.size();
    DR_TRAVERSE_STATE(n_attrs)
    for (const std::string& s : attr_names) {
      put_span(s, "attr_names");
    }
    DR_TRAVERSE_STATE(uniform_end)
    DR_TRAVERSE_STATE(opacity_end)
    DR_TRAVERSE_STATE(main_end)
    DR_TRAVERSE_STATE(result_root)
  }
};

// Microfacet `ClosureNode::Microfacet::mode` scatter mode values
// (`kScatterModeMask`) and flags.
inline constexpr uint16_t kScatterR = 0, kScatterT = 1, kScatterRT = 2;
inline constexpr uint16_t kScatterModeMask = 3, kScatterThinWalled = 4,
                          kScatterTwoSided = 8, kScatterRetroreflective = 16;

inline constexpr uint16_t kSheenContyKulla = 0, kSheenZeltner = 1;

enum ClosureFlag : uint8_t {
  kClosureNone = 0,
  // Set by `SimplifyClosures` on closures whose throughput contributes to a
  // `Layer.top` above a scattering base.
  kClosureNeedsTp = 1 << 0,
};

enum class ClosureOp : uint16_t {
  Null,
  OpaqueBlocker,
  OrenNayarDiffuse,
  BurleyDiffuse,
  Sheen,
  Translucent,
  Dielectric,
  Conductor,
  GeneralizedSchlick,
  ChiangHair,
  Layer,
  Mix,
  Add,
  Multiply
};

constexpr bool is_leaf(ClosureOp op) {
  return op > ClosureOp::OpaqueBlocker && op < ClosureOp::Layer;
}
constexpr bool is_combinator(ClosureOp op) { return op >= ClosureOp::Layer; }

// Tagged POD closure node stored in `MtlxBsdfProgram::closures`. Unless noted
// as a literal or closure index, each field is a `ValueId` (`kInvalidValue` if
// absent).
struct ClosureNode {
  // Shared by `OrenNayarDiffuse` (`mode = energy_comp`), `BurleyDiffuse`,
  // `Sheen` (`mode = kSheen*`), and `Translucent` (`roughness =
  // kInvalidValue`).
  struct Diffuse {
    ValueId weight, color, roughness, normal;
    uint16_t mode;
    uint16_t _pad[7] = {};
  };
  // Shared by `Dielectric` (`p = {tint, ior}`), `Conductor` (`p = {ior, ext}`),
  // and `GeneralizedSchlick` (`p = {color0, color82, color90, exponent}`).
  struct Microfacet {
    ValueId weight, roughness, normal, tangent, tf_thickness, tf_ior;
    ValueId p[4];
    uint16_t mode;
    uint16_t _pad = 0;
  };
  struct ChiangHair {
    ValueId tint_r, tint_tt, tint_trt, roughness_r, roughness_tt, roughness_trt,
        ior, absorption, cuticle_angle, normal, curve_direction;
    uint16_t _pad = 0;
  };
  // Combinator args: `first`/`second` index `MtlxBsdfProgram::closures` — Layer
  // (top, base), Mix (fg, bg), Add (in1, in2), Multiply (in1, -1); `weight` is
  // the `ValueId` of the Mix weight or Multiply factor (`kInvalidValue`
  // otherwise).
  struct Comb {
    int32_t first, second;
    ValueId weight;
    uint16_t _pad[7] = {};
  };

  ClosureOp op = ClosureOp::Null;
  uint8_t flags = 0;
  uint8_t _pad = 0;
  union {
    uint16_t raw[12] = {};
    Diffuse diffuse;
    Microfacet microfacet;
    ChiangHair hair;
    Comb comb;
  };

  ClosureNode() = default;

  static ClosureNode MakeNull() { return ClosureNode(ClosureOp::Null); }

  static ClosureNode MakeOpaqueBlocker() {
    return ClosureNode(ClosureOp::OpaqueBlocker);
  }

  static ClosureNode MakeDiffuse(ValueId weight, ValueId color,
                                 ValueId roughness, uint16_t energy_comp,
                                 ValueId normal) {
    ClosureNode n(ClosureOp::OrenNayarDiffuse);
    n.diffuse = {weight, color, roughness, normal, energy_comp};
    return n;
  }

  static ClosureNode MakeBurleyDiffuse(ValueId weight, ValueId color,
                                       ValueId roughness, ValueId normal) {
    ClosureNode n(ClosureOp::BurleyDiffuse);
    n.diffuse = {weight, color, roughness, normal, 0};
    return n;
  }

  static ClosureNode MakeSheen(ValueId weight, ValueId color, ValueId roughness,
                               uint16_t mode, ValueId normal) {
    ClosureNode n(ClosureOp::Sheen);
    n.diffuse = {weight, color, roughness, normal, mode};
    return n;
  }

  static ClosureNode MakeTranslucent(ValueId weight, ValueId color,
                                     ValueId normal) {
    ClosureNode n(ClosureOp::Translucent);
    n.diffuse = {weight, color, kInvalidValue, normal, 0};
    return n;
  }

  static ClosureNode MakeMicrofacet(ClosureOp op, ValueId weight,
                                    ValueId roughness, uint16_t mode,
                                    ValueId normal, ValueId tangent,
                                    ValueId tf_thickness, ValueId tf_ior,
                                    ValueId p0, ValueId p1,
                                    ValueId p2 = kInvalidValue,
                                    ValueId p3 = kInvalidValue) {
    ClosureNode n(op);
    n.microfacet = {weight,       roughness, normal,           tangent,
                    tf_thickness, tf_ior,    {p0, p1, p2, p3}, mode};
    return n;
  }

  static ClosureNode MakeChiangHair(ValueId tint_r, ValueId tint_tt,
                                    ValueId tint_trt, ValueId roughness_r,
                                    ValueId roughness_tt, ValueId roughness_trt,
                                    ValueId ior, ValueId absorption,
                                    ValueId cuticle_angle, ValueId normal,
                                    ValueId curve_direction) {
    ClosureNode n(ClosureOp::ChiangHair);
    n.hair = {tint_r,         tint_tt, tint_trt,   roughness_r,   roughness_tt,
              roughness_trt,  ior,     absorption, cuticle_angle, normal,
              curve_direction};
    return n;
  }

  static ClosureNode MakeLayer(int32_t top, int32_t base) {
    ClosureNode n(ClosureOp::Layer);
    n.comb = {top, base, kInvalidValue};
    return n;
  }

  static ClosureNode MakeMix(int32_t fg, int32_t bg, ValueId weight) {
    ClosureNode n(ClosureOp::Mix);
    n.comb = {fg, bg, weight};
    return n;
  }

  static ClosureNode MakeAdd(int32_t in1, int32_t in2) {
    ClosureNode n(ClosureOp::Add);
    n.comb = {in1, in2, kInvalidValue};
    return n;
  }

  static ClosureNode MakeMultiply(int32_t in1, ValueId factor) {
    ClosureNode n(ClosureOp::Multiply);
    n.comb = {in1, -1, factor};
    return n;
  }

 private:
  explicit ClosureNode(ClosureOp op) : op(op) {}
};
static_assert(sizeof(ClosureNode) == 28);
static_assert(std::has_unique_object_representations_v<ClosureNode>);

// Returns true if a leaf BSDF closure transmits energy to an underlying `layer`
// base (sheen or reflection-only dielectric / generalized Schlick).
constexpr bool leaf_has_layer_throughput(const ClosureNode& c) {
  return c.op == ClosureOp::Sheen ||
         ((c.op == ClosureOp::Dielectric ||
           c.op == ClosureOp::GeneralizedSchlick) &&
          (c.microfacet.mode & kScatterModeMask) == kScatterR);
}

constexpr ValueId leaf_weight_id(const ClosureNode& c) {
  if (c.op >= ClosureOp::OrenNayarDiffuse && c.op <= ClosureOp::Translucent) {
    return c.diffuse.weight;
  }
  if (c.op >= ClosureOp::Dielectric && c.op <= ClosureOp::GeneralizedSchlick) {
    return c.microfacet.weight;
  }
  return kInvalidValue;
}

// Invokes `fn(id)` for every valid (`!= kInvalidValue`) `ValueId` referenced by
// `c`.
template <typename ClosureT, typename Fn>
void for_each_closure_value_id(ClosureT& c, Fn&& fn) {
  auto visit = [&](auto&... ids) {
    ((ids != kInvalidValue ? (void)fn(ids) : (void)0), ...);
  };
  if (c.op >= ClosureOp::OrenNayarDiffuse && c.op <= ClosureOp::Translucent) {
    visit(c.diffuse.weight, c.diffuse.color, c.diffuse.roughness,
          c.diffuse.normal);
  } else if (c.op >= ClosureOp::Dielectric &&
             c.op <= ClosureOp::GeneralizedSchlick) {
    visit(c.microfacet.weight, c.microfacet.roughness, c.microfacet.normal,
          c.microfacet.tangent, c.microfacet.tf_thickness, c.microfacet.tf_ior,
          c.microfacet.p[0], c.microfacet.p[1], c.microfacet.p[2],
          c.microfacet.p[3]);
  } else if (c.op == ClosureOp::ChiangHair) {
    visit(c.hair.tint_r, c.hair.tint_tt, c.hair.tint_trt, c.hair.roughness_r,
          c.hair.roughness_tt, c.hair.roughness_trt, c.hair.ior,
          c.hair.absorption, c.hair.cuticle_angle, c.hair.normal,
          c.hair.curve_direction);
  } else if (c.op == ClosureOp::Mix || c.op == ClosureOp::Multiply) {
    visit(c.comb.weight);
  }
}

// Compiled MaterialX BSDF program embedding a value program plus a closure DAG,
// optional cutout opacity root, and material-wide thin-walled flag.
struct MtlxBsdfProgram {
  MtlxValueProgram values;
  // Topological post-order closure DAG; non-empty with root at `back()`.
  std::vector<ClosureNode> closures;
  ValueId opacity_root = kInvalidValue;
  bool thin_walled = false;
};

// Register the `materialx_expr` texture and `materialx_bsdf` BSDF with
// Mitsuba's plugin manager. Idempotent and thread-safe. Must be called before
// creating the corresponding plugin; deliberately not done in a static
// initializer to avoid depending on the initialization order of Mitsuba's
// `PluginManager` singleton.
void RegisterMaterialXExprTexture();
void RegisterMaterialXBsdf();

}  // namespace mitsuba::mtlx
