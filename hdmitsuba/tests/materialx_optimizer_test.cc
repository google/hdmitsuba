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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/optimizer.h"
#include "hdmitsuba/materialx/program.h"

namespace mitsuba::mtlx {
namespace {

using ::testing::HasSubstr;

ValueId Tex3(IrBuilder& b, uint16_t idx) {
  return b.Emit(Op::SampleInput, {}, idx, 3, 3);
}

ClosureNode Diffuse(IrBuilder& b, ValueId col, ValueId w = kInvalidValue,
                    ValueId normal = kInvalidValue) {
  return ClosureNode::MakeDiffuse(w == kInvalidValue ? b.One() : w, col,
                                  b.Zero(), 0, normal);
}

ClosureNode Sheen(IrBuilder& b, ValueId col) {
  return ClosureNode::MakeSheen(b.One(), col, b.Const(0.5), kSheenContyKulla,
                                kInvalidValue);
}

void SetMainEnd(MtlxValueProgram& p) {
  p.main_end = static_cast<uint32_t>(p.nodes.size());
}

TEST(MaterialXOptimizerTest, VerifyValueProgramCatchesInvariantViolations) {
  auto make_val = [] {
    MtlxValueProgram p;
    IrBuilder b(p);
    p.result_root = b.Const(0.5);
    SetMainEnd(p);
    return p;
  };
  {
    auto p = make_val();
    p.consts[0] = 2.0;
    EXPECT_THROW(VerifyValueProgram(p, false), std::logic_error);
  }
  {
    auto p = make_val();
    p.nodes[0].channels = 0;
    EXPECT_THROW(VerifyValueProgram(p, false), std::logic_error);
  }
  {
    auto p = make_val();
    // Self-referential operand violates topological order `a < i`.
    p.nodes.push_back(
        {Op::Neg, 1, 1, 0, 0, {static_cast<ValueId>(p.nodes.size())}});
    SetMainEnd(p);
    EXPECT_THROW(VerifyValueProgram(p, false), std::logic_error);
  }
  {
    MtlxValueProgram p;
    IrBuilder b(p);
    p.result_root = b.Emit(Op::GeomUV, {}, 0, 2);
    SetMainEnd(p);
    p.uniform_end = p.opacity_end = p.main_end;
    p.uniform_offsets.assign(p.uniform_end, kNoUniformScalar);
    EXPECT_THROW(VerifyValueProgram(p, true), std::logic_error);
  }
  {
    MtlxValueProgram p;
    IrBuilder b(p);
    ValueId uv = b.Emit(Op::GeomUV, {}, 0, 2);
    p.result_root = uv;
    SetMainEnd(p);
    p.nodes.push_back({Op::Neg, 2, 1, 0, 0, {uv}});
    p.regions.push_back({p.main_end, static_cast<uint32_t>(p.nodes.size()),
                         static_cast<ValueId>(p.main_end), 0, 0, 0});
    EXPECT_THROW(VerifyValueProgram(p, false), std::logic_error);
  }
  {
    MtlxValueProgram p;
    IrBuilder b(p);
    ValueId mat = b.Emit(
        Op::MatTranspose,
        {b.ConstRgb(1, 0, 0), b.ConstRgb(0, 1, 0), b.ConstRgb(0, 0, 1)}, 0, 3);
    p.result_root = b.Neg(mat);
    SetMainEnd(p);
    EXPECT_THROW(VerifyValueProgram(p, false), std::logic_error);
  }
}

TEST(MaterialXOptimizerTest, VerifyBsdfProgramCatchesInvariantViolations) {
  auto make_bsdf = [] {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures.push_back(Diffuse(b, b.One()));
    SetMainEnd(p.values);
    return p;
  };
  {
    auto p = make_bsdf();
    p.values.result_root = 0;
    EXPECT_THROW(VerifyBsdfProgram(p, false), std::logic_error);
  }
  {
    auto p = make_bsdf();
    p.closures.clear();
    EXPECT_THROW(VerifyBsdfProgram(p, false), std::logic_error);
  }
  {
    auto p = make_bsdf();
    p.closures.push_back(ClosureNode::MakeAdd(0, 1));
    EXPECT_THROW(VerifyBsdfProgram(p, false), std::logic_error);
  }
  {
    auto p = make_bsdf();
    OptimizeBsdfProgram(p);
    p.opacity_root = static_cast<ValueId>(p.values.opacity_end);
    EXPECT_THROW(VerifyBsdfProgram(p, true), std::logic_error);
  }
}

TEST(MaterialXOptimizerTest, IrBuilderGvnCanonicalizationAndConstFold) {
  MtlxValueProgram p;
  IrBuilder b(p);
  ValueId p0 = b.Opaque(2.0), p1 = b.Opaque(3.0);
  EXPECT_EQ(b.Add(p0, p1), b.Add(p1, p0));
  EXPECT_EQ(b.Mul(p0, p1), b.Mul(p1, p0));
  EXPECT_EQ(b.Fma(p0, p1, b.Const(5.0)), b.Fma(p1, p0, b.Const(5.0)));

  double val = 0.0;
  ASSERT_TRUE(b.IsConstScalar(
      b.Add(b.Mul(b.Const(3.0), b.Const(4.0)), b.Const(5.0)), &val));
  EXPECT_DOUBLE_EQ(val, 17.0);
  EXPECT_EQ(b.Sub(p0, p0), b.Zero());
  EXPECT_EQ(b.Add(p0, b.Neg(p1)), b.Sub(p0, p1));
  EXPECT_EQ(b.Chan(p0, 2), p0);
  EXPECT_FALSE(b.IsVarying(p0));
  EXPECT_TRUE(b.IsClipped01(b.Const(0.5)));
}

TEST(MaterialXOptimizerTest, IrBuilderAffineCollapseDivMixAndUniformHoist) {
  MtlxValueProgram p;
  IrBuilder b(p);
  ValueId u = b.Chan(b.Emit(Op::GeomUV, {}, 0, 2), 0);
  ValueId p0 = b.Opaque(2.0), p1 = b.Opaque(3.0), p2 = b.Opaque(4.0),
          p3 = b.Opaque(5.0);
  // Chain: `Mix(p3, p2, Div(u * p0 + p1, p2))` collapses to one varying `Fma`.
  ValueId scaled = b.Div(b.Add(b.Mul(u, p0), p1), p2);
  p.result_root = b.Mix(p3, p2, scaled);
  SetMainEnd(p);

  OptimizeValueProgram(p);
  EXPECT_EQ(p.main_end - p.uniform_end, 3u);
  EXPECT_EQ(p.nodes[p.result_root].op, Op::Fma);
  auto args = NodeArgs(p.nodes[p.result_root], p.extra_args);
  EXPECT_NE(args[0] < p.uniform_end, args[1] < p.uniform_end);
  EXPECT_LT(args[2], p.uniform_end);
}

TEST(MaterialXOptimizerTest, IrBuilderClipPowSelectAndNormalRewrites) {
  MtlxValueProgram p;
  IrBuilder b(p);
  ValueId u = b.Chan(b.Emit(Op::GeomUV, {}, 0, 2), 0);
  ValueId clipped =
      b.Emit(Op::Clip, {b.Emit(Op::Fract, {u}), b.Zero(), b.One()});
  ValueId p2 =
      b.Emit(Op::Pow, {b.Emit(Op::Pow, {clipped, b.Const(0.5)}), b.Const(2.0)});
  ValueId sel = b.Select(b.Emit(Op::Lt, {u, p2}), b.Zero(), b.One());
  ValueId n = b.Emit(Op::Normalize, {b.Emit(Op::GeomNormalWorld)}, 3);
  ValueId h2n0 = b.Emit(Op::HeightToNormal, {b.Zero()}, 0);
  p.result_root = b.Add(b.Add(n, sel), h2n0);
  SetMainEnd(p);

  OptimizeValueProgram(p);
  for (const Node& node : p.nodes) {
    EXPECT_NE(node.op, Op::Clip);
    EXPECT_NE(node.op, Op::Pow);
    EXPECT_NE(node.op, Op::Select);
    EXPECT_NE(node.op, Op::Normalize);
    EXPECT_NE(node.op, Op::HeightToNormal);
  }
}

TEST(MaterialXOptimizerTest, UniformOnlyProgramAllocatesZeroVaryingSlots) {
  MtlxValueProgram p;
  IrBuilder b(p);
  p.result_root = b.Add(b.Mul(b.Opaque(0.25), b.Opaque(0.75)), b.Const(0.5));
  SetMainEnd(p);

  OptimizeValueProgram(p);
  EXPECT_EQ(p.uniform_end, p.main_end);
  EXPECT_EQ(p.opacity_end, p.main_end);
  EXPECT_LT(p.result_root, p.uniform_end);
  EXPECT_EQ(p.uniform_scalar_count, 1u);
}

TEST(MaterialXOptimizerTest, PruneAndScheduleHoistsSharedRegionUniforms) {
  MtlxValueProgram p;
  IrBuilder b(p);
  ValueId shared = b.Add(b.Opaque(2.0), b.Const(1.0));
  p.result_root = b.Emit(Op::HeightToNormal, {shared}, 0);
  SetMainEnd(p);

  std::vector<Node> rnodes;
  std::vector<ValueId> rextra;
  IrBuilder rb(rnodes, rextra, p.consts, p.opaques);
  ValueId r_root = rb.Mul(rb.Chan(rb.Emit(Op::GeomUV, {}, 0, 2), 0),
                          rb.Add(rb.OpaqueRef(0), rb.Const(1.0)));
  ASSERT_TRUE(rb.IsVarying(r_root));
  p.AppendRegion(rnodes, rextra, r_root, 0);

  OptimizeValueProgram(p);
  ASSERT_EQ(p.regions.size(), 1u);
  for (uint32_t i = p.regions[0].begin; i < p.regions[0].end; ++i) {
    EXPECT_NE(p.nodes[i].op, Op::Opaque);
    EXPECT_NE(p.nodes[i].op, Op::Add);
  }
  int adds = 0;
  for (const Node& n : p.nodes) adds += (n.op == Op::Add);
  EXPECT_EQ(adds, 1);
}

TEST(MaterialXOptimizerTest, OpacityPrefixSeparatesOpacityFromSurface) {
  MtlxBsdfProgram p;
  IrBuilder b(p.values);
  p.opacity_root = b.Emit(Op::SampleInput, {}, 0, 1, 1);
  p.closures.push_back(Diffuse(b, Tex3(b, 1)));
  SetMainEnd(p.values);

  OptimizeResult res = OptimizeBsdfProgram(p);
  EXPECT_EQ(res.live_inputs, (std::vector<uint16_t>{0, 1}));
  EXPECT_GE(p.opacity_root, p.values.uniform_end);
  EXPECT_LT(p.opacity_root, p.values.opacity_end);
  EXPECT_GE(p.closures[0].diffuse.color, p.values.opacity_end);
  EXPECT_LT(p.closures[0].diffuse.color, p.values.main_end);
}

TEST(MaterialXOptimizerTest,
     SimplifyClosuresPrunesOccludedBaseAndDeadResources) {
  MtlxBsdfProgram p;
  p.values.attr_names = {"dead_attr", "live_attr"};
  IrBuilder b(p.values);
  ValueId dead_a = b.Emit(Op::Attr, {b.Opaque(0.1)}, 0, 3, 3);
  ValueId live_a = b.Emit(Op::Attr, {b.Opaque(0.9)}, 1, 3, 3);
  ValueId top_col = b.Add(Tex3(b, 1), live_a);
  ValueId base_col = b.Add(Tex3(b, 0), dead_a);
  // Zero-scale NormalMap folds to GeomNormalWorld in `IrBuilder::Emit`, which
  // `SimplifyClosures` clears to `kInvalidValue`.
  ValueId trivial_n =
      b.Emit(Op::NormalMap, {b.ConstRgb(0.5, 0.5, 1.0), b.Zero()},
             kNormalMapDefaultFrame);
  p.closures = {Diffuse(b, top_col, b.One(), trivial_n), Diffuse(b, base_col),
                ClosureNode::MakeLayer(0, 1)};
  SetMainEnd(p.values);

  OptimizeResult res = OptimizeBsdfProgram(p);
  ASSERT_EQ(p.closures.size(), 1u);
  EXPECT_EQ(p.closures[0].op, ClosureOp::OrenNayarDiffuse);
  EXPECT_EQ(p.closures[0].diffuse.normal, kInvalidValue);
  EXPECT_EQ(res.live_inputs, (std::vector<uint16_t>{1}));
  EXPECT_EQ(res.live_attrs, (std::vector<uint16_t>{1}));
  ASSERT_EQ(p.values.opaques.size(), 1u);
  EXPECT_DOUBLE_EQ(p.values.opaques[0], 0.9);
  EXPECT_EQ(p.values.attr_names, (std::vector<std::string>{"live_attr"}));
}

TEST(MaterialXOptimizerTest,
     SimplifyClosuresZeroMultiplySoundnessUnderNeedsTp) {
  // Opaque top `Multiply(Diffuse, 0)` becomes `OpaqueBlocker` and occludes
  // base.
  {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures = {Diffuse(b, Tex3(b, 0)),
                  ClosureNode::MakeMultiply(0, b.Zero()),
                  Diffuse(b, Tex3(b, 1)), ClosureNode::MakeLayer(1, 2)};
    SetMainEnd(p.values);
    OptimizeResult res = OptimizeBsdfProgram(p);
    ASSERT_EQ(p.closures.size(), 1u);
    EXPECT_EQ(p.closures[0].op, ClosureOp::Null);
    EXPECT_TRUE(res.live_inputs.empty());
    EXPECT_TRUE(p.values.nodes.empty());
  }
  // Transmissive `Multiply(Sheen, 0)` under `Layer` preserves partial `tp`.
  {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures = {Sheen(b, Tex3(b, 0)), ClosureNode::MakeMultiply(0, b.Zero()),
                  Diffuse(b, Tex3(b, 1)), ClosureNode::MakeLayer(1, 2)};
    SetMainEnd(p.values);
    OptimizeResult res = OptimizeBsdfProgram(p);
    ASSERT_EQ(p.closures.size(), 4u);
    EXPECT_NE(p.closures[0].flags & kClosureNeedsTp, 0u);
    EXPECT_EQ(p.closures[1].op, ClosureOp::Multiply);
    EXPECT_NE(p.closures[1].flags & kClosureNeedsTp, 0u);
    EXPECT_EQ(p.closures[2].flags & kClosureNeedsTp, 0u);
    EXPECT_EQ(res.live_inputs, (std::vector<uint16_t>{0, 1}));
  }
  // Root `Multiply(Sheen, 0)` (`!needs_tp`) folds to `Null`.
  {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures = {Sheen(b, Tex3(b, 0)), ClosureNode::MakeMultiply(0, b.Zero())};
    SetMainEnd(p.values);
    OptimizeResult res = OptimizeBsdfProgram(p);
    ASSERT_EQ(p.closures.size(), 1u);
    EXPECT_EQ(p.closures[0].op, ClosureOp::Null);
    EXPECT_TRUE(res.live_inputs.empty());
  }
}

TEST(MaterialXOptimizerTest, SimplifyClosuresAddOpaqueBlockerSoundness) {
  // At root, opaque `Add(Sheen, OpaqueBlocker)` occludes base and drops
  // blocker.
  {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures = {Sheen(b, Tex3(b, 0)), ClosureNode::MakeOpaqueBlocker(),
                  ClosureNode::MakeAdd(0, 1), Diffuse(b, Tex3(b, 1)),
                  ClosureNode::MakeLayer(2, 3)};
    SetMainEnd(p.values);
    OptimizeResult res = OptimizeBsdfProgram(p);
    ASSERT_EQ(p.closures.size(), 1u);
    EXPECT_EQ(p.closures[0].op, ClosureOp::Sheen);
    EXPECT_EQ(res.live_inputs, (std::vector<uint16_t>{0}));
  }
  // Inside non-opaque `Mix` over `Diffuse`, `Add` must retain `OpaqueBlocker`.
  {
    MtlxBsdfProgram p;
    IrBuilder b(p.values);
    p.closures = {
        Sheen(b, b.One()),          ClosureNode::MakeOpaqueBlocker(),
        ClosureNode::MakeAdd(0, 1), ClosureNode::MakeMix(2, 0, b.Opaque(0.5)),
        Diffuse(b, b.One()),        ClosureNode::MakeLayer(3, 4)};
    SetMainEnd(p.values);
    OptimizeBsdfProgram(p);
    bool has_blocker = false;
    for (const ClosureNode& c : p.closures) {
      has_blocker |= (c.op == ClosureOp::OpaqueBlocker);
    }
    EXPECT_TRUE(has_blocker);
  }
}

TEST(MaterialXOptimizerTest, SimplifyClosuresPrunesZeroWeightLeaf) {
  // `Pow(p0, 0) - 1` folds to `Zero()` in `IrBuilder`, then `SimplifyClosures`
  // prunes the leaf and `PruneAndSchedule` drops its dead texture and opaque.
  MtlxBsdfProgram p;
  IrBuilder b(p.values);
  ValueId w = b.Sub(b.Emit(Op::Pow, {b.Opaque(0.7), b.Zero()}), b.One());
  p.closures = {Diffuse(b, Tex3(b, 0), w)};
  SetMainEnd(p.values);

  OptimizeResult res = OptimizeBsdfProgram(p);
  ASSERT_EQ(p.closures.size(), 1u);
  EXPECT_EQ(p.closures[0].op, ClosureOp::Null);
  EXPECT_TRUE(res.live_inputs.empty());
  EXPECT_TRUE(p.values.opaques.empty());
}

TEST(MaterialXOptimizerTest, DumpProgramFormatsBandsRegionsAndClosures) {
  MtlxBsdfProgram p;
  IrBuilder b(p.values);
  ValueId scale = b.Add(b.Opaque(2.0), b.Const(0.5));
  p.opacity_root = b.Emit(Op::SampleInput, {}, 0, 1, 1);
  ValueId col = b.Mul(Tex3(b, 1), scale);
  ValueId h2n = b.Emit(Op::HeightToNormal, {scale}, 0);
  p.closures = {Sheen(b, col), Diffuse(b, h2n), ClosureNode::MakeLayer(0, 1)};
  p.thin_walled = true;
  SetMainEnd(p.values);

  std::vector<Node> rnodes;
  std::vector<ValueId> rextra;
  IrBuilder rb(rnodes, rextra, p.values.consts, p.values.opaques);
  ValueId r_root = rb.Chan(rb.Emit(Op::GeomUV, {}, 0, 2), 0);
  p.values.AppendRegion(rnodes, rextra, r_root, 0);

  OptimizeBsdfProgram(p);
  std::string dump = DumpBsdfProgram(p);
  EXPECT_THAT(dump, HasSubstr("[uniform]"));
  EXPECT_THAT(dump, HasSubstr("[opacity]"));
  EXPECT_THAT(dump, HasSubstr("[surface]"));
  EXPECT_THAT(dump, HasSubstr("[region 0 root=%"));
  EXPECT_THAT(dump, HasSubstr("[closures]"));
  EXPECT_THAT(dump, HasSubstr("[needs_tp]"));
  EXPECT_THAT(dump, HasSubstr("opacity = %"));
  EXPECT_THAT(dump, HasSubstr("thin_walled = true"));
}

}  // namespace
}  // namespace mitsuba::mtlx
