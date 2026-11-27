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

// Internal data structures and expression-lowering helpers shared between
// `compiler.cc` and `expr_lower.cc`. Do not include from outside `materialx/`.

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/pxr.h>

#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/program.h"
#include "hdmitsuba/texture_cache.h"

PXR_NAMESPACE_OPEN_SCOPE
namespace mtlx_detail {

namespace mx = MaterialX;
namespace mtlx = mitsuba::mtlx;
using mtlx::kInvalidValue;
using mtlx::ValueId;

struct Frame;

struct ScopedNode {
  mx::NodePtr node;
  const Frame* frame = nullptr;

  explicit operator bool() const { return node != nullptr; }
};

// Active `NodeGraph` implementation call frame during lazy lowering.
struct Frame {
  ScopedNode call;
};

using NodeKey = std::pair<const Frame*, const mx::Node*>;
using NodeOutKey = std::tuple<const Frame*, const mx::Node*, std::string>;

struct NodeInfo {
  mx::NodeDefPtr nodedef;
  mx::NodeGraphPtr graph_impl;
};

// Caches MaterialX `NodeDef` / `NodeGraph` lookups and lazy call frames across
// all shader lowers (BSDF, EDF, Displacement) for one document.
struct GraphResolver {
  std::deque<Frame> frames;
  absl::flat_hash_map<NodeKey, const Frame*> child_frames;
  absl::flat_hash_map<const mx::Node*, NodeInfo> node_info;
};

enum class AddressMode : uint8_t {
  Periodic,
  Clamp,
  Mirror,
  Constant,
  Black,
};

inline bool IsOobMode(AddressMode mode) {
  return mode == AddressMode::Constant || mode == AddressMode::Black;
}

// Variant-independent `<image>` sample descriptor referenced by symbolic
// `Op::SampleInput` nodes (`imm` = index into `images`). Operands of the
// symbolic node are `{default_val}` when `default_uv` is true, or
// `{tc, default_val}` otherwise.
struct SymbolicImage {
  TextureKey color_key;
  bool has_alpha = false;
  bool is_vector2 = false;
  bool default_uv = false;
  AddressMode u_mode = AddressMode::Periodic;
  AddressMode v_mode = AddressMode::Periodic;

  bool operator==(const SymbolicImage& o) const {
    return color_key == o.color_key && has_alpha == o.has_alpha &&
           is_vector2 == o.is_vector2 && default_uv == o.default_uv &&
           u_mode == o.u_mode && v_mode == o.v_mode;
  }
};

struct PendingRegion {
  std::vector<mtlx::Node> nodes;
  std::vector<ValueId> extra_args;
  ValueId root = kInvalidValue;
  uint16_t has_texcoord = 0;
};

// A 3x3 or 4x4 matrix as the `ValueId`s of its `n` rows.
struct MatVal {
  ValueId rows[4] = {kInvalidValue, kInvalidValue, kInvalidValue,
                     kInvalidValue};
  int n = 3;
};

// Scope-independent opaque parameter handles for an authored input tuple.
struct AuthoredOpaque {
  uint8_t n = 0;
  bool all_zero = false;
  bool all_one = false;
  uint16_t idx[4] = {};
};

struct AuthoredMatrix {
  int n = 3;
  AuthoredOpaque rows[4];
};

// Per-program lowering state (one per BSDF / EDF / Displacement program),
// borrowing a shared `GraphResolver` and warning deduplication set.
struct LowerCtx {
  mx::DocumentPtr doc;
  GraphResolver& resolver;
  absl::flat_hash_set<std::string>& warned;

  mtlx::MtlxValueProgram val_prog;
  mtlx::IrBuilder main_builder;
  mtlx::IrBuilder* b = nullptr;
  std::vector<PendingRegion> pending_regions;
  std::vector<SymbolicImage> images;
  std::vector<std::pair<std::string, int>> attr_requests;

  absl::flat_hash_map<NodeOutKey, ValueId> node_output_val;
  absl::flat_hash_map<NodeOutKey, MatVal> node_matrix_val;
  absl::flat_hash_map<const mx::ValueElement*, AuthoredOpaque>
      authored_opaque_val;
  absl::flat_hash_map<const mx::ValueElement*, AuthoredMatrix>
      authored_opaque_matrix;

  LowerCtx(mx::DocumentPtr d, GraphResolver& r,
           absl::flat_hash_set<std::string>& w)
      : doc(std::move(d)),
        resolver(r),
        warned(w),
        main_builder(val_prog),
        b(&main_builder) {}
};

// Graph resolution helpers (`expr_lower.cc`).
const std::string& NodeCategory(GraphResolver& r, const mx::NodePtr& node);
std::string_view InputType(GraphResolver& r, const mx::NodePtr& node,
                           const std::string& name);
ScopedNode ResolveConnected(GraphResolver& r, const ScopedNode& s,
                            const std::string& name,
                            const char* mat_input = "surfaceshader");
std::string GetNodeInputString(GraphResolver& r, const ScopedNode& s,
                               const std::string& name,
                               std::string_view def = {});
ScopedNode UnwrapShaderNode(GraphResolver& r, ScopedNode s,
                            std::string output_name,
                            const char* mat_input = "surfaceshader");

// Expression lowering helpers (`expr_lower.cc`).
ValueId LowerInputExpr(LowerCtx& ctx, const ScopedNode& node,
                       const std::string& input_name,
                       ValueId default_val = kInvalidValue);

inline auto BindInputs(LowerCtx& ctx, const ScopedNode& node) {
  return [&ctx, &node](const std::string& name, ValueId def = kInvalidValue) {
    return LowerInputExpr(ctx, node, name, def);
  };
}

inline ValueId NWorld(LowerCtx& ctx) {
  return ctx.b->Emit(mtlx::Op::GeomNormalWorld);
}
inline ValueId TWorld(LowerCtx& ctx) {
  return ctx.b->Emit(mtlx::Op::GeomTangentWorld);
}
inline ValueId Clip01(LowerCtx& ctx, ValueId v) {
  return ctx.b->Emit(mtlx::Op::Clip, {v, ctx.b->Zero(), ctx.b->One()});
}

template <typename FgFn, typename BgFn>
ValueId LowerMixVal(LowerCtx& ctx, ValueId m, FgFn&& fg_fn, BgFn&& bg_fn) {
  if (ctx.b->IsZero(m)) return bg_fn();
  if (ctx.b->IsOne(m)) return fg_fn();
  ValueId fg = fg_fn(), bg = bg_fn();
  return ctx.b->Mix(fg, bg, m);
}

inline void WarnUnsupported(LowerCtx& ctx, const std::string& msg) {
  if (ctx.warned.insert(msg).second) TF_WARN("%s", msg.c_str());
}

inline void AppendPendingRegions(mtlx::MtlxValueProgram& prog,
                                 std::vector<PendingRegion>& pending) {
  prog.main_end = static_cast<uint32_t>(prog.nodes.size());
  prog.regions.reserve(pending.size());
  for (const PendingRegion& reg : pending) {
    prog.AppendRegion(reg.nodes, reg.extra_args, reg.root, reg.has_texcoord);
  }
  pending.clear();
}

}  // namespace mtlx_detail
PXR_NAMESPACE_CLOSE_SCOPE
