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

#include "hdmitsuba/materialx/optimizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <absl/types/span.h>

#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/program.h"

namespace mitsuba::mtlx {
namespace {

void Check(bool cond, const char* msg) {
  if (!cond) throw std::logic_error(std::string("MtlxSSA verify: ") + msg);
}

std::vector<uint8_t> ComputeVarying(const MtlxValueProgram& prog) {
  std::vector<uint8_t> varying(prog.nodes.size(), 0);
  for (size_t i = 0; i < prog.nodes.size(); ++i) {
    const Node& n = prog.nodes[i];
    bool v = is_spatially_varying_op(n.op) || n.op == Op::SampleInput;
    for (ValueId a : node_args(n, prog.extra_args)) v |= (varying[a] != 0);
    varying[i] = v ? 1 : 0;
  }
  return varying;
}

struct ClosureAnalysis {
  std::vector<uint8_t> is_opaque;
  std::vector<uint8_t> has_scat;
  std::vector<uint8_t> needs_tp;
};

template <typename IsZeroFn>
ClosureAnalysis AnalyzeClosures(absl::Span<const ClosureNode> closures,
                                IsZeroFn&& is_zero) {
  size_t n = closures.size();
  ClosureAnalysis a{std::vector<uint8_t>(n, 0), std::vector<uint8_t>(n, 0),
                    std::vector<uint8_t>(n, 0)};
  for (size_t i = 0; i < n; ++i) {
    const ClosureNode& c = closures[i];
    switch (c.op) {
      case ClosureOp::Null:
        break;
      case ClosureOp::OpaqueBlocker:
        a.is_opaque[i] = 1;
        break;
      case ClosureOp::Layer:
      case ClosureOp::Add:
        a.is_opaque[i] =
            a.is_opaque[c.comb.first] || a.is_opaque[c.comb.second];
        a.has_scat[i] = a.has_scat[c.comb.first] || a.has_scat[c.comb.second];
        break;
      case ClosureOp::Mix:
        a.is_opaque[i] =
            a.is_opaque[c.comb.first] && a.is_opaque[c.comb.second];
        a.has_scat[i] = a.has_scat[c.comb.first] || a.has_scat[c.comb.second];
        break;
      case ClosureOp::Multiply:
        a.is_opaque[i] = a.is_opaque[c.comb.first];
        a.has_scat[i] = a.has_scat[c.comb.first] && !is_zero(c.comb.weight);
        break;
      default: {
        ValueId w = leaf_weight_id(c);
        a.is_opaque[i] = !leaf_has_layer_throughput(c);
        a.has_scat[i] = (w == kInvalidValue || !is_zero(w)) ? 1 : 0;
        break;
      }
    }
  }
  for (size_t i = n; i-- > 0;) {
    const ClosureNode& c = closures[i];
    if (c.op == ClosureOp::Layer) {
      if ((a.has_scat[c.comb.second] && !a.is_opaque[c.comb.first]) ||
          a.needs_tp[i]) {
        a.needs_tp[c.comb.first] = 1;
      }
      if (a.needs_tp[i] && !a.is_opaque[c.comb.first]) {
        a.needs_tp[c.comb.second] = 1;
      }
    } else if (a.needs_tp[i] && is_combinator(c.op)) {
      a.needs_tp[c.comb.first] = 1;
      if (c.comb.second >= 0) a.needs_tp[c.comb.second] = 1;
    }
  }
  return a;
}

void SimplifyClosures(MtlxBsdfProgram& prog) {
  auto is_const = [&](ValueId id, uint16_t imm) {
    return prog.values.nodes[id].op == Op::Const &&
           prog.values.nodes[id].imm == imm;
  };
  auto is_zero = [&](ValueId id) { return is_const(id, 0); };
  auto is_one = [&](ValueId id) { return is_const(id, 1); };
  auto clear_if_op = [&](ValueId& id, Op op) {
    if (id != kInvalidValue && prog.values.nodes[id].op == op) {
      id = kInvalidValue;
    }
  };

  for (ClosureNode& c : prog.closures) {
    if (prog.thin_walled && (c.op == ClosureOp::Dielectric ||
                             c.op == ClosureOp::GeneralizedSchlick)) {
      c.microfacet.mode |= kScatterThinWalled | kScatterTwoSided;
    }
    if (c.op >= ClosureOp::OrenNayarDiffuse && c.op <= ClosureOp::Translucent) {
      clear_if_op(c.diffuse.normal, Op::GeomNormalWorld);
    } else if (c.op >= ClosureOp::Dielectric &&
               c.op <= ClosureOp::GeneralizedSchlick) {
      clear_if_op(c.microfacet.normal, Op::GeomNormalWorld);
      clear_if_op(c.microfacet.tangent, Op::GeomTangentWorld);
      if (c.microfacet.tf_thickness != kInvalidValue &&
          is_zero(c.microfacet.tf_thickness)) {
        c.microfacet.tf_thickness = c.microfacet.tf_ior = kInvalidValue;
      }
    } else if (c.op == ClosureOp::ChiangHair) {
      clear_if_op(c.hair.normal, Op::GeomNormalWorld);
      clear_if_op(c.hair.curve_direction, Op::GeomTangentWorld);
    }
  }

  if (prog.opacity_root != kInvalidValue) {
    if (is_zero(prog.opacity_root)) {
      prog.closures = {ClosureNode::MakeNull()};
      return;
    }
    if (is_one(prog.opacity_root)) prog.opacity_root = kInvalidValue;
  }

  ClosureAnalysis info = AnalyzeClosures(prog.closures, is_zero);
  size_t n = prog.closures.size();
  std::vector<ClosureNode> simp;
  simp.reserve(n);
  std::vector<int32_t> remap(n, -1);
  int32_t null_idx = -1, blocker_idx = -1;
  auto emit_node = [&](ClosureNode node) -> int32_t {
    int32_t* cached = node.op == ClosureOp::Null            ? &null_idx
                      : node.op == ClosureOp::OpaqueBlocker ? &blocker_idx
                                                            : nullptr;
    if (cached && *cached >= 0) return *cached;
    int32_t idx = static_cast<int32_t>(simp.size());
    simp.push_back(node);
    if (cached) *cached = idx;
    return idx;
  };
  auto is_null = [&](int32_t idx) { return simp[idx].op == ClosureOp::Null; };

  for (size_t i = 0; i < n; ++i) {
    ClosureNode c = prog.closures[i];
    if (!is_combinator(c.op)) {
      remap[i] = emit_node((is_leaf(c.op) && !info.has_scat[i])
                               ? (info.is_opaque[i] && info.needs_tp[i]
                                      ? ClosureNode::MakeOpaqueBlocker()
                                      : ClosureNode::MakeNull())
                               : c);
      continue;
    }
    int32_t old0 = c.comb.first, old1 = c.comb.second;
    int32_t c0 = remap[old0], c1 = old1 >= 0 ? remap[old1] : -1;
    int32_t folded = -1;
    if (c.op == ClosureOp::Layer) {
      if (info.is_opaque[old0] ||
          (!info.has_scat[old1] &&
           (!info.needs_tp[i] || !info.is_opaque[old1]))) {
        folded = c0;
      } else if (is_null(c0)) {
        folded = c1;
      }
    } else if (c.op == ClosureOp::Add) {
      if (is_null(c0) || (!info.has_scat[old0] &&
                          (!info.needs_tp[i] || info.is_opaque[old1]))) {
        folded = c1;
      } else if (is_null(c1) || (!info.has_scat[old1] &&
                                 (!info.needs_tp[i] || info.is_opaque[old0]))) {
        folded = c0;
      }
    } else if (c.op == ClosureOp::Mix) {
      if (is_zero(c.comb.weight))
        folded = c1;
      else if (is_one(c.comb.weight) || c0 == c1)
        folded = c0;
    } else {  // Multiply
      if (is_one(c.comb.weight) || is_null(c0) ||
          simp[c0].op == ClosureOp::OpaqueBlocker) {
        folded = c0;
      } else if (is_zero(c.comb.weight)) {
        if (!info.needs_tp[i]) {
          folded = emit_node(ClosureNode::MakeNull());
        } else if (info.is_opaque[old0]) {
          folded = emit_node(ClosureNode::MakeOpaqueBlocker());
        }
        // Else keep `Multiply` to preserve partial `tp[c]` for `Layer.top`.
      }
    }
    if (folded >= 0) {
      remap[i] = folded;
    } else {
      c.comb.first = c0;
      c.comb.second = c1;
      remap[i] = emit_node(c);
    }
  }

  int32_t root = remap[n - 1];
  if (simp[root].op == ClosureOp::OpaqueBlocker) {
    prog.closures = {ClosureNode::MakeNull()};
    return;
  }

  std::vector<uint8_t> live(simp.size(), 0);
  live[root] = 1;
  for (int32_t i = root; i >= 0; --i) {
    if (!live[i] || !is_combinator(simp[i].op)) continue;
    live[simp[i].comb.first] = 1;
    if (simp[i].comb.second >= 0) live[simp[i].comb.second] = 1;
  }

  std::vector<ClosureNode> compact;
  std::vector<int32_t> cremap(simp.size(), -1);
  for (int32_t i = 0; i <= root; ++i) {
    if (!live[i]) continue;
    ClosureNode c = simp[i];
    c.flags = kClosureNone;
    if (is_combinator(c.op)) {
      c.comb.first = cremap[c.comb.first];
      if (c.comb.second >= 0) c.comb.second = cremap[c.comb.second];
    }
    cremap[i] = static_cast<int32_t>(compact.size());
    compact.push_back(c);
  }

  ClosureAnalysis final_info = AnalyzeClosures(compact, is_zero);
  for (size_t i = 0; i < compact.size(); ++i) {
    if (final_info.needs_tp[i]) compact[i].flags |= kClosureNeedsTp;
  }
  prog.closures = std::move(compact);
}

OptimizeResult PruneAndSchedule(MtlxValueProgram& prog, MtlxBsdfProgram* bsdf) {
  std::vector<uint8_t> live(prog.nodes.size(), 0);
  std::vector<uint8_t> live_region(prog.regions.size(), 0);
  auto mark = [&](ValueId id) {
    if (id != kInvalidValue) live[id] = 1;
  };
  if (bsdf) {
    mark(bsdf->opacity_root);
    for (const ClosureNode& c : bsdf->closures) {
      for_each_closure_value_id(c, mark);
    }
  } else {
    mark(prog.result_root);
  }
  auto walk_back = [&](uint32_t begin, uint32_t end) {
    for (uint32_t i = end; i-- > begin;) {
      if (!live[i]) continue;
      for (ValueId a : node_args(prog.nodes[i], prog.extra_args)) live[a] = 1;
      if (prog.nodes[i].op == Op::HeightToNormal) {
        live_region[prog.nodes[i].imm] = 1;
      }
    }
  };
  walk_back(0, prog.main_end);
  for (size_t k = prog.regions.size(); k-- > 0;) {
    if (!live_region[k]) continue;
    live[prog.regions[k].root] = 1;
    walk_back(prog.regions[k].begin, prog.regions[k].end);
  }

  std::vector<uint8_t> varying = ComputeVarying(prog);
  std::vector<uint8_t> in_opacity(prog.main_end, 0);
  if (bsdf && bsdf->opacity_root != kInvalidValue &&
      varying[bsdf->opacity_root]) {
    in_opacity[bsdf->opacity_root] = 1;
    for (uint32_t i = prog.main_end; i-- > 0;) {
      if (!in_opacity[i]) continue;
      for (ValueId a : node_args(prog.nodes[i], prog.extra_args)) {
        if (varying[a]) in_opacity[a] = 1;
      }
    }
  }

  // Schedule into 4 bands: [uniform | opacity | surface | regions].
  MtlxValueProgram out;
  out.opaques = prog.opaques;
  out.attr_names = prog.attr_names;
  IrBuilder b(out);
  std::vector<ValueId> remap(prog.nodes.size(), kInvalidValue);
  std::vector<uint16_t> region_remap(prog.regions.size(), kMaxImm);
  uint16_t next_region = 0;
  for (size_t k = 0; k < prog.regions.size(); ++k) {
    if (live_region[k]) region_remap[k] = next_region++;
  }

  auto emit = [&](uint32_t i) -> ValueId {
    const Node& n = prog.nodes[i];
    if (n.op == Op::Const) return b.Const(prog.consts[n.imm]);
    if (n.op == Op::Opaque) return b.OpaqueRef(n.imm);
    absl::InlinedVector<ValueId, 8> args;
    for (ValueId a : node_args(n, prog.extra_args)) args.push_back(remap[a]);
    uint16_t imm = n.op == Op::HeightToNormal ? region_remap[n.imm] : n.imm;
    return b.EmitRaw(n.op, args, imm, n.channels, n.flags);
  };

  for (uint32_t i = 0; i < prog.main_end; ++i) {
    if (live[i] && !varying[i]) remap[i] = emit(i);
  }
  for (size_t k = 0; k < prog.regions.size(); ++k) {
    if (!live_region[k]) continue;
    for (uint32_t i = prog.regions[k].begin; i < prog.regions[k].end; ++i) {
      if (live[i] && !varying[i]) remap[i] = emit(i);
    }
  }
  out.uniform_end = static_cast<uint32_t>(out.nodes.size());

  for (uint32_t i = 0; i < prog.main_end; ++i) {
    if (live[i] && varying[i] && in_opacity[i]) remap[i] = emit(i);
  }
  out.opacity_end = static_cast<uint32_t>(out.nodes.size());

  for (uint32_t i = 0; i < prog.main_end; ++i) {
    if (live[i] && varying[i] && !in_opacity[i]) remap[i] = emit(i);
  }
  out.main_end = static_cast<uint32_t>(out.nodes.size());

  out.regions.reserve(next_region);
  for (size_t k = 0; k < prog.regions.size(); ++k) {
    if (!live_region[k]) continue;
    b.ResetGvn();
    uint32_t begin = static_cast<uint32_t>(out.nodes.size());
    for (uint32_t i = prog.regions[k].begin; i < prog.regions[k].end; ++i) {
      if (live[i] && varying[i]) remap[i] = emit(i);
    }
    out.regions.push_back({begin, static_cast<uint32_t>(out.nodes.size()),
                           remap[prog.regions[k].root], 0,
                           prog.regions[k].has_texcoord, 0});
  }

  if (prog.result_root != kInvalidValue) {
    out.result_root = remap[prog.result_root];
  }
  if (bsdf) {
    if (bsdf->opacity_root != kInvalidValue) {
      bsdf->opacity_root = remap[bsdf->opacity_root];
    }
    for (ClosureNode& c : bsdf->closures) {
      for_each_closure_value_id(c, [&](ValueId& id) { id = remap[id]; });
    }
  }

  // Monotonically compact `opaques`, `attr_names`, and texture `inputs`.
  auto compact_table = [](std::vector<uint16_t>& map, auto parallel_vec,
                          std::vector<uint16_t>* live_out) {
    uint16_t next = 0;
    for (size_t i = 0; i < map.size(); ++i) {
      if (map[i] == kMaxImm) continue;
      if constexpr (!std::is_null_pointer_v<decltype(parallel_vec)>) {
        if (next != i) (*parallel_vec)[next] = std::move((*parallel_vec)[i]);
      }
      if (live_out) live_out->push_back(static_cast<uint16_t>(i));
      map[i] = next++;
    }
    if constexpr (!std::is_null_pointer_v<decltype(parallel_vec)>) {
      parallel_vec->resize(next);
    }
  };
  std::vector<uint16_t> opaque_map(out.opaques.size(), kMaxImm);
  std::vector<uint16_t> attr_map(out.attr_names.size(), kMaxImm);
  std::vector<uint16_t> input_map;
  for (const Node& n : out.nodes) {
    if (n.op == Op::Opaque) {
      opaque_map[n.imm] = 0;
    } else if (n.op == Op::Attr) {
      attr_map[n.imm] = 0;
    } else if (n.op == Op::SampleInput || n.op == Op::SampleInputUV) {
      if (n.imm >= input_map.size()) input_map.resize(n.imm + 1, kMaxImm);
      input_map[n.imm] = 0;
    }
  }
  OptimizeResult result;
  compact_table(opaque_map, &out.opaques, nullptr);
  compact_table(attr_map, &out.attr_names, &result.live_attrs);
  compact_table(input_map, nullptr, &result.live_inputs);
  for (Node& n : out.nodes) {
    if (n.op == Op::Opaque)
      n.imm = opaque_map[n.imm];
    else if (n.op == Op::Attr)
      n.imm = attr_map[n.imm];
    else if (n.op == Op::SampleInput || n.op == Op::SampleInputUV) {
      n.imm = input_map[n.imm];
    }
  }

  auto assign_mats = [&](uint32_t begin, uint32_t end) {
    uint32_t count = 0;
    for (uint32_t i = begin; i < end; ++i) {
      if (is_matrix_producer(out.nodes[i].op)) {
        out.nodes[i].imm = static_cast<uint16_t>(count++);
      }
    }
    return count;
  };
  out.uniform_mat_count = assign_mats(0, out.uniform_end);
  out.main_mat_count = assign_mats(out.uniform_end, out.main_end);
  for (Region& r : out.regions) {
    r.mat_count = static_cast<uint16_t>(assign_mats(r.begin, r.end));
  }

  // Pack only computed uniform values read outside `[0 .. uniform_end)`.
  std::vector<uint8_t> escaped(out.uniform_end, 0);
  auto mark_esc = [&](ValueId id) {
    if (id != kInvalidValue && id < out.uniform_end) escaped[id] = 1;
  };
  for (size_t i = out.uniform_end; i < out.nodes.size(); ++i) {
    for (ValueId a : node_args(out.nodes[i], out.extra_args)) mark_esc(a);
  }
  mark_esc(out.result_root);
  if (bsdf) {
    mark_esc(bsdf->opacity_root);
    for (const ClosureNode& c : bsdf->closures) {
      for_each_closure_value_id(c, mark_esc);
    }
  }
  out.uniform_offsets.assign(out.uniform_end, kNoUniformScalar);
  for (uint32_t u = 0; u < out.uniform_end; ++u) {
    Op op = out.nodes[u].op;
    if (!escaped[u] || op == Op::Const || op == Op::Opaque ||
        is_matrix_producer(op)) {
      continue;
    }
    out.uniform_offsets[u] = out.uniform_scalar_count;
    out.uniform_scalar_count += out.nodes[u].channels;
  }
  prog = std::move(out);
  return result;
}

const char* OpName(Op op) {
  static constexpr const char* kNames[] = {
      "const",     "opaque",  "sample",  "sample_uv", "attr",      "uv",
      "pos_w",     "n_w",     "t_w",     "b_w",       "wi_w",      "normalmap",
      "h2n",       "neg",     "abs",     "sin",       "cos",       "tan",
      "exp",       "round",   "floor",   "ceil",      "sqrt",      "log",
      "asin",      "acos",    "fract",   "add",       "mul",       "eq",
      "min",       "max",     "sub",     "div",       "lt",        "le",
      "pow",       "atan2",   "select",  "mix",       "clip",      "fma",
      "chan",      "dot",     "vec2",    "rot2d",     "normalize", "rgb",
      "cross",     "rot3d",   "rgb2hsv", "hsv2rgb",   "blackbody", "vec4",
      "cellnoise", "noise",   "worley",  "fractal",   "mat_det",   "mat_t",
      "mat_inv",   "mat_mul", "mat_row",
  };
  static_assert(std::size(kNames) == static_cast<size_t>(Op::MatRow) + 1);
  return kNames[static_cast<size_t>(op)];
}

const char* ClosureOpName(ClosureOp op) {
  static constexpr const char* kNames[] = {
      "null",        "opaque_blocker", "diffuse",    "burley",
      "sheen",       "translucent",    "dielectric", "conductor",
      "gen_schlick", "chiang_hair",    "layer",      "mix",
      "add",         "multiply",
  };
  static_assert(std::size(kNames) ==
                static_cast<size_t>(ClosureOp::Multiply) + 1);
  return kNames[static_cast<size_t>(op)];
}

void DumpBand(std::ostringstream& os, const MtlxValueProgram& prog,
              const char* label, uint32_t begin, uint32_t end) {
  if (begin >= end) return;
  os << "[" << label << "]\n";
  for (uint32_t i = begin; i < end; ++i) {
    const Node& n = prog.nodes[i];
    os << "  %" << i << " = " << OpName(n.op);
    if (n.op == Op::Const) {
      os << "(" << prog.consts[n.imm] << ")";
    } else if (n.op == Op::Opaque) {
      os << "[" << n.imm << "](" << prog.opaques[n.imm] << ")";
    } else {
      if (n.imm != 0 || n.op == Op::SampleInput || n.op == Op::SampleInputUV ||
          n.op == Op::Attr || n.op == Op::HeightToNormal || n.op == Op::Chan ||
          n.op == Op::MatRow) {
        os << "[" << n.imm << "]";
      }
      os << "(";
      if (n.op == Op::Attr) os << "\"" << prog.attr_names[n.imm] << "\", ";
      auto args = node_args(n, prog.extra_args);
      for (size_t j = 0; j < args.size(); ++j) {
        if (j > 0) os << ", ";
        os << "%" << args[j];
      }
      os << ")";
    }
    if (n.channels > 1) os << " : f" << int(n.channels);
    if (i < prog.uniform_offsets.size() &&
        prog.uniform_offsets[i] != kNoUniformScalar) {
      os << "  // u[" << prog.uniform_offsets[i] << "]";
    }
    os << "\n";
  }
}

}  // namespace

void VerifyValueProgram(const MtlxValueProgram& prog, bool scheduled) {
  Check(
      prog.consts.size() >= 2 && prog.consts[0] == 0.0 && prog.consts[1] == 1.0,
      "consts must start with {0.0, 1.0}");
  Check(prog.main_end <= prog.nodes.size(), "main_end out of range");
  if (scheduled) {
    Check(prog.uniform_end <= prog.opacity_end &&
              prog.opacity_end <= prog.main_end &&
              prog.uniform_offsets.size() == prog.uniform_end,
          "invalid schedule bounds");
  }

  std::vector<int> node_region(prog.nodes.size(), -1);
  uint32_t cursor = prog.main_end;
  for (size_t k = 0; k < prog.regions.size(); ++k) {
    const Region& r = prog.regions[k];
    Check(r.begin == cursor && r.begin < r.end && r.end <= prog.nodes.size() &&
              r.root >= r.begin && r.root < r.end &&
              !is_matrix_producer(prog.nodes[r.root].op),
          "invalid region bounds or root");
    std::fill(node_region.begin() + r.begin, node_region.begin() + r.end,
              static_cast<int>(k));
    cursor = r.end;
  }
  Check(cursor == prog.nodes.size(), "trailing nodes outside regions");

  for (size_t i = 0; i < prog.nodes.size(); ++i) {
    const Node& n = prog.nodes[i];
    Check(n.channels >= 1 && n.channels <= 4, "invalid node channels");
    if (n.arity > 3) {
      Check(static_cast<size_t>(n.args[0]) + n.arity <= prog.extra_args.size(),
            "extra_args slice out of bounds");
    }
    auto args = node_args(n, prog.extra_args);
    int reg = node_region[i];
    if (scheduled && i < prog.uniform_end) {
      Check(!is_spatially_varying_op(n.op) && n.op != Op::SampleInput,
            "varying op in uniform band");
    }
    for (ValueId a : args) {
      bool order_ok = reg < 0 ? a < i
                              : ((a >= prog.regions[reg].begin && a < i) ||
                                 (scheduled && a < prog.uniform_end));
      Check(order_ok, "operand violates topological order or region isolation");
      if (n.op != Op::MatRow) {
        Check(!is_matrix_producer(prog.nodes[a].op),
              "non-MatRow operand is matrix producer");
      }
    }
    if (int ew = elementwise_arity(n.op)) {
      Check(n.arity == ew, "elementwise arity mismatch");
    }
    switch (n.op) {
      case Op::Const:
        Check(n.arity == 0 && n.imm < prog.consts.size(), "invalid Const");
        break;
      case Op::Opaque:
        Check(n.arity == 0 && n.imm < prog.opaques.size(), "invalid Opaque");
        break;
      case Op::Attr:
        Check(n.arity == 1 && n.imm < prog.attr_names.size() &&
                  (n.flags == 1 || n.flags == 3),
              "invalid Attr");
        break;
      case Op::HeightToNormal:
        Check(n.arity == 1 && n.imm < prog.regions.size() &&
                  (reg < 0 || static_cast<int>(n.imm) < reg),
              "invalid HeightToNormal region");
        break;
      case Op::NormalMap:
        Check(n.arity == (n.imm == kNormalMapDefaultFrame ? 2 : 5),
              "invalid NormalMap arity");
        break;
      case Op::MatRow: {
        const Node& p = prog.nodes[args[0]];
        Check(n.arity == 1 && is_matrix_producer(p.op) && n.imm < p.channels &&
                  n.channels == p.channels,
              "invalid MatRow");
        if (scheduled) {
          bool row_unif = i < prog.uniform_end;
          uint32_t lim = row_unif  ? prog.uniform_mat_count
                         : reg < 0 ? prog.main_mat_count
                                   : prog.regions[reg].mat_count;
          Check((args[0] < prog.uniform_end) == row_unif &&
                    node_region[args[0]] == reg && p.imm < lim,
                "MatRow crosses scope or matrix slot out of bounds");
        }
        break;
      }
      default:
        break;
    }
  }
  if (prog.result_root != kInvalidValue) {
    Check(prog.result_root < prog.main_end &&
              !is_matrix_producer(prog.nodes[prog.result_root].op),
          "invalid result_root");
  }
}

void VerifyBsdfProgram(const MtlxBsdfProgram& prog, bool scheduled) {
  VerifyValueProgram(prog.values, scheduled);
  Check(prog.values.result_root == kInvalidValue && !prog.closures.empty(),
        "invalid BSDF program roots");
  if (prog.opacity_root != kInvalidValue) {
    uint32_t lim = scheduled ? prog.values.opacity_end : prog.values.main_end;
    Check(prog.opacity_root < lim &&
              !is_matrix_producer(prog.values.nodes[prog.opacity_root].op),
          "invalid opacity_root");
  }
  for (size_t i = 0; i < prog.closures.size(); ++i) {
    const ClosureNode& c = prog.closures[i];
    for_each_closure_value_id(c, [&](ValueId id) {
      Check(id < prog.values.main_end &&
                !is_matrix_producer(prog.values.nodes[id].op),
            "closure ValueId out of bounds");
    });
    if (is_combinator(c.op)) {
      Check(
          c.comb.first >= 0 && static_cast<size_t>(c.comb.first) < i &&
              (c.op == ClosureOp::Multiply ||
               (c.comb.second >= 0 && static_cast<size_t>(c.comb.second) < i)),
          "combinator child index out of bounds");
    }
  }
}

std::string DumpValueProgram(const MtlxValueProgram& prog) {
  std::ostringstream os;
  DumpBand(os, prog, "uniform", 0, prog.uniform_end);
  DumpBand(os, prog, "opacity", prog.uniform_end, prog.opacity_end);
  DumpBand(os, prog, "surface", prog.opacity_end, prog.main_end);
  for (size_t k = 0; k < prog.regions.size(); ++k) {
    const Region& r = prog.regions[k];
    std::string hdr = "region " + std::to_string(k) + " root=%" +
                      std::to_string(r.root) +
                      (r.has_texcoord ? " tc=1" : " tc=0");
    DumpBand(os, prog, hdr.c_str(), r.begin, r.end);
  }
  if (prog.result_root != kInvalidValue) {
    os << "result = %" << prog.result_root << "\n";
  }
  return os.str();
}

std::string DumpBsdfProgram(const MtlxBsdfProgram& prog) {
  std::ostringstream os;
  os << DumpValueProgram(prog.values) << "[closures]\n";
  for (size_t i = 0; i < prog.closures.size(); ++i) {
    const ClosureNode& c = prog.closures[i];
    os << "  #" << i << " = " << ClosureOpName(c.op) << "(";
    bool first = true;
    if (is_combinator(c.op)) {
      os << "#" << c.comb.first;
      if (c.comb.second >= 0) os << ", #" << c.comb.second;
      first = false;
    }
    for_each_closure_value_id(c, [&](ValueId id) {
      if (!first) os << ", ";
      os << "%" << id;
      first = false;
    });
    os << ")";
    if (c.flags & kClosureNeedsTp) os << " [needs_tp]";
    os << "\n";
  }
  if (prog.opacity_root != kInvalidValue) {
    os << "opacity = %" << prog.opacity_root << "\n";
  }
  if (prog.thin_walled) os << "thin_walled = true\n";
  return os.str();
}

OptimizeResult OptimizeValueProgram(MtlxValueProgram& prog) {
  VerifyValueProgram(prog, false);
  OptimizeResult res = PruneAndSchedule(prog, nullptr);
  VerifyValueProgram(prog, true);
  return res;
}

OptimizeResult OptimizeBsdfProgram(MtlxBsdfProgram& prog) {
  VerifyBsdfProgram(prog, false);
  SimplifyClosures(prog);
  VerifyBsdfProgram(prog, false);
  OptimizeResult res = PruneAndSchedule(prog.values, &prog);
  VerifyBsdfProgram(prog, true);
  return res;
}

}  // namespace mitsuba::mtlx
