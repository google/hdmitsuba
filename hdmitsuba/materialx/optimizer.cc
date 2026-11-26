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
#include <utility>
#include <vector>

#include <absl/types/span.h>

#include "hdmitsuba/materialx/program.h"

namespace mitsuba::mtlx {
namespace {

void PruneDeadCode(std::vector<Inst>& code, size_t tmp_count,
                   absl::Span<const int> root_slots) {
  if (code.empty() || tmp_count == 0) return;
  std::vector<bool> live_tmp(tmp_count, false);
  for (int slot : root_slots) live_tmp[slot] = true;

  std::vector<uint8_t> keep(code.size(), 0);
  // Any trailing expression after the final statement terminator is the root
  // expression and stays live.
  bool active = true;
  for (size_t i = code.size(); i-- > 0;) {
    Inst inst = code[i];
    if (is_stmt_terminator(inst.op)) {
      size_t n = inst.op == Op::StoreTmp ? 1 : inst.channels;
      active = std::any_of(live_tmp.begin() + inst.imm,
                           live_tmp.begin() + inst.imm + n,
                           [](bool b) { return b; });
    } else if (active && inst.op == Op::LoadTmp) {
      live_tmp[inst.imm] = true;
    }
    keep[i] = active ? 1 : 0;
  }

  size_t out = 0;
  for (size_t i = 0; i < code.size(); ++i) {
    if (keep[i]) code[out++] = code[i];
  }
  code.resize(out);
}

void CompactLiveOpaques(MtlxProgram& prog) {
  if (prog.opaques.empty()) return;
  std::vector<uint16_t> remap(prog.opaques.size(), kMaxImm);
  for (const auto* code : {&prog.code, &prog.sub_code}) {
    for (Inst in : *code) {
      if (in.op == Op::Opaque) remap[in.imm] = 0;
    }
  }
  size_t n = 0;
  for (size_t i = 0; i < prog.opaques.size(); ++i) {
    if (remap[i] == kMaxImm) continue;
    remap[i] = static_cast<uint16_t>(n);
    prog.opaques[n++] = prog.opaques[i];
  }
  if (n == prog.opaques.size()) return;
  prog.opaques.resize(n);
  for (auto* code : {&prog.code, &prog.sub_code}) {
    for (Inst& in : *code) {
      if (in.op == Op::Opaque) in.imm = remap[in.imm];
    }
  }
}

void PartitionCode(MtlxProgram& prog, absl::Span<const int> root_slots) {
  size_t n_tmps = prog.tmp_channels.size();
  prog.invariant_code.clear();
  prog.varying_code.clear();
  prog.tmp_kind.assign(n_tmps, TmpKind::Varying);
  prog.uniform_offset.assign(n_tmps, 0);

  size_t stmt_begin = 0;
  for (size_t i = 0; i < prog.code.size(); ++i) {
    Inst inst = prog.code[i];
    if (!is_stmt_terminator(inst.op)) continue;

    size_t stmt_end = i + 1;
    absl::Span<const Inst> stmt(prog.code.data() + stmt_begin,
                                stmt_end - stmt_begin);
    stmt_begin = stmt_end;

    bool varying = false;
    for (Inst s : stmt) {
      if (is_spatially_varying_op(s.op) || s.op == Op::SampleInput ||
          (s.op == Op::LoadTmp && prog.tmp_kind[s.imm] == TmpKind::Varying)) {
        varying = true;
        break;
      }
    }
    if (varying) {
      prog.varying_code.insert(prog.varying_code.end(), stmt.begin(),
                               stmt.end());
      continue;
    }

    if (inst.op == Op::StoreTmp) {
      if (stmt.size() == 2 &&
          (stmt[0].op == Op::Const || stmt[0].op == Op::Opaque)) {
        prog.tmp_kind[inst.imm] =
            stmt[0].op == Op::Const ? TmpKind::Const : TmpKind::Opaque;
        prog.uniform_offset[inst.imm] = stmt[0].imm;
      } else {
        prog.tmp_kind[inst.imm] = TmpKind::Uniform;
      }
    } else {
      for (uint8_t r = 0; r < inst.channels; ++r) {
        prog.tmp_kind[inst.imm + r] = TmpKind::Uniform;
      }
    }
    prog.invariant_code.insert(prog.invariant_code.end(), stmt.begin(),
                               stmt.end());
  }

  if (stmt_begin < prog.code.size()) {
    prog.varying_code.insert(prog.varying_code.end(),
                             prog.code.begin() + stmt_begin, prog.code.end());
  }

  // Prune intermediate uniform temporaries never read outside
  // `invariant_code` so they do not materialize unused hoisted buffers.
  std::vector<bool> live_out(n_tmps, false);
  for (Inst inst : prog.varying_code) {
    if (inst.op == Op::LoadTmp) live_out[inst.imm] = true;
  }
  for (int slot : root_slots) live_out[slot] = true;
  uint32_t uniform_scalars = 0;
  for (size_t k = 0; k < n_tmps; ++k) {
    if (!live_out[k]) {
      prog.tmp_kind[k] = TmpKind::Varying;
    } else if (prog.tmp_kind[k] == TmpKind::Uniform) {
      prog.uniform_offset[k] = uniform_scalars;
      uniform_scalars += prog.tmp_channels[k];
    }
  }
  prog.uniform_scalar_count = uniform_scalars;
  if (uniform_scalars == 0) prog.invariant_code.clear();
}

}  // namespace

void OptimizeProgram(MtlxProgram& prog) {
  std::vector<int> root_slots;
  if (prog.opacity_slot >= 0) root_slots.push_back(prog.opacity_slot);
  for (const ClosureNode& c : prog.closures) {
    for_each_closure_tmp_slot(c, [&](int slot) { root_slots.push_back(slot); });
  }
  PruneDeadCode(prog.code, prog.tmp_channels.size(), root_slots);

  if (!prog.sub_programs.empty()) {
    std::vector<bool> live_sub(prog.sub_programs.size(), false);
    auto mark_subs = [&](absl::Span<const Inst> code) {
      for (Inst inst : code) {
        if (inst.op == Op::HeightToNormal) live_sub[inst.imm] = true;
      }
    };
    mark_subs(prog.code);
    std::vector<Inst> new_sub_code;
    new_sub_code.reserve(prog.sub_code.size());
    // Subprograms are emitted post-order (callees before callers), so scanning
    // in reverse propagates liveness through nested `HeightToNormal` nodes.
    for (size_t i = prog.sub_programs.size(); i-- > 0;) {
      SubProgram& sub = prog.sub_programs[i];
      if (!live_sub[i]) {
        sub.code_begin = sub.code_end = 0;
        continue;
      }
      std::vector<Inst> slice(prog.sub_code.begin() + sub.code_begin,
                              prog.sub_code.begin() + sub.code_end);
      PruneDeadCode(slice, sub.tmp_count, {});
      mark_subs(slice);
      sub.code_begin = static_cast<uint32_t>(new_sub_code.size());
      new_sub_code.insert(new_sub_code.end(), slice.begin(), slice.end());
      sub.code_end = static_cast<uint32_t>(new_sub_code.size());
    }
    prog.sub_code = std::move(new_sub_code);
  }

  CompactLiveOpaques(prog);
  PartitionCode(prog, root_slots);
}

}  // namespace mitsuba::mtlx
