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

CodePartition PartitionCodeSlice(absl::Span<const Inst> code,
                                 absl::Span<const uint8_t> tmp_channels,
                                 absl::Span<const int> root_slots,
                                 uint32_t& uniform_scalars) {
  CodePartition part;
  if (code.empty()) return part;
  size_t n_tmps = tmp_channels.size();
  part.tmp_kind.assign(n_tmps, TmpKind::Varying);
  part.uniform_offset.assign(n_tmps, 0);

  size_t stmt_begin = 0;
  for (size_t i = 0; i < code.size(); ++i) {
    Inst inst = code[i];
    if (!is_stmt_terminator(inst.op)) continue;

    size_t stmt_end = i + 1;
    absl::Span<const Inst> stmt(code.data() + stmt_begin,
                                stmt_end - stmt_begin);
    stmt_begin = stmt_end;

    bool varying = false;
    for (Inst s : stmt) {
      if (is_spatially_varying_op(s.op) || s.op == Op::SampleInput ||
          (s.op == Op::LoadTmp && part.tmp_kind[s.imm] == TmpKind::Varying)) {
        varying = true;
        break;
      }
    }
    if (varying) {
      part.varying_code.insert(part.varying_code.end(), stmt.begin(),
                               stmt.end());
      continue;
    }

    if (inst.op == Op::StoreTmp) {
      if (stmt.size() == 2 &&
          (stmt[0].op == Op::Const || stmt[0].op == Op::Opaque)) {
        part.tmp_kind[inst.imm] =
            stmt[0].op == Op::Const ? TmpKind::Const : TmpKind::Opaque;
        part.uniform_offset[inst.imm] = stmt[0].imm;
      } else {
        part.tmp_kind[inst.imm] = TmpKind::Uniform;
      }
    } else {
      for (uint8_t r = 0; r < inst.channels; ++r) {
        part.tmp_kind[inst.imm + r] = TmpKind::Uniform;
      }
    }
    part.invariant_code.insert(part.invariant_code.end(), stmt.begin(),
                               stmt.end());
  }

  if (stmt_begin < code.size()) {
    part.varying_code.insert(part.varying_code.end(), code.begin() + stmt_begin,
                             code.end());
  }

  // Prune intermediate uniform temporaries never read outside
  // `invariant_code` so they do not materialize unused hoisted buffers.
  std::vector<bool> live_out(n_tmps, false);
  for (Inst inst : part.varying_code) {
    if (inst.op == Op::LoadTmp) live_out[inst.imm] = true;
  }
  for (int slot : root_slots) live_out[slot] = true;
  uint32_t part_scalars = 0;
  for (size_t k = 0; k < n_tmps; ++k) {
    if (!live_out[k]) {
      part.tmp_kind[k] = TmpKind::Varying;
    } else if (part.tmp_kind[k] == TmpKind::Uniform) {
      part.uniform_offset[k] = uniform_scalars + part_scalars;
      part_scalars += tmp_channels[k];
    }
  }
  uniform_scalars += part_scalars;
  if (part_scalars == 0) part.invariant_code.clear();
  return part;
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

  uint32_t uniform_scalars = 0;
  prog.main_part = PartitionCodeSlice(prog.code, prog.tmp_channels, root_slots,
                                      uniform_scalars);
  prog.sub_parts.clear();
  prog.sub_parts.reserve(prog.sub_programs.size());
  for (const SubProgram& sub : prog.sub_programs) {
    absl::Span<const Inst> slice(prog.sub_code.data() + sub.code_begin,
                                 sub.code_end - sub.code_begin);
    absl::Span<const uint8_t> ch(
        prog.sub_tmp_channels.data() + sub.tmp_channels_begin, sub.tmp_count);
    prog.sub_parts.push_back(
        PartitionCodeSlice(slice, ch, {}, uniform_scalars));
  }
  prog.uniform_scalar_count = uniform_scalars;
}

}  // namespace mitsuba::mtlx
