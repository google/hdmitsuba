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

#include <cstdint>
#include <string>
#include <vector>

namespace mitsuba::mtlx {

struct MtlxValueProgram;
struct MtlxBsdfProgram;

// Compaction tables returned by `OptimizeValueProgram` / `OptimizeBsdfProgram`
// so the compiler can compact parallel input textures and mesh attribute
// requests to only live entries in original order.
struct OptimizeResult {
  // Maps compacted texture input index -> original texture input index.
  std::vector<uint16_t> live_inputs;
  // Maps compacted mesh attribute index -> original mesh attribute index.
  std::vector<uint16_t> live_attrs;
};

// Runs `PruneAndSchedule` and verifies the resulting SSA value program.
OptimizeResult OptimizeValueProgram(MtlxValueProgram& prog);

// Runs `SimplifyClosures -> PruneAndSchedule` and verifies the resulting SSA
// BSDF program.
OptimizeResult OptimizeBsdfProgram(MtlxBsdfProgram& prog);

// Validates SSA, region, schedule, and closure invariants. Throws
// `std::logic_error` on any violation.
void VerifyValueProgram(const MtlxValueProgram& prog, bool scheduled = true);
void VerifyBsdfProgram(const MtlxBsdfProgram& prog, bool scheduled = true);

// Formats a value or BSDF program in human-readable SSA form grouped by
// `[uniform]` / `[opacity]` / `[surface]` / `[region]` / `[closures]` bands.
std::string DumpValueProgram(const MtlxValueProgram& prog);
std::string DumpBsdfProgram(const MtlxBsdfProgram& prog);

}  // namespace mitsuba::mtlx
