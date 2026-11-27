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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

#include "hdmitsuba/materialx/program.h"
#include "hdmitsuba/materialx/srgb_lut.h"

namespace mitsuba::mtlx {

// Evaluates a flat SSA `MtlxValueProgram` for the variant `Float`/`Spectrum`,
// given its texture inputs. Used standalone by `MtlxExprTexture` and embedded
// by `MtlxBsdf`.
template <typename Float, typename Spectrum>
class MtlxEvaluator {
 public:
  MI_IMPORT_TYPES(Texture)
  // Values have up to four channels. A 1-channel value is replicated to all
  // of them, and the unused channels of other values are unspecified.
  using Val = Vector<Float, 4>;

  struct CachedAttr {
    Mask has;
    Val val;
  };

  // Per-query evaluation buffers shared across the opacity and surface stages
  // of a BSDF query. Only varying nodes (`main_end - uniform_end`) are stored
  // per ray; uniform nodes `[0 .. uniform_end)` live in `uniform_scalars_`
  // (except under AD on `opaques_`, where width-1 attached values are stored
  // in `uniform_vals`).
  struct EvalState {
    absl::InlinedVector<Val, 32> uniform_vals;
    absl::InlinedVector<Val, 64> varying_vals;
    absl::InlinedVector<Val, 4> mat_rows;
    absl::InlinedVector<std::optional<CachedAttr>, 4> attr_cache;
    std::optional<SurfaceInteraction3f> si_uv;
    uint32_t evaluated_end = 0;
    bool uniform_ready = false;
  };

  void Compile(const Properties& props);
  void Compile(const Properties& props, MtlxValueProgram prog);
  // Registers a deduplicated spectrum slot (loading the sRGB LUT and reserving
  // a hoisted uniform slot when spatially invariant on spectral variants).
  // Call `FinalizeUniforms()` once all slots are registered.
  int AddSpectrum(ValueId id, bool unbounded = false);
  size_t NumSpectra() const { return spec_slots_.size(); }
  void FinalizeUniforms();

  const MtlxValueProgram& Program() const { return prog_; }
  void Traverse(TraversalCallback* cb);
  void ParametersChanged();
  std::string ToString() const;
  bool IsSpatiallyVarying() const;

  Val ReadVal(ValueId id, const EvalState& state) const;
  UnpolarizedSpectrum EvalSpectrum(size_t slot_idx,
                                   const Wavelength& wavelengths,
                                   const EvalState& state) const;
  Val EvalOpacity(ValueId opacity_root, const SurfaceInteraction3f& si,
                  Mask active, EvalState& state) const;
  void EvalSurface(const SurfaceInteraction3f& si, Mask active,
                   EvalState& state) const;
  Val EvalExpr(const SurfaceInteraction3f& si, Mask active) const;

  // Returns the luminance of 3-channel results, like Mitsuba's `bitmap`.
  Float Eval1(const SurfaceInteraction3f& si, Mask active) const;
  Color3f Eval3(const SurfaceInteraction3f& si, Mask active) const;

  void TraverseCb(void* payload, const drjit::TraverseVisitor& cb);

 private:
  struct SpecSlot {
    ValueId value;
    bool unbounded;
    // Index into `uniform_specs_` for spatially-invariant non-scalar spectral
    // slots, or -1 otherwise.
    int uniform_spec_idx = -1;
  };

  struct UniformSpec {
    dr::Array<Float, 3> coeff;
    Float scale;
    DRJIT_STRUCT(UniformSpec, coeff, scale)
  };

  bool HasOpaqueGrad() const;
  Val LoadUniform(ValueId id) const;
  void UpdateUniformState();
  void EvalUniformNodes(absl::InlinedVector<Val, 32>& uvals) const;
  void EvalUntil(uint32_t end, const SurfaceInteraction3f& si, Mask active,
                 EvalState& state) const;
  void EvalRange(uint32_t begin, uint32_t end, uint32_t base_offset,
                 Val* out_vals, Val* mat_rows, const Val* uniform_vals,
                 const SurfaceInteraction3f& si, Mask active,
                 absl::InlinedVector<std::optional<CachedAttr>, 4>& attr_cache,
                 std::optional<SurfaceInteraction3f>& si_uv) const;

  MtlxValueProgram prog_;
  std::vector<ref<Texture>> inputs_;
  // Opaque scalars (`dr::opaque`), passed as kernel inputs rather than
  // literals.
  std::vector<Float> opaques_;
  // Hoisted size-1 evaluated spatially-invariant scalars.
  std::vector<Float> uniform_scalars_;
  // Deduplicated spectral slots and hoisted detached upsampling coefficients
  // (spectral variants only).
  std::vector<SpecSlot> spec_slots_;
  std::vector<UniformSpec> uniform_specs_;
  // Precomputed finite-difference UV step per `HeightToNormal` region.
  std::vector<ScalarPoint2f> region_eps_;
  detail::SrgbUpsampler<Float, Spectrum> srgb_;
};

}  // namespace mitsuba::mtlx
