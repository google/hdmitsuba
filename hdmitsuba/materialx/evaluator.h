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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <absl/container/inlined_vector.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

#include "hdmitsuba/materialx/program.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

namespace detail {
template <typename Float, typename Spectrum>
class SrgbUpsampler;
}  // namespace detail

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
  // per ray; uniform nodes `[0 .. uniform_end)` live in `m_uniform_scalars`
  // (except under AD on `m_opaques`, where width-1 attached values are stored
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

  MtlxEvaluator();
  ~MtlxEvaluator();

  void compile(const Properties& props);
  void compile(const Properties& props, MtlxValueProgram prog);
  void load_srgb_lut();

  const MtlxValueProgram& program() const { return m_prog; }
  void traverse(TraversalCallback* cb);
  void parameters_changed();
  std::string to_string() const;
  bool is_spatially_varying() const;

  Val read_val(ValueId id, const EvalState& state) const;
  Val eval_opacity(ValueId opacity_root, const SurfaceInteraction3f& si,
                   Mask active, EvalState& state) const;
  void eval_surface(const SurfaceInteraction3f& si, Mask active,
                    EvalState& state) const;
  Val eval_expr(const SurfaceInteraction3f& si, Mask active) const;

  // Returns the luminance of 3-channel results, like Mitsuba's `bitmap`.
  Float eval_1(const SurfaceInteraction3f& si, Mask active) const;
  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const;
  UnpolarizedSpectrum upsample_rgb(const Color3f& c,
                                   const Wavelength& wavelengths,
                                   bool unbounded) const;

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb);

 private:
  bool has_opaque_grad() const;
  void update_uniform_state();
  Val load_uniform(ValueId id) const;
  void eval_uniform_nodes(absl::InlinedVector<Val, 32>& uvals) const;
  void eval_until(uint32_t end, const SurfaceInteraction3f& si, Mask active,
                  EvalState& state) const;
  void eval_range(uint32_t begin, uint32_t end, uint32_t base_offset,
                  Val* out_vals, Val* mat_rows, const Val* uniform_vals,
                  const SurfaceInteraction3f& si, Mask active,
                  absl::InlinedVector<std::optional<CachedAttr>, 4>& attr_cache,
                  std::optional<SurfaceInteraction3f>& si_uv) const;

  MtlxValueProgram m_prog;
  std::vector<ref<Texture>> m_inputs;
  // Opaque scalars (`dr::opaque`), passed as kernel inputs rather than
  // literals.
  std::vector<Float> m_opaques;
  // Hoisted size-1 evaluated spatially-invariant scalars.
  std::vector<Float> m_uniform_scalars;
  // Precomputed finite-difference UV step per `HeightToNormal` region.
  std::vector<ScalarPoint2f> m_region_eps;
  std::unique_ptr<detail::SrgbUpsampler<Float, Spectrum>> m_srgb;
};

}  // namespace mtlx
NAMESPACE_END(mitsuba)
