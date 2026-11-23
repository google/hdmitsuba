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
#include <string>
#include <vector>

#include <absl/types/span.h>
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

// Evaluates an `MtlxProgram` for the variant `Float`/`Spectrum`, given its
// texture inputs.
template <typename Float, typename Spectrum>
class MtlxEvaluator {
 public:
  MI_IMPORT_TYPES(Texture)
  // Values have up to four channels. A 1-channel value is replicated to all
  // of them, and the unused channels of other values are unspecified.
  using Val = Vector<Float, 4>;

  MtlxEvaluator();
  ~MtlxEvaluator();

  void compile(const Properties& props);
  void load_srgb_lut();

  const MtlxProgram& program() const { return m_prog; }

  // Reports texture inputs ("in<i>") and opaque parameters ("param<i>").
  void traverse(TraversalCallback* cb);

  void parameters_changed();

  // Returns the indented code size and texture inputs, one per line, for
  // `MtlxExprTexture::to_string()` and `MtlxBsdf::to_string()`.
  std::string to_string() const;

  // Returns true if the program reads geometric intrinsics or samples a
  // spatially varying (or re-parameterized) input.
  bool is_spatially_varying() const;

  // Evaluates the program and returns its result. If `out_tmps` is given, it
  // receives the evaluated temporaries; if `stop_after_tmp >= 0`, evaluation
  // returns right after storing temporary `stop_after_tmp`.
  Val eval(const SurfaceInteraction3f& si, Mask active, Val* out_tmps = nullptr,
           int stop_after_tmp = -1) const;

  // Returns the luminance of 3-channel results, like Mitsuba's `bitmap`.
  Float eval_1(const SurfaceInteraction3f& si, Mask active) const;
  Color3f eval_3(const SurfaceInteraction3f& si, Mask active) const;

  // Upsamples an RGB color to a spectrum using the traversable sRGB LUT.
  UnpolarizedSpectrum upsample_rgb(const Color3f& c,
                                   const Wavelength& wavelengths,
                                   bool unbounded) const;

  void traverse_cb(void* payload, const drjit::TraverseVisitor& cb);

 private:
  Val eval_code(absl::Span<const Inst> code, size_t tmp_count,
                uint32_t stack_size, const SurfaceInteraction3f& si,
                Mask active, Val* out_tmps = nullptr,
                int stop_after_tmp = -1) const;

  MtlxProgram m_prog;
  std::vector<ref<Texture>> m_inputs;
  // Opaque scalars (`dr::opaque`), passed as kernel inputs rather than
  // literals.
  std::vector<Float> m_opaques;
  std::unique_ptr<detail::SrgbUpsampler<Float, Spectrum>> m_srgb;
};

}  // namespace mtlx
NAMESPACE_END(mitsuba)
