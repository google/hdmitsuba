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

#include <functional>
#include <optional>
#include <string_view>

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/pxr.h>

#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/texture_cache.h"

PXR_NAMESPACE_OPEN_SCOPE

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const pxr::VtValue& val,
                                 bool invert_float = false);

// Invokes `callback` with the key of every texture the material parsers will
// look up for `network`, so the TextureCache can be preloaded beforehand.
void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey& tex_key)>& callback);

MI_VARIANT
class PrimTranslator {
 public:
  struct TranslatedMaterial {
    mitsuba::ref<mitsuba::Object> bsdf = nullptr;
    std::optional<mitsuba::Properties> shape_emitter_props = std::nullopt;
    mitsuba::ref<mitsuba::Object> displacement_texture = nullptr;
  };

  static TranslatedMaterial BuildMaterial(
      const MaterialSpec& spec,
      const TextureCache<Float, Spectrum>& texture_cache);
  static void UpdateMaterialInPlace(
      mitsuba::Object* bsdf, const MaterialSpec& spec,
      const TextureCache<Float, Spectrum>& texture_cache);

  struct TranslatedLight {
    mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape = nullptr;
    mitsuba::ref<mitsuba::Emitter<Float, Spectrum>> emitter = nullptr;
  };

  static TranslatedLight BuildLight(const LightSpec& spec);

  static void UpdateLightInPlace(mitsuba::Object* light_obj,
                                 const LightSpec& spec);

  static mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> BuildSensor(
      const CameraSpec& spec, bool is_interactive = false);

  static void UpdateSensorInPlace(mitsuba::Object* sensor_obj,
                                  const CameraSpec& spec);

  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildMesh(
      const SdfPath& id, const VtIntArray& face_indices,
      const PrimvarMap& primvars, mitsuba::Object* bsdf,
      mitsuba::Object* emitter_ptr, mitsuba::Object* sensor_ptr);

  static void UpdateMeshInPlace(mitsuba::Object* mesh_obj,
                                const VtIntArray& face_indices,
                                const PrimvarMap& primvars,
                                HdDirtyBits dirty_bits);

  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildCurves(
      const CurveSpec& spec, mitsuba::Object* bsdf);

  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildParticleField(
      const ParticleFieldSpec& spec);

  static void UpdateParticleFieldInPlace(mitsuba::Object* shape_obj,
                                         const ParticleFieldSpec& spec);

 private:
  static mitsuba::Properties BuildLightProperties(const LightSpec& spec);

  static mitsuba::ref<mitsuba::BSDF<Float, Spectrum>> DefaultBsdf(
      std::string_view id_str);

  static TranslatedMaterial ParsePreviewSurface(
      const HdMaterialNetwork2& network2,
      const HdMaterialNode2& preview_surface_node,
      const TextureCache<Float, Spectrum>& texture_cache);
};

PXR_NAMESPACE_CLOSE_SCOPE
