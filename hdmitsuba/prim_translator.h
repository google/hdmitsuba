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
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/pxr.h>

#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/texture_cache.h"

PXR_NAMESPACE_OPEN_SCOPE

// Prefix used for Mitsuba mesh vertex attributes created from USD primvars.
inline constexpr std::string_view kVertexAttributePrefix = "vertex_";

// Primvars that a material reads as mesh attributes, paired with the number of
// channels read (1 or 3). A primvar read with both channel counts appears
// twice.
using MeshAttributeRequests = std::set<std::pair<std::string, int>>;

// Returns the name of the Mitsuba vertex attribute holding `primvar` with
// `channels` channels. Mitsuba stores 3-channel attributes whose name contains
// "color" as rgb2spec coefficients in spectral variants, so "color" is
// capitalized to keep the raw values.
std::string MeshAttributeName(std::string_view primvar, int channels);

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const pxr::VtValue& val,
                                 bool invert_float = false);

// Invokes `callback` with the key of every texture the material parsers will
// look up for `network`, so the TextureCache can be preloaded beforehand.
void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey& tex_key)>& callback);

// Returns the upstream node path of the network's displacement terminal
// (`mitsuba:displacement` or `displacement`), or an empty path if none exists.
SdfPath FindDisplacementTerminal(const HdMaterialNetwork2& network);

// Displacement texture of a material.
struct MaterialDisplacement {
  mitsuba::ref<mitsuba::Object> texture = nullptr;
  // Scalar bias subtracted from 1D displacement values before scaling by the
  // surface normal (0.5 for [0, 1]-centered maps, 0.0 for signed displacement).
  float bias = 0.5f;
  // Whether evaluating `texture` requires UV coordinates ("st") on the mesh.
  bool requires_uv = true;
  // Whether `texture` returns a 3D vector displacement in
  // (dPdu, dPdv, N) tangent space, instead of a scalar along the normal.
  bool is_vector = false;

  bool operator==(const MaterialDisplacement& other) const {
    return texture == other.texture && bias == other.bias &&
           requires_uv == other.requires_uv && is_vector == other.is_vector;
  }
  bool operator!=(const MaterialDisplacement& other) const {
    return !(*this == other);
  }
};

struct TranslatedMaterial {
  mitsuba::ref<mitsuba::Object> bsdf = nullptr;
  std::optional<mitsuba::Properties> shape_emitter_props = std::nullopt;
  MaterialDisplacement displacement;
  // Primvars that the BSDF and shape emitter read as mesh attributes.
  MeshAttributeRequests mesh_attributes;
};

MI_VARIANT
class PrimTranslator {
 public:
  static TranslatedMaterial BuildMaterial(
      const MaterialSpec& spec,
      const TextureCache<Float, Spectrum>& texture_cache);

  struct TranslatedLight {
    mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape = nullptr;
    mitsuba::ref<mitsuba::Emitter<Float, Spectrum>> emitter = nullptr;
  };

  static TranslatedLight BuildLight(const LightSpec& spec);

  static mitsuba::ref<mitsuba::Object> CreateAreaEmitter(
      const GfVec3f& emission);

  static void UpdateLightInPlace(mitsuba::Object* light_obj,
                                 const LightSpec& spec);

  static mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> BuildSensor(
      const CameraSpec& spec, bool is_interactive = false);

  static void UpdateSensorInPlace(mitsuba::Object* sensor_obj,
                                  const CameraSpec& spec);

  // Builds a mesh from expanded per-vertex `primvars`. The primvars listed in
  // `mesh_attributes` are uploaded as vertex attributes named by
  // `MeshAttributeName`.
  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildMesh(
      const SdfPath& id, const VtIntArray& face_indices,
      const PrimvarMap& primvars, const MeshAttributeRequests& mesh_attributes,
      mitsuba::Object* bsdf, mitsuba::Object* emitter_ptr,
      mitsuba::Object* sensor_ptr);

  static void UpdateMeshInPlace(mitsuba::Object* mesh_obj,
                                const VtIntArray& face_indices,
                                const PrimvarMap& primvars,
                                const MeshAttributeRequests& mesh_attributes,
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
