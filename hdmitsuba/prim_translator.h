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
#include "hdmitsuba/traversal.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

// Returns the connection of the `mitsuba:<terminal>` terminal, falling back to
// `<terminal>`, or nullptr if neither is connected. A displacement terminal
// driven by a `UsdPreviewSurface` resolves to that shader's
// `inputs:displacement` connection (nullptr if unconnected).
const HdMaterialConnection2* FindTerminalConnection(
    const HdMaterialNetwork2& network, const TfToken& terminal);

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const pxr::VtValue& val,
                                 bool invert_float = false);

// Invokes `callback` with the key of every texture the material parsers will
// look up for `network`, so the TextureCache can be preloaded beforehand.
void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey& tex_key)>& callback);

// Displacement texture of a material.
struct MaterialDisplacement {
  mitsuba::ref<mitsuba::Object> texture = nullptr;
  // Whether evaluating `texture` requires UV coordinates ("st") on the mesh.
  bool requires_uv = true;
  // Whether `texture` returns a 3D vector displacement in
  // (dPdu, dPdv, N) tangent space, instead of a scalar along the normal.
  bool is_vector = false;

  bool operator==(const MaterialDisplacement& other) const {
    return texture == other.texture && requires_uv == other.requires_uv &&
           is_vector == other.is_vector;
  }
  bool operator!=(const MaterialDisplacement& other) const {
    return !(*this == other);
  }
};

struct TranslatedMaterial {
  mitsuba::ref<mitsuba::Object> bsdf = nullptr;
  std::optional<mitsuba::Properties> shape_emitter_props = std::nullopt;
  MaterialDisplacement displacement;
  mitsuba::ref<mitsuba::Object> interior_medium = nullptr;
  // Primvars that the BSDF and shape emitter read as mesh attributes.
  MeshAttributeRequests mesh_attributes;
};

MI_VARIANT
class PrimTranslator {
 public:
  // Sets the "to_world" property to a static or animated transform.
  static void SetTransformProperty(mitsuba::Properties& props,
                                   const MotionTransform& transform);

  // Updates the "to_world" parameter of the object traversed by `cb`, which
  // must already hold a transform of the same kind (static or animated). The
  // caller is responsible for notifying the object of the change.
  static void SetTransformParameter(TraversalCallback& cb,
                                    const MotionTransform& transform);

  // Updates a child spectrum plugin's "<name>.value" parameter (e.g.
  // "radiance", "intensity", or "irradiance") and notifies that child spectrum
  // object via `parameters_changed()` so its internal state stays opaque for
  // Dr.Jit kernel freezing and spectral upsampling coefficients are refreshed.
  // The caller remains responsible for notifying the parent emitter.
  static void SetSpectrumParameter(TraversalCallback& cb, std::string_view name,
                                   const GfVec3f& rgb);

  static TranslatedMaterial BuildMaterial(
      const MaterialSpec& spec,
      const TextureCache<Float, Spectrum>& texture_cache);

  struct TranslatedLight {
    mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape = nullptr;
    mitsuba::ref<mitsuba::Emitter<Float, Spectrum>> emitter = nullptr;
  };

  static TranslatedLight BuildLight(const LightSpec& spec);

  // Creates the `area` emitter of `spec`, wrapped in an `ies` emitter if `spec`
  // has an IES profile.
  static mitsuba::ref<mitsuba::Object> CreateAreaEmitter(const LightSpec& spec);

  // Updates the radiance and IES light frame of an emitter created by
  // `CreateAreaEmitter`.
  static void UpdateAreaEmitterInPlace(
      mitsuba::Emitter<Float, Spectrum>* emitter, const LightSpec& spec);

  static void UpdateLightInPlace(mitsuba::Object* light_obj,
                                 const LightSpec& spec);

  static mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> BuildSensor(
      const CameraSpec& spec, bool is_interactive = false);

  static void UpdateSensorInPlace(mitsuba::Object* sensor_obj,
                                  const CameraSpec& spec);

  // Builds a mesh from expanded per-vertex `primvars`. The primvars listed in
  // `mesh_attributes` are uploaded as vertex attributes named by
  // `MeshAttributeName`. `visibility` is the Mitsuba ray visibility.
  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildMesh(
      const std::string& id, const VtIntArray& face_indices,
      const VertexPrimvarMap& primvars,
      const MeshAttributeRequests& mesh_attributes, mitsuba::Object* bsdf,
      mitsuba::Object* emitter_ptr, mitsuba::Object* sensor_ptr,
      const std::string& visibility,
      mitsuba::Object* interior_medium = nullptr);

  static void UpdateMeshInPlace(mitsuba::Object* mesh_obj,
                                const VtIntArray& face_indices,
                                const VertexPrimvarMap& primvars,
                                const MeshAttributeRequests& mesh_attributes,
                                HdDirtyBits dirty_bits);

  // Adds the primvars in `mesh_attributes` that exist in the expanded
  // per-vertex `primvars` to the Mitsuba mesh `mesh_obj` as vertex attributes,
  // like `BuildMesh`.
  static void AttachMeshAttributes(
      mitsuba::Object* mesh_obj, const VertexPrimvarMap& primvars,
      const MeshAttributeRequests& mesh_attributes);

  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildCurves(
      const CurveSpec& spec, mitsuba::Object* bsdf);

  static mitsuba::ref<mitsuba::Shape<Float, Spectrum>> BuildParticleField(
      const ParticleFieldSpec& spec);

  static void UpdateParticleFieldInPlace(mitsuba::Object* shape_obj,
                                         const ParticleFieldSpec& spec);

 private:
  static mitsuba::Properties BuildLightProperties(const LightSpec& spec);

  static mitsuba::ref<mitsuba::BSDF<Float, Spectrum>> DefaultBsdf(
      std::string_view id_str, std::string_view plugin_name = "diffuse");

  static TranslatedMaterial ParsePreviewSurface(
      const HdMaterialNetwork2& network2,
      const HdMaterialNode2& preview_surface_node,
      const TextureCache<Float, Spectrum>& texture_cache);
};

PXR_NAMESPACE_CLOSE_SCOPE
