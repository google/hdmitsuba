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
#include <memory>
#include <string_view>

#include <MaterialXCore/Node.h>
#include <mitsuba/core/object.h>
#include <pxr/pxr.h>

#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

// Opaque variant-independent symbolic SSA programs lowered once from a
// MaterialX document (BSDF, optional emission, and optional displacement).
struct PreparedMaterialX;

// Instantiated Mitsuba objects and mesh attribute requests produced from a
// `PreparedMaterialX` for a specific Mitsuba variant.
struct CompiledMaterialXResult {
  // Compiled `materialx_bsdf` BSDF (including cutout opacity), or null.
  mitsuba::ref<mitsuba::Object> bsdf;
  // Emitted radiance `materialx_expr` texture, or null if there is no emission.
  mitsuba::ref<mitsuba::Object> radiance_texture;
  // Primvars read as mesh attributes by the BSDF and radiance shaders.
  MeshAttributeRequests mesh_attributes;
  // Compiled displacement texture and metadata (empty if no displacement).
  MaterialDisplacement displacement;
};

// Lowers the MaterialX surface (BSDF + EDF) and displacement shaders once into
// variant-independent symbolic SSA programs, sharing a single graph resolver,
// and invokes `callback` with the `TextureKey` of every live bitmap texture
// consumed by the lowered programs.
//
// If `check_surface_displacement` is true and `surface_out` is a material
// output that unwraps to a `displacementshader` (via `surfacematerial`), lowers
// that displacement shader as well. Returns nullptr if neither surface nor
// displacement is present.
std::shared_ptr<const PreparedMaterialX> LowerMaterialXShaders(
    const MaterialX::OutputPtr& surface_out,
    const MaterialX::OutputPtr& displacement_out,
    bool check_surface_displacement,
    const std::function<void(const TextureKey&)>& callback);

// Resolves symbolic image samples against `lookup_texture`, optimizes, and
// instantiates the Mitsuba plugins for `variant`. Does not mutate `prepared`.
CompiledMaterialXResult InstantiateMaterialXShaders(
    const PreparedMaterialX& prepared, std::string_view variant,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture);

PXR_NAMESPACE_CLOSE_SCOPE
