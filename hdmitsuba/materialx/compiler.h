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
#include <string_view>

#include <MaterialXCore/Node.h>
#include <mitsuba/core/object.h>
#include <pxr/pxr.h>

#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

// The compiled `materialx_bsdf`, optional `materialx_expr` radiance texture,
// and mesh attributes of a MaterialX surface shader.
struct CompiledSurfaceShader {
  // Compiled `materialx_bsdf` BSDF (including cutout opacity).
  mitsuba::ref<mitsuba::Object> bsdf;
  // Emitted radiance, or null if there is no emission.
  mitsuba::ref<mitsuba::Object> radiance_texture;
  // Primvars read as mesh attributes by the compiled shader.
  MeshAttributeRequests mesh_attributes;
};

// Compiles the BSDF closures (including cutout `opacity`) and emission (`edf`,
// weighted by `opacity`) of a flattened MaterialX surface shader (`surface`,
// `surface_unlit`, `mix`, `surfacematerial`, or raw BSDF/EDF closures).
CompiledSurfaceShader CompileSurfaceShader(
    const MaterialX::NodePtr& shader_node, std::string_view variant,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture);

// Dry-run compiles the active branches of `shader_node` and invokes `callback`
// with the `TextureKey` of every bitmap texture it consumes.
void DiscoverSurfaceShaderTextures(
    const MaterialX::NodePtr& shader_node,
    const std::function<void(const TextureKey&)>& callback);

PXR_NAMESPACE_CLOSE_SCOPE
