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

// The `materialx_expr` textures and mesh attributes of a compiled surface
// shader.
struct CompiledSurfaceShader {
  // Emitted radiance, or null if there is no emission.
  mitsuba::ref<mitsuba::Object> radiance_texture;
  // Cutout opacity, or null if it is identically 1.
  mitsuba::ref<mitsuba::Object> opacity_texture;
  // Primvars read as mesh attributes by the compiled shader.
  MeshAttributeRequests mesh_attributes;
};

// Compiles the emission (`edf`, weighted by `opacity`) and cutout `opacity` of
// a flattened MaterialX surface shader (`surface`, `surface_unlit`, `mix`,
// `surfacematerial`, or raw EDF closures). This reads the node definitions of
// the data library of the document, which must not be modified concurrently.
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
