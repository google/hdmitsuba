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

#include <pxr/imaging/hd/material.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include <functional>
#include <string_view>

#include "hdmitsuba/materialx/compiler.h"
#include "hdmitsuba/prim_translator.h"

PXR_NAMESPACE_OPEN_SCOPE

// Returns true if `network` contains a MaterialX shading network (e.g.
// nodes with `ND_*` MaterialX node definitions or `standard_surface`,
// `open_pbr_surface`, `gltf_pbr`).
bool IsMaterialXNetwork(const HdMaterialNetwork2& network);

// Discovers all bitmap textures referenced by a MaterialX shading network
// (including textures inside flattened compound nodegraphs like `<tiledimage>`)
// and invokes `callback` with their `TextureKey` for parallel preloading.
void DiscoverMaterialXTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey&)>& callback);

// Translates a Hydra MaterialX network for a specific Mitsuba variant string.
TranslatedMaterial TranslateMaterialXNetwork(
    const SdfPath& id, const HdMaterialNetwork2& network,
    const mitsuba::mtlx::TextureLookup& lookup_texture,
    std::string_view variant);

PXR_NAMESPACE_CLOSE_SCOPE
