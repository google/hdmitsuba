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
#include <optional>
#include <string_view>

#include <pxr/imaging/hd/material.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>

#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/texture_cache.h"

PXR_NAMESPACE_OPEN_SCOPE

// Opaque flattened MaterialX surface network shared between texture discovery
// and material translation within a single Hydra commit.
struct PreparedMaterialX;

// Flattens a Hydra MaterialX shading network and invokes `callback` with the
// `TextureKey` of every bitmap texture it consumes for parallel preloading.
// Returns nullptr if MaterialX is disabled or `network` has no MaterialX
// surface terminal.
std::shared_ptr<const PreparedMaterialX> PrepareMaterialXNetwork(
    const SdfPath& id, const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey&)>& callback);

// Translates a prepared MaterialX network for a specific Mitsuba variant.
TranslatedMaterial TranslateMaterialXNetwork(
    const SdfPath& id, const PreparedMaterialX& prepared,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture,
    std::string_view variant);

// --- Offline translation (usd_mitsuba, via geometry_ext) ---

// Translates a `UsdShadeMaterial` prim for `variant`, or returns `std::nullopt`
// if MaterialX is disabled or `prim` has no active MaterialX surface shader.
std::optional<TranslatedMaterial> TranslateUsdMaterialX(
    const UsdPrim& prim, UsdTimeCode time, std::string_view variant);

PXR_NAMESPACE_CLOSE_SCOPE
