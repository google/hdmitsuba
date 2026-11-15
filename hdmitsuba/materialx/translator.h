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

#include <optional>
#include <string_view>

#include <pxr/imaging/hd/material.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>

#include "hdmitsuba/prim_translator.h"

PXR_NAMESPACE_OPEN_SCOPE

// Returns true if experimental MaterialX support is enabled and the active
// surface terminal of `network` is a MaterialX shader node.
bool IsMaterialXNetwork(const HdMaterialNetwork2& network);

// Translates a Hydra MaterialX network for a specific Mitsuba variant string.
TranslatedMaterial TranslateMaterialXNetwork(const SdfPath& id,
                                             const HdMaterialNetwork2& network,
                                             std::string_view variant);

// --- Offline translation (usd_mitsuba, via geometry_ext) ---

// Translates a `UsdShadeMaterial` prim for `variant`, or returns `std::nullopt`
// if MaterialX is disabled or `prim` has no active MaterialX surface shader.
std::optional<TranslatedMaterial> TranslateUsdMaterialX(
    const UsdPrim& prim, UsdTimeCode time, std::string_view variant);

PXR_NAMESPACE_CLOSE_SCOPE
