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

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <mitsuba/core/properties.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

using pxr::CachedTexture;
using pxr::MeshAttributeRequests;
using pxr::TextureKey;

// Optional callback resolving an asset path to a filesystem path before
// falling back to `mitsuba::file_resolver()`.
using PathResolver = std::function<std::string(const std::string&)>;

// Callback looking up a preloaded texture in a `TextureCache`.
using TextureLookup = std::function<const CachedTexture*(const TextureKey&)>;

// Options and output attribute accumulator for MaterialX shader compilation.
struct MtlxCompileOptions {
  std::string_view variant;
  TextureLookup lookup_texture;
  PathResolver resolve_path;
  MeshAttributeRequests* mesh_attributes = nullptr;
};

// Collects `TextureKey`s for all `<image>` nodes in a flattened MaterialX
// document (including extra `channel = 3` alpha keys for 4-channel images).
void CollectMtlxImageKeys(
    const MaterialX::DocumentPtr& doc, const PathResolver& resolve_path,
    const std::function<void(const TextureKey&)>& callback);

// Expands all non-library compound nodegraphs in `doc` and at document scope.
void FlattenMtlxDocument(const MaterialX::DocumentPtr& doc);

// Compiles the emission (`edf`) of a flattened MaterialX `surface`,
// `surface_unlit`, or `surfacematerial` node (weighted by its `opacity`) into
// Mitsuba `area` emitter properties, or returns `std::nullopt` if there is no
// emission.
std::optional<Properties> CompileSurfaceEdf(
    const MaterialX::NodePtr& shader_node, const MtlxCompileOptions& opts);

}  // namespace mtlx
NAMESPACE_END(mitsuba)
