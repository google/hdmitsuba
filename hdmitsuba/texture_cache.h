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

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/hash/hash.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

// Identifies a unique Mitsuba bitmap texture instance. Two material inputs
// that resolve to the same key share one texture object. Defaults mirror the
// Mitsuba `bitmap` plugin so that unspecified parameters hash identically.
struct TextureKey {
  // Row-major 3x3 UV transform in Mitsuba's (column-vector) convention, i.e.
  // exactly what the bitmap plugin's `to_uv` parameter consumes.
  using UvMatrix = std::array<float, 9>;
  static constexpr UvMatrix kIdentityUv = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  std::string filename;
  bool raw = false;
  std::string wrap_mode = "repeat";
  std::string filter_type = "bilinear";
  int max_anisotropy = 8;
  std::string format = "auto";
  bool accel = true;
  UvMatrix to_uv = kIdentityUv;
  // Mitsuba's `bitmap` texture plugin only stores 1 or 3 channels and strips
  // the alpha channel when expanding RGBA/YA bitmaps. When true, the image's
  // alpha channel is extracted into a separate single-channel texture; images
  // without alpha yield no texture.
  bool alpha_channel = false;

 private:
  // Single source of truth for equality and hashing. Add new fields here.
  auto Fields() const {
    return std::tie(filename, raw, wrap_mode, filter_type, max_anisotropy,
                    format, accel, to_uv, alpha_channel);
  }

 public:
  bool operator==(const TextureKey& other) const {
    return Fields() == other.Fields();
  }

  template <typename H>
  friend H AbslHashValue(H h, const TextureKey& k) {
    return H::combine(std::move(h), k.Fields());
  }
};

// A loaded texture. `texture` is null if the image failed to load.
struct CachedTexture {
  mitsuba::ref<mitsuba::Object> texture = nullptr;
  size_t channel_count = 0;
};

// Builds the cache key for a `UsdUVTexture` or `mitsuba_bitmap` node connected
// to `input_name`. Returns nullopt if the node has no usable file path.
std::optional<TextureKey> ExtractTextureKey(
    const std::map<TfToken, VtValue>& parameters, const TfToken& nodeTypeId,
    const TfToken& input_name);

// Loads a bitmap through USD's asset resolver so textures embedded in .usdz
// packages (or served by custom resolvers) can be read from memory. Falls back
// to reading directly from the filesystem. Throws on loading error.
mitsuba::ref<mitsuba::Bitmap> ReadBitmap(const std::string& path);

// Owns the Mitsuba bitmap textures shared between materials, keyed by
// TextureKey. Textures are loaded in bulk by Preload() before materials are
// built and evicted by GarbageCollect() once no material references them.
MI_VARIANT
class TextureCache {
 public:
  // Returns the entry for `key`, or nullptr if it was never preloaded.
  const CachedTexture* Find(const TextureKey& key) const;

  // Loads every key that is not yet cached. Each unique image file is decoded
  // once, then each unique key is instantiated from the shared in-memory
  // bitmap; both stages run in parallel. Failed loads are cached as empty
  // entries (already warned about) so lookups can fall back silently; the next
  // GarbageCollect() evicts them.
  void Preload(const absl::flat_hash_set<TextureKey>& keys);

  // Flags that material networks changed, so the next GarbageCollect() runs.
  void MarkDirty() { dirty_ = true; }

  // Evicts textures that are no longer referenced by any material. No-op
  // unless Preload() ran or MarkDirty() was called since the last collection.
  //
  // Invariant: the cache holds exactly one `ref` per texture and every other
  // long-lived holder (BSDFs, displacement textures, emitter `Properties`)
  // also holds an owning `ref`. Hence `ref_count() == 1` means "only the
  // cache references it". This must run after all per-commit temporaries
  // (e.g. `TranslatedMaterial` results) have been destroyed, and nothing may
  // stash a raw `Texture*` to a cached texture.
  void GarbageCollect();

 private:
  // Like ReadBitmap(), but warns and returns null on failure.
  static mitsuba::ref<mitsuba::Bitmap> LoadBitmap(const std::string& filename);
  // Instantiates the Mitsuba texture plugin described by `key` from an already
  // decoded bitmap. Warns and returns an empty entry on failure.
  static CachedTexture LoadTexture(const TextureKey& key,
                                   const mitsuba::Bitmap& bitmap);

  absl::flat_hash_map<TextureKey, CachedTexture> textures_;
  bool dirty_ = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
