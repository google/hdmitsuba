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

#include "hdmitsuba/texture_cache.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <drjit-core/jit.h>
#include <drjit/matrix.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/mstream.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/texture.h>
#include <nanothread/nanothread.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

// Converts the source color space to the Mitsuba Bitmap's `raw` flag.
// Non-color inputs are classified based on the UsdPreviewSurface's
// specification.
bool UseRawBitmap(const TfToken& source_color_space,
                  const TfToken& input_name) {
  static const absl::NoDestructor<
      absl::flat_hash_set<TfToken, TfToken::HashFunctor>>
      kNonColorInputs({
          TfToken("normal"),
          TfToken("normalmap"),
          TfToken("bump"),
          TfToken("bumpmap"),
          TfToken("roughness"),
          TfToken("metallic"),
          TfToken("displacement"),
          TfToken("specular"),
          TfToken("clearcoat"),
          TfToken("clearcoatRoughness"),
          TfToken("ior"),
          TfToken("opacity"),
      });
  const bool is_non_color = kNonColorInputs->contains(input_name);
  if (source_color_space == TfToken("auto")) {
    return is_non_color;
  } else {
    return (source_color_space == TfToken("raw")) && is_non_color;
  }
}

// Converts a matrix-valued `to_uv` shader parameter to the 3x3 Mitsuba-space
// matrix stored in TextureKey. 4x4 inputs are reduced with Transform::extract(),
// which is what Properties::get<Transform3f>() would do with them anyway.
std::optional<TextureKey::UvMatrix> UvMatrixFromValue(const VtValue& val) {
  ScalarAffineTransform3f t;
  if (auto t3 = ExtractTransform3f(val)) {
    t = *t3;
  } else if (auto t4 = ExtractTransform4f(val)) {
    t = t4->extract();
  } else {
    return std::nullopt;
  }
  TextureKey::UvMatrix out;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i * 3 + j] = t.matrix(i, j);
  return out;
}

ScalarAffineTransform3f UvMatrixToTransform(const TextureKey::UvMatrix& m) {
  return ScalarAffineTransform3f(
      dr::Matrix<float, 3>(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]));
}

mitsuba::ref<mitsuba::Object> CreateBitmapTexture(const TextureKey& key,
                                                  const mitsuba::Bitmap& bitmap,
                                                  std::string_view variant) {
  mitsuba::ref<mitsuba::Bitmap> view;
  if (key.alpha_channel) {
    if (!bitmap.has_alpha()) return nullptr;
    // Alpha is stored linearly and is the last channel.
    view = new mitsuba::Bitmap(mitsuba::Bitmap::PixelFormat::Y,
                               bitmap.component_format(), bitmap.size());
    view->set_srgb_gamma(false);
    const size_t bpp = bitmap.bytes_per_pixel();
    const size_t ch_bytes = bpp / bitmap.channel_count();
    const uint8_t* src = bitmap.uint8_data();
    uint8_t* dst = view->uint8_data();
    for (size_t i = 0, n = view->pixel_count(); i < n; ++i) {
      std::memcpy(dst + i * ch_bytes, src + (i + 1) * bpp - ch_bytes, ch_bytes);
    }
  } else {
    // Wrap the decoded pixel buffer in a non-owning Bitmap header view so
    // BitmapTexture::load_bitmap() can safely mutate srgb_gamma in-place
    // (when raw == true) without copying pixel memory or racing across
    // threads that share the same source Bitmap.
    //
    // Lifetime: the expanded texture copies the pixels into its own tensor
    // (either via Bitmap::convert() or the TensorXf constructor) and the
    // un-expanded BitmapTexture holding `view` is dropped below, so neither
    // `view` nor `bitmap` needs to outlive this function.
    view = new mitsuba::Bitmap(bitmap.pixel_format(), bitmap.component_format(),
                               bitmap.size(), bitmap.channel_count(), {},
                               const_cast<uint8_t*>(bitmap.uint8_data()));
    view->set_srgb_gamma(bitmap.srgb_gamma());
    view->set_premultiplied_alpha(bitmap.premultiplied_alpha());
  }

  mitsuba::Properties props("bitmap");
  props.set("bitmap", mitsuba::ref<mitsuba::Object>(view.get()));
  props.set("raw", key.raw);
  props.set("wrap_mode", key.wrap_mode);
  props.set("filter_type", key.filter_type);
  props.set("max_anisotropy", key.max_anisotropy);
  props.set("format", key.format);
  props.set("accel", key.accel);
  if (key.to_uv != TextureKey::kIdentityUv) {
    props.set("to_uv", UvMatrixToTransform(key.to_uv));
  }
  return CreateExpandedObject(props, variant, mitsuba::ObjectType::Texture);
}

}  // namespace

std::optional<TextureKey> ExtractTextureKey(
    const std::map<TfToken, VtValue>& parameters, const TfToken& nodeTypeId,
    const TfToken& input_name) {
  std::string resolved_path;
  if (nodeTypeId == TfToken("UsdUVTexture")) {
    if (auto file_it = parameters.find(TfToken("file"));
        file_it != parameters.end()) {
      resolved_path = ResolvePathFromValue(file_it->second);
    }
  } else if (nodeTypeId == TfToken("mitsuba_bitmap")) {
    if (auto filename_it = parameters.find(TfToken("filename"));
        filename_it != parameters.end()) {
      resolved_path = ResolvePathFromValue(filename_it->second);
    }
  }
  if (resolved_path.empty()) {
    return std::nullopt;
  }

  TextureKey key;
  key.filename = std::move(resolved_path);
  if (nodeTypeId == TfToken("UsdUVTexture")) {
    TfToken source_color_space("auto");
    if (auto cs_it = parameters.find(TfToken("sourceColorSpace"));
        cs_it != parameters.end() && cs_it->second.IsHolding<TfToken>()) {
      source_color_space = cs_it->second.Get<TfToken>();
    }
    key.raw = UseRawBitmap(source_color_space, input_name);
    if (auto wrap_s_it = parameters.find(TfToken("wrapS"));
        wrap_s_it != parameters.end() &&
        wrap_s_it->second.IsHolding<TfToken>()) {
      TfToken wrap_s = wrap_s_it->second.Get<TfToken>();
      if (wrap_s == TfToken("clamp") || wrap_s == TfToken("mirror")) {
        key.wrap_mode = wrap_s.GetString();
      }
    }
  } else if (nodeTypeId == TfToken("mitsuba_bitmap")) {
    // Each setter returns false if `v` does not hold the expected type.
    auto set_as = [](const VtValue& v, auto& dst) {
      using T = std::decay_t<decltype(dst)>;
      if (!v.IsHolding<T>()) return false;
      dst = v.Get<T>();
      return true;
    };
    auto set_string = [](const VtValue& v, std::string& dst) {
      if (v.IsHolding<std::string>()) dst = v.Get<std::string>();
      else if (v.IsHolding<TfToken>()) dst = v.Get<TfToken>().GetString();
      else return false;
      return true;
    };
    auto set_uv = [](const VtValue& v, TextureKey::UvMatrix& dst) {
      auto uv = UvMatrixFromValue(v);
      if (uv) dst = *uv;
      return uv.has_value();
    };
    for (const auto& [param_token, val] : parameters) {
      const std::string& name = param_token.GetString();
      bool ok = true;
      if (name == "filename" || name == "fallback") {
        // Consumed above / by the material parser's fallback texture.
      } else if (absl::StartsWith(name, "colorSpace:")) {
        // Color space metadata of another parameter.
      } else if (name == "raw") {
        ok = set_as(val, key.raw);
      } else if (name == "accel") {
        ok = set_as(val, key.accel);
      } else if (name == "max_anisotropy") {
        ok = set_as(val, key.max_anisotropy);
      } else if (name == "wrap_mode") {
        ok = set_string(val, key.wrap_mode);
      } else if (name == "filter_type") {
        ok = set_string(val, key.filter_type);
      } else if (name == "format") {
        ok = set_string(val, key.format);
      } else if (name == "to_uv") {
        ok = set_uv(val, key.to_uv);
      } else {
        ok = false;
      }
      if (!ok) {
        TF_WARN(
            "Ignoring parameter '%s' (%s) on mitsuba_bitmap texture '%s': "
            "unsupported name or value type.",
            name.c_str(), val.GetTypeName().c_str(), key.filename.c_str());
      }
    }
  }
  return key;
}

mitsuba::ref<mitsuba::Bitmap> ReadBitmap(const std::string& path) {
  if (std::shared_ptr<ArAsset> asset =
          ArGetResolver().OpenAsset(ArResolvedPath(path))) {
    if (std::shared_ptr<const char> buffer = asset->GetBuffer()) {
      mitsuba::ref<mitsuba::MemoryStream> stream = new mitsuba::MemoryStream(
          const_cast<char*>(buffer.get()), asset->GetSize());
      return new mitsuba::Bitmap(stream.get());
    }
  }
  return new mitsuba::Bitmap(path);
}

MI_VARIANT const CachedTexture* TextureCache<Float, Spectrum>::Find(
    const TextureKey& key) const {
  auto it = textures_.find(key);
  return it == textures_.end() ? nullptr : &it->second;
}

MI_VARIANT void TextureCache<Float, Spectrum>::Preload(
    const absl::flat_hash_set<TextureKey>& keys) {
  std::vector<TextureKey> texture_list;
  std::vector<std::string> file_list;
  std::vector<std::vector<size_t>> file_to_textures;
  absl::flat_hash_map<std::string, size_t> file_indices;

  for (const TextureKey& key : keys) {
    if (!textures_.contains(key)) {
      const size_t tex_idx = texture_list.size();
      texture_list.push_back(key);
      auto [it, inserted] =
          file_indices.try_emplace(key.filename, file_list.size());
      if (inserted) {
        file_list.push_back(key.filename);
        file_to_textures.emplace_back();
      }
      file_to_textures[it->second].push_back(tex_idx);
    }
  }
  if (texture_list.empty()) return;

  // Decode each unique image file once in parallel, immediately instantiate all
  // TextureKey plugins referencing it while the decoded Bitmap is hot in cache,
  // and release the uncompressed Bitmap on the worker thread to bound peak RAM.
  std::vector<CachedTexture> loaded_textures(texture_list.size());
  dr::parallel_for(dr::blocked_range<size_t>(0, file_list.size()),
                   [&](dr::blocked_range<size_t> r) {
                     for (size_t i = r.begin(); i != r.end(); ++i) {
                       mitsuba::ref<mitsuba::Bitmap> bmp =
                           LoadBitmap(file_list[i]);
                       if (!bmp) continue;
                       for (size_t tex_idx : file_to_textures[i]) {
                         JitScopeGuard<Float> jit_guard;
                         loaded_textures[tex_idx] =
                             LoadTexture(texture_list[tex_idx], *bmp);
                       }
                     }
                     if constexpr (dr::is_metal_v<Float>) {
                       jit_flush_thread();
                     }
                   });

  for (size_t i = 0; i < texture_list.size(); ++i) {
    textures_[texture_list[i]] = std::move(loaded_textures[i]);
  }
  dirty_ = true;
}

MI_VARIANT void TextureCache<Float, Spectrum>::GarbageCollect() {
  if (!dirty_) return;
  for (auto it = textures_.begin(); it != textures_.end();) {
    if (!it->second.texture || it->second.texture->ref_count() == 1) {
      textures_.erase(it++);
    } else {
      ++it;
    }
  }
  dirty_ = false;
}

MI_VARIANT mitsuba::ref<mitsuba::Bitmap>
TextureCache<Float, Spectrum>::LoadBitmap(const std::string& filename) {
  try {
    return ReadBitmap(filename);
  } catch (const std::exception& e) {
    TF_WARN("Failed to load texture '%s': %s.", filename.c_str(), e.what());
    return nullptr;
  }
}

MI_VARIANT CachedTexture TextureCache<Float, Spectrum>::LoadTexture(
    const TextureKey& key, const mitsuba::Bitmap& bitmap) {
  try {
    size_t channel_count = key.alpha_channel ? 1 : bitmap.channel_count();
    mitsuba::ref<mitsuba::Object> texture = CreateBitmapTexture(
        key, bitmap, mitsuba::Texture<Float, Spectrum>::Variant);
    return CachedTexture{std::move(texture), channel_count};
  } catch (const std::exception& e) {
    TF_WARN("Failed to load texture '%s': %s.", key.filename.c_str(), e.what());
    return CachedTexture{};
  }
}

using mitsuba::Color;
using mitsuba::Spectrum;

MI_INSTANTIATE_CLASS(TextureCache)

PXR_NAMESPACE_CLOSE_SCOPE
