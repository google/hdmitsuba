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

#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/debug_codes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/strip.h>
#include <drjit/matrix.h>
#include <drjit/tensor.h>
#include <drjit/transform.h>
#include <mitsuba/core/animated_transform.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/mstream.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/rfilter.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volumegrid.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/ies_emitter.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/plugin_builder.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/traversal.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

namespace {

using ScalarVector3f = mitsuba::Vector<float, 3>;

// The helpers below encode material-graph rules that are shared between the
// texture pre-pass (DiscoverTextures) and the actual material parsers. Keep
// both sides going through these so that they cannot drift apart: a texture
// that is parsed but not discovered silently falls back to a constant color.

bool IsTextureNode(const HdMaterialNode2& node) {
  return node.nodeTypeId == TfToken("UsdUVTexture") ||
         node.nodeTypeId == TfToken("mitsuba_bitmap");
}

bool IsPrincipledNode(const HdMaterialNode2& node) {
  return node.nodeTypeId == TfToken("UsdPreviewSurface") ||
         node.nodeTypeId == TfToken("mitsuba_principled");
}

// Mitsuba's principled BSDF only accepts constants for these inputs.
bool IsScalarOnlyPrincipledInput(const TfToken& input_name) {
  return input_name == TfToken("specular") || input_name == TfToken("ior") ||
         input_name == TfToken("eta");
}

// Returns the connection of the `mitsuba:<terminal>` terminal, falling back to
// `<terminal>`, or nullptr if neither is connected.
const HdMaterialConnection2* FindTerminalConnection(
    const HdMaterialNetwork2& network, const TfToken& terminal) {
  auto it = network.terminals.find(
      TfToken(absl::StrCat("mitsuba:", terminal.GetString())));
  if (it == network.terminals.end() || it->second.upstreamNode.IsEmpty()) {
    it = network.terminals.find(terminal);
  }
  return (it != network.terminals.end() && !it->second.upstreamNode.IsEmpty())
             ? &it->second
             : nullptr;
}

}  // namespace

std::string MeshAttributeName(std::string_view primvar, int channels) {
  return absl::StrCat(kVertexAttributePrefix,
                      absl::StrReplaceAll(primvar, {{"color", "Color"}}), "_",
                      channels);
}

void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey& tex_key)>& callback) {
  absl::flat_hash_set<SdfPath, SdfPath::Hash> visited_nodes;
  std::function<void(const HdMaterialConnection2&, const TfToken&)> visit =
      [&](const HdMaterialConnection2& conn, const TfToken& input_name) {
        auto it = network.nodes.find(conn.upstreamNode);
        if (it == network.nodes.end()) {
          return;
        }
        const auto& node = it->second;
        if (IsTextureNode(node)) {
          // Texture nodes are keyed by input_name as well, so they are
          // intentionally not deduplicated via visited_nodes.
          if (auto key_opt = ExtractTextureKey(network, node, input_name,
                                               conn.upstreamOutputName)) {
            callback(*key_opt);
          }
          return;
        }
        if (!visited_nodes.insert(conn.upstreamNode).second) {
          return;
        }
        const bool is_principled = IsPrincipledNode(node);
        for (const auto& [conn_input, connections] : node.inputConnections) {
          if (connections.empty()) {
            continue;
          }
          if (is_principled && IsScalarOnlyPrincipledInput(conn_input)) {
            continue;
          }
          visit(connections[0], conn_input);
        }
      };

  for (const TfToken& terminal :
       {HdMaterialTerminalTokens->displacement,
        HdMaterialTerminalTokens->volume, HdMaterialTerminalTokens->surface}) {
    if (const auto* conn = FindTerminalConnection(network, terminal)) {
      visit(*conn, terminal);
    }
  }
}

namespace {

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::VolumeGrid<Float, Spectrum>> LoadVolumeGrid(
    const std::string& path) {
  if (std::shared_ptr<ArAsset> asset =
          ArGetResolver().OpenAsset(ArResolvedPath(path))) {
    if (std::shared_ptr<const char> buffer = asset->GetBuffer()) {
      mitsuba::ref<mitsuba::MemoryStream> stream = new mitsuba::MemoryStream(
          const_cast<char*>(buffer.get()), asset->GetSize());
      return new mitsuba::VolumeGrid<Float, Spectrum>(stream.get());
    }
  }
  return new mitsuba::VolumeGrid<Float, Spectrum>(path);
}

// Returns a texture node's RGBA `fallback` parameter (magenta if unset). Alpha
// defaults to 1 for RGB fallbacks.
GfVec4f ExtractFallbackColor(const std::map<TfToken, VtValue>& parameters) {
  auto it = parameters.find(TfToken("fallback"));
  if (it == parameters.end()) return GfVec4f(1.0f, 0.0f, 1.0f, 1.0f);
  const VtValue& val = it->second;
  if (val.IsHolding<GfVec4f>()) return val.UncheckedGet<GfVec4f>();
  if (val.IsHolding<GfVec3d>()) {
    const GfVec3f c(val.UncheckedGet<GfVec3d>());
    return GfVec4f(c[0], c[1], c[2], 1.0f);
  }
  const GfVec3f c = val.GetWithDefault(GfVec3f(1.0f, 0.0f, 1.0f));
  return GfVec4f(c[0], c[1], c[2], 1.0f);
}

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> GetTexture(
    const HdMaterialNetwork2& network, const HdMaterialNode2& node,
    const TfToken& input_name, const TfToken& output_name,
    const TextureCache<Float, Spectrum>& texture_cache) {
  const bool is_normal =
      (input_name == TfToken("normal") || input_name == TfToken("normalmap"));
  // Attempt to find the texture from the texture cache. This
  // assumes the texture cache was pre-populated by DiscoverTextures.
  if (auto key_opt =
          ExtractTextureKey(network, node, input_name, output_name)) {
    const CachedTexture* cached = texture_cache.Find(*key_opt);
    if (!cached) {
      // The texture discovery pass disagrees with the parser about which
      // inputs are textured. That is a bug; make it visible rather than
      // silently rendering a fallback color. (A file that failed to load is
      // cached as an empty entry and was already warned about.)
      TF_WARN("Texture '%s' (input '%s') was not preloaded; using fallback.",
              key_opt->filename.c_str(), input_name.GetText());
    } else if (cached->texture) {
      if (is_normal && cached->channel_count < 3) {
        TF_WARN(
            "Normal map texture '%s' has only %d channels! Normal maps require "
            "at least 3 channels.",
            key_opt->filename.c_str(), static_cast<int>(cached->channel_count));
      } else {
        return cached->texture;
      }
    }
  }
  // Instantiate a fallback texture.
  if (input_name == TfToken("normal") || input_name == TfToken("bump")) {
    return nullptr;
  }
  const GfVec4f fb = ExtractFallbackColor(node.parameters);
  const int ch = (node.nodeTypeId == TfToken("UsdUVTexture"))
                     ? UsdUVTextureChannel(output_name)
                     : -1;
  mitsuba::Properties fallback_props("srgb");
  if (input_name == TfToken("normalmap")) {
    fallback_props.set("color", mitsuba::Color<float, 3>(0.5f, 0.5f, 1.0f));
  } else if (input_name == TfToken("bumpmap")) {
    fallback_props = mitsuba::Properties("uniform");
    fallback_props.set("value", 0.0f);
  } else if (ch >= 0) {
    fallback_props = mitsuba::Properties("uniform");
    fallback_props.set("value", fb[ch]);
  } else {
    fallback_props.set("color", mitsuba::Color<float, 3>(fb[0], fb[1], fb[2]));
  }
  return mitsuba::PluginManager::instance()->create_object(
      fallback_props, mitsuba::Texture<Float, Spectrum>::Variant,
      mitsuba::Texture<Float, Spectrum>::Type);
}

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> ResolveConnectedTexture(
    const HdMaterialNetwork2& network2, const HdMaterialNode2& downstream_node,
    const TfToken& input_name,
    const TextureCache<Float, Spectrum>& texture_cache) {
  auto conn_it = downstream_node.inputConnections.find(input_name);
  if (conn_it == downstream_node.inputConnections.end() ||
      conn_it->second.empty()) {
    return nullptr;
  }
  const HdMaterialConnection2& conn = conn_it->second[0];
  auto node_it = network2.nodes.find(conn.upstreamNode);
  if (node_it == network2.nodes.end()) {
    return nullptr;
  }
  const HdMaterialNode2& upstream_node = node_it->second;
  if (IsTextureNode(upstream_node)) {
    return GetTexture<Float, Spectrum>(network2, upstream_node, input_name,
                                       conn.upstreamOutputName, texture_cache);
  }
  return nullptr;
}

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> ParseNodeRecursive(
    const HdMaterialConnection2& conn, const HdMaterialNetwork2& network,
    absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>&
        cache,
    const TfToken& input_name,
    const TextureCache<Float, Spectrum>& texture_cache) {
  const SdfPath& nodePath = conn.upstreamNode;
  auto it = network.nodes.find(nodePath);
  if (it == network.nodes.end()) {
    TF_RUNTIME_ERROR("Node %s not found in material network.",
                     nodePath.GetText());
    return nullptr;
  }

  // Specifically handle textures via TextureCache so that a shared texture node
  // connected to multiple inputs with different color-space requirements
  // resolves the appropriate texture variant per input_name.
  const auto& node = it->second;
  if (IsTextureNode(node)) {
    return GetTexture<Float, Spectrum>(network, node, input_name,
                                       conn.upstreamOutputName, texture_cache);
  }

  if (auto cache_it = cache.find(nodePath); cache_it != cache.end()) {
    return cache_it->second;
  }

  // Handle other nodes (native mitsuba plugins, etc.)
  std::string mitsuba_plugin_name;
  std::string_view id_str = node.nodeTypeId.GetString();
  if (id_str.compare(0, 8, "mitsuba_") == 0) {
    id_str.remove_prefix(8);
    mitsuba_plugin_name = std::string(id_str);
  } else {
    return nullptr;
  }

  mitsuba::Properties props(mitsuba_plugin_name);
  for (const auto& [param_token, param_value] : node.parameters) {
    SetMitsubaPropertyFromValue(props, param_token.GetString(), param_value);
  }
  if (mitsuba_plugin_name == "gridvolume" && props.has_property("filename") &&
      !props.has_property("grid")) {
    const std::string path = props.get<std::string>("filename");
    props.remove_property("filename");
    props.set("grid", mitsuba::ref<mitsuba::Object>(
                          LoadVolumeGrid<Float, Spectrum>(path)));
  }

  for (const auto& [input_token, connections] : node.inputConnections) {
    if (connections.empty()) continue;
    if (IsPrincipledNode(node) && IsScalarOnlyPrincipledInput(input_token)) {
      TF_WARN(
          "Mitsuba's principled BSDF does not support texture connection for "
          "'%s'. Ignoring connection.",
          input_token.GetText());
      continue;
    }
    mitsuba::ref<mitsuba::Object> upstream_object =
        ParseNodeRecursive<Float, Spectrum>(connections[0], network, cache,
                                            input_token, texture_cache);
    if (upstream_object) {
      props.set(input_token.GetString(), upstream_object);
    }
  }

  mitsuba::ObjectType object_type =
      mitsuba::PluginManager::instance()->plugin_type(mitsuba_plugin_name);
  mitsuba::ref<mitsuba::Object> result = CreateExpandedObject(
      props, mitsuba::Scene<Float, Spectrum>::Variant, object_type);
  cache[nodePath] = result;
  return result;
}

}  // namespace

MI_VARIANT
mitsuba::ref<mitsuba::BSDF<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::DefaultBsdf(std::string_view id_str,
                                             std::string_view plugin_name) {
  mitsuba::Properties props(plugin_name);
  auto bsdf = mitsuba::PluginManager::instance()
                  ->create_object<mitsuba::BSDF<Float, Spectrum>>(props);
  bsdf->set_id(id_str);
  return bsdf;
}

MI_VARIANT TranslatedMaterial
PrimTranslator<Float, Spectrum>::ParsePreviewSurface(
    const HdMaterialNetwork2& network2,
    const HdMaterialNode2& preview_surface_node,
    const TextureCache<Float, Spectrum>& texture_cache) {
  TranslatedMaterial res;
  // Leave the BSDF null for unsupported diffuse inputs (e.g. Ptex) so that the
  // geometry falls back to its displayColor.
  if (auto it = preview_surface_node.inputConnections.find(
          TfToken("diffuseColor"));
      it != preview_surface_node.inputConnections.end() &&
      !it->second.empty()) {
    auto node_it = network2.nodes.find(it->second[0].upstreamNode);
    if (node_it == network2.nodes.end() || !IsTextureNode(node_it->second)) {
      return res;
    }
  }
  struct ParamMapping {
    TfToken usd_name;
    std::string_view mitsuba_name;
    bool invert_float = false;
  };
  static const absl::NoDestructor<std::vector<ParamMapping>> param_mappings({
      {TfToken("diffuseColor"), "base_color"},
      {TfToken("roughness"), "roughness"},
      {TfToken("metallic"), "metallic"},
      {TfToken("clearcoat"), "clearcoat"},
      {TfToken("clearcoatRoughness"), "clearcoat_gloss"},
      {TfToken("ior"), "eta"},
      {TfToken("specular"), "specular"},
      {TfToken("opacity"), "spec_trans", true},
  });

  mitsuba::Properties props("principled");
  bool spec_trans_is_texture = false;
  for (const auto& mapping : *param_mappings) {
    const bool is_scalar_only = IsScalarOnlyPrincipledInput(mapping.usd_name);
    mitsuba::ref<mitsuba::Object> texture = nullptr;
    if (!is_scalar_only) {
      texture = ResolveConnectedTexture<Float, Spectrum>(
          network2, preview_surface_node, mapping.usd_name, texture_cache);
    } else {
      if (preview_surface_node.inputConnections.count(mapping.usd_name) > 0) {
        TF_WARN(
            "Mitsuba's principled BSDF does not support texture for '%s'. "
            "Using constant value instead.",
            std::string(mapping.mitsuba_name).c_str());
      }
    }
    if (texture) {
      props.set(mapping.mitsuba_name, texture);
      if (mapping.mitsuba_name == "spec_trans") {
        spec_trans_is_texture = true;
      }
    } else {
      auto param_it = preview_surface_node.parameters.find(mapping.usd_name);
      if (param_it != preview_surface_node.parameters.end()) {
        SetMitsubaPropertyFromValue(props, mapping.mitsuba_name,
                                    param_it->second, mapping.invert_float);
      }
    }
  }

  if (props.has_property("eta") && props.has_property("specular")) {
    if (spec_trans_is_texture || props.get<float>("spec_trans", 0.0f) > 0.0f) {
      props.remove_property("specular");
    } else {
      props.remove_property("eta");
    }
  }

  // A non-zero emissive color is used to create a shape emitter.
  const TfToken emissive_token("emissiveColor");
  mitsuba::ref<mitsuba::Object> emissive_texture =
      ResolveConnectedTexture<Float, Spectrum>(network2, preview_surface_node,
                                               emissive_token, texture_cache);
  if (emissive_texture) {
    mitsuba::Properties emitter_props("area");
    emitter_props.set("radiance", emissive_texture);
    res.shape_emitter_props = emitter_props;
  } else {
    auto it = preview_surface_node.parameters.find(emissive_token);
    if (it != preview_surface_node.parameters.end() &&
        it->second.IsHolding<GfVec3f>()) {
      auto color = it->second.Get<GfVec3f>();
      if (color.GetLengthSq() > 0.0f) {
        mitsuba::Properties emitter_props("area");
        emitter_props.set(
            "radiance", mitsuba::Color<float, 3>(color[0], color[1], color[2]));
        res.shape_emitter_props = emitter_props;
      }
    }
  }

  res.bsdf = mitsuba::PluginManager::instance()->create_object(
      props, mitsuba::BSDF<Float, Spectrum>::Variant,
      mitsuba::BSDF<Float, Spectrum>::Type);

  TfToken normal_token("normal");
  mitsuba::ref<mitsuba::Object> texture =
      ResolveConnectedTexture<Float, Spectrum>(network2, preview_surface_node,
                                               normal_token, texture_cache);
  if (texture) {
    mitsuba::Properties normal_props("normalmap");
    normal_props.set("normalmap", texture);
    normal_props.set("nested_bsdf", res.bsdf);
    res.bsdf = mitsuba::PluginManager::instance()->create_object(
        normal_props, mitsuba::BSDF<Float, Spectrum>::Variant,
        mitsuba::BSDF<Float, Spectrum>::Type);
  }
  return res;
}

MI_VARIANT TranslatedMaterial
PrimTranslator<Float, Spectrum>::BuildMaterial(
    const MaterialSpec& spec,
    const TextureCache<Float, Spectrum>& texture_cache) {
  TranslatedMaterial res;
  const HdMaterialNetwork2& network2 = spec.network2;
  const std::string id_str = spec.id.GetAsString();
  absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>
      cache;

  if (const auto* disp_conn = FindTerminalConnection(
          network2, HdMaterialTerminalTokens->displacement)) {
    res.displacement.texture = ParseNodeRecursive<Float, Spectrum>(
        *disp_conn, network2, cache, HdMaterialTerminalTokens->displacement,
        texture_cache);
  }
  if (const auto* vol_conn =
          FindTerminalConnection(network2, HdMaterialTerminalTokens->volume)) {
    cache.clear();
    res.interior_medium = ParseNodeRecursive<Float, Spectrum>(
        *vol_conn, network2, cache, HdMaterialTerminalTokens->volume,
        texture_cache);
    if (res.interior_medium) {
      if (res.interior_medium->type() != mitsuba::ObjectType::Medium) {
        TF_WARN("Volume shader on material '%s' is not a Mitsuba medium.",
                id_str.c_str());
        res.interior_medium = nullptr;
      } else {
        res.interior_medium->set_id(id_str);
      }
    }
  }

  if (const auto* surf_conn =
          FindTerminalConnection(network2, HdMaterialTerminalTokens->surface)) {
    if (auto node_it = network2.nodes.find(surf_conn->upstreamNode);
        node_it != network2.nodes.end()) {
      const HdMaterialNode2& surface_node = node_it->second;
      if (surface_node.nodeTypeId == TfToken("UsdPreviewSurface")) {
        TranslatedMaterial preview_res =
            ParsePreviewSurface(network2, surface_node, texture_cache);
        res.bsdf = std::move(preview_res.bsdf);
        res.shape_emitter_props = std::move(preview_res.shape_emitter_props);
        res.mesh_attributes = std::move(preview_res.mesh_attributes);
      } else {
        cache.clear();
        res.bsdf = ParseNodeRecursive<Float, Spectrum>(
            *surf_conn, network2, cache, HdMaterialTerminalTokens->surface,
            texture_cache);
      }
      if (res.bsdf) {
        res.bsdf->set_id(id_str);
      }
    }
  }
  if (!res.bsdf && res.interior_medium) {
    res.bsdf = DefaultBsdf(id_str, "null");
  }
  return res;
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::SetTransformProperty(
    mitsuba::Properties& props, const MotionTransform& transform) {
  if (transform.IsAnimated()) {
    props.set("to_world", new mitsuba::AnimatedTransform<Float, Spectrum>(
                              transform.samples));
  } else {
    props.set("to_world", transform.First());
  }
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::SetTransformParameter(
    TraversalCallback& cb, const MotionTransform& transform) {
  using AffineTransform4f = mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
  using AnimatedTransform = mitsuba::AnimatedTransform<Float, Spectrum>;
  using TensorXf = typename AnimatedTransform::TensorXf;
  if (!transform.IsAnimated()) {
    cb.set<AffineTransform4f>("to_world", AffineTransform4f(transform.First()));
    return;
  }
  // Copy the keyframe tensors of a new animation into the existing one.
  AnimatedTransform animation(transform.samples);
  TraversalCallback animation_cb;
  animation.traverse(&animation_cb);
  for (std::string_view key : {"times", "scale", "rotation", "translation"}) {
    cb.set<TensorXf>(absl::StrCat("to_world.", key),
                     *animation_cb.get<TensorXf>(key));
  }
  // AnimatedTransform::parameters_changed() copies the updated device tensors
  // back into m_keyframes on the host and refreshes m_keyframe_count.
  if (mitsuba::Object* to_world = cb.objects["to_world"]) {
    to_world->parameters_changed();
  }
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::SetSpectrumParameter(
    TraversalCallback& cb, std::string_view name, const GfVec3f& rgb) {
  using Color3f = mitsuba::Color<Float, 3>;
  using ScalarFloat = dr::scalar_t<Float>;
  auto make_val = [](ScalarFloat v) -> Float {
    if constexpr (dr::is_jit_v<Float>) {
      return dr::opaque<Float>(v);
    } else {
      return Float(v);
    }
  };
  cb.set<Color3f>(
      absl::StrCat(name, ".value"),
      Color3f(make_val(rgb[0]), make_val(rgb[1]), make_val(rgb[2])));
  auto it = cb.objects.find(name);
  if (TF_VERIFY(it != cb.objects.end(), "No spectrum object found for %.*s",
                static_cast<int>(name.size()), name.data())) {
    it->second->parameters_changed();
  }
}

namespace {

enum class PluginLeafRole { kPlain, kRgb, kVector, kSpectrum };

struct ChangedPluginLeaf {
  std::string path;
  PluginLeafRole role;
  VtValue value;
};

template <typename T>
bool TypeMatches(const std::type_info& info) {
  return info == typeid(T) || std::string_view(info.name()) == typeid(T).name();
}

template <typename Float>
Float MakeOpaqueFloat(dr::scalar_t<Float> scalar) {
  if constexpr (dr::is_jit_v<Float>) {
    return dr::opaque<Float>(scalar);
  } else {
    return Float(scalar);
  }
}

std::optional<double> ExtractDoubleValue(const VtValue& value) {
  if (value.IsHolding<float>()) return value.UncheckedGet<float>();
  if (value.IsHolding<double>()) return value.UncheckedGet<double>();
  if (value.IsHolding<int>()) return value.UncheckedGet<int>();
  if (value.IsHolding<int64_t>()) {
    return static_cast<double>(value.UncheckedGet<int64_t>());
  }
  if (value.IsHolding<uint32_t>()) return value.UncheckedGet<uint32_t>();
  if (value.IsHolding<uint64_t>()) {
    return static_cast<double>(value.UncheckedGet<uint64_t>());
  }
  return std::nullopt;
}

std::optional<int64_t> ExtractInt64Value(const VtValue& value) {
  if (value.IsHolding<int>()) return value.UncheckedGet<int>();
  if (value.IsHolding<int64_t>()) return value.UncheckedGet<int64_t>();
  if (value.IsHolding<uint32_t>()) return value.UncheckedGet<uint32_t>();
  if (value.IsHolding<uint64_t>()) {
    uint64_t raw = value.UncheckedGet<uint64_t>();
    if (raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return static_cast<int64_t>(raw);
    }
  }
  return std::nullopt;
}

std::optional<GfVec3d> ExtractRgb3Value(const VtValue& value,
                                        bool allow_scalar) {
  if (value.IsHolding<GfVec3f>()) {
    const GfVec3f& c = value.UncheckedGet<GfVec3f>();
    return GfVec3d(c[0], c[1], c[2]);
  }
  if (value.IsHolding<GfVec3d>()) return value.UncheckedGet<GfVec3d>();
  if (value.IsHolding<GfVec4f>()) {
    const GfVec4f& c = value.UncheckedGet<GfVec4f>();
    return GfVec3d(c[0], c[1], c[2]);
  }
  if (value.IsHolding<GfVec4d>()) {
    const GfVec4d& c = value.UncheckedGet<GfVec4d>();
    return GfVec3d(c[0], c[1], c[2]);
  }
  if (allow_scalar) {
    if (std::optional<double> scalar = ExtractDoubleValue(value)) {
      return GfVec3d(*scalar);
    }
  }
  return std::nullopt;
}

std::optional<GfVec3d> ExtractVec3Value(const VtValue& value) {
  if (value.IsHolding<GfVec3f>()) {
    const GfVec3f& v = value.UncheckedGet<GfVec3f>();
    return GfVec3d(v[0], v[1], v[2]);
  }
  if (value.IsHolding<GfVec3d>()) return value.UncheckedGet<GfVec3d>();
  return std::nullopt;
}

std::optional<GfVec2d> ExtractVec2Value(const VtValue& value) {
  if (value.IsHolding<GfVec2f>()) {
    const GfVec2f& v = value.UncheckedGet<GfVec2f>();
    return GfVec2d(v[0], v[1]);
  }
  if (value.IsHolding<GfVec2d>()) return value.UncheckedGet<GfVec2d>();
  return std::nullopt;
}

bool CollectChangedPluginLeaves(const std::string& prefix,
                                const PluginSpecNode& old_node,
                                const PluginSpecNode& new_node,
                                std::vector<ChangedPluginLeaf>* leaves) {
  for (const auto& [param_key, new_val] : new_node.params) {
    if (new_node.children.find(param_key) != new_node.children.end()) {
      continue;
    }
    auto old_it = old_node.params.find(param_key);
    if (old_it == old_node.params.end() || old_it->second != new_val) {
      std::string path =
          prefix.empty() ? param_key : absl::StrCat(prefix, ".", param_key);
      leaves->push_back({std::move(path), PluginLeafRole::kPlain, new_val});
    }
  }
  for (const auto& [child_key, new_child] : new_node.children) {
    auto old_it = old_node.children.find(child_key);
    if (old_it == old_node.children.end()) return false;
    const PluginSpecNode& old_child = old_it->second;
    if (old_child == new_child) continue;
    std::string child_path =
        prefix.empty() ? child_key : absl::StrCat(prefix, ".", child_key);
    if (new_child.plugin_type == "rgb" || new_child.plugin_type == "vector" ||
        new_child.plugin_type == "spectrum") {
      auto val_it = new_child.params.find("value");
      if (val_it == new_child.params.end() || new_child.params.size() != 1) {
        return false;
      }
      PluginLeafRole role = PluginLeafRole::kSpectrum;
      if (new_child.plugin_type == "rgb") {
        role = PluginLeafRole::kRgb;
      } else if (new_child.plugin_type == "vector") {
        role = PluginLeafRole::kVector;
      }
      leaves->push_back({std::move(child_path), role, val_it->second});
      continue;
    }
    if (!CollectChangedPluginLeaves(child_path, old_child, new_child, leaves)) {
      return false;
    }
  }
  return true;
}

template <typename Float>
bool StageTypedPluginWrite(const TraversalCallback::ValueEntry& entry,
                           PluginLeafRole role, const VtValue& value,
                           std::vector<std::function<void()>>* staged_writes) {
  using ScalarFloat = dr::scalar_t<Float>;
  if constexpr (dr::is_jit_v<Float>) {
    using Color3f = mitsuba::Color<Float, 3>;
    using Vector3f = mitsuba::Vector<Float, 3>;
    using Point3f = mitsuba::Point<Float, 3>;
    using Normal3f = mitsuba::Normal<Float, 3>;
    using Vector2f = mitsuba::Vector<Float, 2>;
    using Point2f = mitsuba::Point<Float, 2>;
    using AffineTransform4f =
        mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
    using AffineTransform3f =
        mitsuba::Transform<mitsuba::Point<Float, 3>, true>;

    if (TypeMatches<Float>(entry.type)) {
      if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kSpectrum) {
        return false;
      }
      std::optional<double> scalar = ExtractDoubleValue(value);
      if (!scalar.has_value()) return false;
      const ScalarFloat v = static_cast<ScalarFloat>(*scalar);
      auto* dst = static_cast<Float*>(entry.ptr);
      staged_writes->push_back([dst, v]() { *dst = dr::opaque<Float>(v); });
      return true;
    }
    if (TypeMatches<Color3f>(entry.type)) {
      if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kRgb) {
        return false;
      }
      std::optional<GfVec3d> rgb =
          ExtractRgb3Value(value, role == PluginLeafRole::kRgb);
      if (!rgb.has_value()) return false;
      const GfVec3d c = *rgb;
      auto* dst = static_cast<Color3f*>(entry.ptr);
      staged_writes->push_back([dst, c]() {
        *dst = Color3f(dr::opaque<Float>(static_cast<ScalarFloat>(c[0])),
                       dr::opaque<Float>(static_cast<ScalarFloat>(c[1])),
                       dr::opaque<Float>(static_cast<ScalarFloat>(c[2])));
      });
      return true;
    }
    auto stage_jit_vec3 = [&](auto* dst) -> bool {
      using VecT = std::remove_pointer_t<decltype(dst)>;
      if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kVector) {
        return false;
      }
      std::optional<GfVec3d> vec = ExtractVec3Value(value);
      if (!vec.has_value()) return false;
      const GfVec3d v = *vec;
      staged_writes->push_back([dst, v]() {
        *dst = VecT(dr::opaque<Float>(static_cast<ScalarFloat>(v[0])),
                    dr::opaque<Float>(static_cast<ScalarFloat>(v[1])),
                    dr::opaque<Float>(static_cast<ScalarFloat>(v[2])));
      });
      return true;
    };
    if (TypeMatches<Vector3f>(entry.type)) {
      return stage_jit_vec3(static_cast<Vector3f*>(entry.ptr));
    }
    if (TypeMatches<Point3f>(entry.type)) {
      return stage_jit_vec3(static_cast<Point3f*>(entry.ptr));
    }
    if (TypeMatches<Normal3f>(entry.type)) {
      return stage_jit_vec3(static_cast<Normal3f*>(entry.ptr));
    }
    auto stage_jit_vec2 = [&](auto* dst) -> bool {
      using VecT = std::remove_pointer_t<decltype(dst)>;
      if (role != PluginLeafRole::kPlain) return false;
      std::optional<GfVec2d> vec = ExtractVec2Value(value);
      if (!vec.has_value()) return false;
      const GfVec2d v = *vec;
      staged_writes->push_back([dst, v]() {
        *dst = VecT(dr::opaque<Float>(static_cast<ScalarFloat>(v[0])),
                    dr::opaque<Float>(static_cast<ScalarFloat>(v[1])));
      });
      return true;
    };
    if (TypeMatches<Vector2f>(entry.type)) {
      return stage_jit_vec2(static_cast<Vector2f*>(entry.ptr));
    }
    if (TypeMatches<Point2f>(entry.type)) {
      return stage_jit_vec2(static_cast<Point2f*>(entry.ptr));
    }
    if (TypeMatches<AffineTransform4f>(entry.type)) {
      if (role != PluginLeafRole::kPlain) return false;
      std::optional<ScalarAffineTransform4f> xform = ExtractTransform4f(value);
      if (!xform.has_value()) return false;
      const ScalarAffineTransform4f xf = *xform;
      auto* dst = static_cast<AffineTransform4f*>(entry.ptr);
      staged_writes->push_back([dst, xf]() {
        AffineTransform4f dev_xf(xf);
        dr::make_opaque(dev_xf);
        *dst = std::move(dev_xf);
      });
      return true;
    }
    if (TypeMatches<AffineTransform3f>(entry.type)) {
      if (role != PluginLeafRole::kPlain) return false;
      std::optional<ScalarAffineTransform3f> xform = ExtractTransform3f(value);
      if (!xform.has_value()) return false;
      const ScalarAffineTransform3f xf = *xform;
      auto* dst = static_cast<AffineTransform3f*>(entry.ptr);
      staged_writes->push_back([dst, xf]() {
        AffineTransform3f dev_xf(xf);
        dr::make_opaque(dev_xf);
        *dst = std::move(dev_xf);
      });
      return true;
    }
  }

  if (TypeMatches<float>(entry.type)) {
    if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kSpectrum) {
      return false;
    }
    std::optional<double> scalar = ExtractDoubleValue(value);
    if (!scalar.has_value()) return false;
    const float v = static_cast<float>(*scalar);
    auto* dst = static_cast<float*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  if (TypeMatches<double>(entry.type)) {
    if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kSpectrum) {
      return false;
    }
    std::optional<double> scalar = ExtractDoubleValue(value);
    if (!scalar.has_value()) return false;
    const double v = *scalar;
    auto* dst = static_cast<double*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  auto stage_scalar_rgb = [&](auto* dst) -> bool {
    using ColorT = std::remove_pointer_t<decltype(dst)>;
    using CompT = dr::scalar_t<ColorT>;
    if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kRgb) {
      return false;
    }
    std::optional<GfVec3d> rgb =
        ExtractRgb3Value(value, role == PluginLeafRole::kRgb);
    if (!rgb.has_value()) return false;
    const GfVec3d c = *rgb;
    staged_writes->push_back([dst, c]() {
      *dst = ColorT(static_cast<CompT>(c[0]), static_cast<CompT>(c[1]),
                    static_cast<CompT>(c[2]));
    });
    return true;
  };
  if (TypeMatches<mitsuba::Color<float, 3>>(entry.type)) {
    return stage_scalar_rgb(static_cast<mitsuba::Color<float, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Color<double, 3>>(entry.type)) {
    return stage_scalar_rgb(static_cast<mitsuba::Color<double, 3>*>(entry.ptr));
  }
  auto stage_scalar_vec3 = [&](auto* dst) -> bool {
    using VecT = std::remove_pointer_t<decltype(dst)>;
    using CompT = dr::scalar_t<VecT>;
    if (role != PluginLeafRole::kPlain && role != PluginLeafRole::kVector) {
      return false;
    }
    std::optional<GfVec3d> vec = ExtractVec3Value(value);
    if (!vec.has_value()) return false;
    const GfVec3d v = *vec;
    staged_writes->push_back([dst, v]() {
      *dst = VecT(static_cast<CompT>(v[0]), static_cast<CompT>(v[1]),
                  static_cast<CompT>(v[2]));
    });
    return true;
  };
  if (TypeMatches<mitsuba::Vector<float, 3>>(entry.type)) {
    return stage_scalar_vec3(
        static_cast<mitsuba::Vector<float, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Vector<double, 3>>(entry.type)) {
    return stage_scalar_vec3(
        static_cast<mitsuba::Vector<double, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Point<float, 3>>(entry.type)) {
    return stage_scalar_vec3(static_cast<mitsuba::Point<float, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Point<double, 3>>(entry.type)) {
    return stage_scalar_vec3(
        static_cast<mitsuba::Point<double, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Normal<float, 3>>(entry.type)) {
    return stage_scalar_vec3(
        static_cast<mitsuba::Normal<float, 3>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Normal<double, 3>>(entry.type)) {
    return stage_scalar_vec3(
        static_cast<mitsuba::Normal<double, 3>*>(entry.ptr));
  }
  auto stage_scalar_vec2 = [&](auto* dst) -> bool {
    using VecT = std::remove_pointer_t<decltype(dst)>;
    using CompT = dr::scalar_t<VecT>;
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<GfVec2d> vec = ExtractVec2Value(value);
    if (!vec.has_value()) return false;
    const GfVec2d v = *vec;
    staged_writes->push_back([dst, v]() {
      *dst = VecT(static_cast<CompT>(v[0]), static_cast<CompT>(v[1]));
    });
    return true;
  };
  if (TypeMatches<mitsuba::Vector<float, 2>>(entry.type)) {
    return stage_scalar_vec2(
        static_cast<mitsuba::Vector<float, 2>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Vector<double, 2>>(entry.type)) {
    return stage_scalar_vec2(
        static_cast<mitsuba::Vector<double, 2>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Point<float, 2>>(entry.type)) {
    return stage_scalar_vec2(static_cast<mitsuba::Point<float, 2>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Point<double, 2>>(entry.type)) {
    return stage_scalar_vec2(
        static_cast<mitsuba::Point<double, 2>*>(entry.ptr));
  }
  if (TypeMatches<mitsuba::Vector<int32_t, 2>>(entry.type)) {
    if (role != PluginLeafRole::kPlain || !value.IsHolding<GfVec2i>()) {
      return false;
    }
    const GfVec2i v = value.UncheckedGet<GfVec2i>();
    auto* dst = static_cast<mitsuba::Vector<int32_t, 2>*>(entry.ptr);
    staged_writes->push_back(
        [dst, v]() { *dst = mitsuba::Vector<int32_t, 2>(v[0], v[1]); });
    return true;
  }
  if (TypeMatches<mitsuba::Point<int32_t, 2>>(entry.type)) {
    if (role != PluginLeafRole::kPlain || !value.IsHolding<GfVec2i>()) {
      return false;
    }
    const GfVec2i v = value.UncheckedGet<GfVec2i>();
    auto* dst = static_cast<mitsuba::Point<int32_t, 2>*>(entry.ptr);
    staged_writes->push_back(
        [dst, v]() { *dst = mitsuba::Point<int32_t, 2>(v[0], v[1]); });
    return true;
  }
  if (TypeMatches<mitsuba::Vector<uint32_t, 2>>(entry.type)) {
    if (role != PluginLeafRole::kPlain || !value.IsHolding<GfVec2i>()) {
      return false;
    }
    const GfVec2i v = value.UncheckedGet<GfVec2i>();
    if (v[0] < 0 || v[1] < 0) return false;
    auto* dst = static_cast<mitsuba::Vector<uint32_t, 2>*>(entry.ptr);
    staged_writes->push_back([dst, v]() {
      *dst = mitsuba::Vector<uint32_t, 2>(static_cast<uint32_t>(v[0]),
                                          static_cast<uint32_t>(v[1]));
    });
    return true;
  }
  if (TypeMatches<mitsuba::Point<uint32_t, 2>>(entry.type)) {
    if (role != PluginLeafRole::kPlain || !value.IsHolding<GfVec2i>()) {
      return false;
    }
    const GfVec2i v = value.UncheckedGet<GfVec2i>();
    if (v[0] < 0 || v[1] < 0) return false;
    auto* dst = static_cast<mitsuba::Point<uint32_t, 2>*>(entry.ptr);
    staged_writes->push_back([dst, v]() {
      *dst = mitsuba::Point<uint32_t, 2>(static_cast<uint32_t>(v[0]),
                                         static_cast<uint32_t>(v[1]));
    });
    return true;
  }
  if (TypeMatches<bool>(entry.type)) {
    if (role != PluginLeafRole::kPlain || !value.IsHolding<bool>()) {
      return false;
    }
    const bool b = value.UncheckedGet<bool>();
    auto* dst = static_cast<bool*>(entry.ptr);
    staged_writes->push_back([dst, b]() { *dst = b; });
    return true;
  }
  if (TypeMatches<int32_t>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<int64_t> i = ExtractInt64Value(value);
    if (!i.has_value() || *i < std::numeric_limits<int32_t>::min() ||
        *i > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    const int32_t v = static_cast<int32_t>(*i);
    auto* dst = static_cast<int32_t*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  if (TypeMatches<uint32_t>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<int64_t> i = ExtractInt64Value(value);
    if (!i.has_value() || *i < 0 ||
        *i > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    const uint32_t v = static_cast<uint32_t>(*i);
    auto* dst = static_cast<uint32_t*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  if (TypeMatches<int64_t>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<int64_t> i = ExtractInt64Value(value);
    if (!i.has_value()) return false;
    const int64_t v = *i;
    auto* dst = static_cast<int64_t*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  if (TypeMatches<uint64_t>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    if (value.IsHolding<uint64_t>()) {
      const uint64_t v = value.UncheckedGet<uint64_t>();
      auto* dst = static_cast<uint64_t*>(entry.ptr);
      staged_writes->push_back([dst, v]() { *dst = v; });
      return true;
    }
    std::optional<int64_t> i = ExtractInt64Value(value);
    if (!i.has_value() || *i < 0) return false;
    const uint64_t v = static_cast<uint64_t>(*i);
    auto* dst = static_cast<uint64_t*>(entry.ptr);
    staged_writes->push_back([dst, v]() { *dst = v; });
    return true;
  }
  if (TypeMatches<ScalarAffineTransform4f>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<ScalarAffineTransform4f> xform = ExtractTransform4f(value);
    if (!xform.has_value()) return false;
    const ScalarAffineTransform4f xf = *xform;
    auto* dst = static_cast<ScalarAffineTransform4f*>(entry.ptr);
    staged_writes->push_back([dst, xf]() { *dst = xf; });
    return true;
  }
  if (TypeMatches<ScalarAffineTransform3f>(entry.type)) {
    if (role != PluginLeafRole::kPlain) return false;
    std::optional<ScalarAffineTransform3f> xform = ExtractTransform3f(value);
    if (!xform.has_value()) return false;
    const ScalarAffineTransform3f xf = *xform;
    auto* dst = static_cast<ScalarAffineTransform3f*>(entry.ptr);
    staged_writes->push_back([dst, xf]() { *dst = xf; });
    return true;
  }
  return false;
}

void AppendUniqueKey(std::vector<std::string>* keys, std::string_view key) {
  for (const std::string& existing : *keys) {
    if (existing == key) return;
  }
  keys->emplace_back(key);
}

std::optional<double> ComputePerspectiveXFov(const CameraSpec& spec,
                                             double aspect) {
  const std::string& sensor_type = spec.sensor_spec.plugin_type.empty()
                                       ? spec.sensor_type
                                       : spec.sensor_spec.plugin_type;
  if (!spec.sensor_spec.HasProperty("fov") &&
      !spec.sensor_spec.HasProperty("focal_length") &&
      !spec.sensor_spec.HasProperty("fov_axis")) {
    return static_cast<double>(spec.fov);
  }
  mitsuba::Properties fov_props(sensor_type);
  if (!spec.sensor_spec.HasProperty("focal_length")) {
    fov_props.set("fov", spec.fov);
    fov_props.set("fov_axis", "x");
  }
  for (const char* key : {"fov", "focal_length", "fov_axis"}) {
    if (auto it = spec.sensor_spec.params.find(key);
        it != spec.sensor_spec.params.end()) {
      SetMitsubaPropertyFromValue(fov_props, key, it->second,
                                  /*invert_float=*/false,
                                  /*raise_if_exists=*/false);
    }
  }
  try {
    return mitsuba::parse_fov(fov_props, aspect > 0.0 ? aspect : 1.0);
  } catch (const std::exception& e) {
    TF_RUNTIME_ERROR("Failed to parse FOV for sensor '%s': %s",
                     spec.id.GetText(), e.what());
    return std::nullopt;
  }
}

}  // namespace

MI_VARIANT bool PrimTranslator<Float, Spectrum>::TryUpdatePluginInPlace(
    mitsuba::Object* root_obj, const PluginSpecNode& old_spec,
    const PluginSpecNode& new_spec) {
  if (root_obj == nullptr) return false;
  if (old_spec == new_spec) return true;
  if (!CanUpdatePluginStructureInPlace(old_spec, new_spec)) return false;

  std::vector<ChangedPluginLeaf> leaves;
  if (!CollectChangedPluginLeaves("", old_spec, new_spec, &leaves)) {
    return false;
  }
  if (leaves.empty()) return true;

  TraversalCallback cb;
  root_obj->traverse(&cb);

  std::vector<std::function<void()>> staged_writes;
  staged_writes.reserve(leaves.size());
  std::vector<std::string> root_keys;
  absl::flat_hash_map<std::string, std::vector<std::string>>
      touched_object_keys;

  auto resolve_traversal_path = [&](std::string_view raw_path) -> std::string {
    std::string prefix;
    size_t start = 0;
    while (true) {
      size_t dot = raw_path.find('.', start);
      if (dot == std::string_view::npos) break;
      std::string_view seg = raw_path.substr(start, dot - start);
      std::string candidate =
          prefix.empty() ? std::string(seg) : absl::StrCat(prefix, ".", seg);
      if (!cb.objects.contains(candidate)) {
        std::string_view alias;
        if (seg == "interior") {
          alias = "interior_medium";
        } else if (seg == "exterior") {
          alias = "exterior_medium";
        } else if (seg == "phase") {
          alias = "phase_function";
        }
        if (!alias.empty()) {
          std::string alias_candidate = prefix.empty()
                                            ? std::string(alias)
                                            : absl::StrCat(prefix, ".", alias);
          if (cb.objects.contains(alias_candidate)) {
            candidate = std::move(alias_candidate);
          }
        }
      }
      prefix = std::move(candidate);
      start = dot + 1;
    }
    std::string_view leaf_seg = raw_path.substr(start);
    return prefix.empty() ? std::string(leaf_seg)
                          : absl::StrCat(prefix, ".", leaf_seg);
  };

  for (const ChangedPluginLeaf& leaf : leaves) {
    std::string resolved_path = resolve_traversal_path(leaf.path);
    auto data_it = cb.data.find(resolved_path);
    while (data_it == cb.data.end() && cb.objects.contains(resolved_path)) {
      resolved_path = absl::StrCat(resolved_path, ".value");
      data_it = cb.data.find(resolved_path);
    }
    if (data_it == cb.data.end()) {
      return false;
    }
    const TraversalCallback::ValueEntry& entry = data_it->second;
    if ((entry.flags &
         static_cast<uint32_t>(mitsuba::ParamFlags::ReadOnly)) != 0) {
      return false;
    }
    if (!StageTypedPluginWrite<Float>(entry, leaf.role, leaf.value,
                                      &staged_writes)) {
      return false;
    }

    size_t first_dot = resolved_path.find('.');
    if (first_dot == std::string::npos) {
      AppendUniqueKey(&root_keys, resolved_path);
    } else {
      AppendUniqueKey(&root_keys,
                      std::string_view(resolved_path).substr(0, first_dot));
      size_t dot_pos = first_dot;
      while (dot_pos != std::string::npos) {
        std::string obj_path = resolved_path.substr(0, dot_pos);
        size_t next_dot = resolved_path.find('.', dot_pos + 1);
        std::string_view child_key =
            next_dot == std::string::npos
                ? std::string_view(resolved_path).substr(dot_pos + 1)
                : std::string_view(resolved_path)
                      .substr(dot_pos + 1, next_dot - dot_pos - 1);
        if (cb.objects.contains(obj_path)) {
          AppendUniqueKey(&touched_object_keys[obj_path], child_key);
        }
        dot_pos = next_dot;
      }
    }
  }

  for (const auto& write_fn : staged_writes) {
    write_fn();
  }

  std::vector<std::pair<std::string, std::vector<std::string>>> ordered_objects(
      touched_object_keys.begin(), touched_object_keys.end());
  std::sort(ordered_objects.begin(), ordered_objects.end(),
            [](const auto& lhs, const auto& rhs) {
              auto depth_lhs =
                  std::count(lhs.first.begin(), lhs.first.end(), '.');
              auto depth_rhs =
                  std::count(rhs.first.begin(), rhs.first.end(), '.');
              if (depth_lhs != depth_rhs) return depth_lhs > depth_rhs;
              return lhs.first > rhs.first;
            });
  try {
    for (const auto& [obj_path, child_keys] : ordered_objects) {
      cb.objects.at(obj_path)->parameters_changed(child_keys);
    }
    root_obj->parameters_changed(root_keys);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

MI_VARIANT typename PrimTranslator<Float, Spectrum>::TranslatedLight
PrimTranslator<Float, Spectrum>::BuildLight(const LightSpec& spec) {
  mitsuba::Properties props = BuildLightProperties(spec);
  std::string id_str = spec.id.GetAsString();
  TranslatedLight res;
  if (spec.IsAreaLight()) {
    res.shape = mitsuba::PluginManager::instance()
                    ->create_object<mitsuba::Shape<Float, Spectrum>>(props);
    res.shape->set_id(id_str);
  } else {
    if (props.plugin_name() == "ies") {
      RegisterIesEmitter();
    }
    res.emitter = mitsuba::PluginManager::instance()
                      ->create_object<mitsuba::Emitter<Float, Spectrum>>(props);
    res.emitter->set_id(id_str);
  }
  return res;
}

namespace {

bool HasExplicitEmitterFrame(const PluginSpecNode& spec) {
  return spec.HasProperty("to_world") || spec.HasProperty("direction") ||
         spec.HasProperty("origin") || spec.HasProperty("position");
}

// Returns `nested_props` if `spec` has no IES profile, and otherwise the
// properties of an `ies` emitter wrapping the emitter they describe. Its light
// frame is `spec.transform` without scale.
template <typename Float, typename Spectrum>
mitsuba::Properties MaybeWrapIesProps(mitsuba::Properties nested_props,
                                      const LightSpec& spec) {
  if (spec.ies.file_path.empty()) {
    return nested_props;
  }
  RegisterIesEmitter();
  using Emitter = mitsuba::Emitter<Float, Spectrum>;
  mitsuba::Properties props("ies");
  props.set("filename", spec.ies.file_path);
  props.set("nested_emitter",
            mitsuba::PluginManager::instance()->create_object(
                nested_props, Emitter::Variant, Emitter::Type));
  props.set("angle_scale", spec.ies.angle_scale);
  props.set("normalize", spec.ies.normalize);
  PrimTranslator<Float, Spectrum>::SetTransformProperty(
      props, spec.transform.Map(RemoveScaleFromTransform));
  return props;
}

// Updates the light frame of the `ies` emitter `emitter` if `spec` has an IES
// profile, and returns the emitter it wraps (or `emitter` otherwise). The
// caller must still call `emitter->parameters_changed()`.
template <typename Float, typename Spectrum>
mitsuba::Emitter<Float, Spectrum>* UpdateIesFrameInPlace(
    mitsuba::Emitter<Float, Spectrum>* emitter, const LightSpec& spec,
    const PluginSpecNode& eff_spec) {
  if (spec.ies.file_path.empty()) {
    return emitter;
  }
  TraversalCallback cb;
  emitter->traverse(&cb);
  if (!(eff_spec.plugin_type == "ies" && HasExplicitEmitterFrame(eff_spec)) &&
      (cb.data.contains("to_world") || cb.objects.contains("to_world"))) {
    PrimTranslator<Float, Spectrum>::SetTransformParameter(
        cb, spec.transform.Map(RemoveScaleFromTransform));
  }
  auto nested_it = cb.objects.find("nested_emitter");
  return nested_it != cb.objects.end()
             ? static_cast<mitsuba::Emitter<Float, Spectrum>*>(
                   nested_it->second)
             : emitter;
}

mitsuba::Object* ResolvePluginSpecTargetEmitter(
    mitsuba::Object* emitter, const LightSpec& spec,
    const PluginSpecNode& emitter_spec) {
  if (emitter == nullptr) return nullptr;
  if (spec.ies.file_path.empty() || emitter_spec.plugin_type == "ies") {
    return emitter;
  }
  TraversalCallback cb;
  emitter->traverse(&cb);
  auto nested_it = cb.objects.find("nested_emitter");
  return nested_it != cb.objects.end() ? nested_it->second : nullptr;
}

void PopulateSpotDefaultBeamWidth(const LightSpec& spec,
                                  const mitsuba::Object* target_emitter,
                                  PluginSpecNode* emitter_spec) {
  if (target_emitter == nullptr ||
      target_emitter->class_name() != "SpotLight" ||
      spec.shaping_cone_angle != 0.0f ||
      emitter_spec->HasProperty("beam_width")) {
    return;
  }
  double cutoff = 20.0;
  if (auto it = emitter_spec->params.find("cutoff_angle");
      it != emitter_spec->params.end()) {
    if (std::optional<double> val = ExtractDoubleValue(it->second)) {
      cutoff = *val;
    }
  }
  emitter_spec->params["beam_width"] =
      VtValue(static_cast<float>(cutoff * 0.75));
}

}  // namespace

MI_VARIANT mitsuba::ref<mitsuba::Object>
PrimTranslator<Float, Spectrum>::CreateAreaEmitter(
    const LightSpec& spec, const PluginSpecNode& emitter_override) {
  const PluginSpecNode& eff_spec =
      emitter_override.Empty() ? spec.emitter_spec : emitter_override;
  mitsuba::Properties emitter_props;
  if (!spec.ies.file_path.empty() && eff_spec.plugin_type == "ies") {
    mitsuba::Properties area_props("area");
    area_props.set(
        "radiance",
        mitsuba::Color<float, 3>(spec.emission[0], spec.emission[1],
                                 spec.emission[2]));
    emitter_props =
        MaybeWrapIesProps<Float, Spectrum>(std::move(area_props), spec);
    if (HasExplicitEmitterFrame(eff_spec)) {
      emitter_props.remove_property("to_world");
    }
    MergePluginProperties(&emitter_props, eff_spec,
                          mitsuba::Emitter<Float, Spectrum>::Variant);
  } else {
    const std::string& emitter_type =
        eff_spec.plugin_type.empty() ? "area" : eff_spec.plugin_type;
    emitter_props = mitsuba::Properties(emitter_type);
    if (emitter_type == "area") {
      emitter_props.set(
          "radiance",
          mitsuba::Color<float, 3>(spec.emission[0], spec.emission[1],
                                   spec.emission[2]));
    }
    MergePluginProperties(&emitter_props, eff_spec,
                          mitsuba::Emitter<Float, Spectrum>::Variant);
    if (emitter_props.plugin_name() != "ies") {
      emitter_props =
          MaybeWrapIesProps<Float, Spectrum>(std::move(emitter_props), spec);
    }
  }
  if (emitter_props.plugin_name() == "ies") {
    RegisterIesEmitter();
  }
  return mitsuba::PluginManager::instance()->create_object(
      emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
      mitsuba::Emitter<Float, Spectrum>::Type);
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateAreaEmitterInPlace(
    mitsuba::Emitter<Float, Spectrum>* emitter, const LightSpec& spec,
    const PluginSpecNode& emitter_override) {
  const PluginSpecNode& eff_spec =
      emitter_override.Empty() ? spec.emitter_spec : emitter_override;
  mitsuba::Emitter<Float, Spectrum>* nested =
      UpdateIesFrameInPlace<Float, Spectrum>(emitter, spec, eff_spec);
  if (!eff_spec.HasProperty("radiance")) {
    TraversalCallback cb;
    nested->traverse(&cb);
    if (cb.data.contains("radiance.value") && cb.objects.contains("radiance")) {
      SetSpectrumParameter(cb, "radiance", spec.emission);
    }
  }
  emitter->parameters_changed();
}

MI_VARIANT bool PrimTranslator<Float, Spectrum>::TryUpdateAreaEmitterInPlace(
    mitsuba::Emitter<Float, Spectrum>* emitter, const LightSpec& spec,
    const PluginSpecNode& old_emitter_spec,
    const PluginSpecNode& new_emitter_spec) {
  mitsuba::Object* target_emitter =
      ResolvePluginSpecTargetEmitter(emitter, spec, new_emitter_spec);
  if (target_emitter == nullptr) return false;
  PluginSpecNode old_eff = old_emitter_spec;
  PluginSpecNode new_eff = new_emitter_spec;
  PopulateSpotDefaultBeamWidth(spec, target_emitter, &old_eff);
  PopulateSpotDefaultBeamWidth(spec, target_emitter, &new_eff);
  if (!TryUpdatePluginInPlace(target_emitter, old_eff, new_eff)) {
    return false;
  }
  try {
    UpdateAreaEmitterInPlace(emitter, spec, new_emitter_spec);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

MI_VARIANT mitsuba::Properties
PrimTranslator<Float, Spectrum>::BuildLightProperties(const LightSpec& spec) {
  mitsuba::Color<float, 3> color(spec.emission[0], spec.emission[1],
                                 spec.emission[2]);
  const bool has_explicit_emitter_frame =
      HasExplicitEmitterFrame(spec.emitter_spec);
  auto make_emitter_props = [&](std::string_view plugin) {
    mitsuba::Properties props(plugin);
    if (!has_explicit_emitter_frame) {
      SetTransformProperty(props, spec.transform);
    }
    return props;
  };
  auto make_area_shape_props = [&](std::string_view default_shape_plugin) {
    std::string_view shape_plugin = spec.shape_spec.plugin_type.empty()
                                        ? default_shape_plugin
                                        : spec.shape_spec.plugin_type;
    mitsuba::Properties props(shape_plugin);
    SetTransformProperty(props, spec.transform);
    props.set("visibility", spec.visibility);
    props.set("emitter", CreateAreaEmitter(spec));
    props.set("bsdf",
              mitsuba::ref<mitsuba::Object>(DefaultBsdf("", "null")));
    MergePluginProperties(&props, spec.shape_spec,
                          mitsuba::Shape<Float, Spectrum>::Variant);
    return props;
  };
  auto merge_emitter_props = [&](mitsuba::Properties default_props) {
    if (!spec.emitter_spec.plugin_type.empty() &&
        spec.emitter_spec.plugin_type != default_props.plugin_name()) {
      mitsuba::Properties custom_props(spec.emitter_spec.plugin_type);
      if (spec.emitter_spec.plugin_type != "constant" &&
          !has_explicit_emitter_frame) {
        SetTransformProperty(custom_props, spec.transform);
      }
      if (default_props.has_property("visibility") &&
          (spec.emitter_spec.plugin_type == "envmap" ||
           spec.emitter_spec.plugin_type == "constant")) {
        custom_props.set("visibility", spec.visibility);
      }
      MergePluginProperties(&custom_props, spec.emitter_spec,
                            mitsuba::Emitter<Float, Spectrum>::Variant);
      return custom_props;
    }
    MergePluginProperties(&default_props, spec.emitter_spec,
                          mitsuba::Emitter<Float, Spectrum>::Variant);
    return default_props;
  };

  if (spec.prim_type == HdPrimTypeTokens->sphereLight) {
    if (!spec.treat_as_point) return make_area_shape_props("sphere");
    const bool is_spot = spec.shaping_cone_angle != 0.0f;
    const bool ies_wrapper_override =
        !spec.ies.file_path.empty() && spec.emitter_spec.plugin_type == "ies";
    mitsuba::Properties props(is_spot ? "spot" : "point");
    if (!has_explicit_emitter_frame || ies_wrapper_override) {
      SetTransformProperty(props, spec.transform);
    }
    props.set("intensity", color);
    if (is_spot) {
      props.set("beam_width", spec.shaping_cone_beam_width);
      props.set("cutoff_angle", spec.shaping_cone_angle);
    }
    if (ies_wrapper_override) {
      props = MaybeWrapIesProps<Float, Spectrum>(std::move(props), spec);
      if (has_explicit_emitter_frame) {
        props.remove_property("to_world");
      }
      MergePluginProperties(&props, spec.emitter_spec,
                            mitsuba::Emitter<Float, Spectrum>::Variant);
      return props;
    }
    props = merge_emitter_props(std::move(props));
    if (props.plugin_name() != "ies") {
      props = MaybeWrapIesProps<Float, Spectrum>(std::move(props), spec);
    }
    return props;
  }

  if (spec.prim_type == HdPrimTypeTokens->domeLight) {
    if (!spec.texture_file_path.empty() &&
        (spec.emitter_spec.plugin_type.empty() ||
         spec.emitter_spec.plugin_type == "envmap")) {
      if (spec.emitter_spec.HasProperty("filename") ||
          spec.emitter_spec.HasProperty("bitmap")) {
        mitsuba::Properties props = make_emitter_props("envmap");
        props.set("scale", (color[0] + color[1] + color[2]) / 3.f);
        props.set("visibility", spec.visibility);
        return merge_emitter_props(std::move(props));
      }
      try {
        mitsuba::ref<mitsuba::Bitmap> bitmap =
            ReadBitmap(spec.texture_file_path);
        mitsuba::Properties props = make_emitter_props("envmap");
        props.set("scale", (color[0] + color[1] + color[2]) / 3.f);
        props.set("bitmap", bitmap.get());
        props.set("visibility", spec.visibility);
        return merge_emitter_props(std::move(props));
      } catch (const std::exception& e) {
        TF_WARN("Failed to load environment texture '%s': %s",
                spec.texture_file_path.c_str(), e.what());
      }
    }
    mitsuba::Properties props("constant");
    props.set("radiance", color);
    props.set("visibility", spec.visibility);
    return merge_emitter_props(std::move(props));
  }

  if (spec.prim_type == HdPrimTypeTokens->distantLight) {
    mitsuba::Properties props = make_emitter_props("directional");
    props.set("irradiance", color);
    props.set("angle", spec.angle);
    return merge_emitter_props(std::move(props));
  }

  if (spec.prim_type == HdPrimTypeTokens->rectLight) {
    return make_area_shape_props("rectangle");
  }

  if (spec.prim_type == HdPrimTypeTokens->diskLight) {
    return make_area_shape_props("disk");
  }

  mitsuba::Properties props("constant");
  return merge_emitter_props(std::move(props));
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateLightInPlace(
    mitsuba::Object* light_obj, const LightSpec& spec) {
  using ScalarFloat = dr::scalar_t<Float>;
  if (auto* shape = dynamic_cast<mitsuba::Shape<Float, Spectrum>*>(light_obj)) {
    TraversalCallback cb_shape;
    shape->traverse(&cb_shape);
    if (!spec.shape_spec.HasProperty("to_world") &&
        (cb_shape.data.contains("to_world") ||
         cb_shape.objects.contains("to_world"))) {
      SetTransformParameter(cb_shape, spec.transform);
      shape->parameters_changed();
    }
    if (shape->is_emitter() && spec.shape_spec.children.count("emitter") == 0) {
      UpdateAreaEmitterInPlace(shape->emitter(), spec);
    }
  } else if (auto* emitter =
                 dynamic_cast<mitsuba::Emitter<Float, Spectrum>*>(light_obj)) {
    TraversalCallback cb;
    UpdateIesFrameInPlace<Float, Spectrum>(emitter, spec, spec.emitter_spec)
        ->traverse(&cb);
    const bool has_explicit_emitter_frame =
        HasExplicitEmitterFrame(spec.emitter_spec);

    if (spec.prim_type == HdPrimTypeTokens->sphereLight &&
        spec.treat_as_point) {
      if (!has_explicit_emitter_frame &&
          (cb.data.contains("to_world") || cb.objects.contains("to_world"))) {
        SetTransformParameter(cb, spec.transform);
      }
      if (!spec.emitter_spec.HasProperty("intensity") &&
          cb.data.contains("intensity.value") &&
          cb.objects.contains("intensity")) {
        SetSpectrumParameter(cb, "intensity", spec.emission);
      }
      if (spec.shaping_cone_angle != 0.0f) {
        if (!spec.emitter_spec.HasProperty("beam_width") &&
            cb.data.contains("beam_width")) {
          cb.set<Float>("beam_width",
                        MakeOpaqueFloat<Float>(
                            static_cast<ScalarFloat>(
                                spec.shaping_cone_beam_width)));
        }
        if (!spec.emitter_spec.HasProperty("cutoff_angle") &&
            cb.data.contains("cutoff_angle")) {
          cb.set<Float>("cutoff_angle",
                        MakeOpaqueFloat<Float>(
                            static_cast<ScalarFloat>(spec.shaping_cone_angle)));
        }
      }
    } else if (spec.prim_type == HdPrimTypeTokens->domeLight) {
      // A domeLight with a missing/corrupt texture falls back to a "constant"
      // emitter in BuildLightProperties, which has "radiance.value" rather than
      // "scale" and "to_world".
      if (cb.data.contains("scale")) {
        if (!has_explicit_emitter_frame &&
            (cb.data.contains("to_world") || cb.objects.contains("to_world"))) {
          SetTransformParameter(cb, spec.transform);
        }
        if (!spec.emitter_spec.HasProperty("scale")) {
          cb.set<Float>(
              "scale",
              MakeOpaqueFloat<Float>(static_cast<ScalarFloat>(
                  (spec.emission[0] + spec.emission[1] + spec.emission[2]) /
                  3.f)));
        }
        emitter->parameters_changed({"scale", "to_world"});
        return;
      } else if (!spec.emitter_spec.HasProperty("radiance") &&
                 cb.data.contains("radiance.value") &&
                 cb.objects.contains("radiance")) {
        SetSpectrumParameter(cb, "radiance", spec.emission);
      }
    } else if (spec.prim_type == HdPrimTypeTokens->distantLight) {
      if (!has_explicit_emitter_frame &&
          (cb.data.contains("to_world") || cb.objects.contains("to_world"))) {
        SetTransformParameter(cb, spec.transform);
      }
      if (!spec.emitter_spec.HasProperty("irradiance") &&
          cb.data.contains("irradiance.value") &&
          cb.objects.contains("irradiance")) {
        SetSpectrumParameter(cb, "irradiance", spec.emission);
      }
      if (!spec.emitter_spec.HasProperty("angle") &&
          cb.data.contains("angle")) {
        cb.set<Float>("angle", MakeOpaqueFloat<Float>(
                                   static_cast<ScalarFloat>(spec.angle)));
      }
    }
    emitter->parameters_changed();
  }
}

MI_VARIANT bool PrimTranslator<Float, Spectrum>::TryUpdateLightInPlace(
    mitsuba::Object* light_obj, const LightSpec& spec) {
  if (light_obj == nullptr || !spec.committed_emitter_spec.has_value() ||
      !spec.committed_shape_spec.has_value()) {
    return false;
  }
  if (auto* shape = dynamic_cast<mitsuba::Shape<Float, Spectrum>*>(light_obj)) {
    for (const char* geom_key : {"center", "radius", "p0", "p1"}) {
      if (spec.shape_spec.params.count(geom_key) > 0) {
        return false;
      }
    }
    if (!TryUpdatePluginInPlace(shape, *spec.committed_shape_spec,
                                spec.shape_spec)) {
      return false;
    }
    if (shape->is_emitter() && spec.shape_spec.children.count("emitter") == 0) {
      if (!TryUpdateAreaEmitterInPlace(shape->emitter(), spec,
                                       *spec.committed_emitter_spec,
                                       spec.emitter_spec)) {
        return false;
      }
    }
    TraversalCallback cb_shape;
    shape->traverse(&cb_shape);
    if (!spec.shape_spec.HasProperty("to_world") &&
        (cb_shape.data.contains("to_world") ||
         cb_shape.objects.contains("to_world"))) {
      SetTransformParameter(cb_shape, spec.transform);
      shape->parameters_changed({"to_world"});
    }
    return true;
  }
  if (auto* emitter =
          dynamic_cast<mitsuba::Emitter<Float, Spectrum>*>(light_obj)) {
    mitsuba::Object* target_emitter =
        ResolvePluginSpecTargetEmitter(emitter, spec, spec.emitter_spec);
    if (target_emitter == nullptr) return false;
    PluginSpecNode old_eff = *spec.committed_emitter_spec;
    PluginSpecNode new_eff = spec.emitter_spec;
    PopulateSpotDefaultBeamWidth(spec, target_emitter, &old_eff);
    PopulateSpotDefaultBeamWidth(spec, target_emitter, &new_eff);
    if (!TryUpdatePluginInPlace(target_emitter, old_eff, new_eff)) {
      return false;
    }
    try {
      UpdateLightInPlace(emitter, spec);
    } catch (const std::exception&) {
      return false;
    }
    return true;
  }
  return false;
}

namespace {

void EraseFilmCropParams(PluginSpecNode* film_spec) {
  film_spec->params.erase("crop_offset_x");
  film_spec->params.erase("crop_offset_y");
  film_spec->params.erase("crop_width");
  film_spec->params.erase("crop_height");
}

}  // namespace

MI_VARIANT mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> PrimTranslator<
    Float, Spectrum>::BuildSensor(const CameraSpec& spec, bool use_box_filter,
                                  const PluginSpecNode& sampler_spec,
                                  bool force_independent_sampler) {
  const std::string& sensor_type = spec.sensor_spec.plugin_type.empty()
                                       ? spec.sensor_type
                                       : spec.sensor_spec.plugin_type;
  mitsuba::Properties props(sensor_type);
  const bool has_explicit_ray_frame =
      spec.sensor_spec.HasProperty("to_world") ||
      spec.sensor_spec.HasProperty("direction") ||
      spec.sensor_spec.HasProperty("origin");
  if (sensor_type != "irradiancemeter" &&
      !spec.target_shape_id.has_value() && !has_explicit_ray_frame) {
    SetTransformProperty(props, spec.transform);
  }
  if (sensor_type == "perspective" || sensor_type == "thinlens") {
    if (!spec.sensor_spec.HasProperty("focal_length")) {
      props.set("fov", spec.fov);
      props.set("fov_axis", "x");
    }
    props.set("principal_point_offset_x", spec.horizontal_aperture_offset);
    props.set("principal_point_offset_y", spec.vertical_aperture_offset);
    if (sensor_type == "thinlens") {
      props.set("aperture_radius", 0.0f);
    }
  }
  if (sensor_type == "perspective" || sensor_type == "thinlens" ||
      sensor_type == "orthographic") {
    props.set("near_clip", spec.near_clip);
    props.set("far_clip", spec.far_clip);
  }
  props.set("shutter_open", spec.shutter[0]);
  props.set("shutter_close", spec.shutter[1]);

  PluginSpecNode effective_spec = spec.sensor_spec;
  effective_spec.plugin_type = sensor_type;

  if (auto film_it = effective_spec.children.find("film");
      film_it != effective_spec.children.end()) {
    EraseFilmCropParams(&film_it->second);
  }
  const bool is_signal_sensor =
      sensor_type == "radiancemeter" || sensor_type == "distant";
  if (effective_spec.children.find("film") != effective_spec.children.end() ||
      use_box_filter || is_signal_sensor) {
    PluginSpecNode& film_spec = effective_spec.children["film"];
    if (film_spec.plugin_type.empty()) {
      film_spec.plugin_type = "hdrfilm";
    }
    if (is_signal_sensor) {
      film_spec.params.try_emplace("width", VtValue(1));
      film_spec.params.try_emplace("height", VtValue(1));
    }
    if (use_box_filter) {
      film_spec.children.erase("reconstruction_filter");
      film_spec.children.erase("rfilter");
      film_spec.children["pixel_filter"] = PluginSpecNode("box");
    }
  }

  if (auto cam_sampler_it = effective_spec.children.find("sampler");
      cam_sampler_it != effective_spec.children.end()) {
    if (force_independent_sampler) {
      cam_sampler_it->second = PluginSpecNode("independent");
    } else if (cam_sampler_it->second.plugin_type.empty()) {
      cam_sampler_it->second.plugin_type = "independent";
    }
  }

  MergePluginProperties(&props, effective_spec,
                        mitsuba::Sensor<Float, Spectrum>::Variant);

  if (!sampler_spec.Empty() && !props.has_property("sampler")) {
    try {
      mitsuba::ref<mitsuba::Sampler<Float, Spectrum>> sampler =
          BuildPluginObject<mitsuba::Sampler<Float, Spectrum>>(sampler_spec);
      if (sampler) {
        props.set("sampler", sampler.get(), /*raise_if_exists=*/false);
      }
    } catch (const std::exception& e) {
      TF_RUNTIME_ERROR("Failed to create sampler '%s' for sensor '%s': %s",
                       sampler_spec.plugin_type.c_str(), spec.id.GetText(),
                       e.what());
    }
  }
  mitsuba::ref<mitsuba::Object> expanded = CreateExpandedObject(
      props, mitsuba::Sensor<Float, Spectrum>::Variant,
      mitsuba::ObjectType::Sensor);
  auto* typed_sensor =
      dynamic_cast<mitsuba::Sensor<Float, Spectrum>*>(expanded.get());
  if (typed_sensor == nullptr) {
    throw std::runtime_error("Mitsuba plugin '" + sensor_type +
                             "' is not a valid Sensor.");
  }
  mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> sensor(typed_sensor);
  sensor->set_id(spec.id.GetAsString());
  return sensor;
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateSensorAspectRatio(
    mitsuba::Object* sensor_obj, const CameraSpec& spec, double aspect) {
  if (aspect <= 0.0) return;
  const std::string& sensor_type = spec.sensor_spec.plugin_type.empty()
                                       ? spec.sensor_type
                                       : spec.sensor_spec.plugin_type;
  if (sensor_type != "perspective" && sensor_type != "thinlens") {
    return;
  }
  if (!spec.sensor_spec.HasProperty("focal_length") &&
      !spec.sensor_spec.HasProperty("fov_axis")) {
    return;
  }
  auto* sensor = dynamic_cast<mitsuba::Sensor<Float, Spectrum>*>(sensor_obj);
  if (!sensor) return;

  std::optional<double> x_fov = ComputePerspectiveXFov(spec, aspect);
  if (!x_fov.has_value()) return;
  TraversalCallback cb;
  sensor->traverse(&cb);
  if (cb.data.contains("x_fov")) {
    cb.set<Float>(
        "x_fov",
        MakeOpaqueFloat<Float>(static_cast<dr::scalar_t<Float>>(*x_fov)));
    sensor->parameters_changed({"x_fov"});
  }
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateSensorInPlace(
    mitsuba::Object* sensor_obj, const CameraSpec& spec) {
  using ScalarFloat = dr::scalar_t<Float>;
  auto* sensor = dynamic_cast<mitsuba::Sensor<Float, Spectrum>*>(sensor_obj);
  if (!sensor) return;

  TraversalCallback cb;
  sensor->traverse(&cb);

  const std::string& sensor_type = spec.sensor_spec.plugin_type.empty()
                                       ? spec.sensor_type
                                       : spec.sensor_spec.plugin_type;
  double aspect = 1.0;
  if (const auto* film = sensor->film(); film && film->size().y() > 0) {
    aspect = static_cast<double>(film->size().x()) /
             static_cast<double>(film->size().y());
  }

  std::vector<std::string> changed_keys;
  if ((spec.dirty_bits & HdCamera::DirtyBits::DirtyTransform) &&
      !spec.sensor_spec.HasProperty("to_world") &&
      !spec.sensor_spec.HasProperty("direction") &&
      !spec.sensor_spec.HasProperty("origin") &&
      (cb.data.contains("to_world") || cb.objects.contains("to_world"))) {
    SetTransformParameter(cb, spec.transform);
    changed_keys.push_back("to_world");
  }
  if (spec.dirty_bits & HdCamera::DirtyBits::DirtyParams) {
    if (!spec.sensor_spec.HasProperty("near_clip") &&
        cb.data.contains("near_clip")) {
      cb.set<ScalarFloat>("near_clip", spec.near_clip);
      changed_keys.push_back("near_clip");
    }
    if (!spec.sensor_spec.HasProperty("far_clip") &&
        cb.data.contains("far_clip")) {
      cb.set<ScalarFloat>("far_clip", spec.far_clip);
      changed_keys.push_back("far_clip");
    }
    if (sensor_type == "perspective" || sensor_type == "thinlens" ||
        sensor_type.empty()) {
      if (!spec.sensor_spec.HasProperty("fov") &&
          !spec.sensor_spec.HasProperty("focal_length") &&
          cb.data.contains("x_fov")) {
        if (std::optional<double> x_fov = ComputePerspectiveXFov(spec, aspect);
            x_fov.has_value()) {
          cb.set<Float>(
              "x_fov",
              MakeOpaqueFloat<Float>(static_cast<ScalarFloat>(*x_fov)));
          changed_keys.push_back("x_fov");
        }
      }
      if (!spec.sensor_spec.HasProperty("principal_point_offset_x") &&
          cb.data.contains("principal_point_offset_x")) {
        cb.set<Float>("principal_point_offset_x",
                      MakeOpaqueFloat<Float>(spec.horizontal_aperture_offset));
        changed_keys.push_back("principal_point_offset_x");
      }
      if (!spec.sensor_spec.HasProperty("principal_point_offset_y") &&
          cb.data.contains("principal_point_offset_y")) {
        cb.set<Float>("principal_point_offset_y",
                      MakeOpaqueFloat<Float>(spec.vertical_aperture_offset));
        changed_keys.push_back("principal_point_offset_y");
      }
    }
  }
  if (!changed_keys.empty()) {
    sensor->parameters_changed(changed_keys);
  }
}

MI_VARIANT bool PrimTranslator<Float, Spectrum>::TryUpdateSensorInPlace(
    mitsuba::Object* sensor_obj, const CameraSpec& spec) {
  using ScalarFloat = dr::scalar_t<Float>;
  auto* sensor = dynamic_cast<mitsuba::Sensor<Float, Spectrum>*>(sensor_obj);
  if (!sensor || !spec.committed_sensor_spec.has_value()) {
    return false;
  }

  const std::string& sensor_type = spec.sensor_spec.plugin_type.empty()
                                       ? spec.sensor_type
                                       : spec.sensor_spec.plugin_type;
  PluginSpecNode old_filtered = *spec.committed_sensor_spec;
  PluginSpecNode new_filtered = spec.sensor_spec;
  if (auto film_it = old_filtered.children.find("film");
      film_it != old_filtered.children.end()) {
    EraseFilmCropParams(&film_it->second);
  }
  if (auto film_it = new_filtered.children.find("film");
      film_it != new_filtered.children.end()) {
    EraseFilmCropParams(&film_it->second);
  }
  bool fov_spec_changed = false;
  if (sensor_type == "perspective" || sensor_type == "thinlens") {
    for (const char* key : {"fov", "focal_length", "fov_axis"}) {
      auto old_it = old_filtered.params.find(key);
      auto new_it = new_filtered.params.find(key);
      const bool had_old = (old_it != old_filtered.params.end());
      const bool has_new = (new_it != new_filtered.params.end());
      if (had_old != has_new || (had_old && old_it->second != new_it->second)) {
        fov_spec_changed = true;
      }
      if (had_old) old_filtered.params.erase(old_it);
      if (has_new) new_filtered.params.erase(new_it);
    }
  }

  double aspect = 1.0;
  if (const auto* film = sensor->film(); film && film->size().y() > 0) {
    aspect = static_cast<double>(film->size().x()) /
             static_cast<double>(film->size().y());
  }
  std::optional<double> updated_x_fov;
  if (fov_spec_changed) {
    updated_x_fov = ComputePerspectiveXFov(spec, aspect);
    if (!updated_x_fov.has_value()) {
      return false;
    }
  }

  if (!TryUpdatePluginInPlace(sensor, old_filtered, new_filtered)) {
    return false;
  }

  if (updated_x_fov.has_value()) {
    TraversalCallback cb;
    sensor->traverse(&cb);
    if (!cb.data.contains("x_fov")) {
      return false;
    }
    cb.set<Float>(
        "x_fov",
        MakeOpaqueFloat<Float>(static_cast<ScalarFloat>(*updated_x_fov)));
    sensor->parameters_changed({"x_fov"});
  }

  UpdateSensorInPlace(sensor, spec);
  return true;
}

namespace {

// Wrap host-side data as a row-major ``(rows, cols)`` tensor of floats.
template <typename Mesh>
typename Mesh::TensorXf32 LoadFloatTensor(const void* data, size_t rows,
                                          size_t cols) {
  using FloatBuffer = typename Mesh::FloatBuffer;
  return typename Mesh::TensorXf32(dr::load<FloatBuffer>(data, rows * cols),
                                   {rows, cols});
}

// Wrap host-side data as a row-major ``(rows, 3)`` tensor of vertex indices.
template <typename Mesh>
typename Mesh::TensorXu32 LoadFaceTensor(const void* data, size_t rows) {
  using IndexBuffer = typename Mesh::IndexBuffer;
  return typename Mesh::TensorXu32(dr::load<IndexBuffer>(data, rows * 3),
                                   {rows, size_t(3)});
}

// Populates `dst` (sized `vertex_count * channels`) from `state`. Returns false
// if `state` does not hold a supported array type of size `vertex_count`.
// If the source array layout already matches `channels` without needing a V
// flip, sets `*direct_ptr` to the contiguous source float buffer instead of
// copying into `dst`.
bool ExtractAttributeBuffer(const std::string& name, size_t channels,
                            size_t vertex_count, const PrimvarState& state,
                            std::vector<float>* dst,
                            const float** direct_ptr) {
  *direct_ptr = nullptr;
  const VtValue& value = state.value;
  if (value.GetArraySize() != vertex_count) {
    TF_WARN("Primvar '%s' has %zu values, but the mesh has %zu vertices.",
            name.c_str(), value.GetArraySize(), vertex_count);
    return false;
  }
  if (value.IsHolding<VtFloatArray>()) {
    const float* src = value.UncheckedGet<VtFloatArray>().cdata();
    if (channels == 1) {
      *direct_ptr = src;
      return true;
    }
    dst->assign(vertex_count * channels, 0.f);
    float* out = dst->data();
    for (size_t i = 0; i < vertex_count; ++i) {
      std::fill_n(out + i * channels, channels, src[i]);
    }
    return true;
  }
  if (value.IsHolding<VtVec2fArray>()) {
    const GfVec2f* src = value.UncheckedGet<VtVec2fArray>().cdata();
    const bool flip =
        state.descriptor.role == HdPrimvarRoleTokens->textureCoordinate;
    if (channels == 2 && !flip) {
      *direct_ptr = src[0].data();
      return true;
    }
    dst->assign(vertex_count * channels, 0.f);
    float* out = dst->data();
    for (size_t i = 0; i < vertex_count; ++i) {
      out[i * channels] = src[i][0];
      if (channels > 1) {
        out[i * channels + 1] = flip ? 1.f - src[i][1] : src[i][1];
      }
    }
    return true;
  }
  if (value.IsHolding<VtVec3fArray>()) {
    const GfVec3f* src = value.UncheckedGet<VtVec3fArray>().cdata();
    if (channels == 3) {
      *direct_ptr = src[0].data();
      return true;
    }
    dst->assign(vertex_count * channels, 0.f);
    float* out = dst->data();
    const size_t copy_ch = std::min<size_t>(channels, 3);
    for (size_t i = 0; i < vertex_count; ++i) {
      std::copy_n(src[i].data(), copy_ch, out + i * channels);
    }
    return true;
  }
  return false;
}

// Adds the primvars in `mesh_attributes` that exist in `primvars` to `packed`
// as vertex attributes with the requested channel count.
void AddPackedMeshAttributes(mitsuba::PackedMesh& packed, size_t vertex_count,
                             const PrimvarMap& primvars,
                             const MeshAttributeRequests& mesh_attributes) {
  if (mesh_attributes.empty()) return;
  std::vector<float> scratch;
  for (const auto& [name, channels] : mesh_attributes) {
    auto it = primvars.find(TfToken(name));
    if (it == primvars.end()) continue;
    const float* direct_ptr = nullptr;
    if (!ExtractAttributeBuffer(name, channels, vertex_count, it->second,
                                &scratch, &direct_ptr)) {
      continue;
    }
    float* dst =
        packed.add_attribute(MeshAttributeName(name, channels), channels);
    const float* src = direct_ptr ? direct_ptr : scratch.data();
    std::memcpy(dst, src, vertex_count * channels * sizeof(float));
  }
}

// Adds the primvars in `mesh_attributes` that exist in `primvars` to `mesh` as
// vertex attributes with the requested channel count. Float primvars are
// broadcast, and vec2 primvars are padded with zeros. Texture coordinates were
// flipped to Mitsuba's convention by `TransformPrimvars` and are flipped back
// to USD's convention.
template <typename Mesh>
void AddMeshAttributes(Mesh* mesh, const PrimvarMap& primvars,
                       const MeshAttributeRequests& mesh_attributes) {
  if (mesh_attributes.empty()) return;
  const size_t vertex_count = mesh->vertex_count();
  std::vector<float> data;
  for (const auto& [name, channels] : mesh_attributes) {
    auto it = primvars.find(TfToken(name));
    if (it == primvars.end()) continue;
    const float* direct_ptr = nullptr;
    if (!ExtractAttributeBuffer(name, channels, vertex_count, it->second, &data,
                                &direct_ptr)) {
      continue;
    }
    const float* src = direct_ptr ? direct_ptr : data.data();
    mesh->add_attribute(MeshAttributeName(name, channels),
                        LoadFloatTensor<Mesh>(src, vertex_count, channels));
  }
}

}  // namespace

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildMesh(
    const std::string& id, const VtIntArray& face_indices,
    const PrimvarMap& primvars, const MeshAttributeRequests& mesh_attributes,
    mitsuba::Object* bsdf, mitsuba::Object* emitter_ptr,
    mitsuba::Object* sensor_ptr, mitsuba::Object* interior_medium,
    const PluginSpecNode& shape_spec) {
  using Mesh = mitsuba::Mesh<Float, Spectrum>;

  auto points_it = primvars.find(HdTokens->points);
  if (points_it == primvars.end()) {
    TF_RUNTIME_ERROR("Mesh %s has no points.", id.c_str());
    return nullptr;
  }

  const auto& points_array =
      points_it->second.value.UncheckedGet<VtVec3fArray>();
  size_t vertex_count = points_array.size();
  if (vertex_count == 0) return nullptr;

  size_t face_count = face_indices.size() / 3;
  if (face_count == 0) return nullptr;

  auto normals_it = primvars.find(HdTokens->normals);
  const GfVec3f* normals_ptr = nullptr;
  if (normals_it != primvars.end() &&
      normals_it->second.value.IsHolding<VtVec3fArray>()) {
    const auto& normals_array =
        normals_it->second.value.UncheckedGet<VtVec3fArray>();
    if (normals_array.size() == vertex_count) {
      normals_ptr = normals_array.cdata();
    }
  }
  const bool has_normals = normals_ptr != nullptr;

  static const TfToken kStToken("st");
  const GfVec2f* texcoords_ptr = nullptr;
  auto texcoords_it = primvars.find(kStToken);
  if (texcoords_it != primvars.end() &&
      texcoords_it->second.value.IsHolding<VtVec2fArray>()) {
    const auto& texcoords_array =
        texcoords_it->second.value.UncheckedGet<VtVec2fArray>();
    if (texcoords_array.size() == vertex_count) {
      texcoords_ptr = texcoords_array.cdata();
    }
  }

  mitsuba::Properties props;
  props.set_id(id);
  // Without authored normals the mesh is flat shaded. Requesting face normals
  // keeps Mitsuba from deriving smooth shading normals of its own: normal
  // computation is handled explicitly by the hydra delegate.
  props.set("face_normals", !has_normals);
  if (emitter_ptr != nullptr) {
    props.set("emitter", emitter_ptr);
  }
  if (sensor_ptr != nullptr) {
    props.set("sensor", sensor_ptr);
  }
  if (interior_medium != nullptr) {
    props.set("interior", interior_medium);
  }
  if (!shape_spec.Empty()) {
    try {
      MergePluginProperties(&props, shape_spec,
                            mitsuba::Shape<Float, Spectrum>::Variant);
    } catch (const std::exception& e) {
      TF_RUNTIME_ERROR("Failed to apply shape properties for mesh '%s': %s",
                       id.c_str(), e.what());
    }
  }
  if (bsdf != nullptr && !props.has_property("bsdf")) {
    props.set("bsdf", bsdf);
  }
  // Mesh::from_packed() requires flip_normals to be baked into the PackedMesh
  // rather than left on the Mesh properties; to_world was already folded into
  // the vertex positions by RunGeometryPipeline.
  const bool flip_normals = props.get<bool>("flip_normals", false);
  props.remove_property("flip_normals");
  props.remove_property("to_world");
  mitsuba::ref<Mesh> mesh = new Mesh(props);

  const bool use_vertex_normals = has_normals && !mesh->has_face_normals();
  mitsuba::PackedMesh packed(
      dr::backend_v<Float>, vertex_count, face_count,
      mitsuba::make_layout(use_vertex_normals, texcoords_ptr != nullptr));

  const GfVec3f* points_ptr = points_array.cdata();
  float* vrec = packed.vertices.data();
  for (size_t i = 0; i < vertex_count; ++i, vrec += mitsuba::MeshVertexStride) {
    const GfVec3f& p = points_ptr[i];
    vrec[mitsuba::PackedPositionOffset + 0] = p[0];
    vrec[mitsuba::PackedPositionOffset + 1] = p[1];
    vrec[mitsuba::PackedPositionOffset + 2] = p[2];
    packed.bbox.expand(mitsuba::PackedMesh::ScalarPoint3f(p[0], p[1], p[2]));
    if (use_vertex_normals) {
      const GfVec3f& n = normals_ptr[i];
      vrec[mitsuba::PackedFrameOffset + 0] = n[0];
      vrec[mitsuba::PackedFrameOffset + 1] = n[1];
      vrec[mitsuba::PackedFrameOffset + 2] = n[2];
    } else {
      vrec[mitsuba::PackedFrameOffset + 0] = 0.f;
      vrec[mitsuba::PackedFrameOffset + 1] = 0.f;
      vrec[mitsuba::PackedFrameOffset + 2] = 0.f;
    }
    if (texcoords_ptr) {
      const GfVec2f& uv = texcoords_ptr[i];
      vrec[mitsuba::PackedTexcoordOffset + 0] = uv[0];
      vrec[mitsuba::PackedTexcoordOffset + 1] = uv[1];
    } else {
      vrec[mitsuba::PackedTexcoordOffset + 0] = 0.f;
      vrec[mitsuba::PackedTexcoordOffset + 1] = 0.f;
    }
  }

  const int* faces_in = face_indices.cdata();
  uint32_t* frec = packed.faces.data();
  for (size_t i = 0; i < face_count; ++i, frec += mitsuba::MeshFaceStride) {
    frec[0] = static_cast<uint32_t>(faces_in[i * 3 + 0]);
    frec[1] = static_cast<uint32_t>(faces_in[i * 3 + 1]);
    frec[2] = static_cast<uint32_t>(faces_in[i * 3 + 2]);
    frec[3] = 0u;
  }

  if (flip_normals) {
    packed.set_transform(ScalarAffineTransform4f(), /*flip_normals=*/true);
    packed.transform_records();
  }

  AddPackedMeshAttributes(packed, vertex_count, primvars, mesh_attributes);
  mesh->from_packed(std::move(packed));
  return mitsuba::ref<mitsuba::Shape<Float, Spectrum>>(mesh.get());
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateMeshInPlace(
    mitsuba::Object* mesh_obj, const VtIntArray& face_indices,
    const PrimvarMap& primvars, const MeshAttributeRequests& mesh_attributes,
    HdDirtyBits dirty_bits) {
  using Mesh = mitsuba::Mesh<Float, Spectrum>;
  using TensorXf32 = typename Mesh::TensorXf32;
  using TensorXu32 = typename Mesh::TensorXu32;
  auto* mesh = dynamic_cast<Mesh*>(mesh_obj);
  if (!mesh) return;

  auto points_it = primvars.find(HdTokens->points);
  if (points_it == primvars.end()) return;
  const auto& points_array =
      points_it->second.value.UncheckedGet<VtVec3fArray>();
  size_t vertex_count = points_array.size();
  size_t face_count = face_indices.size() / 3;
  if (vertex_count == 0 || face_count == 0) return;

  TraversalCallback cb("", nullptr, /*recurse_objects=*/false);
  mesh->traverse(&cb);

  const bool update_topology_and_uvs =
      (dirty_bits & (HdChangeTracker::DirtyTopology |
                     HdChangeTracker::DirtyPrimvar |
                     HdChangeTracker::DirtyNormals)) != 0 ||
      vertex_count != mesh->vertex_count() ||
      face_count != mesh->face_count();

  std::vector<std::string> keys = {"positions"};
  cb.set<TensorXf32>(
      "positions",
      LoadFloatTensor<Mesh>(points_array.cdata(), vertex_count, 3));
  if (update_topology_and_uvs) {
    cb.set<TensorXu32>("faces",
                       LoadFaceTensor<Mesh>(face_indices.cdata(), face_count));
    keys.push_back("faces");
  }

  auto normals_it = primvars.find(HdTokens->normals);
  if (normals_it != primvars.end() && mesh->has_normals()) {
    const auto& normals_array =
        normals_it->second.value.UncheckedGet<VtVec3fArray>();
    cb.set<TensorXf32>("normals",
                       LoadFloatTensor<Mesh>(normals_array.cdata(),
                                             normals_array.size(), 3));
  }
  // Always add "normals" to the keys: This prevents Mitsuba from recomputing
  // the normals. We handle normal recomputation explicitly in the hydra
  // delegate.
  keys.push_back("normals");

  if (update_topology_and_uvs) {
    static const TfToken kStToken("st");
    auto texcoords_it = primvars.find(kStToken);
    if (texcoords_it != primvars.end() && mesh->has_texcoords()) {
      const auto& texcoords_array =
          texcoords_it->second.value.UncheckedGet<VtVec2fArray>();
      cb.set<TensorXf32>("texcoords",
                         LoadFloatTensor<Mesh>(texcoords_array.cdata(),
                                               texcoords_array.size(), 2));
      keys.push_back("texcoords");
    }
    // The vertex count or user primvars may have changed, so remove and re-add
    // custom attributes.
    for (const auto& [key, value] : cb.data) {
      if (absl::StartsWith(key, kVertexAttributePrefix)) {
        mesh->remove_attribute(key);
      }
    }
  }

  mesh->parameters_changed(keys);
  if (update_topology_and_uvs) {
    AddMeshAttributes(mesh, primvars, mesh_attributes);
  }
}

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildCurves(const CurveSpec& spec,
                                             mitsuba::Object* bsdf) {
  static const absl::NoDestructor<std::string> kCurveResourcePath([]() {
    if (auto plugin =
            PlugRegistry::GetInstance().GetPluginWithName("hdMitsuba")) {
      return plugin->GetResourcePath() + "/curve.txt";
    }
    return std::string();
  }());

  const std::string id_str = spec.id.GetAsString();
  mitsuba::Properties props(spec.plugin_name);
  if (!kCurveResourcePath->empty()) {
    props.set("filename", *kCurveResourcePath);
  } else {
    TF_RUNTIME_ERROR("Failed to find plugin 'hdMitsuba' to locate resources.");
  }
  if (bsdf != nullptr) {
    props.set("bsdf", bsdf);
  }

  mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape =
      mitsuba::PluginManager::instance()
          ->template create_object<mitsuba::Shape<Float, Spectrum>>(props);
  shape->set_id(id_str);

  TraversalCallback cb("", nullptr, /*recurse_objects=*/false);
  shape->traverse(&cb);

  using FloatStorage = mitsuba::DynamicBuffer<dr::float32_array_t<Float>>;
  using IntStorage = mitsuba::DynamicBuffer<dr::uint32_array_t<Float>>;
  using ScalarSize = typename mitsuba::Shape<Float, Spectrum>::ScalarSize;

  using Point3f = mitsuba::Point<float, 3>;
  using Vector3f = mitsuba::Vector<float, 3>;
  const size_t num_floats = spec.control_points.size();
  cb.set<ScalarSize>("control_point_count", num_floats / 4);

  if (spec.transform == ScalarAffineTransform4f()) {
    cb.set<FloatStorage>(
        "control_points",
        dr::load<FloatStorage>(spec.control_points.data(), num_floats));
  } else {
    const float world_scale =
        (dr::norm(spec.transform * Vector3f(1.f, 0.f, 0.f)) +
         dr::norm(spec.transform * Vector3f(0.f, 1.f, 0.f)) +
         dr::norm(spec.transform * Vector3f(0.f, 0.f, 1.f))) /
        3.f;
    std::vector<float> world_control_points(num_floats);
    const float* src = spec.control_points.data();
    float* dst = world_control_points.data();
    for (size_t i = 0; i < num_floats; i += 4) {
      Point3f p = spec.transform * Point3f(src[i], src[i + 1], src[i + 2]);
      dst[i + 0] = p[0];
      dst[i + 1] = p[1];
      dst[i + 2] = p[2];
      dst[i + 3] = src[i + 3] * world_scale;
    }
    cb.set<FloatStorage>(
        "control_points",
        dr::load<FloatStorage>(world_control_points.data(), num_floats));
  }
  cb.set<IntStorage>("segment_indices",
                     dr::load<IntStorage>(spec.segment_indices.data(),
                                          spec.segment_indices.size()));
  shape->parameters_changed();
  return shape;
}

namespace {

// The 'ellipsoids' plugin keeps each particle as one interleaved record of
// centre (3), scale (3) and quaternion (4) floats, and its "data" property
// accepts exactly that layout. Handing it pre-interleaved data skips the
// gather/scatter pass it would otherwise run to assemble the same buffer from
// separate "centers", "scales" and "quaternions" tensors.
constexpr size_t kEllipsoidStride = 10;

bool ParticleFieldSizesValid(const ParticleFieldSpec& spec,
                             const std::string& id_str) {
  const size_t count = spec.points.size();
  if (spec.scales.size() == count && spec.orientations.size() == count &&
      spec.opacities.size() == count) {
    return true;
  }
  TF_WARN(
      "Particle field %s has mismatched primvar sizes (positions: %zu, "
      "scales: %zu, orientations: %zu, opacities: %zu), skipping.",
      id_str.c_str(), count, spec.scales.size(), spec.orientations.size(),
      spec.opacities.size());
  return false;
}

// Splits the prim transform into the uniform scale and rigid rotation that can
// be folded into per-particle scales and orientations. Centres are transformed
// by the full matrix, so only the scale/rotation part needs decomposing.
void DecomposeParticleFieldTransform(const GfMatrix4d& transform,
                                     const std::string& id_str,
                                     GfQuatf* rotation, float* scale_factor) {
  GfMatrix4d r_mat, u_mat, p_mat;
  GfVec3d scale_vec, trans_vec;
  *rotation = GfQuatf(1.0f);
  *scale_factor = 1.0f;

  if (!transform.Factor(&r_mat, &scale_vec, &u_mat, &trans_vec, &p_mat)) {
    TF_WARN(
        "Singular transform matrix for particle field %s, ignoring transform "
        "scale/rotation",
        id_str.c_str());
    return;
  }

  *rotation = GfQuatf(u_mat.ExtractRotation().GetQuat());
  *scale_factor = scale_vec[0];
  const double scale_eps = 1e-3 * std::abs(scale_vec[0]);
  if (std::abs(scale_vec[1] - scale_vec[0]) > scale_eps ||
      std::abs(scale_vec[2] - scale_vec[0]) > scale_eps) {
    TF_WARN(
        "Non-uniform transform scale (%f, %f, %f) on particle field %s is not "
        "supported; applying x-axis scale uniformly.",
        scale_vec[0], scale_vec[1], scale_vec[2], id_str.c_str());
  }
}

// Bakes the prim transform into the interleaved per-particle records. The
// plugin rejects "to_world" and "scale_factor" unless it is loading a PLY
// file, so the transform has to be applied on this side.
template <typename Storage>
Storage PackParticleField(const ParticleFieldSpec& spec,
                          const std::string& id_str) {
  GfQuatf rotation;
  float scale_factor = 1.0f;
  DecomposeParticleFieldTransform(spec.transform, id_str, &rotation,
                                  &scale_factor);

  const size_t count = spec.points.size();
  std::vector<float> packed(count * kEllipsoidStride);
  for (size_t i = 0; i < count; ++i) {
    float* record = packed.data() + i * kEllipsoidStride;

    const GfVec3f center(spec.transform.Transform(spec.points[i]));
    record[0] = center[0];
    record[1] = center[1];
    record[2] = center[2];

    record[3] = spec.scales[i][0] * scale_factor;
    record[4] = spec.scales[i][1] * scale_factor;
    record[5] = spec.scales[i][2] * scale_factor;

    // USD does not guarantee unit-length orientations, and the plugin only
    // normalizes on its PLY path - a non-unit quaternion here would shear the
    // ellipsoid rather than just rotate it.
    const GfQuatf quat = (rotation * spec.orientations[i]).GetNormalized();
    const GfVec3f& imaginary = quat.GetImaginary();
    record[6] = imaginary[0];
    record[7] = imaginary[1];
    record[8] = imaginary[2];
    record[9] = quat.GetReal();
  }
  return dr::load<Storage>(packed.data(), packed.size());
}

// SH coefficients in the plugin's flat (count, num_coeffs * 3) layout. GfVec3f
// is three packed floats, so the prim's array is already that layout and loads
// as-is. Without coefficients, or with a size that disagrees with the declared
// degree, fall back to a uniform white DC term.
template <typename Storage>
Storage LoadShCoefficients(const ParticleFieldSpec& spec) {
  const size_t count = spec.points.size();
  if (!spec.sh_coeffs.empty()) {
    const size_t num_coeffs = (spec.sh_degree + 1) * (spec.sh_degree + 1);
    if (spec.sh_coeffs.size() == count * num_coeffs) {
      return dr::load<Storage>(spec.sh_coeffs.cdata(),
                               count * num_coeffs * 3);
    }
    TF_WARN(
        "SH coefficients size mismatch: %zu vs expected %zu. Defaulting to "
        "uniform white DC color.",
        spec.sh_coeffs.size(), count * num_coeffs);
  }
  return dr::full<Storage>(1.0f, count * 3);
}

// Minimum opacity threshold when adaptive extent clamping is active.
// In Mitsuba's EllipsoidsData::compute_extents(), extents are scaled by
// sqrt(2 * log(opacity / 0.01)). Values <= 0.01 produce 0 or NaN extents, which
// create degenerate or NaN triangles in the BVH. Clamping >= 0.0101 ensures a
// positive minimum extent (~0.141 sigma) while shrinking low-opacity shells.
constexpr float kMinAdaptiveClampingOpacity = 0.0101f;

template <typename Storage>
Storage LoadOpacities(const ParticleFieldSpec& spec) {
  const size_t count = spec.opacities.size();
  Storage loaded = dr::load<Storage>(spec.opacities.cdata(), count);
  return dr::maximum(std::move(loaded), kMinAdaptiveClampingOpacity);
}

}  // namespace

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildParticleField(
    const ParticleFieldSpec& spec) {
  const std::string id_str = spec.id.GetAsString();
  const size_t count = spec.points.size();
  if (count == 0 || !ParticleFieldSizesValid(spec, id_str)) {
    return nullptr;
  }

  using FloatStorage = mitsuba::DynamicBuffer<dr::float32_array_t<Float>>;
  using TensorXf32 = dr::Tensor<FloatStorage>;

  mitsuba::Properties props("ellipsoidsmesh");
  props.set("extent_adaptive_clamping", true);
  props.set("data", mitsuba::Any(TensorXf32(
                        PackParticleField<FloatStorage>(spec, id_str),
                        {count, kEllipsoidStride})));
  props.set("opacities",
            mitsuba::Any(TensorXf32(LoadOpacities<FloatStorage>(spec),
                                    {count, 1})));

  FloatStorage sh_coeffs = LoadShCoefficients<FloatStorage>(spec);
  const size_t sh_width = dr::width(sh_coeffs) / count;
  props.set("sh_coeffs", mitsuba::Any(TensorXf32(std::move(sh_coeffs),
                                                 {count, sh_width})));

  mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape =
      mitsuba::PluginManager::instance()
          ->template create_object<mitsuba::Shape<Float, Spectrum>>(props);
  shape->set_id(id_str);

  return shape;
}

// Pushes new particle data into an existing shape instead of re-creating it.
// Pure SH edits stay cheap without rebuilding the proxy mesh, while opacity
// edits trigger a proxy mesh recomputation when adaptive clamping is active.
MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateParticleFieldInPlace(
    mitsuba::Object* shape_obj, const ParticleFieldSpec& spec) {
  using FloatStorage = mitsuba::DynamicBuffer<dr::float32_array_t<Float>>;

  auto* shape = dynamic_cast<mitsuba::Shape<Float, Spectrum>*>(shape_obj);
  const std::string id_str = spec.id.GetAsString();
  if (!TF_VERIFY(shape != nullptr, "Not a shape: %s", id_str.c_str())) {
    return;
  }
  if (spec.points.empty() || !ParticleFieldSizesValid(spec, id_str)) {
    return;
  }

  TraversalCallback cb;
  shape->traverse(&cb);

  std::vector<std::string> changed;
  if (spec.geometry_dirty) {
    cb.set<FloatStorage>("data", PackParticleField<FloatStorage>(spec, id_str));
    changed.emplace_back("data");
  }
  if (spec.opacities_dirty) {
    cb.set<FloatStorage>("opacities", LoadOpacities<FloatStorage>(spec));
    changed.emplace_back("opacities");
    // Mitsuba's EllipsoidsMesh only calls recompute_mesh() when "data" is in
    // `changed`. When adaptive extent clamping is active, proxy shell extents
    // depend on opacities, so include "data" to trigger mesh recomputation.
    if (!spec.geometry_dirty) {
      changed.emplace_back("data");
    }
  }
  if (spec.sh_dirty) {
    cb.set<FloatStorage>("sh_coeffs", LoadShCoefficients<FloatStorage>(spec));
    changed.emplace_back("sh_coeffs");
  }

  if (changed.empty()) {
    return;
  }
  shape->parameters_changed(changed);
}

using mitsuba::Color;
using mitsuba::MuellerMatrix;
using mitsuba::Spectrum;

MI_INSTANTIATE_CLASS(PrimTranslator)

PXR_NAMESPACE_CLOSE_SCOPE
