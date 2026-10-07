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
#include <map>
#include <string>
#include <string_view>
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
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fwd.h>
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
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

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

}  // namespace

SdfPath FindDisplacementTerminal(const HdMaterialNetwork2& network) {
  // An empty `mitsuba:displacement` terminal does not shadow `displacement`.
  auto it = network.terminals.find(TfToken("mitsuba:displacement"));
  if (it == network.terminals.end() || it->second.upstreamNode.IsEmpty()) {
    it = network.terminals.find(HdMaterialTerminalTokens->displacement);
  }
  return it != network.terminals.end() ? it->second.upstreamNode : SdfPath();
}

std::string MeshAttributeName(std::string_view primvar, int channels) {
  return absl::StrCat(kVertexAttributePrefix,
                      absl::StrReplaceAll(primvar, {{"color", "Color"}}), "_",
                      channels);
}

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const VtValue& val,
                                 bool invert_float) {
  // Color space metadata is consumed when building texture keys.
  if (absl::StartsWith(name, "colorSpace:")) return;
  if (val.IsHolding<GfVec3f>()) {
    auto c = val.Get<GfVec3f>();
    props.set(name, mitsuba::Color<float, 3>(c[0], c[1], c[2]));
  } else if (val.IsHolding<GfVec4f>()) {
    // Mitsuba never uses alpha channels, so we ignore it.
    auto c = val.Get<GfVec4f>();
    props.set(name, mitsuba::Color<float, 3>(c[0], c[1], c[2]));
  } else if (val.IsHolding<GfVec3d>()) {
    auto c = val.Get<GfVec3d>();
    props.set(name, mitsuba::Color<float, 3>(static_cast<float>(c[0]),
                                             static_cast<float>(c[1]),
                                             static_cast<float>(c[2])));
  } else if (val.IsHolding<float>()) {
    props.set(name, invert_float ? 1.0f - val.Get<float>() : val.Get<float>());
  } else if (val.IsHolding<double>()) {
    props.set(name, invert_float ? 1.0f - static_cast<float>(val.Get<double>())
                                 : static_cast<float>(val.Get<double>()));
  } else if (val.IsHolding<bool>()) {
    props.set(name, val.Get<bool>());
  } else if (val.IsHolding<int>()) {
    props.set(name, val.Get<int>());
  } else if (auto t3 = ExtractTransform3f(val)) {
    props.set(name, *t3);
  } else if (auto t4 = ExtractTransform4f(val)) {
    props.set(name, *t4);
  } else if (val.IsHolding<TfToken>()) {
    props.set(name, val.Get<TfToken>().GetString());
  } else if (val.IsHolding<std::string>()) {
    props.set(name, val.Get<std::string>());
  } else if (val.IsHolding<SdfAssetPath>()) {
    props.set(name, ResolvePathFromValue(val));
  }
}

void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey& tex_key)>& callback) {
  absl::flat_hash_set<SdfPath, SdfPath::Hash> visited_nodes;
  std::function<void(const SdfPath&, const TfToken&)> visit =
      [&](const SdfPath& node_path, const TfToken& input_name) {
        auto it = network.nodes.find(node_path);
        if (it == network.nodes.end()) {
          return;
        }
        const auto& node = it->second;
        if (IsTextureNode(node)) {
          // Texture nodes are keyed by input_name as well, so they are
          // intentionally not deduplicated via visited_nodes.
          if (auto key_opt = ExtractTextureKey(
                  node.parameters, node.nodeTypeId, input_name)) {
            callback(*key_opt);
          }
          return;
        }
        if (!visited_nodes.insert(node_path).second) {
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
          visit(connections[0].upstreamNode, conn_input);
        }
      };

  if (SdfPath disp_node = FindDisplacementTerminal(network);
      !disp_node.IsEmpty()) {
    visit(disp_node, HdMaterialTerminalTokens->displacement);
  }
  auto surf_it = network.terminals.find(HdMaterialTerminalTokens->surface);
  if (surf_it != network.terminals.end() &&
      !surf_it->second.upstreamNode.IsEmpty()) {
    visit(surf_it->second.upstreamNode, HdMaterialTerminalTokens->surface);
  }
}

namespace {

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> GetTexture(
    const std::map<TfToken, VtValue>& parameters, const TfToken& nodeTypeId,
    const TfToken& input_name,
    const TextureCache<Float, Spectrum>& texture_cache) {
  const bool is_normal =
      (input_name == TfToken("normal") || input_name == TfToken("normalmap"));
  // Attempt to find the texture from the texture cache. This
  // assumes the texture cache was pre-populated by DiscoverTextures.
  if (auto key_opt =
          ExtractTextureKey(parameters, nodeTypeId, input_name)) {
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
  mitsuba::Properties fallback_props("srgb");
  auto fallback_it = parameters.find(TfToken("fallback"));
  if (fallback_it != parameters.end()) {
    SetMitsubaPropertyFromValue(fallback_props, "color", fallback_it->second);
  } else {
    if (is_normal) {
      fallback_props.set("color", mitsuba::Color<float, 3>(0.5f, 0.5f, 1.0f));
    } else {
      fallback_props.set("color", mitsuba::Color<float, 3>(1.0f, 0.0f, 1.0f));
    }
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
  auto node_it = network2.nodes.find(conn_it->second[0].upstreamNode);
  if (node_it == network2.nodes.end()) {
    return nullptr;
  }
  const HdMaterialNode2& upstream_node = node_it->second;
  if (IsTextureNode(upstream_node)) {
    return GetTexture<Float, Spectrum>(upstream_node.parameters,
                                       upstream_node.nodeTypeId, input_name,
                                       texture_cache);
  }
  return nullptr;
}

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> ParseNodeRecursive(
    const SdfPath& nodePath, const HdMaterialNetwork2& network,
    absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>&
        cache,
    TfToken input_name, const TextureCache<Float, Spectrum>& texture_cache) {
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
    return GetTexture<Float, Spectrum>(node.parameters, node.nodeTypeId,
                                       input_name, texture_cache);
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
        ParseNodeRecursive<Float, Spectrum>(connections[0].upstreamNode,
                                            network, cache, input_token,
                                            texture_cache);
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
PrimTranslator<Float, Spectrum>::DefaultBsdf(std::string_view id_str) {
  mitsuba::Properties props("diffuse");
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

  if (SdfPath disp_node = FindDisplacementTerminal(network2);
      !disp_node.IsEmpty()) {
    absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>
        cache;
    res.displacement.texture = ParseNodeRecursive<Float, Spectrum>(
        disp_node, network2, cache, HdMaterialTerminalTokens->displacement,
        texture_cache);
  }

  auto terminal_it = network2.terminals.find(HdMaterialTerminalTokens->surface);
  if (terminal_it == network2.terminals.end()) {
    res.bsdf = DefaultBsdf(id_str);
    return res;
  }
  const SdfPath& surface_node_path = terminal_it->second.upstreamNode;
  auto node_it = network2.nodes.find(surface_node_path);
  if (node_it == network2.nodes.end()) {
    res.bsdf = DefaultBsdf(id_str);
    return res;
  }
  const HdMaterialNode2& surface_node = node_it->second;

  if (surface_node.nodeTypeId == TfToken("UsdPreviewSurface")) {
    auto preview_res =
        ParsePreviewSurface(network2, surface_node, texture_cache);
    preview_res.bsdf->set_id(id_str);
    preview_res.displacement = res.displacement;
    return preview_res;
  } else {
    absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>
        cache;
    res.bsdf = ParseNodeRecursive<Float, Spectrum>(
        surface_node_path, network2, cache, HdMaterialTerminalTokens->surface,
        texture_cache);
    if (res.bsdf) {
      res.bsdf->set_id(id_str);
    }
  }
  if (!res.bsdf) {
    res.bsdf = DefaultBsdf(id_str);
  }
  return res;
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
    res.emitter = mitsuba::PluginManager::instance()
                      ->create_object<mitsuba::Emitter<Float, Spectrum>>(props);
    res.emitter->set_id(id_str);
  }
  return res;
}

MI_VARIANT mitsuba::ref<mitsuba::Object>
PrimTranslator<Float, Spectrum>::CreateAreaEmitter(const GfVec3f& emission) {
  mitsuba::Properties emitter_props("area");
  emitter_props.set(
      "radiance",
      mitsuba::Color<float, 3>(emission[0], emission[1], emission[2]));
  return mitsuba::PluginManager::instance()->create_object(
      emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
      mitsuba::Emitter<Float, Spectrum>::Type);
}

MI_VARIANT mitsuba::Properties
PrimTranslator<Float, Spectrum>::BuildLightProperties(const LightSpec& spec) {
  const std::string id_str = spec.id.GetAsString();
  ScalarAffineTransform4f to_world = spec.transform;
  mitsuba::Color<float, 3> color(spec.emission[0], spec.emission[1],
                                 spec.emission[2]);
  auto make_area_shape_props = [&](std::string_view shape_plugin) {
    mitsuba::Properties props(shape_plugin);
    props.set("to_world", to_world);
    props.set("emitter", CreateAreaEmitter(spec.emission));
    return props;
  };

  if (spec.prim_type == HdPrimTypeTokens->sphereLight) {
    if (spec.treat_as_point) {
      if (spec.shaping_cone_angle != 0.0f) {
        mitsuba::Properties spot_props("spot");
        spot_props.set("to_world", to_world);
        spot_props.set("intensity", color);
        spot_props.set("beam_width", spec.shaping_cone_beam_width);
        spot_props.set("cutoff_angle", spec.shaping_cone_angle);
        return spot_props;
      } else {
        mitsuba::Properties point_props("point");
        point_props.set("to_world", to_world);
        point_props.set("intensity", color);
        return point_props;
      }
    } else {
      return make_area_shape_props("sphere");
    }
  }

  if (spec.prim_type == HdPrimTypeTokens->domeLight) {
    if (!spec.texture_file_path.empty()) {
      try {
        mitsuba::ref<mitsuba::Bitmap> bitmap =
            ReadBitmap(spec.texture_file_path);
        mitsuba::Properties props("envmap");
        props.set("to_world", to_world);
        props.set("scale", (color[0] + color[1] + color[2]) / 3.f);
        props.set("bitmap", mitsuba::ref<mitsuba::Object>(bitmap));
        return props;
      } catch (const std::exception& e) {
        TF_WARN("Failed to load environment texture '%s': %s",
                spec.texture_file_path.c_str(), e.what());
      }
    }
    mitsuba::Properties props("constant");
    props.set("radiance", color);
    return props;
  }

  if (spec.prim_type == HdPrimTypeTokens->distantLight) {
    mitsuba::Properties props("directional");
    props.set("irradiance", color);
    props.set("to_world", to_world);
    props.set("angle", spec.angle);
    return props;
  }

  if (spec.prim_type == HdPrimTypeTokens->rectLight) {
    return make_area_shape_props("rectangle");
  }

  if (spec.prim_type == HdPrimTypeTokens->diskLight) {
    return make_area_shape_props("disk");
  }

  mitsuba::Properties props("constant");
  return props;
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateLightInPlace(
    mitsuba::Object* light_obj, const LightSpec& spec) {
  using AffineTransform4f = mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
  using Color3f = mitsuba::Color<Float, 3>;

  ScalarAffineTransform4f to_world = spec.transform;
  mitsuba::Color<float, 3> color(spec.emission[0], spec.emission[1],
                                 spec.emission[2]);
  if (auto* shape = dynamic_cast<mitsuba::Shape<Float, Spectrum>*>(light_obj)) {
    TraversalCallback cb_shape;
    shape->traverse(&cb_shape);
    cb_shape.set<AffineTransform4f>("to_world",
                                    AffineTransform4f(to_world.matrix));
    shape->parameters_changed();
    if (shape->is_emitter()) {
      auto* area_emitter = shape->emitter();
      TraversalCallback cb_emitter;
      area_emitter->traverse(&cb_emitter);
      cb_emitter.set<Color3f>("radiance.value",
                              Color3f(color[0], color[1], color[2]));
      area_emitter->parameters_changed();
    }
  } else if (auto* emitter =
                 dynamic_cast<mitsuba::Emitter<Float, Spectrum>*>(light_obj)) {
    TraversalCallback cb;
    emitter->traverse(&cb);

    if (spec.prim_type == HdPrimTypeTokens->sphereLight &&
        spec.treat_as_point) {
      cb.set<AffineTransform4f>("to_world",
                                AffineTransform4f(to_world.matrix));
      cb.set<Color3f>("intensity.value", Color3f(color[0], color[1], color[2]));
      if (spec.shaping_cone_angle != 0.0f) {
        cb.set<Float>("beam_width", spec.shaping_cone_beam_width);
        cb.set<Float>("cutoff_angle", spec.shaping_cone_angle);
      }
    } else if (spec.prim_type == HdPrimTypeTokens->domeLight) {
      // A domeLight with a missing/corrupt texture falls back to a "constant"
      // emitter in BuildLightProperties, which has "radiance.value" rather than
      // "scale" and "to_world".
      if (cb.data.contains("scale")) {
        cb.set<AffineTransform4f>("to_world",
                                  AffineTransform4f(to_world.matrix));
        cb.set<Float>("scale", (color[0] + color[1] + color[2]) / 3.f);
        emitter->parameters_changed({"scale", "to_world"});
        return;
      } else {
        cb.set<Color3f>("radiance.value",
                        Color3f(color[0], color[1], color[2]));
      }
    } else if (spec.prim_type == HdPrimTypeTokens->distantLight) {
      cb.set<AffineTransform4f>("to_world", AffineTransform4f(to_world.matrix));
      cb.set<Color3f>("irradiance.value",
                      Color3f(color[0], color[1], color[2]));
      cb.set<Float>("angle", spec.angle);
    }
    emitter->parameters_changed();
  }
}

MI_VARIANT mitsuba::ref<mitsuba::Sensor<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildSensor(const CameraSpec& spec,
                                             bool is_interactive) {
  mitsuba::Properties props;
  if (spec.sensor_type == "irradiancemeter") {
    props = mitsuba::Properties("irradiancemeter");
  } else {
    props = mitsuba::Properties("perspective");
    props.set("to_world", spec.transform);
    props.set("fov", spec.fov);
    props.set("fov_axis", "x");
    props.set("principal_point_offset_x", spec.horizontal_aperture_offset);
    props.set("principal_point_offset_y", spec.vertical_aperture_offset);
  }
  props.set("near_clip", spec.near_clip);
  props.set("far_clip", spec.far_clip);

  std::string filter_type = is_interactive ? "box" : spec.pixel_filter_type;
  if (!filter_type.empty()) {
    mitsuba::Properties film_props("hdrfilm");
    film_props.set(
        "pixel_filter",
        static_cast<mitsuba::Object*>(
            mitsuba::PluginManager::instance()
                ->create_object<mitsuba::ReconstructionFilter<Float, Spectrum>>(
                    mitsuba::Properties(filter_type))
                .get()));
    props.set(
        "film",
        static_cast<mitsuba::Object*>(
            mitsuba::PluginManager::instance()
                ->create_object<mitsuba::Film<Float, Spectrum>>(film_props)
                .get()));
  }
  mitsuba::ref<mitsuba::Sensor<Float, Spectrum>> sensor =
      mitsuba::PluginManager::instance()
          ->create_object<mitsuba::Sensor<Float, Spectrum>>(props);
  sensor->set_id(spec.id.GetAsString());
  return sensor;
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateSensorInPlace(
    mitsuba::Object* sensor_obj, const CameraSpec& spec) {
  using AffineTransform4f = mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
  using ScalarFloat = float;
  auto* sensor = dynamic_cast<mitsuba::Sensor<Float, Spectrum>*>(sensor_obj);
  if (!sensor) return;

  TraversalCallback cb;
  sensor->traverse(&cb);

  if (spec.dirty_bits & HdCamera::DirtyBits::DirtyTransform) {
    cb.set<AffineTransform4f>(
        "to_world",
        AffineTransform4f(spec.transform.matrix));
  }
  if (spec.dirty_bits & HdCamera::DirtyBits::DirtyParams) {
    cb.set<ScalarFloat>("near_clip", spec.near_clip);
    cb.set<ScalarFloat>("far_clip", spec.far_clip);
    if (spec.sensor_type == "perspective" || spec.sensor_type.empty()) {
      cb.set<Float>("x_fov", spec.fov);
      cb.set<Float>("principal_point_offset_x",
                    spec.horizontal_aperture_offset);
      cb.set<Float>("principal_point_offset_y", spec.vertical_aperture_offset);
    }
  }
  sensor->parameters_changed();
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

// Adds the primvars in `mesh_attributes` that exist in `primvars` to `mesh` as
// vertex attributes with the requested channel count. Float primvars are
// broadcast, and vec2 primvars are padded with zeros. Texture coordinates were
// flipped to Mitsuba's convention by `TransformPrimvars` and are flipped back
// to USD's convention.
template <typename Mesh>
void AddMeshAttributes(Mesh* mesh, const PrimvarMap& primvars,
                       const MeshAttributeRequests& mesh_attributes) {
  const size_t vertex_count = mesh->vertex_count();
  std::vector<float> data;
  for (const auto& [name, channels] : mesh_attributes) {
    auto it = primvars.find(TfToken(name));
    if (it == primvars.end()) continue;
    const VtValue& value = it->second.value;
    if (value.GetArraySize() != vertex_count) {
      TF_WARN("Primvar '%s' has %zu values, but the mesh has %zu vertices.",
              name.c_str(), value.GetArraySize(), vertex_count);
      continue;
    }
    data.assign(vertex_count * channels, 0.f);
    if (value.IsHolding<VtFloatArray>()) {
      const auto& a = value.UncheckedGet<VtFloatArray>();
      for (size_t i = 0; i < vertex_count; ++i) {
        std::fill_n(&data[i * channels], channels, a[i]);
      }
    } else if (value.IsHolding<VtVec2fArray>()) {
      const auto& a = value.UncheckedGet<VtVec2fArray>();
      const bool flip =
          it->second.descriptor.role == HdPrimvarRoleTokens->textureCoordinate;
      for (size_t i = 0; i < vertex_count; ++i) {
        data[i * channels] = a[i][0];
        if (channels > 1) {
          data[i * channels + 1] = flip ? 1.f - a[i][1] : a[i][1];
        }
      }
    } else if (value.IsHolding<VtVec3fArray>()) {
      const auto& a = value.UncheckedGet<VtVec3fArray>();
      for (size_t i = 0; i < vertex_count; ++i) {
        std::copy_n(a[i].data(), channels, &data[i * channels]);
      }
    } else {
      continue;
    }
    mesh->add_attribute(
        MeshAttributeName(name, channels),
        LoadFloatTensor<Mesh>(data.data(), vertex_count, channels));
  }
}

}  // namespace

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildMesh(
    const SdfPath& id, const VtIntArray& face_indices,
    const PrimvarMap& primvars, const MeshAttributeRequests& mesh_attributes,
    mitsuba::Object* bsdf, mitsuba::Object* emitter_ptr,
    mitsuba::Object* sensor_ptr) {
  using Mesh = mitsuba::Mesh<Float, Spectrum>;
  using TensorXf32 = typename Mesh::TensorXf32;

  const std::string id_str = id.GetAsString();
  auto points_it = primvars.find(HdTokens->points);
  if (points_it == primvars.end()) {
    TF_RUNTIME_ERROR("Mesh %s has no points.", id_str.c_str());
    return nullptr;
  }

  const auto& points_array = points_it->second.value.Get<VtVec3fArray>();
  size_t vertex_count = points_array.size();
  if (vertex_count == 0) return nullptr;

  size_t face_count = face_indices.size() / 3;
  if (face_count == 0) return nullptr;

  auto normals_it = primvars.find(HdTokens->normals);
  const bool has_normals = normals_it != primvars.end();
  mitsuba::Properties props;
  props.set_id(id_str);
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
  mitsuba::ref<Mesh> mesh = new Mesh(props);
  mesh->set_id(id_str);
  if (bsdf != nullptr) {
    mesh->set_bsdf(dynamic_cast<mitsuba::BSDF<Float, Spectrum>*>(bsdf));
  }
  TensorXf32 normals;
  if (has_normals) {
    const auto& normals_array = normals_it->second.value.Get<VtVec3fArray>();
    normals = LoadFloatTensor<Mesh>(normals_array.data(), normals_array.size(),
                                    3);
  }
  TensorXf32 texcoords;
  auto texcoords_it = primvars.find(TfToken("st"));
  if (texcoords_it != primvars.end()) {
    const auto& texcoords_array =
        texcoords_it->second.value.Get<VtVec2fArray>();
    texcoords = LoadFloatTensor<Mesh>(texcoords_array.data(),
                                      texcoords_array.size(), 2);
  }
  mesh->from_fields(LoadFaceTensor<Mesh>(face_indices.data(), face_count),
                    LoadFloatTensor<Mesh>(points_array.data(), vertex_count, 3),
                    normals, texcoords);
  AddMeshAttributes(mesh.get(), primvars, mesh_attributes);
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
  const auto& points_array = points_it->second.value.Get<VtVec3fArray>();
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
      "positions", LoadFloatTensor<Mesh>(points_array.data(), vertex_count, 3));
  if (update_topology_and_uvs) {
    cb.set<TensorXu32>("faces",
                       LoadFaceTensor<Mesh>(face_indices.data(), face_count));
    keys.push_back("faces");
  }

  auto normals_it = primvars.find(HdTokens->normals);
  if (normals_it != primvars.end() && mesh->has_normals()) {
    const auto& normals_array = normals_it->second.value.Get<VtVec3fArray>();
    cb.set<TensorXf32>("normals",
                       LoadFloatTensor<Mesh>(normals_array.data(),
                                             normals_array.size(), 3));
  }
  // Always add "normals" to the keys: This prevents Mitsuba from recomputing
  // the normals. We handle normal recomputation explicitly in the hydra
  // delegate.
  keys.push_back("normals");

  if (update_topology_and_uvs) {
    auto texcoords_it = primvars.find(TfToken("st"));
    if (texcoords_it != primvars.end() && mesh->has_texcoords()) {
      const auto& texcoords_array =
          texcoords_it->second.value.Get<VtVec2fArray>();
      cb.set<TensorXf32>("texcoords",
                         LoadFloatTensor<Mesh>(texcoords_array.data(),
                                               texcoords_array.size(), 2));
      keys.push_back("texcoords");
    }
  }

  // The vertex count may change, so the attributes are removed and re-added.
  for (const auto& [key, value] : cb.data) {
    if (absl::StartsWith(key, kVertexAttributePrefix)) {
      mesh->remove_attribute(key);
    }
  }
  mesh->parameters_changed(keys);
  AddMeshAttributes(mesh, primvars, mesh_attributes);
}

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildCurves(const CurveSpec& spec,
                                             mitsuba::Object* bsdf) {
  const std::string id_str = spec.id.GetAsString();
  mitsuba::Properties props(spec.plugin_name);
  if (auto plugin =
          PlugRegistry::GetInstance().GetPluginWithName("hdMitsuba")) {
    props.set("filename", plugin->GetResourcePath() + "/curve.txt");
  } else {
    TF_RUNTIME_ERROR("Failed to find plugin 'hdMitsuba' to locate resources.");
  }

  mitsuba::ref<mitsuba::Shape<Float, Spectrum>> shape =
      mitsuba::PluginManager::instance()
          ->template create_object<mitsuba::Shape<Float, Spectrum>>(props);
  shape->set_id(id_str);
  if (bsdf != nullptr) {
    shape->set_bsdf(dynamic_cast<mitsuba::BSDF<Float, Spectrum>*>(bsdf));
  }

  TraversalCallback cb("", nullptr, /*recurse_objects=*/false);
  shape->traverse(&cb);

  using FloatStorage = mitsuba::DynamicBuffer<dr::float32_array_t<Float>>;
  using IntStorage = mitsuba::DynamicBuffer<dr::uint32_array_t<Float>>;
  using ScalarSize = typename mitsuba::Shape<Float, Spectrum>::ScalarSize;

  using Point3f = mitsuba::Point<float, 3>;
  using Vector3f = mitsuba::Vector<float, 3>;
  const float world_scale =
      (dr::norm(spec.transform * Vector3f(1.f, 0.f, 0.f)) +
       dr::norm(spec.transform * Vector3f(0.f, 1.f, 0.f)) +
       dr::norm(spec.transform * Vector3f(0.f, 0.f, 1.f))) /
      3.f;
  std::vector<float> world_control_points = spec.control_points;
  for (size_t i = 0; i < world_control_points.size(); i += 4) {
    Point3f p(world_control_points[i], world_control_points[i + 1],
              world_control_points[i + 2]);
    p = spec.transform * p;
    world_control_points[i + 0] = p[0];
    world_control_points[i + 1] = p[1];
    world_control_points[i + 2] = p[2];
    world_control_points[i + 3] *= world_scale;
  }

  cb.set<ScalarSize>("control_point_count", world_control_points.size() / 4);
  cb.set<FloatStorage>("control_points",
                       dr::load<FloatStorage>(world_control_points.data(),
                                              world_control_points.size()));
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
