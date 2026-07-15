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

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_join.h>
#include <absl/strings/strip.h>
#include <drjit/matrix.h>
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
using ScalarVector2u = mitsuba::Vector<uint32_t, 2>;

std::string ResolvePathFromValue(const VtValue& value) {
  if (value.IsHolding<SdfAssetPath>()) {
    const auto& asset_path = value.Get<SdfAssetPath>();
    const std::string& resolved = asset_path.GetResolvedPath();
    return resolved.empty() ? asset_path.GetAssetPath() : resolved;
  } else if (value.IsHolding<std::string>()) {
    return value.Get<std::string>();
  }
  return {};
}

}  // namespace

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

void SetMitsubaPropertyFromValue(mitsuba::Properties& props,
                                 std::string_view name, const VtValue& val,
                                 bool invert_float) {
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
  } else if (val.IsHolding<GfMatrix3f>()) {
    auto m = val.Get<GfMatrix3f>();
    dr::Matrix<float, 3> matrix(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1],
                                m[1][2], m[2][0], m[2][1], m[2][2]);
    props.set(name, mitsuba::Transform<mitsuba::Point<float, 3>, true>(matrix));
  } else if (val.IsHolding<GfMatrix3d>()) {
    auto m = val.Get<GfMatrix3d>();
    dr::Matrix<float, 3> matrix(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1],
                                m[1][2], m[2][0], m[2][1], m[2][2]);
    props.set(name, mitsuba::Transform<mitsuba::Point<float, 3>, true>(matrix));
  } else if (val.IsHolding<GfMatrix4f>()) {
    auto m = val.Get<GfMatrix4f>();
    dr::Matrix<float, 4> matrix(
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
    props.set(name, ScalarAffineTransform4f(matrix));
  } else if (val.IsHolding<GfMatrix4d>()) {
    auto m = val.Get<GfMatrix4d>();
    dr::Matrix<float, 4> matrix(
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
    props.set(name, ScalarAffineTransform4f(matrix));
  } else if (val.IsHolding<std::string>()) {
    props.set(name, val.Get<std::string>());
  } else if (val.IsHolding<TfToken>()) {
    props.set(name, val.Get<TfToken>().GetString());
  }
}

std::optional<mitsuba::Properties> ExtractTextureProperties(
    const std::map<TfToken, VtValue>& parameters, const TfToken& nodeTypeId,
    const TfToken& input_name) {
  std::string resolved_path;
  bool found = false;
  if (nodeTypeId == TfToken("UsdUVTexture")) {
    auto file_it = parameters.find(TfToken("file"));
    if (file_it != parameters.end()) {
      resolved_path = ResolvePathFromValue(file_it->second);
      found = !resolved_path.empty();
    }
  } else if (nodeTypeId == TfToken("mitsuba_bitmap")) {
    auto filename_it = parameters.find(TfToken("filename"));
    if (filename_it != parameters.end()) {
      resolved_path = ResolvePathFromValue(filename_it->second);
      found = !resolved_path.empty();
    }
  }
  if (!found) {
    return std::nullopt;
  }
  mitsuba::Properties props("bitmap");
  props.set("filename", resolved_path);
  props.set("is_normal", (input_name == TfToken("normal") ||
                          input_name == TfToken("normalmap")));
  if (nodeTypeId == TfToken("UsdUVTexture")) {
    auto source_color_space_it = parameters.find(TfToken("sourceColorSpace"));
    TfToken source_color_space = TfToken("auto");
    if (source_color_space_it != parameters.end() &&
        source_color_space_it->second.IsHolding<TfToken>()) {
      source_color_space = source_color_space_it->second.Get<TfToken>();
    }
    bool is_raw = UseRawBitmap(source_color_space, input_name);
    props.set("raw", is_raw);
    auto wrap_s_it = parameters.find(TfToken("wrapS"));
    if (wrap_s_it != parameters.end() &&
        wrap_s_it->second.IsHolding<TfToken>()) {
      TfToken wrap_s = wrap_s_it->second.Get<TfToken>();
      if (wrap_s == TfToken("repeat") || wrap_s == TfToken("clamp") ||
          wrap_s == TfToken("mirror")) {
        props.set("wrap_mode", wrap_s.GetString());
      }
    }
  } else if (nodeTypeId == TfToken("mitsuba_bitmap")) {
    for (const auto& [param_token, param_value] : parameters) {
      std::string name = param_token.GetString();
      if (name != "filename") {
        SetMitsubaPropertyFromValue(props, name, param_value);
      }
    }
  }
  return props;
}

namespace {

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> GetTexture(
    const std::map<TfToken, VtValue>& parameters, const TfToken& nodeTypeId,
    const TfToken& input_name, const TextureCache& texture_cache) {
  // Attempt to find the texture from the texture cache. This
  // assumes the texture cache was pre-populated.
  if (auto props_opt =
          ExtractTextureProperties(parameters, nodeTypeId, input_name)) {
    auto cache_it = texture_cache.find(*props_opt);
    if (cache_it != texture_cache.end()) {
      return cache_it->second;
    }
  }
  // Instantiate a fallback texture.
  mitsuba::Properties fallback_props("srgb");
  auto fallback_it = parameters.find(TfToken("fallback"));
  if (fallback_it != parameters.end()) {
    SetMitsubaPropertyFromValue(fallback_props, "color", fallback_it->second);
  } else {
    if (input_name == TfToken("normal") || input_name == TfToken("normalmap")) {
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
    const TfToken& input_name, const TextureCache& texture_cache) {
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
  const auto& type_id = upstream_node.nodeTypeId;
  static const TfToken usd_uv_texture("UsdUVTexture");
  static const TfToken mitsuba_bitmap("mitsuba_bitmap");
  if (type_id == usd_uv_texture || type_id == mitsuba_bitmap) {
    return GetTexture<Float, Spectrum>(upstream_node.parameters, type_id,
                                       input_name, texture_cache);
  }
  return nullptr;
}

template <typename Float, typename Spectrum>
mitsuba::ref<mitsuba::Object> ParseNodeRecursive(
    const SdfPath& nodePath, const HdMaterialNetwork2& network,
    absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>&
        cache,
    TfToken input_name, const TextureCache& texture_cache) {
  if (auto it = cache.find(nodePath); it != cache.end()) {
    return it->second;
  }

  auto it = network.nodes.find(nodePath);
  if (it == network.nodes.end()) {
    TF_RUNTIME_ERROR("Node %s not found in material network.",
                     nodePath.GetText());
    return nullptr;
  }

  // Specifically handle textures due to caching.
  const auto& node = it->second;
  bool is_texture = (node.nodeTypeId == TfToken("UsdUVTexture") ||
                     node.nodeTypeId == TfToken("mitsuba_bitmap"));
  if (is_texture) {
    mitsuba::ref<mitsuba::Object> texture = GetTexture<Float, Spectrum>(
        node.parameters, node.nodeTypeId, input_name, texture_cache);
    cache[nodePath] = texture;
    return texture;
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
    mitsuba::ref<mitsuba::Object> upstream_object =
        ParseNodeRecursive<Float, Spectrum>(connections[0].upstreamNode,
                                            network, cache, input_token,
                                            texture_cache);
    if (upstream_object) {
      bool is_scalar_only =
          (mitsuba_plugin_name == "principled") &&
          (input_token == TfToken("specular") || input_token == TfToken("eta"));
      if (is_scalar_only) {
        TF_WARN(
            "Mitsuba's principled BSDF does not support texture connection for "
            "'%s'. Ignoring connection.",
            input_token.GetText());
      } else {
        props.set(input_token.GetString(), upstream_object);
      }
    }
  }

  mitsuba::ObjectType object_type =
      mitsuba::PluginManager::instance()->plugin_type(mitsuba_plugin_name);
  mitsuba::ref<mitsuba::Object> result =
      mitsuba::PluginManager::instance()->create_object(
          props, mitsuba::Scene<Float, Spectrum>::Variant, object_type);
  std::vector<mitsuba::ref<mitsuba::Object>> result_objects = result->expand();
  if (!result_objects.empty()) {
    result = result_objects[0];
  }
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

MI_VARIANT typename PrimTranslator<Float, Spectrum>::TranslatedMaterial
PrimTranslator<Float, Spectrum>::ParsePreviewSurface(
    const HdMaterialNetwork2& network2,
    const HdMaterialNode2& preview_surface_node,
    const TextureCache& texture_cache) {
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
    bool is_scalar_only =
        (mapping.mitsuba_name == "specular") || (mapping.mitsuba_name == "eta");
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

MI_VARIANT mitsuba::ref<mitsuba::Object>
PrimTranslator<Float, Spectrum>::LoadTexture(const mitsuba::Properties& props) {
  std::string path = props.get<std::string>("filename");
  bool is_normal = false;
  if (props.has_property("is_normal")) {
    is_normal = props.get<bool>("is_normal");
  }
  mitsuba::Properties local_props = props;
  if (local_props.has_property("is_normal")) {
    local_props.remove_property("is_normal");
  }
  try {
    mitsuba::ref<mitsuba::Bitmap> bmp = new mitsuba::Bitmap(path);
    if (is_normal) {
      if (bmp->channel_count() < 3) {
        TF_WARN(
            "Normal map texture '%s' has only %d channels! Normal maps require "
            "at least 3 channels.",
            path.c_str(), (int)bmp->channel_count());
        return nullptr;
      }
    }
    local_props.set("bitmap", mitsuba::ref<mitsuba::Object>(bmp));
    if (local_props.has_property("filename")) {
      local_props.remove_property("filename");
    }
    mitsuba::ref<mitsuba::Object> texture =
        mitsuba::PluginManager::instance()->create_object(
            local_props, mitsuba::Texture<Float, Spectrum>::Variant,
            mitsuba::Texture<Float, Spectrum>::Type);
    std::vector<mitsuba::ref<mitsuba::Object>> expanded_texture =
        texture->expand();
    if (!expanded_texture.empty()) {
      texture = expanded_texture[0];
    }
    return texture;
  } catch (const std::exception& e) {
    TF_WARN("Failed to load texture '%s': %s.", path.c_str(), e.what());
    return nullptr;
  }
}

MI_VARIANT typename PrimTranslator<Float, Spectrum>::TranslatedMaterial
PrimTranslator<Float, Spectrum>::BuildMaterial(
    const MaterialSpec& spec, const TextureCache& texture_cache) {
  TranslatedMaterial res;
  const HdMaterialNetwork2& network2 = spec.network2;
  const std::string id_str = spec.id.GetAsString();

  auto disp_it = network2.terminals.find(TfToken("mitsuba:displacement"));
  if (disp_it == network2.terminals.end()) {
    disp_it = network2.terminals.find(HdMaterialTerminalTokens->displacement);
  }
  if (disp_it != network2.terminals.end()) {
    auto terminal_connection = disp_it->second;
    if (!network2.nodes.empty() &&
        !terminal_connection.upstreamNode.IsEmpty()) {
      absl::flat_hash_map<SdfPath, mitsuba::ref<mitsuba::Object>, SdfPath::Hash>
          cache;
      res.displacement_texture = ParseNodeRecursive<Float, Spectrum>(
          terminal_connection.upstreamNode, network2, cache,
          HdMaterialTerminalTokens->displacement, texture_cache);
    }
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
    preview_res.displacement_texture = res.displacement_texture;
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

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateMaterialInPlace(
    mitsuba::Object* /*bsdf*/, const MaterialSpec& /*spec*/,
    const TextureCache& /*texture_cache*/) {}

MI_VARIANT typename PrimTranslator<Float, Spectrum>::TranslatedLight
PrimTranslator<Float, Spectrum>::BuildLight(const LightSpec& spec) {
  mitsuba::Properties props = BuildLightProperties(spec);
  std::string id_str = spec.id.GetAsString();
  std::string_view plugin_name = props.plugin_name();
  bool is_shape = plugin_name == "rectangle" || plugin_name == "disk" ||
                  plugin_name == "sphere";
  TranslatedLight res;
  if (is_shape) {
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

MI_VARIANT mitsuba::Properties
PrimTranslator<Float, Spectrum>::BuildLightProperties(const LightSpec& spec) {
  const std::string id_str = spec.id.GetAsString();
  ScalarAffineTransform4f to_world = spec.transform;
  mitsuba::Color<float, 3> color(spec.emission[0], spec.emission[1],
                                 spec.emission[2]);

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
      mitsuba::Properties sphere_props("sphere");
      sphere_props.set("to_world", to_world);
      mitsuba::Properties emitter_props("area");
      emitter_props.set("radiance", color);
      mitsuba::ref<mitsuba::Object> area_emitter =
          mitsuba::PluginManager::instance()->create_object(
              emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
              mitsuba::Emitter<Float, Spectrum>::Type);
      sphere_props.set("emitter", area_emitter);
      return sphere_props;
    }
  }

  if (spec.prim_type == HdPrimTypeTokens->domeLight) {
    if (!spec.texture_file_path.empty()) {
      try {
        mitsuba::ref<mitsuba::Bitmap> bitmap =
            new mitsuba::Bitmap(spec.texture_file_path);
        mitsuba::Properties props("envmap");
        props.set("to_world", to_world);
        props.set("scale", (color[0] + color[1] + color[2]) / 3.f);
        // If needed, resample to Mitsuba's minimum envmap size.
        if (bitmap->width() < 2 || bitmap->height() < 3) {
          uint32_t target_w = std::max(2u, bitmap->width());
          uint32_t target_h = std::max(3u, bitmap->height());
          TF_DEBUG(HDMITSUBA_SYNC)
              .Msg(
                  "Resampling environment map '%s' from %ux%u to %ux%u due to "
                  "Mitsuba size limits\n",
                  spec.texture_file_path.c_str(), bitmap->width(),
                  bitmap->height(), target_w, target_h);
          bitmap = bitmap->convert(mitsuba::Bitmap::PixelFormat::RGB,
                                   mitsuba::struct_type_v<float>, false);
          bitmap = bitmap->resample(ScalarVector2u(target_w, target_h));
        }
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
    return props;
  }

  if (spec.prim_type == HdPrimTypeTokens->rectLight) {
    mitsuba::Properties props("rectangle");
    mitsuba::Properties emitter_props("area");
    emitter_props.set("radiance", color);
    mitsuba::ref<mitsuba::Object> area_emitter =
        mitsuba::PluginManager::instance()->create_object(
            emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
            mitsuba::Emitter<Float, Spectrum>::Type);
    props.set("emitter", area_emitter);
    props.set("to_world", to_world);
    return props;
  }

  if (spec.prim_type == HdPrimTypeTokens->diskLight) {
    mitsuba::Properties props("disk");
    mitsuba::Properties emitter_props("area");
    emitter_props.set("radiance", color);
    mitsuba::ref<mitsuba::Object> area_emitter =
        mitsuba::PluginManager::instance()->create_object(
            emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
            mitsuba::Emitter<Float, Spectrum>::Type);
    props.set("emitter", area_emitter);
    props.set("to_world", to_world);
    return props;
  }

  mitsuba::Properties props("constant");
  return props;
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateLightInPlace(
    mitsuba::Object* light_obj, const LightSpec& spec) {
  using AffineTransform4f = mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
  using Color3f = mitsuba::Color<Float, 3>;
  using Point3f = mitsuba::Point<Float, 3>;

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
      if (spec.shaping_cone_angle != 0.0f) {
        cb.set<AffineTransform4f>("to_world",
                                  AffineTransform4f(to_world.matrix));
        cb.set<Color3f>("intensity.value",
                        Color3f(color[0], color[1], color[2]));
        cb.set<Float>("beam_width", spec.shaping_cone_beam_width);
        cb.set<Float>("cutoff_angle", spec.shaping_cone_angle);
      } else {
        auto pos = to_world.translation();
        cb.set<Point3f>("position", Point3f(pos[0], pos[1], pos[2]));
        cb.set<Color3f>("intensity.value",
                        Color3f(color[0], color[1], color[2]));
      }
    } else if (spec.prim_type == HdPrimTypeTokens->domeLight) {
      if (!spec.texture_file_path.empty()) {
        cb.set<AffineTransform4f>("to_world",
                                  AffineTransform4f(to_world.matrix));
        float scale = (color[0] + color[1] + color[2]) / 3.f;
        cb.set<Float>("scale", scale);
      } else {
        cb.set<Color3f>("radiance.value",
                        Color3f(color[0], color[1], color[2]));
      }
    } else if (spec.prim_type == HdPrimTypeTokens->distantLight) {
      cb.set<AffineTransform4f>("to_world", AffineTransform4f(to_world.matrix));
      cb.set<Color3f>("irradiance.value",
                      Color3f(color[0], color[1], color[2]));
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

  if (!filter_type.empty()) {
    mitsuba::Properties film_props("hdrfilm");
    film_props.set(
        "pixel_filter",
        static_cast<mitsuba::Object*>(
            mitsuba::PluginManager::instance()
                ->create_object<mitsuba::ReconstructionFilter<Float, Spectrum>>(
                    mitsuba::Properties(is_interactive ? "box" : spec.pixel_filter_type))
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

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildMesh(const SdfPath& id,
                                           const VtIntArray& face_indices,
                                           const PrimvarMap& primvars,
                                           mitsuba::Object* bsdf,
                                           mitsuba::Object* emitter_ptr,
                                           mitsuba::Object* sensor_ptr) {
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

  mitsuba::Properties props;
  if (emitter_ptr != nullptr) {
    props.set("emitter", emitter_ptr);
  }
  if (sensor_ptr != nullptr) {
    props.set("sensor", sensor_ptr);
  }

  auto* mesh = new mitsuba::Mesh<Float, Spectrum>(
      id_str, vertex_count, face_count, props,
      primvars.find(HdTokens->normals) != primvars.end(), false);
  mesh->set_id(id_str);
  if (bsdf != nullptr) {
    mesh->set_bsdf(dynamic_cast<mitsuba::BSDF<Float, Spectrum>*>(bsdf));
  }
  UpdateMeshInPlace(mesh, face_indices, primvars);
  mesh->initialize();
  return mitsuba::ref<mitsuba::Shape<Float, Spectrum>>(mesh);
}

MI_VARIANT void PrimTranslator<Float, Spectrum>::UpdateMeshInPlace(
    mitsuba::Object* mesh_obj, const VtIntArray& face_indices,
    const PrimvarMap& primvars) {
  using FloatStorage = typename mitsuba::Mesh<Float, Spectrum>::FloatStorage;
  using IntStorage = mitsuba::DynamicBuffer<dr::uint32_array_t<Float>>;
  auto* mesh = dynamic_cast<mitsuba::Mesh<Float, Spectrum>*>(mesh_obj);
  if (!mesh) return;

  auto points_it = primvars.find(HdTokens->points);
  if (points_it == primvars.end()) return;
  const auto& points_array = points_it->second.value.Get<VtVec3fArray>();
  size_t vertex_count = points_array.size();
  size_t face_count = face_indices.size() / 3;
  if (vertex_count == 0 || face_count == 0) return;

  TraversalCallback cb;
  mesh->traverse(&cb);

  std::vector<std::string> keys = {"vertex_positions", "faces"};
  cb.set<FloatStorage>(
      "vertex_positions",
      dr::load<FloatStorage>(points_array.data(), vertex_count * 3));
  cb.set<IntStorage>("faces",
                     dr::load<IntStorage>(face_indices.data(), face_count * 3));

  auto normals_it = primvars.find(HdTokens->normals);
  if (normals_it != primvars.end()) {
    const auto& normals_array = normals_it->second.value.Get<VtVec3fArray>();
    cb.set<FloatStorage>(
        "vertex_normals",
        dr::load<FloatStorage>(normals_array.data(), normals_array.size() * 3));
  }
  // Always add "vertex_normals" to the keys: This prevents Mitsuba from
  // recomputing the normals. We handle normal recomputation explicitly in the
  // hydra delegate.
  keys.push_back("vertex_normals");

  auto texcoords_it = primvars.find(TfToken("st"));
  if (texcoords_it != primvars.end()) {
    const auto& texcoords_array =
        texcoords_it->second.value.Get<VtVec2fArray>();
    cb.set<FloatStorage>("vertex_texcoords",
                         dr::load<FloatStorage>(texcoords_array.data(),
                                                texcoords_array.size() * 2));
    keys.push_back("vertex_texcoords");
  }

  mesh->parameters_changed(keys);
}

MI_VARIANT mitsuba::ref<mitsuba::Shape<Float, Spectrum>>
PrimTranslator<Float, Spectrum>::BuildCurves(const CurveSpec& spec,
                                             mitsuba::Object* bsdf) {
  const std::string id_str = spec.id.GetAsString();
  mitsuba::Properties props(spec.plugin_name);
  props.set("to_world", spec.transform);
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

  TraversalCallback cb;
  shape->traverse(&cb);

  using FloatStorage = typename mitsuba::Mesh<Float, Spectrum>::FloatStorage;
  using IntStorage = mitsuba::DynamicBuffer<dr::uint32_array_t<Float>>;
  using ScalarSize = typename mitsuba::Shape<Float, Spectrum>::ScalarSize;

  cb.set<ScalarSize>("control_point_count", spec.control_points.size() / 4);
  cb.set<FloatStorage>("control_points",
                       dr::load<FloatStorage>(spec.control_points.data(),
                                              spec.control_points.size()));
  cb.set<IntStorage>("segment_indices",
                     dr::load<IntStorage>(spec.segment_indices.data(),
                                          spec.segment_indices.size()));
  shape->parameters_changed();
  return shape;
}

using mitsuba::Color;
using mitsuba::MuellerMatrix;
using mitsuba::Spectrum;

MI_INSTANTIATE_CLASS(PrimTranslator)

PXR_NAMESPACE_CLOSE_SCOPE
