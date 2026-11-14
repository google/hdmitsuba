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

#include "hdmitsuba/materialx/translator.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/strip.h>
#include <absl/synchronization/mutex.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/plugin.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hdmitsuba/materialx/compiler.h"
#include "hdmitsuba/prim_translator.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace mx = MaterialX;
namespace mtlx = mitsuba::mtlx;

std::string ResolveUsdAssetPath(const std::string& file) {
  std::string resolved = ArGetResolver().Resolve(file).GetPathString();
  if (!resolved.empty() &&
      (mitsuba::fs::exists(resolved) ||
       ArGetResolver().OpenAsset(ArResolvedPath(resolved)))) {
    return resolved;
  }
  return {};
}

// `HdMtlxCreateMtlxDocumentFromHdNetwork` mutates unsynchronized static state
// (`swizzleDoc` and `HdMtlxStdLibraries()`); serialize `HdMtlxStdLibraries()`
// access and document creation while keeping flattening and compilation
// parallel.
absl::NoDestructor<absl::Mutex> hd_mtlx_mu;

template <typename Map, typename Pred>
void EraseIf(Map& map, Pred pred) {
  for (auto it = map.begin(); it != map.end();) {
    it = pred(*it) ? map.erase(it) : std::next(it);
  }
}

// Replaces SdfAssetPath values with their resolved path in AssetPath so
// HdMtlxConvertToString preserves full resolved filesystem paths, and upgrades
// legacy MaterialX 1.38 nodeTypeIds (such as ND_normalmap) for 1.39 stdlib.
void PrepareMaterialXNetwork(HdMaterialNetwork2& net2) {
  for (auto& kv_node : net2.nodes) {
    auto& node = kv_node.second;
    EraseIf(node.inputConnections, [&](auto& entry) {
      auto& conns = entry.second;
      conns.erase(std::remove_if(conns.begin(), conns.end(),
                                 [&](const HdMaterialConnection2& c) {
                                   return !net2.nodes.count(c.upstreamNode);
                                 }),
                  conns.end());
      return conns.empty();
    });
    EraseIf(node.parameters, [&](auto& kv) {
      if (node.inputConnections.count(kv.first)) return true;
      const std::string& name = kv.first.GetString();
      VtValue& val = kv.second;
      if ((name == "clampoutput" || name == "doclamp") &&
          val.IsHolding<int>()) {
        val = VtValue(val.UncheckedGet<int>() != 0);
      } else if (val.IsHolding<SdfAssetPath>()) {
        if (std::string resolved = ResolvePathFromValue(val);
            !resolved.empty()) {
          val = VtValue(SdfAssetPath(resolved, resolved));
        }
      }
      return false;
    });
  }

  absl::MutexLock lock(*hd_mtlx_mu);
  const mx::DocumentPtr& std_libs = HdMtlxStdLibraries();
  auto conn_output_type =
      [&](const HdMaterialConnection2& conn) -> std::string {
    auto up_it = net2.nodes.find(conn.upstreamNode);
    if (up_it == net2.nodes.end()) return {};
    mx::NodeDefPtr nd =
        std_libs->getNodeDef(up_it->second.nodeTypeId.GetString());
    if (!nd) return {};
    if (nd->getType() == mx::MULTI_OUTPUT_TYPE_STRING) {
      mx::OutputPtr out =
          nd->getActiveOutput(conn.upstreamOutputName.GetString());
      return out ? out->getType() : std::string();
    }
    return nd->getType();
  };

  absl::flat_hash_set<SdfPath, SdfPath::Hash> visited;
  auto upgrade_arith = [&](auto& self, const SdfPath& path) -> void {
    if (!visited.insert(path).second) return;
    auto node_it = net2.nodes.find(path);
    if (node_it == net2.nodes.end()) return;
    HdMaterialNode2& node = node_it->second;
    for (const auto& [_, conns] : node.inputConnections) {
      for (const auto& c : conns) self(self, c.upstreamNode);
    }
    std::string_view op = node.nodeTypeId.GetString();
    if (!absl::ConsumePrefix(&op, "ND_") ||
        !absl::ConsumeSuffix(&op, "_float") ||
        (op != "add" && op != "subtract" && op != "multiply" &&
         op != "divide")) {
      return;
    }
    auto port_type = [&](const TfToken& port) -> std::string {
      if (auto it = node.inputConnections.find(port);
          it != node.inputConnections.end() && !it->second.empty()) {
        return conn_output_type(it->second[0]);
      }
      if (auto it = node.parameters.find(port);
          it != node.parameters.end() && it->second.IsHolding<GfVec3f>()) {
        auto tn_it = node.parameters.find(
            TfToken(absl::StrCat("typeName:", port.GetString())));
        return (tn_it != node.parameters.end() &&
                tn_it->second == VtValue(TfToken("color3f")))
                   ? "color3"
                   : "vector3";
      }
      return "float";
    };
    std::string t1 = port_type(TfToken("in1"));
    std::string t2 = port_type(TfToken("in2"));
    auto is_vec3 = [](std::string_view t) {
      return t == "vector3" || t == "color3";
    };
    if (!is_vec3(t1) && !is_vec3(t2)) return;
    const std::string& vec_t = is_vec3(t1) ? t1 : t2;
    if (is_vec3(t1) && t2 == "float") {
      node.nodeTypeId = TfToken(absl::StrCat("ND_", op, "_", vec_t, "FA"));
    } else {
      node.nodeTypeId = TfToken(absl::StrCat("ND_", op, "_", vec_t));
      for (const char* p : {"in1", "in2"}) {
        if (auto it = node.parameters.find(TfToken(p));
            it != node.parameters.end() && it->second.IsHolding<float>()) {
          it->second = VtValue(GfVec3f(it->second.UncheckedGet<float>()));
        }
      }
    }
  };
  for (auto& [_, node] : net2.nodes) {
    if (node.nodeTypeId == TfToken("ND_normalmap") ||
        node.nodeTypeId == TfToken("ND_normalmap_vector3")) {
      bool is_vec2 = false;
      if (auto c_it = node.inputConnections.find(TfToken("scale"));
          c_it != node.inputConnections.end() && !c_it->second.empty()) {
        is_vec2 = conn_output_type(c_it->second[0]) == "vector2";
      } else if (auto p_it = node.parameters.find(TfToken("scale"));
                 p_it != node.parameters.end()) {
        is_vec2 = p_it->second.IsHolding<GfVec2f>();
      }
      node.nodeTypeId =
          TfToken(is_vec2 ? "ND_normalmap_vector2" : "ND_normalmap_float");
    }
  }
  for (auto& [node_path, node] : net2.nodes) {
    upgrade_arith(upgrade_arith, node_path);
    EraseIf(node.parameters, [](const auto& kv) {
      return absl::StartsWith(kv.first.GetString(), "typeName:");
    });
  }

  std::vector<std::pair<SdfPath, HdMaterialNode2>> adapters;
  size_t conv_idx = 0;
  for (auto& [node_path, node] : net2.nodes) {
    mx::NodeDefPtr nd = std_libs->getNodeDef(node.nodeTypeId.GetString());
    if (!nd) continue;
    for (auto& [in_token, conns] : node.inputConnections) {
      mx::InputPtr in_port = nd->getActiveInput(in_token.GetString());
      if (!in_port) continue;
      const std::string& in_t = in_port->getType();
      for (HdMaterialConnection2& conn : conns) {
        std::string out_t = conn_output_type(conn);
        if (out_t.empty() || in_t.empty() || out_t == in_t) continue;
        std::string conv_id = absl::StrCat("ND_convert_", out_t, "_", in_t);
        if (!std_libs->getNodeDef(conv_id)) continue;
        SdfPath conv_path = node_path.AppendChild(
            TfToken(absl::StrCat("__mtlx_conv_", conv_idx++)));
        adapters.push_back(
            {conv_path, {TfToken(conv_id), {}, {{TfToken("in"), {conn}}}}});
        conn = {conv_path, TfToken("out")};
      }
    }
  }
  for (auto& [path, conv_node] : adapters) {
    net2.nodes.emplace(path, std::move(conv_node));
  }
}

bool IsMaterialXNode(const HdMaterialNode2& node) {
  const std::string& id_str = node.nodeTypeId.GetString();
  return absl::StartsWith(id_str, "ND_") || id_str == "standard_surface" ||
         id_str == "open_pbr_surface" || id_str == "gltf_pbr" ||
         id_str == "disney_principled" || id_str == "surface" ||
         id_str == "surface_unlit" || absl::StartsWith(id_str, "Lama");
}

// Creates the flattened MaterialX document and returns its terminal shader node
// for `terminal_name` in `net2`.
std::pair<mx::DocumentPtr, mx::NodePtr> GetTerminalShaderNode(
    const HdMaterialNetwork2& net2, const TfToken& terminal_name,
    const std::string& shader_input) {
  const auto* conn = FindTerminalConnection(net2, terminal_name);
  auto node_it = conn ? net2.nodes.find(conn->upstreamNode) : net2.nodes.end();
  if (node_it == net2.nodes.end() || !IsMaterialXNode(node_it->second)) {
    return {};
  }
  mx::DocumentPtr doc;
  {
    absl::MutexLock lock(*hd_mtlx_mu);
    doc = HdMtlxCreateMtlxDocumentFromHdNetwork(
        net2, node_it->second, conn->upstreamNode, SdfPath("/Material"),
        HdMtlxStdLibraries(), /*hdMtlxData=*/nullptr);
  }
  mtlx::FlattenMtlxDocument(doc);
  for (const mx::NodePtr& mat : doc->getMaterialNodes()) {
    if (mx::NodePtr node = mat->getConnectedNode(shader_input)) {
      return {doc, node};
    }
  }
  return {};
}

}  // namespace

bool IsMaterialXNetwork(const HdMaterialNetwork2& network) {
  for (const auto& [node_path, node] : network.nodes) {
    if (IsMaterialXNode(node)) {
      return true;
    }
  }
  return false;
}

void DiscoverMaterialXTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey&)>& callback) {
  try {
    HdMaterialNetwork2 net2 = network;
    PrepareMaterialXNetwork(net2);
    if (auto [doc, _] = GetTerminalShaderNode(
            net2, HdMaterialTerminalTokens->surface, "surfaceshader");
        doc) {
      mtlx::CollectMtlxImageKeys(doc, ResolveUsdAssetPath, callback);
    }
  } catch (const std::exception& e) {
    TF_WARN("Failed to discover MaterialX textures: %s", e.what());
  }
}

TranslatedMaterial TranslateMaterialXNetwork(
    const SdfPath& id, const HdMaterialNetwork2& network,
    const mtlx::TextureLookup& lookup_texture, std::string_view variant) {
  try {
    TranslatedMaterial res;
    HdMaterialNetwork2 net2 = network;
    PrepareMaterialXNetwork(net2);
    mtlx::MtlxCompileOptions opts{variant, lookup_texture, ResolveUsdAssetPath,
                                  &res.mesh_attributes};

    auto [doc, shader_node] = GetTerminalShaderNode(
        net2, HdMaterialTerminalTokens->surface, "surfaceshader");
    if (!shader_node) return res;

    res.shape_emitter_props = mtlx::CompileSurfaceEdf(shader_node, opts);
    mitsuba::Properties bsdf_props("diffuse");
    bsdf_props.set("reflectance", 0.0f);
    res.bsdf = mitsuba::PluginManager::instance()->create_object(
        bsdf_props, variant, mitsuba::ObjectType::BSDF);
    if (res.bsdf) res.bsdf->set_id(id.GetAsString());
    return res;
  } catch (const std::exception& e) {
    TF_WARN("Failed to translate MaterialX network '%s': %s", id.GetText(),
            e.what());
    return {};
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
