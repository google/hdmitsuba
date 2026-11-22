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

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <absl/synchronization/mutex.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/getenv.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolverContextBinder.h>
#include <pxr/usd/ar/resolverScopedCache.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usdImaging/usdImaging/dataSourceMaterial.h>
#include <pxr/usdImaging/usdImaging/dataSourceStageGlobals.h>

#include "hdmitsuba/config.h"
#include "hdmitsuba/material.h"
#include "hdmitsuba/materialx/compiler.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace mx = MaterialX;

// Prepares a MaterialX library document before `flattenSubgraphs()`:
// 1. Resolves non-empty relative filenames (`importLibrary` and
//    `flattenSubgraphs` drop enclosing `<materialx>` / `<nodegraph>`
//    `fileprefix` attributes).
// 2. Forwards `thin_walled` / `geometry_thin_walled` onto `<surface>` nodes
//    inside `surfaceshader` implementation nodegraphs when omitted (e.g.,
//    `standard_surface.mtlx`), matching `open_pbr_surface.mtlx`.
void PrepareLibraryDocument(const mx::DocumentPtr& doc) {
  std::vector<mx::ElementPtr> with_prefix;
  for (const mx::ElementPtr& elem : doc->traverseTree()) {
    if (elem->hasFilePrefix()) with_prefix.push_back(elem);
    auto ve = elem->asA<mx::ValueElement>();
    if (!ve || ve->getType() != mx::FILENAME_TYPE_STRING) continue;
    const std::string& raw = ve->getValueString();
    if (raw.empty() || mx::FilePath(raw).isAbsolute()) continue;
    mx::FilePath p(ve->getActiveFilePrefix() + raw);
    mx::FilePath dir = mx::FilePath(ve->getActiveSourceUri()).getParentPath();
    if (!p.isAbsolute() && !dir.isEmpty()) p = dir / p;
    ve->setValueString(p.asString(mx::FilePath::FormatPosix));
  }
  for (const mx::ElementPtr& elem : with_prefix) {
    elem->removeAttribute(mx::Element::FILE_PREFIX_ATTRIBUTE);
  }
  for (const mx::NodeDefPtr& nd : doc->getNodeDefs()) {
    if (nd->getType() != mx::SURFACE_SHADER_TYPE_STRING) continue;
    mx::InputPtr tw = nd->getActiveInput("thin_walled");
    if (!tw) tw = nd->getActiveInput("geometry_thin_walled");
    mx::ElementPtr impl = tw ? nd->getImplementation() : nullptr;
    auto ng = impl ? impl->asA<mx::NodeGraph>() : nullptr;
    if (!ng) continue;
    for (const mx::NodePtr& s : ng->getNodes("surface")) {
      if (!s->getInput("thin_walled")) {
        s->addInput("thin_walled", "boolean")->setInterfaceName(tw->getName());
      }
    }
  }
}

// Returns the immutable MaterialX standard libraries (initialized once).
const mx::ConstDocumentPtr& GetMaterialXStdLibraries() {
  static const absl::NoDestructor<mx::ConstDocumentPtr> std_libs([] {
    mx::FileSearchPath search_path;
    for (const char* env :
         {"PXR_MTLX_PLUGIN_SEARCH_PATHS", "PXR_MTLX_STDLIB_SEARCH_PATHS"}) {
      search_path.append(mx::FileSearchPath(TfGetenv(env)));
    }
    search_path.append(mx::getEnvironmentPath());
    if (mx::FilePath build_dir(HDMITSUBA_MATERIALX_STDLIB_DIR);
        !build_dir.isEmpty() && build_dir.exists()) {
      search_path.append(build_dir);
    }
    for (const mx::FilePath& p : mx::getDefaultDataSearchPath()) {
      mx::FilePath libs = p / "libraries";
      search_path.append(libs.exists() ? libs : p);
    }
    mx::DocumentPtr doc = mx::createDocument();
    mx::loadLibraries({}, search_path, doc);
    PrepareLibraryDocument(doc);
    return mx::ConstDocumentPtr(std::move(doc));
  }());
  return *std_libs;
}

// Loads and caches an immutable custom Sdr `.mtlx` library by URI.
mx::ConstDocumentPtr LoadCustomLibrary(const std::string& uri) {
  static absl::NoDestructor<absl::Mutex> cache_mu;
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, mx::ConstDocumentPtr>>
      cache;
  {
    absl::MutexLock lock(*cache_mu);
    if (auto it = cache->find(uri); it != cache->end()) return it->second;
  }
  mx::DocumentPtr raw_doc = mx::createDocument();
  mx::readFromXmlFile(raw_doc, uri);
  PrepareLibraryDocument(raw_doc);
  mx::DocumentPtr lib_doc = mx::createDocument();
  lib_doc->importLibrary(raw_doc);
  absl::MutexLock lock(*cache_mu);
  return cache->try_emplace(uri, mx::ConstDocumentPtr(std::move(lib_doc)))
      .first->second;
}

bool IsMaterialXNode(const HdMaterialNode2& node) {
  return absl::StartsWith(node.nodeTypeId.GetString(), "ND_") ||
         (!node.nodeTypeId.IsEmpty() &&
          SdrRegistry::GetInstance().GetShaderNodeByIdentifierAndType(
              node.nodeTypeId, TfToken("mtlx")) != nullptr);
}

const HdMaterialConnection2* FindMaterialXSurfaceTerminal(
    const HdMaterialNetwork2& network) {
  const auto* conn =
      FindTerminalConnection(network, HdMaterialTerminalTokens->surface);
  auto it = conn ? network.nodes.find(conn->upstreamNode) : network.nodes.end();
  return (it != network.nodes.end() && IsMaterialXNode(it->second)) ? conn
                                                                    : nullptr;
}

std::optional<std::string> ConvertToMtlxString(const VtValue& val) {
  if (val.IsHolding<bool>()) return val.UncheckedGet<bool>() ? "true" : "false";
  if (val.IsHolding<int>()) return absl::StrCat(val.UncheckedGet<int>());
  if (val.IsHolding<SdfAssetPath>()) return ResolvePathFromValue(val);
  if (val.IsHolding<std::string>()) return val.UncheckedGet<std::string>();
  if (val.IsHolding<TfToken>()) return val.UncheckedGet<TfToken>().GetString();

  auto format_span = [](const auto* p, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) {
      if (i) out += ", ";
      absl::StrAppendFormat(&out, "%.9g", p[i]);
    }
    return out;
  };
  if (val.IsHolding<float>()) {
    return absl::StrFormat("%.9g", val.UncheckedGet<float>());
  }
  if (val.IsHolding<double>()) {
    return absl::StrFormat("%.9g", val.UncheckedGet<double>());
  }
  if (val.IsHolding<GfVec2f>()) {
    return format_span(val.UncheckedGet<GfVec2f>().data(), 2);
  }
  if (val.IsHolding<GfVec2d>()) {
    return format_span(val.UncheckedGet<GfVec2d>().data(), 2);
  }
  if (val.IsHolding<GfVec3f>()) {
    return format_span(val.UncheckedGet<GfVec3f>().data(), 3);
  }
  if (val.IsHolding<GfVec3d>()) {
    return format_span(val.UncheckedGet<GfVec3d>().data(), 3);
  }
  if (val.IsHolding<GfVec4f>()) {
    return format_span(val.UncheckedGet<GfVec4f>().data(), 4);
  }
  if (val.IsHolding<GfVec4d>()) {
    return format_span(val.UncheckedGet<GfVec4d>().data(), 4);
  }
  if (val.IsHolding<GfMatrix3d>()) {
    return format_span(val.UncheckedGet<GfMatrix3d>().data(), 9);
  }
  if (val.IsHolding<GfMatrix4d>()) {
    return format_span(val.UncheckedGet<GfMatrix4d>().data(), 16);
  }
  TF_WARN("Unsupported MaterialX parameter type '%s'",
          val.GetTypeName().c_str());
  return std::nullopt;
}

// Parses legacy MaterialX 1.38 `ND_swizzle_<in>_<out>` identifiers into
// `{in_type, out_type}` so `mx::Document::upgradeVersion()` upgrades them to
// `extract` / `combine`.
std::optional<std::pair<std::string_view, std::string_view>>
ParseSwizzleNodeDef(std::string_view type_str) {
  if (!absl::ConsumePrefix(&type_str, "ND_swizzle_")) return std::nullopt;
  size_t sep = type_str.find('_');
  if (sep == std::string_view::npos || sep == 0 || sep + 1 >= type_str.size() ||
      type_str.find('_', sep + 1) != std::string_view::npos) {
    return std::nullopt;
  }
  return std::pair<std::string_view, std::string_view>(
      type_str.substr(0, sep), type_str.substr(sep + 1));
}

// Resolves a MaterialX `NodeDef` from `std_libs`, handling legacy 1.38
// `ND_normalmap` and custom Sdr `.mtlx` assets (recorded in `custom_libs`).
mx::NodeDefPtr ResolveNodeDef(const TfToken& hd_node_type,
                              const mx::ConstDocumentPtr& std_libs,
                              std::vector<mx::ConstDocumentPtr>& custom_libs) {
  const std::string& type_str = hd_node_type.GetString();
  if (mx::NodeDefPtr nd = std_libs->getNodeDef(type_str)) return nd;
  if (type_str == "ND_normalmap") {
    if (mx::NodeDefPtr nd = std_libs->getNodeDef("ND_normalmap_float")) {
      return nd;
    }
  }

  SdrShaderNodeConstPtr sdr_node =
      SdrRegistry::GetInstance().GetShaderNodeByIdentifierAndType(
          hd_node_type, TfToken("mtlx"));
  if (!sdr_node) return nullptr;
  const std::string& impl_name = sdr_node->GetImplementationName();
  if (mx::NodeDefPtr nd = std_libs->getNodeDef(impl_name)) return nd;
  const std::string& uri = sdr_node->GetResolvedImplementationURI();
  if (uri.empty()) return nullptr;
  mx::ConstDocumentPtr lib = LoadCustomLibrary(uri);
  if (mx::NodeDefPtr nd = lib->getNodeDef(impl_name)) {
    if (std::find(custom_libs.begin(), custom_libs.end(), lib) ==
        custom_libs.end()) {
      custom_libs.push_back(std::move(lib));
    }
    return nd;
  }
  return nullptr;
}

std::string InferMtlxParamType(const mx::NodeDef* nd,
                               const HdMaterialNode2& hd_node,
                               const std::string& name, const VtValue& val) {
  if (nd) {
    if (mx::InputPtr p = nd->getActiveInput(name)) return p->getType();
  }

  std::string_view usd_type;
  if (auto it =
          hd_node.parameters.find(TfToken(absl::StrCat("typeName:", name)));
      it != hd_node.parameters.end() && it->second.IsHolding<TfToken>()) {
    usd_type = it->second.UncheckedGet<TfToken>().GetString();
  }
  if (val.IsHolding<bool>()) return "boolean";
  if (val.IsHolding<int>()) return "integer";
  if (val.IsHolding<float>() || val.IsHolding<double>()) return "float";
  if (val.IsHolding<GfVec2f>() || val.IsHolding<GfVec2d>()) return "vector2";
  if (val.IsHolding<GfVec3f>() || val.IsHolding<GfVec3d>()) {
    return (usd_type == "float3" || usd_type == "double3" ||
            usd_type == "vector3f" || usd_type == "vector3d" ||
            usd_type == "normal3f" || usd_type == "normal3d" ||
            usd_type == "point3f" || usd_type == "point3d")
               ? "vector3"
               : "color3";
  }
  if (val.IsHolding<GfVec4f>() || val.IsHolding<GfVec4d>()) {
    return (usd_type == "float4" || usd_type == "double4") ? "vector4"
                                                           : "color4";
  }
  if (val.IsHolding<GfMatrix3d>()) return "matrix33";
  if (val.IsHolding<GfMatrix4d>()) return "matrix44";
  if (val.IsHolding<SdfAssetPath>() || usd_type == "asset") return "filename";
  return "string";
}

const std::string* GetMtlxVersion(const HdMaterialNetwork2& net2) {
  auto it = net2.config.find(TfToken("mtlx:version"));
  return (it != net2.config.end() && it->second.IsHolding<std::string>())
             ? &it->second.UncheckedGet<std::string>()
             : nullptr;
}

// Populates a standalone `mx::Document` with the subgraph rooted at
// `term_conn`.
class DocumentBuilder {
 public:
  using BuiltNode = std::pair<mx::NodePtr, mx::NodeDefPtr>;

  struct Result {
    mx::DocumentPtr doc;
    std::vector<mx::ConstDocumentPtr> custom_libs;
  };

  DocumentBuilder(const HdMaterialNetwork2& net2,
                  const mx::ConstDocumentPtr& std_libs,
                  const std::string* version)
      : net2_(net2), std_libs_(std_libs), doc_(mx::createDocument()) {
    doc_->setVersionString(version ? *version : "1.38");
  }

  Result Build(const HdMaterialConnection2& term_conn) {
    BuiltNode root = BuildNode(term_conn.upstreamNode);
    if (!ConnectPort(
            [&](const std::string& type) {
              return doc_->addOutput("out", type);
            },
            root, term_conn.upstreamOutputName)) {
      return {};
    }
    return {std::move(doc_), std::move(custom_libs_)};
  }

 private:
  template <typename AddPortFn>
  static bool ConnectPort(AddPortFn&& add_port, const BuiltNode& up,
                          const TfToken& output_name) {
    const auto& [up_node, up_nd] = up;
    if (!up_node) return false;
    if (up_node->isMultiOutputType()) {
      mx::OutputPtr out_port =
          up_nd ? up_nd->getActiveOutput(output_name.GetString()) : nullptr;
      if (!out_port) return false;
      mx::PortElementPtr port = add_port(out_port->getType());
      port->setConnectedNode(up_node);
      port->setOutputString(out_port->getName());
    } else {
      add_port(up_node->getType())->setConnectedNode(up_node);
    }
    return true;
  }

  BuiltNode BuildNode(const SdfPath& path) {
    if (visiting_.contains(path)) {
      TF_WARN("Cycle detected in MaterialX network at '%s'; breaking cycle.",
              path.GetText());
      return {};
    }
    if (auto it = built_.find(path); it != built_.end()) return it->second;
    auto node_it = net2_.nodes.find(path);
    if (node_it == net2_.nodes.end()) return {};
    const HdMaterialNode2& hd_node = node_it->second;

    std::string mx_name = absl::StrCat("n_", built_.size());
    mx::NodePtr mx_node;
    mx::NodeDefPtr nd;
    auto swizzle = ParseSwizzleNodeDef(hd_node.nodeTypeId.GetString());
    if (swizzle) {
      mx_node = doc_->addNode("swizzle", mx_name, std::string(swizzle->second));
    } else {
      nd = ResolveNodeDef(hd_node.nodeTypeId, std_libs_, custom_libs_);
      if (!nd) {
        TF_WARN("No MaterialX NodeDef '%s' for node '%s'; ignoring the node.",
                hd_node.nodeTypeId.GetText(), path.GetText());
        built_.emplace(path, BuiltNode{});
        return {};
      }
      mx_node = doc_->addNodeInstance(nd, mx_name);
    }
    built_.emplace(path, BuiltNode{mx_node, nd});

    visiting_.insert(path);
    for (const auto& [in_tok, conns] : hd_node.inputConnections) {
      if (conns.empty()) continue;
      const HdMaterialConnection2& c = conns[0];
      const std::string& in_name = in_tok.GetString();
      ConnectPort(
          [&](const std::string& type) {
            return mx_node->addInput(in_name, type);
          },
          BuildNode(c.upstreamNode), c.upstreamOutputName);
    }
    visiting_.erase(path);

    for (const auto& [param_tok, val] : hd_node.parameters) {
      const std::string& p_name = param_tok.GetString();
      if (mx_node->getInput(p_name) ||
          absl::StartsWith(p_name, "colorSpace:") ||
          absl::StartsWith(p_name, "typeName:")) {
        continue;
      }
      std::string in_type =
          (swizzle && p_name == "in") ? std::string(swizzle->first)
          : (swizzle && p_name == "channels")
              ? std::string("string")
              : InferMtlxParamType(nd.get(), hd_node, p_name, val);
      std::optional<std::string> val_str =
          (in_type == "boolean" && val.IsHolding<int>())
              ? std::optional<std::string>(val.UncheckedGet<int>() ? "true"
                                                                   : "false")
              : ConvertToMtlxString(val);
      if (!val_str) continue;
      mx::InputPtr mx_in = mx_node->setInputValue(p_name, *val_str, in_type);
      if (auto cs_it = hd_node.parameters.find(
              TfToken(absl::StrCat("colorSpace:", p_name)));
          cs_it != hd_node.parameters.end() &&
          cs_it->second.IsHolding<TfToken>()) {
        const TfToken& cs = cs_it->second.UncheckedGet<TfToken>();
        if (!cs.IsEmpty()) mx_in->setColorSpace(cs.GetString());
      }
    }

    return {mx_node, nd};
  }

  const HdMaterialNetwork2& net2_;
  const mx::ConstDocumentPtr& std_libs_;
  mx::DocumentPtr doc_;
  std::vector<mx::ConstDocumentPtr> custom_libs_;
  absl::flat_hash_map<SdfPath, BuiltNode, SdfPath::Hash> built_;
  absl::flat_hash_set<SdfPath, SdfPath::Hash> visiting_;
};

// Upgrades `doc` while preserving constant `which >= 5` on 10-input 1.39
// `switch` nodes (`in6`..`in10` authored) when `mtlx:version` was not set.
void UpgradeMaterialXDocument(const mx::DocumentPtr& doc,
                              bool has_explicit_version) {
  std::vector<std::pair<mx::InputPtr, std::string>> saved_switch_which;
  if (!has_explicit_version) {
    for (const mx::NodePtr& node : doc->getNodes("switch")) {
      mx::InputPtr which = node->getInput("which");
      if (!which || !which->hasValueString()) continue;
      for (int i = 6; i <= 10; ++i) {
        if (node->getInput(absl::StrCat("in", i))) {
          saved_switch_which.emplace_back(which, which->getValueString());
          break;
        }
      }
    }
  }
  doc->upgradeVersion();
  for (const auto& [which, val_str] : saved_switch_which) {
    which->setValueString(val_str);
  }
}

template <typename Float, typename Spectrum>
TranslatedMaterial TranslateWithTextureCache(
    const SdfPath& id, const PreparedMaterialX& prepared,
    const absl::flat_hash_set<TextureKey>& keys, std::string_view variant) {
  TextureCache<Float, Spectrum> texture_cache;
  texture_cache.Preload(keys);
  return TranslateMaterialXNetwork(
      id, prepared, [&](const TextureKey& k) { return texture_cache.Find(k); },
      variant);
}

}  // namespace

struct PreparedMaterialX {
  mx::DocumentPtr doc;
  mx::NodePtr shader_node;
};

std::shared_ptr<const PreparedMaterialX> PrepareMaterialXNetwork(
    const SdfPath& id, const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey&)>& callback) {
  if (!IsMaterialXEnabled()) return nullptr;
  const auto* term_conn = FindMaterialXSurfaceTerminal(network);
  if (!term_conn) return nullptr;
  auto prepared = std::make_shared<PreparedMaterialX>();
  try {
    const mx::ConstDocumentPtr& std_libs = GetMaterialXStdLibraries();
    const std::string* version = GetMtlxVersion(network);
    auto [doc, custom_libs] =
        DocumentBuilder(network, std_libs, version).Build(*term_conn);
    if (!doc) return prepared;

    // Upgrade the small standalone user graph BEFORE attaching `std_libs`.
    UpgradeMaterialXDocument(doc, version != nullptr);

    if (custom_libs.empty()) {
      doc->setDataLibrary(std_libs);
    } else {
      mx::DocumentPtr overlay = mx::createDocument();
      overlay->setDataLibrary(std_libs);
      for (const mx::ConstDocumentPtr& lib : custom_libs) {
        overlay->importLibrary(lib);
      }
      doc->setDataLibrary(overlay);
    }
    doc->flattenSubgraphs();
    mx::OutputPtr out = doc->getOutput("out");
    mx::NodePtr shader_node = out ? out->getConnectedNode() : nullptr;
    if (!shader_node) return prepared;
    if (!doc->getNodes("image").empty()) {
      DiscoverSurfaceShaderTextures(shader_node, callback);
    }
    prepared->doc = std::move(doc);
    prepared->shader_node = std::move(shader_node);
  } catch (const std::exception& e) {
    TF_WARN("Failed to translate MaterialX network '%s': %s", id.GetText(),
            e.what());
  }
  return prepared;
}

TranslatedMaterial TranslateMaterialXNetwork(
    const SdfPath& id, const PreparedMaterialX& prepared,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture,
    std::string_view variant) {
  if (!prepared.shader_node) return {};
  try {
    TranslatedMaterial res;
    CompiledSurfaceShader compiled =
        CompileSurfaceShader(prepared.shader_node, variant, lookup_texture);
    res.mesh_attributes = std::move(compiled.mesh_attributes);
    if (compiled.radiance_texture) {
      mitsuba::ref<mitsuba::Object> radiance = compiled.radiance_texture;
      if (absl::StrContains(variant, "spectral")) {
        // Linear sRGB colors are relative to the D65 white point. As for
        // Mitsuba's own RGB emitter colors, multiply the upsampled color by
        // the D65 spectrum.
        mitsuba::Properties d65_props("d65");
        d65_props.set("nested", radiance);
        radiance = CreateExpandedObject(d65_props, variant,
                                        mitsuba::ObjectType::Texture);
      }
      mitsuba::Properties emitter_props("area");
      emitter_props.set("radiance", radiance);
      res.shape_emitter_props = std::move(emitter_props);
    }

    res.bsdf = std::move(compiled.bsdf);
    return res;
  } catch (const std::exception& e) {
    TF_WARN("Failed to translate MaterialX network '%s': %s", id.GetText(),
            e.what());
    return {};
  }
}

std::optional<TranslatedMaterial> TranslateUsdMaterialX(
    const UsdPrim& prim, UsdTimeCode time, std::string_view variant) {
  if (!IsMaterialXEnabled() || !prim) return std::nullopt;
  ArResolverContextBinder binder(prim.GetStage()->GetPathResolverContext());
  ArResolverScopedCache resolver_cache;

  struct Globals final : UsdImagingDataSourceStageGlobals {
    UsdTimeCode time;
    explicit Globals(UsdTimeCode t) : time(t) {}
    UsdTimeCode GetTime() const override { return time; }
    void FlagAsTimeVarying(const SdfPath&,
                           const HdDataSourceLocator&) const override {}
    void FlagAsAssetPathDependent(const SdfPath&) const override {}
  } globals(time);

  HdMaterialNetwork2 net = ConvertMaterialSchema(
      HdMaterialSchema(UsdImagingDataSourceMaterial::New(prim, globals)));
  absl::flat_hash_set<TextureKey> keys;
  auto prepared = PrepareMaterialXNetwork(
      prim.GetPath(), net, [&](const TextureKey& k) { keys.insert(k); });
  if (!prepared) return std::nullopt;
  TranslatedMaterial trans;
  if (prepared->shader_node) {
    using namespace mitsuba;
    trans = MI_INVOKE_VARIANT(variant, TranslateWithTextureCache,
                              prim.GetPath(), *prepared, keys, variant);
    if (trans.bsdf) trans.bsdf->set_id(prim.GetPath().GetAsString());
  }
  return trans;
}

PXR_NAMESPACE_CLOSE_SCOPE
