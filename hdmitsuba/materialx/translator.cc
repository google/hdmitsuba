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

#include "hdmitsuba/material.h"
#include "hdmitsuba/materialx/compiler.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace mx = MaterialX;

// Prepares a MaterialX library document before use:
// 1. Resolves non-empty relative filenames (`importLibrary` and lazy
//    `NodeGraph` compilation drop enclosing `<materialx>` / `<nodegraph>`
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

bool IsMaterialXTerminal(const HdMaterialNetwork2& network,
                         const HdMaterialConnection2* conn) {
  auto it = conn ? network.nodes.find(conn->upstreamNode) : network.nodes.end();
  return it != network.nodes.end() && IsMaterialXNode(it->second);
}

// Resolves a Hydra parameter's MaterialX type and formats its `VtValue` string
// in a single type dispatch. Returns `{mtlx_type, value_string}`, or
// `std::nullopt` if `val` holds an unsupported type.
std::optional<std::pair<std::string, std::string>> FormatMtlxParam(
    const mx::NodeDef* nd, const HdMaterialNode2& hd_node,
    const std::string& name, const VtValue& val,
    std::string_view forced_type = {}) {
  std::string explicit_type(forced_type);
  if (explicit_type.empty() && nd) {
    if (mx::InputPtr p = nd->getActiveInput(name)) explicit_type = p->getType();
  }
  std::string_view usd_type;
  if (explicit_type.empty()) {
    if (auto it =
            hd_node.parameters.find(TfToken(absl::StrCat("typeName:", name)));
        it != hd_node.parameters.end() && it->second.IsHolding<TfToken>()) {
      usd_type = it->second.UncheckedGet<TfToken>().GetString();
    }
  }

  auto make = [&](std::string_view inferred_type, std::string val_str) {
    return std::make_pair(explicit_type.empty() ? std::string(inferred_type)
                                                : std::move(explicit_type),
                          std::move(val_str));
  };
  auto format_span = [](const auto* p, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) {
      if (i) out += ", ";
      absl::StrAppendFormat(&out, "%.9g", p[i]);
    }
    return out;
  };
  auto try_vec = [&](auto tag, int n, std::string_view inferred_type)
      -> std::optional<std::pair<std::string, std::string>> {
    using T = decltype(tag);
    if (!val.IsHolding<T>()) return std::nullopt;
    return make(inferred_type, format_span(val.UncheckedGet<T>().data(), n));
  };

  if (val.IsHolding<bool>()) {
    return make("boolean", val.UncheckedGet<bool>() ? "true" : "false");
  }
  if (val.IsHolding<int>()) {
    int v = val.UncheckedGet<int>();
    return make("integer", explicit_type == "boolean" ? (v ? "true" : "false")
                                                      : absl::StrCat(v));
  }
  if (val.IsHolding<SdfAssetPath>()) {
    return make("filename", ResolvePathFromValue(val));
  }
  if (val.IsHolding<std::string>()) {
    return make(usd_type == "asset" ? "filename" : "string",
                val.UncheckedGet<std::string>());
  }
  if (val.IsHolding<TfToken>()) {
    return make(usd_type == "asset" ? "filename" : "string",
                val.UncheckedGet<TfToken>().GetString());
  }
  if (val.IsHolding<float>()) {
    return make("float", absl::StrFormat("%.9g", val.UncheckedGet<float>()));
  }
  if (val.IsHolding<double>()) {
    return make("float", absl::StrFormat("%.9g", val.UncheckedGet<double>()));
  }
  if (auto r = try_vec(GfVec2f{}, 2, "vector2")) return r;
  if (auto r = try_vec(GfVec2d{}, 2, "vector2")) return r;
  std::string_view v3_type =
      (usd_type == "float3" || usd_type == "double3" ||
       usd_type == "vector3f" || usd_type == "vector3d" ||
       usd_type == "normal3f" || usd_type == "normal3d" ||
       usd_type == "point3f" || usd_type == "point3d")
          ? "vector3"
          : "color3";
  if (auto r = try_vec(GfVec3f{}, 3, v3_type)) return r;
  if (auto r = try_vec(GfVec3d{}, 3, v3_type)) return r;
  std::string_view v4_type =
      (usd_type == "float4" || usd_type == "double4") ? "vector4" : "color4";
  if (auto r = try_vec(GfVec4f{}, 4, v4_type)) return r;
  if (auto r = try_vec(GfVec4d{}, 4, v4_type)) return r;
  if (auto r = try_vec(GfMatrix3d{}, 9, "matrix33")) return r;
  if (auto r = try_vec(GfMatrix4d{}, 16, "matrix44")) return r;

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

const std::string* GetMtlxVersion(const HdMaterialNetwork2& net2) {
  auto it = net2.config.find(TfToken("mtlx:version"));
  return (it != net2.config.end() && it->second.IsHolding<std::string>())
             ? &it->second.UncheckedGet<std::string>()
             : nullptr;
}

using BuiltNode = std::pair<mx::NodePtr, mx::NodeDefPtr>;

struct DocBuildCtx {
  const HdMaterialNetwork2& net2;
  const mx::ConstDocumentPtr& std_libs;
  mx::DocumentPtr doc;
  std::vector<mx::ConstDocumentPtr> custom_libs = {};
  absl::flat_hash_map<SdfPath, BuiltNode, SdfPath::Hash> built = {};
  absl::flat_hash_set<SdfPath, SdfPath::Hash> visiting = {};
};

template <typename AddPortFn>
bool ConnectPort(AddPortFn&& add_port, const BuiltNode& up,
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

BuiltNode BuildDocumentNode(DocBuildCtx& ctx, const SdfPath& path) {
  if (ctx.visiting.contains(path)) {
    TF_WARN("Cycle detected in MaterialX network at '%s'; breaking cycle.",
            path.GetText());
    return {};
  }
  if (auto it = ctx.built.find(path); it != ctx.built.end()) return it->second;
  auto node_it = ctx.net2.nodes.find(path);
  if (node_it == ctx.net2.nodes.end()) return {};
  const HdMaterialNode2& hd_node = node_it->second;

  std::string mx_name = absl::StrCat("n_", ctx.built.size());
  mx::NodePtr mx_node;
  mx::NodeDefPtr nd;
  auto swizzle = ParseSwizzleNodeDef(hd_node.nodeTypeId.GetString());
  if (swizzle) {
    mx_node =
        ctx.doc->addNode("swizzle", mx_name, std::string(swizzle->second));
  } else {
    nd = ResolveNodeDef(hd_node.nodeTypeId, ctx.std_libs, ctx.custom_libs);
    if (!nd) {
      TF_WARN("No MaterialX NodeDef '%s' for node '%s'; ignoring the node.",
              hd_node.nodeTypeId.GetText(), path.GetText());
      ctx.built.emplace(path, BuiltNode{});
      return {};
    }
    mx_node = ctx.doc->addNodeInstance(nd, mx_name);
  }
  ctx.built.emplace(path, BuiltNode{mx_node, nd});

  ctx.visiting.insert(path);
  for (const auto& [in_tok, conns] : hd_node.inputConnections) {
    if (conns.empty()) continue;
    const HdMaterialConnection2& c = conns[0];
    const std::string& in_name = in_tok.GetString();
    ConnectPort(
        [&](const std::string& type) {
          return mx_node->addInput(in_name, type);
        },
        BuildDocumentNode(ctx, c.upstreamNode), c.upstreamOutputName);
  }
  ctx.visiting.erase(path);

  for (const auto& [param_tok, val] : hd_node.parameters) {
    const std::string& p_name = param_tok.GetString();
    if (mx_node->getInput(p_name) || absl::StartsWith(p_name, "colorSpace:") ||
        absl::StartsWith(p_name, "typeName:")) {
      continue;
    }
    std::string_view forced_type = (swizzle && p_name == "in") ? swizzle->first
                                   : (swizzle && p_name == "channels")
                                       ? std::string_view("string")
                                       : std::string_view();
    auto formatted =
        FormatMtlxParam(nd.get(), hd_node, p_name, val, forced_type);
    if (!formatted) continue;
    mx::InputPtr mx_in =
        mx_node->setInputValue(p_name, formatted->second, formatted->first);
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

// Populates a standalone, upgraded `mx::Document` with the MaterialX subgraphs
// rooted at `surface_conn` and `disp_conn`, attached to `std_libs`.
mx::DocumentPtr BuildMaterialXDocument(
    const HdMaterialNetwork2& network, const mx::ConstDocumentPtr& std_libs,
    const HdMaterialConnection2* surface_conn,
    const HdMaterialConnection2* disp_conn) {
  const std::string* version = GetMtlxVersion(network);
  DocBuildCtx ctx{network, std_libs, mx::createDocument()};
  ctx.doc->setVersionString(version ? *version : "1.38");
  auto add_terminal = [&](const char* out_name,
                          const HdMaterialConnection2* conn) {
    return conn && ConnectPort(
                       [&](const std::string& type) {
                         return ctx.doc->addOutput(out_name, type);
                       },
                       BuildDocumentNode(ctx, conn->upstreamNode),
                       conn->upstreamOutputName);
  };
  bool ok = add_terminal("out_surface", surface_conn);
  ok |= add_terminal("out_displacement", disp_conn);
  if (!ok) return nullptr;

  // Upgrade the small standalone user graph BEFORE attaching `std_libs`.
  UpgradeMaterialXDocument(ctx.doc, version != nullptr);

  if (ctx.custom_libs.empty()) {
    ctx.doc->setDataLibrary(std_libs);
  } else {
    mx::DocumentPtr overlay = mx::createDocument();
    overlay->setDataLibrary(std_libs);
    for (const mx::ConstDocumentPtr& lib : ctx.custom_libs) {
      overlay->importLibrary(lib);
    }
    ctx.doc->setDataLibrary(overlay);
  }
  return std::move(ctx.doc);
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

std::shared_ptr<const PreparedMaterialX> PrepareMaterialXNetwork(
    const SdfPath& id, const HdMaterialNetwork2& network,
    const std::function<void(const TextureKey&)>& callback) {
  auto [surface_conn, surf_rank] =
      FindTerminalConnection(network, HdMaterialTerminalTokens->surface);
  auto [disp_conn, disp_rank] =
      FindTerminalConnection(network, HdMaterialTerminalTokens->displacement);
  if (!IsMaterialXTerminal(network, surface_conn)) surface_conn = nullptr;
  if (!IsMaterialXTerminal(network, disp_conn)) disp_conn = nullptr;
  if (!surface_conn && !disp_conn) return nullptr;
  try {
    mx::DocumentPtr doc = BuildMaterialXDocument(
        network, GetMaterialXStdLibraries(), surface_conn, disp_conn);
    if (!doc) return nullptr;
    return LowerMaterialXShaders(doc->getOutput("out_surface"),
                                 doc->getOutput("out_displacement"),
                                 disp_rank < surf_rank, callback);
  } catch (const std::exception& e) {
    TF_WARN("Failed to translate MaterialX network '%s': %s", id.GetText(),
            e.what());
    return nullptr;
  }
}

TranslatedMaterial TranslateMaterialXNetwork(
    const SdfPath& id, const PreparedMaterialX& prepared,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture,
    std::string_view variant) {
  TranslatedMaterial res;
  try {
    CompiledMaterialXResult compiled =
        InstantiateMaterialXShaders(prepared, variant, lookup_texture);
    res.bsdf = std::move(compiled.bsdf);
    res.mesh_attributes = std::move(compiled.mesh_attributes);
    res.displacement = std::move(compiled.displacement);
    if (compiled.radiance_texture) {
      mitsuba::ref<mitsuba::Object> radiance =
          std::move(compiled.radiance_texture);
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
  } catch (const std::exception& e) {
    TF_WARN("Failed to translate MaterialX network '%s': %s", id.GetText(),
            e.what());
  }
  return res;
}

std::optional<std::pair<TranslatedMaterial, bool>> TranslateUsdMaterialX(
    const UsdPrim& prim, UsdTimeCode time, std::string_view variant) {
  if (!prim) return std::nullopt;
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
  using namespace mitsuba;
  TranslatedMaterial trans =
      MI_INVOKE_VARIANT(variant, TranslateWithTextureCache, prim.GetPath(),
                        *prepared, keys, variant);
  if (trans.bsdf) trans.bsdf->set_id(prim.GetPath().GetAsString());
  return std::make_pair(std::move(trans), HasMaterialXDisplacement(*prepared));
}

PXR_NAMESPACE_CLOSE_SCOPE
