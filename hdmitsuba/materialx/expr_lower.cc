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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/ascii.h>
#include <absl/strings/strip.h>
#include <pxr/base/gf/color.h>
#include <pxr/base/gf/colorSpace.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>

#include "hdmitsuba/materialx/compiler_internal.h"
#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/program.h"
#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE
namespace mtlx_detail {
namespace {

mx::OutputPtr ResolveNodeGraphOutput(const mx::NodeGraphPtr& ng,
                                     const std::string& output_name) {
  if (!output_name.empty()) {
    if (mx::OutputPtr out = ng->getActiveOutput(output_name)) return out;
  }
  std::vector<mx::OutputPtr> outs = ng->getActiveOutputs();
  return outs.empty() ? nullptr : outs[0];
}

bool IsFourChannelType(std::string_view type) {
  return type == "color4" || type == "vector4";
}

bool IsValueType(std::string_view type) {
  return type == "float" || type == "integer" || type == "boolean" ||
         type == "vector2" || type == "vector3" || type == "color3" ||
         type == "vector4" || type == "color4";
}

bool IsGeomIntrinsic(std::string_view kind) {
  return kind == "texcoord" || kind == "position" || kind == "normal" ||
         kind == "tangent" || kind == "bitangent" || kind == "viewdirection" ||
         kind == "geomcolor";
}

uint8_t TypeChannels(std::string_view type) {
  if (type == "float" || type == "integer" || type == "boolean") return 1;
  if (type == "vector2") return 2;
  return IsFourChannelType(type) ? 4 : 3;
}

AddressMode ParseAddressMode(std::string_view mode) {
  if (mode == "clamp") return AddressMode::Clamp;
  if (mode == "mirror") return AddressMode::Mirror;
  if (mode == "constant") return AddressMode::Constant;
  if (mode == "black") return AddressMode::Black;
  return AddressMode::Periodic;
}

struct ResolvedInput {
  enum Kind { None, Connected, Value, GeomProp } kind = None;
  ScopedNode upstream;
  std::string output_name;
  mx::ValueElementPtr elem;
};

const NodeInfo& GetNodeInfo(GraphResolver& r, const mx::NodePtr& node) {
  auto [it, inserted] = r.node_info.try_emplace(node.get());
  if (inserted) {
    it->second.nodedef = node->getNodeDef();
    if (it->second.nodedef) {
      if (mx::InterfaceElementPtr impl =
              it->second.nodedef->getImplementation()) {
        it->second.graph_impl = impl->asA<mx::NodeGraph>();
      }
    }
  }
  return it->second;
}

ResolvedInput ResolveInput(GraphResolver& r, const ScopedNode& s,
                           const std::string& name);

ResolvedInput ResolvePort(GraphResolver& r, mx::PortElementPtr port,
                          const Frame* frame) {
  while (port && (port->hasNodeGraphString() ||
                  (port->hasOutputString() && !port->hasNodeName() &&
                   !port->hasInterfaceName()))) {
    port = port->getConnectedOutput();
    frame = nullptr;
  }
  if (!port) return {};
  if (port->hasInterfaceName()) {
    const std::string& iface = port->getInterfaceName();
    ResolvedInput res;
    if (frame) {
      res = ResolveInput(r, frame->call, iface);
    } else if (mx::GraphElementPtr g =
                   port->getAncestorOfType<mx::GraphElement>()) {
      if (mx::InputPtr iface_in = g->getActiveInput(iface)) {
        res = ResolvePort(r, std::move(iface_in), nullptr);
      }
    }
    if (res.kind != ResolvedInput::None) return res;
  }
  if (port->hasNodeName()) {
    bool ng_in =
        port->isA<mx::Input>() && port->getParent()->isA<mx::NodeGraph>();
    if (mx::NodePtr upstream =
            ng_in ? port->getDocument()->getNode(port->getNodeName())
                  : port->mx::PortElement::getConnectedNode()) {
      return {ResolvedInput::Connected,
              {std::move(upstream), ng_in ? nullptr : frame},
              port->getOutputString(),
              nullptr};
    }
  }
  if (auto in = port->asA<mx::Input>(); in && in->hasDefaultGeomPropString()) {
    return {ResolvedInput::GeomProp, {}, {}, std::move(in)};
  }
  if (port->hasValueString()) {
    return {ResolvedInput::Value, {}, {}, std::move(port)};
  }
  return {};
}

ResolvedInput ResolveOutput(GraphResolver& r, const ScopedNode& call,
                            const mx::OutputPtr& out) {
  if (!out) return {};
  const Frame*& slot = r.child_frames[{call.frame, call.node.get()}];
  if (!slot) {
    r.frames.push_back({call});
    slot = &r.frames.back();
  }
  return ResolvePort(r, out, slot);
}

ResolvedInput ResolveInput(GraphResolver& r, const ScopedNode& s,
                           const std::string& name) {
  if (mx::InputPtr input = s.node->getInput(name)) {
    if (ResolvedInput res = ResolvePort(r, std::move(input), s.frame);
        res.kind != ResolvedInput::None) {
      return res;
    }
  }
  const mx::NodeDefPtr& nd = GetNodeInfo(r, s.node).nodedef;
  mx::InputPtr def = nd ? nd->getActiveInput(name) : nullptr;
  if (!def) return {};
  if (def->hasDefaultGeomPropString()) {
    return {ResolvedInput::GeomProp, {}, {}, std::move(def)};
  }
  if (def->hasValueString()) {
    return {ResolvedInput::Value, {}, {}, std::move(def)};
  }
  return {};
}

ResolvedInput ResolveTerminalInput(GraphResolver& r, const ScopedNode& s,
                                   const std::string& name) {
  ResolvedInput res = ResolveInput(r, s, name);
  while (res.kind == ResolvedInput::Connected) {
    if (mx::NodeGraphPtr ng = GetNodeInfo(r, res.upstream.node).graph_impl) {
      res = ResolveOutput(r, res.upstream,
                          ResolveNodeGraphOutput(ng, res.output_name));
      continue;
    }
    const std::string& cat = NodeCategory(r, res.upstream.node);
    if (cat != "dot" && cat != "constant") break;
    res = ResolveInput(r, res.upstream, cat == "constant" ? "value" : "in");
  }
  return res;
}

mx::ValueElementPtr GetValueInput(GraphResolver& r, const ScopedNode& s,
                                  const std::string& name) {
  ResolvedInput res = ResolveTerminalInput(r, s, name);
  return res.kind == ResolvedInput::Value ? res.elem : nullptr;
}

bool IsAuthoredInput(const LowerCtx& ctx, const mx::ValueElementPtr& elem) {
  mx::ElementPtr parent = elem->getParent();
  return parent && parent->isA<mx::Node>() && parent->getParent() == ctx.doc;
}

ValueId DefaultTexcoord(LowerCtx& ctx, uint8_t ch = 2) {
  return ctx.b->Emit(mtlx::Op::GeomUV, {}, 0, ch);
}
ValueId PWorld(LowerCtx& ctx) { return ctx.b->Emit(mtlx::Op::GeomPosWorld); }
ValueId BWorld(LowerCtx& ctx) {
  return ctx.b->Emit(mtlx::Op::GeomBitangentWorld);
}

MatVal IdentityMatrix(LowerCtx& ctx, int n) {
  MatVal m;
  m.n = n;
  ValueId z = ctx.b->Zero(), o = ctx.b->One();
  for (int i = 0; i < 3; ++i) {
    ValueId row = ctx.b->Rgb(i == 0 ? o : z, i == 1 ? o : z, i == 2 ? o : z);
    m.rows[i] = n == 4 ? ctx.b->Vec4(row, z) : row;
  }
  m.rows[3] = ctx.b->Vec4(z, o);
  return m;
}

ValueId PackTuple(mtlx::IrBuilder& b, const ValueId* e, int n) {
  if (n == 1) return e[0];
  if (n == 2) return b.Vec2(e[0], e[1]);
  ValueId c3 = b.Rgb(e[0], e[1], e[2]);
  return n == 3 ? c3 : b.Vec4(c3, e[3]);
}

AuthoredOpaque RecordAuthoredTuple(LowerCtx& ctx,
                                   std::initializer_list<float> vals) {
  AuthoredOpaque a;
  a.n = static_cast<uint8_t>(vals.size());
  a.all_zero =
      std::all_of(vals.begin(), vals.end(), [](float x) { return x == 0.0f; });
  a.all_one =
      std::all_of(vals.begin(), vals.end(), [](float x) { return x == 1.0f; });
  if (a.all_zero || a.all_one) return a;
  int i = 0;
  for (float v : vals) a.idx[i++] = ctx.b->AddOpaqueValue(v);
  return a;
}

ValueId MaterializeAuthored(LowerCtx& ctx, const AuthoredOpaque& a) {
  if (a.all_zero) return ctx.b->Zero();
  if (a.all_one) return ctx.b->One();
  ValueId e[4];
  for (int i = 0; i < a.n; ++i) e[i] = ctx.b->OpaqueRef(a.idx[i]);
  return PackTuple(*ctx.b, e, a.n);
}

ValueId EmitConstTuple(mtlx::IrBuilder& b, std::initializer_list<float> vals) {
  ValueId e[4];
  int n = 0;
  for (float v : vals) e[n++] = b.Const(v);
  return PackTuple(b, e, n);
}

ValueId LowerLinearRemap(LowerCtx& ctx, const ScopedNode& node, ValueId ilo,
                         ValueId ihi, ValueId olo, ValueId ohi) {
  ValueId scale = ctx.b->Div(ctx.b->Sub(ohi, olo), ctx.b->Sub(ihi, ilo));
  return ctx.b->Fma(ctx.b->Sub(LowerInputExpr(ctx, node, "in"), ilo), scale,
                    olo);
}

// Maps MaterialX `cmlib` and USD `GfColorSpace` names to a canonical
// `GfColorSpace` token when conversion to linear Rec.709 is needed, or an empty
// token if `color_space` is empty, data/raw, or already linear Rec.709.
TfToken ResolveSourceColorSpace(LowerCtx& ctx, std::string_view color_space) {
  if (color_space.empty()) return {};
  std::string cs = absl::AsciiStrToLower(color_space);
  static const absl::NoDestructor<
      absl::flat_hash_map<std::string_view, TfToken>>
      kMtlxAliases({
          {"lin_rec709", GfColorSpaceNames->LinearRec709},
          {"lin_srgb", GfColorSpaceNames->LinearRec709},
          {"none", GfColorSpaceNames->Raw},
          {"acescg", GfColorSpaceNames->LinearAP1},
          {"g22_ap1", GfColorSpaceNames->G22AP1},
          {"g22_rec709", GfColorSpaceNames->G22Rec709},
          {"g18_rec709", GfColorSpaceNames->G18Rec709},
          // Use token string directly: some OpenUSD 26.05 snapshots (e.g.
          // Houdini 22.0) omit the `GfColorSpaceNames->G24Rec709` constant.
          {"rec709_display", TfToken("g24_rec709_scene")},
          {"srgb", GfColorSpaceNames->SRGBRec709},
          {"srgb_texture", GfColorSpaceNames->SRGBRec709},
          {"lin_adobergb", GfColorSpaceNames->LinearAdobeRGB},
          {"adobergb", GfColorSpaceNames->G22AdobeRGB},
          {"srgb_displayp3", GfColorSpaceNames->SRGBP3D65},
          {"lin_displayp3", GfColorSpaceNames->LinearP3D65},
      });
  auto it = kMtlxAliases->find(cs);
  TfToken token = it != kMtlxAliases->end() ? it->second : TfToken(cs);
  if (token == GfColorSpaceNames->LinearRec709 ||
      token == GfColorSpaceNames->Raw || token == GfColorSpaceNames->Data ||
      token == GfColorSpaceNames->Identity ||
      token == GfColorSpaceNames->Unknown) {
    return {};
  }
  if (!GfColorSpace::IsValid(token)) {
    WarnUnsupported(ctx, "Unsupported MaterialX color space '" +
                             std::string(color_space) +
                             "'; treating as linear Rec.709.");
    return {};
  }
  return token;
}

GfVec3f ToLinearRec709(LowerCtx& ctx, GfVec3f rgb_val,
                       std::string_view color_space) {
  if (rgb_val == GfVec3f(0.0f)) return rgb_val;
  TfToken src_cs = ResolveSourceColorSpace(ctx, color_space);
  if (src_cs.IsEmpty()) return rgb_val;
  rgb_val = GfColorSpace(GfColorSpaceNames->LinearRec709)
                .Convert(GfColorSpace(src_cs), rgb_val)
                .GetRGB();
  // Snap 1-ULP float32 transfer/matrix rounding at `(1, 1, 1)` (e.g. sRGB
  // `opacity = (1, 1, 1)` under `UsdColorSpaceAPI`) so `EmitConstTuple` /
  // `RecordAuthoredTuple` folds opaque white to `One()`, while preserving
  // spaces where `(1, 1, 1)` does not map to Rec.709 white.
  if (GfIsClose(rgb_val, GfVec3f(1.0f), 1e-5f)) rgb_val = GfVec3f(1.0f);
  return rgb_val;
}

ValueId MeshAttr(LowerCtx& ctx, std::string_view name, int dim, ValueId def,
                 uint8_t out_ch = 0) {
  std::pair<std::string, int> req{std::string(name), dim};
  auto it = std::find(ctx.attr_requests.begin(), ctx.attr_requests.end(), req);
  if (it == ctx.attr_requests.end()) {
    if (ctx.attr_requests.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many attributes");
    }
    ctx.val_prog.attr_names.push_back(MeshAttributeName(name, dim));
    it = ctx.attr_requests.insert(ctx.attr_requests.end(), std::move(req));
  }
  uint8_t read_ch = static_cast<uint8_t>(dim);
  uint8_t ch =
      out_ch ? out_ch : std::max<uint8_t>(ctx.b->Channels(def), read_ch);
  return ctx.b->Emit(mtlx::Op::Attr, {def},
                     static_cast<uint16_t>(it - ctx.attr_requests.begin()), ch,
                     read_ch);
}

ValueId LowerNodeOutputExpr(LowerCtx& ctx, const ScopedNode& node,
                            const std::string& output_name);
MatVal LowerMatrixNode(LowerCtx& ctx, const ScopedNode& node,
                       const std::string& output_name);

ValueId LowerGeomIntrinsic(LowerCtx& ctx, std::string_view kind,
                           std::string_view space, std::string_view type,
                           bool non_zero_index) {
  if (non_zero_index && (kind == "texcoord" || kind == "tangent" ||
                         kind == "bitangent" || kind == "geomcolor")) {
    WarnUnsupported(ctx, "Unsupported MaterialX '" + std::string(kind) +
                             "' index; using index 0.");
  }
  if (space.empty()) space = kind == "viewdirection" ? "world" : "object";
  auto warn_world_space = [&] {
    if (space != "world") {
      WarnUnsupported(ctx, "Unsupported MaterialX " + std::string(kind) +
                               " space '" + std::string(space) +
                               "'; evaluating in world space.");
    }
  };
  if (kind == "texcoord") {
    return DefaultTexcoord(ctx, type == "vector3" ? 3 : 2);
  }
  if (kind == "geomcolor") {
    ValueId color =
        MeshAttr(ctx, "displayColor", type == "float" ? 1 : 3, ctx.b->Zero());
    // The alpha channel of `geomcolor` is the display opacity.
    return type == "color4" ? ctx.b->Vec4(color, MeshAttr(ctx, "displayOpacity",
                                                          1, ctx.b->One()))
                            : color;
  }
  if (kind == "position") {
    if (space == "object" || space == "model") {
      return MeshAttr(ctx, "__p_object", 3, PWorld(ctx));
    }
    warn_world_space();
    return PWorld(ctx);
  }
  if (kind == "normal") {
    if (space == "object" || space == "model") {
      return ctx.b->Emit(mtlx::Op::Normalize,
                         {MeshAttr(ctx, "__n_object", 3, NWorld(ctx))}, 3);
    }
    if (space == "tangent") return ctx.b->ConstRgb(0.0, 0.0, 1.0);
    warn_world_space();
    return NWorld(ctx);
  }
  if (kind == "viewdirection") {
    warn_world_space();
    return ctx.b->Emit(mtlx::Op::GeomIncidentDir);
  }
  bool bitangent = kind == "bitangent";
  if (space == "tangent") {
    return bitangent ? ctx.b->ConstRgb(0.0, 1.0, 0.0)
                     : ctx.b->ConstRgb(1.0, 0.0, 0.0);
  }
  warn_world_space();
  return bitangent ? BWorld(ctx) : TWorld(ctx);
}

template <typename DefaultFn>
ValueId LowerGeomProp(LowerCtx& ctx, const std::string& prop_name,
                      std::string_view type, DefaultFn&& default_fn) {
  uint8_t ch = TypeChannels(type);
  if (mx::GeomPropDefPtr gpd = ctx.doc->getGeomPropDef(prop_name)) {
    const std::string& gp = gpd->getGeomProp();
    if (IsGeomIntrinsic(gp)) {
      std::string_view idx = gpd->getIndex();
      ValueId e = LowerGeomIntrinsic(ctx, gp, gpd->getSpace(), type,
                                     !idx.empty() && idx != "0");
      if (ch == 1) return ctx.b->Chan(e, 0);
      if (ch == 2 && ctx.b->Channels(e) > 2) {
        return ctx.b->Vec2(ctx.b->Chan(e, 0), ctx.b->Chan(e, 1));
      }
      return e;
    }
  }
  return MeshAttr(ctx, prop_name, ch == 1 ? 1 : 3, default_fn(), ch);
}

ValueId LowerValueElement(LowerCtx& ctx, const mx::ValueElementPtr& elem,
                          ValueId default_val, bool is_authored) {
  const std::string& type = elem->getType();
  if (type == "boolean") {
    const std::string& s = elem->getValueString();
    return (s == "true" || s == "1") ? ctx.b->One() : ctx.b->Zero();
  }
  mx::ValuePtr val = elem->getValue();
  if (!val) return default_val;
  // Integers and booleans select code paths (e.g. switch branches) and are
  // baked as constants.
  if (type == "integer") return ctx.b->Const(val->asA<int>());

  auto emit = [&](std::initializer_list<float> vals) -> ValueId {
    if (!is_authored) return EmitConstTuple(*ctx.b, vals);
    auto [it, inserted] = ctx.authored_opaque_val.try_emplace(elem.get());
    if (inserted) it->second = RecordAuthoredTuple(ctx, vals);
    return MaterializeAuthored(ctx, it->second);
  };

  if (type == "float") return emit({val->asA<float>()});
  if (type == "color3") {
    mx::Color3 v = val->asA<mx::Color3>();
    GfVec3f c =
        ToLinearRec709(ctx, {v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return emit({c[0], c[1], c[2]});
  }
  if (type == "color4") {
    mx::Color4 v = val->asA<mx::Color4>();
    GfVec3f c =
        ToLinearRec709(ctx, {v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return emit({c[0], c[1], c[2], v[3]});
  }
  if (type == "vector2") {
    mx::Vector2 v = val->asA<mx::Vector2>();
    return emit({v[0], v[1]});
  }
  if (type == "vector3") {
    mx::Vector3 v = val->asA<mx::Vector3>();
    return emit({v[0], v[1], v[2]});
  }
  if (type == "vector4") {
    mx::Vector4 v = val->asA<mx::Vector4>();
    return emit({v[0], v[1], v[2], v[3]});
  }
  return default_val;
}

ValueId LowerResolvedInput(LowerCtx& ctx, const ResolvedInput& r,
                           ValueId default_val) {
  if (r.kind == ResolvedInput::Connected) {
    NodeOutKey key{r.upstream.frame, r.upstream.node.get(), r.output_name};
    auto it = ctx.node_output_val.find(key);
    if (it == ctx.node_output_val.end()) {
      ValueId v = LowerNodeOutputExpr(ctx, r.upstream, r.output_name);
      it = ctx.node_output_val.emplace(std::move(key), v).first;
    }
    return it->second;
  }
  if (r.kind == ResolvedInput::GeomProp) {
    return LowerGeomProp(ctx,
                         r.elem->asA<mx::Input>()->getDefaultGeomPropString(),
                         r.elem->getType(), [&] { return ctx.b->Zero(); });
  }
  if (r.kind != ResolvedInput::Value) return default_val;
  if (!IsAuthoredInput(ctx, r.elem)) {
    return LowerValueElement(ctx, r.elem, default_val, /*is_authored=*/false);
  }
  if (auto it = ctx.authored_opaque_val.find(r.elem.get());
      it != ctx.authored_opaque_val.end()) {
    return MaterializeAuthored(ctx, it->second);
  }
  return LowerValueElement(ctx, r.elem, default_val, /*is_authored=*/true);
}

std::optional<int> ConstIntInput(LowerCtx& ctx, const ScopedNode& node,
                                 const std::string& name) {
  ResolvedInput r = ResolveInput(ctx.resolver, node, name);
  if (r.kind == ResolvedInput::None) return std::nullopt;
  ValueId v = LowerResolvedInput(ctx, r, ctx.b->Zero());
  double d = 0.0;
  if (ctx.b->IsConstScalar(v, &d)) {
    return static_cast<int>(std::clamp<double>(
        d, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
  }
  return std::nullopt;
}

ValueId LowerIfCondition(LowerCtx& ctx, const ScopedNode& node,
                         std::string_view cat) {
  ValueId v1 = LowerInputExpr(ctx, node, "value1");
  ValueId v2 = LowerInputExpr(ctx, node, "value2");
  if (cat == "ifgreater") return ctx.b->Emit(mtlx::Op::Lt, {v2, v1});
  if (cat == "ifgreatereq") return ctx.b->Emit(mtlx::Op::Le, {v2, v1});
  return ctx.b->Emit(mtlx::Op::Eq, {v1, v2});
}

MatVal LowerMatrixTranspose(mtlx::IrBuilder& b, const MatVal& m) {
  MatVal out;
  out.n = m.n;
  for (int i = 0; i < m.n; ++i) {
    uint16_t c = static_cast<uint16_t>(i);
    ValueId r =
        b.Rgb(b.Chan(m.rows[0], c), b.Chan(m.rows[1], c), b.Chan(m.rows[2], c));
    out.rows[i] = m.n == 4 ? b.Vec4(r, b.Chan(m.rows[3], c)) : r;
  }
  return out;
}

MatVal LowerMatrixMul(mtlx::IrBuilder& b, const MatVal& a, const MatVal& bm) {
  MatVal out;
  out.n = a.n;
  for (int i = 0; i < a.n; ++i) {
    ValueId acc = b.Mul(b.Chan(a.rows[i], 0), bm.rows[0]);
    for (int k = 1; k < a.n; ++k) {
      acc = b.Fma(b.Chan(a.rows[i], static_cast<uint16_t>(k)), bm.rows[k], acc);
    }
    out.rows[i] = acc;
  }
  return out;
}

struct Mat4Minors {
  ValueId a[4][4];
  ValueId s[6];
  ValueId c[6];
  ValueId det;
};

Mat4Minors ComputeMat4Minors(mtlx::IrBuilder& b, const MatVal& m) {
  Mat4Minors r;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      r.a[i][j] = b.Chan(m.rows[i], static_cast<uint16_t>(j));
    }
  }
  auto det2 = [&](int r0, int r1, int c0, int c1) {
    return b.Sub(b.Mul(r.a[r0][c0], r.a[r1][c1]),
                 b.Mul(r.a[r1][c0], r.a[r0][c1]));
  };
  r.s[0] = det2(0, 1, 0, 1);
  r.s[1] = det2(0, 1, 0, 2);
  r.s[2] = det2(0, 1, 0, 3);
  r.s[3] = det2(0, 1, 1, 2);
  r.s[4] = det2(0, 1, 1, 3);
  r.s[5] = det2(0, 1, 2, 3);
  r.c[5] = det2(2, 3, 2, 3);
  r.c[4] = det2(2, 3, 1, 3);
  r.c[3] = det2(2, 3, 1, 2);
  r.c[2] = det2(2, 3, 0, 3);
  r.c[1] = det2(2, 3, 0, 2);
  r.c[0] = det2(2, 3, 0, 1);
  r.det = b.Add(b.Sub(b.Add(b.Sub(b.Mul(r.s[0], r.c[5]), b.Mul(r.s[1], r.c[4])),
                            b.Mul(r.s[2], r.c[3])),
                      b.Sub(b.Mul(r.s[4], r.c[1]), b.Mul(r.s[3], r.c[2]))),
                b.Mul(r.s[5], r.c[0]));
  return r;
}

ValueId LowerMatrixDet(mtlx::IrBuilder& b, const MatVal& m) {
  if (m.n == 3) {
    ValueId c0 = b.Emit(mtlx::Op::Cross, {m.rows[1], m.rows[2]});
    return b.Emit(mtlx::Op::Dot, {m.rows[0], c0}, 3);
  }
  return ComputeMat4Minors(b, m).det;
}

MatVal LowerMatrixInverse(mtlx::IrBuilder& b, const MatVal& m) {
  MatVal out;
  out.n = m.n;
  if (m.n == 3) {
    ValueId c0 = b.Emit(mtlx::Op::Cross, {m.rows[1], m.rows[2]});
    ValueId c1 = b.Emit(mtlx::Op::Cross, {m.rows[2], m.rows[0]});
    ValueId c2 = b.Emit(mtlx::Op::Cross, {m.rows[0], m.rows[1]});
    // `Op::Div` uses `MxSafeDiv`, returning 0 for singular matrices (`det ==
    // 0`).
    ValueId inv_det = b.Div(b.One(), b.Emit(mtlx::Op::Dot, {m.rows[0], c0}, 3));
    for (int i = 0; i < 3; ++i) {
      uint16_t ci = static_cast<uint16_t>(i);
      out.rows[i] =
          b.Mul(b.Rgb(b.Chan(c0, ci), b.Chan(c1, ci), b.Chan(c2, ci)), inv_det);
    }
    return out;
  }
  Mat4Minors d = ComputeMat4Minors(b, m);
  ValueId inv_det = b.Div(b.One(), d.det);
  auto cof3 = [&](ValueId x0, ValueId y0, ValueId x1, ValueId y1, ValueId x2,
                  ValueId y2) {
    return b.Add(b.Sub(b.Mul(x0, y0), b.Mul(x1, y1)), b.Mul(x2, y2));
  };
  auto row = [&](ValueId e0, ValueId e1, ValueId e2, ValueId e3) {
    return b.Mul(b.Vec4(b.Rgb(e0, e1, e2), e3), inv_det);
  };
  out.rows[0] =
      row(cof3(d.a[1][1], d.c[5], d.a[1][2], d.c[4], d.a[1][3], d.c[3]),
          b.Neg(cof3(d.a[0][1], d.c[5], d.a[0][2], d.c[4], d.a[0][3], d.c[3])),
          cof3(d.a[3][1], d.s[5], d.a[3][2], d.s[4], d.a[3][3], d.s[3]),
          b.Neg(cof3(d.a[2][1], d.s[5], d.a[2][2], d.s[4], d.a[2][3], d.s[3])));
  out.rows[1] =
      row(b.Neg(cof3(d.a[1][0], d.c[5], d.a[1][2], d.c[2], d.a[1][3], d.c[1])),
          cof3(d.a[0][0], d.c[5], d.a[0][2], d.c[2], d.a[0][3], d.c[1]),
          b.Neg(cof3(d.a[3][0], d.s[5], d.a[3][2], d.s[2], d.a[3][3], d.s[1])),
          cof3(d.a[2][0], d.s[5], d.a[2][2], d.s[2], d.a[2][3], d.s[1]));
  out.rows[2] =
      row(cof3(d.a[1][0], d.c[4], d.a[1][1], d.c[2], d.a[1][3], d.c[0]),
          b.Neg(cof3(d.a[0][0], d.c[4], d.a[0][1], d.c[2], d.a[0][3], d.c[0])),
          cof3(d.a[3][0], d.s[4], d.a[3][1], d.s[2], d.a[3][3], d.s[0]),
          b.Neg(cof3(d.a[2][0], d.s[4], d.a[2][1], d.s[2], d.a[2][3], d.s[0])));
  out.rows[3] =
      row(b.Neg(cof3(d.a[1][0], d.c[3], d.a[1][1], d.c[1], d.a[1][2], d.c[0])),
          cof3(d.a[0][0], d.c[3], d.a[0][1], d.c[1], d.a[0][2], d.c[0]),
          b.Neg(cof3(d.a[3][0], d.s[3], d.a[3][1], d.s[1], d.a[3][2], d.s[0])),
          cof3(d.a[2][0], d.s[3], d.a[2][1], d.s[1], d.a[2][2], d.s[0]));
  return out;
}

MatVal LowerResolvedMatrix(LowerCtx& ctx, const ResolvedInput& r, int n) {
  if (r.kind == ResolvedInput::Connected) {
    return LowerMatrixNode(ctx, r.upstream, r.output_name);
  }
  mx::ValuePtr val =
      r.kind == ResolvedInput::Value ? r.elem->getValue() : nullptr;
  if (!val) return IdentityMatrix(ctx, n);
  const std::string& t = r.elem->getType();
  if (t != "matrix33" && t != "matrix44") return IdentityMatrix(ctx, n);
  int dim = t == "matrix44" ? 4 : 3;
  auto for_each_row = [&](auto&& fn) {
    if (dim == 4) {
      mx::Matrix44 m = val->asA<mx::Matrix44>();
      for (int i = 0; i < 4; ++i) fn(i, {m[i][0], m[i][1], m[i][2], m[i][3]});
    } else {
      mx::Matrix33 m = val->asA<mx::Matrix33>();
      for (int i = 0; i < 3; ++i) fn(i, {m[i][0], m[i][1], m[i][2]});
    }
  };
  MatVal out;
  out.n = dim;
  if (IsAuthoredInput(ctx, r.elem)) {
    auto [it, inserted] = ctx.authored_opaque_matrix.try_emplace(r.elem.get());
    AuthoredMatrix& am = it->second;
    if (inserted) {
      am.n = dim;
      for_each_row([&](int i, std::initializer_list<float> row) {
        am.rows[i] = RecordAuthoredTuple(ctx, row);
      });
    }
    for (int i = 0; i < dim; ++i) {
      out.rows[i] = MaterializeAuthored(ctx, am.rows[i]);
    }
    return out;
  }
  for_each_row([&](int i, std::initializer_list<float> row) {
    out.rows[i] = EmitConstTuple(*ctx.b, row);
  });
  return out;
}

MatVal LowerMatrixInput(LowerCtx& ctx, const ScopedNode& node,
                        const std::string& input_name) {
  return LowerResolvedMatrix(
      ctx, ResolveInput(ctx.resolver, node, input_name),
      InputType(ctx.resolver, node.node, input_name) == "matrix44" ? 4 : 3);
}

MatVal LowerMatrixNode(LowerCtx& ctx, const ScopedNode& node,
                       const std::string& output_name) {
  NodeOutKey key{node.frame, node.node.get(), output_name};
  if (auto it = ctx.node_matrix_val.find(key);
      it != ctx.node_matrix_val.end()) {
    return it->second;
  }
  int n = node.node->getType() == "matrix44" ? 4 : 3;
  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  MatVal res = IdentityMatrix(ctx, n);
  if (mx::NodeGraphPtr ng = GetNodeInfo(ctx.resolver, node.node).graph_impl) {
    mx::OutputPtr out = ResolveNodeGraphOutput(ng, output_name);
    res = LowerResolvedMatrix(ctx, ResolveOutput(ctx.resolver, node, out),
                              out ? (out->getType() == "matrix44" ? 4 : 3) : n);
  } else if (cat == "constant") {
    res = LowerMatrixInput(ctx, node, "value");
  } else if (cat == "dot") {
    res = LowerMatrixInput(ctx, node, "in");
  } else if (cat == "creatematrix") {
    // The rows of a 4x4 matrix are either `vector4` or `vector3` values, which
    // are extended by the fourth column of the identity.
    bool extend =
        n == 4 && InputType(ctx.resolver, node.node, "in1") != "vector4";
    for (int i = 0; i < n; ++i) {
      ValueId row = LowerInputExpr(ctx, node, "in" + std::to_string(i + 1));
      res.rows[i] =
          extend ? ctx.b->Vec4(row, i == 3 ? ctx.b->One() : ctx.b->Zero())
                 : row;
    }
  } else if (cat == "transpose") {
    res = LowerMatrixTranspose(*ctx.b, LowerMatrixInput(ctx, node, "in"));
  } else if (cat == "invertmatrix") {
    res = LowerMatrixInverse(*ctx.b, LowerMatrixInput(ctx, node, "in"));
  } else if (cat == "add" || cat == "subtract") {
    MatVal m1 = LowerMatrixInput(ctx, node, "in1");
    bool is_add = cat == "add";
    if (InputType(ctx.resolver, node.node, "in2") == "float") {
      ValueId s = LowerInputExpr(ctx, node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] =
            is_add ? ctx.b->Add(m1.rows[i], s) : ctx.b->Sub(m1.rows[i], s);
      }
    } else {
      MatVal m2 = LowerMatrixInput(ctx, node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = is_add ? ctx.b->Add(m1.rows[i], m2.rows[i])
                             : ctx.b->Sub(m1.rows[i], m2.rows[i]);
      }
    }
  } else if (cat == "multiply" || cat == "divide") {
    MatVal m1 = LowerMatrixInput(ctx, node, "in1");
    MatVal m2 = LowerMatrixInput(ctx, node, "in2");
    if (cat == "divide") m2 = LowerMatrixInverse(*ctx.b, m2);
    res = LowerMatrixMul(*ctx.b, m1, m2);
  } else if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    ValueId cond = LowerIfCondition(ctx, node, cat);
    if (ctx.b->IsOne(cond) || ctx.b->IsZero(cond)) {
      res = LowerMatrixInput(ctx, node, ctx.b->IsOne(cond) ? "in1" : "in2");
    } else {
      MatVal m1 = LowerMatrixInput(ctx, node, "in1"),
             m2 = LowerMatrixInput(ctx, node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = ctx.b->Select(cond, m1.rows[i], m2.rows[i]);
      }
    }
  } else {
    WarnUnsupported(ctx, "Unsupported MaterialX matrix node category '" + cat +
                             "'; using identity.");
  }

  ctx.node_matrix_val[std::move(key)] = res;
  return res;
}

// Lowers an `<image>` node into a symbolic `Op::SampleInput` referencing
// `ctx.images[imm]`, resolved later against `lookup_texture`.
ValueId LowerImage(LowerCtx& ctx, const ScopedNode& node) {
  mx::ValueElementPtr file_in = GetValueInput(ctx.resolver, node, "file");
  if (!file_in || file_in->getValueString().empty()) {
    return LowerInputExpr(ctx, node, "default");
  }

  const std::string& type = node.node->getType();
  AddressMode u_mode = ParseAddressMode(
      GetNodeInputString(ctx.resolver, node, "uaddressmode", "periodic"));
  AddressMode v_mode = ParseAddressMode(
      GetNodeInputString(ctx.resolver, node, "vaddressmode", "periodic"));
  bool mixed_wrap = u_mode != v_mode || IsOobMode(u_mode) || IsOobMode(v_mode);
  if (!GetNodeInputString(ctx.resolver, node, "layer").empty()) {
    WarnUnsupported(
        ctx, "Unsupported MaterialX image 'layer'; reading default layer.");
  }
  if (!GetNodeInputString(ctx.resolver, node, "framerange").empty()) {
    WarnUnsupported(
        ctx, "Unsupported MaterialX image 'framerange'; reading static file.");
  }

  TextureKey key;
  key.filename =
      ArGetResolver().Resolve(file_in->getValueString()).GetPathString();
  // Keep raw filename when unresolved so `TextureCache::Preload` logs missing
  // files (`TF_WARN`) and `<UDIM>` tiles (`TF_DEBUG(HDMITSUBA_MATERIAL)`).
  if (key.filename.empty()) key.filename = file_in->getValueString();
  // Images load raw: like MaterialX, an image without a color space is in the
  // working color space, while `key.color_space` converts all others at load
  // time in `TextureCache`.
  key.raw = true;
  if (type == "color3" || type == "color4") {
    std::string cs = file_in->getActiveColorSpace();
    if (cs.empty()) {
      mx::InputPtr img_file = node.node->getInput("file");
      cs = img_file ? img_file->getActiveColorSpace()
                    : node.node->getActiveColorSpace();
    }
    key.color_space = ResolveSourceColorSpace(ctx, cs).GetString();
  }
  if (type == "float") key.channel = 0;
  // Mitsuba's `bitmap` supports a single `wrap_mode`; mixed or out-of-bounds
  // (`"constant"` / `"black"`) address modes clamp at the border and remap UVs
  // during resolution.
  key.wrap_mode = (mixed_wrap || u_mode == AddressMode::Clamp) ? "clamp"
                  : u_mode == AddressMode::Mirror              ? "mirror"
                                                               : "repeat";
  key.filter_type = GetNodeInputString(ctx.resolver, node, "filtertype",
                                       "linear") == "closest"
                        ? "nearest"
                        : "bilinear";

  ValueId def_tc = DefaultTexcoord(ctx);
  ValueId tc = LowerInputExpr(ctx, node, "texcoord", def_tc);
  ValueId def_val = LowerInputExpr(ctx, node, "default");
  bool default_uv = !mixed_wrap && tc == def_tc;

  SymbolicImage img{std::move(key),
                    IsFourChannelType(type),
                    type == "vector2",
                    default_uv,
                    u_mode,
                    v_mode};
  auto it = std::find(ctx.images.begin(), ctx.images.end(), img);
  if (it == ctx.images.end()) {
    if (ctx.images.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many texture inputs");
    }
    it = ctx.images.insert(ctx.images.end(), std::move(img));
  }
  uint16_t idx = static_cast<uint16_t>(it - ctx.images.begin());
  uint8_t ch = TypeChannels(type);
  if (default_uv) {
    return ctx.b->Emit(mtlx::Op::SampleInput, {def_val}, idx, ch);
  }
  return ctx.b->Emit(mtlx::Op::SampleInput, {tc, def_val}, idx, ch);
}

ValueId LowerHeightToNormal(LowerCtx& ctx, const ScopedNode& node) {
  ValueId scale = LowerInputExpr(ctx, node, "scale", ctx.b->One());
  if (ctx.b->IsZero(scale)) return ctx.b->ConstRgb(0.5, 0.5, 1.0);

  // Lower `in` (and `texcoord` if non-default) into an isolated `PendingRegion`
  // so `Op::HeightToNormal` can evaluate the subgraph at perturbed surface
  // interactions. Uniform subexpressions are hoisted back into the shared
  // uniform band by `PruneAndSchedule`.
  PendingRegion reg;
  mtlx::IrBuilder sub_builder(reg.nodes, reg.extra_args, ctx.val_prog.consts,
                              ctx.val_prog.opaques);
  mtlx::IrBuilder* saved_b = std::exchange(ctx.b, &sub_builder);
  auto saved_out = std::exchange(ctx.node_output_val, {});
  auto saved_mat = std::exchange(ctx.node_matrix_val, {});

  ValueId h = LowerInputExpr(ctx, node, "in");
  std::optional<uint16_t> sub_idx;
  if (sub_builder.IsVarying(h)) {
    ValueId def_tc = DefaultTexcoord(ctx);
    ValueId tc = LowerInputExpr(ctx, node, "texcoord", def_tc);
    bool has_tc = tc != def_tc;
    reg.root = has_tc ? ctx.b->Rgb(ctx.b->Chan(h, 0), ctx.b->Chan(tc, 0),
                                   ctx.b->Chan(tc, 1))
                      : ctx.b->Chan(h, 0);
    reg.has_texcoord = has_tc ? 1 : 0;
    if (ctx.pending_regions.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many heighttonormal nodes");
    }
    sub_idx = static_cast<uint16_t>(ctx.pending_regions.size());
    ctx.pending_regions.push_back(std::move(reg));
  }

  ctx.b = saved_b;
  ctx.node_output_val = std::move(saved_out);
  ctx.node_matrix_val = std::move(saved_mat);
  return sub_idx ? ctx.b->Emit(mtlx::Op::HeightToNormal, {scale}, *sub_idx)
                 : ctx.b->ConstRgb(0.5, 0.5, 1.0);
}

std::optional<ValueId> LowerChannelNode(LowerCtx& ctx, const ScopedNode& node,
                                        const std::string& cat,
                                        const std::string& type) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "constant") return in("value");
  if (cat == "dot") return in("in");
  if (cat == "extract") {
    int idx = ConstIntInput(ctx, node, "index").value_or(0);
    std::string_view in_type = InputType(ctx.resolver, node.node, "in");
    if (in_type == "matrix33" || in_type == "matrix44") {
      MatVal mat = LowerMatrixInput(ctx, node, "in");
      return mat.rows[std::clamp(idx, 0, mat.n - 1)];
    }
    return b.Chan(in("in"), static_cast<uint16_t>(
                                std::clamp(idx, 0, TypeChannels(in_type) - 1)));
  }
  if (cat == "combine2") {
    ValueId i1 = in("in1"), i2 = in("in2");
    if (!IsFourChannelType(type)) return b.Vec2(i1, i2);
    // A 3-channel value and a float, or two `vector2` values.
    if (InputType(ctx.resolver, node.node, "in1") != "vector2") {
      return b.Vec4(i1, i2);
    }
    return b.Vec4(b.Rgb(b.Chan(i1, 0), b.Chan(i1, 1), b.Chan(i2, 0)),
                  b.Chan(i2, 1));
  }
  if (cat == "combine3") {
    ValueId i1 = in("in1"), i2 = in("in2"), i3 = in("in3");
    return b.Rgb(i1, i2, i3);
  }
  if (cat == "combine4") {
    ValueId i1 = in("in1"), i2 = in("in2"), i3 = in("in3"), i4 = in("in4");
    return b.Vec4(b.Rgb(i1, i2, i3), i4);
  }
  return std::nullopt;
}

std::optional<ValueId> LowerMathNode(LowerCtx& ctx, const ScopedNode& node,
                                     const std::string& cat) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "multiply") {
    ValueId i1, i2;
    if (GetValueInput(ctx.resolver, node, "in2") &&
        !GetValueInput(ctx.resolver, node, "in1")) {
      i2 = in("in2", b.One());
      if (b.IsZero(i2)) return b.Zero();
      i1 = in("in1");
    } else {
      i1 = in("in1");
      if (b.IsZero(i1)) return b.Zero();
      i2 = in("in2", b.One());
    }
    return b.Mul(i1, i2);
  }
  if (cat == "divide") {
    ValueId i1 = in("in1");
    return b.IsZero(i1) ? b.Zero() : b.Div(i1, in("in2", b.One()));
  }
  if (cat == "mix") {
    return LowerMixVal(
        ctx, in("mix"), [&] { return in("fg"); }, [&] { return in("bg"); });
  }
  if (cat == "clamp") {
    ValueId x = in("in"), lo = in("low"), hi = in("high", b.One());
    return b.Emit(mtlx::Op::Clip, {x, lo, hi});
  }
  if (cat == "invert") {
    ValueId amt = in("amount", b.One()), x = in("in");
    return b.Sub(amt, x);
  }
  if (cat == "sign") {
    ValueId x = in("in");
    return b.Sub(b.Emit(mtlx::Op::Lt, {b.Zero(), x}),
                 b.Emit(mtlx::Op::Lt, {x, b.Zero()}));
  }
  if (cat == "modulo") {
    ValueId x = in("in1"), y = in("in2", b.One());
    if (b.IsOne(y)) return b.Emit(mtlx::Op::Fract, {x});
    return b.Sub(x, b.Mul(y, b.Emit(mtlx::Op::Floor, {b.Div(x, y)})));
  }
  if (cat == "remap") {
    ValueId ilo = in("inlow"), olo = in("outlow");
    return LowerLinearRemap(ctx, node, ilo, in("inhigh", b.One()), olo,
                            in("outhigh", b.One()));
  }
  return std::nullopt;
}

std::optional<ValueId> LowerLogicNode(LowerCtx& ctx, const ScopedNode& node,
                                      const std::string& cat,
                                      const std::string& type) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    ValueId cond = LowerIfCondition(ctx, node, cat);
    if (type == "boolean") return cond;
    if (b.IsOne(cond)) return in("in1");
    if (b.IsZero(cond)) return in("in2");
    ValueId i1 = in("in1"), i2 = in("in2");
    return b.Select(cond, i1, i2);
  }
  if (cat == "and" || cat == "or") {
    auto to_bool = [&](ValueId e) {
      return (b.IsZero(e) || b.IsOne(e)) ? e : b.Select(e, b.One(), b.Zero());
    };
    ValueId b1 = to_bool(in("in1")), b2 = to_bool(in("in2"));
    if (cat == "and") return b.Mul(b1, b2);
    return (b.IsOne(b1) || b.IsOne(b2)) ? b.One()
           : b.IsZero(b1)               ? b2
           : b.IsZero(b2)               ? b1
                                        : b.Emit(mtlx::Op::Max, {b1, b2});
  }
  if (cat == "not") return b.Not(in("in"));
  return std::nullopt;
}

std::optional<ValueId> LowerVectorNode(LowerCtx& ctx, const ScopedNode& node,
                                       const std::string& cat) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "normalize" || cat == "magnitude" || cat == "dotproduct") {
    std::string_view in_type =
        InputType(ctx.resolver, node.node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_imm = in_type == "vector2" ? 2 : in_type == "vector4" ? 4 : 3;
    if (cat == "normalize") {
      return b.Emit(mtlx::Op::Normalize, {in("in")}, vec_imm);
    }
    if (cat == "magnitude") {
      ValueId v = in("in");
      return b.Emit(mtlx::Op::Sqrt, {b.Emit(mtlx::Op::Dot, {v, v}, vec_imm)});
    }
    ValueId i1 = in("in1"), i2 = in("in2");
    return b.Emit(mtlx::Op::Dot, {i1, i2}, vec_imm);
  }
  if (cat == "rotate3d") {
    ValueId x = in("in"), amt = in("amount");
    if (b.IsZero(amt)) return x;
    return b.Emit(mtlx::Op::Rotate3D,
                  {x, amt, in("axis", b.ConstRgb(0.0, 1.0, 0.0))});
  }
  if (cat == "normalmap") {
    std::string space =
        GetNodeInputString(ctx.resolver, node, "space", "tangent");
    if (space != "tangent") {
      WarnUnsupported(ctx, "Unsupported MaterialX normalmap space '" + space +
                               "'; evaluating in tangent space.");
    }
    ValueId scale = in("scale", b.One());
    // A zero `scale` keeps only the sign of the z component of `in`, which is
    // positive for `heighttonormal` (e.g. in `NG_bump_vector3`), so skip
    // evaluating its subgraph.
    ScopedNode up = b.IsZero(scale) ? ResolveConnected(ctx.resolver, node, "in")
                                    : ScopedNode{};
    ValueId in_e = up && NodeCategory(ctx.resolver, up.node) == "heighttonormal"
                       ? b.ConstRgb(0.5, 0.5, 1.0)
                       : in("in");
    ValueId n = in("normal", NWorld(ctx)), t = in("tangent", TWorld(ctx)),
            bv = in("bitangent", BWorld(ctx));
    ValueId args[] = {in_e, scale, n, t, bv};
    return b.Emit(mtlx::Op::NormalMap, args);
  }
  if (cat == "heighttonormal") return LowerHeightToNormal(ctx, node);
  return std::nullopt;
}

std::optional<ValueId> LowerGeomNode(LowerCtx& ctx, const ScopedNode& node,
                                     const std::string& cat,
                                     const std::string& type) {
  auto in = BindInputs(ctx, node);
  if (cat == "image") return LowerImage(ctx, node);
  if (IsGeomIntrinsic(cat)) {
    return LowerGeomIntrinsic(
        ctx, cat, GetNodeInputString(ctx.resolver, node, "space"), type,
        ConstIntInput(ctx, node, "index").value_or(0) != 0);
  }
  if (cat == "geompropvalue") {
    std::string p = GetNodeInputString(ctx.resolver, node, "geomprop");
    if (p.empty()) return in("default");
    return LowerGeomProp(ctx, p, type, [&] { return in("default"); });
  }
  if (cat == "transformpoint" || cat == "transformvector" ||
      cat == "transformnormal") {
    std::string from = GetNodeInputString(ctx.resolver, node, "fromspace");
    std::string to = GetNodeInputString(ctx.resolver, node, "tospace");
    auto canon_space = [](std::string_view s) {
      return (s.empty() || s == "model") ? std::string_view("object") : s;
    };
    if (canon_space(from) != canon_space(to)) {
      WarnUnsupported(ctx, "Unsupported MaterialX space transform '" + cat +
                               "(" + from + "->" + to +
                               ")'; passing input through.");
    }
    return in("in");
  }
  return std::nullopt;
}

std::optional<ValueId> LowerMatrixValueNode(LowerCtx& ctx,
                                            const ScopedNode& node,
                                            const std::string& cat,
                                            const std::string& type) {
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "transformmatrix") {
    // Row vectors are transformed as `v * M`. The last coordinate of a vector
    // with fewer coordinates than the matrix is 1.
    ValueId v = LowerInputExpr(ctx, node, "in");
    MatVal mat = LowerMatrixInput(ctx, node, "mat");
    ValueId last = type == (mat.n == 4 ? "vector3" : "vector2")
                       ? b.One()
                       : b.Chan(v, static_cast<uint16_t>(mat.n - 1));
    ValueId res = b.Add(b.Mul(mat.rows[0], b.Chan(v, 0)),
                        b.Mul(mat.rows[1], b.Chan(v, 1)));
    res = mat.n == 4 ? b.Add(b.Add(res, b.Mul(mat.rows[2], b.Chan(v, 2))),
                             b.Mul(mat.rows[3], last))
                     : b.Add(res, b.Mul(mat.rows[2], last));
    if (type == "vector2") return b.Vec2(b.Chan(res, 0), b.Chan(res, 1));
    if (type == "vector3" && b.Channels(res) == 4) {
      return b.Rgb(b.Chan(res, 0), b.Chan(res, 1), b.Chan(res, 2));
    }
    return res;
  }
  if (cat == "determinant") {
    return LowerMatrixDet(b, LowerMatrixInput(ctx, node, "in"));
  }
  return std::nullopt;
}

std::optional<ValueId> LowerCompositingNode(LowerCtx& ctx,
                                            const ScopedNode& node,
                                            const std::string& cat) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "plus" || cat == "minus" || cat == "difference" ||
      cat == "screen" || cat == "burn" || cat == "dodge") {
    ValueId fg = in("fg"), bg = in("bg"), m = in("mix", b.One());
    // Like MaterialX, burn and dodge are 0 where they divide by 0.
    if (cat == "burn" || cat == "dodge") {
      ValueId denom = cat == "burn" ? fg : b.Sub(b.One(), fg);
      ValueId blend = cat == "burn"
                          ? b.Sub(b.One(), b.Div(b.Sub(b.One(), bg), denom))
                          : b.Div(bg, denom);
      return b.Select(b.Emit(mtlx::Op::Lt, {b.Emit(mtlx::Op::Abs, {denom}),
                                            b.Const(mtlx::kMtlxFloatEps)}),
                      b.Zero(), b.Mix(blend, bg, m));
    }
    ValueId blend = cat == "plus"    ? b.Add(bg, fg)
                    : cat == "minus" ? b.Sub(bg, fg)
                    : cat == "difference"
                        ? b.Emit(mtlx::Op::Abs, {b.Sub(bg, fg)})
                    : b.IsOne(fg) ? b.One()
                                  : b.Fma(b.Sub(b.One(), bg), fg, bg);
    return b.Mix(blend, bg, m);
  }
  if (cat == "inside") return b.Mul(in("in"), in("mask", b.One()));
  if (cat == "outside") return b.Mul(in("in"), b.Sub(b.One(), in("mask")));
  if (cat == "premult" || cat == "unpremult") {
    ValueId in_e = in("in");
    ValueId a = b.Chan(in_e, 3);
    ValueId rgb_scaled =
        cat == "premult" ? b.Mul(in_e, a, 3) : b.Div(in_e, a, 3);
    return b.Vec4(rgb_scaled, a);
  }
  if (cat == "over" || cat == "in" || cat == "out" || cat == "mask" ||
      cat == "matte" || cat == "disjointover") {
    ValueId fg = in("fg"), bg = in("bg");
    ValueId fa = b.Chan(fg, 3), ba = b.Chan(bg, 3);
    ValueId composite;
    if (cat == "over") {
      composite = b.Add(fg, b.Mul(bg, b.Sub(b.One(), fa)));
    } else if (cat == "in") {
      composite = b.Mul(fg, ba);
    } else if (cat == "out") {
      composite = b.Mul(fg, b.Sub(b.One(), ba));
    } else if (cat == "mask") {
      composite = b.Mul(bg, fa);
    } else if (cat == "matte") {
      composite = b.Add(b.Mul(fg, fa), b.Mul(bg, b.Sub(b.One(), fa)));
      composite = b.Vec4(composite, b.Add(fa, b.Mul(ba, b.Sub(b.One(), fa))));
    } else {
      composite = b.Select(
          b.Emit(mtlx::Op::Le, {b.Add(fa, ba), b.One()}), b.Add(fg, bg),
          b.Select(b.Emit(mtlx::Op::Lt, {b.Emit(mtlx::Op::Abs, {ba}),
                                         b.Const(mtlx::kMtlxFloatEps)}),
                   b.Zero(),
                   b.Add(fg, b.Mul(bg, b.Div(b.Sub(b.One(), fa), ba)))));
      composite =
          b.Vec4(composite, b.Emit(mtlx::Op::Min, {b.Add(fa, ba), b.One()}));
    }
    return b.Mix(composite, bg, in("mix", b.One()));
  }
  if (cat == "ramplr" || cat == "ramptb" || cat == "splitlr" ||
      cat == "splittb") {
    bool lr = (cat == "ramplr" || cat == "splitlr");
    ValueId v0 = in(lr ? "valuer" : "valuet"),
            v1 = in(lr ? "valuel" : "valueb");
    ValueId tc = in("texcoord", DefaultTexcoord(ctx));
    ValueId tc_comp = b.Chan(tc, lr ? 0 : 1);
    if (cat == "ramplr" || cat == "ramptb") {
      return b.Mix(v0, v1, Clip01(ctx, tc_comp));
    }
    return b.Select(b.Emit(mtlx::Op::Le, {in("center", b.Const(0.5)), tc_comp}),
                    v0, v1);
  }
  return std::nullopt;
}

std::optional<ValueId> LowerColorPbrNode(LowerCtx& ctx, const ScopedNode& node,
                                         const std::string& cat,
                                         const std::string& type,
                                         const std::string& output_name) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  if (cat == "rgbtohsv" || cat == "hsvtorgb") {
    return b.Emit(cat == "rgbtohsv" ? mtlx::Op::RgbToHsv : mtlx::Op::HsvToRgb,
                  {in("in")}, 0, type == "color4" ? 4 : 3);
  }
  if (cat == "luminance") {
    // The luminance of a gray value is the value itself, which keeps e.g. the
    // default `opacity` color of `standard_surface` a constant 1.
    ValueId in_e = in("in");
    ResolvedInput cr = ResolveInput(ctx.resolver, node, "lumacoeffs");
    bool has_coeffs =
        cr.kind != ResolvedInput::None &&
        !(cr.kind == ResolvedInput::Value &&
          cr.elem->getParent() == GetNodeInfo(ctx.resolver, node.node).nodedef);
    if (!has_coeffs && b.IsScalar(in_e)) return in_e;
    ValueId lum =
        b.Emit(mtlx::Op::Dot,
               {in_e, has_coeffs ? LowerResolvedInput(ctx, cr, b.Zero())
                                 : b.ConstRgb(0.2722287, 0.6740818, 0.0536895)},
               3);
    return type == "color4" ? b.Vec4(lum, b.Chan(in_e, 3)) : lum;
  }
  if (cat == "blackbody") {
    return b.Emit(mtlx::Op::Blackbody, {in("temperature", b.Const(5000.0))});
  }
  if (cat == "roughness_anisotropy") {
    ValueId r = in("roughness");
    ValueId r2 = b.Emit(mtlx::Op::Clip,
                        {b.Mul(r, r), b.Const(mtlx::kMtlxFloatEps), b.One()});
    ValueId aspect = b.Emit(
        mtlx::Op::Sqrt,
        {b.Sub(b.One(), b.Emit(mtlx::Op::Clip,
                               {in("anisotropy"), b.Zero(), b.Const(0.98)}))});
    if (b.IsOne(aspect)) return r2;
    return b.Vec2(b.Emit(mtlx::Op::Min, {b.Div(r2, aspect), b.One()}),
                  b.Mul(r2, aspect));
  }
  if (cat == "roughness_dual") {
    ValueId r = in("roughness");
    if (b.IsScalar(r)) {
      return b.Emit(mtlx::Op::Clip,
                    {b.Mul(r, r), b.Const(mtlx::kMtlxFloatEps), b.One()});
    }
    ValueId rx = b.Chan(r, 0);
    ValueId ry = b.Select(b.Emit(mtlx::Op::Lt, {b.Chan(r, 1), b.Zero()}), rx,
                          b.Chan(r, 1));
    return b.Vec2(
        b.Emit(mtlx::Op::Clip,
               {b.Mul(rx, rx), b.Const(mtlx::kMtlxFloatEps), b.One()}),
        b.Emit(mtlx::Op::Clip,
               {b.Mul(ry, ry), b.Const(mtlx::kMtlxFloatEps), b.One()}));
  }
  if (cat == "artistic_ior") {
    ValueId g = in("edge_color");
    if (b.IsZero(g) && output_name == "extinction") return b.Zero();
    ValueId r =
        b.Emit(mtlx::Op::Clip, {in("reflectivity"), b.Zero(), b.Const(0.99)});
    if (b.IsOne(g)) {
      ValueId inv = b.Div(b.One(), b.Add(b.One(), r));
      return output_name != "extinction"
                 ? b.Mul(b.Sub(b.One(), r), inv)
                 : b.Mul(b.Mul(b.Const(2.0), b.Emit(mtlx::Op::Sqrt, {r})), inv);
    }
    ValueId r_sqrt = b.Emit(mtlx::Op::Sqrt, {r});
    if (b.IsZero(g)) {
      return b.Div(b.Add(b.One(), r_sqrt), b.Sub(b.One(), r_sqrt));
    }
    ValueId eta =
        b.Mix(b.Div(b.Sub(b.One(), r), b.Add(b.One(), r)),
              b.Div(b.Add(b.One(), r_sqrt), b.Sub(b.One(), r_sqrt)), g);
    if (output_name != "extinction") return eta;
    ValueId np1 = b.Add(eta, b.One()), nm1 = b.Sub(eta, b.One());
    return b.Emit(mtlx::Op::Sqrt,
                  {b.Div(b.Sub(b.Mul(b.Mul(np1, np1), r), b.Mul(nm1, nm1)),
                         b.Sub(b.One(), r))});
  }
  if (cat == "chiang_hair_roughness") {
    auto clip_in = [&](const char* name, double def) {
      return b.Emit(mtlx::Op::Clip,
                    {in(name, b.Const(def)), b.Const(1e-3), b.One()});
    };
    ValueId lr = clip_in("longitudinal", 0.1), ar = clip_in("azimuthal", 0.2);
    ValueId v_sqrt = b.Add(
        b.Add(b.Mul(b.Const(0.726), lr), b.Mul(b.Const(0.812), b.Mul(lr, lr))),
        b.Mul(b.Const(3.7), b.Emit(mtlx::Op::Pow, {lr, b.Const(20.0)})));
    if (output_name == "roughness_TT" || output_name == "roughness_TRT") {
      bool is_tt = output_name == "roughness_TT";
      v_sqrt = b.Mul(v_sqrt, in(is_tt ? "scale_TT" : "scale_TRT",
                                b.Const(is_tt ? 0.5 : 2.0)));
    }
    ValueId s = b.Add(
        b.Add(b.Mul(b.Const(0.265), ar), b.Mul(b.Const(1.194), b.Mul(ar, ar))),
        b.Mul(b.Const(5.372), b.Emit(mtlx::Op::Pow, {ar, b.Const(22.0)})));
    return b.Vec2(b.Mul(v_sqrt, v_sqrt), s);
  }
  if (cat == "chiang_hair_absorption_from_color") {
    ValueId c =
        b.Emit(mtlx::Op::Clip, {in("color", b.One()), b.Const(1e-3), b.One()});
    ValueId bv = in("azimuthal_roughness", b.Const(0.2));
    ValueId b_fac = b.Add(
        b.Const(5.969),
        b.Mul(
            bv,
            b.Add(
                b.Const(-0.215),
                b.Mul(
                    bv,
                    b.Add(
                        b.Const(2.532),
                        b.Mul(bv,
                              b.Add(b.Const(-10.73),
                                    b.Mul(bv,
                                          b.Add(b.Const(5.574),
                                                b.Mul(bv,
                                                      b.Const(0.245)))))))))));
    ValueId sigma = b.Div(b.Emit(mtlx::Op::Log, {c}), b_fac);
    return b.Mul(sigma, sigma);
  }
  if (cat == "deon_hair_absorption_from_melanin") {
    ValueId log_conc = b.Emit(
        mtlx::Op::Log,
        {b.Emit(mtlx::Op::Max,
                {b.Sub(b.One(), in("melanin_concentration", b.Const(0.25))),
                 b.Const(1e-4)})});
    ValueId pigment = LowerMixVal(
        ctx, in("melanin_redness", b.Const(0.5)),
        [&] {
          return b.Emit(mtlx::Op::Log,
                        {in("pheomelanin_color",
                            b.ConstRgb(0.829444, 0.67032, 0.349938))});
        },
        [&] {
          return b.Emit(mtlx::Op::Log,
                        {in("eumelanin_color",
                            b.ConstRgb(0.657704, 0.498077, 0.254107))});
        });
    return b.Emit(mtlx::Op::Max, {b.Mul(log_conc, pigment), b.Zero()});
  }
  return std::nullopt;
}

std::optional<ValueId> LowerNoiseNode(LowerCtx& ctx, const ScopedNode& node,
                                      const std::string& cat,
                                      const std::string& type) {
  std::string_view noise = cat;
  bool is_3d = absl::ConsumeSuffix(&noise, "3d");
  if ((!is_3d && !absl::ConsumeSuffix(&noise, "2d")) ||
      (noise != "cellnoise" && noise != "worleynoise" && noise != "noise" &&
       noise != "fractal")) {
    return std::nullopt;
  }
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  uint16_t imm = mtlx::PackNoiseImm(is_3d, type != "float");
  ValueId coord =
      is_3d ? in("position", MeshAttr(ctx, "__p_object", 3, PWorld(ctx)))
            : in("texcoord", DefaultTexcoord(ctx));
  if (noise == "cellnoise") {
    return b.Emit(mtlx::Op::CellNoise, {coord}, imm);
  }
  if (noise == "worleynoise") {
    ValueId jitter = in("jitter", b.One()), style = in("style");
    uint8_t ch = type == "vector2" ? 2 : (type == "float" ? 1 : 3);
    double style_val = 0.0;
    if (b.IsConstScalar(style, &style_val) && std::round(style_val) != 1.0) {
      return b.Emit(mtlx::Op::Worley, {coord, jitter},
                    static_cast<uint16_t>(imm | mtlx::kWorleyDistanceOnly), ch);
    }
    return b.Emit(mtlx::Op::Worley, {coord, jitter, style}, imm, ch);
  }
  bool is_noise = noise == "noise";
  std::optional<int> oct =
      is_noise ? std::nullopt : ConstIntInput(ctx, node, "octaves");
  bool four_ch = IsFourChannelType(type);
  bool split_vec2 = !is_noise && type == "vector2";
  ValueId lac = kInvalidValue, dim = kInvalidValue, dyn_oct = kInvalidValue;
  if (!is_noise) {
    lac = in("lacunarity", b.Const(2.0));
    dim = in("diminish", b.Const(0.5));
    if (!oct) dyn_oct = in("octaves");
  }
  auto eval_one = [&](bool vec3, ValueId p, uint8_t ch = 0) {
    if (is_noise) {
      return b.Emit(mtlx::Op::Noise, {p}, mtlx::PackNoiseImm(is_3d, vec3), ch);
    }
    if (oct) {
      uint16_t clamped =
          static_cast<uint16_t>(std::clamp(*oct, 0, mtlx::kMaxFractalOctaves));
      return b.Emit(mtlx::Op::Fractal, {p, lac, dim},
                    mtlx::PackNoiseImm(is_3d, vec3, clamped), ch);
    }
    ValueId args[] = {p, lac, dim, dyn_oct};
    return b.Emit(mtlx::Op::Fractal, args,
                  mtlx::PackNoiseImm(is_3d, vec3, mtlx::kFractalDynamicOctaves),
                  ch);
  };
  auto offset_coord = [&]() {
    constexpr double kNoiseOffset[3] = {19.0, 73.0, 29.0};
    constexpr double kFractalOffset[3] = {19.0, 193.0, 17.0};
    const double* off = is_noise ? kNoiseOffset : kFractalOffset;
    if (is_3d) return b.Add(coord, b.ConstRgb(off[0], off[1], off[2]));
    return b.Add(coord, b.Vec2(b.Const(off[0]), b.Const(off[1])));
  };
  ValueId res;
  if (four_ch) {
    // Like MaterialX, a fourth channel is scalar noise at an offset.
    res = b.Vec4(eval_one(true, coord), eval_one(false, offset_coord()));
  } else if (split_vec2) {
    // Like MaterialX, `vector2` fractals are two scalar fractals.
    res = b.Vec2(eval_one(false, coord), eval_one(false, offset_coord()));
  } else {
    uint8_t ch = type == "vector2" ? 2 : 0;
    res = eval_one(type != "float", coord, ch);
  }
  res = b.Mul(res, in("amplitude", b.One()));
  return is_noise ? b.Add(res, in("pivot")) : res;
}

ValueId LowerNodeOutputExpr(LowerCtx& ctx, const ScopedNode& node,
                            const std::string& output_name) {
  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  const std::string& type = node.node->getType();
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;

  // Fast path for `range` with unit gamma, collapsing `abs + pow + sign + mul`
  // (e.g. default `range` nodes inside `NG_unifiednoise*`) to a linear remap.
  if (cat == "range") {
    if (b.IsOne(in("gamma", b.One()))) {
      ValueId ilo = in("inlow"), ihi = in("inhigh", b.One()),
              olo = in("outlow"), ohi = in("outhigh", b.One());
      ValueId res = LowerLinearRemap(ctx, node, ilo, ihi, olo, ohi);
      ValueId doclamp = in("doclamp");
      if (b.IsZero(doclamp)) return res;
      ValueId clamped = b.Emit(mtlx::Op::Clip, {res, olo, ohi});
      return b.IsOne(doclamp) ? clamped : b.Select(doclamp, clamped, res);
    }
  }
  if (cat == "smoothstep") {
    // Handle before `graph_impl` expansion so multi-channel `NG_smoothstep_*`
    // stays vectorized. Like OSL, a step function at `low` if `low >= high`.
    ValueId lo = in("low"), hi = in("high", b.One()), x = in("in");
    ValueId t = b.Div(b.Sub(x, lo), b.Sub(hi, lo));
    return b.Select(
        b.Emit(mtlx::Op::Lt, {x, lo}), b.Zero(),
        b.Select(b.Emit(mtlx::Op::Le, {hi, x}), b.One(),
                 b.Mul(b.Mul(t, t), b.Fma(b.Const(-2.0), t, b.Const(3.0)))));
  }
  if (cat == "hsvadjust") {
    ValueId amt = in("amount", b.ConstRgb(0.0, 1.0, 1.0));
    if (b.Channels(amt) == 3) {
      if (auto v = b.ConstValue(amt);
          v && (*v)[0] == 0.0 && (*v)[1] == 1.0 && (*v)[2] == 1.0) {
        return in("in");
      }
    }
  }
  if (cat == "contrast") {
    if (b.IsOne(in("amount", b.One()))) return in("in");
  }
  // Fast-path safe no-op, scalar-broadcast, and narrowing `ND_convert_*`
  // before `stdlib_ng.mtlx` expands them into `separate` + `combine`.
  // Widening (`2 -> 3/4` zero-pads `.z`, `3 -> 4` sets `.w = 1`),
  // `integer -> boolean` (`!= 0`), and `* -> surfaceshader` fall through.
  if (cat == "convert") {
    std::string_view in_type = InputType(ctx.resolver, node.node, "in");
    if (IsValueType(in_type) && IsValueType(type) &&
        !(in_type == "integer" && type == "boolean")) {
      uint8_t in_ch = TypeChannels(in_type), out_ch = TypeChannels(type);
      if (in_ch == out_ch || in_ch == 1) return in("in");
      if (in_ch > out_ch) {
        ValueId v = in("in");
        if (b.Channels(v) <= out_ch) return v;
        if (out_ch == 3) {
          return b.Rgb(b.Chan(v, 0), b.Chan(v, 1), b.Chan(v, 2));
        }
        return b.Vec2(b.Chan(v, 0), b.Chan(v, 1));
      }
    }
  }
  // Handle noise before NodeGraph expansion so stdlib noise/fractal NodeGraphs
  // emit native noise ops directly.
  if (auto e = LowerNoiseNode(ctx, node, cat, type)) return *e;

  if (mx::NodeGraphPtr ng = GetNodeInfo(ctx.resolver, node.node).graph_impl) {
    return LowerResolvedInput(
        ctx,
        ResolveOutput(ctx.resolver, node,
                      ResolveNodeGraphOutput(ng, output_name)),
        b.Zero());
  }

  // Unary and binary ops that map to a single SSA op.
  static constexpr struct {
    std::string_view name;
    mtlx::Op op;
  } kUnaryOps[] = {
      {"absval", mtlx::Op::Abs},  {"floor", mtlx::Op::Floor},
      {"ceil", mtlx::Op::Ceil},   {"round", mtlx::Op::Round},
      {"fract", mtlx::Op::Fract}, {"sin", mtlx::Op::Sin},
      {"cos", mtlx::Op::Cos},     {"tan", mtlx::Op::Tan},
      {"asin", mtlx::Op::Asin},   {"acos", mtlx::Op::Acos},
      {"sqrt", mtlx::Op::Sqrt},   {"ln", mtlx::Op::Log},
      {"exp", mtlx::Op::Exp},
  };
  for (const auto& [name, op] : kUnaryOps) {
    if (cat == name) {
      return b.Emit(op, {in("in", op == mtlx::Op::Log ? b.One() : b.Zero())});
    }
  }
  if (cat == "power") {
    ValueId exp = in("in2", b.One());
    // Skip compiling dead `in1` (and its textures/opaques) when `exp == 0`.
    if (b.IsZero(exp)) return b.One();
    return b.Emit(mtlx::Op::Pow, {in("in1"), exp});
  }
  static constexpr struct {
    std::string_view name;
    const char *in1, *in2;
    mtlx::Op op;
    bool def2_one;
  } kBinaryOps[] = {
      {"add", "in1", "in2", mtlx::Op::Add, false},
      {"subtract", "in1", "in2", mtlx::Op::Sub, false},
      {"min", "in1", "in2", mtlx::Op::Min, false},
      {"max", "in1", "in2", mtlx::Op::Max, false},
      {"atan2", "iny", "inx", mtlx::Op::Atan2, true},
      {"crossproduct", "in1", "in2", mtlx::Op::Cross, false},
      {"rotate2d", "in", "amount", mtlx::Op::Rotate2D, false},
  };
  for (const auto& [name, in1, in2, op, def2_one] : kBinaryOps) {
    if (cat == name) {
      ValueId e1 = in(in1), e2 = in(in2, def2_one ? b.One() : b.Zero());
      return b.Emit(op, {e1, e2});
    }
  }

  if (auto e = LowerChannelNode(ctx, node, cat, type)) return *e;
  if (auto e = LowerMathNode(ctx, node, cat)) return *e;
  if (auto e = LowerLogicNode(ctx, node, cat, type)) return *e;
  if (auto e = LowerVectorNode(ctx, node, cat)) return *e;
  if (auto e = LowerGeomNode(ctx, node, cat, type)) return *e;
  if (auto e = LowerMatrixValueNode(ctx, node, cat, type)) return *e;
  if (auto e = LowerCompositingNode(ctx, node, cat)) return *e;
  if (auto e = LowerColorPbrNode(ctx, node, cat, type, output_name)) return *e;

  WarnUnsupported(ctx, "Unsupported MaterialX node category '" + cat +
                           "'; passing through its 'in' input.");
  return in("in");
}

}  // namespace

const std::string& NodeCategory(GraphResolver& r, const mx::NodePtr& node) {
  if (node->hasNodeDefString()) {
    if (const mx::NodeDefPtr& nd = GetNodeInfo(r, node).nodedef) {
      return nd->getNodeString();
    }
  }
  return node->getCategory();
}

std::string_view InputType(GraphResolver& r, const mx::NodePtr& node,
                           const std::string& name) {
  if (mx::InputPtr in = node->getInput(name)) return in->getType();
  if (const mx::NodeDefPtr& nd = GetNodeInfo(r, node).nodedef) {
    if (mx::InputPtr in = nd->getActiveInput(name)) return in->getType();
  }
  return {};
}

ScopedNode ResolveConnected(GraphResolver& r, const ScopedNode& s,
                            const std::string& name, const char* mat_input) {
  ResolvedInput res = ResolveInput(r, s, name);
  return res.kind == ResolvedInput::Connected
             ? UnwrapShaderNode(r, std::move(res.upstream),
                                std::move(res.output_name), mat_input)
             : ScopedNode{};
}

std::string GetNodeInputString(GraphResolver& r, const ScopedNode& s,
                               const std::string& name, std::string_view def) {
  mx::ValueElementPtr in = GetValueInput(r, s, name);
  return in ? in->getValueString() : std::string(def);
}

ScopedNode UnwrapShaderNode(GraphResolver& r, ScopedNode s,
                            std::string output_name, const char* mat_input) {
  while (s.node) {
    ResolvedInput res;
    const std::string& cat = NodeCategory(r, s.node);
    if (cat == "surfacematerial" || cat == "dot") {
      res = ResolveInput(r, s, cat == "dot" ? "in" : mat_input);
    } else if (mx::NodeGraphPtr ng = GetNodeInfo(r, s.node).graph_impl) {
      res = ResolveOutput(r, s, ResolveNodeGraphOutput(ng, output_name));
    } else {
      break;
    }
    if (res.kind != ResolvedInput::Connected) return {};
    s = std::move(res.upstream);
    output_name = std::move(res.output_name);
  }
  return s;
}

ValueId LowerInputExpr(LowerCtx& ctx, const ScopedNode& node,
                       const std::string& input_name, ValueId default_val) {
  if (default_val == kInvalidValue) default_val = ctx.b->Zero();
  return LowerResolvedInput(ctx, ResolveInput(ctx.resolver, node, input_name),
                            default_val);
}

}  // namespace mtlx_detail
PXR_NAMESPACE_CLOSE_SCOPE
