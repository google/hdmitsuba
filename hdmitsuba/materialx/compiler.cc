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

#include "hdmitsuba/materialx/compiler.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/ascii.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <pxr/base/gf/color.h>
#include <pxr/base/gf/colorSpace.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/materialx/evaluator.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace mx = MaterialX;
using mitsuba::Object;
using mitsuba::ObjectType;
using mitsuba::PluginManager;
using mitsuba::Properties;
using mitsuba::ref;
using mitsuba::mtlx::elementwise_arity;
using mitsuba::mtlx::Inst;
using mitsuba::mtlx::kMaxImm;
using mitsuba::mtlx::MtlxProgram;
using mitsuba::mtlx::Op;
using mitsuba::mtlx::RegisterMaterialXExprTexture;
using mitsuba::mtlx::result_channels;

struct Expr;

template <typename... Exprs>
Expr call(Op op, uint16_t imm, const Exprs&... args);
template <typename... Exprs>
Expr call(Op op, const Expr& arg0, const Exprs&... rest);

// Reverse-Polish-Notation expression fragment that evaluates to one stack slot.
// Constants 0.0 and 1.0 are always at indices 0 and 1 in `MtlxProgram::consts`.
//
// Arithmetic, `mix()` and `select()` simplify constant 0 and 1 operands. This
// is not an optimization (Dr.Jit simplifies the traced program itself), but
// determines what is known before rendering: whether a surface emits or is
// cut out at all, and which textures and primvars its shader reads.
struct Expr {
  std::vector<Inst> code;
  uint8_t channels = 3;
  uint32_t max_depth = 1;

  Expr() = default;
  explicit Expr(Inst i, uint8_t channels = 3) : code{i}, channels(channels) {}

  static Expr zero() { return Expr({Op::Const, 0, 0}, 1); }
  static Expr one() { return Expr({Op::Const, 0, 1}, 1); }
  static Expr p_world() { return Expr({Op::GeomPosWorld, 0, 0}); }
  static Expr n_world() { return Expr({Op::GeomNormalWorld, 0, 0}); }
  static Expr t_world() { return Expr({Op::GeomTangentWorld, 0, 0}); }
  static Expr b_world() { return Expr({Op::GeomBitangentWorld, 0, 0}); }

  bool uniform() const { return channels == 1; }
  bool is_zero() const { return *this == zero(); }
  bool is_one() const { return *this == one(); }

  Expr chan(int c) const {
    if (uniform()) return *this;
    Expr e = *this;
    e.code.push_back({Op::Chan, 0, static_cast<uint16_t>(c)});
    e.channels = 1;
    return e;
  }
  Expr x() const { return chan(0); }
  Expr y() const { return chan(1); }
  Expr w() const { return chan(3); }

  Expr operator-() const { return is_zero() ? zero() : call(Op::Neg, *this); }
  Expr operator!() const {
    return is_zero() ? one() : is_one() ? zero() : call(Op::Eq, *this, zero());
  }
  Expr operator+(const Expr& b) const {
    return is_zero() ? b : b.is_zero() ? *this : call(Op::Add, *this, b);
  }
  Expr operator-(const Expr& b) const {
    return b.is_zero() ? *this : is_zero() ? -b : call(Op::Sub, *this, b);
  }
  Expr operator*(const Expr& b) const {
    return (is_zero() || b.is_zero()) ? zero()
           : is_one()                 ? b
           : b.is_one()               ? *this
                                      : call(Op::Mul, *this, b);
  }
  Expr operator/(const Expr& b) const {
    return (is_zero() || b.is_zero()) ? zero()
           : b.is_one()               ? *this
                                      : call(Op::Div, *this, b);
  }
  Expr operator<(const Expr& b) const { return call(Op::Lt, *this, b); }
  Expr operator<=(const Expr& b) const { return call(Op::Le, *this, b); }
  Expr operator>(const Expr& b) const { return b < *this; }
  Expr operator>=(const Expr& b) const { return b <= *this; }

  bool operator==(const Expr& o) const {
    return channels == o.channels && code == o.code;
  }
  bool operator!=(const Expr& o) const { return !(*this == o); }

  template <typename H>
  friend H AbslHashValue(H h, const Expr& e) {
    return H::combine(std::move(h), e.channels, e.code);
  }
};

template <typename... Exprs>
Expr call(Op op, uint16_t imm, const Exprs&... args) {
  Expr e;
  uint32_t depth = 0;
  ((e.code.insert(e.code.end(), args.code.begin(), args.code.end()),
    e.max_depth = std::max(e.max_depth, depth + args.max_depth), ++depth),
   ...);
  bool ew = elementwise_arity(op) != 0;
  e.channels =
      ew ? std::max<uint8_t>({args.channels...}) : result_channels(op, imm);
  e.code.push_back({op, static_cast<uint8_t>(ew ? e.channels : 0), imm});
  return e;
}

template <typename... Exprs>
Expr call(Op op, const Expr& arg0, const Exprs&... rest) {
  return call(op, uint16_t{0}, arg0, rest...);
}

Expr rgb(const Expr& r, const Expr& g, const Expr& b) {
  return (r.uniform() && r == g && g == b) ? r : call(Op::Rgb, r, g, b);
}
Expr vec2(const Expr& x, const Expr& y) { return call(Op::Vec2, x, y); }
Expr vec4(const Expr& xyz, const Expr& w) {
  return (xyz.uniform() && xyz == w) ? xyz : call(Op::Vec4, xyz, w);
}
Expr select(const Expr& cond, const Expr& t, const Expr& f) {
  return cond.is_zero() ? f : cond.is_one() ? t : call(Op::Select, cond, t, f);
}
Expr saturate(const Expr& e) {
  return (e.is_zero() || e.is_one())
             ? e
             : call(Op::Clip, e, Expr::zero(), Expr::one());
}
Expr mix(const Expr& fg, const Expr& bg, const Expr& m) {
  return m.is_zero() ? bg : m.is_one() ? fg : call(Op::Mix, fg, bg, m);
}
template <typename FgFn, typename BgFn>
Expr CompileMix(const Expr& m, FgFn&& fg_fn, BgFn&& bg_fn) {
  if (m.is_zero()) return bg_fn();
  if (m.is_one()) return fg_fn();
  Expr fg = fg_fn(), bg = bg_fn();
  return mix(fg, bg, m);
}

// Expression of the default `texcoord` node (UV0).
Expr DefaultTexcoord(uint8_t channels = 2) {
  return Expr({Op::GeomUV, 0, 0}, channels);
}

mx::InputPtr GetValueInput(mx::NodePtr node, std::string name) {
  while (mx::NodePtr up = node->getConnectedNode(name)) {
    const std::string& cat = up->getCategory();
    if (cat != "dot" && cat != "constant") return nullptr;
    node = std::move(up);
    name = cat == "constant" ? "value" : "in";
  }
  if (mx::InputPtr in = node->getInput(name); in && in->hasValueString()) {
    return in;
  }
  if (mx::NodeDefPtr nd = node->getNodeDef()) {
    if (mx::InputPtr in = nd->getActiveInput(name);
        in && in->hasValueString()) {
      return in;
    }
  }
  return nullptr;
}

std::string GetNodeInputString(const mx::NodePtr& node, const std::string& name,
                               std::string_view def = {}) {
  mx::InputPtr in = GetValueInput(node, name);
  return in ? in->getValueString() : std::string(def);
}

// Returns the type of the input `name` of `node`, which may only be declared
// by its node definition, or an empty view if there is no such input.
std::string_view InputType(const mx::NodePtr& node, const std::string& name) {
  if (mx::InputPtr in = node->getInput(name)) return in->getType();
  if (mx::NodeDefPtr nd = node->getNodeDef()) {
    if (mx::InputPtr in = nd->getActiveInput(name)) return in->getType();
  }
  return {};
}

bool IsFourChannelType(std::string_view type) {
  return type == "color4" || type == "vector4";
}

uint8_t TypeChannels(std::string_view type) {
  if (type == "float" || type == "integer" || type == "boolean") return 1;
  if (type == "vector2") return 2;
  return IsFourChannelType(type) ? 4 : 3;
}

mx::NodePtr UnwrapShaderNode(mx::NodePtr node) {
  while (node) {
    const std::string& cat = node->getCategory();
    if (cat != "dot" && cat != "surfacematerial") break;
    node = node->getConnectedNode(cat == "surfacematerial" ? "surfaceshader"
                                                           : "in");
  }
  return node;
}

// Compiles a flattened MaterialX DAG into a `mitsuba::mtlx::MtlxProgram` and
// a list of Mitsuba Texture child inputs (`in0`, `in1`, ...).
class MtlxExprCompiler {
 public:
  using LookupTextureFn =
      std::function<const CachedTexture*(const TextureKey&)>;
  using DiscoverTextureFn = std::function<void(const TextureKey&)>;

  MtlxExprCompiler(const LookupTextureFn& lookup_texture,
                   absl::flat_hash_set<std::string>& warned)
      : lookup_texture_(&lookup_texture), warned_(&warned) {}
  explicit MtlxExprCompiler(const DiscoverTextureFn& discover_texture)
      : discover_texture_(&discover_texture) {}

  // Compiles a MaterialX EDF or surface emission DAG (`surface`,
  // `surface_unlit`, `uniform_edf`, `generalized_schlick_edf`, `multiply`,
  // `add`, `mix`) into a radiance `Expr`, or `Expr::zero()` if none.
  Expr CompileEdf(const mx::NodePtr& edf_node);

  // Compiles surface cutout `opacity` (`[0, 1]`) for `surface`,
  // `surface_unlit`, and `mix` (`surfaceshader`) nodes, or `Expr::one()`.
  Expr CompileOpacity(const mx::NodePtr& shader_node);

  ref<Object> CreateExprTexture(const Expr& expr, bool unbounded,
                                std::string_view variant,
                                MeshAttributeRequests& mesh_attributes);

 private:
  Expr Const(double value);
  Expr ConstRgb(double r, double g, double b);
  Expr Opaque(double value);
  // Emits a scalar or multi-channel parameter value. All-zero and all-one
  // tuples are baked into the program to enable constant simplifications, while
  // other values become opaque scalars exposed via traversal.
  Expr EmitTuple(std::initializer_list<float> vals);
  Expr EmitTmpExpr(const Expr& expr);

  TfToken ResolveSourceColorSpace(std::string_view color_space);
  GfVec3f ToLinearRec709(GfVec3f rgb_val, std::string_view color_space);
  Expr CompileInputExpr(const mx::NodePtr& node, const std::string& input_name,
                        const Expr& default_expr = Expr::zero());
  Expr CompileNodeOutputExpr(const mx::NodePtr& node);
  std::optional<int> ConstIntInput(const mx::NodePtr& node,
                                   const std::string& name);
  Expr MeshAttr(std::string_view name, int dim, const Expr& def);
  Expr SampleImage(const mx::NodePtr& node);
  Expr CompileValueElement(const mx::ValueElementPtr& elem,
                           const Expr& default_expr);

  // Warns once per message about an unsupported feature of the compiled shader
  // (not during texture discovery, which compiles it a second time).
  void WarnUnsupported(const std::string& msg) {
    if (warned_ && warned_->insert(msg).second) {
      TF_WARN("%s", msg.c_str());
    }
  }

  const LookupTextureFn* lookup_texture_ = nullptr;
  const DiscoverTextureFn* discover_texture_ = nullptr;
  MtlxProgram prog_;
  std::vector<std::pair<std::string, int>> attr_requests_;
  std::vector<uint8_t> tmp_channels_;
  absl::flat_hash_map<const mx::Node*, Expr> node_output_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> edf_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> opacity_expr_;
  absl::flat_hash_map<std::pair<const mx::Node*, std::string>, Expr>
      value_input_expr_;
  absl::flat_hash_map<Expr, int> expr_to_slot_;
  std::vector<ref<Object>> inputs_;
  uint16_t discover_input_idx_ = 0;
  absl::flat_hash_set<std::string>* warned_ = nullptr;
};

Expr MtlxExprCompiler::Const(double value) {
  auto it = std::find(prog_.consts.begin(), prog_.consts.end(), value);
  if (it == prog_.consts.end()) {
    if (prog_.consts.size() >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many constants");
    }
    it = prog_.consts.insert(prog_.consts.end(), value);
  }
  return Expr({Op::Const, 0, static_cast<uint16_t>(it - prog_.consts.begin())},
              1);
}

Expr MtlxExprCompiler::ConstRgb(double r, double g, double b) {
  if (r == g && g == b) return Const(r);
  Expr er = Const(r), eg = Const(g), eb = Const(b);
  return rgb(er, eg, eb);
}

Expr MtlxExprCompiler::Opaque(double value) {
  if (prog_.opaques.size() >= kMaxImm) {
    throw std::runtime_error("materialx_expr: too many opaque values");
  }
  prog_.opaques.push_back(value);
  return Expr({Op::Opaque, 0, static_cast<uint16_t>(prog_.opaques.size() - 1)},
              1);
}

// Maps MaterialX `cmlib` and USD `GfColorSpace` names to a canonical
// `GfColorSpace` token when conversion to linear Rec.709 is needed, or an empty
// token if `color_space` is empty, data/raw, or already linear Rec.709.
TfToken MtlxExprCompiler::ResolveSourceColorSpace(
    std::string_view color_space) {
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
          {"rec709_display", GfColorSpaceNames->G24Rec709},
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
    WarnUnsupported("Unsupported MaterialX color space '" +
                    std::string(color_space) +
                    "'; treating as linear Rec.709.");
    return {};
  }
  return token;
}

GfVec3f MtlxExprCompiler::ToLinearRec709(GfVec3f rgb_val,
                                         std::string_view color_space) {
  if (rgb_val == GfVec3f(0.0f)) return rgb_val;
  TfToken src_cs = ResolveSourceColorSpace(color_space);
  if (src_cs.IsEmpty()) return rgb_val;
  rgb_val = GfColorSpace(GfColorSpaceNames->LinearRec709)
                .Convert(GfColorSpace(src_cs), rgb_val)
                .GetRGB();
  // Snap 1-ULP float32 transfer/matrix rounding at `(1, 1, 1)` (e.g. sRGB
  // `opacity = (1, 1, 1)` under `UsdColorSpaceAPI`) so `EmitTuple` folds
  // opaque white to `Expr::one()`, while preserving spaces where `(1, 1, 1)`
  // does not map to Rec.709 white.
  if (GfIsClose(rgb_val, GfVec3f(1.0f), 1e-5f)) rgb_val = GfVec3f(1.0f);
  return rgb_val;
}

Expr MtlxExprCompiler::MeshAttr(std::string_view name, int dim,
                                const Expr& def) {
  std::pair<std::string, int> req{std::string(name), dim};
  auto it = std::find(attr_requests_.begin(), attr_requests_.end(), req);
  if (it == attr_requests_.end()) {
    if (attr_requests_.size() >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many attributes");
    }
    prog_.attr_names.push_back(MeshAttributeName(name, dim));
    it = attr_requests_.insert(attr_requests_.end(), std::move(req));
  }
  Expr e = def;
  uint8_t read_ch = static_cast<uint8_t>(dim);
  e.code.push_back(
      {Op::Attr, read_ch, static_cast<uint16_t>(it - attr_requests_.begin())});
  e.channels = std::max<uint8_t>(def.channels, read_ch);
  return e;
}

Expr MtlxExprCompiler::EmitTuple(std::initializer_list<float> vals) {
  if (std::all_of(vals.begin(), vals.end(),
                  [](float x) { return x == 0.0f; })) {
    return Expr::zero();
  }
  if (std::all_of(vals.begin(), vals.end(),
                  [](float x) { return x == 1.0f; })) {
    return Expr::one();
  }
  Expr e[4];
  int n = 0;
  for (float v : vals) e[n++] = Opaque(v);
  if (n == 1) return e[0];
  if (n == 2) return vec2(e[0], e[1]);
  Expr c3 = rgb(e[0], e[1], e[2]);
  return n == 3 ? c3 : vec4(c3, e[3]);
}

Expr MtlxExprCompiler::EmitTmpExpr(const Expr& expr) {
  if (expr.code.size() == 1 && expr.code[0].op == Op::LoadTmp) return expr;
  auto [it, inserted] =
      expr_to_slot_.try_emplace(expr, static_cast<int>(prog_.num_tmps));
  if (inserted) {
    if (prog_.num_tmps >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many temporaries");
    }
    prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
    prog_.code.push_back(
        {Op::StoreTmp, 0, static_cast<uint16_t>(prog_.num_tmps)});
    prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
    tmp_channels_.push_back(expr.channels);
    ++prog_.num_tmps;
  }
  int slot = it->second;
  return Expr({Op::LoadTmp, 0, static_cast<uint16_t>(slot)},
              tmp_channels_[slot]);
}

Expr MtlxExprCompiler::CompileValueElement(const mx::ValueElementPtr& elem,
                                           const Expr& default_expr) {
  const std::string& type = elem->getType();
  if (type == "boolean") {
    const std::string& s = elem->getValueString();
    return (s == "true" || s == "1") ? Expr::one() : Expr::zero();
  }
  mx::ValuePtr val = elem->getValue();
  if (!val) return default_expr;

  if (type == "float") return EmitTuple({val->asA<float>()});
  // Integers and booleans select code paths (e.g. switch branches) and are
  // baked.
  if (type == "integer") return Const(val->asA<int>());
  if (type == "color3") {
    mx::Color3 v = val->asA<mx::Color3>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return EmitTuple({c[0], c[1], c[2]});
  }
  if (type == "color4") {
    mx::Color4 v = val->asA<mx::Color4>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return EmitTuple({c[0], c[1], c[2], v[3]});
  }
  if (type == "vector2") {
    mx::Vector2 v = val->asA<mx::Vector2>();
    return EmitTuple({v[0], v[1]});
  }
  if (type == "vector3") {
    mx::Vector3 v = val->asA<mx::Vector3>();
    return EmitTuple({v[0], v[1], v[2]});
  }
  if (type == "vector4") {
    mx::Vector4 v = val->asA<mx::Vector4>();
    return EmitTuple({v[0], v[1], v[2], v[3]});
  }
  return default_expr;
}

Expr MtlxExprCompiler::CompileInputExpr(const mx::NodePtr& node,
                                        const std::string& input_name,
                                        const Expr& default_expr) {
  if (mx::InputPtr input = node->getInput(input_name)) {
    if (mx::NodePtr upstream = input->getConnectedNode()) {
      auto it = node_output_expr_.find(upstream.get());
      if (it == node_output_expr_.end()) {
        Expr se = CompileNodeOutputExpr(upstream);
        if (se.code.size() > 1) se = EmitTmpExpr(se);
        it = node_output_expr_.emplace(upstream.get(), std::move(se)).first;
      }
      return it->second;
    }
  }
  mx::InputPtr val_in = GetValueInput(node, input_name);
  if (!val_in) return default_expr;
  auto [it, inserted] = value_input_expr_.try_emplace({node.get(), input_name});
  if (inserted) it->second = CompileValueElement(val_in, default_expr);
  return it->second;
}

std::optional<int> MtlxExprCompiler::ConstIntInput(const mx::NodePtr& node,
                                                   const std::string& name) {
  Expr e = CompileInputExpr(node, name, Expr{});
  return (e.code.size() == 1 && e.code[0].op == Op::Const)
             ? std::optional<int>(static_cast<int>(prog_.consts[e.code[0].imm]))
             : std::nullopt;
}

Expr MtlxExprCompiler::SampleImage(const mx::NodePtr& node) {
  auto default_val = [&] { return CompileInputExpr(node, "default"); };
  mx::InputPtr file_in = GetValueInput(node, "file");
  if (!file_in || file_in->getValueString().empty()) return default_val();

  const std::string& type = node->getType();
  bool has_alpha = IsFourChannelType(type);
  std::string u_mode = GetNodeInputString(node, "uaddressmode", "periodic");
  std::string v_mode = GetNodeInputString(node, "vaddressmode", "periodic");
  auto is_oob_mode = [](std::string_view m) {
    return m == "constant" || m == "black";
  };
  bool has_oob = is_oob_mode(u_mode) || is_oob_mode(v_mode);
  bool mixed_wrap = u_mode != v_mode || has_oob;
  if (!GetNodeInputString(node, "layer").empty()) {
    WarnUnsupported(
        "Unsupported MaterialX image 'layer'; reading default layer.");
  }
  if (!GetNodeInputString(node, "framerange").empty()) {
    WarnUnsupported(
        "Unsupported MaterialX image 'framerange'; reading static file.");
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
    key.color_space =
        ResolveSourceColorSpace(file_in->getActiveColorSpace()).GetString();
  }
  if (type == "float") key.channel = 0;
  // Mitsuba's `bitmap` supports a single `wrap_mode`; mixed or out-of-bounds
  // (`"constant"` / `"black"`) address modes clamp at the border and remap UVs
  // below.
  key.wrap_mode = (mixed_wrap || u_mode == "clamp") ? "clamp"
                  : u_mode == "mirror"              ? "mirror"
                                                    : "repeat";
  key.filter_type =
      GetNodeInputString(node, "filtertype", "linear") == "closest"
          ? "nearest"
          : "bilinear";

  if (discover_texture_) {
    (*discover_texture_)(key);
    if (has_alpha) {
      key.channel = 3;
      key.color_space.clear();
      (*discover_texture_)(key);
    }
    CompileInputExpr(node, "texcoord", DefaultTexcoord());
    default_val();
    return Expr({Op::SampleInput, 0, discover_input_idx_++},
                TypeChannels(type));
  }

  const CachedTexture* cached_color = (*lookup_texture_)(key);
  if (!cached_color || !cached_color->texture) return default_val();
  const CachedTexture* cached_alpha = nullptr;
  if (has_alpha && cached_color->has_alpha) {
    key.channel = 3;
    key.color_space.clear();
    cached_alpha = (*lookup_texture_)(key);
    if (!cached_alpha || !cached_alpha->texture) return default_val();
  }

  Expr tc = CompileInputExpr(node, "texcoord", DefaultTexcoord());
  bool default_uv = !mixed_wrap && tc == DefaultTexcoord();
  Expr u, v;
  if (!default_uv) {
    tc = EmitTmpExpr(tc);
    auto wrap_axis = [&](Expr c, const std::string& mode) {
      if (!mixed_wrap || mode == "clamp" || is_oob_mode(mode)) return c;
      if (mode == "mirror") {
        return Expr::one() -
               call(Op::Abs,
                    call(Op::Fract, c * Const(0.5)) * Const(2.0) - Expr::one());
      }
      return call(Op::Fract, c);
    };
    u = wrap_axis(tc.x(), u_mode);
    v = Expr::one() - wrap_axis(tc.y(), v_mode);
    if (cached_alpha) {
      u = EmitTmpExpr(u);
      v = EmitTmpExpr(v);
    }
  }

  auto sample_tex = [&](const CachedTexture& cached) -> Expr {
    auto it = std::find(inputs_.begin(), inputs_.end(), cached.texture);
    if (it == inputs_.end()) {
      if (inputs_.size() >= kMaxImm) {
        throw std::runtime_error("materialx_expr: too many texture inputs");
      }
      it = inputs_.insert(inputs_.end(), cached.texture);
    }
    uint16_t idx = static_cast<uint16_t>(it - inputs_.begin());
    uint8_t ch = cached.channel_count == 1 ? 1 : 3;
    if (default_uv) return Expr({Op::SampleInput, ch, idx}, ch);
    Expr e = call(Op::SampleInputUV, idx, u, v);
    e.channels = e.code.back().channels = ch;
    return e;
  };

  Expr result = sample_tex(*cached_color);
  if (has_alpha) {
    result =
        vec4(result, cached_alpha ? sample_tex(*cached_alpha) : Expr::one());
  }
  if (type == "vector2") result.channels = 2;
  if (has_oob) {
    auto oob_axis = [](const Expr& c) {
      return (c < Expr::zero()) + (c > Expr::one());
    };
    auto oob_val = [&](const std::string& mode) {
      return mode == "black" ? Expr::zero() : default_val();
    };
    if (u_mode == v_mode) {
      result =
          select(oob_axis(tc.x()) + oob_axis(tc.y()), oob_val(u_mode), result);
    } else {
      if (is_oob_mode(u_mode)) {
        result = select(oob_axis(tc.x()), oob_val(u_mode), result);
      }
      if (is_oob_mode(v_mode)) {
        result = select(oob_axis(tc.y()), oob_val(v_mode), result);
      }
    }
  }
  return result;
}

Expr MtlxExprCompiler::CompileNodeOutputExpr(const mx::NodePtr& node) {
  const std::string& cat = node->getCategory();
  const std::string& type = node->getType();
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };

  // Constants, routing, and channel ops (`stdlib_ng` flattens compound
  // `convert` nodes, leaving scalar bool/int/float conversions as
  // pass-through).
  if (cat == "constant") return in("value");
  if (cat == "dot" || cat == "convert") return in("in");
  if (cat == "extract") {
    int idx = ConstIntInput(node, "index").value_or(0);
    return in("in").chan(
        std::clamp(idx, 0, TypeChannels(InputType(node, "in")) - 1));
  }
  if (cat == "combine2") {
    if (IsFourChannelType(type)) {
      if (InputType(node, "in1") != "vector2") {
        Expr i1 = in("in1"), i2 = in("in2");
        return vec4(i1, i2);
      }
      Expr v1 = EmitTmpExpr(in("in1")), v2 = EmitTmpExpr(in("in2"));
      return vec4(rgb(v1.x(), v1.y(), v2.x()), v2.y());
    }
    Expr i1 = in("in1"), i2 = in("in2");
    return vec2(i1, i2);
  }
  if (cat == "combine3") {
    Expr i1 = in("in1"), i2 = in("in2"), i3 = in("in3");
    return rgb(i1, i2, i3);
  }
  if (cat == "combine4") {
    Expr i1 = in("in1"), i2 = in("in2"), i3 = in("in3"), i4 = in("in4");
    return vec4(rgb(i1, i2, i3), i4);
  }

  // Arithmetic and interpolation, simplifying constant operands without
  // compiling unused inputs.
  if (cat == "multiply") {
    if (GetValueInput(node, "in2") && !GetValueInput(node, "in1")) {
      Expr i2 = in("in2", Expr::one());
      return i2.is_zero() ? Expr::zero() : in("in1") * i2;
    }
    Expr i1 = in("in1");
    return i1.is_zero() ? Expr::zero() : i1 * in("in2", Expr::one());
  }
  if (cat == "add") return in("in1") + in("in2");
  if (cat == "subtract") return in("in1") - in("in2");
  if (cat == "divide") {
    Expr i1 = in("in1");
    return i1.is_zero() ? Expr::zero() : i1 / in("in2", Expr::one());
  }
  if (cat == "mix") {
    return CompileMix(
        in("mix"), [&] { return in("fg"); }, [&] { return in("bg"); });
  }
  if (cat == "clamp") {
    Expr x = in("in"), lo = in("low"), hi = in("high", Expr::one());
    return call(Op::Clip, x, lo, hi);
  }
  if (cat == "power") {
    Expr i1 = in("in1"), i2 = in("in2", Expr::one());
    return call(Op::Pow, i1, i2);
  }
  if (cat == "invert") return in("amount", Expr::one()) - in("in");
  if (cat == "absval") return call(Op::Abs, in("in"));
  if (cat == "fract") return call(Op::Fract, in("in"));

  auto check_zero_index = [&] {
    if (ConstIntInput(node, "index").value_or(0) != 0) {
      WarnUnsupported("Unsupported MaterialX '" + cat +
                      "' index; using index 0.");
    }
  };

  // Images and texture coordinates
  if (cat == "image") return SampleImage(node);
  if (cat == "texcoord") {
    check_zero_index();
    return DefaultTexcoord(type == "vector3" ? 3 : 2);
  }
  if (cat == "rotate2d") {
    Expr i1 = in("in"), amt = in("amount");
    return call(Op::Rotate2D, i1, amt);
  }

  // Conditionals & boolean negation
  if (cat == "not") return !in("in");
  if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    Expr v1 = in("value1"), v2 = in("value2");
    Expr cond;
    if (v1 == v2) {
      cond = cat != "ifgreater" ? Expr::one() : Expr::zero();
    } else if (v1.code.size() == 1 && v1.code[0].op == Op::Const &&
               v2.code.size() == 1 && v2.code[0].op == Op::Const) {
      double c1 = prog_.consts[v1.code[0].imm];
      double c2 = prog_.consts[v2.code[0].imm];
      bool b = cat == "ifgreater"     ? (c1 > c2)
               : cat == "ifgreatereq" ? (c1 >= c2)
                                      : (c1 == c2);
      cond = b ? Expr::one() : Expr::zero();
    } else {
      cond = cat == "ifgreater"     ? (v1 > v2)
             : cat == "ifgreatereq" ? (v1 >= v2)
                                    : call(Op::Eq, v1, v2);
    }
    if (type == "boolean") return cond;
    if (cond.is_one()) return in("in1");
    if (cond.is_zero()) return in("in2");
    Expr i1 = in("in1"), i2 = in("in2");
    return select(cond, i1, i2);
  }

  // Vector ops & normal map
  if (cat == "normalize" || cat == "dotproduct") {
    std::string_view in_type =
        InputType(node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_imm = in_type == "vector2" ? 2 : in_type == "vector4" ? 4 : 0;
    if (cat == "normalize") return call(Op::Normalize, vec_imm, in("in"));
    Expr i1 = in("in1"), i2 = in("in2");
    return call(Op::Dot, vec_imm, i1, i2);
  }
  if (cat == "normalmap") {
    std::string space = GetNodeInputString(node, "space", "tangent");
    if (space != "tangent") {
      WarnUnsupported("Unsupported MaterialX normalmap space '" + space +
                      "'; evaluating in tangent space.");
    }
    Expr in_e = in("in"), scale = in("scale", Expr::one());
    Expr n = in("normal", Expr::n_world()), t = in("tangent", Expr::t_world()),
         b = in("bitangent", Expr::b_world());
    return call(Op::NormalMap, in_e, scale, n, t, b);
  }

  // Geometric intrinsics & space transforms
  auto warn_world_space = [&](const std::string& space) {
    if (space != "world") {
      WarnUnsupported("Unsupported MaterialX " + cat + " space '" + space +
                      "'; evaluating in world space.");
    }
  };
  if (cat == "position") {
    std::string space = GetNodeInputString(node, "space", "object");
    if (space == "object" || space == "model") {
      return MeshAttr("__p_object", 3, Expr::p_world());
    }
    warn_world_space(space);
    return Expr::p_world();
  }
  if (cat == "normal") {
    std::string space = GetNodeInputString(node, "space", "object");
    if (space == "object" || space == "model") {
      return call(Op::Normalize, MeshAttr("__n_object", 3, Expr::n_world()));
    }
    if (space == "tangent") return ConstRgb(0.0, 0.0, 1.0);
    warn_world_space(space);
    return Expr::n_world();
  }
  if (cat == "tangent" || cat == "bitangent") {
    check_zero_index();
    std::string space = GetNodeInputString(node, "space", "object");
    bool bitangent = cat == "bitangent";
    if (space == "tangent") {
      return bitangent ? ConstRgb(0.0, 1.0, 0.0) : ConstRgb(1.0, 0.0, 0.0);
    }
    warn_world_space(space);
    return bitangent ? Expr::b_world() : Expr::t_world();
  }
  if (cat == "viewdirection") {
    warn_world_space(GetNodeInputString(node, "space", "world"));
    return Expr({Op::GeomIncidentDir, 0, 0});
  }
  if (cat == "geomcolor") {
    check_zero_index();
    Expr color =
        MeshAttr("displayColor", type == "float" ? 1 : 3, Expr::zero());
    // The alpha channel of `geomcolor` is the display opacity.
    return type == "color4"
               ? vec4(color, MeshAttr("displayOpacity", 1, Expr::one()))
               : color;
  }
  if (cat == "geompropvalue") {
    std::string p = GetNodeInputString(node, "geomprop");
    if (p.empty()) return in("default");
    uint8_t ch = TypeChannels(type);
    Expr attr = MeshAttr(p, ch == 1 ? 1 : 3, in("default"));
    attr.channels = ch;
    return attr;
  }
  if (cat == "transformpoint" || cat == "transformvector" ||
      cat == "transformnormal") {
    std::string from = GetNodeInputString(node, "fromspace");
    std::string to = GetNodeInputString(node, "tospace");
    auto canon_space = [](std::string_view s) {
      return (s.empty() || s == "model") ? std::string_view("object") : s;
    };
    if (canon_space(from) != canon_space(to)) {
      WarnUnsupported("Unsupported MaterialX space transform '" + cat + "(" +
                      from + "->" + to + ")'; passing input through.");
    }
    return in("in");
  }

  // 2D ramps & luminance
  if (cat == "ramplr" || cat == "ramptb") {
    bool lr = cat == "ramplr";
    Expr fg = in(lr ? "valuer" : "valuet"), bg = in(lr ? "valuel" : "valueb");
    Expr tc = in("texcoord", DefaultTexcoord());
    return mix(fg, bg, saturate(lr ? tc.x() : tc.y()));
  }
  if (cat == "luminance") {
    // The luminance of a gray value is the value itself, which keeps e.g. the
    // default `opacity` color of `standard_surface` a constant 1.
    Expr in_e = in("in");
    bool has_coeffs = node->getInput("lumacoeffs") != nullptr;
    if (!has_coeffs && in_e.uniform()) return in_e;
    if (type == "color4") in_e = EmitTmpExpr(in_e);
    Expr lum = call(Op::Dot, in_e,
                    has_coeffs ? in("lumacoeffs")
                               : ConstRgb(0.2722287, 0.6740818, 0.0536895));
    return type == "color4" ? vec4(lum, in_e.w()) : lum;
  }

  if (cat == "bump") {
    WarnUnsupported(
        "Unsupported MaterialX node category 'bump'; passing through "
        "'normal'.");
    return in("normal", Expr::n_world());
  }
  WarnUnsupported("Unsupported MaterialX node category '" + cat +
                  "'; passing through its 'in' input.");
  return in("in");
}

Expr MtlxExprCompiler::CompileEdf(const mx::NodePtr& edf_node) {
  mx::NodePtr node = UnwrapShaderNode(edf_node);
  if (!node) return Expr::zero();
  if (auto it = edf_expr_.find(node.get()); it != edf_expr_.end()) {
    return it->second;
  }
  const std::string& cat = node->getCategory();
  Expr res = Expr::zero();
  if (cat == "surface" || cat == "surface_unlit") {
    if (cat == "surface") {
      res = CompileEdf(node->getConnectedNode("edf"));
    } else {
      Expr em = CompileInputExpr(node, "emission", Expr::one());
      if (!em.is_zero()) {
        em = em *
             (Expr::one() - saturate(CompileInputExpr(node, "transmission")));
      }
      if (!em.is_zero()) {
        res = em * CompileInputExpr(node, "emission_color", Expr::one());
      }
    }
    if (!res.is_zero()) {
      res = res * saturate(CompileInputExpr(node, "opacity", Expr::one()));
    }
  } else if (cat == "uniform_edf" || cat == "conical_edf" ||
             cat == "measured_edf") {
    mx::InputPtr file_in =
        cat == "measured_edf" ? GetValueInput(node, "file") : nullptr;
    if (file_in && !file_in->getValueString().empty()) {
      WarnUnsupported(
          "Unsupported MaterialX 'measured_edf' IES profile; approximating as "
          "uniform_edf.");
    } else if (cat != "uniform_edf" && !discover_texture_) {
      TF_DEBUG(HDMITSUBA_MATERIAL)
          .Msg("Approximating MaterialX '%s' as uniform_edf.\n", cat.c_str());
    }
    res = CompileInputExpr(node, "color", Expr::one());
  } else if (cat == "generalized_schlick_edf") {
    Expr base = CompileEdf(node->getConnectedNode("base"));
    if (!base.is_zero()) {
      Expr c0 = CompileInputExpr(node, "color0", Expr::one());
      Expr c90 = CompileInputExpr(node, "color90", Expr::one());
      // Weight `base` by the cosine-weighted hemispherical average of
      // `mix(c0, c90, (1 - cos)^exp)`, which equals
      // `mix(c0, c90, 2 / ((exp + 1) * (exp + 2)))`.
      Expr w = c0;
      if (c0 != c90) {
        Expr exp = EmitTmpExpr(
            call(Op::Max, CompileInputExpr(node, "exponent", Const(5.0)),
                 Expr::zero()));
        w = mix(c90, c0,
                Const(2.0) / ((exp + Expr::one()) * (exp + Const(2.0))));
      }
      res = base * w;
    }
  } else if (cat == "add") {
    res = CompileEdf(node->getConnectedNode("in1")) +
          CompileEdf(node->getConnectedNode("in2"));
  } else if (cat == "multiply") {
    Expr base = CompileEdf(node->getConnectedNode("in1"));
    if (!base.is_zero()) {
      res = base * CompileInputExpr(node, "in2", Expr::one());
    }
  } else if (cat == "mix") {
    res = CompileMix(
        CompileInputExpr(node, "mix"),
        [&] { return CompileEdf(node->getConnectedNode("fg")); },
        [&] { return CompileEdf(node->getConnectedNode("bg")); });
  }
  if (res.code.size() > 1) res = EmitTmpExpr(res);
  edf_expr_[node.get()] = res;
  return res;
}

Expr MtlxExprCompiler::CompileOpacity(const mx::NodePtr& shader_node) {
  mx::NodePtr node = UnwrapShaderNode(shader_node);
  if (!node) return Expr::one();
  if (auto it = opacity_expr_.find(node.get()); it != opacity_expr_.end()) {
    return it->second;
  }
  const std::string& cat = node->getCategory();
  Expr res = Expr::one();
  if (cat == "surface" || cat == "surface_unlit") {
    res = saturate(CompileInputExpr(node, "opacity", Expr::one()));
  } else if (cat == "mix" &&
             node->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
    res = CompileMix(
        CompileInputExpr(node, "mix"),
        [&] { return CompileOpacity(node->getConnectedNode("fg")); },
        [&] { return CompileOpacity(node->getConnectedNode("bg")); });
  }
  if (res.code.size() > 1) res = EmitTmpExpr(res);
  opacity_expr_[node.get()] = res;
  return res;
}

ref<Object> MtlxExprCompiler::CreateExprTexture(
    const Expr& expr, bool unbounded, std::string_view variant,
    MeshAttributeRequests& mesh_attributes) {
  RegisterMaterialXExprTexture();
  Properties props("materialx_expr");
  for (size_t i = 0; i < inputs_.size(); ++i) {
    props.set("in" + std::to_string(i), std::move(inputs_[i]));
  }
  prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
  mesh_attributes.insert(attr_requests_.begin(), attr_requests_.end());
  prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
  prog_.result_uniform = expr.uniform();
  prog_.unbounded = unbounded;
  props.set_any("program", std::move(prog_));
  return PluginManager::instance()->create_object(props, variant,
                                                  ObjectType::Texture);
}

}  // namespace

CompiledSurfaceShader CompileSurfaceShader(
    const mx::NodePtr& shader_node, std::string_view variant,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture) {
  absl::flat_hash_set<std::string> warned;
  CompiledSurfaceShader out;
  {
    MtlxExprCompiler compiler(lookup_texture, warned);
    Expr radiance = compiler.CompileEdf(shader_node);
    if (!radiance.is_zero()) {
      out.radiance_texture = compiler.CreateExprTexture(
          radiance, /*unbounded=*/true, variant, out.mesh_attributes);
    }
  }
  {
    MtlxExprCompiler compiler(lookup_texture, warned);
    Expr opacity = compiler.CompileOpacity(shader_node);
    if (!opacity.is_one()) {
      out.opacity_texture = compiler.CreateExprTexture(
          opacity, /*unbounded=*/false, variant, out.mesh_attributes);
    }
  }
  return out;
}

void DiscoverSurfaceShaderTextures(
    const mx::NodePtr& shader_node,
    const std::function<void(const TextureKey&)>& callback) {
  MtlxExprCompiler compiler(callback);
  compiler.CompileEdf(shader_node);
  compiler.CompileOpacity(shader_node);
}

PXR_NAMESPACE_CLOSE_SCOPE
