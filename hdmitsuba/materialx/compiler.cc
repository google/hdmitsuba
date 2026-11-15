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
// cut out at all.
struct Expr {
  std::vector<Inst> code;
  uint8_t channels = 3;
  uint32_t max_depth = 1;

  Expr() = default;
  explicit Expr(Inst i, uint8_t channels = 3) : code{i}, channels(channels) {}

  static Expr zero() { return Expr({Op::Const, 0, 0}, 1); }
  static Expr one() { return Expr({Op::Const, 0, 1}, 1); }
  static Expr uv() { return Expr({Op::GeomUV, 0, 0}, 2); }

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
  e.channels = ew ? std::max<uint8_t>({args.channels...}) : result_channels(op);
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

// Maps MaterialX `cmlib` and USD `GfColorSpace` names to a `GfColorSpace` when
// conversion to linear Rec.709 is needed, or `std::nullopt` if `color_space`
// is empty, data/raw, or already linear Rec.709.
std::optional<GfColorSpace> ResolveSourceColorSpace(
    std::string_view color_space) {
  if (color_space.empty()) return std::nullopt;
  std::string cs = absl::AsciiStrToLower(color_space);
  static const absl::NoDestructor<
      absl::flat_hash_map<std::string_view, TfToken>>
      kMtlxAliases({
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
      token == GfColorSpaceNames->Unknown || !GfColorSpace::IsValid(token)) {
    return std::nullopt;
  }
  return GfColorSpace(token);
}

GfVec3f ToLinearRec709(GfVec3f rgb_val, std::string_view color_space) {
  if (rgb_val != GfVec3f(0.0f) && rgb_val != GfVec3f(1.0f)) {
    if (auto src_cs = ResolveSourceColorSpace(color_space)) {
      rgb_val = GfColorSpace(GfColorSpaceNames->LinearRec709)
                    .Convert(*src_cs, rgb_val)
                    .GetRGB();
    }
  }
  return rgb_val;
}

// Expression of the default `texcoord` node (UV0), undoing Mitsuba's mesh UV
// v-flip.
Expr DefaultTexcoord(uint8_t channels = 2) {
  Expr uv = vec2(Expr::uv().x(), Expr::one() - Expr::uv().y());
  uv.channels = channels;
  return uv;
}

mx::InputPtr GetValueInput(const mx::NodePtr& node, const std::string& name) {
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

mx::NodePtr UnwrapShaderNode(mx::NodePtr node) {
  while (node) {
    const std::string& cat = node->getCategory();
    if (cat != "dot" && cat != "surfacematerial") break;
    node = node->getConnectedNode(cat == "surfacematerial" ? "surfaceshader"
                                                           : "in");
  }
  return node;
}

// Compiles a flattened MaterialX DAG into a `mitsuba::mtlx::MtlxProgram`.
class MtlxExprCompiler {
 public:
  MtlxExprCompiler(std::string_view variant,
                   absl::flat_hash_set<std::string>& warned)
      : variant_(variant), warned_(warned) {}

  // Compiles a MaterialX EDF or surface emission DAG (`surface`,
  // `surface_unlit`, `uniform_edf`, `generalized_schlick_edf`, `multiply`,
  // `add`, `mix`) into a radiance `Expr`, or `Expr::zero()` if none.
  Expr CompileEdf(const mx::NodePtr& edf_node);

  // Compiles surface cutout `opacity` (`[0, 1]`) for `surface`,
  // `surface_unlit`, and `mix` (`surfaceshader`) nodes, or `Expr::one()`.
  Expr CompileOpacity(const mx::NodePtr& shader_node);

  ref<Object> CreateExprTexture(const Expr& expr, bool unbounded);

 private:
  Expr Const(double value);
  Expr ConstRgb(double r, double g, double b);
  Expr Opaque(double value);
  // Emits a scalar or multi-channel parameter value. All-zero and all-one
  // tuples are baked into the program to enable constant simplifications, while
  // other values become opaque scalars exposed via traversal.
  Expr EmitTuple(std::initializer_list<float> vals);
  Expr EmitTmpExpr(const Expr& expr);

  Expr CompileInputExpr(const mx::NodePtr& node, const std::string& input_name,
                        const Expr& default_expr = Expr::zero());
  Expr CompileNodeOutputExpr(const mx::NodePtr& node);
  std::optional<int> ConstIntInput(const mx::NodePtr& node,
                                   const std::string& name);
  Expr CompileValueElement(const mx::ValueElementPtr& elem,
                           const Expr& default_expr);

  void WarnUnsupported(const std::string& msg) {
    if (warned_.insert(msg).second) TF_WARN("%s", msg.c_str());
  }

  std::string_view variant_;
  MtlxProgram prog_;
  std::vector<uint8_t> tmp_channels_;
  absl::flat_hash_map<const mx::Node*, Expr> node_output_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> edf_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> opacity_expr_;
  absl::flat_hash_map<std::pair<const mx::Node*, std::string>, Expr>
      value_input_expr_;
  absl::flat_hash_map<Expr, int> expr_to_slot_;
  absl::flat_hash_set<std::string>& warned_;
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
    std::string_view in_type = InputType(node, "in");
    int n = in_type == "vector2" ? 2 : IsFourChannelType(in_type) ? 4 : 3;
    return in("in").chan(std::clamp(idx, 0, n - 1));
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
    if (!node->getConnectedNode("in1")) {
      Expr i1 = in("in1");
      return i1.is_zero() ? Expr::zero() : i1 * in("in2", Expr::one());
    }
    Expr i2 = in("in2", Expr::one());
    return i2.is_zero() ? Expr::zero() : in("in1") * i2;
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
  if (cat == "invert") return in("amount", Expr::one()) - in("in");

  // Texture coordinates
  if (cat == "texcoord") {
    if (ConstIntInput(node, "index").value_or(0) != 0) {
      WarnUnsupported(
          "Unsupported MaterialX 'texcoord' index; using the primary UV set.");
    }
    return DefaultTexcoord(type == "vector3" ? 3 : 2);
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
    } else if (cat != "uniform_edf") {
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

ref<Object> MtlxExprCompiler::CreateExprTexture(const Expr& expr,
                                                bool unbounded) {
  RegisterMaterialXExprTexture();
  prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
  prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
  prog_.result_uniform = expr.uniform();
  prog_.unbounded = unbounded;
  Properties props("materialx_expr");
  props.set_any("program", std::move(prog_));
  return PluginManager::instance()->create_object(props, variant_,
                                                  ObjectType::Texture);
}

}  // namespace

CompiledSurfaceShader CompileSurfaceShader(const mx::NodePtr& shader_node,
                                           std::string_view variant) {
  absl::flat_hash_set<std::string> warned;
  CompiledSurfaceShader out;
  {
    MtlxExprCompiler compiler(variant, warned);
    Expr radiance = compiler.CompileEdf(shader_node);
    if (!radiance.is_zero()) {
      out.radiance_texture =
          compiler.CreateExprTexture(radiance, /*unbounded=*/true);
    }
  }
  {
    MtlxExprCompiler compiler(variant, warned);
    Expr opacity = compiler.CompileOpacity(shader_node);
    if (!opacity.is_one()) {
      out.opacity_texture =
          compiler.CreateExprTexture(opacity, /*unbounded=*/false);
    }
  }
  return out;
}

PXR_NAMESPACE_CLOSE_SCOPE
