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
#include <absl/strings/strip.h>
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
using mitsuba::mtlx::ClosureNode;
using mitsuba::mtlx::ClosureOp;
using mitsuba::mtlx::elementwise_arity;
using mitsuba::mtlx::Inst;
using mitsuba::mtlx::kFractalDynamicOctaves;
using mitsuba::mtlx::kFractalOctavesShift;
using mitsuba::mtlx::kMaxFractalOctaves;
using mitsuba::mtlx::kMaxImm;
using mitsuba::mtlx::kMtlxFloatEps;
using mitsuba::mtlx::kNoise3D;
using mitsuba::mtlx::kNoiseVec3;
using mitsuba::mtlx::MtlxProgram;
using mitsuba::mtlx::Op;
using mitsuba::mtlx::RegisterMaterialXBsdf;
using mitsuba::mtlx::RegisterMaterialXExprTexture;
using mitsuba::mtlx::result_channels;
namespace leaf_arg = mitsuba::mtlx::leaf_arg;
namespace combinator_arg = mitsuba::mtlx::combinator_arg;

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
  Expr z() const { return chan(2); }
  Expr w() const { return chan(3); }
  // Returns the first three channels of a 4-channel expression.
  Expr xyz() const {
    Expr e = *this;
    e.channels = std::min<uint8_t>(channels, 3);
    return e;
  }

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

std::string ConnectedOutputName(const mx::InputPtr& input) {
  if (!input->hasNodeGraphString()) return input->getOutputString();
  mx::OutputPtr out = input->getConnectedOutput();
  return out ? out->getOutputString() : std::string();
}

// Returns the category of `node`, which is determined by its explicit node
// definition, if any. For example, `NG_ramp_gradient` has an `ifgreater` node
// with the node definition `ND_ifgreatereq_color4I`.
std::string NodeCategory(const mx::NodePtr& node) {
  if (node->hasNodeDefString()) {
    if (mx::NodeDefPtr nd = node->getNodeDef()) return nd->getNodeString();
  }
  return node->getCategory();
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
    std::string cat = NodeCategory(node);
    if (cat != "dot" && cat != "surfacematerial") break;
    node = node->getConnectedNode(cat == "surfacematerial" ? "surfaceshader"
                                                           : "in");
  }
  return node;
}

// Compiles a flattened MaterialX DAG of pattern/math/procedural/image nodes
// into a `mitsuba::mtlx::MtlxProgram` and a list of Mitsuba Texture child
// inputs (`in0`, `in1`, ...).
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

  // Compiles a MaterialX surface shader or BSDF closure DAG into
  // `prog_.closures` and `prog_.opacity_slot`.
  void CompileBsdf(const mx::NodePtr& shader_node);

  ref<Object> CreateExprTexture(const Expr& expr, bool unbounded,
                                std::string_view variant,
                                MeshAttributeRequests& mesh_attributes);

  ref<Object> CreateBsdf(std::string_view variant,
                         MeshAttributeRequests& mesh_attributes);

 private:
  // A 3x3 or 4x4 matrix as the expressions of its `n` rows.
  struct MatExpr {
    Expr rows[4];
    int n = 3;

    static MatExpr Identity(int n) {
      MatExpr m;
      m.n = n;
      for (int i = 0; i < 3; ++i) {
        Expr row = rgb(i == 0 ? Expr::one() : Expr::zero(),
                       i == 1 ? Expr::one() : Expr::zero(),
                       i == 2 ? Expr::one() : Expr::zero());
        m.rows[i] = n == 4 ? vec4(row, Expr::zero()) : row;
      }
      m.rows[3] = vec4(Expr::zero(), Expr::one());
      return m;
    }
  };

  Expr Const(double value);
  Expr ConstRgb(double r, double g, double b);
  Expr Opaque(double value);
  // Emits a scalar or multi-channel parameter value. All-zero and all-one
  // tuples are baked into the program to enable constant simplifications, while
  // other values become opaque scalars exposed via traversal.
  Expr EmitTuple(std::initializer_list<float> vals);
  int EmitTmp(const Expr& expr);
  Expr EmitTmpExpr(const Expr& expr);
  Expr LoadTmp(int slot) const {
    return Expr({Op::LoadTmp, 0, static_cast<uint16_t>(slot)},
                prog_.tmp_channels[slot]);
  }
  Expr CompileOpacity(const mx::NodePtr& shader_node);
  int CompileBsdfClosure(const mx::NodePtr& node);
  int EmitClosure(ClosureOp op,
                  std::initializer_list<std::pair<int, int>> args);
  ref<Object> CreatePlugin(std::string_view name, ObjectType type,
                           std::string_view variant,
                           MeshAttributeRequests& mesh_attributes);

  TfToken ResolveSourceColorSpace(std::string_view color_space);
  GfVec3f ToLinearRec709(GfVec3f rgb_val, std::string_view color_space);
  Expr CompileInputExpr(const mx::NodePtr& node, const std::string& input_name,
                        const Expr& default_expr = Expr::zero());
  Expr CompileNodeOutputExpr(const mx::NodePtr& node,
                             const std::string& output_name);
  std::optional<int> ConstIntInput(const mx::NodePtr& node,
                                   const std::string& name);
  Expr CompileIfCondition(const mx::NodePtr& node, std::string_view cat);
  MatExpr CompileMatrixInput(const mx::NodePtr& node,
                             const std::string& input_name);
  MatExpr CompileMatrixNode(const mx::NodePtr& node);
  // Emits `MatTranspose`, `MatInverse` or `MatMul` (of `a` and `b`).
  MatExpr MatrixOp(Op op, const MatExpr& a, const MatExpr* b = nullptr);
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
  absl::flat_hash_map<std::pair<const mx::Node*, std::string>, Expr>
      node_output_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> edf_expr_;
  absl::flat_hash_map<const mx::Node*, Expr> opacity_expr_;
  absl::flat_hash_map<const mx::Node*, int> bsdf_closure_;
  absl::flat_hash_map<std::pair<const mx::Node*, std::string>, Expr>
      value_input_expr_;
  absl::flat_hash_map<const mx::Node*, MatExpr> node_matrix_expr_;
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

int MtlxExprCompiler::EmitTmp(const Expr& expr) {
  if (expr.code.size() == 1 && expr.code[0].op == Op::LoadTmp) {
    return expr.code[0].imm;
  }
  size_t next_slot = prog_.tmp_channels.size();
  auto [it, inserted] =
      expr_to_slot_.try_emplace(expr, static_cast<int>(next_slot));
  if (inserted) {
    if (next_slot >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many temporaries");
    }
    prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
    prog_.code.push_back({Op::StoreTmp, 0, static_cast<uint16_t>(next_slot)});
    prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
    prog_.tmp_channels.push_back(expr.channels);
  }
  return it->second;
}

Expr MtlxExprCompiler::EmitTmpExpr(const Expr& expr) {
  if (expr.code.size() == 1 && expr.code[0].op <= Op::LoadTmp) return expr;
  return LoadTmp(EmitTmp(expr));
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
      std::string out_name = ConnectedOutputName(input);
      std::pair<const mx::Node*, std::string> key(upstream.get(), out_name);
      auto it = node_output_expr_.find(key);
      if (it == node_output_expr_.end()) {
        Expr se = CompileNodeOutputExpr(upstream, out_name);
        if (se.code.size() > 1) se = EmitTmpExpr(se);
        it = node_output_expr_.emplace(std::move(key), std::move(se)).first;
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

Expr MtlxExprCompiler::CompileIfCondition(const mx::NodePtr& node,
                                          std::string_view cat) {
  Expr v1 = CompileInputExpr(node, "value1");
  Expr v2 = CompileInputExpr(node, "value2");
  if (v1 == v2) return cat != "ifgreater" ? Expr::one() : Expr::zero();
  if (v1.code.size() == 1 && v1.code[0].op == Op::Const &&
      v2.code.size() == 1 && v2.code[0].op == Op::Const) {
    double c1 = prog_.consts[v1.code[0].imm];
    double c2 = prog_.consts[v2.code[0].imm];
    bool b = cat == "ifgreater"     ? (c1 > c2)
             : cat == "ifgreatereq" ? (c1 >= c2)
                                    : (c1 == c2);
    return b ? Expr::one() : Expr::zero();
  }
  return cat == "ifgreater"     ? (v1 > v2)
         : cat == "ifgreatereq" ? (v1 >= v2)
                                : call(Op::Eq, v1, v2);
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::MatrixOp(Op op, const MatExpr& a,
                                                     const MatExpr* b) {
  if (prog_.tmp_channels.size() + a.n > kMaxImm) {
    throw std::runtime_error("materialx_expr: too many temporaries");
  }
  uint32_t depth = 0;
  auto emit_rows = [&](const MatExpr& m) {
    for (int i = 0; i < a.n; ++i) {
      prog_.code.insert(prog_.code.end(), m.rows[i].code.begin(),
                        m.rows[i].code.end());
      prog_.stack_size =
          std::max(prog_.stack_size, depth + m.rows[i].max_depth);
      ++depth;
    }
  };
  emit_rows(a);
  if (b) emit_rows(*b);
  uint16_t first = static_cast<uint16_t>(prog_.tmp_channels.size());
  prog_.code.push_back({op, static_cast<uint8_t>(a.n), first});
  prog_.tmp_channels.insert(prog_.tmp_channels.end(), a.n,
                            static_cast<uint8_t>(a.n));
  MatExpr out;
  out.n = a.n;
  for (int i = 0; i < a.n; ++i) out.rows[i] = LoadTmp(first + i);
  return out;
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::CompileMatrixInput(
    const mx::NodePtr& node, const std::string& input_name) {
  if (mx::NodePtr upstream = node->getConnectedNode(input_name)) {
    return CompileMatrixNode(upstream);
  }
  int n = InputType(node, input_name) == "matrix44" ? 4 : 3;
  mx::InputPtr val_in = GetValueInput(node, input_name);
  if (!val_in || !val_in->getValue()) return MatExpr::Identity(n);
  const std::string& t = val_in->getType();
  if (t != "matrix33" && t != "matrix44") return MatExpr::Identity(n);
  MatExpr out;
  if (t == "matrix44") {
    out.n = 4;
    mx::Matrix44 m = val_in->getValue()->asA<mx::Matrix44>();
    for (int i = 0; i < 4; ++i) {
      out.rows[i] = EmitTuple({m[i][0], m[i][1], m[i][2], m[i][3]});
    }
  } else {
    mx::Matrix33 m = val_in->getValue()->asA<mx::Matrix33>();
    for (int i = 0; i < 3; ++i) {
      out.rows[i] = EmitTuple({m[i][0], m[i][1], m[i][2]});
    }
  }
  return out;
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::CompileMatrixNode(
    const mx::NodePtr& node) {
  if (auto it = node_matrix_expr_.find(node.get());
      it != node_matrix_expr_.end()) {
    return it->second;
  }
  int n = node->getType() == "matrix44" ? 4 : 3;
  std::string cat = NodeCategory(node);

  MatExpr res = MatExpr::Identity(n);
  if (cat == "constant") {
    res = CompileMatrixInput(node, "value");
  } else if (cat == "dot") {
    res = CompileMatrixInput(node, "in");
  } else if (cat == "creatematrix") {
    // The rows of a 4x4 matrix are either `vector4` or `vector3` values, which
    // are extended by the fourth column of the identity.
    bool extend = n == 4 && InputType(node, "in1") != "vector4";
    for (int i = 0; i < n; ++i) {
      Expr row = CompileInputExpr(node, "in" + std::to_string(i + 1));
      res.rows[i] =
          extend ? vec4(row, i == 3 ? Expr::one() : Expr::zero()) : row;
    }
  } else if (cat == "transpose") {
    res = MatrixOp(Op::MatTranspose, CompileMatrixInput(node, "in"));
  } else if (cat == "invertmatrix") {
    res = MatrixOp(Op::MatInverse, CompileMatrixInput(node, "in"));
  } else if (cat == "add" || cat == "subtract") {
    MatExpr m1 = CompileMatrixInput(node, "in1");
    bool is_add = cat == "add";
    if (InputType(node, "in2") == "float") {
      Expr s = EmitTmpExpr(CompileInputExpr(node, "in2"));
      for (int i = 0; i < n; ++i) {
        res.rows[i] = is_add ? m1.rows[i] + s : m1.rows[i] - s;
      }
    } else {
      MatExpr m2 = CompileMatrixInput(node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] =
            is_add ? m1.rows[i] + m2.rows[i] : m1.rows[i] - m2.rows[i];
      }
    }
  } else if (cat == "multiply" || cat == "divide") {
    MatExpr m1 = CompileMatrixInput(node, "in1");
    MatExpr m2 = CompileMatrixInput(node, "in2");
    if (cat == "divide") m2 = MatrixOp(Op::MatInverse, m2);
    res = MatrixOp(Op::MatMul, m1, &m2);
  } else if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    Expr cond = CompileIfCondition(node, cat);
    if (cond.is_one() || cond.is_zero()) {
      res = CompileMatrixInput(node, cond.is_one() ? "in1" : "in2");
    } else {
      Expr cmp = EmitTmpExpr(cond);
      MatExpr m1 = CompileMatrixInput(node, "in1"),
              m2 = CompileMatrixInput(node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = select(cmp, m1.rows[i], m2.rows[i]);
      }
    }
  } else {
    WarnUnsupported("Unsupported MaterialX matrix node category '" + cat +
                    "'; using identity.");
  }

  for (int i = 0; i < n; ++i) {
    if (res.rows[i].code.size() > 1) res.rows[i] = EmitTmpExpr(res.rows[i]);
  }
  node_matrix_expr_[node.get()] = res;
  return res;
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

Expr MtlxExprCompiler::CompileNodeOutputExpr(const mx::NodePtr& node,
                                             const std::string& output_name) {
  std::string cat = NodeCategory(node);
  const std::string& type = node->getType();
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  auto in_slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return EmitTmpExpr(CompileInputExpr(node, name, def));
  };

  // Categories are ordered by how common they are in material networks.

  // Constants, routing, and channel ops
  if (cat == "constant") return in("value");
  if (cat == "dot" || cat == "convert") return in("in");
  if (cat == "extract") {
    int idx = ConstIntInput(node, "index").value_or(0);
    std::string_view in_type = InputType(node, "in");
    if (in_type == "matrix33" || in_type == "matrix44") {
      MatExpr mat = CompileMatrixInput(node, "in");
      return mat.rows[std::clamp(idx, 0, mat.n - 1)];
    }
    return in("in").chan(std::clamp(idx, 0, TypeChannels(in_type) - 1));
  }
  if (cat == "combine2") {
    if (IsFourChannelType(type)) {
      // A 3-channel value and a float, or two `vector2` values.
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
  if (cat == "sign") {
    Expr x = in_slot("in");
    return (x > Expr::zero()) - (x < Expr::zero());
  }
  if (cat == "modulo") {
    Expr x = in_slot("in1"), y = in_slot("in2", Expr::one());
    return x - y * call(Op::Floor, x / y);
  }
  if (cat == "smoothstep") {
    // Like OSL, a step function at `low` if `low >= high`.
    Expr lo = in_slot("low"), hi = in_slot("high", Expr::one()),
         x = in_slot("in");
    Expr t = EmitTmpExpr((x - lo) / (hi - lo));
    return select(
        x < lo, Expr::zero(),
        select(x >= hi, Expr::one(), t * t * (Const(3.0) - Const(2.0) * t)));
  }
  if (cat == "remap") {
    Expr ilo = in_slot("inlow"), olo = in_slot("outlow");
    return olo + (in("in") - ilo) / (in("inhigh", Expr::one()) - ilo) *
                     (in("outhigh", Expr::one()) - olo);
  }

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
  // Unary and binary ops that map to a single instruction
  static constexpr struct {
    std::string_view name;
    Op op;
  } kUnaryOps[] = {
      {"absval", Op::Abs},  {"floor", Op::Floor}, {"ceil", Op::Ceil},
      {"round", Op::Round}, {"fract", Op::Fract}, {"sin", Op::Sin},
      {"cos", Op::Cos},     {"tan", Op::Tan},     {"asin", Op::Asin},
      {"acos", Op::Acos},   {"sqrt", Op::Sqrt},   {"ln", Op::Log},
      {"exp", Op::Exp},
  };
  for (const auto& [name, op] : kUnaryOps) {
    if (cat == name) {
      return call(op, in("in", op == Op::Log ? Expr::one() : Expr::zero()));
    }
  }
  static constexpr struct {
    std::string_view name;
    const char *in1, *in2;
    Op op;
    bool def2_one;
  } kBinaryOps[] = {
      {"min", "in1", "in2", Op::Min, false},
      {"max", "in1", "in2", Op::Max, false},
      {"atan2", "iny", "inx", Op::Atan2, true},
      {"crossproduct", "in1", "in2", Op::Cross, false},
      {"rotate2d", "in", "amount", Op::Rotate2D, false},
  };
  for (const auto& [name, in1, in2, op, def2_one] : kBinaryOps) {
    if (cat == name) {
      Expr e1 = in(in1), e2 = in(in2, def2_one ? Expr::one() : Expr::zero());
      return call(op, e1, e2);
    }
  }

  // Conditionals & logical ops
  if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    Expr cond = CompileIfCondition(node, cat);
    if (type == "boolean") return cond;
    if (cond.is_one()) return in("in1");
    if (cond.is_zero()) return in("in2");
    Expr i1 = in("in1"), i2 = in("in2");
    return select(cond, i1, i2);
  }
  if (cat == "and" || cat == "or") {
    auto to_bool = [](const Expr& e) {
      return (e.is_zero() || e.is_one()) ? e
                                         : select(e, Expr::one(), Expr::zero());
    };
    Expr b1 = to_bool(in("in1")), b2 = to_bool(in("in2"));
    if (cat == "and") return b1 * b2;
    return (b1.is_one() || b2.is_one()) ? Expr::one()
           : b1.is_zero()               ? b2
           : b2.is_zero()               ? b1
                                        : call(Op::Max, b1, b2);
  }
  if (cat == "not") return !in("in");

  // Vector ops & normal map
  if (cat == "normalize" || cat == "magnitude" || cat == "dotproduct") {
    std::string_view in_type =
        InputType(node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_imm = in_type == "vector2" ? 2 : in_type == "vector4" ? 4 : 0;
    if (cat == "normalize") return call(Op::Normalize, vec_imm, in("in"));
    if (cat == "magnitude") {
      Expr v = in_slot("in");
      return call(Op::Sqrt, call(Op::Dot, vec_imm, v, v));
    }
    Expr i1 = in("in1"), i2 = in("in2");
    return call(Op::Dot, vec_imm, i1, i2);
  }
  if (cat == "rotate3d") {
    Expr x = in("in"), amt = in("amount"),
         axis = in("axis", ConstRgb(0.0, 1.0, 0.0));
    return call(Op::Rotate3D, x, amt, axis);
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

  // Matrix ops
  if (cat == "transformmatrix") {
    // Row vectors are transformed as `v * M`. The last coordinate of a vector
    // with fewer coordinates than the matrix is 1.
    Expr v = in_slot("in");
    MatExpr mat = CompileMatrixInput(node, "mat");
    Expr last = type == (mat.n == 4 ? "vector3" : "vector2")
                    ? Expr::one()
                    : v.chan(mat.n - 1);
    Expr res = mat.rows[0] * v.x() + mat.rows[1] * v.y();
    res = mat.n == 4 ? res + mat.rows[2] * v.z() + mat.rows[3] * last
                     : res + mat.rows[2] * last;
    if (type == "vector2") {
      res = EmitTmpExpr(res);
      return vec2(res.x(), res.y());
    }
    return type == "vector3" ? res.xyz() : res;
  }
  if (cat == "determinant") {
    MatExpr m = CompileMatrixInput(node, "in");
    Expr det =
        m.n == 4 ? call(Op::MatDet, m.rows[0], m.rows[1], m.rows[2], m.rows[3])
                 : call(Op::MatDet, m.rows[0], m.rows[1], m.rows[2]);
    det.code.back().channels = static_cast<uint8_t>(m.n);
    return det;
  }

  // Compositing & 2D patterns
  if (cat == "plus" || cat == "minus" || cat == "difference" ||
      cat == "screen" || cat == "burn" || cat == "dodge") {
    Expr fg = in_slot("fg"), bg = in_slot("bg"), m = in("mix", Expr::one());
    // Like MaterialX, burn and dodge are 0 where they divide by 0.
    if (cat == "burn" || cat == "dodge") {
      Expr denom = cat == "burn" ? fg : EmitTmpExpr(Expr::one() - fg);
      Expr blend =
          cat == "burn" ? Expr::one() - (Expr::one() - bg) / denom : bg / denom;
      return select(call(Op::Abs, denom) < Const(kMtlxFloatEps), Expr::zero(),
                    mix(blend, bg, m));
    }
    Expr blend = cat == "plus"         ? (bg + fg)
                 : cat == "minus"      ? (bg - fg)
                 : cat == "difference" ? call(Op::Abs, bg - fg)
                                       : (bg + fg - bg * fg);
    return mix(blend, bg, m);
  }
  if (cat == "inside") return in("in") * in("mask", Expr::one());
  if (cat == "outside") return in("in") * (Expr::one() - in("mask"));
  if (cat == "premult" || cat == "unpremult") {
    Expr in_e = in_slot("in");
    Expr color = in_e.xyz(), a = in_e.w();
    return vec4(cat == "premult" ? color * a : color / a, a);
  }
  if (cat == "over" || cat == "in" || cat == "out" || cat == "mask" ||
      cat == "matte" || cat == "disjointover") {
    Expr fg = in_slot("fg"), bg = in_slot("bg");
    Expr fa = fg.w(), ba = bg.w();
    Expr composite =
        cat == "over"   ? fg + bg * (Expr::one() - fa)
        : cat == "in"   ? fg * ba
        : cat == "out"  ? fg * (Expr::one() - ba)
        : cat == "mask" ? bg * fa
        : cat == "matte"
            ? fg * fa + bg * (Expr::one() - fa)
            : select(fa + ba <= Expr::one(), fg + bg,
                     select(call(Op::Abs, ba) < Const(kMtlxFloatEps),
                            Expr::zero(), fg + bg * ((Expr::one() - fa) / ba)));
    // The alpha channels of `matte` and `disjointover` are not composited like
    // their colors.
    if (cat == "matte") {
      composite = vec4(composite.xyz(), fa + ba * (Expr::one() - fa));
    } else if (cat == "disjointover") {
      composite = vec4(composite.xyz(), call(Op::Min, fa + ba, Expr::one()));
    }
    return mix(composite, bg, in("mix", Expr::one()));
  }
  if (cat == "ramplr" || cat == "ramptb" || cat == "splitlr" ||
      cat == "splittb") {
    bool lr = (cat == "ramplr" || cat == "splitlr");
    Expr v0 = in(lr ? "valuer" : "valuet"), v1 = in(lr ? "valuel" : "valueb");
    Expr tc = in("texcoord", DefaultTexcoord());
    Expr tc_comp = lr ? tc.x() : tc.y();
    if (cat == "ramplr" || cat == "ramptb") {
      return mix(v0, v1, saturate(tc_comp));
    }
    return select(tc_comp >= in("center", Const(0.5)), v0, v1);
  }

  // Color science & PBR helper nodes
  if (cat == "rgbtohsv" || cat == "hsvtorgb") {
    Expr e = call(cat == "rgbtohsv" ? Op::RgbToHsv : Op::HsvToRgb, in("in"));
    if (type == "color4") e.channels = 4;
    return e;
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
  if (cat == "blackbody") {
    return call(Op::Blackbody, in("temperature", Const(5000.0)));
  }
  if (cat == "roughness_anisotropy") {
    Expr r = in_slot("roughness");
    Expr r2 =
        EmitTmpExpr(call(Op::Clip, r * r, Const(kMtlxFloatEps), Expr::one()));
    Expr aspect = EmitTmpExpr(
        call(Op::Sqrt, Expr::one() - call(Op::Clip, in("anisotropy"),
                                          Expr::zero(), Const(0.98))));
    return vec2(call(Op::Min, r2 / aspect, Expr::one()), r2 * aspect);
  }
  if (cat == "roughness_dual") {
    Expr r = in_slot("roughness"), rx = EmitTmpExpr(r.x());
    Expr ry = EmitTmpExpr(select(r.y() < Expr::zero(), rx, r.y()));
    return vec2(call(Op::Clip, rx * rx, Const(kMtlxFloatEps), Expr::one()),
                call(Op::Clip, ry * ry, Const(kMtlxFloatEps), Expr::one()));
  }
  if (cat == "artistic_ior") {
    Expr r = EmitTmpExpr(
        call(Op::Clip, in("reflectivity"), Expr::zero(), Const(0.99)));
    Expr r_sqrt = EmitTmpExpr(call(Op::Sqrt, r));
    Expr eta =
        mix((Expr::one() - r) / (Expr::one() + r),
            (Expr::one() + r_sqrt) / (Expr::one() - r_sqrt), in("edge_color"));
    if (output_name != "extinction") return eta;
    eta = EmitTmpExpr(eta);
    Expr np1 = EmitTmpExpr(eta + Expr::one()),
         nm1 = EmitTmpExpr(eta - Expr::one());
    return call(Op::Sqrt, (np1 * np1 * r - nm1 * nm1) / (Expr::one() - r));
  }

  // Procedural noise nodes
  std::string_view noise = cat;
  bool is_3d = absl::ConsumeSuffix(&noise, "3d");
  if ((is_3d || absl::ConsumeSuffix(&noise, "2d")) &&
      (noise == "cellnoise" || noise == "worleynoise" || noise == "noise" ||
       noise == "fractal")) {
    uint16_t dim_bit = is_3d ? kNoise3D : 0;
    uint16_t imm = dim_bit | (type != "float" ? kNoiseVec3 : 0);
    Expr coord =
        is_3d ? in("position", MeshAttr("__p_object", 3, Expr::p_world()))
              : in("texcoord", DefaultTexcoord());
    if (noise == "cellnoise") {
      return call(Op::CellNoise, imm, coord);
    }
    if (noise == "worleynoise") {
      Expr jitter = in("jitter", Expr::one()), style = in("style");
      Expr w = call(Op::Worley, imm, coord, jitter, style);
      if (type == "vector2") w.channels = 2;
      return w;
    }
    bool is_noise = noise == "noise";
    std::optional<int> oct =
        is_noise ? std::nullopt : ConstIntInput(node, "octaves");
    bool four_ch = IsFourChannelType(type);
    bool split_vec2 = !is_noise && type == "vector2";
    // Operands of two noise calls are stored in temporaries.
    auto shared = [&](const Expr& e) {
      return four_ch || split_vec2 ? EmitTmpExpr(e) : e;
    };
    coord = shared(coord);
    Expr lac, dim, dyn_oct;
    if (!is_noise) {
      lac = shared(in("lacunarity", Const(2.0)));
      dim = shared(in("diminish", Const(0.5)));
      if (!oct) dyn_oct = shared(in("octaves"));
    }
    auto eval_one = [&](bool vec3, const Expr& p) {
      uint16_t bits = dim_bit | (vec3 ? kNoiseVec3 : 0);
      if (is_noise) return call(Op::Noise, bits, p);
      if (oct) {
        bits |= std::clamp(*oct, 0, kMaxFractalOctaves) << kFractalOctavesShift;
        return call(Op::Fractal, bits, p, lac, dim);
      }
      return call(Op::Fractal,
                  bits | (kFractalDynamicOctaves << kFractalOctavesShift), p,
                  lac, dim, dyn_oct);
    };
    auto offset_coord = [&]() {
      constexpr double kNoiseOffset[3] = {19.0, 73.0, 29.0};
      constexpr double kFractalOffset[3] = {19.0, 193.0, 17.0};
      const double* off = is_noise ? kNoiseOffset : kFractalOffset;
      if (is_3d) return coord + ConstRgb(off[0], off[1], off[2]);
      return coord + vec2(Const(off[0]), Const(off[1]));
    };
    Expr res;
    if (four_ch) {
      // Like MaterialX, a fourth channel is scalar noise at an offset.
      res = vec4(eval_one(true, coord), eval_one(false, offset_coord()));
    } else if (split_vec2) {
      // Like MaterialX, `vector2` fractals are two scalar fractals.
      res = vec2(eval_one(false, coord), eval_one(false, offset_coord()));
    } else {
      res = eval_one(type != "float", coord);
      if (type == "vector2") res.channels = 2;
    }
    res = res * in("amplitude", Expr::one());
    return is_noise ? res + in("pivot") : res;
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
  std::string cat = NodeCategory(node);
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
  std::string cat = NodeCategory(node);
  Expr res = Expr::one();
  if (cat == "surface" || cat == "surface_unlit") {
    res = saturate(CompileInputExpr(node, "opacity", Expr::one()));
  } else if (cat == "mix" &&
             node->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
    res = CompileMix(
        saturate(CompileInputExpr(node, "mix")),
        [&] { return CompileOpacity(node->getConnectedNode("fg")); },
        [&] { return CompileOpacity(node->getConnectedNode("bg")); });
  }
  if (!res.uniform()) res = call(Op::Dot, res, Const(1.0 / 3.0));
  if (res.code.size() > 1) res = EmitTmpExpr(res);
  opacity_expr_[node.get()] = res;
  return res;
}

int MtlxExprCompiler::EmitClosure(
    ClosureOp op, std::initializer_list<std::pair<int, int>> args) {
  ClosureNode& c = prog_.closures.emplace_back(op);
  for (auto [k, v] : args) c.args[k] = v;
  return static_cast<int>(prog_.closures.size()) - 1;
}

int MtlxExprCompiler::CompileBsdfClosure(const mx::NodePtr& raw_node) {
  mx::NodePtr node = UnwrapShaderNode(raw_node);
  if (!node) return -1;
  if (auto it = bsdf_closure_.find(node.get()); it != bsdf_closure_.end()) {
    return it->second;
  }

  std::string cat = NodeCategory(node);
  auto slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return EmitTmp(CompileInputExpr(node, name, def));
  };
  auto scale = [&](int c, const Expr& w) {
    return (c < 0 || w.is_one())
               ? c
               : EmitClosure(ClosureOp::Multiply,
                             {{combinator_arg::kFirst, c},
                              {combinator_arg::kWeight, EmitTmp(w)}});
  };

  auto frame_slot = [&](const std::string& name, const Expr& def) {
    Expr e = CompileInputExpr(node, name, def);
    return e == def ? -1 : EmitTmp(e);
  };

  int idx = -1;
  if (cat == "oren_nayar_diffuse_bsdf" || cat == "subsurface_bsdf" ||
      cat == "translucent_bsdf" || cat == "conductor_bsdf") {
    Expr w_expr = CompileInputExpr(node, "weight", Expr::one());
    if (w_expr.is_zero()) {
      // Opaque leaves still block `layer` bases (`throughput = 0`) at
      // `weight == 0`, without compiling dead inputs.
      int z = EmitTmp(Expr::zero());
      idx = EmitClosure(ClosureOp::OrenNayarDiffuse, {{leaf_arg::kWeight, z},
                                                      {leaf_arg::kColor, z},
                                                      {leaf_arg::kRoughness, z},
                                                      {leaf_arg::kMode, 0},
                                                      {leaf_arg::kNormal, -1}});
    } else if (cat == "conductor_bsdf") {
      auto is_nonzero_input = [&](const std::string& name) {
        mx::InputPtr in = GetValueInput(node, name);
        if (!in) return node->getConnectedNode(name) != nullptr;
        if (in->getType() == "boolean") {
          const std::string& s = in->getValueString();
          return s == "true" || s == "1";
        }
        mx::ValuePtr v = in->getValue();
        return v && v->isA<float>() && v->asA<float>() > 0.0f;
      };
      if (is_nonzero_input("thinfilm_thickness")) {
        WarnUnsupported(
            "Unsupported MaterialX 'conductor_bsdf' thinfilm_thickness; "
            "ignoring thin film.");
      }
      if (is_nonzero_input("retroreflective")) {
        WarnUnsupported(
            "Unsupported MaterialX 'conductor_bsdf' retroreflective; using "
            "standard reflection.");
      }
      idx = EmitClosure(
          ClosureOp::Conductor,
          {{leaf_arg::kWeight, EmitTmp(w_expr)},
           {leaf_arg::kColor, slot("ior", Expr::one())},
           {leaf_arg::kRoughness, slot("roughness")},
           {leaf_arg::kExtinction, slot("extinction")},
           {leaf_arg::kNormal, frame_slot("normal", Expr::n_world())},
           {leaf_arg::kTangent, frame_slot("tangent", Expr::t_world())}});
    } else {
      bool is_translucent = cat == "translucent_bsdf";
      idx = EmitClosure(
          is_translucent ? ClosureOp::Translucent : ClosureOp::OrenNayarDiffuse,
          {{leaf_arg::kWeight, EmitTmp(w_expr)},
           {leaf_arg::kColor, slot("color", Expr::one())},
           {leaf_arg::kRoughness, is_translucent ? -1 : slot("roughness")},
           {leaf_arg::kMode,
            cat == "oren_nayar_diffuse_bsdf" &&
                !CompileInputExpr(node, "energy_compensation").is_zero()},
           {leaf_arg::kNormal, frame_slot("normal", Expr::n_world())}});
    }
  } else if (cat == "layer" || cat == "add") {
    bool is_layer = cat == "layer";
    int c0 =
        CompileBsdfClosure(node->getConnectedNode(is_layer ? "top" : "in1"));
    int c1 =
        CompileBsdfClosure(node->getConnectedNode(is_layer ? "base" : "in2"));
    if (c0 < 0) {
      idx = c1;
    } else if (c1 < 0) {
      idx = c0;
    } else {
      idx = EmitClosure(
          is_layer ? ClosureOp::Layer : ClosureOp::Add,
          {{combinator_arg::kFirst, c0}, {combinator_arg::kSecond, c1}});
    }
  } else if (cat == "mix") {
    Expr mix_expr = CompileInputExpr(node, "mix");
    if (mix_expr.is_zero()) {
      idx = CompileBsdfClosure(node->getConnectedNode("bg"));
    } else if (mix_expr.is_one()) {
      idx = CompileBsdfClosure(node->getConnectedNode("fg"));
    } else {
      int fg_idx = CompileBsdfClosure(node->getConnectedNode("fg"));
      int bg_idx = CompileBsdfClosure(node->getConnectedNode("bg"));
      if (fg_idx == bg_idx) {
        idx = fg_idx;
      } else {
        if (fg_idx < 0) fg_idx = EmitClosure(ClosureOp::Null, {});
        if (bg_idx < 0) bg_idx = EmitClosure(ClosureOp::Null, {});
        idx = EmitClosure(ClosureOp::Mix,
                          {{combinator_arg::kFirst, fg_idx},
                           {combinator_arg::kSecond, bg_idx},
                           {combinator_arg::kWeight, EmitTmp(mix_expr)}});
      }
    }
  } else if (cat == "multiply") {
    Expr w = CompileInputExpr(node, "in2", Expr::one());
    idx = scale(CompileBsdfClosure(node->getConnectedNode("in1")), w);
  } else if (cat == "surface") {
    if (Expr op = CompileOpacity(node); !op.is_zero()) {
      idx = scale(CompileBsdfClosure(node->getConnectedNode("bsdf")), op);
    }
  } else if (cat == "surface_unlit") {
    // Unlit surfaces emit radiance only and have a null BSDF.
  } else if (node->getType() == mx::BSDF_TYPE_STRING &&
             InputType(node, "weight") == "float" &&
             CompileInputExpr(node, "weight", Expr::one()).is_zero()) {
    // Skip inactive unsupported lobes (e.g. zero-weight lobes in
    // `standard_surface`) without warning.
  } else {
    WarnUnsupported("Unsupported MaterialX closure '" + cat +
                    "'; using a null BSDF.");
  }

  bsdf_closure_[node.get()] = idx;
  return idx;
}

void MtlxExprCompiler::CompileBsdf(const mx::NodePtr& shader_node) {
  mx::NodePtr node = UnwrapShaderNode(shader_node);
  Expr op_expr = CompileOpacity(node);
  if (!op_expr.is_one()) prog_.opacity_slot = EmitTmp(op_expr);
  if (op_expr.is_zero() || CompileBsdfClosure(node) < 0) {
    prog_.closures = {ClosureNode(ClosureOp::Null)};
  }
}

ref<Object> MtlxExprCompiler::CreatePlugin(
    std::string_view name, ObjectType type, std::string_view variant,
    MeshAttributeRequests& mesh_attributes) {
  Properties props(name);
  for (size_t i = 0; i < inputs_.size(); ++i) {
    props.set("in" + std::to_string(i), std::move(inputs_[i]));
  }
  mesh_attributes.insert(attr_requests_.begin(), attr_requests_.end());
  props.set_any("program", std::move(prog_));
  return PluginManager::instance()->create_object(props, variant, type);
}

ref<Object> MtlxExprCompiler::CreateExprTexture(
    const Expr& expr, bool unbounded, std::string_view variant,
    MeshAttributeRequests& mesh_attributes) {
  RegisterMaterialXExprTexture();
  prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
  prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
  prog_.result_uniform = expr.uniform();
  prog_.unbounded = unbounded;
  return CreatePlugin("materialx_expr", ObjectType::Texture, variant,
                      mesh_attributes);
}

ref<Object> MtlxExprCompiler::CreateBsdf(
    std::string_view variant, MeshAttributeRequests& mesh_attributes) {
  RegisterMaterialXBsdf();
  return CreatePlugin("materialx_bsdf", ObjectType::BSDF, variant,
                      mesh_attributes);
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
    compiler.CompileBsdf(shader_node);
    out.bsdf = compiler.CreateBsdf(variant, out.mesh_attributes);
  }
  {
    MtlxExprCompiler compiler(lookup_texture, warned);
    Expr radiance = compiler.CompileEdf(shader_node);
    if (!radiance.is_zero()) {
      out.radiance_texture = compiler.CreateExprTexture(
          radiance, /*unbounded=*/true, variant, out.mesh_attributes);
    }
  }
  return out;
}

void DiscoverSurfaceShaderTextures(
    const mx::NodePtr& shader_node,
    const std::function<void(const TextureKey&)>& callback) {
  MtlxExprCompiler compiler(callback);
  compiler.CompileBsdf(shader_node);
  compiler.CompileEdf(shader_node);
}

PXR_NAMESPACE_CLOSE_SCOPE
