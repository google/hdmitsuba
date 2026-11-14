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

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/synchronization/mutex.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "hdmitsuba/materialx/vm.h"
#include "hdmitsuba/utils.h"

NAMESPACE_BEGIN(mitsuba)
namespace mtlx {

template <typename... Exprs>
inline Expr call(Op op, uint16_t imm, const Exprs&... args);
template <typename... Exprs>
inline Expr call(Op op, const Expr& arg0, const Exprs&... rest);

// Reverse-Polish-Notation expression fragment that evaluates to one stack slot.
// Constants 0.0 and 1.0 are always at indices 0 and 1 in `MtlxProgram::consts`.
struct Expr {
  std::vector<Inst> code;
  uint8_t channels = 3;
  uint32_t max_depth = 1;

  Expr() = default;
  explicit Expr(Inst i, uint8_t channels = 3) : code{i}, channels(channels) {}

  static Expr zero() { return Expr({Op::Const, 0, 0}, 1); }
  static Expr one() { return Expr({Op::Const, 0, 1}, 1); }
  static Expr in(int k, bool uniform = false) {
    uint8_t ch = uniform ? 1 : 3;
    return Expr({Op::SampleInput, ch, static_cast<uint16_t>(k)}, ch);
  }
  static Expr uv() { return Expr({Op::GeomUV, 0, 0}, 2); }
  static Expr p_world() { return Expr({Op::GeomPosWorld, 0, 0}); }
  static Expr n_world() { return Expr({Op::GeomNormalWorld, 0, 0}); }
  static Expr t_world() { return Expr({Op::GeomTangentWorld, 0, 0}); }
  static Expr b_world() { return Expr({Op::GeomBitangentWorld, 0, 0}); }
  static Expr incident_dir() { return Expr({Op::GeomIncidentDir, 0, 0}); }

  bool empty() const { return code.empty(); }
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

  Expr operator-() const { return call(Op::Neg, *this); }
  Expr operator!() const { return call(Op::Eq, *this, zero()); }
  Expr operator+(const Expr& b) const {
    return is_zero() ? b : b.is_zero() ? *this : call(Op::Add, *this, b);
  }
  Expr operator-(const Expr& b) const {
    return b.is_zero() ? *this : *this == b ? zero() : call(Op::Sub, *this, b);
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
inline Expr call(Op op, uint16_t imm, const Exprs&... args) {
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
inline Expr call(Op op, const Expr& arg0, const Exprs&... rest) {
  return call(op, uint16_t{0}, arg0, rest...);
}

inline Expr rgb(const Expr& r, const Expr& g, const Expr& b) {
  return (r.uniform() && r == g && g == b) ? r : call(Op::Rgb, r, g, b);
}
inline Expr vec2(const Expr& x, const Expr& y) { return call(Op::Vec2, x, y); }
inline Expr select(const Expr& cond, const Expr& t, const Expr& f) {
  return call(Op::Select, cond, t, f);
}
inline Expr sample2d(int k, const Expr& u, const Expr& v,
                     bool uniform = false) {
  Expr e = call(Op::SampleInputUV, static_cast<uint16_t>(k), u, v);
  e.channels = e.code.back().channels = uniform ? 1 : 3;
  return e;
}
inline Expr eq(const Expr& a, const Expr& b) { return call(Op::Eq, a, b); }
inline Expr ne(const Expr& a, const Expr& b) { return call(Op::Ne, a, b); }
inline Expr logical_and(const Expr& a, const Expr& b) {
  return call(Op::Mul, ne(a, Expr::zero()), ne(b, Expr::zero()));
}
inline Expr logical_or(const Expr& a, const Expr& b) {
  return call(Op::Max, ne(a, Expr::zero()), ne(b, Expr::zero()));
}

Expr MtlxProgram::c(double value) {
  auto it = std::find(consts.begin(), consts.end(), value);
  if (it == consts.end()) {
    if (consts.size() >= kMaxImm) Throw("mtlx_vm: too many constants");
    it = consts.insert(consts.end(), value);
  }
  return Expr({Op::Const, 0, static_cast<uint16_t>(it - consts.begin())}, 1);
}

Expr MtlxProgram::c_rgb(double r, double g, double b) {
  return (r == g && g == b) ? c(r) : rgb(c(r), c(g), c(b));
}

Expr MtlxProgram::c_vec2(double x, double y) { return vec2(c(x), c(y)); }

Expr MtlxProgram::param(double value) {
  if (params.size() >= kMaxImm) Throw("mtlx_vm: too many parameters");
  params.push_back(value);
  return Expr({Op::Param, 0, static_cast<uint16_t>(params.size() - 1)}, 1);
}

Expr MtlxProgram::attr(bool three, std::string_view name, const Expr& def) {
  auto it = std::find(attr_names.begin(), attr_names.end(), name);
  if (it == attr_names.end()) {
    it = attr_names.insert(it, std::string(name));
  }
  Expr e = def;
  // Shapes lacking the attribute evaluate it to zero, so a zero default
  // needs no `Shape::has_attribute()` call.
  e.code.push_back({three ? Op::Attr3 : Op::Attr1,
                    static_cast<uint8_t>(def.is_zero()),
                    static_cast<uint16_t>(it - attr_names.begin())});
  e.channels = (!three && def.uniform()) ? 1 : 3;
  return e;
}

int MtlxProgram::add_tmp(const Expr& val) {
  if (num_tmps >= kMaxImm) Throw("mtlx_vm: too many temporaries");
  code.insert(code.end(), val.code.begin(), val.code.end());
  code.push_back({Op::StoreTmp, 0, static_cast<uint16_t>(num_tmps)});
  stack_size = std::max(stack_size, val.max_depth);
  tmp_channels.push_back(val.channels);
  return static_cast<int>(num_tmps++);
}

Expr MtlxProgram::tmp(int k) const {
  return Expr({Op::LoadTmp, 0, static_cast<uint16_t>(k)}, tmp_channels[k]);
}

MtlxProgram& MtlxProgram::finish(const Expr& result) {
  code.insert(code.end(), result.code.begin(), result.code.end());
  stack_size = std::max(stack_size, result.max_depth);
  result_uniform = result.uniform();
  return *this;
}

// Compiles a flattened MaterialX DAG of pattern/math/procedural/image nodes
// into a `mitsuba::mtlx::MtlxProgram` and a list of Mitsuba Texture child
// inputs (`in0`, `in1`, ...).
class MtlxExprCompiler {
 public:
  explicit MtlxExprCompiler(const MtlxCompileOptions& opts);

  MtlxProgram& program() { return prog_; }
  std::string_view variant() const { return variant_; }

  Expr Const(double v) { return prog_.c(v); }
  Expr ConstRgb(double r, double g, double b) { return prog_.c_rgb(r, g, b); }
  Expr ConstVec2(double x, double y) { return prog_.c_vec2(x, y); }

  Expr CompileInputExpr(const MaterialX::NodePtr& node,
                        const std::string& input_name,
                        const Expr& default_expr);

  Expr CompileInputAlphaExpr(const MaterialX::NodePtr& node,
                             const std::string& input_name);

  int EmitTmp(const Expr& expr);
  Expr EmitTmpExpr(const Expr& expr) { return prog_.tmp(EmitTmp(expr)); }

  void PopulateProperties(Properties& props,
                          const Expr& final_expr = Expr()) const;

  ref<Object> CreateExprTexture(const Expr& expr) const;

 private:
  Expr CompileInputSlotExpr(const MaterialX::NodePtr& node,
                            const std::string& input_name,
                            const Expr& default_expr) {
    return EmitTmpExpr(CompileInputExpr(node, input_name, default_expr));
  }
  Expr CompileNodeOutputExpr(const MaterialX::NodePtr& node,
                             const std::string& output_name);
  Expr CompileNodeAlphaExpr(const MaterialX::NodePtr& node);
  Expr Magnitude4(const MaterialX::NodePtr& node,
                  const std::string& input_name);
  Expr NoiseCoord(const MaterialX::NodePtr& node);
  Expr PositionExpr(const std::string& space);
  Expr NormalExpr(const std::string& space);
  Expr MeshAttr(std::string_view name, int dim, const Expr& def);
  Expr TangentNormalMapExpr(const Expr& in_e, const Expr& scale,
                            const MaterialX::NodePtr& node);
  Expr NoiseExpr(const MaterialX::NodePtr& node, Expr coord, bool vec3,
                 const Expr& amplitude, bool offset = false);
  ref<Object> LookupTexture(const TextureKey& key) const;
  Expr SampleImage(const MaterialX::NodePtr& node, const TextureKey& key,
                   const Expr& out_of_bounds_val);
  Expr CompileValueElement(const MaterialX::ValueElementPtr& elem,
                           const Expr& default_expr);
  int RegisterTextureInput(ref<Object> tex_obj);

  TextureLookup lookup_texture_;
  PathResolver resolve_path_;
  std::string variant_;
  MeshAttributeRequests* mesh_attributes_ = nullptr;
  MtlxProgram prog_;
  absl::flat_hash_map<std::tuple<const MaterialX::Node*, std::string, uint8_t>,
                      Expr>
      node_output_expr_;
  absl::flat_hash_map<const MaterialX::Node*, Expr> node_alpha_expr_;
  const MaterialX::Node* alpha_node_ = nullptr;
  absl::flat_hash_map<const Object*, int> input_key_to_idx_;
  absl::flat_hash_map<Expr, int> expr_to_slot_;
  std::vector<ref<Object>> inputs_;
};

namespace {

namespace mx = MaterialX;
using pxr::CreateExpandedObject;
using pxr::MeshAttributeName;

constexpr double kNoiseOffset[3] = {19.0, 73.0, 29.0};
constexpr double kFractalOffset[3] = {19.0, 193.0, 17.0};

void WarnOnce(const char* fmt, std::string_view name) {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<
      absl::flat_hash_set<std::pair<const char*, std::string>>>
      warned;
  absl::MutexLock lock(*mu);
  if (warned->emplace(fmt, name).second) {
    Log(Warn, fmt, std::string(name).c_str());
  }
}

// Expression of the default `texcoord` node (UV0), which flips v.
Expr DefaultTexcoord() {
  return vec2(Expr::uv().x(), Expr::one() - Expr::uv().y());
}

// Emits a material parameter value. 0 and 1 are baked into the program, as
// they enable simplifications (e.g. of mix weights and products). Other values
// become opaque parameters, so that editing them reuses compiled kernels.
Expr EmitParam(MtlxProgram& prog, double v) {
  return v == 0.0 ? Expr::zero() : v == 1.0 ? Expr::one() : prog.param(v);
}

// Multi-channel tuples only bake all-zero or all-one values, so that editing a
// non-trivial color/vector parameter (e.g. (1, 0, 0.2) -> (0.2, 0.5, 0.8))
// preserves the number of parameters and reuses the compiled kernel.
Expr EmitRgb(MtlxProgram& prog, double r, double g, double b) {
  return (r == 0.0 && g == 0.0 && b == 0.0) ? Expr::zero()
         : (r == 1.0 && g == 1.0 && b == 1.0)
             ? Expr::one()
             : rgb(prog.param(r), prog.param(g), prog.param(b));
}

mx::InputPtr GetValueInput(const mx::NodePtr& node, const std::string& name) {
  for (const mx::InterfaceElementPtr& el :
       {mx::InterfaceElementPtr(node),
        mx::InterfaceElementPtr(node->getNodeDef())}) {
    if (mx::InputPtr in = el ? el->getActiveInput(name) : nullptr;
        in && in->hasValueString()) {
      return in;
    }
  }
  return nullptr;
}

std::string GetNodeInputString(const mx::NodePtr& node, const std::string& name,
                               const std::string& default_val = "") {
  mx::InputPtr in = GetValueInput(node, name);
  return in ? in->getValueString() : default_val;
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
// by its node definition, or an empty string if there is no such input.
std::string InputType(const mx::NodePtr& node, const std::string& name) {
  if (mx::InputPtr in = node->getInput(name)) return in->getType();
  if (mx::NodeDefPtr nd = node->getNodeDef()) {
    if (mx::InputPtr in = nd->getActiveInput(name)) return in->getType();
  }
  return "";
}

bool IsFourChannelType(std::string_view type) {
  return type == "color4" || type == "vector4";
}

// Categories of element-wise nodes, whose alpha channel is computed by
// applying the node to the alpha channels of its inputs (see
// `CompileNodeAlphaExpr`).
bool IsElementwiseCategory(std::string_view cat) {
  static const absl::NoDestructor<absl::flat_hash_set<std::string_view>> kCats(
      {"absval",     "acos",     "add",        "asin",        "atan2",
       "burn",       "ceil",     "clamp",      "constant",    "cos",
       "difference", "divide",   "dodge",      "exp",         "floor",
       "fract",      "ifequal",  "ifgreater",  "ifgreatereq", "in",
       "inside",     "invert",   "ln",         "mask",        "max",
       "min",        "minus",    "mix",        "modulo",      "multiply",
       "out",        "outside",  "over",       "plus",        "power",
       "ramplr",     "ramptb",   "remap",      "round",       "screen",
       "sign",       "sin",      "smoothstep", "splitlr",     "splittb",
       "sqrt",       "subtract", "switch",     "tan"});
  return kCats->contains(cat);
}

// Returns whether an image of output `type` and `color_space` should skip
// sRGB-to-linear decoding. Non-color images (`float`, `vector*`) are always
// raw; `color3` and `color4` images are raw only when tagged with a linear or
// data color space.
bool IsRawImage(std::string_view type, std::string_view color_space) {
  if (type != "color3" && type != "color4") return true;
  std::string cs = absl::AsciiStrToLower(color_space);
  return cs == "none" || cs == "raw" || cs == "data" || cs == "identity" ||
         cs == "linear" || cs == "acescg" || absl::StartsWith(cs, "lin_");
}

std::optional<TextureKey> ExtractMtlxImageKey(
    const mx::NodePtr& node, const PathResolver& resolve_path) {
  mx::InputPtr file_in = GetValueInput(node, "file");
  if (!file_in) return std::nullopt;
  std::string file = file_in->getActiveFilePrefix() + file_in->getValueString();
  if (file.empty()) return std::nullopt;
  std::string resolved = resolve_path ? resolve_path(file) : std::string();
  if (resolved.empty()) {
    fs::path p = file_resolver()->resolve(file);
    if (fs::exists(p)) resolved = p.string();
  }
  if (resolved.empty()) return std::nullopt;

  std::string cs = file_in->getActiveColorSpace();
  if (cs.empty()) cs = node->getActiveColorSpace();
  std::string u_mode = GetNodeInputString(node, "uaddressmode", "periodic");
  std::string v_mode = GetNodeInputString(node, "vaddressmode", "periodic");
  TextureKey key;
  key.filename = std::move(resolved);
  key.raw = IsRawImage(node->getType(), cs);
  // Mitsuba's `bitmap` supports a single `wrap_mode`; mixed or `"constant"`
  // address modes clamp at the border and remap UVs in `SampleImage`.
  key.wrap_mode =
      (u_mode != v_mode || u_mode == "clamp" || u_mode == "constant") ? "clamp"
      : u_mode == "mirror" ? "mirror"
                           : "repeat";
  key.filter_type =
      GetNodeInputString(node, "filtertype", "linear") == "closest"
          ? "nearest"
          : "bilinear";
  return key;
}

mx::NodePtr UnwrapShaderNode(mx::NodePtr node) {
  while (node &&
         (node->getCategory() == "dot" || node->getCategory() == "convert" ||
          node->getCategory() == "surfacematerial")) {
    node = node->getConnectedNode(
        node->getCategory() == "surfacematerial" ? "surfaceshader" : "in");
  }
  return node;
}

// Compiles a MaterialX EDF or surface emission DAG (`surface`, `surface_unlit`,
// `uniform_edf`, `generalized_schlick_edf`, `multiply`, `add`, `mix`) into a
// radiance `Expr`, or returns `Expr::zero()` if there is no emission.
Expr CompileEdfNode(const mx::NodePtr& edf_node,
                    MtlxExprCompiler& expr_compiler,
                    absl::flat_hash_map<const mx::Node*, Expr>& memo) {
  mx::NodePtr node = UnwrapShaderNode(edf_node);
  if (!node) return Expr::zero();
  if (auto it = memo.find(node.get()); it != memo.end()) return it->second;
  auto compile_input = [&](const std::string& name) {
    return CompileEdfNode(node->getConnectedNode(name), expr_compiler, memo);
  };
  const std::string& cat = node->getCategory();
  Expr res = Expr::zero();
  if (cat == "surface" || cat == "surface_unlit") {
    if (cat == "surface") {
      res = compile_input("edf");
    } else {
      Expr em = expr_compiler.CompileInputExpr(node, "emission", Expr::one());
      if (!em.is_zero()) {
        res = em * expr_compiler.CompileInputExpr(node, "emission_color",
                                                  Expr::one());
      }
    }
    if (!res.is_zero()) {
      Expr op = expr_compiler.CompileInputExpr(node, "opacity", Expr::one());
      res = op.is_zero()  ? Expr::zero()
            : op.is_one() ? res
                          : res * call(Op::Clip, op, Expr::zero(), Expr::one());
    }
  } else if (cat == "uniform_edf" || cat == "conical_edf" ||
             cat == "measured_edf") {
    if (cat != "uniform_edf") {
      WarnOnce("Approximating MaterialX '%s' as uniform_edf.", cat);
    }
    res = expr_compiler.CompileInputExpr(node, "color", Expr::one());
  } else if (cat == "generalized_schlick_edf") {
    Expr base = compile_input("base");
    if (!base.is_zero()) {
      Expr c0 = expr_compiler.CompileInputExpr(node, "color0", Expr::one());
      Expr c90 = expr_compiler.CompileInputExpr(node, "color90", Expr::one());
      Expr exp = expr_compiler.EmitTmpExpr(
          call(Op::Max,
               expr_compiler.CompileInputExpr(node, "exponent",
                                              expr_compiler.Const(5.0)),
               Expr::zero()));
      // Weight `base` by the cosine-weighted hemispherical average of
      // `mix(c0, c90, (1 - cos)^exp)`, which equals
      // `mix(c0, c90, 2 / ((exp + 1) * (exp + 2)))`.
      Expr w = (c0 == c90) ? c0
                           : call(Op::Mix, c90, c0,
                                  expr_compiler.Const(2.0) /
                                      ((exp + Expr::one()) *
                                       (exp + expr_compiler.Const(2.0))));
      res = base * w;
    }
  } else if (cat == "add") {
    res = compile_input("in1") + compile_input("in2");
  } else if (cat == "multiply") {
    Expr w = expr_compiler.CompileInputExpr(node, "in2", Expr::one());
    if (!w.is_zero()) res = compile_input("in1") * w;
  } else if (cat == "mix") {
    Expr m = expr_compiler.CompileInputExpr(node, "mix", Expr::zero());
    if (m.is_zero()) {
      res = compile_input("bg");
    } else if (m.is_one()) {
      res = compile_input("fg");
    } else {
      Expr fg = compile_input("fg");
      Expr bg = compile_input("bg");
      res = fg == bg ? bg : call(Op::Mix, fg, bg, m);
    }
  }
  if (res.code.size() > 1) res = expr_compiler.EmitTmpExpr(res);
  memo[node.get()] = res;
  return res;
}

enum NodeCtxDep : uint8_t {
  kCtxDepNone = 0,
  kCtxDepValue = 1 << 0,
  kCtxDepAlphaValue = 1 << 1,
};

}  // namespace

void FlattenMtlxDocument(const mx::DocumentPtr& doc) {
  for (const mx::NodeGraphPtr& ng : doc->getChildrenOfType<mx::NodeGraph>()) {
    if (!ng->hasSourceUri()) ng->flattenSubgraphs();
  }
  doc->flattenSubgraphs();
}

void CollectMtlxImageKeys(
    const mx::DocumentPtr& doc, const PathResolver& resolve_path,
    const std::function<void(const TextureKey&)>& callback) {
  auto visit_graph = [&](auto& self, const mx::GraphElementPtr& graph) -> void {
    for (const mx::NodePtr& node : graph->getNodes()) {
      if (NodeCategory(node) != "image") continue;
      if (auto key = ExtractMtlxImageKey(node, resolve_path)) {
        callback(*key);
        if (IsFourChannelType(node->getType())) {
          key->raw = true;
          key->channel = 3;
          callback(*key);
        }
      }
    }
    for (const mx::NodeGraphPtr& ng :
         graph->getChildrenOfType<mx::NodeGraph>()) {
      if (!ng->hasSourceUri()) self(self, ng);
    }
  };
  visit_graph(visit_graph, doc);
}

MtlxExprCompiler::MtlxExprCompiler(const MtlxCompileOptions& opts)
    : lookup_texture_(opts.lookup_texture),
      resolve_path_(opts.resolve_path),
      variant_(opts.variant),
      mesh_attributes_(opts.mesh_attributes) {}

int MtlxExprCompiler::EmitTmp(const Expr& expr) {
  if (expr.code.size() == 1 && expr.code[0].op == Op::LoadTmp) {
    return expr.code[0].imm;
  }
  auto [it, inserted] =
      expr_to_slot_.try_emplace(expr, static_cast<int>(prog_.num_tmps));
  if (!inserted) return it->second;
  return prog_.add_tmp(expr);
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

  auto emit_rgb = [&](const auto& v) {
    return EmitRgb(prog_, v[0], v[1], v[2]);
  };

  if (type == "float") return EmitParam(prog_, val->asA<float>());
  // Integers and booleans select code paths (e.g. octaves, enums) and are
  // baked.
  if (type == "integer") return Const(val->asA<int>());
  if (type == "color3") return emit_rgb(val->asA<mx::Color3>());
  if (type == "color4") return emit_rgb(val->asA<mx::Color4>());
  if (type == "vector2") {
    mx::Vector2 v = val->asA<mx::Vector2>();
    return (v[0] == 0.0f && v[1] == 0.0f) ? Expr::zero()
           : (v[0] == 1.0f && v[1] == 1.0f)
               ? Expr::one()
               : vec2(prog_.param(v[0]), prog_.param(v[1]));
  }
  if (type == "vector3") return emit_rgb(val->asA<mx::Vector3>());
  if (type == "vector4") return emit_rgb(val->asA<mx::Vector4>());
  return default_expr;
}

Expr MtlxExprCompiler::CompileInputExpr(const mx::NodePtr& node,
                                        const std::string& input_name,
                                        const Expr& default_expr) {
  if (node.get() == alpha_node_ &&
      IsFourChannelType(InputType(node, input_name))) {
    return CompileInputAlphaExpr(node, input_name);
  }
  if (mx::InputPtr input = node->getInput(input_name)) {
    if (mx::NodePtr upstream = input->getConnectedNode()) {
      std::string out_name = ConnectedOutputName(input);
      std::tuple<const mx::Node*, std::string, uint8_t> key(
          upstream.get(), out_name, kCtxDepNone);
      auto it = node_output_expr_.find(key);
      if (it == node_output_expr_.end()) {
        Expr se = CompileNodeOutputExpr(upstream, out_name);
        if (se.code.size() > 1 && se != DefaultTexcoord()) {
          se = EmitTmpExpr(se);
        }
        it = node_output_expr_.emplace(std::move(key), std::move(se)).first;
      }
      return it->second;
    }
  }
  mx::InputPtr val_in = GetValueInput(node, input_name);
  if (!val_in) return default_expr;
  auto [it, inserted] =
      node_output_expr_.try_emplace({node.get(), input_name, kCtxDepValue});
  if (inserted) it->second = CompileValueElement(val_in, default_expr);
  return it->second;
}

int MtlxExprCompiler::RegisterTextureInput(ref<Object> tex_obj) {
  auto [it, inserted] = input_key_to_idx_.try_emplace(
      tex_obj.get(), static_cast<int>(inputs_.size()));
  if (inserted) inputs_.push_back(std::move(tex_obj));
  return it->second;
}

ref<Object> MtlxExprCompiler::LookupTexture(const TextureKey& key) const {
  const CachedTexture* c = lookup_texture_ ? lookup_texture_(key) : nullptr;
  return c ? c->texture : nullptr;
}

Expr MtlxExprCompiler::SampleImage(const mx::NodePtr& node,
                                   const TextureKey& key,
                                   const Expr& out_of_bounds_val) {
  ref<Object> tex = LookupTexture(key);
  if (!tex) return out_of_bounds_val;
  std::string u_mode = GetNodeInputString(node, "uaddressmode", "periodic");
  std::string v_mode = GetNodeInputString(node, "vaddressmode", "periodic");
  bool has_const = u_mode == "constant" || v_mode == "constant";
  bool mixed_wrap = u_mode != v_mode || has_const;

  // Alpha keys on RGB images resolve to `UniformSpectrum` (1.0).
  bool is_uniform_one =
      key.channel == 3 && tex->class_name() == "UniformSpectrum";
  if (is_uniform_one && !has_const) return Expr::one();

  Expr tc = CompileInputExpr(node, "texcoord", DefaultTexcoord());
  Expr sample = Expr::one();
  if (!is_uniform_one) {
    int in_idx = RegisterTextureInput(std::move(tex));
    bool mono = node->getType() == "float" || key.channel >= 0;
    if (!mixed_wrap && tc == DefaultTexcoord()) {
      sample = Expr::in(in_idx, mono);
    } else {
      tc = EmitTmpExpr(tc);
      auto wrap_axis = [&](Expr c, const std::string& mode) {
        if (!mixed_wrap || mode == "clamp" || mode == "constant") return c;
        if (mode == "mirror") {
          return Expr::one() -
                 call(Op::Abs, call(Op::Fract, c * Const(0.5)) * Const(2.0) -
                                   Expr::one());
        }
        return call(Op::Fract, c);
      };
      sample = sample2d(in_idx, wrap_axis(tc.x(), u_mode),
                        Expr::one() - wrap_axis(tc.y(), v_mode), mono);
    }
  }
  if (has_const) {
    tc = EmitTmpExpr(tc);
    auto oob_axis = [&](Expr c) {
      return logical_or(c<Expr::zero(), c> Expr::one());
    };
    Expr oob = (u_mode == "constant" && v_mode == "constant")
                   ? logical_or(oob_axis(tc.x()), oob_axis(tc.y()))
               : u_mode == "constant" ? oob_axis(tc.x())
                                      : oob_axis(tc.y());
    sample = select(oob, out_of_bounds_val, sample);
  }
  if (key.channel < 0 && node->getType() == "vector2") sample.channels = 2;
  return sample;
}

Expr MtlxExprCompiler::CompileInputAlphaExpr(const mx::NodePtr& node,
                                             const std::string& input_name) {
  std::string type = InputType(node, input_name);
  if (type == "float") return CompileInputExpr(node, input_name, Expr::one());
  if (!IsFourChannelType(type)) return Expr::one();
  if (mx::NodePtr upstream = node->getConnectedNode(input_name)) {
    return CompileNodeAlphaExpr(upstream);
  }
  if (mx::InputPtr val_in = GetValueInput(node, input_name)) {
    if (mx::ValuePtr val = val_in->getValue()) {
      auto [it, inserted] = node_output_expr_.try_emplace(
          {node.get(), input_name, kCtxDepAlphaValue});
      if (inserted) {
        it->second =
            EmitParam(prog_, type == "color4" ? val->asA<mx::Color4>()[3]
                                              : val->asA<mx::Vector4>()[3]);
      }
      return it->second;
    }
  }
  return Expr::one();
}

Expr MtlxExprCompiler::CompileNodeAlphaExpr(const mx::NodePtr& node) {
  if (auto it = node_alpha_expr_.find(node.get());
      it != node_alpha_expr_.end()) {
    return it->second;
  }
  std::string cat = NodeCategory(node);
  Expr alpha = Expr::one();
  if (cat == "image" && IsFourChannelType(node->getType())) {
    auto rgb_key = ExtractMtlxImageKey(node, resolve_path_);
    Expr def_a = CompileInputAlphaExpr(node, "default");
    if (!rgb_key || !LookupTexture(*rgb_key)) {
      alpha = def_a;
    } else {
      rgb_key->raw = true;
      rgb_key->channel = 3;
      if (LookupTexture(*rgb_key)) alpha = SampleImage(node, *rgb_key, def_a);
    }
  } else if (IsElementwiseCategory(cat) || cat == "transformmatrix") {
    const mx::Node* outer = std::exchange(alpha_node_, node.get());
    alpha = CompileNodeOutputExpr(node, "out");
    alpha_node_ = outer;
  } else if (cat == "dot" || cat == "convert" || cat == "luminance" ||
             cat == "rgbtohsv" || cat == "hsvtorgb" || cat == "premult" ||
             cat == "unpremult") {
    alpha = CompileInputAlphaExpr(node, "in");
  } else if (cat == "combine2") {
    // `color4` or `vector4` from a 3-channel value and a float, or from two
    // `vector2` values.
    Expr in2 = CompileInputExpr(node, "in2", Expr::zero());
    alpha = InputType(node, "in2") == "vector2" ? in2.y() : in2;
  } else if (cat == "combine4") {
    alpha = CompileInputExpr(node, "in4", Expr::zero());
  } else if (cat == "normalize") {
    alpha = CompileInputAlphaExpr(node, "in") / Magnitude4(node, "in");
  } else if (cat == "matte" || cat == "disjointover") {
    Expr fa = CompileInputAlphaExpr(node, "fg");
    Expr ba = CompileInputAlphaExpr(node, "bg");
    Expr composite = cat == "matte" ? fa + ba * (Expr::one() - fa)
                                    : call(Op::Min, fa + ba, Expr::one());
    alpha = call(Op::Mix, composite, ba,
                 CompileInputExpr(node, "mix", Expr::one()));
  } else if (cat == "noise2d" || cat == "noise3d" || cat == "fractal2d" ||
             cat == "fractal3d") {
    alpha = NoiseExpr(node, NoiseCoord(node), false,
                      CompileInputAlphaExpr(node, "amplitude"),
                      /*offset=*/true);
  }
  if (alpha.code.size() > 1) alpha = EmitTmpExpr(alpha);
  node_alpha_expr_[node.get()] = alpha;
  return alpha;
}

Expr MtlxExprCompiler::Magnitude4(const mx::NodePtr& node,
                                  const std::string& input_name) {
  Expr v = CompileInputSlotExpr(node, input_name, Expr::zero());
  Expr a = CompileInputAlphaExpr(node, input_name);
  return call(Op::Sqrt, call(Op::Dot, v, v) + a * a);
}

Expr MtlxExprCompiler::MeshAttr(std::string_view name, int dim,
                                const Expr& def) {
  if (mesh_attributes_) mesh_attributes_->emplace(name, dim);
  return prog_.attr(dim == 3, MeshAttributeName(name, dim), def);
}

Expr MtlxExprCompiler::PositionExpr(const std::string& space) {
  return space == "world" ? Expr::p_world()
                          : MeshAttr("__p_object", 3, Expr::p_world());
}

Expr MtlxExprCompiler::NormalExpr(const std::string& space) {
  if (space == "world") return Expr::n_world();
  return call(Op::Normalize, 3, MeshAttr("__n_object", 3, Expr::n_world()));
}

Expr MtlxExprCompiler::NoiseCoord(const mx::NodePtr& node) {
  return absl::StrContains(NodeCategory(node), "2d")
             ? CompileInputExpr(node, "texcoord", DefaultTexcoord())
             : CompileInputExpr(node, "position", PositionExpr("object"));
}

Expr MtlxExprCompiler::NoiseExpr(const mx::NodePtr& node, Expr coord, bool vec3,
                                 const Expr& amplitude, bool offset) {
  std::string cat = NodeCategory(node);
  bool is_3d = absl::StrContains(cat, "3d");
  bool is_noise = cat == "noise2d" || cat == "noise3d";
  if (offset) {
    const double* off = is_noise ? kNoiseOffset : kFractalOffset;
    coord = coord + (is_3d ? ConstRgb(off[0], off[1], off[2])
                           : ConstVec2(off[0], off[1]));
  }
  uint16_t imm = (is_3d ? kNoise3D : 0) | (vec3 ? kNoiseVec3 : 0);
  if (is_noise) {
    return call(Op::Noise, imm, coord, amplitude,
                CompileInputExpr(node, "pivot", Expr::zero()));
  }
  return call(Op::Fractal, imm, coord,
              CompileInputExpr(node, "octaves", Const(3.0)),
              CompileInputExpr(node, "lacunarity", Const(2.0)),
              CompileInputExpr(node, "diminish", Const(0.5))) *
         amplitude;
}

Expr MtlxExprCompiler::CompileNodeOutputExpr(const mx::NodePtr& node,
                                             const std::string& output_name) {
  std::string cat = NodeCategory(node);
  const std::string& type = node->getType();
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  auto in_slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputSlotExpr(node, name, def);
  };

  // Multi-output nodes & Image / Bitmap sampling
  if (cat == "artistic_ior") {
    Expr r = in_slot("reflectivity");
    Expr eta = call(Op::ArtisticIorN, r, in_slot("edge_color"));
    return output_name == "extinction"
               ? call(Op::ArtisticIorK, r, EmitTmpExpr(eta))
               : eta;
  }
  if (cat == "image") {
    auto key = ExtractMtlxImageKey(node, resolve_path_);
    Expr def = in("default");
    return key ? SampleImage(node, *key, def) : def;
  }

  // Geometric intrinsics & space transforms.
  auto default_uv = [&]() {
    Expr uv = DefaultTexcoord();
    if (type == "vector3") uv.channels = 3;
    return uv;
  };
  if (cat == "texcoord") return default_uv();
  if (cat == "position")
    return PositionExpr(GetNodeInputString(node, "space", "object"));
  if (cat == "normal")
    return NormalExpr(GetNodeInputString(node, "space", "object"));
  if (cat == "tangent") return Expr::t_world();
  if (cat == "bitangent") return Expr::b_world();
  if (cat == "viewdirection") return Expr::incident_dir();
  if (cat == "updirection") return ConstRgb(0.0, 0.0, 1.0);
  if (cat == "frame" || cat == "time") return Expr::zero();
  if (cat == "geompropvalue" || cat == "geomcolor") {
    std::string p = (cat == "geomcolor")
                        ? "displayColor"
                        : GetNodeInputString(node, "geomprop", "st");
    if (p == "st" || p == "uv" || p == "st0" || p == "UV0" || p == "UVMap")
      return default_uv();
    if (p == "position" || p == "P" || p == "Pobject" || p == "Pworld")
      return PositionExpr(p == "Pobject" ? "object" : "world");
    if (p == "normal" || p == "N" || p == "Nobject" || p == "Nworld")
      return NormalExpr(p == "Nobject" ? "object" : "world");
    if (p == "tangent" || p == "T" || p == "Tworld" || p == "Tobject")
      return Expr::t_world();
    if (p == "bitangent" || p == "B" || p == "Bworld" || p == "Bobject")
      return Expr::b_world();
    bool scalar = type == "float" || type == "integer" || type == "boolean";
    Expr attr = MeshAttr(p, scalar ? 1 : 3, in("default"));
    if (type == "vector2") attr.channels = 2;
    return attr;
  }
  if (cat == "transformmatrix") {
    bool want_alpha = std::exchange(alpha_node_, nullptr) == node.get();
    Expr v = in_slot("in");
    mx::NodePtr mat_node = node->getConnectedNode("mat");
    bool transposed = false;
    while (mat_node && (mat_node->getCategory() == "transpose" ||
                        mat_node->getCategory() == "dot")) {
      if (mat_node->getCategory() == "transpose") transposed = !transposed;
      mat_node = mat_node->getConnectedNode("in");
    }
    // Rows of the matrix, which multiplies `v` as a row vector.
    Expr rows[4] = {Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()};
    Expr col3[4] = {Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()};
    bool is_mat4 = false;
    if (mat_node && mat_node->getCategory() == "creatematrix" &&
        (mat_node->getType() == "matrix33" ||
         mat_node->getType() == "matrix44")) {
      is_mat4 = mat_node->getType() == "matrix44";
      bool vec4_in = InputType(mat_node, "in1") == "vector4";
      for (int i = 0; i < (is_mat4 ? 4 : 3); ++i) {
        std::string in_name = "in" + std::to_string(i + 1);
        rows[i] = CompileInputExpr(mat_node, in_name, Expr::zero());
        if (vec4_in) col3[i] = CompileInputAlphaExpr(mat_node, in_name);
      }
    } else if (mx::InputPtr mat_val = !mat_node ? GetValueInput(node, "mat")
                                      : mat_node->getCategory() == "constant"
                                          ? GetValueInput(mat_node, "value")
                                          : nullptr;
               mat_val && (mat_val->getType() == "matrix33" ||
                           mat_val->getType() == "matrix44")) {
      is_mat4 = mat_val->getType() == "matrix44";
      if (is_mat4) {
        mx::Matrix44 m = mat_val->getValue()->asA<mx::Matrix44>();
        for (int i = 0; i < 4; ++i) {
          rows[i] = EmitRgb(prog_, m[i][0], m[i][1], m[i][2]);
          col3[i] = EmitParam(prog_, m[i][3]);
        }
      } else {
        mx::Matrix33 m = mat_val->getValue()->asA<mx::Matrix33>();
        for (int i = 0; i < 3; ++i)
          rows[i] = EmitRgb(prog_, m[i][0], m[i][1], m[i][2]);
      }
    } else {
      return want_alpha ? CompileInputAlphaExpr(node, "in") : v;
    }
    Expr v3 =
        type == "vector2" ? EmitTmpExpr(rgb(v.x(), v.y(), Expr::one())) : v;
    Expr w = type == "vector4" ? EmitTmpExpr(CompileInputAlphaExpr(node, "in"))
                               : (is_mat4 ? Expr::one() : Expr::zero());
    if (want_alpha) {
      if (!is_mat4) return Expr::one();
      Expr acc = transposed ? call(Op::Dot, rows[3], v3) : Expr::zero();
      if (!transposed) {
        for (int i = 0; i < 3; ++i) acc = acc + col3[i] * v3.chan(i);
      }
      return acc + col3[3] * w;
    }
    Expr res = transposed ? rgb(call(Op::Dot, rows[0], v3) + col3[0] * w,
                                call(Op::Dot, rows[1], v3) + col3[1] * w,
                                call(Op::Dot, rows[2], v3) + col3[2] * w)
                          : (rows[0] * v3.x() + rows[1] * v3.y() +
                             rows[2] * v3.z() + rows[3] * w);
    if (type != "vector2") return res;
    res = EmitTmpExpr(res);
    return vec2(res.x(), res.y());
  }

  // Constants, routing, and channel ops
  if (cat == "constant") return in("value");
  if (cat == "dot" || cat == "convert") return in("in");
  if (cat == "transformpoint" || cat == "transformvector" ||
      cat == "transformnormal") {
    std::string from = GetNodeInputString(node, "fromspace");
    std::string to = GetNodeInputString(node, "tospace");
    auto canon_space = [](std::string_view s) {
      return s == "model" ? std::string_view("object") : s;
    };
    if (!from.empty() && !to.empty() && canon_space(from) != canon_space(to))
      WarnOnce(
          "Unsupported MaterialX space transform '%s'; passing input through "
          "without space conversion.",
          std::string(cat) + "(" + from + "->" + to + ")");
    return cat == "transformnormal" ? call(Op::Normalize, 3, in("in"))
                                    : in("in");
  }
  if (cat == "combine2") {
    if (IsFourChannelType(type)) {
      // The alpha channel is `in2` (see `CompileNodeAlphaExpr`).
      if (InputType(node, "in1") != "vector2") return in("in1");
      Expr v1 = in_slot("in1");
      return rgb(v1.x(), v1.y(), in("in2"));
    }
    return vec2(in("in1"), in("in2"));
  }
  if (cat == "combine3" || cat == "combine4")
    return rgb(in("in1"), in("in2"), in("in3"));
  if (cat == "extract") {
    mx::InputPtr idx_in = GetValueInput(node, "index");
    int idx = idx_in ? idx_in->getValue()->asA<int>() : 0;
    return idx == 3 ? CompileInputAlphaExpr(node, "in")
                    : in("in").chan(std::clamp(idx, 0, 2));
  }

  // Arithmetic, compositing & math nodes
  if (cat == "add") return in("in1") + in("in2");
  if (cat == "subtract") return in("in1") - in("in2");
  if (cat == "plus" || cat == "minus" || cat == "difference" ||
      cat == "screen" || cat == "burn" || cat == "dodge") {
    Expr fg = in_slot("fg"), bg = in_slot("bg"), m = in("mix", Expr::one());
    // Like MaterialX, burn and dodge are 0 where they divide by 0.
    if (cat == "burn")
      return select(
          call(Op::Abs, fg) < Const(1e-8), Expr::zero(),
          call(Op::Mix, Expr::one() - (Expr::one() - bg) / fg, bg, m));
    if (cat == "dodge")
      return select(call(Op::Abs, Expr::one() - fg) < Const(1e-8), Expr::zero(),
                    call(Op::Mix, bg / (Expr::one() - fg), bg, m));
    Expr blend = cat == "plus"         ? (bg + fg)
                 : cat == "minus"      ? (bg - fg)
                 : cat == "difference" ? call(Op::Abs, bg - fg)
                                       : (bg + fg - bg * fg);
    return call(Op::Mix, blend, bg, m);
  }
  if (cat == "inside") return in("in") * in("mask", Expr::one());
  if (cat == "outside") return in("in") * (Expr::one() - in("mask"));
  if (cat == "premult" || cat == "unpremult") {
    Expr in_e = in("in"), a = CompileInputAlphaExpr(node, "in");
    return cat == "premult" ? (in_e * a) : (in_e / a);
  }
  if (cat == "over" || cat == "in" || cat == "out" || cat == "mask" ||
      cat == "matte" || cat == "disjointover") {
    Expr fg = in_slot("fg"), bg = in_slot("bg");
    Expr fa = CompileInputAlphaExpr(node, "fg");
    Expr ba = CompileInputAlphaExpr(node, "bg");
    Expr composite =
        cat == "over"   ? fg + bg * (Expr::one() - fa)
        : cat == "in"   ? fg * ba
        : cat == "out"  ? fg * (Expr::one() - ba)
        : cat == "mask" ? bg * fa
        : cat == "matte"
            ? fg * fa + bg * (Expr::one() - fa)
            : select(fa + ba <= Expr::one(), fg + bg,
                     select(call(Op::Abs, ba) < Const(1e-8), Expr::zero(),
                            fg + bg * ((Expr::one() - fa) / ba)));
    return call(Op::Mix, composite, bg, in("mix", Expr::one()));
  }
  if (cat == "ramplr" || cat == "ramptb" || cat == "splitlr" ||
      cat == "splittb") {
    bool lr = (cat == "ramplr" || cat == "splitlr");
    Expr v0 = in(lr ? "valuer" : "valuet"), v1 = in(lr ? "valuel" : "valueb");
    Expr tc = in("texcoord", DefaultTexcoord());
    Expr tc_comp = lr ? tc.x() : tc.y();
    if (cat == "ramplr" || cat == "ramptb")
      return call(Op::Mix, v0, v1,
                  call(Op::Clip, tc_comp, Expr::zero(), Expr::one()));
    Expr c = in_slot("center", Const(0.5));
    return call(Op::Mix, v0, v1,
                call(Op::Smoothstep, c, c + Const(1e-5), tc_comp));
  }
  if (cat == "multiply") {
    Expr i2 = in("in2", Expr::one());
    return i2.is_zero() ? Expr::zero() : in("in1", Expr::one()) * i2;
  }
  if (cat == "divide") {
    Expr i1 = in("in1", Expr::one());
    return i1.is_zero() ? Expr::zero() : i1 / in("in2", Expr::one());
  }
  if (cat == "power") {
    Expr i2 = in("in2", Expr::one());
    if (i2.is_zero()) return Expr::one();
    Expr i1 = in("in1");
    return (i1.is_one() || i2.is_one()) ? i1 : call(Op::Pow, i1, i2);
  }
  if (cat == "clamp") {
    Expr in_e = in("in"), low = in("low"), high = in("high", Expr::one());
    if (in_e.is_zero() && low.is_zero()) return Expr::zero();
    if (in_e.is_one() && high.is_one()) return Expr::one();
    return call(Op::Clip, in_e, low, high);
  }
  if (cat == "mix") {
    Expr m = in("mix");
    if (m.is_zero()) return in("bg");
    if (m.is_one()) return in("fg");
    Expr fg = in("fg"), bg = in("bg");
    return fg == bg ? bg : call(Op::Mix, fg, bg, m);
  }
  if (cat == "smoothstep")
    return call(Op::Smoothstep, in("low"), in("high", Expr::one()), in("in"));
  if (cat == "remap")
    return call(Op::Remap, in("in"), in("inlow"), in("inhigh", Expr::one()),
                in("outlow"), in("outhigh", Expr::one()));
  if (cat == "invert") return in("amount", Expr::one()) - in("in");
  // Unary ops, and whether they map 0 to 0.
  static constexpr struct {
    std::string_view name;
    Op op;
    bool zero_preserving;
  } kUnaryOps[] = {
      {"absval", Op::Abs, true},
      {"sign", Op::Sign, true},
      {"floor", Op::Floor, true},
      {"ceil", Op::Ceil, true},
      {"round", Op::Round, true},
      {"fract", Op::Fract, true},
      {"sin", Op::Sin, true},
      {"cos", Op::Cos, false},
      {"tan", Op::Tan, true},
      {"asin", Op::Asin, true},
      {"acos", Op::Acos, false},
      {"sqrt", Op::Sqrt, true},
      {"ln", Op::Log, false},
      {"exp", Op::Exp, false},
      {"rgbtohsv", Op::RgbToHsv, false},
      {"hsvtorgb", Op::HsvToRgb, false},
  };
  for (const auto& [name, op, zero_preserving] : kUnaryOps) {
    if (cat == name) {
      Expr in_e = in("in");
      return in_e.is_zero() && zero_preserving ? Expr::zero() : call(op, in_e);
    }
  }
  static constexpr struct {
    std::string_view name;
    const char *in1, *in2;
    Op op;
    bool def1_one, def2_one;
  } kBinaryOps[] = {
      {"modulo", "in1", "in2", Op::Modulo, false, true},
      {"min", "in1", "in2", Op::Min, false, false},
      {"max", "in1", "in2", Op::Max, false, false},
      {"atan2", "iny", "inx", Op::Atan2, false, true},
      {"crossproduct", "in1", "in2", Op::Cross, false, false},
      {"rotate2d", "in", "amount", Op::Rotate2D, false, false},
  };
  for (const auto& [name, in1, in2, op, def1_one, def2_one] : kBinaryOps) {
    if (cat == name) {
      return call(op, in(in1, def1_one ? Expr::one() : Expr::zero()),
                  in(in2, def2_one ? Expr::one() : Expr::zero()));
    }
  }

  // Vector ops & normal map
  if (cat == "normalize" || cat == "magnitude" || cat == "dotproduct") {
    std::string in_type = InputType(node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_dim = in_type == "vector2" ? 2 : 3;
    if (cat == "normalize")
      return in_type == "vector4" ? in_slot("in") / Magnitude4(node, "in")
                                  : call(Op::Normalize, vec_dim, in("in"));
    if (cat == "magnitude")
      return in_type == "vector4" ? Magnitude4(node, "in")
                                  : call(Op::Length, vec_dim, in("in"));
    Expr dot = call(Op::Dot, vec_dim, in("in1"), in("in2"));
    return in_type == "vector4" ? dot + CompileInputAlphaExpr(node, "in1") *
                                            CompileInputAlphaExpr(node, "in2")
                                : dot;
  }
  if (cat == "rotate3d") {
    Expr amt = in("amount"), in_e = in("in");
    return amt.is_zero() ? in_e
                         : call(Op::Rotate3D, in_e, amt,
                                in("axis", ConstRgb(0.0, 1.0, 0.0)));
  }
  if (cat == "normalmap")
    return TangentNormalMapExpr(in("in"), in("scale", Expr::one()), node);

  // Conditionals & logical ops
  if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    auto branch = [&](bool first) {
      return type == "boolean" ? (first ? Expr::one() : Expr::zero())
                               : in(first ? "in1" : "in2");
    };
    Expr v1 = in("value1"), v2 = in("value2");
    if (v1 == v2) return branch(cat != "ifgreater");
    if (v1.code.size() == 1 && v1.code[0].op == Op::Const &&
        v2.code.size() == 1 && v2.code[0].op == Op::Const) {
      double c1 = prog_.consts[v1.code[0].imm];
      double c2 = prog_.consts[v2.code[0].imm];
      return branch(cat == "ifgreater"     ? (c1 > c2)
                    : cat == "ifgreatereq" ? (c1 >= c2)
                                           : (c1 == c2));
    }
    Expr cmp = cat == "ifgreater"     ? (v1 > v2)
               : cat == "ifgreatereq" ? (v1 >= v2)
                                      : eq(v1, v2);
    Expr b1 = branch(true), b2 = branch(false);
    return b1 == b2 ? b1 : select(cmp, b1, b2);
  }
  if (cat == "switch") {
    Expr w = in("which");
    if (w.code.size() == 1 && w.code[0].op == Op::Const) {
      int idx = static_cast<int>(std::floor(prog_.consts[w.code[0].imm]));
      if (idx >= 10) return Expr::zero();
      return in("in" + std::to_string(std::max(idx, 0) + 1));
    }
    w = EmitTmpExpr(w);
    Expr res = Expr::zero();
    for (int i = 9; i >= 0; --i) {
      std::string name = "in" + std::to_string(i + 1);
      if (!node->getConnectedNode(name) && !GetValueInput(node, name)) continue;
      Expr val = in(name);
      if (res != val) {
        res = select(Const(static_cast<double>(i + 1)) > w, val, res);
      }
    }
    return res;
  }
  if (cat == "and" || cat == "or") {
    return cat == "and" ? logical_and(in("in1"), in("in2"))
                        : logical_or(in("in1"), in("in2"));
  }
  if (cat == "xor") return ne(in("in1"), in("in2"));
  if (cat == "not") return !in("in");

  // Color science & PBR helper nodes
  if (cat == "luminance") {
    Expr in_e = in("in");
    return in_e.is_zero() ? Expr::zero()
                          : call(Op::Dot, in_e, in("lumacoeffs"));
  }
  if (cat == "blackbody") return call(Op::Blackbody, in("temperature"));
  if (cat == "roughness_anisotropy") {
    Expr r = in("roughness"), a = in("anisotropy");
    if (a.is_zero()) {
      if (r.is_zero()) return Const(1e-5);
      if (r.is_one()) return Expr::one();
      Expr r_tmp = EmitTmpExpr(r);
      return call(Op::Clip, r_tmp * r_tmp, Const(1e-5), Expr::one());
    }
    return call(Op::RoughnessAnisotropy, r, a);
  }
  if (cat == "roughness_dual") return call(Op::RoughnessDual, in("roughness"));

  // Procedural noise nodes
  uint16_t noise_imm = (absl::StrContains(cat, "3d") ? kNoise3D : 0) |
                       (type != "float" ? kNoiseVec3 : 0);
  if (cat == "cellnoise2d" || cat == "cellnoise3d")
    return call(Op::CellNoise, noise_imm, NoiseCoord(node));
  if (cat == "worleynoise2d" || cat == "worleynoise3d") {
    Expr w = call(Op::Worley, noise_imm, NoiseCoord(node),
                  in("jitter", Expr::one()), in("style"));
    if (type == "vector2") w.channels = 2;
    return w;
  }
  if (cat == "noise2d" || cat == "noise3d" || cat == "fractal2d" ||
      cat == "fractal3d") {
    Expr amplitude = in("amplitude", Expr::one());
    if (type == "float")
      return NoiseExpr(node, NoiseCoord(node), false, amplitude);
    if (type != "vector2")
      return NoiseExpr(node, NoiseCoord(node), true, amplitude);
    // Like MaterialX, `vector2` noise is the first two channels of vector
    // noise, whereas `vector2` fractals are two scalar fractals.
    Expr coord = EmitTmpExpr(NoiseCoord(node));
    if (cat == "noise2d" || cat == "noise3d") {
      Expr n = EmitTmpExpr(NoiseExpr(node, coord, true, amplitude));
      return vec2(n.x(), n.y());
    }
    return vec2(NoiseExpr(node, coord, false, Expr::one()),
                NoiseExpr(node, coord, false, Expr::one(), /*offset=*/true)) *
           amplitude;
  }

  WarnOnce(
      "Unsupported MaterialX node category '%s'; passing through its 'in' "
      "input.",
      cat);
  return in("in");
}

Expr MtlxExprCompiler::TangentNormalMapExpr(const Expr& in_e, const Expr& scale,
                                            const mx::NodePtr& node) {
  Expr n = CompileInputExpr(node, "normal", Expr::n_world());
  Expr t = CompileInputExpr(node, "tangent", Expr::t_world());
  return (node->getConnectedNode("bitangent") ||
          GetValueInput(node, "bitangent"))
             ? call(Op::NormalMap, 1, in_e, scale, n, t,
                    CompileInputExpr(node, "bitangent", Expr::b_world()))
             : call(Op::NormalMap, in_e, scale, n, t);
}

void MtlxExprCompiler::PopulateProperties(Properties& props,
                                          const Expr& final_expr) const {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    props.set("in" + std::to_string(i), inputs_[i]);
  }
  MtlxProgram prog = prog_;
  if (!final_expr.empty()) prog.finish(final_expr);
  props.set_any("program", std::move(prog));
}

ref<Object> MtlxExprCompiler::CreateExprTexture(const Expr& expr) const {
  RegisterMaterialXExprTexture();
  Properties props("materialx_expr");
  PopulateProperties(props, expr);
  return PluginManager::instance()->create_object(props, variant_,
                                                  ObjectType::Texture);
}

std::optional<Properties> CompileSurfaceEdf(const mx::NodePtr& shader_node,
                                            const MtlxCompileOptions& opts) {
  MtlxCompileOptions edf_opts = opts;
  MeshAttributeRequests edf_attrs;
  edf_opts.mesh_attributes = opts.mesh_attributes ? &edf_attrs : nullptr;
  MtlxExprCompiler edf_compiler(edf_opts);
  absl::flat_hash_map<const mx::Node*, Expr> memo;
  Expr rad_expr = CompileEdfNode(shader_node, edf_compiler, memo);
  if (rad_expr.is_zero()) return std::nullopt;
  if (opts.mesh_attributes) {
    opts.mesh_attributes->insert(edf_attrs.begin(), edf_attrs.end());
  }
  ref<Object> radiance = edf_compiler.CreateExprTexture(rad_expr);
  if (absl::StrContains(edf_compiler.variant(), "spectral")) {
    // Linear sRGB colors are relative to the D65 white point. As for Mitsuba's
    // own RGB emitter colors, multiply the upsampled color by the D65 spectrum.
    Properties d65_props("d65");
    d65_props.set("nested", radiance);
    radiance = CreateExpandedObject(d65_props, edf_compiler.variant(),
                                    ObjectType::Texture);
  }
  Properties emit_props("area");
  emit_props.set("radiance", radiance);
  return emit_props;
}

}  // namespace mtlx
NAMESPACE_END(mitsuba)
