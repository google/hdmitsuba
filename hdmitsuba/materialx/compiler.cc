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
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>
#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/strings/ascii.h>
#include <absl/strings/strip.h>
#include <absl/types/span.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/vector.h>
#include <pxr/base/gf/color.h>
#include <pxr/base/gf/colorSpace.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/materialx/math.h"
#include "hdmitsuba/materialx/program.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace mx = MaterialX;
namespace mtlx = mitsuba::mtlx;
using mitsuba::Object;
using mitsuba::ObjectType;
using mitsuba::PluginManager;
using mitsuba::Properties;
using mitsuba::ref;

struct Expr;

template <typename... Exprs>
Expr call(mtlx::Op op, uint16_t imm, const Exprs&... args);
template <typename... Exprs>
Expr call(mtlx::Op op, const Expr& arg0, const Exprs&... rest);
template <typename... Exprs>
Expr call_with_channels(mtlx::Op op, uint8_t inst_ch, uint16_t imm,
                        const Exprs&... args);

// Reverse-Polish-Notation expression fragment that evaluates to one stack slot.
// Constants 0.0 and 1.0 are always at indices 0 and 1 in `MtlxProgram::consts`.
//
// Arithmetic, `mix()` and `select()` simplify constant 0 and 1 operands. This
// is not an optimization (Dr.Jit simplifies the traced program itself), but
// determines what is known before rendering: whether a surface emits or is
// cut out at all, and which textures and primvars its shader reads.
struct Expr {
  std::vector<mtlx::Inst> code;
  uint8_t channels = 3;
  uint32_t max_depth = 1;

  Expr() = default;
  explicit Expr(mtlx::Inst i, uint8_t channels = 3)
      : code{i}, channels(channels) {}

  static Expr zero() { return Expr({mtlx::Op::Const, 0, 0}, 1); }
  static Expr one() { return Expr({mtlx::Op::Const, 0, 1}, 1); }
  static Expr p_world() { return Expr({mtlx::Op::GeomPosWorld, 0, 0}); }
  static Expr n_world() { return Expr({mtlx::Op::GeomNormalWorld, 0, 0}); }
  static Expr t_world() { return Expr({mtlx::Op::GeomTangentWorld, 0, 0}); }
  static Expr b_world() { return Expr({mtlx::Op::GeomBitangentWorld, 0, 0}); }

  bool uniform() const { return channels == 1; }
  bool is_zero() const { return *this == zero(); }
  bool is_one() const { return *this == one(); }
  bool is_const() const {
    return !code.empty() &&
           std::all_of(code.begin(), code.end(), [](mtlx::Inst i) {
             return i.op == mtlx::Op::Const || i.op == mtlx::Op::Vec2 ||
                    i.op == mtlx::Op::Rgb || i.op == mtlx::Op::Vec4;
           });
  }

  Expr with_channels(uint8_t ch) const {
    Expr e = *this;
    e.channels = ch;
    return e;
  }

  Expr chan(int c) const {
    if (uniform()) return *this;
    Expr e = *this;
    e.code.push_back({mtlx::Op::Chan, 0, static_cast<uint16_t>(c)});
    e.channels = 1;
    return e;
  }
  Expr x() const { return chan(0); }
  Expr y() const { return chan(1); }
  Expr z() const { return chan(2); }
  Expr w() const { return chan(3); }
  // Returns the first three channels of a 4-channel expression.
  Expr xyz() const { return with_channels(std::min<uint8_t>(channels, 3)); }

  Expr operator-() const {
    return is_zero() ? zero() : call(mtlx::Op::Neg, *this);
  }
  Expr operator!() const {
    return is_zero()  ? one()
           : is_one() ? zero()
                      : call(mtlx::Op::Eq, *this, zero());
  }
  Expr operator+(const Expr& b) const {
    return is_zero() ? b : b.is_zero() ? *this : call(mtlx::Op::Add, *this, b);
  }
  Expr operator-(const Expr& b) const {
    return b.is_zero() ? *this : is_zero() ? -b : call(mtlx::Op::Sub, *this, b);
  }
  Expr operator*(const Expr& b) const {
    return (is_zero() || b.is_zero()) ? zero()
           : is_one()                 ? b
           : b.is_one()               ? *this
                                      : call(mtlx::Op::Mul, *this, b);
  }
  Expr operator/(const Expr& b) const {
    return (is_zero() || b.is_zero()) ? zero()
           : b.is_one()               ? *this
                                      : call(mtlx::Op::Div, *this, b);
  }
  Expr operator<(const Expr& b) const { return call(mtlx::Op::Lt, *this, b); }
  Expr operator<=(const Expr& b) const { return call(mtlx::Op::Le, *this, b); }
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
Expr call_with_channels(mtlx::Op op, uint8_t inst_ch, uint16_t imm,
                        const Exprs&... args) {
  Expr e;
  uint32_t depth = 0;
  ((e.code.insert(e.code.end(), args.code.begin(), args.code.end()),
    e.max_depth = std::max(e.max_depth, depth + args.max_depth), ++depth),
   ...);
  bool ew = mtlx::elementwise_arity(op) != 0;
  e.channels = ew ? std::max<uint8_t>({args.channels...})
                  : mtlx::result_channels(op, imm);
  e.code.push_back(
      {op, inst_ch ? inst_ch : static_cast<uint8_t>(ew ? e.channels : 0), imm});
  return e;
}

template <typename... Exprs>
Expr call(mtlx::Op op, uint16_t imm, const Exprs&... args) {
  return call_with_channels(op, uint8_t{0}, imm, args...);
}

template <typename... Exprs>
Expr call(mtlx::Op op, const Expr& arg0, const Exprs&... rest) {
  return call(op, uint16_t{0}, arg0, rest...);
}

Expr rgb(const Expr& r, const Expr& g, const Expr& b) {
  return (r.uniform() && r == g && g == b) ? r : call(mtlx::Op::Rgb, r, g, b);
}
Expr vec2(const Expr& x, const Expr& y) {
  return (x.uniform() && x == y) ? x : call(mtlx::Op::Vec2, x, y);
}
Expr vec4(const Expr& xyz, const Expr& w) {
  return (xyz.uniform() && xyz == w) ? xyz : call(mtlx::Op::Vec4, xyz, w);
}
Expr select(const Expr& cond, const Expr& t, const Expr& f) {
  return (cond.is_zero() || t == f) ? f
         : cond.is_one()            ? t
                                    : call(mtlx::Op::Select, cond, t, f);
}
Expr saturate(const Expr& e) {
  return (e.is_zero() || e.is_one())
             ? e
             : call(mtlx::Op::Clip, e, Expr::zero(), Expr::one());
}
Expr mix(const Expr& fg, const Expr& bg, const Expr& m) {
  return (m.is_zero() || fg == bg) ? bg
         : m.is_one()              ? fg
                                   : call(mtlx::Op::Mix, fg, bg, m);
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
  return Expr({mtlx::Op::GeomUV, 0, 0}, channels);
}

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

struct Frame;

struct ScopedNode {
  mx::NodePtr node;
  const Frame* frame = nullptr;

  explicit operator bool() const { return node != nullptr; }
};

// Active `NodeGraph` implementation call frame during lazy compilation.
struct Frame {
  ScopedNode call;
};

// Returns true if `pred` holds for an instruction of `root` or a live
// instruction of `body`, which stores the `tmp_count` temporaries read by
// `root`. `body` may also end with root instructions after its last
// `StoreTmp`.
template <typename Pred>
bool AnyLiveInst(absl::Span<const mtlx::Inst> body,
                 absl::Span<const mtlx::Inst> root, size_t tmp_count,
                 const Pred& pred) {
  std::vector<bool> live(tmp_count);
  bool active = true;
  auto visit = [&](const mtlx::Inst& inst) {
    if (inst.op == mtlx::Op::StoreTmp) {
      active = live[inst.imm];
    } else if (inst.op == mtlx::Op::MatTranspose ||
               inst.op == mtlx::Op::MatInverse || inst.op == mtlx::Op::MatMul) {
      active = std::any_of(live.begin() + inst.imm,
                           live.begin() + inst.imm + inst.channels,
                           [](bool b) { return b; });
    } else if (active) {
      if (pred(inst)) return true;
      if (inst.op == mtlx::Op::LoadTmp) live[inst.imm] = true;
    }
    return false;
  };
  return std::any_of(root.rbegin(), root.rend(), visit) ||
         std::any_of(body.rbegin(), body.rend(), visit);
}

// Resolves MaterialX ports, inputs, and lazy `NodeGraph` call frames without
// allocating a `MtlxProgram`.
class MtlxGraphResolver {
 public:
  using NodeKey = std::pair<const Frame*, const mx::Node*>;

  struct ResolvedInput {
    enum Kind { None, Connected, Value, GeomProp } kind = None;
    ScopedNode upstream;
    std::string output_name;
    mx::ValueElementPtr elem;
  };

  struct NodeInfo {
    mx::NodeDefPtr nodedef;
    mx::NodeGraphPtr graph_impl;
  };

  const NodeInfo& GetNodeInfo(const mx::NodePtr& node);
  // Returns the category of `node`, which is determined by its explicit node
  // definition, if any. For example, `NG_ramp_gradient` has an `ifgreater` node
  // with the node definition `ND_ifgreatereq_color4I`.
  const std::string& NodeCategory(const mx::NodePtr& node);
  // Returns the type of the input `name` of `node`, which may only be declared
  // by its node definition, or an empty view if there is no such input.
  std::string_view InputType(const mx::NodePtr& node, const std::string& name);
  ResolvedInput ResolvePort(mx::PortElementPtr port, const Frame* frame);
  ResolvedInput ResolveOutput(const ScopedNode& call, const mx::OutputPtr& out);
  ResolvedInput ResolveInput(const ScopedNode& s, const std::string& name);
  ResolvedInput ResolveTerminalInput(const ScopedNode& s,
                                     const std::string& name);
  ScopedNode ResolveConnected(const ScopedNode& s, const std::string& name,
                              const char* mat_input = "surfaceshader");
  mx::ValueElementPtr GetValueInput(const ScopedNode& s,
                                    const std::string& name);
  std::string GetNodeInputString(const ScopedNode& s, const std::string& name,
                                 std::string_view def = {});
  bool IsNonZeroInput(const ScopedNode& s, const std::string& name);
  ScopedNode UnwrapShaderNode(ScopedNode s, std::string output_name,
                              const char* mat_input = "surfaceshader");

 private:
  std::deque<Frame> frames_;
  absl::flat_hash_map<NodeKey, const Frame*> child_frames_;
  absl::flat_hash_map<const mx::Node*, NodeInfo> node_info_;
};

// Compiles a MaterialX DAG (lazily traversing `NodeGraph` implementations
// without flattening them into the document) into a
// `mitsuba::mtlx::MtlxProgram` and a list of Mitsuba Texture child inputs
// (`in0`, `in1`, ...).
class MtlxExprCompiler : private MtlxGraphResolver {
 public:
  using LookupTextureFn =
      std::function<const CachedTexture*(const TextureKey&)>;
  using DiscoverTextureFn = std::function<void(const TextureKey&)>;

  struct DisplacementExpr {
    Expr expr = Expr::zero();
    bool is_vector = false;
  };

  MtlxExprCompiler(mx::DocumentPtr doc, const LookupTextureFn& lookup_texture,
                   absl::flat_hash_set<std::string>& warned)
      : doc_(std::move(doc)),
        lookup_texture_(&lookup_texture),
        warned_(&warned) {}
  MtlxExprCompiler(mx::DocumentPtr doc,
                   const DiscoverTextureFn& discover_texture)
      : doc_(std::move(doc)), discover_texture_(&discover_texture) {}

  // Compiles a MaterialX EDF or surface emission DAG (`surface`,
  // `surface_unlit`, `uniform_edf`, `generalized_schlick_edf`, `multiply`,
  // `add`, `mix`) into a radiance `Expr`, or `Expr::zero()` if none.
  Expr CompileEdf(const mx::NodePtr& edf_node, std::string_view output_name) {
    return CompileEdf(
        UnwrapShaderNode({edf_node, nullptr}, std::string(output_name)));
  }

  // Compiles a MaterialX displacement shader DAG (`displacement`, `mix`,
  // `dot`, `surfacematerial`) into a scalar or 3D tangent-space `Expr`.
  DisplacementExpr CompileDisplacement(const mx::NodePtr& disp_node,
                                       std::string_view output_name) {
    return CompileDisplacement(UnwrapShaderNode(
        {disp_node, nullptr}, std::string(output_name), "displacementshader"));
  }

  // Compiles a MaterialX surface shader or BSDF closure DAG into
  // `prog_.closures` and `prog_.opacity_slot`.
  void CompileBsdf(const mx::NodePtr& shader_node,
                   std::string_view output_name);

  bool ProgramUsesUv(const Expr& expr) const;

  const std::vector<std::pair<std::string, int>>& attr_requests() const {
    return attr_requests_;
  }

  ref<Object> CreateExprTexture(const Expr& expr, bool unbounded,
                                std::string_view variant);

  ref<Object> CreateBsdf(std::string_view variant);

 private:
  using NodeOutKey = std::tuple<const Frame*, const mx::Node*, std::string>;

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

  bool IsAuthoredInput(const mx::ValueElementPtr& elem) const {
    mx::ElementPtr parent = elem->getParent();
    return parent && parent->isA<mx::Node>() && parent->getParent() == doc_;
  }
  Expr Const(double value);
  Expr ConstRgb(double r, double g, double b);
  Expr Opaque(double value);
  // Emits a scalar or multi-channel parameter value. Library-internal constants
  // and unauthored NodeDef defaults (`!is_authored`) are baked as `Op::Const`
  // to enable constant folding and CSE; authored inputs emit `Op::Opaque`
  // (except all-zero and all-one tuples) for traversal and kernel freezing.
  Expr EmitTuple(std::initializer_list<float> vals, bool is_authored);
  std::optional<mitsuba::Vector<double, 4>> TryEvalConst(
      const Expr& expr) const;
  Expr FoldExpr(const Expr& expr);
  int EmitTmpFolded(const Expr& folded);
  int EmitTmp(const Expr& expr);
  Expr EmitTmpExpr(const Expr& expr);
  Expr FoldAndSpill(const Expr& expr);
  Expr LoadTmp(int slot) const {
    return Expr({mtlx::Op::LoadTmp, 0, static_cast<uint16_t>(slot)},
                prog_.tmp_channels[slot]);
  }
  Expr CompileEdf(const ScopedNode& node);
  DisplacementExpr CompileDisplacement(const ScopedNode& node);
  Expr CompileOpacity(const ScopedNode& node);
  int CompileBsdfClosure(const ScopedNode& node);
  int EmitOpaqueBlocker();
  int FrameSlot(const ScopedNode& node, const std::string& name,
                const Expr& def);
  int CompileDiffuseClosure(const ScopedNode& node, const std::string& cat);
  int CompileTranslucentClosure(const ScopedNode& node);
  int CompileMicrofacetClosure(const ScopedNode& node, const std::string& cat);
  int CompileChiangHairClosure(const ScopedNode& node);
  void CompileVdfWarning(const ScopedNode& node, const std::string& cat);
  void MarkTwoSidedDielectrics(size_t start);
  int EmitClosure(mtlx::ClosureNode node);
  void CompactLiveOpaques();
  ref<Object> CreatePlugin(std::string_view name, ObjectType type,
                           std::string_view variant);

  TfToken ResolveSourceColorSpace(std::string_view color_space);
  GfVec3f ToLinearRec709(GfVec3f rgb_val, std::string_view color_space);
  Expr CompileGeomIntrinsic(std::string_view kind, std::string_view space,
                            std::string_view type, bool non_zero_index);
  Expr CompileGeomProp(const std::string& prop_name, std::string_view type,
                       const std::function<Expr()>& default_fn = {});
  Expr CompileResolvedInput(const ResolvedInput& r, const Expr& default_expr);
  Expr CompileInputExpr(const ScopedNode& node, const std::string& input_name,
                        const Expr& default_expr = Expr::zero());
  Expr CompileNodeOutputExpr(const ScopedNode& node,
                             const std::string& output_name);
  std::optional<Expr> CompileChannelNode(const ScopedNode& node,
                                         const std::string& cat,
                                         const std::string& type);
  std::optional<Expr> CompileMathNode(const ScopedNode& node,
                                      const std::string& cat);
  std::optional<Expr> CompileLogicNode(const ScopedNode& node,
                                       const std::string& cat,
                                       const std::string& type);
  std::optional<Expr> CompileVectorNode(const ScopedNode& node,
                                        const std::string& cat);
  std::optional<Expr> CompileGeomNode(const ScopedNode& node,
                                      const std::string& cat,
                                      const std::string& type);
  std::optional<Expr> CompileMatrixValueNode(const ScopedNode& node,
                                             const std::string& cat,
                                             const std::string& type);
  std::optional<Expr> CompileCompositingNode(const ScopedNode& node,
                                             const std::string& cat);
  std::optional<Expr> CompileColorPbrNode(const ScopedNode& node,
                                          const std::string& cat,
                                          const std::string& type,
                                          const std::string& output_name);
  std::optional<Expr> CompileNoiseNode(const ScopedNode& node,
                                       const std::string& cat,
                                       const std::string& type);
  std::optional<int> ConstIntInput(const ScopedNode& node,
                                   const std::string& name);
  Expr CompileIfCondition(const ScopedNode& node, std::string_view cat);
  MatExpr CompileResolvedMatrix(const ResolvedInput& r, int n);
  MatExpr CompileMatrixInput(const ScopedNode& node,
                             const std::string& input_name);
  MatExpr CompileMatrixNode(const ScopedNode& node,
                            const std::string& output_name);
  // Emits `MatTranspose`, `MatInverse` or `MatMul` (of `a` and `b`).
  MatExpr MatrixOp(mtlx::Op op, const MatExpr& a, const MatExpr* b = nullptr);
  Expr MeshAttr(std::string_view name, int dim, const Expr& def);
  Expr CompileHeightToNormal(const ScopedNode& node);
  Expr SampleImage(const ScopedNode& node);
  Expr CompileValueElement(const mx::ValueElementPtr& elem,
                           const Expr& default_expr, bool is_authored);

  // Warns once per message about an unsupported feature of the compiled shader
  // (not during texture discovery, which compiles it a second time).
  void WarnUnsupported(const std::string& msg) {
    if (warned_ && warned_->insert(msg).second) {
      TF_WARN("%s", msg.c_str());
    }
  }

  mx::DocumentPtr doc_;
  const LookupTextureFn* lookup_texture_ = nullptr;
  const DiscoverTextureFn* discover_texture_ = nullptr;
  mtlx::MtlxProgram prog_;
  std::vector<std::pair<std::string, int>> attr_requests_;
  absl::flat_hash_map<NodeOutKey, Expr> node_output_expr_;
  absl::flat_hash_map<NodeKey, Expr> edf_expr_;
  absl::flat_hash_map<NodeKey, DisplacementExpr> disp_expr_;
  absl::flat_hash_map<NodeKey, Expr> opacity_expr_;
  absl::flat_hash_map<NodeKey, int> bsdf_closure_;
  absl::flat_hash_map<const mx::ValueElement*, Expr> authored_opaque_expr_;
  absl::flat_hash_map<const mx::ValueElement*, MatExpr> authored_opaque_matrix_;
  absl::flat_hash_map<NodeOutKey, MatExpr> node_matrix_expr_;
  absl::flat_hash_map<Expr, int> expr_to_slot_;
  std::vector<ref<Object>> inputs_;
  uint16_t discover_input_idx_ = 0;
  bool thin_walled_ = false;
  absl::flat_hash_set<std::string>* warned_ = nullptr;
};

const MtlxGraphResolver::NodeInfo& MtlxGraphResolver::GetNodeInfo(
    const mx::NodePtr& node) {
  auto [it, inserted] = node_info_.try_emplace(node.get());
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

const std::string& MtlxGraphResolver::NodeCategory(const mx::NodePtr& node) {
  if (node->hasNodeDefString()) {
    if (const mx::NodeDefPtr& nd = GetNodeInfo(node).nodedef) {
      return nd->getNodeString();
    }
  }
  return node->getCategory();
}

std::string_view MtlxGraphResolver::InputType(const mx::NodePtr& node,
                                              const std::string& name) {
  if (mx::InputPtr in = node->getInput(name)) return in->getType();
  if (const mx::NodeDefPtr& nd = GetNodeInfo(node).nodedef) {
    if (mx::InputPtr in = nd->getActiveInput(name)) return in->getType();
  }
  return {};
}

MtlxGraphResolver::ResolvedInput MtlxGraphResolver::ResolvePort(
    mx::PortElementPtr port, const Frame* frame) {
  while (port && (port->hasNodeGraphString() ||
                  (port->hasOutputString() && !port->hasNodeName() &&
                   !port->hasInterfaceName()))) {
    port = port->getConnectedOutput();
    frame = nullptr;
  }
  if (!port) return {};
  if (port->hasInterfaceName()) {
    const std::string& iface = port->getInterfaceName();
    ResolvedInput r;
    if (frame) {
      r = ResolveInput(frame->call, iface);
    } else if (mx::GraphElementPtr g =
                   port->getAncestorOfType<mx::GraphElement>()) {
      if (mx::InputPtr iface_in = g->getActiveInput(iface)) {
        r = ResolvePort(std::move(iface_in), nullptr);
      }
    }
    if (r.kind != ResolvedInput::None) return r;
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

MtlxGraphResolver::ResolvedInput MtlxGraphResolver::ResolveOutput(
    const ScopedNode& call, const mx::OutputPtr& out) {
  if (!out) return {};
  const Frame*& slot = child_frames_[{call.frame, call.node.get()}];
  if (!slot) {
    frames_.push_back({call});
    slot = &frames_.back();
  }
  return ResolvePort(out, slot);
}

MtlxGraphResolver::ResolvedInput MtlxGraphResolver::ResolveInput(
    const ScopedNode& s, const std::string& name) {
  if (mx::InputPtr input = s.node->getInput(name)) {
    if (ResolvedInput r = ResolvePort(std::move(input), s.frame);
        r.kind != ResolvedInput::None) {
      return r;
    }
  }
  const mx::NodeDefPtr& nd = GetNodeInfo(s.node).nodedef;
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

MtlxGraphResolver::ResolvedInput MtlxGraphResolver::ResolveTerminalInput(
    const ScopedNode& s, const std::string& name) {
  ResolvedInput r = ResolveInput(s, name);
  while (r.kind == ResolvedInput::Connected) {
    if (mx::NodeGraphPtr ng = GetNodeInfo(r.upstream.node).graph_impl) {
      r = ResolveOutput(r.upstream, ResolveNodeGraphOutput(ng, r.output_name));
      continue;
    }
    const std::string& cat = NodeCategory(r.upstream.node);
    if (cat != "dot" && cat != "constant") break;
    r = ResolveInput(r.upstream, cat == "constant" ? "value" : "in");
  }
  return r;
}

ScopedNode MtlxGraphResolver::ResolveConnected(const ScopedNode& s,
                                               const std::string& name,
                                               const char* mat_input) {
  ResolvedInput r = ResolveInput(s, name);
  return r.kind == ResolvedInput::Connected
             ? UnwrapShaderNode(std::move(r.upstream), std::move(r.output_name),
                                mat_input)
             : ScopedNode{};
}

mx::ValueElementPtr MtlxGraphResolver::GetValueInput(const ScopedNode& s,
                                                     const std::string& name) {
  ResolvedInput r = ResolveTerminalInput(s, name);
  return r.kind == ResolvedInput::Value ? r.elem : nullptr;
}

std::string MtlxGraphResolver::GetNodeInputString(const ScopedNode& s,
                                                  const std::string& name,
                                                  std::string_view def) {
  mx::ValueElementPtr in = GetValueInput(s, name);
  return in ? in->getValueString() : std::string(def);
}

// Returns true if input `name` of `s` may be non-zero, without emitting VM
// bytecode, opaque parameters, or texture loads into `MtlxProgram`.
bool MtlxGraphResolver::IsNonZeroInput(const ScopedNode& s,
                                       const std::string& name) {
  ResolvedInput r = ResolveTerminalInput(s, name);
  if (r.kind == ResolvedInput::None) return false;
  if (r.kind == ResolvedInput::GeomProp) return true;
  if (r.kind == ResolvedInput::Value) {
    mx::ValuePtr v = r.elem->getValue();
    if (!v) return false;
    if (v->isA<float>()) return v->asA<float>() != 0.0f;
    if (v->isA<mx::Vector3>()) {
      return v->asA<mx::Vector3>() != mx::Vector3(0.0f);
    }
    if (v->isA<mx::Color3>()) {
      return v->asA<mx::Color3>() != mx::Color3(0.0f);
    }
    return false;
  }
  const ScopedNode& up = r.upstream;
  const std::string& cat = NodeCategory(up.node);
  if (cat == "ifgreater") {
    mx::ValueElementPtr i1 = GetValueInput(up, "value1"),
                        i2 = GetValueInput(up, "value2");
    mx::ValuePtr v1 = i1 ? i1->getValue() : nullptr,
                 v2 = i2 ? i2->getValue() : nullptr;
    if (v1 && v2 && v1->isA<float>() && v2->isA<float>()) {
      return IsNonZeroInput(
          up, v1->asA<float>() > v2->asA<float>() ? "in1" : "in2");
    }
  } else if (cat == "multiply") {
    return IsNonZeroInput(up, "in1") && IsNonZeroInput(up, "in2");
  }
  return true;
}

ScopedNode MtlxGraphResolver::UnwrapShaderNode(ScopedNode s,
                                               std::string output_name,
                                               const char* mat_input) {
  while (s.node) {
    ResolvedInput r;
    const std::string& cat = NodeCategory(s.node);
    if (cat == "surfacematerial" || cat == "dot") {
      r = ResolveInput(s, cat == "dot" ? "in" : mat_input);
    } else if (mx::NodeGraphPtr ng = GetNodeInfo(s.node).graph_impl) {
      r = ResolveOutput(s, ResolveNodeGraphOutput(ng, output_name));
    } else {
      break;
    }
    if (r.kind != ResolvedInput::Connected) return {};
    s = std::move(r.upstream);
    output_name = std::move(r.output_name);
  }
  return s;
}

Expr MtlxExprCompiler::Const(double value) {
  auto it = std::find(prog_.consts.begin(), prog_.consts.end(), value);
  if (it == prog_.consts.end()) {
    if (prog_.consts.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many constants");
    }
    it = prog_.consts.insert(prog_.consts.end(), value);
  }
  return Expr(
      {mtlx::Op::Const, 0, static_cast<uint16_t>(it - prog_.consts.begin())},
      1);
}

Expr MtlxExprCompiler::ConstRgb(double r, double g, double b) {
  if (r == g && g == b) return Const(r);
  Expr er = Const(r), eg = Const(g), eb = Const(b);
  return rgb(er, eg, eb);
}

Expr MtlxExprCompiler::Opaque(double value) {
  if (prog_.opaques.size() >= mtlx::kMaxImm) {
    throw std::runtime_error("materialx_expr: too many opaque values");
  }
  prog_.opaques.push_back(value);
  return Expr(
      {mtlx::Op::Opaque, 0, static_cast<uint16_t>(prog_.opaques.size() - 1)},
      1);
}

std::optional<mitsuba::Vector<double, 4>> MtlxExprCompiler::TryEvalConst(
    const Expr& expr) const {
  if (expr.code.empty()) return std::nullopt;
  using Vec4d = mitsuba::Vector<double, 4>;
  absl::InlinedVector<Vec4d, 16> stack(expr.max_depth);
  Vec4d* sp = stack.data();
  for (mtlx::Inst inst : expr.code) {
    if (inst.op == mtlx::Op::Const) {
      *sp++ = Vec4d(prog_.consts[inst.imm]);
    } else if (!mtlx::detail::eval_pure_op<double>(inst, sp)) {
      return std::nullopt;
    }
  }
  const Vec4d& r = stack[0];
  for (uint8_t c = 0; c < expr.channels; ++c) {
    if (!(std::abs(r[c]) <= std::numeric_limits<float>::max())) {
      return std::nullopt;
    }
  }
  return r;
}

Expr MtlxExprCompiler::FoldExpr(const Expr& expr) {
  if (expr.code.size() <= 1) return expr;
  auto v = TryEvalConst(expr);
  if (!v) return expr;
  if (expr.channels <= 1) return Const((*v)[0]);
  if (expr.channels == 2) return vec2(Const((*v)[0]), Const((*v)[1]));
  Expr c3 = ConstRgb((*v)[0], (*v)[1], (*v)[2]);
  return expr.channels == 3 ? c3 : vec4(c3, Const((*v)[3]));
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
    if (attr_requests_.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many attributes");
    }
    prog_.attr_names.push_back(MeshAttributeName(name, dim));
    it = attr_requests_.insert(attr_requests_.end(), std::move(req));
  }
  Expr e = def;
  uint8_t read_ch = static_cast<uint8_t>(dim);
  e.code.push_back({mtlx::Op::Attr, read_ch,
                    static_cast<uint16_t>(it - attr_requests_.begin())});
  return e.with_channels(std::max<uint8_t>(def.channels, read_ch));
}

Expr MtlxExprCompiler::EmitTuple(std::initializer_list<float> vals,
                                 bool is_authored) {
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
  for (float v : vals) e[n++] = is_authored ? Opaque(v) : Const(v);
  if (n == 1) return e[0];
  if (n == 2) return vec2(e[0], e[1]);
  Expr c3 = rgb(e[0], e[1], e[2]);
  return n == 3 ? c3 : vec4(c3, e[3]);
}

int MtlxExprCompiler::EmitTmpFolded(const Expr& folded) {
  if (folded.code.size() == 1 && folded.code[0].op == mtlx::Op::LoadTmp) {
    return folded.code[0].imm;
  }
  size_t next_slot = prog_.tmp_channels.size();
  auto [it, inserted] =
      expr_to_slot_.try_emplace(folded, static_cast<int>(next_slot));
  if (inserted) {
    if (next_slot >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many temporaries");
    }
    prog_.code.insert(prog_.code.end(), folded.code.begin(), folded.code.end());
    prog_.code.push_back(
        {mtlx::Op::StoreTmp, 0, static_cast<uint16_t>(next_slot)});
    prog_.stack_size = std::max(prog_.stack_size, folded.max_depth);
    prog_.tmp_channels.push_back(folded.channels);
  }
  return it->second;
}

int MtlxExprCompiler::EmitTmp(const Expr& expr) {
  return EmitTmpFolded(FoldExpr(expr));
}

// Spills non-trivial expressions (including 1-op geometry intrinsics) for
// intra-node operand reuse.
Expr MtlxExprCompiler::EmitTmpExpr(const Expr& expr) {
  Expr folded = FoldExpr(expr);
  if ((folded.code.size() == 1 && folded.code[0].op <= mtlx::Op::LoadTmp) ||
      folded.is_const()) {
    return folded;
  }
  return LoadTmp(EmitTmpFolded(folded));
}

// Like `EmitTmpExpr`, but keeps 1-op intrinsics inline across cached node
// boundaries so downstream `== Expr::n_world()` / `DefaultTexcoord()` checks
// still match.
Expr MtlxExprCompiler::FoldAndSpill(const Expr& expr) {
  return expr.code.size() > 1 ? EmitTmpExpr(expr) : expr;
}

Expr MtlxExprCompiler::CompileValueElement(const mx::ValueElementPtr& elem,
                                           const Expr& default_expr,
                                           bool is_authored) {
  const std::string& type = elem->getType();
  if (type == "boolean") {
    const std::string& s = elem->getValueString();
    return (s == "true" || s == "1") ? Expr::one() : Expr::zero();
  }
  mx::ValuePtr val = elem->getValue();
  if (!val) return default_expr;

  if (type == "float") return EmitTuple({val->asA<float>()}, is_authored);
  // Integers and booleans select code paths (e.g. switch branches) and are
  // baked.
  if (type == "integer") return Const(val->asA<int>());
  if (type == "color3") {
    mx::Color3 v = val->asA<mx::Color3>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return EmitTuple({c[0], c[1], c[2]}, is_authored);
  }
  if (type == "color4") {
    mx::Color4 v = val->asA<mx::Color4>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return EmitTuple({c[0], c[1], c[2], v[3]}, is_authored);
  }
  if (type == "vector2") {
    mx::Vector2 v = val->asA<mx::Vector2>();
    return EmitTuple({v[0], v[1]}, is_authored);
  }
  if (type == "vector3") {
    mx::Vector3 v = val->asA<mx::Vector3>();
    return EmitTuple({v[0], v[1], v[2]}, is_authored);
  }
  if (type == "vector4") {
    mx::Vector4 v = val->asA<mx::Vector4>();
    return EmitTuple({v[0], v[1], v[2], v[3]}, is_authored);
  }
  return default_expr;
}

Expr MtlxExprCompiler::CompileResolvedInput(const ResolvedInput& r,
                                            const Expr& default_expr) {
  if (r.kind == ResolvedInput::Connected) {
    NodeOutKey key{r.upstream.frame, r.upstream.node.get(), r.output_name};
    auto it = node_output_expr_.find(key);
    if (it == node_output_expr_.end()) {
      Expr se = FoldAndSpill(CompileNodeOutputExpr(r.upstream, r.output_name));
      it = node_output_expr_.emplace(std::move(key), std::move(se)).first;
    }
    return it->second;
  }
  if (r.kind == ResolvedInput::GeomProp) {
    return FoldAndSpill(
        CompileGeomProp(r.elem->asA<mx::Input>()->getDefaultGeomPropString(),
                        r.elem->getType()));
  }
  if (r.kind != ResolvedInput::Value) return default_expr;
  if (!IsAuthoredInput(r.elem)) {
    return CompileValueElement(r.elem, default_expr, /*is_authored=*/false);
  }
  // Cache raw Opaque/tuple expressions (never LoadTmp) so cached entries are
  // scope-safe across HeightToNormal and dead opaques remain compactable.
  auto [it, inserted] = authored_opaque_expr_.try_emplace(r.elem.get());
  if (inserted) {
    it->second =
        CompileValueElement(r.elem, default_expr, /*is_authored=*/true);
  }
  return it->second;
}

Expr MtlxExprCompiler::CompileInputExpr(const ScopedNode& s,
                                        const std::string& input_name,
                                        const Expr& default_expr) {
  return CompileResolvedInput(ResolveInput(s, input_name), default_expr);
}

std::optional<int> MtlxExprCompiler::ConstIntInput(const ScopedNode& node,
                                                   const std::string& name) {
  Expr e = CompileInputExpr(node, name, Expr{});
  if (e.channels != 1) return std::nullopt;
  if (auto v = TryEvalConst(e)) {
    return static_cast<int>(
        std::clamp<double>((*v)[0], std::numeric_limits<int>::min(),
                           std::numeric_limits<int>::max()));
  }
  return std::nullopt;
}

Expr MtlxExprCompiler::CompileIfCondition(const ScopedNode& node,
                                          std::string_view cat) {
  Expr v1 = CompileInputExpr(node, "value1");
  Expr v2 = CompileInputExpr(node, "value2");
  if (v1 == v2) return cat != "ifgreater" ? Expr::one() : Expr::zero();
  return FoldExpr(cat == "ifgreater"     ? (v1 > v2)
                  : cat == "ifgreatereq" ? (v1 >= v2)
                                         : call(mtlx::Op::Eq, v1, v2));
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::MatrixOp(mtlx::Op op,
                                                     const MatExpr& a,
                                                     const MatExpr* b) {
  if (prog_.tmp_channels.size() + a.n > mtlx::kMaxImm) {
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

MtlxExprCompiler::MatExpr MtlxExprCompiler::CompileResolvedMatrix(
    const ResolvedInput& r, int n) {
  if (r.kind == ResolvedInput::Connected) {
    return CompileMatrixNode(r.upstream, r.output_name);
  }
  mx::ValuePtr val =
      r.kind == ResolvedInput::Value ? r.elem->getValue() : nullptr;
  if (!val) return MatExpr::Identity(n);
  const std::string& t = r.elem->getType();
  if (t != "matrix33" && t != "matrix44") return MatExpr::Identity(n);
  bool is_authored = IsAuthoredInput(r.elem);
  if (is_authored) {
    if (auto it = authored_opaque_matrix_.find(r.elem.get());
        it != authored_opaque_matrix_.end()) {
      return it->second;
    }
  }
  MatExpr out;
  if (t == "matrix44") {
    out.n = 4;
    mx::Matrix44 m = val->asA<mx::Matrix44>();
    for (int i = 0; i < 4; ++i) {
      out.rows[i] =
          EmitTuple({m[i][0], m[i][1], m[i][2], m[i][3]}, is_authored);
    }
  } else {
    mx::Matrix33 m = val->asA<mx::Matrix33>();
    for (int i = 0; i < 3; ++i) {
      out.rows[i] = EmitTuple({m[i][0], m[i][1], m[i][2]}, is_authored);
    }
  }
  if (is_authored) authored_opaque_matrix_.emplace(r.elem.get(), out);
  return out;
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::CompileMatrixInput(
    const ScopedNode& node, const std::string& input_name) {
  return CompileResolvedMatrix(
      ResolveInput(node, input_name),
      InputType(node.node, input_name) == "matrix44" ? 4 : 3);
}

MtlxExprCompiler::MatExpr MtlxExprCompiler::CompileMatrixNode(
    const ScopedNode& node, const std::string& output_name) {
  NodeOutKey key{node.frame, node.node.get(), output_name};
  if (auto it = node_matrix_expr_.find(key); it != node_matrix_expr_.end()) {
    return it->second;
  }
  int n = node.node->getType() == "matrix44" ? 4 : 3;
  const std::string& cat = NodeCategory(node.node);
  MatExpr res = MatExpr::Identity(n);
  if (mx::NodeGraphPtr ng = GetNodeInfo(node.node).graph_impl) {
    mx::OutputPtr out = ResolveNodeGraphOutput(ng, output_name);
    res =
        CompileResolvedMatrix(ResolveOutput(node, out),
                              out ? (out->getType() == "matrix44" ? 4 : 3) : n);
  } else if (cat == "constant") {
    res = CompileMatrixInput(node, "value");
  } else if (cat == "dot") {
    res = CompileMatrixInput(node, "in");
  } else if (cat == "creatematrix") {
    // The rows of a 4x4 matrix are either `vector4` or `vector3` values, which
    // are extended by the fourth column of the identity.
    bool extend = n == 4 && InputType(node.node, "in1") != "vector4";
    for (int i = 0; i < n; ++i) {
      Expr row = CompileInputExpr(node, "in" + std::to_string(i + 1));
      res.rows[i] =
          extend ? vec4(row, i == 3 ? Expr::one() : Expr::zero()) : row;
    }
  } else if (cat == "transpose") {
    res = MatrixOp(mtlx::Op::MatTranspose, CompileMatrixInput(node, "in"));
  } else if (cat == "invertmatrix") {
    res = MatrixOp(mtlx::Op::MatInverse, CompileMatrixInput(node, "in"));
  } else if (cat == "add" || cat == "subtract") {
    MatExpr m1 = CompileMatrixInput(node, "in1");
    bool is_add = cat == "add";
    if (InputType(node.node, "in2") == "float") {
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
    if (cat == "divide") m2 = MatrixOp(mtlx::Op::MatInverse, m2);
    res = MatrixOp(mtlx::Op::MatMul, m1, &m2);
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

  for (int i = 0; i < res.n; ++i) res.rows[i] = FoldAndSpill(res.rows[i]);
  node_matrix_expr_[std::move(key)] = res;
  return res;
}

Expr MtlxExprCompiler::SampleImage(const ScopedNode& node) {
  auto default_val = [&] { return CompileInputExpr(node, "default"); };
  mx::ValueElementPtr file_in = GetValueInput(node, "file");
  if (!file_in || file_in->getValueString().empty()) return default_val();

  const std::string& type = node.node->getType();
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
    std::string cs = file_in->getActiveColorSpace();
    if (cs.empty()) {
      mx::InputPtr img_file = node.node->getInput("file");
      cs = img_file ? img_file->getActiveColorSpace()
                    : node.node->getActiveColorSpace();
    }
    key.color_space = ResolveSourceColorSpace(cs).GetString();
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
    return Expr({mtlx::Op::SampleInput, 0, discover_input_idx_++},
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
               call(mtlx::Op::Abs,
                    call(mtlx::Op::Fract, c * Const(0.5)) * Const(2.0) -
                        Expr::one());
      }
      return call(mtlx::Op::Fract, c);
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
      if (inputs_.size() >= mtlx::kMaxImm) {
        throw std::runtime_error("materialx_expr: too many texture inputs");
      }
      it = inputs_.insert(inputs_.end(), cached.texture);
    }
    uint16_t idx = static_cast<uint16_t>(it - inputs_.begin());
    uint8_t ch = cached.channel_count == 1 ? 1 : 3;
    if (default_uv) return Expr({mtlx::Op::SampleInput, ch, idx}, ch);
    return call_with_channels(mtlx::Op::SampleInputUV, ch, idx, u, v)
        .with_channels(ch);
  };

  Expr result = sample_tex(*cached_color);
  if (has_alpha) {
    result =
        vec4(result, cached_alpha ? sample_tex(*cached_alpha) : Expr::one());
  }
  if (type == "vector2") result = result.with_channels(2);
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

std::optional<Expr> MtlxExprCompiler::CompileChannelNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "constant") return in("value");
  if (cat == "dot" || cat == "convert") return in("in");
  if (cat == "extract") {
    int idx = ConstIntInput(node, "index").value_or(0);
    std::string_view in_type = InputType(node.node, "in");
    if (in_type == "matrix33" || in_type == "matrix44") {
      MatExpr mat = CompileMatrixInput(node, "in");
      return mat.rows[std::clamp(idx, 0, mat.n - 1)];
    }
    return in("in").chan(std::clamp(idx, 0, TypeChannels(in_type) - 1));
  }
  if (cat == "combine2") {
    if (IsFourChannelType(type)) {
      // A 3-channel value and a float, or two `vector2` values.
      if (InputType(node.node, "in1") != "vector2") {
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
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileMathNode(const ScopedNode& node,
                                                      const std::string& cat) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  auto in_slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return EmitTmpExpr(CompileInputExpr(node, name, def));
  };
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
    return call(mtlx::Op::Clip, x, lo, hi);
  }
  if (cat == "invert") return in("amount", Expr::one()) - in("in");
  if (cat == "sign") {
    Expr x = in_slot("in");
    return (x > Expr::zero()) - (x < Expr::zero());
  }
  if (cat == "modulo") {
    Expr x = in_slot("in1"), y = in_slot("in2", Expr::one());
    return x - y * call(mtlx::Op::Floor, x / y);
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
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileLogicNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
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
                                        : call(mtlx::Op::Max, b1, b2);
  }
  if (cat == "not") return !in("in");
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileVectorNode(
    const ScopedNode& node, const std::string& cat) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "normalize" || cat == "magnitude" || cat == "dotproduct") {
    std::string_view in_type =
        InputType(node.node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_imm = in_type == "vector2" ? 2 : in_type == "vector4" ? 4 : 0;
    if (cat == "normalize") return call(mtlx::Op::Normalize, vec_imm, in("in"));
    if (cat == "magnitude") {
      Expr v = EmitTmpExpr(in("in"));
      return call(mtlx::Op::Sqrt, call(mtlx::Op::Dot, vec_imm, v, v));
    }
    Expr i1 = in("in1"), i2 = in("in2");
    return call(mtlx::Op::Dot, vec_imm, i1, i2);
  }
  if (cat == "rotate3d") {
    Expr x = in("in"), amt = in("amount"),
         axis = in("axis", ConstRgb(0.0, 1.0, 0.0));
    return call(mtlx::Op::Rotate3D, x, amt, axis);
  }
  if (cat == "normalmap") {
    std::string space = GetNodeInputString(node, "space", "tangent");
    if (space != "tangent") {
      WarnUnsupported("Unsupported MaterialX normalmap space '" + space +
                      "'; evaluating in tangent space.");
    }
    Expr scale = in("scale", Expr::one());
    // A zero `scale` keeps only the sign of the z component of `in`, which is
    // positive for `heighttonormal` (e.g. in `NG_bump_vector3`), so skip
    // evaluating its subgraph.
    ScopedNode up =
        scale.is_zero() ? ResolveConnected(node, "in") : ScopedNode{};
    Expr in_e = up && NodeCategory(up.node) == "heighttonormal"
                    ? ConstRgb(0.5, 0.5, 1.0)
                    : in("in");
    Expr n = in("normal", Expr::n_world()), t = in("tangent", Expr::t_world()),
         b = in("bitangent", Expr::b_world());
    return call(mtlx::Op::NormalMap, in_e, scale, n, t, b);
  }
  if (cat == "heighttonormal") return CompileHeightToNormal(node);
  return std::nullopt;
}

Expr MtlxExprCompiler::CompileHeightToNormal(const ScopedNode& node) {
  Expr scale = CompileInputExpr(node, "scale", Expr::one());
  if (scale.is_zero()) return ConstRgb(0.5, 0.5, 1.0);

  // Compile `in` (and `texcoord` if non-default) into an isolated temporary
  // scope so `Op::HeightToNormal` can re-evaluate the subgraph at perturbed
  // surface interactions without referencing outer temporaries.
  auto scope = [&] {
    return std::tie(prog_.code, prog_.tmp_channels, prog_.stack_size,
                    node_output_expr_, node_matrix_expr_, expr_to_slot_);
  };
  auto outer = std::apply(
      [](auto&... state) { return std::tuple(std::exchange(state, {})...); },
      scope());
  prog_.stack_size = 1;

  std::optional<uint16_t> sub_idx;
  Expr h = CompileInputExpr(node, "in");
  auto op_varies = [](const mtlx::Inst& i) {
    return mtlx::is_spatially_varying_op(i.op) || i.op == mtlx::Op::SampleInput;
  };
  if (AnyLiveInst(prog_.code, h.code, prog_.tmp_channels.size(), op_varies)) {
    Expr tc = CompileInputExpr(node, "texcoord", DefaultTexcoord());
    if (discover_texture_) {
      sub_idx = discover_input_idx_++;
    } else {
      bool has_tc = tc != DefaultTexcoord();
      if (has_tc) tc = EmitTmpExpr(tc);
      Expr out = has_tc ? rgb(h.x(), tc.x(), tc.y()) : h.x();
      if (prog_.sub_programs.size() >= mtlx::kMaxImm) {
        throw std::runtime_error(
            "materialx_expr: too many heighttonormal nodes");
      }
      uint32_t begin = static_cast<uint32_t>(prog_.sub_code.size());
      prog_.sub_code.insert(prog_.sub_code.end(), prog_.code.begin(),
                            prog_.code.end());
      prog_.sub_code.insert(prog_.sub_code.end(), out.code.begin(),
                            out.code.end());
      prog_.sub_programs.push_back(
          {begin, static_cast<uint32_t>(prog_.sub_code.size()),
           static_cast<uint32_t>(prog_.tmp_channels.size()),
           std::max(prog_.stack_size, out.max_depth), has_tc ? 1u : 0u});
      sub_idx = static_cast<uint16_t>(prog_.sub_programs.size() - 1);
    }
  }

  scope() = std::move(outer);
  return sub_idx ? call(mtlx::Op::HeightToNormal, *sub_idx, scale)
                 : ConstRgb(0.5, 0.5, 1.0);
}

Expr MtlxExprCompiler::CompileGeomIntrinsic(std::string_view kind,
                                            std::string_view space,
                                            std::string_view type,
                                            bool non_zero_index) {
  if (non_zero_index && (kind == "texcoord" || kind == "tangent" ||
                         kind == "bitangent" || kind == "geomcolor")) {
    WarnUnsupported("Unsupported MaterialX '" + std::string(kind) +
                    "' index; using index 0.");
  }
  if (space.empty()) space = kind == "viewdirection" ? "world" : "object";
  auto warn_world_space = [&] {
    if (space != "world") {
      WarnUnsupported("Unsupported MaterialX " + std::string(kind) +
                      " space '" + std::string(space) +
                      "'; evaluating in world space.");
    }
  };
  if (kind == "texcoord") return DefaultTexcoord(type == "vector3" ? 3 : 2);
  if (kind == "geomcolor") {
    Expr color =
        MeshAttr("displayColor", type == "float" ? 1 : 3, Expr::zero());
    // The alpha channel of `geomcolor` is the display opacity.
    return type == "color4"
               ? vec4(color, MeshAttr("displayOpacity", 1, Expr::one()))
               : color;
  }
  if (kind == "position") {
    if (space == "object" || space == "model") {
      return MeshAttr("__p_object", 3, Expr::p_world());
    }
    warn_world_space();
    return Expr::p_world();
  }
  if (kind == "normal") {
    if (space == "object" || space == "model") {
      return call(mtlx::Op::Normalize,
                  MeshAttr("__n_object", 3, Expr::n_world()));
    }
    if (space == "tangent") return ConstRgb(0.0, 0.0, 1.0);
    warn_world_space();
    return Expr::n_world();
  }
  if (kind == "viewdirection") {
    warn_world_space();
    return Expr({mtlx::Op::GeomIncidentDir, 0, 0});
  }
  bool bitangent = kind == "bitangent";
  if (space == "tangent") {
    return bitangent ? ConstRgb(0.0, 1.0, 0.0) : ConstRgb(1.0, 0.0, 0.0);
  }
  warn_world_space();
  return bitangent ? Expr::b_world() : Expr::t_world();
}

Expr MtlxExprCompiler::CompileGeomProp(
    const std::string& prop_name, std::string_view type,
    const std::function<Expr()>& default_fn) {
  uint8_t ch = TypeChannels(type);
  if (mx::GeomPropDefPtr gpd = doc_->getGeomPropDef(prop_name)) {
    const std::string& gp = gpd->getGeomProp();
    if (IsGeomIntrinsic(gp)) {
      std::string_view idx = gpd->getIndex();
      Expr e = CompileGeomIntrinsic(gp, gpd->getSpace(), type,
                                    !idx.empty() && idx != "0");
      return ch == 1 ? e.x() : e.with_channels(ch);
    }
  }
  return MeshAttr(prop_name, ch == 1 ? 1 : 3,
                  default_fn ? default_fn() : Expr::zero())
      .with_channels(ch);
}

std::optional<Expr> MtlxExprCompiler::CompileGeomNode(const ScopedNode& node,
                                                      const std::string& cat,
                                                      const std::string& type) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };

  if (cat == "image") return SampleImage(node);
  if (IsGeomIntrinsic(cat)) {
    return CompileGeomIntrinsic(cat, GetNodeInputString(node, "space"), type,
                                ConstIntInput(node, "index").value_or(0) != 0);
  }
  if (cat == "geompropvalue") {
    std::string p = GetNodeInputString(node, "geomprop");
    if (p.empty()) return in("default");
    return CompileGeomProp(p, type, [&] { return in("default"); });
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
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileMatrixValueNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  if (cat == "transformmatrix") {
    // Row vectors are transformed as `v * M`. The last coordinate of a vector
    // with fewer coordinates than the matrix is 1.
    Expr v = EmitTmpExpr(CompileInputExpr(node, "in"));
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
    uint8_t ch = static_cast<uint8_t>(m.n);
    return m.n == 4 ? call_with_channels(mtlx::Op::MatDet, ch, 0, m.rows[0],
                                         m.rows[1], m.rows[2], m.rows[3])
                    : call_with_channels(mtlx::Op::MatDet, ch, 0, m.rows[0],
                                         m.rows[1], m.rows[2]);
  }
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileCompositingNode(
    const ScopedNode& node, const std::string& cat) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  auto in_slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return EmitTmpExpr(CompileInputExpr(node, name, def));
  };
  if (cat == "plus" || cat == "minus" || cat == "difference" ||
      cat == "screen" || cat == "burn" || cat == "dodge") {
    Expr fg = in_slot("fg"), bg = in_slot("bg"), m = in("mix", Expr::one());
    // Like MaterialX, burn and dodge are 0 where they divide by 0.
    if (cat == "burn" || cat == "dodge") {
      Expr denom = cat == "burn" ? fg : EmitTmpExpr(Expr::one() - fg);
      Expr blend =
          cat == "burn" ? Expr::one() - (Expr::one() - bg) / denom : bg / denom;
      return select(call(mtlx::Op::Abs, denom) < Const(mtlx::kMtlxFloatEps),
                    Expr::zero(), mix(blend, bg, m));
    }
    Expr blend = cat == "plus"         ? (bg + fg)
                 : cat == "minus"      ? (bg - fg)
                 : cat == "difference" ? call(mtlx::Op::Abs, bg - fg)
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
    if (cat == "disjointover") {
      fa = EmitTmpExpr(fa);
      ba = EmitTmpExpr(ba);
    }
    Expr composite =
        cat == "over"   ? fg + bg * (Expr::one() - fa)
        : cat == "in"   ? fg * ba
        : cat == "out"  ? fg * (Expr::one() - ba)
        : cat == "mask" ? bg * fa
        : cat == "matte"
            ? fg * fa + bg * (Expr::one() - fa)
            : select(
                  fa + ba <= Expr::one(), fg + bg,
                  select(call(mtlx::Op::Abs, ba) < Const(mtlx::kMtlxFloatEps),
                         Expr::zero(), fg + bg * ((Expr::one() - fa) / ba)));
    // The alpha channels of `matte` and `disjointover` are not composited like
    // their colors.
    if (cat == "matte") {
      composite = vec4(composite.xyz(), fa + ba * (Expr::one() - fa));
    } else if (cat == "disjointover") {
      composite =
          vec4(composite.xyz(), call(mtlx::Op::Min, fa + ba, Expr::one()));
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
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileColorPbrNode(
    const ScopedNode& node, const std::string& cat, const std::string& type,
    const std::string& output_name) {
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  auto in_slot = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return EmitTmpExpr(CompileInputExpr(node, name, def));
  };
  if (cat == "rgbtohsv" || cat == "hsvtorgb") {
    Expr e = call(cat == "rgbtohsv" ? mtlx::Op::RgbToHsv : mtlx::Op::HsvToRgb,
                  in("in"));
    return type == "color4" ? e.with_channels(4) : e;
  }
  if (cat == "luminance") {
    // The luminance of a gray value is the value itself, which keeps e.g. the
    // default `opacity` color of `standard_surface` a constant 1.
    Expr in_e = in("in");
    ResolvedInput cr = ResolveInput(node, "lumacoeffs");
    bool has_coeffs = cr.kind != ResolvedInput::None &&
                      !(cr.kind == ResolvedInput::Value &&
                        cr.elem->getParent() == GetNodeInfo(node.node).nodedef);
    if (!has_coeffs && in_e.uniform()) return in_e;
    if (type == "color4") in_e = EmitTmpExpr(in_e);
    Expr lum = call(mtlx::Op::Dot, in_e,
                    has_coeffs ? CompileResolvedInput(cr, Expr::zero())
                               : ConstRgb(0.2722287, 0.6740818, 0.0536895));
    return type == "color4" ? vec4(lum, in_e.w()) : lum;
  }
  if (cat == "blackbody") {
    return call(mtlx::Op::Blackbody, in("temperature", Const(5000.0)));
  }
  if (cat == "roughness_anisotropy") {
    Expr r = in_slot("roughness");
    Expr r2 = EmitTmpExpr(
        call(mtlx::Op::Clip, r * r, Const(mtlx::kMtlxFloatEps), Expr::one()));
    Expr aspect = EmitTmpExpr(call(
        mtlx::Op::Sqrt, Expr::one() - call(mtlx::Op::Clip, in("anisotropy"),
                                           Expr::zero(), Const(0.98))));
    return vec2(call(mtlx::Op::Min, r2 / aspect, Expr::one()), r2 * aspect);
  }
  if (cat == "roughness_dual") {
    Expr r = in_slot("roughness"), rx = EmitTmpExpr(r.x());
    Expr ry = EmitTmpExpr(select(r.y() < Expr::zero(), rx, r.y()));
    return vec2(
        call(mtlx::Op::Clip, rx * rx, Const(mtlx::kMtlxFloatEps), Expr::one()),
        call(mtlx::Op::Clip, ry * ry, Const(mtlx::kMtlxFloatEps), Expr::one()));
  }
  if (cat == "artistic_ior") {
    Expr r = EmitTmpExpr(
        call(mtlx::Op::Clip, in("reflectivity"), Expr::zero(), Const(0.99)));
    Expr r_sqrt = EmitTmpExpr(call(mtlx::Op::Sqrt, r));
    Expr eta =
        mix((Expr::one() - r) / (Expr::one() + r),
            (Expr::one() + r_sqrt) / (Expr::one() - r_sqrt), in("edge_color"));
    if (output_name != "extinction") return eta;
    eta = EmitTmpExpr(eta);
    Expr np1 = EmitTmpExpr(eta + Expr::one()),
         nm1 = EmitTmpExpr(eta - Expr::one());
    return call(mtlx::Op::Sqrt,
                (np1 * np1 * r - nm1 * nm1) / (Expr::one() - r));
  }
  if (cat == "chiang_hair_roughness") {
    auto clip_in = [&](const char* name, double def) {
      return EmitTmpExpr(
          call(mtlx::Op::Clip, in(name, Const(def)), Const(1e-3), Expr::one()));
    };
    Expr lr = clip_in("longitudinal", 0.1), ar = clip_in("azimuthal", 0.2);
    Expr v_sqrt =
        EmitTmpExpr(Const(0.726) * lr + Const(0.812) * lr * lr +
                    Const(3.7) * call(mtlx::Op::Pow, lr, Const(20.0)));
    if (output_name == "roughness_TT" || output_name == "roughness_TRT") {
      bool is_tt = output_name == "roughness_TT";
      v_sqrt = EmitTmpExpr(v_sqrt * in(is_tt ? "scale_TT" : "scale_TRT",
                                       Const(is_tt ? 0.5 : 2.0)));
    }
    Expr s = EmitTmpExpr(Const(0.265) * ar + Const(1.194) * ar * ar +
                         Const(5.372) * call(mtlx::Op::Pow, ar, Const(22.0)));
    return vec2(v_sqrt * v_sqrt, s);
  }
  if (cat == "chiang_hair_absorption_from_color") {
    Expr c = call(mtlx::Op::Clip, in("color", Expr::one()), Const(1e-3),
                  Expr::one());
    Expr b = EmitTmpExpr(in("azimuthal_roughness", Const(0.2)));
    Expr b_fac =
        Const(5.969) +
        b * (Const(-0.215) +
             b * (Const(2.532) +
                  b * (Const(-10.73) + b * (Const(5.574) + b * Const(0.245)))));
    Expr sigma = EmitTmpExpr(call(mtlx::Op::Log, c) / b_fac);
    return sigma * sigma;
  }
  if (cat == "deon_hair_absorption_from_melanin") {
    Expr log_conc =
        call(mtlx::Op::Log,
             call(mtlx::Op::Max,
                  Expr::one() - in("melanin_concentration", Const(0.25)),
                  Const(1e-4)));
    Expr pigment = CompileMix(
        in("melanin_redness", Const(0.5)),
        [&] {
          return call(mtlx::Op::Log, in("pheomelanin_color",
                                        ConstRgb(0.829444, 0.67032, 0.349938)));
        },
        [&] {
          return call(
              mtlx::Op::Log,
              in("eumelanin_color", ConstRgb(0.657704, 0.498077, 0.254107)));
        });
    return call(mtlx::Op::Max, log_conc * pigment, Expr::zero());
  }
  return std::nullopt;
}

std::optional<Expr> MtlxExprCompiler::CompileNoiseNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  std::string_view noise = cat;
  bool is_3d = absl::ConsumeSuffix(&noise, "3d");
  if ((!is_3d && !absl::ConsumeSuffix(&noise, "2d")) ||
      (noise != "cellnoise" && noise != "worleynoise" && noise != "noise" &&
       noise != "fractal")) {
    return std::nullopt;
  }
  auto in = [&](const std::string& name, const Expr& def = Expr::zero()) {
    return CompileInputExpr(node, name, def);
  };
  uint16_t imm = mtlx::pack_noise_imm(is_3d, type != "float");
  Expr coord = is_3d
                   ? in("position", MeshAttr("__p_object", 3, Expr::p_world()))
                   : in("texcoord", DefaultTexcoord());
  if (noise == "cellnoise") {
    return call(mtlx::Op::CellNoise, imm, coord);
  }
  if (noise == "worleynoise") {
    Expr jitter = in("jitter", Expr::one()), style = in("style");
    Expr w = call(mtlx::Op::Worley, imm, coord, jitter, style);
    return type == "vector2" ? w.with_channels(2) : w;
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
    if (is_noise) {
      return call(mtlx::Op::Noise, mtlx::pack_noise_imm(is_3d, vec3), p);
    }
    if (oct) {
      uint16_t clamped =
          static_cast<uint16_t>(std::clamp(*oct, 0, mtlx::kMaxFractalOctaves));
      return call(mtlx::Op::Fractal, mtlx::pack_noise_imm(is_3d, vec3, clamped),
                  p, lac, dim);
    }
    return call(mtlx::Op::Fractal,
                mtlx::pack_noise_imm(is_3d, vec3, mtlx::kFractalDynamicOctaves),
                p, lac, dim, dyn_oct);
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
    if (type == "vector2") res = res.with_channels(2);
  }
  res = res * in("amplitude", Expr::one());
  return is_noise ? res + in("pivot") : res;
}

Expr MtlxExprCompiler::CompileNodeOutputExpr(const ScopedNode& node,
                                             const std::string& output_name) {
  const std::string& cat = NodeCategory(node.node);
  const std::string& type = node.node->getType();

  // Handle noise before NodeGraph expansion so stdlib noise/fractal NodeGraphs
  // emit native VM noise ops directly.
  if (auto e = CompileNoiseNode(node, cat, type)) return *e;

  if (mx::NodeGraphPtr ng = GetNodeInfo(node.node).graph_impl) {
    return CompileResolvedInput(
        ResolveOutput(node, ResolveNodeGraphOutput(ng, output_name)),
        Expr::zero());
  }

  // Unary and binary ops that map to a single instruction.
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
      return call(op, CompileInputExpr(
                          node, "in",
                          op == mtlx::Op::Log ? Expr::one() : Expr::zero()));
    }
  }
  static constexpr struct {
    std::string_view name;
    const char *in1, *in2;
    mtlx::Op op;
    bool def2_one;
  } kBinaryOps[] = {
      {"min", "in1", "in2", mtlx::Op::Min, false},
      {"max", "in1", "in2", mtlx::Op::Max, false},
      {"power", "in1", "in2", mtlx::Op::Pow, true},
      {"atan2", "iny", "inx", mtlx::Op::Atan2, true},
      {"crossproduct", "in1", "in2", mtlx::Op::Cross, false},
      {"rotate2d", "in", "amount", mtlx::Op::Rotate2D, false},
  };
  for (const auto& [name, in1, in2, op, def2_one] : kBinaryOps) {
    if (cat == name) {
      Expr e1 = CompileInputExpr(node, in1);
      Expr e2 =
          CompileInputExpr(node, in2, def2_one ? Expr::one() : Expr::zero());
      return call(op, e1, e2);
    }
  }

  if (auto e = CompileChannelNode(node, cat, type)) return *e;
  if (auto e = CompileMathNode(node, cat)) return *e;
  if (auto e = CompileLogicNode(node, cat, type)) return *e;
  if (auto e = CompileVectorNode(node, cat)) return *e;
  if (auto e = CompileGeomNode(node, cat, type)) return *e;
  if (auto e = CompileMatrixValueNode(node, cat, type)) return *e;
  if (auto e = CompileCompositingNode(node, cat)) return *e;
  if (auto e = CompileColorPbrNode(node, cat, type, output_name)) return *e;

  WarnUnsupported("Unsupported MaterialX node category '" + cat +
                  "'; passing through its 'in' input.");
  return CompileInputExpr(node, "in");
}

Expr MtlxExprCompiler::CompileEdf(const ScopedNode& node) {
  if (!node) return Expr::zero();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = edf_expr_.find(key); it != edf_expr_.end()) {
    return it->second;
  }
  const std::string& cat = NodeCategory(node.node);
  Expr res = Expr::zero();
  if (cat == "surface" || cat == "surface_unlit") {
    if (cat == "surface") {
      res = CompileEdf(ResolveConnected(node, "edf"));
    } else {
      Expr em = CompileInputExpr(node, "emission", Expr::one());
      if (!em.is_zero()) {
        em = em * FoldExpr(Expr::one() -
                           saturate(CompileInputExpr(node, "transmission")));
      }
      if (!em.is_zero()) {
        res = em * CompileInputExpr(node, "emission_color", Expr::one());
      }
    }
    if (!res.is_zero()) {
      res = res *
            FoldExpr(saturate(CompileInputExpr(node, "opacity", Expr::one())));
    }
  } else if (cat == "uniform_edf" || cat == "conical_edf" ||
             cat == "measured_edf") {
    if (cat == "measured_edf" && !GetNodeInputString(node, "file").empty()) {
      WarnUnsupported(
          "Unsupported MaterialX 'measured_edf' IES profile; approximating as "
          "uniform_edf.");
    } else if (cat != "uniform_edf" && !discover_texture_) {
      TF_DEBUG(HDMITSUBA_MATERIAL)
          .Msg("Approximating MaterialX '%s' as uniform_edf.\n", cat.c_str());
    }
    res = CompileInputExpr(node, "color", Expr::one());
  } else if (cat == "generalized_schlick_edf") {
    Expr base = CompileEdf(ResolveConnected(node, "base"));
    if (!base.is_zero()) {
      Expr c0 = CompileInputExpr(node, "color0", Expr::one());
      Expr c90 = CompileInputExpr(node, "color90", Expr::one());
      // Weight `base` by the cosine-weighted hemispherical average of
      // `mix(c0, c90, (1 - cos)^exp)`, which equals
      // `mix(c0, c90, 2 / ((exp + 1) * (exp + 2)))`.
      Expr w = c0;
      if (c0 != c90) {
        Expr exp = EmitTmpExpr(
            call(mtlx::Op::Max, CompileInputExpr(node, "exponent", Const(5.0)),
                 Expr::zero()));
        w = mix(c90, c0,
                Const(2.0) / ((exp + Expr::one()) * (exp + Const(2.0))));
      }
      res = base * w;
    }
  } else if (cat == "add") {
    res = CompileEdf(ResolveConnected(node, "in1")) +
          CompileEdf(ResolveConnected(node, "in2"));
  } else if (cat == "multiply") {
    Expr base = CompileEdf(ResolveConnected(node, "in1"));
    if (!base.is_zero()) {
      res = base * CompileInputExpr(node, "in2", Expr::one());
    }
  } else if (cat == "mix") {
    res = CompileMix(
        CompileInputExpr(node, "mix"),
        [&] { return CompileEdf(ResolveConnected(node, "fg")); },
        [&] { return CompileEdf(ResolveConnected(node, "bg")); });
  }
  res = FoldAndSpill(res);
  edf_expr_[key] = res;
  return res;
}

Expr MtlxExprCompiler::CompileOpacity(const ScopedNode& node) {
  if (!node) return Expr::one();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = opacity_expr_.find(key); it != opacity_expr_.end()) {
    return it->second;
  }
  const std::string& cat = NodeCategory(node.node);
  Expr res = Expr::one();
  if (cat == "surface" || cat == "surface_unlit") {
    res = saturate(CompileInputExpr(node, "opacity", Expr::one()));
  } else if (cat == "mix" &&
             node.node->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
    res = CompileMix(
        FoldExpr(saturate(CompileInputExpr(node, "mix"))),
        [&] { return CompileOpacity(ResolveConnected(node, "fg")); },
        [&] { return CompileOpacity(ResolveConnected(node, "bg")); });
  }
  if (!res.uniform()) res = call(mtlx::Op::Dot, res, Const(1.0 / 3.0));
  res = FoldAndSpill(res);
  opacity_expr_[key] = res;
  return res;
}

void MtlxExprCompiler::MarkTwoSidedDielectrics(size_t start) {
  for (size_t i = start; i < prog_.closures.size(); ++i) {
    const mtlx::ClosureNode& c = prog_.closures[i];
    if ((c.op == mtlx::ClosureOp::Dielectric ||
         c.op == mtlx::ClosureOp::GeneralizedSchlick) &&
        (c.microfacet.mode & mtlx::kScatterModeMask) != mtlx::kScatterR) {
      return;
    }
  }
  for (size_t i = start; i < prog_.closures.size(); ++i) {
    mtlx::ClosureNode& c = prog_.closures[i];
    if (c.op == mtlx::ClosureOp::Dielectric) {
      c.microfacet.mode |= mtlx::kScatterTwoSided;
    }
  }
}

int MtlxExprCompiler::EmitClosure(mtlx::ClosureNode node) {
  prog_.closures.push_back(node);
  return static_cast<int>(prog_.closures.size()) - 1;
}

int MtlxExprCompiler::EmitOpaqueBlocker() {
  // Opaque and transmissive leaves still block `layer` bases (`throughput = 0`)
  // at `weight == 0`, without compiling dead inputs.
  int z = EmitTmp(Expr::zero());
  return EmitClosure(mtlx::ClosureNode::MakeDiffuse(z, z, z, 0, -1));
}

int MtlxExprCompiler::FrameSlot(const ScopedNode& node, const std::string& name,
                                const Expr& def) {
  Expr e = CompileInputExpr(node, name, def);
  return e == def ? -1 : EmitTmp(e);
}

int MtlxExprCompiler::CompileDiffuseClosure(const ScopedNode& node,
                                            const std::string& cat) {
  Expr w_expr = CompileInputExpr(node, "weight", Expr::one());
  if (w_expr.is_zero()) return cat == "sheen_bsdf" ? -1 : EmitOpaqueBlocker();
  int w_slot = EmitTmp(w_expr);
  int color_slot = EmitTmp(CompileInputExpr(node, "color", Expr::one()));
  int rough_slot = EmitTmp(CompileInputExpr(node, "roughness"));
  int n_slot = FrameSlot(node, "normal", Expr::n_world());
  if (cat == "burley_diffuse_bsdf") {
    return EmitClosure(mtlx::ClosureNode::MakeBurleyDiffuse(
        w_slot, color_slot, rough_slot, n_slot));
  }
  if (cat == "sheen_bsdf") {
    int mode = GetNodeInputString(node, "mode") == "zeltner"
                   ? mtlx::kSheenZeltner
                   : mtlx::kSheenContyKulla;
    return EmitClosure(mtlx::ClosureNode::MakeSheen(w_slot, color_slot,
                                                    rough_slot, mode, n_slot));
  }
  int energy_comp = cat == "oren_nayar_diffuse_bsdf" &&
                    !CompileInputExpr(node, "energy_compensation").is_zero();
  return EmitClosure(mtlx::ClosureNode::MakeDiffuse(
      w_slot, color_slot, rough_slot, energy_comp, n_slot));
}

int MtlxExprCompiler::CompileTranslucentClosure(const ScopedNode& node) {
  Expr w_expr = CompileInputExpr(node, "weight", Expr::one());
  if (w_expr.is_zero()) return EmitOpaqueBlocker();
  int w_slot = EmitTmp(w_expr);
  int color_slot = EmitTmp(CompileInputExpr(node, "color", Expr::one()));
  int n_slot = FrameSlot(node, "normal", Expr::n_world());
  return EmitClosure(
      mtlx::ClosureNode::MakeTranslucent(w_slot, color_slot, n_slot));
}

int MtlxExprCompiler::CompileMicrofacetClosure(const ScopedNode& node,
                                               const std::string& cat) {
  Expr w_expr = CompileInputExpr(node, "weight", Expr::one());
  bool is_cond = cat == "conductor_bsdf";
  int sm = mtlx::kScatterR;
  if (!is_cond) {
    std::string sm_str = GetNodeInputString(node, "scatter_mode");
    sm = sm_str == "T"    ? mtlx::kScatterT
         : sm_str == "RT" ? mtlx::kScatterRT
                          : mtlx::kScatterR;
  }
  if (w_expr.is_zero()) {
    return (!is_cond && sm == mtlx::kScatterR) ? -1 : EmitOpaqueBlocker();
  }
  int tf_t = -1, tf_ior = -1;
  if (sm != mtlx::kScatterT) {
    if (Expr thickness = CompileInputExpr(node, "thinfilm_thickness");
        !thickness.is_zero()) {
      tf_t = EmitTmp(thickness);
      tf_ior = EmitTmp(CompileInputExpr(node, "thinfilm_ior", Const(1.5)));
    }
  }
  int mode =
      sm |
      (!CompileInputExpr(node, "retroreflective").is_zero()
           ? mtlx::kScatterRetroreflective
           : 0) |
      ((is_cond || (cat == "generalized_schlick_bsdf" && sm == mtlx::kScatterR))
           ? mtlx::kScatterTwoSided
           : 0);
  int w_slot = EmitTmp(w_expr);
  int rough_slot = EmitTmp(CompileInputExpr(node, "roughness"));
  int n_slot = FrameSlot(node, "normal", Expr::n_world());
  int t_slot = FrameSlot(node, "tangent", Expr::t_world());
  if (cat == "dielectric_bsdf") {
    return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
        mtlx::ClosureOp::Dielectric, w_slot, rough_slot, mode, n_slot, t_slot,
        tf_t, tf_ior, EmitTmp(CompileInputExpr(node, "tint", Expr::one())),
        EmitTmp(CompileInputExpr(node, "ior", Expr::one()))));
  }
  if (is_cond) {
    return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
        mtlx::ClosureOp::Conductor, w_slot, rough_slot, mode, n_slot, t_slot,
        tf_t, tf_ior, EmitTmp(CompileInputExpr(node, "ior", Expr::one())),
        EmitTmp(CompileInputExpr(node, "extinction"))));
  }
  return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
      mtlx::ClosureOp::GeneralizedSchlick, w_slot, rough_slot, mode, n_slot,
      t_slot, tf_t, tf_ior,
      EmitTmp(CompileInputExpr(node, "color0", Expr::one())),
      EmitTmp(CompileInputExpr(node, "color82", Expr::one())),
      EmitTmp(CompileInputExpr(node, "color90", Expr::one())),
      EmitTmp(CompileInputExpr(node, "exponent", Const(5.0)))));
}

int MtlxExprCompiler::CompileChiangHairClosure(const ScopedNode& node) {
  auto slot = [&](const char* name, const Expr& def = Expr::zero()) {
    return EmitTmp(CompileInputExpr(node, name, def));
  };
  return EmitClosure(mtlx::ClosureNode::MakeChiangHair(
      slot("tint_R", Expr::one()), slot("tint_TT", Expr::one()),
      slot("tint_TRT", Expr::one()),
      slot("roughness_R", vec2(Const(0.1), Const(0.1))),
      slot("roughness_TT", vec2(Const(0.05), Const(0.05))),
      slot("roughness_TRT", vec2(Const(0.2), Const(0.2))),
      slot("ior", Const(1.55)), slot("absorption_coefficient"),
      slot("cuticle_angle", Const(0.5)),
      FrameSlot(node, "normal", Expr::n_world()),
      FrameSlot(node, "curve_direction", Expr::t_world())));
}

void MtlxExprCompiler::CompileVdfWarning(const ScopedNode& node,
                                         const std::string& cat) {
  if (cat == "anisotropic_vdf" || cat == "absorption_vdf") {
    if (IsNonZeroInput(node, "absorption") ||
        (cat == "anisotropic_vdf" && IsNonZeroInput(node, "scattering"))) {
      WarnUnsupported("Unsupported MaterialX VDF closure '" + cat +
                      "'; ignoring volume absorption/scattering.");
    }
  } else {
    WarnUnsupported("Unsupported MaterialX VDF closure '" + cat +
                    "'; ignoring volume.");
  }
}

int MtlxExprCompiler::CompileBsdfClosure(const ScopedNode& node) {
  if (!node) return -1;
  NodeKey key{node.frame, node.node.get()};
  if (auto it = bsdf_closure_.find(key); it != bsdf_closure_.end()) {
    return it->second;
  }

  const std::string& cat = NodeCategory(node.node);
  auto scale = [&](int c, const Expr& w) {
    return (c < 0 || w.is_one())
               ? c
               : EmitClosure(mtlx::ClosureNode::MakeMultiply(c, EmitTmp(w)));
  };

  int idx = -1;
  if (cat == "oren_nayar_diffuse_bsdf" || cat == "subsurface_bsdf" ||
      cat == "burley_diffuse_bsdf" || cat == "sheen_bsdf") {
    idx = CompileDiffuseClosure(node, cat);
  } else if (cat == "translucent_bsdf") {
    idx = CompileTranslucentClosure(node);
  } else if (cat == "dielectric_bsdf" || cat == "conductor_bsdf" ||
             cat == "generalized_schlick_bsdf") {
    idx = CompileMicrofacetClosure(node, cat);
  } else if (cat == "chiang_hair_bsdf") {
    idx = CompileChiangHairClosure(node);
  } else if (node.node->getType() == mx::VDF_TYPE_STRING) {
    CompileVdfWarning(node, cat);
  } else if (cat == "layer" || cat == "add") {
    bool is_layer = cat == "layer";
    int c0 =
        CompileBsdfClosure(ResolveConnected(node, is_layer ? "top" : "in1"));
    int c1 =
        CompileBsdfClosure(ResolveConnected(node, is_layer ? "base" : "in2"));
    if (c0 < 0) {
      idx = c1;
    } else if (c1 < 0) {
      idx = c0;
    } else {
      idx = EmitClosure(is_layer ? mtlx::ClosureNode::MakeLayer(c0, c1)
                                 : mtlx::ClosureNode::MakeAdd(c0, c1));
    }
  } else if (cat == "mix") {
    Expr mix_expr = CompileInputExpr(node, "mix");
    Expr mix_sat = FoldExpr(saturate(mix_expr));
    if (mix_sat.is_zero()) {
      idx = CompileBsdfClosure(ResolveConnected(node, "bg"));
    } else if (mix_sat.is_one()) {
      idx = CompileBsdfClosure(ResolveConnected(node, "fg"));
    } else {
      int fg_idx = CompileBsdfClosure(ResolveConnected(node, "fg"));
      int bg_idx = CompileBsdfClosure(ResolveConnected(node, "bg"));
      if (fg_idx == bg_idx) {
        idx = fg_idx;
      } else {
        if (fg_idx < 0) fg_idx = EmitClosure(mtlx::ClosureNode::MakeNull());
        if (bg_idx < 0) bg_idx = EmitClosure(mtlx::ClosureNode::MakeNull());
        idx = EmitClosure(
            mtlx::ClosureNode::MakeMix(fg_idx, bg_idx, EmitTmp(mix_expr)));
      }
    }
  } else if (cat == "multiply") {
    Expr w = CompileInputExpr(node, "in2", Expr::one());
    idx = scale(CompileBsdfClosure(ResolveConnected(node, "in1")), w);
  } else if (cat == "surface") {
    if (Expr op = CompileOpacity(node); !op.is_zero()) {
      if (!CompileInputExpr(node, "thin_walled").is_zero()) thin_walled_ = true;
      // Scope BSDF deduplication to this surface so `MarkTwoSidedDielectrics`
      // cannot mutate a closure shared with another mixed surface.
      auto saved = std::exchange(bsdf_closure_, {});
      size_t start = prog_.closures.size();
      int bsdf_idx = CompileBsdfClosure(ResolveConnected(node, "bsdf"));
      MarkTwoSidedDielectrics(start);
      bsdf_closure_ = std::move(saved);
      idx = scale(bsdf_idx, op);
    }
  } else if (cat == "surface_unlit") {
    // Unlit surfaces emit radiance only and have a null BSDF.
  } else if (node.node->getType() == mx::BSDF_TYPE_STRING &&
             InputType(node.node, "weight") == "float" &&
             CompileInputExpr(node, "weight", Expr::one()).is_zero()) {
    // Skip inactive unsupported lobes (e.g. zero-weight lobes in
    // `standard_surface`) without warning.
  } else {
    WarnUnsupported("Unsupported MaterialX closure '" + cat +
                    "'; using a null BSDF.");
  }

  bsdf_closure_[key] = idx;
  return idx;
}

void MtlxExprCompiler::CompileBsdf(const mx::NodePtr& shader_node,
                                   std::string_view output_name) {
  ScopedNode node =
      UnwrapShaderNode({shader_node, nullptr}, std::string(output_name));
  Expr op_expr = CompileOpacity(node);
  if (!op_expr.is_one()) prog_.opacity_slot = EmitTmp(op_expr);
  if (op_expr.is_zero() || CompileBsdfClosure(node) < 0) {
    prog_.closures = {mtlx::ClosureNode::MakeNull()};
  } else if (thin_walled_) {
    for (mtlx::ClosureNode& c : prog_.closures) {
      if (c.op == mtlx::ClosureOp::Dielectric ||
          c.op == mtlx::ClosureOp::GeneralizedSchlick) {
        c.microfacet.mode |= mtlx::kScatterThinWalled | mtlx::kScatterTwoSided;
      }
    }
  } else if (node.node->getType() != mx::SURFACE_SHADER_TYPE_STRING) {
    MarkTwoSidedDielectrics(0);
  }
}

MtlxExprCompiler::DisplacementExpr MtlxExprCompiler::CompileDisplacement(
    const ScopedNode& node) {
  if (!node) return {};
  NodeKey key{node.frame, node.node.get()};
  if (auto it = disp_expr_.find(key); it != disp_expr_.end()) {
    return it->second;
  }
  const std::string& cat = NodeCategory(node.node);
  DisplacementExpr res;
  if (cat == "displacement") {
    if (IsNonZeroInput(node, "displacement")) {
      Expr scale = CompileInputExpr(node, "scale", Expr::one());
      if (!scale.is_zero()) {
        bool is_vec = InputType(node.node, "displacement") == "vector3";
        Expr disp = CompileInputExpr(node, "displacement") * scale;
        if (!disp.is_zero()) res = {is_vec ? disp : disp.x(), is_vec};
      }
    }
  } else if (cat == "mix") {
    auto disp_in = [&](const std::string& name) {
      return CompileDisplacement(
          ResolveConnected(node, name, "displacementshader"));
    };
    Expr m = CompileInputExpr(node, "mix");
    if (m.is_zero()) {
      res = disp_in("bg");
    } else if (m.is_one()) {
      res = disp_in("fg");
    } else {
      DisplacementExpr fg = disp_in("fg"), bg = disp_in("bg");
      if (!fg.expr.is_zero() || !bg.expr.is_zero()) {
        bool is_vec = fg.is_vector || bg.is_vector;
        auto promote = [&](const DisplacementExpr& d) {
          return (!is_vec || d.is_vector)
                     ? d.expr
                     : rgb(Expr::zero(), Expr::zero(), d.expr);
        };
        res = {mix(promote(fg), promote(bg), m), is_vec};
      }
    }
  } else {
    WarnUnsupported("Unsupported MaterialX displacement shader '" + cat +
                    "'; ignoring displacement.");
  }
  res.expr = FoldAndSpill(res.expr);
  disp_expr_[key] = res;
  return res;
}

bool MtlxExprCompiler::ProgramUsesUv(const Expr& expr) const {
  auto inst_uses_uv = [&](auto& self, const mtlx::Inst& inst) -> bool {
    if (inst.op == mtlx::Op::GeomUV || inst.op == mtlx::Op::SampleInput) {
      return true;
    }
    if (inst.op != mtlx::Op::HeightToNormal) return false;
    const mtlx::SubProgram& sub = prog_.sub_programs[inst.imm];
    return !sub.has_texcoord ||
           AnyLiveInst(
               absl::MakeConstSpan(prog_.sub_code)
                   .subspan(sub.code_begin, sub.code_end - sub.code_begin),
               {}, sub.tmp_count,
               [&](const mtlx::Inst& i) { return self(self, i); });
  };
  return AnyLiveInst(
      prog_.code, expr.code, prog_.tmp_channels.size(),
      [&](const mtlx::Inst& i) { return inst_uses_uv(inst_uses_uv, i); });
}

void MtlxExprCompiler::CompactLiveOpaques() {
  if (prog_.opaques.empty()) return;
  std::vector<uint16_t> remap(prog_.opaques.size(), mtlx::kMaxImm);
  for (const auto* code : {&prog_.code, &prog_.sub_code}) {
    for (mtlx::Inst in : *code) {
      if (in.op == mtlx::Op::Opaque) remap[in.imm] = 0;
    }
  }
  size_t n = 0;
  for (size_t i = 0; i < prog_.opaques.size(); ++i) {
    if (remap[i] == mtlx::kMaxImm) continue;
    remap[i] = static_cast<uint16_t>(n);
    prog_.opaques[n++] = prog_.opaques[i];
  }
  if (n == prog_.opaques.size()) return;
  prog_.opaques.resize(n);
  for (auto* code : {&prog_.code, &prog_.sub_code}) {
    for (mtlx::Inst& in : *code) {
      if (in.op == mtlx::Op::Opaque) in.imm = remap[in.imm];
    }
  }
}

ref<Object> MtlxExprCompiler::CreatePlugin(std::string_view name,
                                           ObjectType type,
                                           std::string_view variant) {
  CompactLiveOpaques();
  Properties props(name);
  for (size_t i = 0; i < inputs_.size(); ++i) {
    props.set("in" + std::to_string(i), std::move(inputs_[i]));
  }
  props.set_any("program", std::move(prog_));
  return PluginManager::instance()->create_object(props, variant, type);
}

ref<Object> MtlxExprCompiler::CreateExprTexture(const Expr& expr,
                                                bool unbounded,
                                                std::string_view variant) {
  mtlx::RegisterMaterialXExprTexture();
  prog_.code.insert(prog_.code.end(), expr.code.begin(), expr.code.end());
  prog_.stack_size = std::max(prog_.stack_size, expr.max_depth);
  prog_.result_uniform = expr.uniform();
  prog_.unbounded = unbounded;
  return CreatePlugin("materialx_expr", ObjectType::Texture, variant);
}

ref<Object> MtlxExprCompiler::CreateBsdf(std::string_view variant) {
  mtlx::RegisterMaterialXBsdf();
  return CreatePlugin("materialx_bsdf", ObjectType::BSDF, variant);
}

}  // namespace

CompiledSurfaceShader CompileSurfaceShader(
    const mx::NodePtr& shader_node, std::string_view output_name,
    std::string_view variant,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture) {
  absl::flat_hash_set<std::string> warned;
  CompiledSurfaceShader out;
  {
    MtlxExprCompiler compiler(shader_node->getDocument(), lookup_texture,
                              warned);
    compiler.CompileBsdf(shader_node, output_name);
    out.bsdf = compiler.CreateBsdf(variant);
    out.mesh_attributes.insert(compiler.attr_requests().begin(),
                               compiler.attr_requests().end());
  }
  {
    MtlxExprCompiler compiler(shader_node->getDocument(), lookup_texture,
                              warned);
    Expr radiance = compiler.CompileEdf(shader_node, output_name);
    if (!radiance.is_zero()) {
      out.radiance_texture =
          compiler.CreateExprTexture(radiance, /*unbounded=*/true, variant);
      out.mesh_attributes.insert(compiler.attr_requests().begin(),
                                 compiler.attr_requests().end());
    }
  }
  return out;
}

void DiscoverSurfaceShaderTextures(
    const mx::NodePtr& shader_node, std::string_view output_name,
    const std::function<void(const TextureKey&)>& callback) {
  MtlxExprCompiler compiler(shader_node->getDocument(), callback);
  compiler.CompileBsdf(shader_node, output_name);
  compiler.CompileEdf(shader_node, output_name);
}

bool HasDisplacementShader(const mx::NodePtr& shader_node,
                           std::string_view output_name) {
  MtlxGraphResolver resolver;
  ScopedNode n = resolver.UnwrapShaderNode(
      {shader_node, nullptr}, std::string(output_name), "displacementshader");
  return n && n.node->getType() == mx::DISPLACEMENT_SHADER_TYPE_STRING;
}

MaterialDisplacement CompileDisplacementShader(
    const mx::NodePtr& shader_node, std::string_view output_name,
    std::string_view variant,
    const std::function<const CachedTexture*(const TextureKey&)>&
        lookup_texture) {
  absl::flat_hash_set<std::string> warned;
  MtlxExprCompiler compiler(shader_node->getDocument(), lookup_texture, warned);
  MtlxExprCompiler::DisplacementExpr disp =
      compiler.CompileDisplacement(shader_node, output_name);
  if (disp.expr.is_zero()) return {};
  bool requires_uv = compiler.ProgramUsesUv(disp.expr);
  return {compiler.CreateExprTexture(disp.expr, /*unbounded=*/true, variant),
          requires_uv, disp.is_vector,
          MeshAttributeRequests(compiler.attr_requests().begin(),
                                compiler.attr_requests().end())};
}

void DiscoverDisplacementShaderTextures(
    const mx::NodePtr& shader_node, std::string_view output_name,
    const std::function<void(const TextureKey&)>& callback) {
  MtlxExprCompiler compiler(shader_node->getDocument(), callback);
  compiler.CompileDisplacement(shader_node, output_name);
}

PXR_NAMESPACE_CLOSE_SCOPE
