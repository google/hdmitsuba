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
#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/optimizer.h"
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
using mtlx::kInvalidValue;
using mtlx::ValueId;

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

// Resolves MaterialX ports, inputs, and lazy `NodeGraph` call frames without
// allocating an SSA program.
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

// Lowers a MaterialX DAG (lazily traversing `NodeGraph` implementations without
// flattening them into the document) directly into flat SSA IR
// (`MtlxValueProgram` / `MtlxBsdfProgram`) and Mitsuba Texture inputs.
class MtlxExprCompiler : private MtlxGraphResolver {
 public:
  using LookupTextureFn =
      std::function<const CachedTexture*(const TextureKey&)>;
  using DiscoverTextureFn = std::function<void(const TextureKey&)>;

  struct DisplacementVal {
    ValueId val = kInvalidValue;
    bool is_vector = false;
  };

  MtlxExprCompiler(mx::DocumentPtr doc, const LookupTextureFn& lookup_texture,
                   absl::flat_hash_set<std::string>& warned)
      : doc_(std::move(doc)),
        lookup_texture_(&lookup_texture),
        warned_(&warned),
        main_builder_(val_prog_),
        b_(&main_builder_) {}

  MtlxExprCompiler(mx::DocumentPtr doc,
                   const DiscoverTextureFn& discover_texture)
      : doc_(std::move(doc)),
        discover_texture_(&discover_texture),
        main_builder_(val_prog_),
        b_(&main_builder_) {}

  // Compiles a MaterialX EDF or surface emission DAG (`surface`,
  // `surface_unlit`, `uniform_edf`, `generalized_schlick_edf`, `multiply`,
  // `add`, `mix`) into a radiance `ValueId`, or `Zero()` if none.
  ValueId CompileEdf(const mx::NodePtr& edf_node,
                     std::string_view output_name) {
    return CompileEdf(
        UnwrapShaderNode({edf_node, nullptr}, std::string(output_name)));
  }

  // Compiles a MaterialX displacement shader DAG (`displacement`, `mix`,
  // `dot`, `surfacematerial`) into a scalar or 3D tangent-space `ValueId`.
  DisplacementVal CompileDisplacement(const mx::NodePtr& disp_node,
                                      std::string_view output_name) {
    return CompileDisplacement(UnwrapShaderNode(
        {disp_node, nullptr}, std::string(output_name), "displacementshader"));
  }

  // Compiles a MaterialX surface shader or BSDF closure DAG into `closures_`,
  // `opacity_root_`, and `thin_walled_`.
  void CompileBsdf(const mx::NodePtr& shader_node,
                   std::string_view output_name);

  bool IsZero(ValueId id) const { return b_->IsZero(id); }

  void FinalizeExpr(ValueId root);
  bool ProgramUsesUv() const;

  const std::vector<std::pair<std::string, int>>& attr_requests() const {
    return attr_requests_;
  }

  ref<Object> CreateExprTexture(std::string_view variant);
  ref<Object> CreateBsdf(std::string_view variant);

 private:
  using NodeOutKey = std::tuple<const Frame*, const mx::Node*, std::string>;

  // A 3x3 or 4x4 matrix as the `ValueId`s of its `n` rows.
  struct MatVal {
    ValueId rows[4] = {kInvalidValue, kInvalidValue, kInvalidValue,
                       kInvalidValue};
    int n = 3;
  };

  // Scope-independent opaque parameter handles for an authored input tuple.
  struct AuthoredOpaque {
    uint8_t n = 0;
    bool all_zero = false;
    bool all_one = false;
    uint16_t idx[4] = {};
  };

  struct AuthoredMatrix {
    int n = 3;
    AuthoredOpaque rows[4];
  };

  struct PendingRegion {
    std::vector<mtlx::Node> nodes;
    std::vector<ValueId> extra_args;
    ValueId root = kInvalidValue;
    uint16_t has_texcoord = 0;
  };

  bool IsAuthoredInput(const mx::ValueElementPtr& elem) const {
    mx::ElementPtr parent = elem->getParent();
    return parent && parent->isA<mx::Node>() && parent->getParent() == doc_;
  }

  ValueId DefaultTexcoord(uint8_t ch = 2) {
    return b_->Emit(mtlx::Op::GeomUV, {}, 0, ch);
  }
  ValueId PWorld() { return b_->Emit(mtlx::Op::GeomPosWorld); }
  ValueId NWorld() { return b_->Emit(mtlx::Op::GeomNormalWorld); }
  ValueId TWorld() { return b_->Emit(mtlx::Op::GeomTangentWorld); }
  ValueId BWorld() { return b_->Emit(mtlx::Op::GeomBitangentWorld); }

  MatVal IdentityMatrix(int n) {
    MatVal m;
    m.n = n;
    ValueId z = b_->Zero(), o = b_->One();
    for (int i = 0; i < 3; ++i) {
      ValueId row = b_->Rgb(i == 0 ? o : z, i == 1 ? o : z, i == 2 ? o : z);
      m.rows[i] = n == 4 ? b_->Vec4(row, z) : row;
    }
    m.rows[3] = b_->Vec4(z, o);
    return m;
  }

  ValueId PackTuple(const ValueId* e, int n) {
    if (n == 1) return e[0];
    if (n == 2) return b_->Vec2(e[0], e[1]);
    ValueId c3 = b_->Rgb(e[0], e[1], e[2]);
    return n == 3 ? c3 : b_->Vec4(c3, e[3]);
  }
  AuthoredOpaque RecordAuthoredTuple(std::initializer_list<float> vals);
  ValueId MaterializeAuthored(const AuthoredOpaque& a);
  ValueId EmitConstTuple(std::initializer_list<float> vals);
  ValueId Clip01(ValueId v) {
    return b_->Emit(mtlx::Op::Clip, {v, b_->Zero(), b_->One()});
  }

  template <typename FgFn, typename BgFn>
  ValueId CompileMixVal(ValueId m, FgFn&& fg_fn, BgFn&& bg_fn) {
    if (b_->IsZero(m)) return bg_fn();
    if (b_->IsOne(m)) return fg_fn();
    ValueId fg = fg_fn(), bg = bg_fn();
    return b_->Mix(fg, bg, m);
  }

  ValueId CompileLinearRemap(const ScopedNode& node, ValueId ilo, ValueId ihi,
                             ValueId olo, ValueId ohi);

  ValueId CompileEdf(const ScopedNode& node);
  DisplacementVal CompileDisplacement(const ScopedNode& node);
  ValueId CompileOpacity(const ScopedNode& node);
  int32_t CompileBsdfClosure(const ScopedNode& node);
  bool IsClosureOpaque(int32_t idx) const;
  int32_t EmitOpaqueBlocker();
  ValueId FrameValue(const ScopedNode& node, const std::string& name,
                     ValueId def);
  int32_t CompileDiffuseClosure(const ScopedNode& node, const std::string& cat);
  int32_t CompileMicrofacetClosure(const ScopedNode& node,
                                   const std::string& cat);
  int32_t CompileChiangHairClosure(const ScopedNode& node);
  void CompileVdfWarning(const ScopedNode& node, const std::string& cat);
  void MarkTwoSidedDielectrics(size_t start);
  int32_t EmitClosure(mtlx::ClosureNode node);

  void AppendPendingRegions();
  void ApplyCompaction(const mtlx::OptimizeResult& res);
  template <typename ProgT>
  ref<Object> CreatePlugin(std::string_view name, ObjectType type,
                           std::string_view variant, ProgT&& prog);

  TfToken ResolveSourceColorSpace(std::string_view color_space);
  GfVec3f ToLinearRec709(GfVec3f rgb_val, std::string_view color_space);
  ValueId CompileGeomIntrinsic(std::string_view kind, std::string_view space,
                               std::string_view type, bool non_zero_index);
  ValueId CompileGeomProp(const std::string& prop_name, std::string_view type,
                          const std::function<ValueId()>& default_fn = {});
  ValueId CompileResolvedInput(const ResolvedInput& r, ValueId default_val);
  ValueId CompileInputExpr(const ScopedNode& node,
                           const std::string& input_name,
                           ValueId default_val = kInvalidValue);
  ValueId CompileNodeOutputExpr(const ScopedNode& node,
                                const std::string& output_name);
  std::optional<ValueId> CompileChannelNode(const ScopedNode& node,
                                            const std::string& cat,
                                            const std::string& type);
  std::optional<ValueId> CompileMathNode(const ScopedNode& node,
                                         const std::string& cat);
  std::optional<ValueId> CompileLogicNode(const ScopedNode& node,
                                          const std::string& cat,
                                          const std::string& type);
  std::optional<ValueId> CompileVectorNode(const ScopedNode& node,
                                           const std::string& cat);
  std::optional<ValueId> CompileGeomNode(const ScopedNode& node,
                                         const std::string& cat,
                                         const std::string& type);
  std::optional<ValueId> CompileMatrixValueNode(const ScopedNode& node,
                                                const std::string& cat,
                                                const std::string& type);
  std::optional<ValueId> CompileCompositingNode(const ScopedNode& node,
                                                const std::string& cat);
  std::optional<ValueId> CompileColorPbrNode(const ScopedNode& node,
                                             const std::string& cat,
                                             const std::string& type,
                                             const std::string& output_name);
  std::optional<ValueId> CompileNoiseNode(const ScopedNode& node,
                                          const std::string& cat,
                                          const std::string& type);
  std::optional<int> ConstIntInput(const ScopedNode& node,
                                   const std::string& name);
  ValueId CompileIfCondition(const ScopedNode& node, std::string_view cat);
  MatVal CompileResolvedMatrix(const ResolvedInput& r, int n);
  MatVal CompileMatrixInput(const ScopedNode& node,
                            const std::string& input_name);
  MatVal CompileMatrixNode(const ScopedNode& node,
                           const std::string& output_name);
  MatVal MatrixOp(mtlx::Op op, const MatVal& a, const MatVal* b = nullptr);
  ValueId MeshAttr(std::string_view name, int dim, ValueId def,
                   uint8_t out_ch = 0);
  ValueId CompileHeightToNormal(const ScopedNode& node);
  ValueId SampleImage(const ScopedNode& node);
  ValueId CompileValueElement(const mx::ValueElementPtr& elem,
                              ValueId default_val, bool is_authored);

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
  absl::flat_hash_set<std::string>* warned_ = nullptr;

  mtlx::MtlxValueProgram val_prog_;
  mtlx::IrBuilder main_builder_;
  mtlx::IrBuilder* b_ = nullptr;
  std::vector<PendingRegion> pending_regions_;

  std::vector<mtlx::ClosureNode> closures_;
  std::vector<bool> closure_opaque_;
  int32_t opaque_blocker_ = -1;
  ValueId opacity_root_ = kInvalidValue;
  bool thin_walled_ = false;

  std::vector<std::pair<std::string, int>> attr_requests_;
  absl::flat_hash_map<NodeOutKey, ValueId> node_output_val_;
  absl::flat_hash_map<NodeOutKey, MatVal> node_matrix_val_;
  absl::flat_hash_map<NodeKey, ValueId> edf_val_;
  absl::flat_hash_map<NodeKey, DisplacementVal> disp_val_;
  absl::flat_hash_map<NodeKey, ValueId> opacity_val_;
  absl::flat_hash_map<NodeKey, int32_t> bsdf_closure_;
  absl::flat_hash_map<const mx::ValueElement*, AuthoredOpaque>
      authored_opaque_val_;
  absl::flat_hash_map<const mx::ValueElement*, AuthoredMatrix>
      authored_opaque_matrix_;
  std::vector<ref<Object>> inputs_;
  uint16_t discover_input_idx_ = 0;
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

// Returns true if input `name` of `s` may be non-zero, without emitting SSA
// nodes, opaque parameters, or texture loads into the program.
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

MtlxExprCompiler::AuthoredOpaque MtlxExprCompiler::RecordAuthoredTuple(
    std::initializer_list<float> vals) {
  AuthoredOpaque a;
  a.n = static_cast<uint8_t>(vals.size());
  a.all_zero =
      std::all_of(vals.begin(), vals.end(), [](float x) { return x == 0.0f; });
  a.all_one =
      std::all_of(vals.begin(), vals.end(), [](float x) { return x == 1.0f; });
  if (a.all_zero || a.all_one) return a;
  int i = 0;
  for (float v : vals) a.idx[i++] = b_->AddOpaqueValue(v);
  return a;
}

ValueId MtlxExprCompiler::MaterializeAuthored(const AuthoredOpaque& a) {
  if (a.all_zero) return b_->Zero();
  if (a.all_one) return b_->One();
  ValueId e[4];
  for (int i = 0; i < a.n; ++i) e[i] = b_->OpaqueRef(a.idx[i]);
  return PackTuple(e, a.n);
}

ValueId MtlxExprCompiler::EmitConstTuple(std::initializer_list<float> vals) {
  ValueId e[4];
  int n = 0;
  for (float v : vals) e[n++] = b_->Const(v);
  return PackTuple(e, n);
}

ValueId MtlxExprCompiler::CompileLinearRemap(const ScopedNode& node,
                                             ValueId ilo, ValueId ihi,
                                             ValueId olo, ValueId ohi) {
  ValueId scale = b_->Div(b_->Sub(ohi, olo), b_->Sub(ihi, ilo));
  return b_->Fma(b_->Sub(CompileInputExpr(node, "in"), ilo), scale, olo);
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
  // `opacity = (1, 1, 1)` under `UsdColorSpaceAPI`) so `EmitConstTuple` /
  // `RecordAuthoredTuple` folds opaque white to `One()`, while preserving
  // spaces where `(1, 1, 1)` does not map to Rec.709 white.
  if (GfIsClose(rgb_val, GfVec3f(1.0f), 1e-5f)) rgb_val = GfVec3f(1.0f);
  return rgb_val;
}

ValueId MtlxExprCompiler::MeshAttr(std::string_view name, int dim, ValueId def,
                                   uint8_t out_ch) {
  std::pair<std::string, int> req{std::string(name), dim};
  auto it = std::find(attr_requests_.begin(), attr_requests_.end(), req);
  if (it == attr_requests_.end()) {
    if (attr_requests_.size() >= mtlx::kMaxImm) {
      throw std::runtime_error("materialx_expr: too many attributes");
    }
    val_prog_.attr_names.push_back(MeshAttributeName(name, dim));
    it = attr_requests_.insert(attr_requests_.end(), std::move(req));
  }
  uint8_t read_ch = static_cast<uint8_t>(dim);
  uint8_t ch = out_ch ? out_ch : std::max<uint8_t>(b_->channels(def), read_ch);
  return b_->Emit(mtlx::Op::Attr, {def},
                  static_cast<uint16_t>(it - attr_requests_.begin()), ch,
                  read_ch);
}

ValueId MtlxExprCompiler::CompileValueElement(const mx::ValueElementPtr& elem,
                                              ValueId default_val,
                                              bool is_authored) {
  const std::string& type = elem->getType();
  if (type == "boolean") {
    const std::string& s = elem->getValueString();
    return (s == "true" || s == "1") ? b_->One() : b_->Zero();
  }
  mx::ValuePtr val = elem->getValue();
  if (!val) return default_val;
  // Integers and booleans select code paths (e.g. switch branches) and are
  // baked as constants.
  if (type == "integer") return b_->Const(val->asA<int>());

  auto emit = [&](std::initializer_list<float> vals) -> ValueId {
    if (!is_authored) return EmitConstTuple(vals);
    auto [it, inserted] = authored_opaque_val_.try_emplace(elem.get());
    if (inserted) it->second = RecordAuthoredTuple(vals);
    return MaterializeAuthored(it->second);
  };

  if (type == "float") return emit({val->asA<float>()});
  if (type == "color3") {
    mx::Color3 v = val->asA<mx::Color3>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
    return emit({c[0], c[1], c[2]});
  }
  if (type == "color4") {
    mx::Color4 v = val->asA<mx::Color4>();
    GfVec3f c = ToLinearRec709({v[0], v[1], v[2]}, elem->getActiveColorSpace());
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

ValueId MtlxExprCompiler::CompileResolvedInput(const ResolvedInput& r,
                                               ValueId default_val) {
  if (r.kind == ResolvedInput::Connected) {
    NodeOutKey key{r.upstream.frame, r.upstream.node.get(), r.output_name};
    auto it = node_output_val_.find(key);
    if (it == node_output_val_.end()) {
      ValueId v = CompileNodeOutputExpr(r.upstream, r.output_name);
      it = node_output_val_.emplace(std::move(key), v).first;
    }
    return it->second;
  }
  if (r.kind == ResolvedInput::GeomProp) {
    return CompileGeomProp(r.elem->asA<mx::Input>()->getDefaultGeomPropString(),
                           r.elem->getType());
  }
  if (r.kind != ResolvedInput::Value) return default_val;
  if (!IsAuthoredInput(r.elem)) {
    return CompileValueElement(r.elem, default_val, /*is_authored=*/false);
  }
  if (auto it = authored_opaque_val_.find(r.elem.get());
      it != authored_opaque_val_.end()) {
    return MaterializeAuthored(it->second);
  }
  return CompileValueElement(r.elem, default_val, /*is_authored=*/true);
}

ValueId MtlxExprCompiler::CompileInputExpr(const ScopedNode& s,
                                           const std::string& input_name,
                                           ValueId default_val) {
  if (default_val == kInvalidValue) default_val = b_->Zero();
  return CompileResolvedInput(ResolveInput(s, input_name), default_val);
}

std::optional<int> MtlxExprCompiler::ConstIntInput(const ScopedNode& node,
                                                   const std::string& name) {
  ResolvedInput r = ResolveInput(node, name);
  if (r.kind == ResolvedInput::None) return std::nullopt;
  ValueId v = CompileResolvedInput(r, b_->Zero());
  double d = 0.0;
  if (b_->IsConstScalar(v, &d)) {
    return static_cast<int>(std::clamp<double>(
        d, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
  }
  return std::nullopt;
}

ValueId MtlxExprCompiler::CompileIfCondition(const ScopedNode& node,
                                             std::string_view cat) {
  ValueId v1 = CompileInputExpr(node, "value1");
  ValueId v2 = CompileInputExpr(node, "value2");
  if (cat == "ifgreater") return b_->Emit(mtlx::Op::Lt, {v2, v1});
  if (cat == "ifgreatereq") return b_->Emit(mtlx::Op::Le, {v2, v1});
  return b_->Emit(mtlx::Op::Eq, {v1, v2});
}

MtlxExprCompiler::MatVal MtlxExprCompiler::MatrixOp(mtlx::Op op,
                                                    const MatVal& a,
                                                    const MatVal* b) {
  absl::InlinedVector<ValueId, 8> args;
  for (int i = 0; i < a.n; ++i) args.push_back(a.rows[i]);
  if (b) {
    for (int i = 0; i < a.n; ++i) args.push_back(b->rows[i]);
  }
  ValueId mat_id = b_->Emit(op, args, 0, static_cast<uint8_t>(a.n));
  MatVal out;
  out.n = a.n;
  for (int i = 0; i < a.n; ++i) {
    out.rows[i] = b_->Emit(mtlx::Op::MatRow, {mat_id}, static_cast<uint16_t>(i),
                           static_cast<uint8_t>(a.n));
  }
  return out;
}

MtlxExprCompiler::MatVal MtlxExprCompiler::CompileResolvedMatrix(
    const ResolvedInput& r, int n) {
  if (r.kind == ResolvedInput::Connected) {
    return CompileMatrixNode(r.upstream, r.output_name);
  }
  mx::ValuePtr val =
      r.kind == ResolvedInput::Value ? r.elem->getValue() : nullptr;
  if (!val) return IdentityMatrix(n);
  const std::string& t = r.elem->getType();
  if (t != "matrix33" && t != "matrix44") return IdentityMatrix(n);
  bool is_authored = IsAuthoredInput(r.elem);
  if (is_authored) {
    auto [it, inserted] = authored_opaque_matrix_.try_emplace(r.elem.get());
    if (inserted) {
      if (t == "matrix44") {
        it->second.n = 4;
        mx::Matrix44 m = val->asA<mx::Matrix44>();
        for (int i = 0; i < 4; ++i) {
          it->second.rows[i] =
              RecordAuthoredTuple({m[i][0], m[i][1], m[i][2], m[i][3]});
        }
      } else {
        it->second.n = 3;
        mx::Matrix33 m = val->asA<mx::Matrix33>();
        for (int i = 0; i < 3; ++i) {
          it->second.rows[i] = RecordAuthoredTuple({m[i][0], m[i][1], m[i][2]});
        }
      }
    }
    MatVal out;
    out.n = it->second.n;
    for (int i = 0; i < out.n; ++i) {
      out.rows[i] = MaterializeAuthored(it->second.rows[i]);
    }
    return out;
  }
  MatVal out;
  if (t == "matrix44") {
    out.n = 4;
    mx::Matrix44 m = val->asA<mx::Matrix44>();
    for (int i = 0; i < 4; ++i) {
      out.rows[i] = EmitConstTuple({m[i][0], m[i][1], m[i][2], m[i][3]});
    }
  } else {
    out.n = 3;
    mx::Matrix33 m = val->asA<mx::Matrix33>();
    for (int i = 0; i < 3; ++i) {
      out.rows[i] = EmitConstTuple({m[i][0], m[i][1], m[i][2]});
    }
  }
  return out;
}

MtlxExprCompiler::MatVal MtlxExprCompiler::CompileMatrixInput(
    const ScopedNode& node, const std::string& input_name) {
  return CompileResolvedMatrix(
      ResolveInput(node, input_name),
      InputType(node.node, input_name) == "matrix44" ? 4 : 3);
}

MtlxExprCompiler::MatVal MtlxExprCompiler::CompileMatrixNode(
    const ScopedNode& node, const std::string& output_name) {
  NodeOutKey key{node.frame, node.node.get(), output_name};
  if (auto it = node_matrix_val_.find(key); it != node_matrix_val_.end()) {
    return it->second;
  }
  int n = node.node->getType() == "matrix44" ? 4 : 3;
  const std::string& cat = NodeCategory(node.node);
  MatVal res = IdentityMatrix(n);
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
      ValueId row = CompileInputExpr(node, "in" + std::to_string(i + 1));
      res.rows[i] =
          extend ? b_->Vec4(row, i == 3 ? b_->One() : b_->Zero()) : row;
    }
  } else if (cat == "transpose") {
    res = MatrixOp(mtlx::Op::MatTranspose, CompileMatrixInput(node, "in"));
  } else if (cat == "invertmatrix") {
    res = MatrixOp(mtlx::Op::MatInverse, CompileMatrixInput(node, "in"));
  } else if (cat == "add" || cat == "subtract") {
    MatVal m1 = CompileMatrixInput(node, "in1");
    bool is_add = cat == "add";
    if (InputType(node.node, "in2") == "float") {
      ValueId s = CompileInputExpr(node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = is_add ? b_->Add(m1.rows[i], s) : b_->Sub(m1.rows[i], s);
      }
    } else {
      MatVal m2 = CompileMatrixInput(node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = is_add ? b_->Add(m1.rows[i], m2.rows[i])
                             : b_->Sub(m1.rows[i], m2.rows[i]);
      }
    }
  } else if (cat == "multiply" || cat == "divide") {
    MatVal m1 = CompileMatrixInput(node, "in1");
    MatVal m2 = CompileMatrixInput(node, "in2");
    if (cat == "divide") m2 = MatrixOp(mtlx::Op::MatInverse, m2);
    res = MatrixOp(mtlx::Op::MatMul, m1, &m2);
  } else if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    ValueId cond = CompileIfCondition(node, cat);
    if (b_->IsOne(cond) || b_->IsZero(cond)) {
      res = CompileMatrixInput(node, b_->IsOne(cond) ? "in1" : "in2");
    } else {
      MatVal m1 = CompileMatrixInput(node, "in1"),
             m2 = CompileMatrixInput(node, "in2");
      for (int i = 0; i < n; ++i) {
        res.rows[i] = b_->Select(cond, m1.rows[i], m2.rows[i]);
      }
    }
  } else {
    WarnUnsupported("Unsupported MaterialX matrix node category '" + cat +
                    "'; using identity.");
  }

  node_matrix_val_[std::move(key)] = res;
  return res;
}

ValueId MtlxExprCompiler::SampleImage(const ScopedNode& node) {
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
    uint8_t ch = TypeChannels(type);
    return b_->Emit(mtlx::Op::SampleInput, {}, discover_input_idx_++, ch,
                    ch == 1 ? 1 : 3);
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

  ValueId def_tc = DefaultTexcoord();
  ValueId tc = CompileInputExpr(node, "texcoord", def_tc);
  bool default_uv = !mixed_wrap && tc == def_tc;
  ValueId u = kInvalidValue, v = kInvalidValue;
  if (!default_uv) {
    auto wrap_axis = [&](ValueId c, const std::string& mode) {
      if (!mixed_wrap || mode == "clamp" || is_oob_mode(mode)) return c;
      if (mode == "mirror") {
        ValueId f = b_->Emit(mtlx::Op::Fract, {b_->Mul(c, b_->Const(0.5))});
        return b_->Sub(
            b_->One(),
            b_->Emit(mtlx::Op::Abs,
                     {b_->Sub(b_->Mul(f, b_->Const(2.0)), b_->One())}));
      }
      return b_->Emit(mtlx::Op::Fract, {c});
    };
    u = wrap_axis(b_->Chan(tc, 0), u_mode);
    v = b_->Sub(b_->One(), wrap_axis(b_->Chan(tc, 1), v_mode));
  }

  auto sample_tex = [&](const CachedTexture& cached,
                        uint8_t out_ch = 0) -> ValueId {
    auto it = std::find(inputs_.begin(), inputs_.end(), cached.texture);
    if (it == inputs_.end()) {
      if (inputs_.size() >= mtlx::kMaxImm) {
        throw std::runtime_error("materialx_expr: too many texture inputs");
      }
      it = inputs_.insert(inputs_.end(), cached.texture);
    }
    uint16_t idx = static_cast<uint16_t>(it - inputs_.begin());
    uint8_t read_ch = cached.channel_count == 1 ? 1 : 3;
    uint8_t ch = out_ch ? out_ch : read_ch;
    if (default_uv) {
      return b_->Emit(mtlx::Op::SampleInput, {}, idx, ch, read_ch);
    }
    return b_->Emit(mtlx::Op::SampleInputUV, {u, v}, idx, ch, read_ch);
  };

  uint8_t color_ch =
      (type == "vector2" && cached_color->channel_count != 1) ? 2 : 0;
  ValueId result = sample_tex(*cached_color, color_ch);
  if (has_alpha) {
    result =
        b_->Vec4(result, cached_alpha ? sample_tex(*cached_alpha) : b_->One());
  }
  if (has_oob) {
    auto oob_axis = [&](ValueId c) {
      return b_->Add(b_->Emit(mtlx::Op::Lt, {c, b_->Zero()}),
                     b_->Emit(mtlx::Op::Lt, {b_->One(), c}));
    };
    auto oob_val = [&](const std::string& mode) {
      return mode == "black" ? b_->Zero() : default_val();
    };
    if (u_mode == v_mode) {
      result = b_->Select(
          b_->Add(oob_axis(b_->Chan(tc, 0)), oob_axis(b_->Chan(tc, 1))),
          oob_val(u_mode), result);
    } else {
      if (is_oob_mode(u_mode)) {
        result = b_->Select(oob_axis(b_->Chan(tc, 0)), oob_val(u_mode), result);
      }
      if (is_oob_mode(v_mode)) {
        result = b_->Select(oob_axis(b_->Chan(tc, 1)), oob_val(v_mode), result);
      }
    }
  }
  return result;
}

std::optional<ValueId> MtlxExprCompiler::CompileChannelNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "constant") return in("value");
  if (cat == "dot" || cat == "convert") return in("in");
  if (cat == "extract") {
    int idx = ConstIntInput(node, "index").value_or(0);
    std::string_view in_type = InputType(node.node, "in");
    if (in_type == "matrix33" || in_type == "matrix44") {
      MatVal mat = CompileMatrixInput(node, "in");
      return mat.rows[std::clamp(idx, 0, mat.n - 1)];
    }
    return b_->Chan(in("in"), static_cast<uint16_t>(std::clamp(
                                  idx, 0, TypeChannels(in_type) - 1)));
  }
  if (cat == "combine2") {
    if (IsFourChannelType(type)) {
      // A 3-channel value and a float, or two `vector2` values.
      if (InputType(node.node, "in1") != "vector2") {
        ValueId i1 = in("in1"), i2 = in("in2");
        return b_->Vec4(i1, i2);
      }
      ValueId v1 = in("in1"), v2 = in("in2");
      return b_->Vec4(
          b_->Rgb(b_->Chan(v1, 0), b_->Chan(v1, 1), b_->Chan(v2, 0)),
          b_->Chan(v2, 1));
    }
    ValueId i1 = in("in1"), i2 = in("in2");
    return b_->Vec2(i1, i2);
  }
  if (cat == "combine3") {
    ValueId i1 = in("in1"), i2 = in("in2"), i3 = in("in3");
    return b_->Rgb(i1, i2, i3);
  }
  if (cat == "combine4") {
    ValueId i1 = in("in1"), i2 = in("in2"), i3 = in("in3"), i4 = in("in4");
    return b_->Vec4(b_->Rgb(i1, i2, i3), i4);
  }
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileMathNode(
    const ScopedNode& node, const std::string& cat) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "multiply") {
    ValueId i1, i2;
    if (GetValueInput(node, "in2") && !GetValueInput(node, "in1")) {
      i2 = in("in2", b_->One());
      if (b_->IsZero(i2)) return b_->Zero();
      i1 = in("in1");
    } else {
      i1 = in("in1");
      if (b_->IsZero(i1)) return b_->Zero();
      i2 = in("in2", b_->One());
    }
    return b_->Mul(i1, i2);
  }
  if (cat == "add") {
    ValueId a = in("in1"), b = in("in2");
    return b_->Add(a, b);
  }
  if (cat == "subtract") {
    ValueId a = in("in1"), b = in("in2");
    return b_->Sub(a, b);
  }
  if (cat == "divide") {
    ValueId i1 = in("in1");
    return b_->IsZero(i1) ? b_->Zero() : b_->Div(i1, in("in2", b_->One()));
  }
  if (cat == "mix") {
    return CompileMixVal(
        in("mix"), [&] { return in("fg"); }, [&] { return in("bg"); });
  }
  if (cat == "clamp") {
    ValueId x = in("in"), lo = in("low"), hi = in("high", b_->One());
    return b_->Emit(mtlx::Op::Clip, {x, lo, hi});
  }
  if (cat == "invert") {
    ValueId amt = in("amount", b_->One()), x = in("in");
    return b_->Sub(amt, x);
  }
  if (cat == "sign") {
    ValueId x = in("in");
    return b_->Sub(b_->Emit(mtlx::Op::Lt, {b_->Zero(), x}),
                   b_->Emit(mtlx::Op::Lt, {x, b_->Zero()}));
  }
  if (cat == "modulo") {
    ValueId x = in("in1"), y = in("in2", b_->One());
    if (b_->IsOne(y)) return b_->Emit(mtlx::Op::Fract, {x});
    return b_->Sub(x, b_->Mul(y, b_->Emit(mtlx::Op::Floor, {b_->Div(x, y)})));
  }
  if (cat == "remap") {
    ValueId ilo = in("inlow"), olo = in("outlow");
    return CompileLinearRemap(node, ilo, in("inhigh", b_->One()), olo,
                              in("outhigh", b_->One()));
  }
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileLogicNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "ifgreater" || cat == "ifgreatereq" || cat == "ifequal") {
    ValueId cond = CompileIfCondition(node, cat);
    if (type == "boolean") return cond;
    if (b_->IsOne(cond)) return in("in1");
    if (b_->IsZero(cond)) return in("in2");
    ValueId i1 = in("in1"), i2 = in("in2");
    return b_->Select(cond, i1, i2);
  }
  if (cat == "and" || cat == "or") {
    auto to_bool = [&](ValueId e) {
      return (b_->IsZero(e) || b_->IsOne(e))
                 ? e
                 : b_->Select(e, b_->One(), b_->Zero());
    };
    ValueId b1 = to_bool(in("in1")), b2 = to_bool(in("in2"));
    if (cat == "and") return b_->Mul(b1, b2);
    return (b_->IsOne(b1) || b_->IsOne(b2)) ? b_->One()
           : b_->IsZero(b1)                 ? b2
           : b_->IsZero(b2)                 ? b1
                                            : b_->Emit(mtlx::Op::Max, {b1, b2});
  }
  if (cat == "not") return b_->Not(in("in"));
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileVectorNode(
    const ScopedNode& node, const std::string& cat) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "normalize" || cat == "magnitude" || cat == "dotproduct") {
    std::string_view in_type =
        InputType(node.node, cat == "dotproduct" ? "in1" : "in");
    uint16_t vec_imm = in_type == "vector2" ? 2 : in_type == "vector4" ? 4 : 3;
    if (cat == "normalize") {
      return b_->Emit(mtlx::Op::Normalize, {in("in")}, vec_imm);
    }
    if (cat == "magnitude") {
      ValueId v = in("in");
      return b_->Emit(mtlx::Op::Sqrt,
                      {b_->Emit(mtlx::Op::Dot, {v, v}, vec_imm)});
    }
    ValueId i1 = in("in1"), i2 = in("in2");
    return b_->Emit(mtlx::Op::Dot, {i1, i2}, vec_imm);
  }
  if (cat == "rotate3d") {
    ValueId x = in("in"), amt = in("amount");
    if (b_->IsZero(amt)) return x;
    return b_->Emit(mtlx::Op::Rotate3D,
                    {x, amt, in("axis", b_->ConstRgb(0.0, 1.0, 0.0))});
  }
  if (cat == "normalmap") {
    std::string space = GetNodeInputString(node, "space", "tangent");
    if (space != "tangent") {
      WarnUnsupported("Unsupported MaterialX normalmap space '" + space +
                      "'; evaluating in tangent space.");
    }
    ValueId scale = in("scale", b_->One());
    // A zero `scale` keeps only the sign of the z component of `in`, which is
    // positive for `heighttonormal` (e.g. in `NG_bump_vector3`), so skip
    // evaluating its subgraph.
    ScopedNode up =
        b_->IsZero(scale) ? ResolveConnected(node, "in") : ScopedNode{};
    ValueId in_e = up && NodeCategory(up.node) == "heighttonormal"
                       ? b_->ConstRgb(0.5, 0.5, 1.0)
                       : in("in");
    ValueId n = in("normal", NWorld()), t = in("tangent", TWorld()),
            b = in("bitangent", BWorld());
    ValueId args[] = {in_e, scale, n, t, b};
    return b_->Emit(mtlx::Op::NormalMap, args);
  }
  if (cat == "heighttonormal") return CompileHeightToNormal(node);
  return std::nullopt;
}

ValueId MtlxExprCompiler::CompileHeightToNormal(const ScopedNode& node) {
  ValueId scale = CompileInputExpr(node, "scale", b_->One());
  if (b_->IsZero(scale)) return b_->ConstRgb(0.5, 0.5, 1.0);

  // Lower `in` (and `texcoord` if non-default) into an isolated `PendingRegion`
  // so `Op::HeightToNormal` can evaluate the subgraph at perturbed surface
  // interactions. Uniform subexpressions are hoisted back into the shared
  // uniform band by `PruneAndSchedule`.
  PendingRegion reg;
  mtlx::IrBuilder sub_builder(reg.nodes, reg.extra_args, val_prog_.consts,
                              val_prog_.opaques);
  mtlx::IrBuilder* saved_b = std::exchange(b_, &sub_builder);
  auto saved_out = std::exchange(node_output_val_, {});
  auto saved_mat = std::exchange(node_matrix_val_, {});

  ValueId h = CompileInputExpr(node, "in");
  std::optional<uint16_t> sub_idx;
  if (sub_builder.IsVarying(h)) {
    ValueId def_tc = DefaultTexcoord();
    ValueId tc = CompileInputExpr(node, "texcoord", def_tc);
    if (discover_texture_) {
      sub_idx = discover_input_idx_++;
    } else {
      bool has_tc = tc != def_tc;
      reg.root = has_tc
                     ? b_->Rgb(b_->Chan(h, 0), b_->Chan(tc, 0), b_->Chan(tc, 1))
                     : b_->Chan(h, 0);
      reg.has_texcoord = has_tc ? 1 : 0;
      if (pending_regions_.size() >= mtlx::kMaxImm) {
        throw std::runtime_error(
            "materialx_expr: too many heighttonormal nodes");
      }
      sub_idx = static_cast<uint16_t>(pending_regions_.size());
      pending_regions_.push_back(std::move(reg));
    }
  }

  b_ = saved_b;
  node_output_val_ = std::move(saved_out);
  node_matrix_val_ = std::move(saved_mat);
  return sub_idx ? b_->Emit(mtlx::Op::HeightToNormal, {scale}, *sub_idx)
                 : b_->ConstRgb(0.5, 0.5, 1.0);
}

ValueId MtlxExprCompiler::CompileGeomIntrinsic(std::string_view kind,
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
    ValueId color =
        MeshAttr("displayColor", type == "float" ? 1 : 3, b_->Zero());
    // The alpha channel of `geomcolor` is the display opacity.
    return type == "color4"
               ? b_->Vec4(color, MeshAttr("displayOpacity", 1, b_->One()))
               : color;
  }
  if (kind == "position") {
    if (space == "object" || space == "model") {
      return MeshAttr("__p_object", 3, PWorld());
    }
    warn_world_space();
    return PWorld();
  }
  if (kind == "normal") {
    if (space == "object" || space == "model") {
      return b_->Emit(mtlx::Op::Normalize,
                      {MeshAttr("__n_object", 3, NWorld())}, 3);
    }
    if (space == "tangent") return b_->ConstRgb(0.0, 0.0, 1.0);
    warn_world_space();
    return NWorld();
  }
  if (kind == "viewdirection") {
    warn_world_space();
    return b_->Emit(mtlx::Op::GeomIncidentDir);
  }
  bool bitangent = kind == "bitangent";
  if (space == "tangent") {
    return bitangent ? b_->ConstRgb(0.0, 1.0, 0.0)
                     : b_->ConstRgb(1.0, 0.0, 0.0);
  }
  warn_world_space();
  return bitangent ? BWorld() : TWorld();
}

ValueId MtlxExprCompiler::CompileGeomProp(
    const std::string& prop_name, std::string_view type,
    const std::function<ValueId()>& default_fn) {
  uint8_t ch = TypeChannels(type);
  if (mx::GeomPropDefPtr gpd = doc_->getGeomPropDef(prop_name)) {
    const std::string& gp = gpd->getGeomProp();
    if (IsGeomIntrinsic(gp)) {
      std::string_view idx = gpd->getIndex();
      ValueId e = CompileGeomIntrinsic(gp, gpd->getSpace(), type,
                                       !idx.empty() && idx != "0");
      if (ch == 1) return b_->Chan(e, 0);
      if (ch == 2 && b_->channels(e) > 2) {
        return b_->Vec2(b_->Chan(e, 0), b_->Chan(e, 1));
      }
      return e;
    }
  }
  return MeshAttr(prop_name, ch == 1 ? 1 : 3,
                  default_fn ? default_fn() : b_->Zero(), ch);
}

std::optional<ValueId> MtlxExprCompiler::CompileGeomNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
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

std::optional<ValueId> MtlxExprCompiler::CompileMatrixValueNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  if (cat == "transformmatrix") {
    // Row vectors are transformed as `v * M`. The last coordinate of a vector
    // with fewer coordinates than the matrix is 1.
    ValueId v = CompileInputExpr(node, "in");
    MatVal mat = CompileMatrixInput(node, "mat");
    ValueId last = type == (mat.n == 4 ? "vector3" : "vector2")
                       ? b_->One()
                       : b_->Chan(v, static_cast<uint16_t>(mat.n - 1));
    ValueId res = b_->Add(b_->Mul(mat.rows[0], b_->Chan(v, 0)),
                          b_->Mul(mat.rows[1], b_->Chan(v, 1)));
    res = mat.n == 4
              ? b_->Add(b_->Add(res, b_->Mul(mat.rows[2], b_->Chan(v, 2))),
                        b_->Mul(mat.rows[3], last))
              : b_->Add(res, b_->Mul(mat.rows[2], last));
    if (type == "vector2") return b_->Vec2(b_->Chan(res, 0), b_->Chan(res, 1));
    if (type == "vector3" && b_->channels(res) == 4) {
      return b_->Rgb(b_->Chan(res, 0), b_->Chan(res, 1), b_->Chan(res, 2));
    }
    return res;
  }
  if (cat == "determinant") {
    MatVal m = CompileMatrixInput(node, "in");
    return b_->Emit(mtlx::Op::MatDet, absl::MakeConstSpan(m.rows, m.n), 0, 1);
  }
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileCompositingNode(
    const ScopedNode& node, const std::string& cat) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "plus" || cat == "minus" || cat == "difference" ||
      cat == "screen" || cat == "burn" || cat == "dodge") {
    ValueId fg = in("fg"), bg = in("bg"), m = in("mix", b_->One());
    // Like MaterialX, burn and dodge are 0 where they divide by 0.
    if (cat == "burn" || cat == "dodge") {
      ValueId denom = cat == "burn" ? fg : b_->Sub(b_->One(), fg);
      ValueId blend =
          cat == "burn"
              ? b_->Sub(b_->One(), b_->Div(b_->Sub(b_->One(), bg), denom))
              : b_->Div(bg, denom);
      return b_->Select(
          b_->Emit(mtlx::Op::Lt, {b_->Emit(mtlx::Op::Abs, {denom}),
                                  b_->Const(mtlx::kMtlxFloatEps)}),
          b_->Zero(), b_->Mix(blend, bg, m));
    }
    ValueId blend = cat == "plus"    ? b_->Add(bg, fg)
                    : cat == "minus" ? b_->Sub(bg, fg)
                    : cat == "difference"
                        ? b_->Emit(mtlx::Op::Abs, {b_->Sub(bg, fg)})
                    : b_->IsOne(fg) ? b_->One()
                                    : b_->Fma(b_->Sub(b_->One(), bg), fg, bg);
    return b_->Mix(blend, bg, m);
  }
  if (cat == "inside") return b_->Mul(in("in"), in("mask", b_->One()));
  if (cat == "outside") {
    return b_->Mul(in("in"), b_->Sub(b_->One(), in("mask")));
  }
  if (cat == "premult" || cat == "unpremult") {
    ValueId in_e = in("in");
    ValueId a = b_->Chan(in_e, 3);
    ValueId rgb_scaled =
        cat == "premult" ? b_->Mul(in_e, a, 3) : b_->Div(in_e, a, 3);
    return b_->Vec4(rgb_scaled, a);
  }
  if (cat == "over" || cat == "in" || cat == "out" || cat == "mask" ||
      cat == "matte" || cat == "disjointover") {
    ValueId fg = in("fg"), bg = in("bg");
    ValueId fa = b_->Chan(fg, 3), ba = b_->Chan(bg, 3);
    ValueId composite;
    if (cat == "over") {
      composite = b_->Add(fg, b_->Mul(bg, b_->Sub(b_->One(), fa)));
    } else if (cat == "in") {
      composite = b_->Mul(fg, ba);
    } else if (cat == "out") {
      composite = b_->Mul(fg, b_->Sub(b_->One(), ba));
    } else if (cat == "mask") {
      composite = b_->Mul(bg, fa);
    } else if (cat == "matte") {
      composite = b_->Add(b_->Mul(fg, fa), b_->Mul(bg, b_->Sub(b_->One(), fa)));
      composite =
          b_->Vec4(composite, b_->Add(fa, b_->Mul(ba, b_->Sub(b_->One(), fa))));
    } else {
      composite = b_->Select(
          b_->Emit(mtlx::Op::Le, {b_->Add(fa, ba), b_->One()}), b_->Add(fg, bg),
          b_->Select(
              b_->Emit(mtlx::Op::Lt, {b_->Emit(mtlx::Op::Abs, {ba}),
                                      b_->Const(mtlx::kMtlxFloatEps)}),
              b_->Zero(),
              b_->Add(fg, b_->Mul(bg, b_->Div(b_->Sub(b_->One(), fa), ba)))));
      composite = b_->Vec4(
          composite, b_->Emit(mtlx::Op::Min, {b_->Add(fa, ba), b_->One()}));
    }
    return b_->Mix(composite, bg, in("mix", b_->One()));
  }
  if (cat == "ramplr" || cat == "ramptb" || cat == "splitlr" ||
      cat == "splittb") {
    bool lr = (cat == "ramplr" || cat == "splitlr");
    ValueId v0 = in(lr ? "valuer" : "valuet"),
            v1 = in(lr ? "valuel" : "valueb");
    ValueId tc = in("texcoord", DefaultTexcoord());
    ValueId tc_comp = b_->Chan(tc, lr ? 0 : 1);
    if (cat == "ramplr" || cat == "ramptb") {
      return b_->Mix(v0, v1, Clip01(tc_comp));
    }
    return b_->Select(
        b_->Emit(mtlx::Op::Le, {in("center", b_->Const(0.5)), tc_comp}), v0,
        v1);
  }
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileColorPbrNode(
    const ScopedNode& node, const std::string& cat, const std::string& type,
    const std::string& output_name) {
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  if (cat == "rgbtohsv" || cat == "hsvtorgb") {
    return b_->Emit(cat == "rgbtohsv" ? mtlx::Op::RgbToHsv : mtlx::Op::HsvToRgb,
                    {in("in")}, 0, type == "color4" ? 4 : 3);
  }
  if (cat == "luminance") {
    // The luminance of a gray value is the value itself, which keeps e.g. the
    // default `opacity` color of `standard_surface` a constant 1.
    ValueId in_e = in("in");
    ResolvedInput cr = ResolveInput(node, "lumacoeffs");
    bool has_coeffs = cr.kind != ResolvedInput::None &&
                      !(cr.kind == ResolvedInput::Value &&
                        cr.elem->getParent() == GetNodeInfo(node.node).nodedef);
    if (!has_coeffs && b_->is_scalar(in_e)) return in_e;
    ValueId lum = b_->Emit(
        mtlx::Op::Dot,
        {in_e, has_coeffs ? CompileResolvedInput(cr, b_->Zero())
                          : b_->ConstRgb(0.2722287, 0.6740818, 0.0536895)},
        3);
    return type == "color4" ? b_->Vec4(lum, b_->Chan(in_e, 3)) : lum;
  }
  if (cat == "blackbody") {
    return b_->Emit(mtlx::Op::Blackbody,
                    {in("temperature", b_->Const(5000.0))});
  }
  if (cat == "roughness_anisotropy") {
    ValueId r = in("roughness");
    ValueId r2 =
        b_->Emit(mtlx::Op::Clip,
                 {b_->Mul(r, r), b_->Const(mtlx::kMtlxFloatEps), b_->One()});
    ValueId aspect = b_->Emit(
        mtlx::Op::Sqrt,
        {b_->Sub(b_->One(),
                 b_->Emit(mtlx::Op::Clip,
                          {in("anisotropy"), b_->Zero(), b_->Const(0.98)}))});
    if (b_->IsOne(aspect)) return r2;
    return b_->Vec2(b_->Emit(mtlx::Op::Min, {b_->Div(r2, aspect), b_->One()}),
                    b_->Mul(r2, aspect));
  }
  if (cat == "roughness_dual") {
    ValueId r = in("roughness");
    if (b_->is_scalar(r)) {
      return b_->Emit(
          mtlx::Op::Clip,
          {b_->Mul(r, r), b_->Const(mtlx::kMtlxFloatEps), b_->One()});
    }
    ValueId rx = b_->Chan(r, 0);
    ValueId ry =
        b_->Select(b_->Emit(mtlx::Op::Lt, {b_->Chan(r, 1), b_->Zero()}), rx,
                   b_->Chan(r, 1));
    return b_->Vec2(
        b_->Emit(mtlx::Op::Clip,
                 {b_->Mul(rx, rx), b_->Const(mtlx::kMtlxFloatEps), b_->One()}),
        b_->Emit(mtlx::Op::Clip,
                 {b_->Mul(ry, ry), b_->Const(mtlx::kMtlxFloatEps), b_->One()}));
  }
  if (cat == "artistic_ior") {
    ValueId g = in("edge_color");
    if (b_->IsZero(g) && output_name == "extinction") return b_->Zero();
    ValueId r = b_->Emit(mtlx::Op::Clip,
                         {in("reflectivity"), b_->Zero(), b_->Const(0.99)});
    if (b_->IsOne(g)) {
      ValueId inv = b_->Div(b_->One(), b_->Add(b_->One(), r));
      return output_name != "extinction"
                 ? b_->Mul(b_->Sub(b_->One(), r), inv)
                 : b_->Mul(
                       b_->Mul(b_->Const(2.0), b_->Emit(mtlx::Op::Sqrt, {r})),
                       inv);
    }
    ValueId r_sqrt = b_->Emit(mtlx::Op::Sqrt, {r});
    if (b_->IsZero(g)) {
      return b_->Div(b_->Add(b_->One(), r_sqrt), b_->Sub(b_->One(), r_sqrt));
    }
    ValueId eta = b_->Mix(
        b_->Div(b_->Sub(b_->One(), r), b_->Add(b_->One(), r)),
        b_->Div(b_->Add(b_->One(), r_sqrt), b_->Sub(b_->One(), r_sqrt)), g);
    if (output_name != "extinction") return eta;
    ValueId np1 = b_->Add(eta, b_->One()), nm1 = b_->Sub(eta, b_->One());
    return b_->Emit(
        mtlx::Op::Sqrt,
        {b_->Div(b_->Sub(b_->Mul(b_->Mul(np1, np1), r), b_->Mul(nm1, nm1)),
                 b_->Sub(b_->One(), r))});
  }
  if (cat == "chiang_hair_roughness") {
    auto clip_in = [&](const char* name, double def) {
      return b_->Emit(mtlx::Op::Clip,
                      {in(name, b_->Const(def)), b_->Const(1e-3), b_->One()});
    };
    ValueId lr = clip_in("longitudinal", 0.1), ar = clip_in("azimuthal", 0.2);
    ValueId v_sqrt =
        b_->Add(b_->Add(b_->Mul(b_->Const(0.726), lr),
                        b_->Mul(b_->Const(0.812), b_->Mul(lr, lr))),
                b_->Mul(b_->Const(3.7),
                        b_->Emit(mtlx::Op::Pow, {lr, b_->Const(20.0)})));
    if (output_name == "roughness_TT" || output_name == "roughness_TRT") {
      bool is_tt = output_name == "roughness_TT";
      v_sqrt = b_->Mul(v_sqrt, in(is_tt ? "scale_TT" : "scale_TRT",
                                  b_->Const(is_tt ? 0.5 : 2.0)));
    }
    ValueId s =
        b_->Add(b_->Add(b_->Mul(b_->Const(0.265), ar),
                        b_->Mul(b_->Const(1.194), b_->Mul(ar, ar))),
                b_->Mul(b_->Const(5.372),
                        b_->Emit(mtlx::Op::Pow, {ar, b_->Const(22.0)})));
    return b_->Vec2(b_->Mul(v_sqrt, v_sqrt), s);
  }
  if (cat == "chiang_hair_absorption_from_color") {
    ValueId c = b_->Emit(mtlx::Op::Clip,
                         {in("color", b_->One()), b_->Const(1e-3), b_->One()});
    ValueId b = in("azimuthal_roughness", b_->Const(0.2));
    ValueId b_fac = b_->Add(
        b_->Const(5.969),
        b_->Mul(
            b,
            b_->Add(
                b_->Const(-0.215),
                b_->Mul(
                    b,
                    b_->Add(
                        b_->Const(2.532),
                        b_->Mul(
                            b,
                            b_->Add(
                                b_->Const(-10.73),
                                b_->Mul(b,
                                        b_->Add(b_->Const(5.574),
                                                b_->Mul(b, b_->Const(
                                                               0.245)))))))))));
    ValueId sigma = b_->Div(b_->Emit(mtlx::Op::Log, {c}), b_fac);
    return b_->Mul(sigma, sigma);
  }
  if (cat == "deon_hair_absorption_from_melanin") {
    ValueId log_conc = b_->Emit(
        mtlx::Op::Log,
        {b_->Emit(mtlx::Op::Max, {b_->Sub(b_->One(), in("melanin_concentration",
                                                        b_->Const(0.25))),
                                  b_->Const(1e-4)})});
    ValueId pigment = CompileMixVal(
        in("melanin_redness", b_->Const(0.5)),
        [&] {
          return b_->Emit(mtlx::Op::Log,
                          {in("pheomelanin_color",
                              b_->ConstRgb(0.829444, 0.67032, 0.349938))});
        },
        [&] {
          return b_->Emit(mtlx::Op::Log,
                          {in("eumelanin_color",
                              b_->ConstRgb(0.657704, 0.498077, 0.254107))});
        });
    return b_->Emit(mtlx::Op::Max, {b_->Mul(log_conc, pigment), b_->Zero()});
  }
  return std::nullopt;
}

std::optional<ValueId> MtlxExprCompiler::CompileNoiseNode(
    const ScopedNode& node, const std::string& cat, const std::string& type) {
  std::string_view noise = cat;
  bool is_3d = absl::ConsumeSuffix(&noise, "3d");
  if ((!is_3d && !absl::ConsumeSuffix(&noise, "2d")) ||
      (noise != "cellnoise" && noise != "worleynoise" && noise != "noise" &&
       noise != "fractal")) {
    return std::nullopt;
  }
  auto in = [&](const std::string& name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  uint16_t imm = mtlx::pack_noise_imm(is_3d, type != "float");
  ValueId coord = is_3d ? in("position", MeshAttr("__p_object", 3, PWorld()))
                        : in("texcoord", DefaultTexcoord());
  if (noise == "cellnoise") {
    return b_->Emit(mtlx::Op::CellNoise, {coord}, imm);
  }
  if (noise == "worleynoise") {
    ValueId jitter = in("jitter", b_->One()), style = in("style");
    uint8_t ch = type == "vector2" ? 2 : (type == "float" ? 1 : 3);
    return b_->Emit(mtlx::Op::Worley, {coord, jitter, style}, imm, ch);
  }
  bool is_noise = noise == "noise";
  std::optional<int> oct =
      is_noise ? std::nullopt : ConstIntInput(node, "octaves");
  bool four_ch = IsFourChannelType(type);
  bool split_vec2 = !is_noise && type == "vector2";
  ValueId lac = kInvalidValue, dim = kInvalidValue, dyn_oct = kInvalidValue;
  if (!is_noise) {
    lac = in("lacunarity", b_->Const(2.0));
    dim = in("diminish", b_->Const(0.5));
    if (!oct) dyn_oct = in("octaves");
  }
  auto eval_one = [&](bool vec3, ValueId p, uint8_t ch = 0) {
    if (is_noise) {
      return b_->Emit(mtlx::Op::Noise, {p}, mtlx::pack_noise_imm(is_3d, vec3),
                      ch);
    }
    if (oct) {
      uint16_t clamped =
          static_cast<uint16_t>(std::clamp(*oct, 0, mtlx::kMaxFractalOctaves));
      return b_->Emit(mtlx::Op::Fractal, {p, lac, dim},
                      mtlx::pack_noise_imm(is_3d, vec3, clamped), ch);
    }
    ValueId args[] = {p, lac, dim, dyn_oct};
    return b_->Emit(
        mtlx::Op::Fractal, args,
        mtlx::pack_noise_imm(is_3d, vec3, mtlx::kFractalDynamicOctaves), ch);
  };
  auto offset_coord = [&]() {
    constexpr double kNoiseOffset[3] = {19.0, 73.0, 29.0};
    constexpr double kFractalOffset[3] = {19.0, 193.0, 17.0};
    const double* off = is_noise ? kNoiseOffset : kFractalOffset;
    if (is_3d) return b_->Add(coord, b_->ConstRgb(off[0], off[1], off[2]));
    return b_->Add(coord, b_->Vec2(b_->Const(off[0]), b_->Const(off[1])));
  };
  ValueId res;
  if (four_ch) {
    // Like MaterialX, a fourth channel is scalar noise at an offset.
    res = b_->Vec4(eval_one(true, coord), eval_one(false, offset_coord()));
  } else if (split_vec2) {
    // Like MaterialX, `vector2` fractals are two scalar fractals.
    res = b_->Vec2(eval_one(false, coord), eval_one(false, offset_coord()));
  } else {
    uint8_t ch = type == "vector2" ? 2 : 0;
    res = eval_one(type != "float", coord, ch);
  }
  res = b_->Mul(res, in("amplitude", b_->One()));
  return is_noise ? b_->Add(res, in("pivot")) : res;
}

ValueId MtlxExprCompiler::CompileNodeOutputExpr(
    const ScopedNode& node, const std::string& output_name) {
  const std::string& cat = NodeCategory(node.node);
  const std::string& type = node.node->getType();

  // Fast path for `range` with unit gamma, collapsing `abs + pow + sign + mul`
  // (e.g. default `range` nodes inside `NG_unifiednoise*`) to a linear remap.
  if (cat == "range") {
    if (b_->IsOne(CompileInputExpr(node, "gamma", b_->One()))) {
      ValueId ilo = CompileInputExpr(node, "inlow");
      ValueId ihi = CompileInputExpr(node, "inhigh", b_->One());
      ValueId olo = CompileInputExpr(node, "outlow");
      ValueId ohi = CompileInputExpr(node, "outhigh", b_->One());
      ValueId res = CompileLinearRemap(node, ilo, ihi, olo, ohi);
      ValueId doclamp = CompileInputExpr(node, "doclamp");
      if (b_->IsZero(doclamp)) return res;
      ValueId clamped = b_->Emit(mtlx::Op::Clip, {res, olo, ohi});
      return b_->IsOne(doclamp) ? clamped : b_->Select(doclamp, clamped, res);
    }
  }
  if (cat == "smoothstep") {
    // Handle before `graph_impl` expansion so multi-channel `NG_smoothstep_*`
    // stays vectorized. Like OSL, a step function at `low` if `low >= high`.
    ValueId lo = CompileInputExpr(node, "low");
    ValueId hi = CompileInputExpr(node, "high", b_->One());
    ValueId x = CompileInputExpr(node, "in");
    ValueId t = b_->Div(b_->Sub(x, lo), b_->Sub(hi, lo));
    return b_->Select(
        b_->Emit(mtlx::Op::Lt, {x, lo}), b_->Zero(),
        b_->Select(b_->Emit(mtlx::Op::Le, {hi, x}), b_->One(),
                   b_->Mul(b_->Mul(t, t),
                           b_->Fma(b_->Const(-2.0), t, b_->Const(3.0)))));
  }
  if (cat == "hsvadjust") {
    ValueId amt = CompileInputExpr(node, "amount", b_->ConstRgb(0.0, 1.0, 1.0));
    if (b_->channels(amt) == 3) {
      if (auto v = b_->ConstValue(amt);
          v && (*v)[0] == 0.0 && (*v)[1] == 1.0 && (*v)[2] == 1.0) {
        return CompileInputExpr(node, "in");
      }
    }
  }
  if (cat == "contrast") {
    if (b_->IsOne(CompileInputExpr(node, "amount", b_->One()))) {
      return CompileInputExpr(node, "in");
    }
  }
  // Handle noise before NodeGraph expansion so stdlib noise/fractal NodeGraphs
  // emit native noise ops directly.
  if (auto e = CompileNoiseNode(node, cat, type)) return *e;

  if (mx::NodeGraphPtr ng = GetNodeInfo(node.node).graph_impl) {
    return CompileResolvedInput(
        ResolveOutput(node, ResolveNodeGraphOutput(ng, output_name)),
        b_->Zero());
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
      return b_->Emit(
          op, {CompileInputExpr(node, "in",
                                op == mtlx::Op::Log ? b_->One() : b_->Zero())});
    }
  }
  if (cat == "power") {
    ValueId exp = CompileInputExpr(node, "in2", b_->One());
    // Skip compiling dead `in1` (and its textures/opaques) when `exp == 0`.
    if (b_->IsZero(exp)) return b_->One();
    return b_->Emit(mtlx::Op::Pow, {CompileInputExpr(node, "in1"), exp});
  }
  if (cat == "rotate2d") {
    ValueId amt = CompileInputExpr(node, "amount");
    return b_->Emit(mtlx::Op::Rotate2D, {CompileInputExpr(node, "in"), amt});
  }
  static constexpr struct {
    std::string_view name;
    const char *in1, *in2;
    mtlx::Op op;
    bool def2_one;
  } kBinaryOps[] = {
      {"min", "in1", "in2", mtlx::Op::Min, false},
      {"max", "in1", "in2", mtlx::Op::Max, false},
      {"atan2", "iny", "inx", mtlx::Op::Atan2, true},
      {"crossproduct", "in1", "in2", mtlx::Op::Cross, false},
  };
  for (const auto& [name, in1, in2, op, def2_one] : kBinaryOps) {
    if (cat == name) {
      ValueId e1 = CompileInputExpr(node, in1);
      ValueId e2 =
          CompileInputExpr(node, in2, def2_one ? b_->One() : b_->Zero());
      return b_->Emit(op, {e1, e2});
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

ValueId MtlxExprCompiler::CompileEdf(const ScopedNode& node) {
  if (!node) return b_->Zero();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = edf_val_.find(key); it != edf_val_.end()) return it->second;

  const std::string& cat = NodeCategory(node.node);
  ValueId res = b_->Zero();
  if (cat == "surface" || cat == "surface_unlit") {
    if (cat == "surface") {
      res = CompileEdf(ResolveConnected(node, "edf"));
    } else {
      ValueId em = CompileInputExpr(node, "emission", b_->One());
      if (!b_->IsZero(em)) {
        em = b_->Mul(
            em,
            b_->Sub(b_->One(), Clip01(CompileInputExpr(node, "transmission"))));
      }
      if (!b_->IsZero(em)) {
        res = b_->Mul(em, CompileInputExpr(node, "emission_color", b_->One()));
      }
    }
    if (!b_->IsZero(res)) {
      res = b_->Mul(res, Clip01(CompileInputExpr(node, "opacity", b_->One())));
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
    res = CompileInputExpr(node, "color", b_->One());
  } else if (cat == "generalized_schlick_edf") {
    ValueId base = CompileEdf(ResolveConnected(node, "base"));
    if (!b_->IsZero(base)) {
      ValueId c0 = CompileInputExpr(node, "color0", b_->One());
      ValueId c90 = CompileInputExpr(node, "color90", b_->One());
      // Weight `base` by the cosine-weighted hemispherical average of
      // `mix(c0, c90, (1 - cos)^exp)`, which equals
      // `mix(c0, c90, 2 / ((exp + 1) * (exp + 2)))`.
      ValueId w = c0;
      if (c0 != c90) {
        ValueId exp = b_->Emit(
            mtlx::Op::Max,
            {CompileInputExpr(node, "exponent", b_->Const(5.0)), b_->Zero()});
        w = b_->Mix(
            c90, c0,
            b_->Div(b_->Const(2.0), b_->Mul(b_->Add(exp, b_->One()),
                                            b_->Add(exp, b_->Const(2.0)))));
      }
      res = b_->Mul(base, w);
    }
  } else if (cat == "add") {
    res = b_->Add(CompileEdf(ResolveConnected(node, "in1")),
                  CompileEdf(ResolveConnected(node, "in2")));
  } else if (cat == "multiply") {
    ValueId base = CompileEdf(ResolveConnected(node, "in1"));
    if (!b_->IsZero(base)) {
      res = b_->Mul(base, CompileInputExpr(node, "in2", b_->One()));
    }
  } else if (cat == "mix") {
    res = CompileMixVal(
        CompileInputExpr(node, "mix"),
        [&] { return CompileEdf(ResolveConnected(node, "fg")); },
        [&] { return CompileEdf(ResolveConnected(node, "bg")); });
  }
  edf_val_[key] = res;
  return res;
}

ValueId MtlxExprCompiler::CompileOpacity(const ScopedNode& node) {
  if (!node) return b_->One();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = opacity_val_.find(key); it != opacity_val_.end()) {
    return it->second;
  }
  const std::string& cat = NodeCategory(node.node);
  ValueId res = b_->One();
  if (cat == "surface" || cat == "surface_unlit") {
    res = Clip01(CompileInputExpr(node, "opacity", b_->One()));
  } else if (cat == "mix" &&
             node.node->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
    res = CompileMixVal(
        Clip01(CompileInputExpr(node, "mix")),
        [&] { return CompileOpacity(ResolveConnected(node, "fg")); },
        [&] { return CompileOpacity(ResolveConnected(node, "bg")); });
  }
  if (!b_->is_scalar(res)) {
    res = b_->Emit(mtlx::Op::Dot, {res, b_->Const(1.0 / 3.0)}, 3);
  }
  opacity_val_[key] = res;
  return res;
}

void MtlxExprCompiler::MarkTwoSidedDielectrics(size_t start) {
  for (size_t i = start; i < closures_.size(); ++i) {
    const mtlx::ClosureNode& c = closures_[i];
    if ((c.op == mtlx::ClosureOp::Dielectric ||
         c.op == mtlx::ClosureOp::GeneralizedSchlick) &&
        (c.microfacet.mode & mtlx::kScatterModeMask) != mtlx::kScatterR) {
      return;
    }
  }
  for (size_t i = start; i < closures_.size(); ++i) {
    mtlx::ClosureNode& c = closures_[i];
    if (c.op == mtlx::ClosureOp::Dielectric) {
      c.microfacet.mode |= mtlx::kScatterTwoSided;
    }
  }
}

int32_t MtlxExprCompiler::EmitClosure(mtlx::ClosureNode node) {
  bool opaque = true;
  switch (node.op) {
    case mtlx::ClosureOp::Null:
      opaque = false;
      break;
    case mtlx::ClosureOp::Layer:
    case mtlx::ClosureOp::Add:
      opaque =
          IsClosureOpaque(node.comb.first) || IsClosureOpaque(node.comb.second);
      break;
    case mtlx::ClosureOp::Mix:
      opaque =
          IsClosureOpaque(node.comb.first) && IsClosureOpaque(node.comb.second);
      break;
    case mtlx::ClosureOp::Multiply:
      opaque = IsClosureOpaque(node.comb.first);
      break;
    default:
      opaque = !mtlx::leaf_has_layer_throughput(node);
      break;
  }
  closures_.push_back(node);
  closure_opaque_.push_back(opaque);
  return static_cast<int32_t>(closures_.size()) - 1;
}

bool MtlxExprCompiler::IsClosureOpaque(int32_t idx) const {
  return idx >= 0 && closure_opaque_[idx];
}

int32_t MtlxExprCompiler::EmitOpaqueBlocker() {
  // Opaque and transmissive leaves still block `layer` bases (`throughput = 0`)
  // at `weight == 0`, without compiling dead inputs or emitting an active leaf.
  if (opaque_blocker_ < 0) {
    opaque_blocker_ = EmitClosure(mtlx::ClosureNode::MakeOpaqueBlocker());
  }
  return opaque_blocker_;
}

ValueId MtlxExprCompiler::FrameValue(const ScopedNode& node,
                                     const std::string& name, ValueId def) {
  ValueId e = CompileInputExpr(node, name, def);
  return e == def ? kInvalidValue : e;
}

int32_t MtlxExprCompiler::CompileDiffuseClosure(const ScopedNode& node,
                                                const std::string& cat) {
  ValueId w = CompileInputExpr(node, "weight", b_->One());
  if (b_->IsZero(w)) return cat == "sheen_bsdf" ? -1 : EmitOpaqueBlocker();
  ValueId color = CompileInputExpr(node, "color", b_->One());
  ValueId rough = cat == "translucent_bsdf"
                      ? kInvalidValue
                      : CompileInputExpr(node, "roughness");
  ValueId normal = FrameValue(node, "normal", NWorld());
  if (cat == "translucent_bsdf") {
    return EmitClosure(mtlx::ClosureNode::MakeTranslucent(w, color, normal));
  }
  if (cat == "burley_diffuse_bsdf") {
    return EmitClosure(
        mtlx::ClosureNode::MakeBurleyDiffuse(w, color, rough, normal));
  }
  if (cat == "sheen_bsdf") {
    uint16_t mode = GetNodeInputString(node, "mode") == "zeltner"
                        ? mtlx::kSheenZeltner
                        : mtlx::kSheenContyKulla;
    return EmitClosure(
        mtlx::ClosureNode::MakeSheen(w, color, rough, mode, normal));
  }
  uint16_t energy_comp =
      (cat == "oren_nayar_diffuse_bsdf" &&
       !b_->IsZero(CompileInputExpr(node, "energy_compensation")))
          ? 1
          : 0;
  return EmitClosure(
      mtlx::ClosureNode::MakeDiffuse(w, color, rough, energy_comp, normal));
}

int32_t MtlxExprCompiler::CompileMicrofacetClosure(const ScopedNode& node,
                                                   const std::string& cat) {
  ValueId w = CompileInputExpr(node, "weight", b_->One());
  bool is_cond = cat == "conductor_bsdf";
  uint16_t sm = mtlx::kScatterR;
  if (!is_cond) {
    std::string sm_str = GetNodeInputString(node, "scatter_mode");
    sm = sm_str == "T"    ? mtlx::kScatterT
         : sm_str == "RT" ? mtlx::kScatterRT
                          : mtlx::kScatterR;
  }
  if (b_->IsZero(w)) {
    return (!is_cond && sm == mtlx::kScatterR) ? -1 : EmitOpaqueBlocker();
  }
  ValueId tf_t = kInvalidValue, tf_ior = kInvalidValue;
  if (sm != mtlx::kScatterT) {
    if (ValueId thickness = CompileInputExpr(node, "thinfilm_thickness");
        !b_->IsZero(thickness)) {
      tf_t = thickness;
      tf_ior = CompileInputExpr(node, "thinfilm_ior", b_->Const(1.5));
    }
  }
  uint16_t mode = static_cast<uint16_t>(
      sm |
      (!b_->IsZero(CompileInputExpr(node, "retroreflective"))
           ? mtlx::kScatterRetroreflective
           : 0) |
      ((is_cond || (cat == "generalized_schlick_bsdf" && sm == mtlx::kScatterR))
           ? mtlx::kScatterTwoSided
           : 0));
  ValueId rough = CompileInputExpr(node, "roughness");
  ValueId normal = FrameValue(node, "normal", NWorld());
  ValueId tangent = FrameValue(node, "tangent", TWorld());
  if (cat == "dielectric_bsdf") {
    return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
        mtlx::ClosureOp::Dielectric, w, rough, mode, normal, tangent, tf_t,
        tf_ior, CompileInputExpr(node, "tint", b_->One()),
        CompileInputExpr(node, "ior", b_->One())));
  }
  if (is_cond) {
    return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
        mtlx::ClosureOp::Conductor, w, rough, mode, normal, tangent, tf_t,
        tf_ior, CompileInputExpr(node, "ior", b_->One()),
        CompileInputExpr(node, "extinction")));
  }
  return EmitClosure(mtlx::ClosureNode::MakeMicrofacet(
      mtlx::ClosureOp::GeneralizedSchlick, w, rough, mode, normal, tangent,
      tf_t, tf_ior, CompileInputExpr(node, "color0", b_->One()),
      CompileInputExpr(node, "color82", b_->One()),
      CompileInputExpr(node, "color90", b_->One()),
      CompileInputExpr(node, "exponent", b_->Const(5.0))));
}

int32_t MtlxExprCompiler::CompileChiangHairClosure(const ScopedNode& node) {
  auto in = [&](const char* name, ValueId def = kInvalidValue) {
    return CompileInputExpr(node, name, def);
  };
  return EmitClosure(mtlx::ClosureNode::MakeChiangHair(
      in("tint_R", b_->One()), in("tint_TT", b_->One()),
      in("tint_TRT", b_->One()), in("roughness_R", b_->Const(0.1)),
      in("roughness_TT", b_->Const(0.05)), in("roughness_TRT", b_->Const(0.2)),
      in("ior", b_->Const(1.55)), in("absorption_coefficient"),
      in("cuticle_angle", b_->Const(0.5)), FrameValue(node, "normal", NWorld()),
      FrameValue(node, "curve_direction", TWorld())));
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

int32_t MtlxExprCompiler::CompileBsdfClosure(const ScopedNode& node) {
  if (!node) return -1;
  NodeKey key{node.frame, node.node.get()};
  if (auto it = bsdf_closure_.find(key); it != bsdf_closure_.end()) {
    return it->second;
  }

  const std::string& cat = NodeCategory(node.node);
  auto is_blocker = [&](int32_t c) {
    return c >= 0 && closures_[c].op == mtlx::ClosureOp::OpaqueBlocker;
  };
  auto scale = [&](int32_t c, ValueId w) {
    return (c < 0 || b_->IsOne(w) || is_blocker(c))
               ? c
               : EmitClosure(mtlx::ClosureNode::MakeMultiply(c, w));
  };

  int32_t idx = -1;
  if (cat == "oren_nayar_diffuse_bsdf" || cat == "subsurface_bsdf" ||
      cat == "burley_diffuse_bsdf" || cat == "sheen_bsdf" ||
      cat == "translucent_bsdf") {
    idx = CompileDiffuseClosure(node, cat);
  } else if (cat == "dielectric_bsdf" || cat == "conductor_bsdf" ||
             cat == "generalized_schlick_bsdf") {
    idx = CompileMicrofacetClosure(node, cat);
  } else if (cat == "chiang_hair_bsdf") {
    idx = CompileChiangHairClosure(node);
  } else if (node.node->getType() == mx::VDF_TYPE_STRING) {
    CompileVdfWarning(node, cat);
  } else if (cat == "layer" || cat == "add") {
    bool is_layer = cat == "layer";
    int32_t c0 =
        CompileBsdfClosure(ResolveConnected(node, is_layer ? "top" : "in1"));
    // Prune occluded BSDF bases (and VDF bases under a zero-weight blocker),
    // while still visiting active `layer_vdf` bases for VDF warnings.
    if (is_layer && IsClosureOpaque(c0) &&
        (is_blocker(c0) ||
         InputType(node.node, "base") != mx::VDF_TYPE_STRING)) {
      idx = c0;
    } else if (int32_t c1 = CompileBsdfClosure(
                   ResolveConnected(node, is_layer ? "base" : "in2"));
               c0 < 0 || c1 < 0) {
      idx = c0 < 0 ? c1 : c0;
    } else {
      idx = EmitClosure(is_layer ? mtlx::ClosureNode::MakeLayer(c0, c1)
                                 : mtlx::ClosureNode::MakeAdd(c0, c1));
    }
  } else if (cat == "mix") {
    ValueId mix_val = CompileInputExpr(node, "mix");
    ValueId mix_sat = Clip01(mix_val);
    if (b_->IsZero(mix_sat)) {
      idx = CompileBsdfClosure(ResolveConnected(node, "bg"));
    } else if (b_->IsOne(mix_sat)) {
      idx = CompileBsdfClosure(ResolveConnected(node, "fg"));
    } else {
      int32_t fg_idx = CompileBsdfClosure(ResolveConnected(node, "fg"));
      int32_t bg_idx = CompileBsdfClosure(ResolveConnected(node, "bg"));
      if (fg_idx == bg_idx) {
        idx = fg_idx;
      } else {
        if (fg_idx < 0) fg_idx = EmitClosure(mtlx::ClosureNode::MakeNull());
        if (bg_idx < 0) bg_idx = EmitClosure(mtlx::ClosureNode::MakeNull());
        idx = EmitClosure(mtlx::ClosureNode::MakeMix(fg_idx, bg_idx, mix_val));
      }
    }
  } else if (cat == "multiply") {
    int32_t c = CompileBsdfClosure(ResolveConnected(node, "in1"));
    idx = (c < 0 || is_blocker(c))
              ? c
              : scale(c, CompileInputExpr(node, "in2", b_->One()));
  } else if (cat == "surface") {
    if (ValueId op = CompileOpacity(node); !b_->IsZero(op)) {
      if (!b_->IsZero(CompileInputExpr(node, "thin_walled"))) {
        thin_walled_ = true;
      }
      // Scope BSDF deduplication to this surface so `MarkTwoSidedDielectrics`
      // cannot mutate a closure shared with another mixed surface.
      auto saved = std::exchange(bsdf_closure_, {});
      size_t start = closures_.size();
      int32_t bsdf_idx = CompileBsdfClosure(ResolveConnected(node, "bsdf"));
      MarkTwoSidedDielectrics(start);
      bsdf_closure_ = std::move(saved);
      idx = scale(bsdf_idx, op);
    }
  } else if (cat == "surface_unlit") {
    // Unlit surfaces emit radiance only and have a null BSDF.
  } else if (node.node->getType() == mx::BSDF_TYPE_STRING &&
             InputType(node.node, "weight") == "float" &&
             b_->IsZero(CompileInputExpr(node, "weight", b_->One()))) {
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
  ValueId op_val = CompileOpacity(node);
  if (!b_->IsOne(op_val)) opacity_root_ = op_val;
  if (b_->IsZero(op_val) || CompileBsdfClosure(node) < 0) {
    closures_ = {mtlx::ClosureNode::MakeNull()};
  } else if (!thin_walled_ &&
             node.node->getType() != mx::SURFACE_SHADER_TYPE_STRING) {
    MarkTwoSidedDielectrics(0);
  }
}

MtlxExprCompiler::DisplacementVal MtlxExprCompiler::CompileDisplacement(
    const ScopedNode& node) {
  if (!node) return {b_->Zero(), false};
  NodeKey key{node.frame, node.node.get()};
  if (auto it = disp_val_.find(key); it != disp_val_.end()) return it->second;

  const std::string& cat = NodeCategory(node.node);
  DisplacementVal res{b_->Zero(), false};
  if (cat == "displacement") {
    if (IsNonZeroInput(node, "displacement")) {
      ValueId scale = CompileInputExpr(node, "scale", b_->One());
      if (!b_->IsZero(scale)) {
        bool is_vec = InputType(node.node, "displacement") == "vector3";
        ValueId disp = b_->Mul(CompileInputExpr(node, "displacement"), scale);
        if (!b_->IsZero(disp)) {
          res = {is_vec ? disp : b_->Chan(disp, 0), is_vec};
        }
      }
    }
  } else if (cat == "mix") {
    auto disp_in = [&](const std::string& name) {
      return CompileDisplacement(
          ResolveConnected(node, name, "displacementshader"));
    };
    ValueId m = CompileInputExpr(node, "mix");
    if (b_->IsZero(m)) {
      res = disp_in("bg");
    } else if (b_->IsOne(m)) {
      res = disp_in("fg");
    } else {
      DisplacementVal fg = disp_in("fg"), bg = disp_in("bg");
      if (!b_->IsZero(fg.val) || !b_->IsZero(bg.val)) {
        bool is_vec = fg.is_vector || bg.is_vector;
        auto promote = [&](const DisplacementVal& d) {
          return (!is_vec || d.is_vector)
                     ? d.val
                     : b_->Rgb(b_->Zero(), b_->Zero(), d.val);
        };
        res = {b_->Mix(promote(fg), promote(bg), m), is_vec};
      }
    }
  } else {
    WarnUnsupported("Unsupported MaterialX displacement shader '" + cat +
                    "'; ignoring displacement.");
  }
  disp_val_[key] = res;
  return res;
}

void MtlxExprCompiler::AppendPendingRegions() {
  val_prog_.main_end = static_cast<uint32_t>(val_prog_.nodes.size());
  val_prog_.regions.reserve(pending_regions_.size());
  for (const PendingRegion& reg : pending_regions_) {
    val_prog_.append_region(reg.nodes, reg.extra_args, reg.root,
                            reg.has_texcoord);
  }
  pending_regions_.clear();
}

void MtlxExprCompiler::ApplyCompaction(const mtlx::OptimizeResult& res) {
  auto compact = [](auto& vec, absl::Span<const uint16_t> live) {
    for (size_t i = 0; i < live.size(); ++i) {
      if (i != live[i]) vec[i] = std::move(vec[live[i]]);
    }
    vec.resize(live.size());
  };
  compact(inputs_, res.live_inputs);
  compact(attr_requests_, res.live_attrs);
}

void MtlxExprCompiler::FinalizeExpr(ValueId root) {
  val_prog_.result_root = root;
  AppendPendingRegions();
  ApplyCompaction(mtlx::OptimizeValueProgram(val_prog_));
}

bool MtlxExprCompiler::ProgramUsesUv() const {
  return std::any_of(
      val_prog_.nodes.begin(), val_prog_.nodes.end(), [&](const mtlx::Node& n) {
        return n.op == mtlx::Op::GeomUV || n.op == mtlx::Op::SampleInput ||
               (n.op == mtlx::Op::HeightToNormal &&
                !val_prog_.regions[n.imm].has_texcoord);
      });
}

template <typename ProgT>
ref<Object> MtlxExprCompiler::CreatePlugin(std::string_view name,
                                           ObjectType type,
                                           std::string_view variant,
                                           ProgT&& prog) {
  Properties props(name);
  for (size_t i = 0; i < inputs_.size(); ++i) {
    props.set("in" + std::to_string(i), std::move(inputs_[i]));
  }
  props.set_any("program", std::forward<ProgT>(prog));
  return PluginManager::instance()->create_object(props, variant, type);
}

ref<Object> MtlxExprCompiler::CreateExprTexture(std::string_view variant) {
  mtlx::RegisterMaterialXExprTexture();
  return CreatePlugin("materialx_expr", ObjectType::Texture, variant,
                      std::move(val_prog_));
}

ref<Object> MtlxExprCompiler::CreateBsdf(std::string_view variant) {
  mtlx::RegisterMaterialXBsdf();
  AppendPendingRegions();
  mtlx::MtlxBsdfProgram bsdf_prog;
  bsdf_prog.values = std::move(val_prog_);
  bsdf_prog.closures = std::move(closures_);
  bsdf_prog.opacity_root = opacity_root_;
  bsdf_prog.thin_walled = thin_walled_;
  ApplyCompaction(mtlx::OptimizeBsdfProgram(bsdf_prog));
  return CreatePlugin("materialx_bsdf", ObjectType::BSDF, variant,
                      std::move(bsdf_prog));
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
    ValueId radiance = compiler.CompileEdf(shader_node, output_name);
    if (!compiler.IsZero(radiance)) {
      compiler.FinalizeExpr(radiance);
      out.radiance_texture = compiler.CreateExprTexture(variant);
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
  MtlxExprCompiler::DisplacementVal disp =
      compiler.CompileDisplacement(shader_node, output_name);
  if (compiler.IsZero(disp.val)) return {};
  compiler.FinalizeExpr(disp.val);
  bool requires_uv = compiler.ProgramUsesUv();
  return {compiler.CreateExprTexture(variant), requires_uv, disp.is_vector,
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
