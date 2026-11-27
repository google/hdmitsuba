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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/types/span.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <pxr/base/tf/diagnostic.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/materialx/compiler_internal.h"
#include "hdmitsuba/materialx/ir_builder.h"
#include "hdmitsuba/materialx/optimizer.h"
#include "hdmitsuba/materialx/program.h"
#include "hdmitsuba/materialx/translator.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using mitsuba::Object;
using mitsuba::ObjectType;
using mitsuba::PluginManager;
using mitsuba::Properties;
using mitsuba::ref;
using namespace mtlx_detail;

using LookupTextureFn = std::function<const CachedTexture*(const TextureKey&)>;

struct SymbolicExpr {
  mtlx::MtlxValueProgram prog;
  // Empty when `prog` has no live images and is already optimized.
  std::vector<SymbolicImage> images;
  std::vector<std::pair<std::string, int>> attr_requests;
};

struct SymbolicBsdf {
  mtlx::MtlxBsdfProgram prog;
  // Empty when `prog` has no live images and is already optimized.
  std::vector<SymbolicImage> images;
  std::vector<std::pair<std::string, int>> attr_requests;
};

struct SymbolicDisplacement {
  SymbolicExpr expr;
  bool is_vector = false;
};

// --- Shader frontends (EDF, Opacity, BSDF closures, Displacement) ---

ValueId LowerEdf(LowerCtx& ctx, const ScopedNode& node,
                 absl::flat_hash_map<NodeKey, ValueId>& cache) {
  mtlx::IrBuilder& b = *ctx.b;
  if (!node) return b.Zero();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = cache.find(key); it != cache.end()) return it->second;

  auto in = BindInputs(ctx, node);
  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  ValueId res = b.Zero();
  if (cat == "surface" || cat == "surface_unlit") {
    if (cat == "surface") {
      res = LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "edf"), cache);
    } else {
      ValueId em = in("emission", b.One());
      if (!b.IsZero(em)) {
        em = b.Mul(em, b.Sub(b.One(), Clip01(ctx, in("transmission"))));
      }
      if (!b.IsZero(em)) res = b.Mul(em, in("emission_color", b.One()));
    }
    if (!b.IsZero(res)) res = b.Mul(res, Clip01(ctx, in("opacity", b.One())));
  } else if (cat == "uniform_edf" || cat == "conical_edf" ||
             cat == "measured_edf") {
    if (cat == "measured_edf" &&
        !GetNodeInputString(ctx.resolver, node, "file").empty()) {
      WarnUnsupported(
          ctx,
          "Unsupported MaterialX 'measured_edf' IES profile; approximating as "
          "uniform_edf.");
    } else if (cat != "uniform_edf") {
      TF_DEBUG(HDMITSUBA_MATERIAL)
          .Msg("Approximating MaterialX '%s' as uniform_edf.\n", cat.c_str());
    }
    res = in("color", b.One());
  } else if (cat == "generalized_schlick_edf") {
    ValueId base =
        LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "base"), cache);
    if (!b.IsZero(base)) {
      ValueId c0 = in("color0", b.One()), c90 = in("color90", b.One());
      // Weight `base` by the cosine-weighted hemispherical average of
      // `mix(c0, c90, (1 - cos)^exp)`, which equals
      // `mix(c0, c90, 2 / ((exp + 1) * (exp + 2)))`.
      ValueId w = c0;
      if (c0 != c90) {
        ValueId exp =
            b.Emit(mtlx::Op::Max, {in("exponent", b.Const(5.0)), b.Zero()});
        w = b.Mix(c90, c0,
                  b.Div(b.Const(2.0),
                        b.Mul(b.Add(exp, b.One()), b.Add(exp, b.Const(2.0)))));
      }
      res = b.Mul(base, w);
    }
  } else if (cat == "add") {
    res = b.Add(
        LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "in1"), cache),
        LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "in2"), cache));
  } else if (cat == "multiply") {
    ValueId base =
        LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "in1"), cache);
    if (!b.IsZero(base)) res = b.Mul(base, in("in2", b.One()));
  } else if (cat == "mix") {
    res = LowerMixVal(
        ctx, in("mix"),
        [&] {
          return LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "fg"),
                          cache);
        },
        [&] {
          return LowerEdf(ctx, ResolveConnected(ctx.resolver, node, "bg"),
                          cache);
        });
  }
  cache[key] = res;
  return res;
}

struct BsdfLowerState {
  std::vector<mtlx::ClosureNode> closures;
  std::vector<bool> closure_opaque;
  int32_t opaque_blocker = -1;
  bool thin_walled = false;
  absl::flat_hash_map<NodeKey, ValueId> opacity_val;
  absl::flat_hash_map<NodeKey, int32_t> bsdf_closure;
};

ValueId LowerOpacity(LowerCtx& ctx, BsdfLowerState& st,
                     const ScopedNode& node) {
  mtlx::IrBuilder& b = *ctx.b;
  if (!node) return b.One();
  NodeKey key{node.frame, node.node.get()};
  if (auto it = st.opacity_val.find(key); it != st.opacity_val.end()) {
    return it->second;
  }
  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  ValueId res = b.One();
  if (cat == "surface" || cat == "surface_unlit") {
    res = Clip01(ctx, LowerInputExpr(ctx, node, "opacity", b.One()));
  } else if (cat == "mix" &&
             node.node->getType() == mx::SURFACE_SHADER_TYPE_STRING) {
    res = LowerMixVal(
        ctx, Clip01(ctx, LowerInputExpr(ctx, node, "mix")),
        [&] {
          return LowerOpacity(ctx, st,
                              ResolveConnected(ctx.resolver, node, "fg"));
        },
        [&] {
          return LowerOpacity(ctx, st,
                              ResolveConnected(ctx.resolver, node, "bg"));
        });
  }
  if (!b.IsScalar(res)) {
    res = b.Emit(mtlx::Op::Dot, {res, b.Const(1.0 / 3.0)}, 3);
  }
  st.opacity_val[key] = res;
  return res;
}

bool IsClosureOpaque(const BsdfLowerState& st, int32_t idx) {
  return idx >= 0 && st.closure_opaque[idx];
}

int32_t EmitClosure(BsdfLowerState& st, mtlx::ClosureNode node) {
  bool opaque = true;
  switch (node.op) {
    case mtlx::ClosureOp::Null:
      opaque = false;
      break;
    case mtlx::ClosureOp::Layer:
    case mtlx::ClosureOp::Add:
      opaque = IsClosureOpaque(st, node.comb.first) ||
               IsClosureOpaque(st, node.comb.second);
      break;
    case mtlx::ClosureOp::Mix:
      opaque = IsClosureOpaque(st, node.comb.first) &&
               IsClosureOpaque(st, node.comb.second);
      break;
    case mtlx::ClosureOp::Multiply:
      opaque = IsClosureOpaque(st, node.comb.first);
      break;
    default:
      opaque = !mtlx::LeafHasLayerThroughput(node);
      break;
  }
  st.closures.push_back(node);
  st.closure_opaque.push_back(opaque);
  return static_cast<int32_t>(st.closures.size()) - 1;
}

int32_t EmitOpaqueBlocker(BsdfLowerState& st) {
  // Opaque and transmissive leaves still block `layer` bases (`throughput = 0`)
  // at `weight == 0`, without compiling dead inputs or emitting an active leaf.
  if (st.opaque_blocker < 0) {
    st.opaque_blocker = EmitClosure(st, mtlx::ClosureNode::MakeOpaqueBlocker());
  }
  return st.opaque_blocker;
}

void MarkTwoSidedDielectrics(BsdfLowerState& st, size_t start) {
  for (size_t i = start; i < st.closures.size(); ++i) {
    const mtlx::ClosureNode& c = st.closures[i];
    if ((c.op == mtlx::ClosureOp::Dielectric ||
         c.op == mtlx::ClosureOp::GeneralizedSchlick) &&
        (c.microfacet.mode & mtlx::kScatterModeMask) != mtlx::kScatterR) {
      return;
    }
  }
  for (size_t i = start; i < st.closures.size(); ++i) {
    mtlx::ClosureNode& c = st.closures[i];
    if (c.op == mtlx::ClosureOp::Dielectric) {
      c.microfacet.mode |= mtlx::kScatterTwoSided;
    }
  }
}

ValueId FrameValue(LowerCtx& ctx, const ScopedNode& node,
                   const std::string& name, ValueId def) {
  ValueId e = LowerInputExpr(ctx, node, name, def);
  return e == def ? kInvalidValue : e;
}

int32_t LowerDiffuseClosure(LowerCtx& ctx, BsdfLowerState& st,
                            const ScopedNode& node, const std::string& cat) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  ValueId w = in("weight", b.One());
  if (b.IsZero(w)) return cat == "sheen_bsdf" ? -1 : EmitOpaqueBlocker(st);
  ValueId color = in("color", b.One());
  ValueId rough = cat == "translucent_bsdf" ? kInvalidValue : in("roughness");
  ValueId normal = FrameValue(ctx, node, "normal", NWorld(ctx));
  if (cat == "translucent_bsdf") {
    return EmitClosure(st,
                       mtlx::ClosureNode::MakeTranslucent(w, color, normal));
  }
  if (cat == "burley_diffuse_bsdf") {
    return EmitClosure(
        st, mtlx::ClosureNode::MakeBurleyDiffuse(w, color, rough, normal));
  }
  if (cat == "sheen_bsdf") {
    uint16_t mode = GetNodeInputString(ctx.resolver, node, "mode") == "zeltner"
                        ? mtlx::kSheenZeltner
                        : mtlx::kSheenContyKulla;
    return EmitClosure(
        st, mtlx::ClosureNode::MakeSheen(w, color, rough, mode, normal));
  }
  uint16_t energy_comp =
      (cat == "oren_nayar_diffuse_bsdf" && !b.IsZero(in("energy_compensation")))
          ? 1
          : 0;
  return EmitClosure(
      st, mtlx::ClosureNode::MakeDiffuse(w, color, rough, energy_comp, normal));
}

int32_t LowerMicrofacetClosure(LowerCtx& ctx, BsdfLowerState& st,
                               const ScopedNode& node, const std::string& cat) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  ValueId w = in("weight", b.One());
  bool is_cond = cat == "conductor_bsdf";
  uint16_t sm = mtlx::kScatterR;
  if (!is_cond) {
    std::string sm_str = GetNodeInputString(ctx.resolver, node, "scatter_mode");
    sm = sm_str == "T"    ? mtlx::kScatterT
         : sm_str == "RT" ? mtlx::kScatterRT
                          : mtlx::kScatterR;
  }
  if (b.IsZero(w)) {
    return (!is_cond && sm == mtlx::kScatterR) ? -1 : EmitOpaqueBlocker(st);
  }
  ValueId tf_t = kInvalidValue, tf_ior = kInvalidValue;
  if (sm != mtlx::kScatterT) {
    if (ValueId thickness = in("thinfilm_thickness"); !b.IsZero(thickness)) {
      tf_t = thickness;
      tf_ior = in("thinfilm_ior", b.Const(1.5));
    }
  }
  uint16_t mode = static_cast<uint16_t>(
      sm |
      (!b.IsZero(in("retroreflective")) ? mtlx::kScatterRetroreflective : 0) |
      ((is_cond || (cat == "generalized_schlick_bsdf" && sm == mtlx::kScatterR))
           ? mtlx::kScatterTwoSided
           : 0));
  ValueId rough = in("roughness");
  ValueId normal = FrameValue(ctx, node, "normal", NWorld(ctx));
  ValueId tangent = FrameValue(ctx, node, "tangent", TWorld(ctx));
  if (cat == "dielectric_bsdf") {
    return EmitClosure(
        st, mtlx::ClosureNode::MakeMicrofacet(
                mtlx::ClosureOp::Dielectric, w, rough, mode, normal, tangent,
                tf_t, tf_ior, in("tint", b.One()), in("ior", b.One())));
  }
  if (is_cond) {
    return EmitClosure(
        st, mtlx::ClosureNode::MakeMicrofacet(
                mtlx::ClosureOp::Conductor, w, rough, mode, normal, tangent,
                tf_t, tf_ior, in("ior", b.One()), in("extinction")));
  }
  return EmitClosure(
      st,
      mtlx::ClosureNode::MakeMicrofacet(
          mtlx::ClosureOp::GeneralizedSchlick, w, rough, mode, normal, tangent,
          tf_t, tf_ior, in("color0", b.One()), in("color82", b.One()),
          in("color90", b.One()), in("exponent", b.Const(5.0))));
}

int32_t LowerChiangHairClosure(LowerCtx& ctx, BsdfLowerState& st,
                               const ScopedNode& node) {
  auto in = BindInputs(ctx, node);
  mtlx::IrBuilder& b = *ctx.b;
  return EmitClosure(
      st, mtlx::ClosureNode::MakeChiangHair(
              in("tint_R", b.One()), in("tint_TT", b.One()),
              in("tint_TRT", b.One()), in("roughness_R", b.Const(0.1)),
              in("roughness_TT", b.Const(0.05)),
              in("roughness_TRT", b.Const(0.2)), in("ior", b.Const(1.55)),
              in("absorption_coefficient"), in("cuticle_angle", b.Const(0.5)),
              FrameValue(ctx, node, "normal", NWorld(ctx)),
              FrameValue(ctx, node, "curve_direction", TWorld(ctx))));
}

void CheckVdfWarning(LowerCtx& ctx, const ScopedNode& node,
                     const std::string& cat) {
  if (cat == "anisotropic_vdf" || cat == "absorption_vdf") {
    // Nodes/opaques emitted here are unreferenced by closures and pruned by
    // `OptimizeBsdfProgram` and `CollectLiveImages`.
    if (!ctx.b->IsZero(LowerInputExpr(ctx, node, "absorption")) ||
        (cat == "anisotropic_vdf" &&
         !ctx.b->IsZero(LowerInputExpr(ctx, node, "scattering")))) {
      WarnUnsupported(ctx, "Unsupported MaterialX VDF closure '" + cat +
                               "'; ignoring volume absorption/scattering.");
    }
  } else {
    WarnUnsupported(ctx, "Unsupported MaterialX VDF closure '" + cat +
                             "'; ignoring volume.");
  }
}

int32_t LowerBsdfClosure(LowerCtx& ctx, BsdfLowerState& st,
                         const ScopedNode& node) {
  if (!node) return -1;
  NodeKey key{node.frame, node.node.get()};
  if (auto it = st.bsdf_closure.find(key); it != st.bsdf_closure.end()) {
    return it->second;
  }

  mtlx::IrBuilder& b = *ctx.b;
  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  auto is_blocker = [&](int32_t c) {
    return c >= 0 && st.closures[c].op == mtlx::ClosureOp::OpaqueBlocker;
  };
  auto scale = [&](int32_t c, ValueId w) {
    return (c < 0 || b.IsOne(w) || is_blocker(c))
               ? c
               : EmitClosure(st, mtlx::ClosureNode::MakeMultiply(c, w));
  };

  int32_t idx = -1;
  if (cat == "oren_nayar_diffuse_bsdf" || cat == "subsurface_bsdf" ||
      cat == "burley_diffuse_bsdf" || cat == "sheen_bsdf" ||
      cat == "translucent_bsdf") {
    idx = LowerDiffuseClosure(ctx, st, node, cat);
  } else if (cat == "dielectric_bsdf" || cat == "conductor_bsdf" ||
             cat == "generalized_schlick_bsdf") {
    idx = LowerMicrofacetClosure(ctx, st, node, cat);
  } else if (cat == "chiang_hair_bsdf") {
    idx = LowerChiangHairClosure(ctx, st, node);
  } else if (node.node->getType() == mx::VDF_TYPE_STRING) {
    CheckVdfWarning(ctx, node, cat);
  } else if (cat == "layer" || cat == "add") {
    bool is_layer = cat == "layer";
    int32_t c0 = LowerBsdfClosure(
        ctx, st,
        ResolveConnected(ctx.resolver, node, is_layer ? "top" : "in1"));
    // Prune occluded BSDF bases (and VDF bases under a zero-weight blocker),
    // while still visiting active `layer_vdf` bases for VDF warnings.
    if (is_layer && IsClosureOpaque(st, c0) &&
        (is_blocker(c0) ||
         InputType(ctx.resolver, node.node, "base") != mx::VDF_TYPE_STRING)) {
      idx = c0;
    } else if (int32_t c1 = LowerBsdfClosure(
                   ctx, st,
                   ResolveConnected(ctx.resolver, node,
                                    is_layer ? "base" : "in2"));
               c0 < 0 || c1 < 0) {
      idx = c0 < 0 ? c1 : c0;
    } else {
      idx = EmitClosure(st, is_layer ? mtlx::ClosureNode::MakeLayer(c0, c1)
                                     : mtlx::ClosureNode::MakeAdd(c0, c1));
    }
  } else if (cat == "mix") {
    ValueId mix_val = LowerInputExpr(ctx, node, "mix");
    ValueId mix_sat = Clip01(ctx, mix_val);
    if (b.IsZero(mix_sat)) {
      idx =
          LowerBsdfClosure(ctx, st, ResolveConnected(ctx.resolver, node, "bg"));
    } else if (b.IsOne(mix_sat)) {
      idx =
          LowerBsdfClosure(ctx, st, ResolveConnected(ctx.resolver, node, "fg"));
    } else {
      int32_t fg_idx =
          LowerBsdfClosure(ctx, st, ResolveConnected(ctx.resolver, node, "fg"));
      int32_t bg_idx =
          LowerBsdfClosure(ctx, st, ResolveConnected(ctx.resolver, node, "bg"));
      if (fg_idx == bg_idx) {
        idx = fg_idx;
      } else {
        if (fg_idx < 0) fg_idx = EmitClosure(st, mtlx::ClosureNode::MakeNull());
        if (bg_idx < 0) bg_idx = EmitClosure(st, mtlx::ClosureNode::MakeNull());
        idx = EmitClosure(st,
                          mtlx::ClosureNode::MakeMix(fg_idx, bg_idx, mix_val));
      }
    }
  } else if (cat == "multiply") {
    int32_t c =
        LowerBsdfClosure(ctx, st, ResolveConnected(ctx.resolver, node, "in1"));
    idx = (c < 0 || is_blocker(c))
              ? c
              : scale(c, LowerInputExpr(ctx, node, "in2", b.One()));
  } else if (cat == "surface") {
    if (ValueId op = LowerOpacity(ctx, st, node); !b.IsZero(op)) {
      if (!b.IsZero(LowerInputExpr(ctx, node, "thin_walled"))) {
        st.thin_walled = true;
      }
      // Scope BSDF deduplication to this surface so `MarkTwoSidedDielectrics`
      // cannot mutate a closure shared with another mixed surface.
      auto saved = std::exchange(st.bsdf_closure, {});
      size_t start = st.closures.size();
      int32_t bsdf_idx = LowerBsdfClosure(
          ctx, st, ResolveConnected(ctx.resolver, node, "bsdf"));
      MarkTwoSidedDielectrics(st, start);
      st.bsdf_closure = std::move(saved);
      idx = scale(bsdf_idx, op);
    }
  } else if (cat == "surface_unlit") {
    // Unlit surfaces emit radiance only and have a null BSDF.
  } else if (node.node->getType() == mx::BSDF_TYPE_STRING &&
             InputType(ctx.resolver, node.node, "weight") == "float" &&
             b.IsZero(LowerInputExpr(ctx, node, "weight", b.One()))) {
    // Skip inactive unsupported lobes (e.g. zero-weight lobes in
    // `standard_surface`) without warning.
  } else {
    WarnUnsupported(
        ctx, "Unsupported MaterialX closure '" + cat + "'; using a null BSDF.");
  }

  st.bsdf_closure[key] = idx;
  return idx;
}

struct DisplacementVal {
  ValueId val = kInvalidValue;
  bool is_vector = false;
};

DisplacementVal LowerDisplacement(
    LowerCtx& ctx, const ScopedNode& node,
    absl::flat_hash_map<NodeKey, DisplacementVal>& cache) {
  mtlx::IrBuilder& b = *ctx.b;
  if (!node) return {b.Zero(), false};
  NodeKey key{node.frame, node.node.get()};
  if (auto it = cache.find(key); it != cache.end()) return it->second;

  const std::string& cat = NodeCategory(ctx.resolver, node.node);
  DisplacementVal res{b.Zero(), false};
  if (cat == "displacement") {
    ValueId scale = LowerInputExpr(ctx, node, "scale", b.One());
    if (!b.IsZero(scale)) {
      ValueId disp_in = LowerInputExpr(ctx, node, "displacement");
      if (!b.IsZero(disp_in)) {
        bool is_vec =
            InputType(ctx.resolver, node.node, "displacement") == "vector3";
        ValueId disp = b.Mul(disp_in, scale);
        if (!b.IsZero(disp)) {
          res = {is_vec ? disp : b.Chan(disp, 0), is_vec};
        }
      }
    }
  } else if (cat == "mix") {
    auto disp_in = [&](const std::string& name) {
      return LowerDisplacement(
          ctx, ResolveConnected(ctx.resolver, node, name, "displacementshader"),
          cache);
    };
    ValueId m = LowerInputExpr(ctx, node, "mix");
    if (b.IsZero(m)) {
      res = disp_in("bg");
    } else if (b.IsOne(m)) {
      res = disp_in("fg");
    } else {
      DisplacementVal fg = disp_in("fg"), bg = disp_in("bg");
      if (!b.IsZero(fg.val) || !b.IsZero(bg.val)) {
        bool is_vec = fg.is_vector || bg.is_vector;
        auto promote = [&](const DisplacementVal& d) {
          return (!is_vec || d.is_vector) ? d.val
                                          : b.Rgb(b.Zero(), b.Zero(), d.val);
        };
        res = {b.Mix(promote(fg), promote(bg), m), is_vec};
      }
    }
  } else {
    WarnUnsupported(ctx, "Unsupported MaterialX displacement shader '" + cat +
                             "'; ignoring displacement.");
  }
  cache[key] = res;
  return res;
}

// --- Live-node reachability, texture collection, and symbolic replay ---

TextureKey MakeAlphaKey(TextureKey key) {
  key.channel = 3;
  key.color_space.clear();
  return key;
}

bool CollectLiveImages(const mtlx::MtlxValueProgram& prog,
                       const mtlx::MtlxBsdfProgram* bsdf,
                       absl::Span<const SymbolicImage> images,
                       const std::function<void(const TextureKey&)>& callback) {
  if (images.empty()) return false;
  mtlx::LiveMask live = mtlx::ComputeLiveMask(prog, bsdf);
  std::vector<uint8_t> live_img(images.size(), 0);
  bool any_live = false;
  for (size_t i = 0; i < prog.nodes.size(); ++i) {
    if (live.node[i] && prog.nodes[i].op == mtlx::Op::SampleInput) {
      live_img[prog.nodes[i].imm] = 1;
      any_live = true;
    }
  }
  for (size_t idx = 0; idx < images.size(); ++idx) {
    if (!live_img[idx]) continue;
    const SymbolicImage& img = images[idx];
    callback(img.color_key);
    if (img.has_alpha) callback(MakeAlphaKey(img.color_key));
  }
  return any_live;
}

template <typename Vec>
void CompactVec(Vec& vec, absl::Span<const uint16_t> live) {
  for (size_t i = 0; i < live.size(); ++i) {
    if (i != live[i]) vec[i] = std::move(vec[live[i]]);
  }
  vec.resize(live.size());
}

ValueId ResolveSymbolicImage(mtlx::IrBuilder& b, const SymbolicImage& img,
                             ValueId tc, ValueId def_val,
                             const LookupTextureFn& lookup_texture,
                             std::vector<ref<Object>>& inputs) {
  const CachedTexture* cached_color = lookup_texture(img.color_key);
  if (!cached_color || !cached_color->texture) return def_val;
  const CachedTexture* cached_alpha = nullptr;
  if (img.has_alpha && cached_color->has_alpha) {
    cached_alpha = lookup_texture(MakeAlphaKey(img.color_key));
    if (!cached_alpha || !cached_alpha->texture) return def_val;
  }

  bool has_oob = IsOobMode(img.u_mode) || IsOobMode(img.v_mode);
  bool mixed_wrap = img.u_mode != img.v_mode || has_oob;
  ValueId u = kInvalidValue, v = kInvalidValue;
  if (!img.default_uv) {
    auto wrap_axis = [&](ValueId c, AddressMode mode) {
      if (!mixed_wrap || mode == AddressMode::Clamp || IsOobMode(mode)) {
        return c;
      }
      if (mode == AddressMode::Mirror) {
        ValueId f = b.Emit(mtlx::Op::Fract, {b.Mul(c, b.Const(0.5))});
        return b.Sub(b.One(), b.Emit(mtlx::Op::Abs,
                                     {b.Sub(b.Mul(f, b.Const(2.0)), b.One())}));
      }
      return b.Emit(mtlx::Op::Fract, {c});
    };
    u = wrap_axis(b.Chan(tc, 0), img.u_mode);
    v = b.Sub(b.One(), wrap_axis(b.Chan(tc, 1), img.v_mode));
  }

  auto sample_tex = [&](const CachedTexture& cached,
                        uint8_t out_ch = 0) -> ValueId {
    auto it = std::find(inputs.begin(), inputs.end(), cached.texture);
    if (it == inputs.end()) {
      if (inputs.size() >= mtlx::kMaxImm) {
        throw std::runtime_error("materialx_expr: too many texture inputs");
      }
      it = inputs.insert(inputs.end(), cached.texture);
    }
    uint16_t idx = static_cast<uint16_t>(it - inputs.begin());
    uint8_t read_ch = cached.channel_count == 1 ? 1 : 3;
    uint8_t ch = out_ch ? out_ch : read_ch;
    if (img.default_uv) {
      return b.Emit(mtlx::Op::SampleInput, {}, idx, ch, read_ch);
    }
    return b.Emit(mtlx::Op::SampleInputUV, {u, v}, idx, ch, read_ch);
  };

  uint8_t color_ch =
      (img.is_vector2 && cached_color->channel_count != 1) ? 2 : 0;
  ValueId result = sample_tex(*cached_color, color_ch);
  if (img.has_alpha) {
    result = b.Vec4(result, cached_alpha ? sample_tex(*cached_alpha) : b.One());
  }
  if (has_oob) {
    auto oob_axis = [&](ValueId c) {
      return b.Add(b.Emit(mtlx::Op::Lt, {c, b.Zero()}),
                   b.Emit(mtlx::Op::Lt, {b.One(), c}));
    };
    auto oob_val = [&](AddressMode mode) {
      return mode == AddressMode::Black ? b.Zero() : def_val;
    };
    if (img.u_mode == img.v_mode) {
      result = b.Select(b.Add(oob_axis(b.Chan(tc, 0)), oob_axis(b.Chan(tc, 1))),
                        oob_val(img.u_mode), result);
    } else {
      if (IsOobMode(img.u_mode)) {
        result = b.Select(oob_axis(b.Chan(tc, 0)), oob_val(img.u_mode), result);
      }
      if (IsOobMode(img.v_mode)) {
        result = b.Select(oob_axis(b.Chan(tc, 1)), oob_val(img.v_mode), result);
      }
    }
  }
  return result;
}

void ReplayValueProgram(const mtlx::MtlxValueProgram& src,
                        const mtlx::MtlxBsdfProgram* bsdf_src,
                        absl::Span<const SymbolicImage> images,
                        const LookupTextureFn& lookup_texture,
                        mtlx::MtlxValueProgram& out,
                        std::vector<ref<Object>>& inputs,
                        std::vector<ValueId>& remap) {
  mtlx::LiveMask live = mtlx::ComputeLiveMask(src, bsdf_src);
  out.opaques = src.opaques;
  out.attr_names = src.attr_names;
  remap.assign(src.nodes.size(), kInvalidValue);
  std::vector<uint16_t> region_remap(src.regions.size(),
                                     static_cast<uint16_t>(mtlx::kMaxImm));
  std::vector<PendingRegion> new_regions;

  auto replay_range = [&](mtlx::IrBuilder& b, uint32_t begin, uint32_t end) {
    for (uint32_t i = begin; i < end; ++i) {
      if (!live.node[i]) continue;
      const mtlx::Node& n = src.nodes[i];
      if (n.op == mtlx::Op::Const) {
        remap[i] = b.Const(src.consts[n.imm]);
      } else if (n.op == mtlx::Op::Opaque) {
        remap[i] = b.OpaqueRef(n.imm);
      } else if (n.op == mtlx::Op::SampleInput) {
        const SymbolicImage& img = images[n.imm];
        ValueId tc = img.default_uv ? kInvalidValue : remap[n.args[0]];
        ValueId def_val = remap[img.default_uv ? n.args[0] : n.args[1]];
        remap[i] =
            ResolveSymbolicImage(b, img, tc, def_val, lookup_texture, inputs);
      } else if (n.op == mtlx::Op::HeightToNormal) {
        uint16_t new_r = region_remap[n.imm];
        remap[i] = new_r == mtlx::kMaxImm ? b.ConstRgb(0.5, 0.5, 1.0)
                                          : b.Emit(mtlx::Op::HeightToNormal,
                                                   {remap[n.args[0]]}, new_r);
      } else {
        absl::InlinedVector<ValueId, 8> args;
        uint8_t orig_max_ch = 1, new_max_ch = 1;
        for (ValueId a : mtlx::NodeArgs(n, src.extra_args)) {
          orig_max_ch = std::max(orig_max_ch, src.nodes[a].channels);
          ValueId ra = remap[a];
          new_max_ch = std::max(new_max_ch, b.Channels(ra));
          args.push_back(ra);
        }
        uint8_t explicit_ch =
            mtlx::ElementwiseArity(n.op) == 0
                ? n.channels
                : (n.channels < orig_max_ch ? std::min(n.channels, new_max_ch)
                                            : 0);
        remap[i] = b.Emit(n.op, args, n.imm, explicit_ch, n.flags);
      }
    }
  };

  // Regions are stored bottom-up (nested `HeightToNormal` references `imm <
  // k`).
  for (size_t k = 0; k < src.regions.size(); ++k) {
    if (!live.region[k]) continue;
    const mtlx::Region& r = src.regions[k];
    PendingRegion reg;
    mtlx::IrBuilder rb(reg.nodes, reg.extra_args, out.consts, out.opaques);
    replay_range(rb, r.begin, r.end);
    ValueId h_orig = (r.has_texcoord && src.nodes[r.root].op == mtlx::Op::Rgb)
                         ? src.nodes[r.root].args[0]
                         : r.root;
    if (rb.IsVarying(remap[h_orig])) {
      reg.root = remap[r.root];
      reg.has_texcoord = r.has_texcoord;
      region_remap[k] = static_cast<uint16_t>(new_regions.size());
      new_regions.push_back(std::move(reg));
    }
  }

  mtlx::IrBuilder mb(out);
  replay_range(mb, 0, src.main_end);
  AppendPendingRegions(out, new_regions);
}

bool ProgramUsesUv(const mtlx::MtlxValueProgram& prog) {
  return std::any_of(
      prog.nodes.begin(), prog.nodes.end(), [&](const mtlx::Node& n) {
        return n.op == mtlx::Op::GeomUV || n.op == mtlx::Op::SampleInput ||
               (n.op == mtlx::Op::HeightToNormal &&
                !prog.regions[n.imm].has_texcoord);
      });
}

template <typename ProgT>
ref<Object> CreatePlugin(std::string_view name, ObjectType type,
                         std::string_view variant, ProgT&& prog,
                         std::vector<ref<Object>>& inputs) {
  Properties props(name);
  for (size_t i = 0; i < inputs.size(); ++i) {
    props.set("in" + std::to_string(i), std::move(inputs[i]));
  }
  props.set_any("program", std::forward<ProgT>(prog));
  return PluginManager::instance()->create_object(props, variant, type);
}

struct ResolvedExpr {
  mtlx::MtlxValueProgram prog;
  std::vector<ref<Object>> inputs;
  std::vector<std::pair<std::string, int>> attr_requests;
};

std::optional<ResolvedExpr> ResolveExpr(const SymbolicExpr& sym,
                                        const LookupTextureFn& lookup_texture) {
  if (sym.images.empty()) {
    return ResolvedExpr{sym.prog, {}, sym.attr_requests};
  }
  ResolvedExpr out;
  std::vector<ValueId> remap;
  ReplayValueProgram(sym.prog, nullptr, sym.images, lookup_texture, out.prog,
                     out.inputs, remap);
  out.prog.result_root = remap[sym.prog.result_root];
  if (mtlx::IsConstZero(out.prog, out.prog.result_root)) return std::nullopt;
  out.attr_requests = sym.attr_requests;
  mtlx::OptimizeResult res = mtlx::OptimizeValueProgram(out.prog);
  CompactVec(out.inputs, res.live_inputs);
  CompactVec(out.attr_requests, res.live_attrs);
  return out;
}

}  // namespace

struct PreparedMaterialX {
  bool has_displacement = false;
  std::optional<SymbolicBsdf> bsdf;
  std::optional<SymbolicExpr> edf;
  std::optional<SymbolicDisplacement> displacement;
};

bool HasMaterialXSurface(const PreparedMaterialX& prepared) {
  return prepared.bsdf.has_value();
}

bool HasMaterialXDisplacement(const PreparedMaterialX& prepared) {
  return prepared.has_displacement;
}

std::shared_ptr<const PreparedMaterialX> LowerMaterialXShaders(
    const mx::OutputPtr& surface_out, const mx::OutputPtr& displacement_out,
    bool check_surface_displacement,
    const std::function<void(const TextureKey&)>& callback) {
  mx::NodePtr surface_node =
      surface_out ? surface_out->getConnectedNode() : nullptr;
  mx::NodePtr displacement_node =
      displacement_out ? displacement_out->getConnectedNode() : nullptr;
  if (!surface_node && !displacement_node) return nullptr;

  GraphResolver resolver;
  absl::flat_hash_set<std::string> warned;
  mx::DocumentPtr doc =
      (surface_node ? surface_node : displacement_node)->getDocument();

  auto prepared = std::make_shared<PreparedMaterialX>();
  prepared->has_displacement = displacement_node != nullptr;

  // Resolve displacement target before lowering so `surfacematerial`
  // displacement reuses the shared `GraphResolver`.
  ScopedNode disp_root;
  if (surface_node && check_surface_displacement &&
      surface_out->getType() == mx::MATERIAL_TYPE_STRING) {
    ScopedNode surf_disp =
        UnwrapShaderNode(resolver, {surface_node, nullptr},
                         surface_out->getOutputString(), "displacementshader");
    if (surf_disp &&
        surf_disp.node->getType() == mx::DISPLACEMENT_SHADER_TYPE_STRING) {
      disp_root = std::move(surf_disp);
      prepared->has_displacement = true;
    }
  }
  if (!disp_root && displacement_node) {
    disp_root = UnwrapShaderNode(resolver, {displacement_node, nullptr},
                                 displacement_out->getOutputString(),
                                 "displacementshader");
  }

  auto finalize_or_optimize =
      [&](const mtlx::MtlxValueProgram& vprog,
          const mtlx::MtlxBsdfProgram* bprog,
          std::vector<SymbolicImage>& images,
          std::vector<std::pair<std::string, int>>& attrs, auto&& optimize_fn) {
        if (!CollectLiveImages(vprog, bprog, images, callback)) {
          images.clear();
          mtlx::OptimizeResult res = optimize_fn();
          CompactVec(attrs, res.live_attrs);
        }
      };
  auto finalize_expr = [&](LowerCtx& ctx, ValueId root) -> SymbolicExpr {
    ctx.val_prog.result_root = root;
    AppendPendingRegions(ctx.val_prog, ctx.pending_regions);
    SymbolicExpr out{std::move(ctx.val_prog), std::move(ctx.images),
                     std::move(ctx.attr_requests)};
    finalize_or_optimize(out.prog, nullptr, out.images, out.attr_requests,
                         [&] { return mtlx::OptimizeValueProgram(out.prog); });
    return out;
  };

  if (surface_node) {
    ScopedNode surf_root = UnwrapShaderNode(resolver, {surface_node, nullptr},
                                            surface_out->getOutputString());
    // Lower BSDF closure DAG + opacity.
    {
      LowerCtx ctx(doc, resolver, warned);
      BsdfLowerState st;
      ValueId op_val = LowerOpacity(ctx, st, surf_root);
      ValueId opacity_root = ctx.b->IsOne(op_val) ? kInvalidValue : op_val;
      if (ctx.b->IsZero(op_val) || LowerBsdfClosure(ctx, st, surf_root) < 0) {
        st.closures = {mtlx::ClosureNode::MakeNull()};
      } else if (!st.thin_walled &&
                 surf_root.node->getType() != mx::SURFACE_SHADER_TYPE_STRING) {
        MarkTwoSidedDielectrics(st, 0);
      }
      AppendPendingRegions(ctx.val_prog, ctx.pending_regions);
      SymbolicBsdf bsdf;
      bsdf.prog.values = std::move(ctx.val_prog);
      bsdf.prog.closures = std::move(st.closures);
      bsdf.prog.opacity_root = opacity_root;
      bsdf.prog.thin_walled = st.thin_walled;
      bsdf.images = std::move(ctx.images);
      bsdf.attr_requests = std::move(ctx.attr_requests);
      finalize_or_optimize(
          bsdf.prog.values, &bsdf.prog, bsdf.images, bsdf.attr_requests,
          [&] { return mtlx::OptimizeBsdfProgram(bsdf.prog); });
      prepared->bsdf = std::move(bsdf);
    }
    // Lower EDF emission.
    {
      LowerCtx ctx(doc, resolver, warned);
      absl::flat_hash_map<NodeKey, ValueId> edf_cache;
      ValueId radiance = LowerEdf(ctx, surf_root, edf_cache);
      if (!ctx.b->IsZero(radiance)) {
        prepared->edf = finalize_expr(ctx, radiance);
      }
    }
  }

  if (disp_root) {
    LowerCtx ctx(doc, resolver, warned);
    absl::flat_hash_map<NodeKey, DisplacementVal> disp_cache;
    DisplacementVal disp = LowerDisplacement(ctx, disp_root, disp_cache);
    if (!ctx.b->IsZero(disp.val)) {
      prepared->displacement =
          SymbolicDisplacement{finalize_expr(ctx, disp.val), disp.is_vector};
    }
  }

  return prepared;
}

CompiledMaterialXResult InstantiateMaterialXShaders(
    const PreparedMaterialX& prepared, std::string_view variant,
    const LookupTextureFn& lookup_texture) {
  CompiledMaterialXResult out;

  if (prepared.bsdf) {
    mtlx::RegisterMaterialXBsdf();
    mtlx::MtlxBsdfProgram prog;
    std::vector<ref<Object>> inputs;
    std::vector<std::pair<std::string, int>> attrs =
        prepared.bsdf->attr_requests;
    if (prepared.bsdf->images.empty()) {
      prog = prepared.bsdf->prog;
    } else {
      prog.closures = prepared.bsdf->prog.closures;
      prog.thin_walled = prepared.bsdf->prog.thin_walled;
      std::vector<ValueId> remap;
      ReplayValueProgram(prepared.bsdf->prog.values, &prepared.bsdf->prog,
                         prepared.bsdf->images, lookup_texture, prog.values,
                         inputs, remap);
      if (prepared.bsdf->prog.opacity_root != kInvalidValue) {
        prog.opacity_root = remap[prepared.bsdf->prog.opacity_root];
      }
      for (mtlx::ClosureNode& c : prog.closures) {
        mtlx::ForEachClosureValueId(c, [&](ValueId& id) { id = remap[id]; });
      }
      mtlx::OptimizeResult res = mtlx::OptimizeBsdfProgram(prog);
      CompactVec(inputs, res.live_inputs);
      CompactVec(attrs, res.live_attrs);
    }
    out.bsdf = CreatePlugin("materialx_bsdf", ObjectType::BSDF, variant,
                            std::move(prog), inputs);
    out.mesh_attributes.insert(attrs.begin(), attrs.end());
  }

  if (prepared.edf) {
    if (std::optional<ResolvedExpr> expr =
            ResolveExpr(*prepared.edf, lookup_texture)) {
      mtlx::RegisterMaterialXExprTexture();
      out.radiance_texture =
          CreatePlugin("materialx_expr", ObjectType::Texture, variant,
                       std::move(expr->prog), expr->inputs);
      out.mesh_attributes.insert(expr->attr_requests.begin(),
                                 expr->attr_requests.end());
    }
  }

  if (prepared.displacement) {
    if (std::optional<ResolvedExpr> expr =
            ResolveExpr(prepared.displacement->expr, lookup_texture)) {
      mtlx::RegisterMaterialXExprTexture();
      bool requires_uv = ProgramUsesUv(expr->prog);
      out.displacement = {
          CreatePlugin("materialx_expr", ObjectType::Texture, variant,
                       std::move(expr->prog), expr->inputs),
          requires_uv, prepared.displacement->is_vector,
          MeshAttributeRequests(expr->attr_requests.begin(),
                                expr->attr_requests.end())};
    }
  }

  return out;
}

PXR_NAMESPACE_CLOSE_SCOPE
