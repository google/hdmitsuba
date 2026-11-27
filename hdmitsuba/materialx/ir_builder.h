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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/types/span.h>
#include <mitsuba/core/vector.h>

#include "hdmitsuba/materialx/math.h"
#include "hdmitsuba/materialx/program.h"

namespace mitsuba::mtlx {

// Builds flat SSA `Node`s into `(nodes, extra_args)` with O(1) hash-consing
// (GVN/CSE), commutative canonicalization, pure constant folding, and all
// bottom-up algebraic/affine/Clip01 rewrites on construction (analogous to
// `drjit-core/src/op.cpp`). Shared by `compiler.cc` and `optimizer.cc`.
class IrBuilder {
  struct GvnKey {
    Op op;
    uint8_t channels;
    uint8_t flags;
    uint16_t imm;
    absl::InlinedVector<ValueId, 4> args;

    bool operator==(const GvnKey& o) const {
      return op == o.op && channels == o.channels && flags == o.flags &&
             imm == o.imm && args == o.args;
    }

    template <typename H>
    friend H AbslHashValue(H h, const GvnKey& k) {
      return H::combine(std::move(h), k.op, k.channels, k.flags, k.imm, k.args);
    }
  };

  struct AffineForm {
    ValueId x, scale, bias;
  };

 public:
  IrBuilder(std::vector<Node>& nodes, std::vector<ValueId>& extra_args,
            std::vector<double>& consts, std::vector<double>& opaques)
      : nodes_(nodes),
        extra_args_(extra_args),
        consts_(consts),
        opaques_(opaques) {
    if (consts_.size() < 2) consts_ = {0.0, 1.0};
    for (size_t i = 0; i < consts_.size(); ++i) {
      const_map_.try_emplace(ConstKey(consts_[i]), static_cast<uint16_t>(i));
    }
    varying_.reserve(nodes_.size());
    for (const Node& n : nodes_) varying_.push_back(ComputeVarying(n));
  }

  explicit IrBuilder(MtlxValueProgram& prog)
      : IrBuilder(prog.nodes, prog.extra_args, prog.consts, prog.opaques) {}

  void ResetGvn() { gvn_.clear(); }
  ValueId Zero() { return Const(0.0); }
  ValueId One() { return Const(1.0); }

  ValueId Const(double value) {
    if (value == 0.0) value = 0.0;  // Canonicalize -0.0 to +0.0.
    auto [it, inserted] = const_map_.try_emplace(
        ConstKey(value), static_cast<uint16_t>(consts_.size()));
    if (inserted) {
      if (consts_.size() >= kMaxImm) {
        throw std::runtime_error("materialx_expr: too many constants");
      }
      consts_.push_back(value);
    }
    return EmitRaw(Op::Const, {}, it->second, 1, 0);
  }

  ValueId ConstRgb(double r, double g, double b) {
    return (r == g && g == b) ? Const(r) : Rgb(Const(r), Const(g), Const(b));
  }

  uint16_t AddOpaqueValue(double value) {
    if (opaques_.size() >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many opaque values");
    }
    opaques_.push_back(value);
    return static_cast<uint16_t>(opaques_.size() - 1);
  }

  ValueId Opaque(double value) { return OpaqueRef(AddOpaqueValue(value)); }
  ValueId OpaqueRef(uint16_t idx) { return EmitRaw(Op::Opaque, {}, idx, 1, 0); }

  bool IsZero(ValueId id) const { return IsConstZero(nodes_, id); }
  bool IsOne(ValueId id) const { return IsConstOne(nodes_, id); }
  bool IsConstScalar(ValueId id, double* out = nullptr) const {
    return ConstScalar(nodes_, consts_, id, out);
  }
  bool IsVarying(ValueId id) const { return varying_[id] != 0; }
  bool IsClipped01(ValueId id) const {
    return (nodes_[id].flags & kNodeClipped01) != 0;
  }
  bool IsUnitNormal(ValueId id) const {
    return (nodes_[id].flags & kNodeUnitNormal) != 0;
  }

  std::optional<Vector<double, 4>> ConstValue(ValueId id) const {
    const Node& n = nodes_[id];
    double x = 0.0, y = 0.0, z = 0.0;
    if (n.op == Op::Const) return Vector<double, 4>(consts_[n.imm]);
    if (n.op == Op::Vec2 && IsConstScalar(n.args[0], &x) &&
        IsConstScalar(n.args[1], &y)) {
      return Vector<double, 4>(x, y, 0.0, 0.0);
    }
    if (n.op == Op::Rgb && IsConstScalar(n.args[0], &x) &&
        IsConstScalar(n.args[1], &y) && IsConstScalar(n.args[2], &z)) {
      return Vector<double, 4>(x, y, z, 0.0);
    }
    if (n.op == Op::Vec4 && IsConstScalar(n.args[1], &x)) {
      if (auto c3 = ConstValue(n.args[0])) {
        return Vector<double, 4>(c3->x(), c3->y(), c3->z(), x);
      }
    }
    return std::nullopt;
  }

  uint8_t Channels(ValueId id) const { return nodes_[id].channels; }
  bool IsScalar(ValueId id) const { return nodes_[id].channels == 1; }

  // Emits a node directly with commutative canonicalization and GVN
  // deduplication (no algebraic rewrites or constant folding).
  ValueId EmitRaw(Op op, absl::Span<const ValueId> args, uint16_t imm,
                  uint8_t ch, uint8_t flags) {
    GvnKey key{op,
               ch,
               static_cast<uint8_t>(flags & kNodeReadChannelsMask),
               imm,
               {args.begin(), args.end()}};
    if ((IsCommutativeOp(op) && key.args.size() == 2 &&
         key.args[0] > key.args[1]) ||
        (op == Op::Fma && key.args.size() == 3 && key.args[0] > key.args[1])) {
      std::swap(key.args[0], key.args[1]);
    }
    if (auto it = gvn_.find(key); it != gvn_.end()) {
      nodes_[it->second].flags |= flags & ~kNodeReadChannelsMask;
      return it->second;
    }
    if (nodes_.size() >= kMaxImm) {
      throw std::runtime_error("materialx_expr: too many SSA nodes");
    }
    Node n{op, ch, static_cast<uint8_t>(key.args.size()), flags, imm, {}};
    if (key.args.size() <= 3) {
      for (size_t i = 0; i < key.args.size(); ++i) n.args[i] = key.args[i];
    } else {
      if (extra_args_.size() + key.args.size() > kMaxImm) {
        throw std::runtime_error("materialx_expr: too many SSA operands");
      }
      n.args[0] = static_cast<uint16_t>(extra_args_.size());
      extra_args_.insert(extra_args_.end(), key.args.begin(), key.args.end());
    }
    n.flags = ComputeNodeFlags(n);
    ValueId id = static_cast<ValueId>(nodes_.size());
    nodes_.push_back(n);
    varying_.push_back(ComputeVarying(n));
    gvn_.emplace(std::move(key), id);
    return id;
  }

  ValueId Emit(Op op, absl::Span<const ValueId> args = {}, uint16_t imm = 0,
               uint8_t explicit_ch = 0, uint8_t flags = 0) {
    switch (op) {
      case Op::Chan:
        if (IsScalar(args[0])) return args[0];
        if (nodes_[args[0]].op == Op::Vec2 && imm < 2) {
          return Chan(nodes_[args[0]].args[imm], 0);
        }
        if (nodes_[args[0]].op == Op::Rgb && imm < 3) {
          return Chan(nodes_[args[0]].args[imm], 0);
        }
        if (nodes_[args[0]].op == Op::Vec4) {
          return imm == 3 ? Chan(nodes_[args[0]].args[1], 0)
                          : Chan(nodes_[args[0]].args[0], imm);
        }
        break;
      case Op::Dot:
        for (int i = 0; i < 2; ++i) {
          if (IsScalar(args[i])) {
            if (auto cv = ConstValue(args[1 - i])) {
              double sum = cv->x() + cv->y() + (imm >= 3 ? cv->z() : 0.0) +
                           (imm == 4 ? cv->w() : 0.0);
              if (std::abs(sum - 1.0) <= 1e-12) return args[i];
            }
          }
        }
        break;
      case Op::Rgb:
      case Op::Vec2:
        if (IsScalar(args[0]) &&
            std::all_of(args.begin() + 1, args.end(),
                        [&](ValueId a) { return a == args[0]; })) {
          return args[0];
        }
        if (nodes_[args[0]].op == Op::Chan && nodes_[args[0]].imm == 0) {
          ValueId src = nodes_[args[0]].args[0];
          if (Channels(src) == args.size()) {
            bool identity = true;
            for (size_t i = 1; i < args.size(); ++i) {
              identity &= nodes_[args[i]].op == Op::Chan &&
                          nodes_[args[i]].imm == i &&
                          nodes_[args[i]].args[0] == src;
            }
            if (identity) return src;
          }
        }
        break;
      case Op::Vec4:
        if (IsScalar(args[0]) && args[0] == args[1]) return args[0];
        break;
      case Op::Neg:
        if (IsZero(args[0])) return Zero();
        if (nodes_[args[0]].op == Op::Neg) return nodes_[args[0]].args[0];
        break;
      case Op::Add:
        if (IsZero(args[0])) return args[1];
        if (IsZero(args[1])) return args[0];
        if (nodes_[args[1]].op == Op::Neg) {
          return Sub(args[0], nodes_[args[1]].args[0], explicit_ch);
        }
        if (nodes_[args[0]].op == Op::Neg) {
          return Sub(args[1], nodes_[args[0]].args[0], explicit_ch);
        }
        break;
      case Op::Sub:
        if (args[0] == args[1]) return Zero();
        if (IsZero(args[1])) return args[0];
        if (IsZero(args[0])) return Neg(args[1]);
        if (nodes_[args[1]].op == Op::Neg) {
          return Add(args[0], nodes_[args[1]].args[0], explicit_ch);
        }
        break;
      case Op::Mul:
        if (IsZero(args[0]) || IsZero(args[1])) return Zero();
        if (IsOne(args[0])) return args[1];
        if (IsOne(args[1])) return args[0];
        break;
      case Op::Div:
        if (IsZero(args[0]) || IsZero(args[1])) return Zero();
        if (IsOne(args[1])) return args[0];
        break;
      case Op::Pow: {
        double ev = 0.0;
        if (IsConstScalar(args[1], &ev)) {
          if (ev == 0.0) return One();
          if (ev == 1.0) return args[0];
          if (ev == 0.5) return Emit(Op::Sqrt, {args[0]});
          if (ev == 2.0) return Mul(args[0], args[0]);
          if (ev == 3.0) return Mul(Mul(args[0], args[0]), args[0]);
        }
        break;
      }
      case Op::Fma:
        if (IsZero(args[0]) || IsZero(args[1])) return args[2];
        if (IsZero(args[2])) return Mul(args[0], args[1], explicit_ch);
        if (IsOne(args[0])) return Add(args[1], args[2], explicit_ch);
        if (IsOne(args[1])) return Add(args[0], args[2], explicit_ch);
        break;
      case Op::Clip:
        if (IsClipped01(args[0]) && IsZero(args[1]) && IsOne(args[2])) {
          return args[0];
        }
        break;
      case Op::Select: {
        if (IsZero(args[0]) || args[1] == args[2]) return args[2];
        if (IsOne(args[0])) return args[1];
        const Node& cn = nodes_[args[0]];
        if (IsOne(args[1]) && IsZero(args[2]) &&
            (cn.op == Op::Lt || cn.op == Op::Le || cn.op == Op::Eq)) {
          return args[0];
        }
        if (IsZero(args[1]) && IsOne(args[2])) {
          if (cn.op == Op::Lt) return Emit(Op::Le, {cn.args[1], cn.args[0]});
          if (cn.op == Op::Le) return Emit(Op::Lt, {cn.args[1], cn.args[0]});
        }
        break;
      }
      case Op::Mix:
        if (IsZero(args[2]) || args[0] == args[1]) return args[1];
        if (IsOne(args[2])) return args[0];
        break;
      case Op::Lt:
        if (args[0] == args[1]) return Zero();
        break;
      case Op::Le:
      case Op::Eq:
        if (args[0] == args[1]) return One();
        break;
      case Op::Min:
      case Op::Max:
        if (args[0] == args[1]) return args[0];
        break;
      case Op::Rotate2D:
      case Op::Rotate3D:
        if (IsZero(args[1])) return args[0];
        break;
      case Op::Normalize: {
        const Node& an = nodes_[args[0]];
        if ((an.op == Op::Normalize && an.imm == imm) ||
            (imm == 3 && an.op >= Op::GeomNormalWorld &&
             an.op <= Op::GeomIncidentDir)) {
          return args[0];
        }
        break;
      }
      case Op::NormalMap:
        if (IsZero(args[1])) {
          return imm == kNormalMapDefaultFrame ? Emit(Op::GeomNormalWorld)
                                               : args[2];
        }
        if (imm == 0 && nodes_[args[2]].op == Op::GeomNormalWorld &&
            nodes_[args[3]].op == Op::GeomTangentWorld &&
            nodes_[args[4]].op == Op::GeomBitangentWorld) {
          return Emit(Op::NormalMap, {args[0], args[1]},
                      kNormalMapDefaultFrame);
        }
        break;
      case Op::HeightToNormal:
        if (IsZero(args[0])) return ConstRgb(0.5, 0.5, 1.0);
        break;
      default:
        break;
    }

    uint8_t ch = explicit_ch ? explicit_ch : DeduceChannels(op, args, imm);

    // Fold pure ops whose operands are all compile-time constants.
    if (IsPureOp(op) && op != Op::Rgb && op != Op::Vec2 && op != Op::Vec4 &&
        args.size() <= 3) {
      Vector<double, 4> cargs[3], out(0.0);
      bool all_const = true;
      for (size_t i = 0; i < args.size() && all_const; ++i) {
        if (auto cv = ConstValue(args[i])) {
          cargs[i] = *cv;
        } else {
          all_const = false;
        }
      }
      Node tmp{op, ch, static_cast<uint8_t>(args.size()), flags, imm, {}};
      if (all_const && detail::EvalPureOp<double>(tmp, cargs, out)) {
        bool ok = true;
        for (uint8_t c = 0; c < ch; ++c) {
          ok &= std::abs(out[c]) <= std::numeric_limits<float>::max();
        }
        if (ok) {
          if (ch <= 1) return Const(out[0]);
          if (ch == 2) return Vec2(Const(out[0]), Const(out[1]));
          ValueId c3 = ConstRgb(out[0], out[1], out[2]);
          return ch == 3 ? c3 : Vec4(c3, Const(out[3]));
        }
      }
    }

    // Collapse affine ops on a single varying value (`x * scale + bias`) and
    // hoist uniform scale/bias arithmetic out of the varying chain.
    if (auto af = TryDecomposeAffine(op, args, ch)) {
      bool mix_clipped = op == Op::Mix && IsClipped01(args[0]) &&
                         IsClipped01(args[1]) && IsClipped01(args[2]);
      ValueId res = EmitAffine(*af, ch);
      if (mix_clipped) nodes_[res].flags |= kNodeClipped01;
      return res;
    }

    return EmitRaw(op, args, imm, ch, flags);
  }

  ValueId Chan(ValueId v, uint16_t c) { return Emit(Op::Chan, {v}, c); }
  ValueId Neg(ValueId a) { return Emit(Op::Neg, {a}); }
  ValueId Not(ValueId a) { return Emit(Op::Eq, {a, Zero()}); }
  ValueId Add(ValueId a, ValueId b, uint8_t ch = 0) {
    return Emit(Op::Add, {a, b}, 0, ch);
  }
  ValueId Sub(ValueId a, ValueId b, uint8_t ch = 0) {
    return Emit(Op::Sub, {a, b}, 0, ch);
  }
  ValueId Mul(ValueId a, ValueId b, uint8_t ch = 0) {
    return Emit(Op::Mul, {a, b}, 0, ch);
  }
  ValueId Div(ValueId a, ValueId b, uint8_t ch = 0) {
    return Emit(Op::Div, {a, b}, 0, ch);
  }
  ValueId Fma(ValueId a, ValueId b, ValueId c, uint8_t ch = 0) {
    return Emit(Op::Fma, {a, b, c}, 0, ch);
  }
  ValueId Select(ValueId c, ValueId t, ValueId f) {
    return Emit(Op::Select, {c, t, f});
  }
  ValueId Mix(ValueId fg, ValueId bg, ValueId m) {
    return Emit(Op::Mix, {fg, bg, m});
  }
  ValueId Rgb(ValueId r, ValueId g, ValueId b) {
    return Emit(Op::Rgb, {r, g, b});
  }
  ValueId Vec2(ValueId x, ValueId y) { return Emit(Op::Vec2, {x, y}); }
  ValueId Vec4(ValueId xyz, ValueId w) { return Emit(Op::Vec4, {xyz, w}); }

 private:
  static uint64_t ConstKey(double v) {
    if (v == 0.0) v = 0.0;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
  }

  bool IsNegOne(ValueId id) const {
    double v = 0.0;
    return IsConstScalar(id, &v) && v == -1.0;
  }

  uint8_t ComputeVarying(const Node& n) const {
    if (IsSpatiallyVaryingOp(n.op) || n.op == Op::SampleInput) return 1;
    for (ValueId a : NodeArgs(n, extra_args_)) {
      if (IsVarying(a)) return 1;
    }
    return 0;
  }

  uint8_t ComputeNodeFlags(const Node& n) const {
    auto args = NodeArgs(n, extra_args_);
    bool clip = (n.flags & kNodeClipped01) != 0;
    bool unit = (n.flags & kNodeUnitNormal) != 0;
    switch (n.op) {
      case Op::Const:
        clip = consts_[n.imm] >= 0.0 && consts_[n.imm] <= 1.0;
        break;
      case Op::Fract:
      case Op::Lt:
      case Op::Le:
      case Op::Eq:
      case Op::CellNoise:
        clip = true;
        break;
      case Op::Chan:
        clip = IsClipped01(args[0]);
        break;
      case Op::Clip:
      case Op::Select:
        clip = IsClipped01(args[1]) && IsClipped01(args[2]);
        if (n.op == Op::Select) {
          unit = IsUnitNormal(args[1]) && IsUnitNormal(args[2]);
        }
        break;
      case Op::Rgb:
      case Op::Vec2:
      case Op::Vec4:
      case Op::Min:
      case Op::Max:
      case Op::Mul:
      case Op::Mix:
        clip = std::all_of(args.begin(), args.end(),
                           [&](ValueId a) { return IsClipped01(a); });
        break;
      case Op::GeomNormalWorld:
      case Op::GeomTangentWorld:
      case Op::GeomBitangentWorld:
      case Op::GeomIncidentDir:
      case Op::NormalMap:
        unit = true;
        break;
      default:
        break;
    }
    return static_cast<uint8_t>((n.flags & kNodeReadChannelsMask) |
                                (clip ? kNodeClipped01 : 0) |
                                (unit ? kNodeUnitNormal : 0));
  }

  std::optional<AffineForm> TryDecomposeAffine(Op op,
                                               absl::Span<const ValueId> in,
                                               uint8_t out_ch) {
    if (in.empty() || in.size() > 3) return std::nullopt;
    // Copy operands locally before emitting uniform helpers that may grow
    // `nodes_`.
    ValueId a[3] = {};
    uint8_t max_ch = 1;
    for (size_t i = 0; i < in.size(); ++i) {
      a[i] = in[i];
      max_ch = std::max(max_ch, Channels(a[i]));
    }
    if (out_ch != max_ch) return std::nullopt;
    switch (op) {
      case Op::Neg:
        if (IsVarying(a[0])) return AffineForm{a[0], Const(-1.0), Zero()};
        return std::nullopt;
      case Op::Add:
      case Op::Mul: {
        bool v0 = IsVarying(a[0]), v1 = IsVarying(a[1]);
        if (v0 == v1) return std::nullopt;
        ValueId vx = v0 ? a[0] : a[1], u = v0 ? a[1] : a[0];
        return op == Op::Add ? AffineForm{vx, One(), u}
                             : AffineForm{vx, u, Zero()};
      }
      case Op::Sub: {
        bool v0 = IsVarying(a[0]), v1 = IsVarying(a[1]);
        if (v0 && !v1) return AffineForm{a[0], One(), Neg(a[1])};
        if (!v0 && v1) return AffineForm{a[1], Const(-1.0), a[0]};
        return std::nullopt;
      }
      case Op::Div:
        if (IsVarying(a[0]) && !IsVarying(a[1])) {
          return AffineForm{a[0], Div(One(), a[1]), Zero()};
        }
        return std::nullopt;
      case Op::Fma: {
        bool v0 = IsVarying(a[0]), v1 = IsVarying(a[1]);
        if (!IsVarying(a[2]) && v0 != v1) {
          return AffineForm{v0 ? a[0] : a[1], v0 ? a[1] : a[0], a[2]};
        }
        return std::nullopt;
      }
      case Op::Mix:
        if (!IsVarying(a[0]) && !IsVarying(a[1]) && IsVarying(a[2])) {
          return AffineForm{a[2], Sub(a[0], a[1]), a[1]};
        }
        return std::nullopt;
      default:
        return std::nullopt;
    }
  }

  ValueId EmitAffine(AffineForm af, uint8_t ch) {
    if (IsZero(af.scale)) return af.bias;
    if (IsOne(af.scale) && IsZero(af.bias)) return af.x;

    // Collapse chained affine `up(x) * scale + bias` into one varying op.
    Node dn = nodes_[af.x];
    if (dn.arity <= 3) {
      if (auto up = TryDecomposeAffine(
              dn.op, absl::MakeConstSpan(dn.args, dn.arity), dn.channels)) {
        auto ok = [ch](uint8_t c) { return c == 1 || c == ch; };
        if (ok(Channels(af.x)) && ok(Channels(up->x)) &&
            ok(Channels(up->scale)) && ok(Channels(up->bias)) &&
            ok(Channels(af.scale)) && ok(Channels(af.bias))) {
          af.x = up->x;
          af.bias = Fma(up->bias, af.scale, af.bias);
          af.scale = Mul(up->scale, af.scale);
          if (IsZero(af.scale)) return af.bias;
          if (IsOne(af.scale) && IsZero(af.bias)) return af.x;
        }
      }
    }

    if (IsZero(af.bias)) {
      return IsNegOne(af.scale) ? EmitRaw(Op::Neg, {af.x}, 0, ch, 0)
                                : EmitRaw(Op::Mul, {af.x, af.scale}, 0, ch, 0);
    }
    if (IsOne(af.scale)) {
      return nodes_[af.bias].op == Op::Neg
                 ? EmitRaw(Op::Sub, {af.x, nodes_[af.bias].args[0]}, 0, ch, 0)
                 : EmitRaw(Op::Add, {af.x, af.bias}, 0, ch, 0);
    }
    if (IsNegOne(af.scale)) {
      return EmitRaw(Op::Sub, {af.bias, af.x}, 0, ch, 0);
    }
    return EmitRaw(Op::Fma, {af.x, af.scale, af.bias}, 0, ch, 0);
  }

  uint8_t DeduceChannels(Op op, absl::Span<const ValueId> args,
                         uint16_t imm) const {
    if (ElementwiseArity(op) != 0) {
      uint8_t ch = 1;
      for (ValueId a : args) ch = std::max(ch, nodes_[a].channels);
      return ch;
    }
    switch (op) {
      case Op::Const:
      case Op::Opaque:
      case Op::Chan:
      case Op::Dot:
        return 1;
      case Op::GeomUV:
      case Op::Vec2:
      case Op::Rotate2D:
        return 2;
      case Op::Vec4:
        return 4;
      case Op::Normalize:
        return imm == 4 ? 4 : (imm == 2 ? 2 : 3);
      case Op::CellNoise:
      case Op::Noise:
      case Op::Worley:
      case Op::Fractal:
        return NoiseIsVec3(imm) ? 3 : 1;
      case Op::RgbToHsv:
      case Op::HsvToRgb:
        return nodes_[args[0]].channels;
      default:
        return 3;
    }
  }

  std::vector<Node>& nodes_;
  std::vector<ValueId>& extra_args_;
  std::vector<double>& consts_;
  std::vector<double>& opaques_;
  std::vector<uint8_t> varying_;
  absl::flat_hash_map<uint64_t, uint16_t> const_map_;
  absl::flat_hash_map<GvnKey, ValueId> gvn_;
};

}  // namespace mitsuba::mtlx
