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

#if defined(MI_ENABLE_JIT)

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <drjit/array_traverse.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/pxr.h>

#include "hdmitsuba/debug_codes.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace dr = drjit;

template <typename Float, typename Spectrum>
struct JITInputs {
  std::vector<uint32_t> flat_indices;
  std::vector<uint32_t> owned_class_vars;
};

// Gathers all JIT variables from the scene, sensor, and integrator.
template <typename Float, typename Spectrum>
JITInputs<Float, Spectrum> GatherAllJitInputs(
    mitsuba::Scene<Float, Spectrum>* scene,
    mitsuba::Sensor<Float, Spectrum>* sensor,
    mitsuba::Integrator<Float, Spectrum>* integrator) {
  std::vector<uint32_t> indices;
  std::vector<uint32_t> owned_class_vars;
  absl::flat_hash_set<void*> visited_ptrs;

  auto collect_cb = [](void* payload, uint64_t index_combined, const char*,
                       const char*) {
    auto* vec = static_cast<std::vector<uint32_t>*>(payload);
    uint32_t index = (uint32_t)index_combined;
    if (index != 0) {
      vec->push_back(index);
    }
  };

  bool old_traversal = ::jit_flag(::JitFlag::EnableObjectTraversal);
  ::jit_set_flag(::JitFlag::EnableObjectTraversal, true);

  if constexpr (dr::is_jit_v<Float>) {
    ::JitBackend backend = dr::backend_v<Float>;
    auto collect_class_var = [&](void* ptr) {
      if (!ptr || !visited_ptrs.insert(ptr).second) return;
      uint32_t class_var = ::jit_var_class(backend, ptr);
      if (class_var != 0) {
        indices.push_back(class_var);
        owned_class_vars.push_back(class_var);
      }
    };

    if (scene) {
      collect_class_var(scene);
      dr::traverse_1_fn_ro(*scene, &indices, collect_cb);
    }
    if (sensor) {
      collect_class_var(sensor);
      dr::traverse_1_fn_ro(*sensor, &indices, collect_cb);
    }
    if (integrator) {
      collect_class_var(integrator);
      dr::traverse_1_fn_ro(*integrator, &indices, collect_cb);
    }

    const char* variant_name = mitsuba::detail::variant<Float, Spectrum>::name;
    constexpr const char* domains[] = {
        "Scene",
        "ReconstructionFilter",
        "Sensor",
        "Film",
        "Emitter",
        "Sampler",
        "Shape",
        "Texture",
        "Volume",
        "Medium",
        "BSDF",
        "PhaseFunction",
        "Integrator"
    };

    for (const auto& domain : domains) {
      uint32_t registry_bound = ::jit_registry_id_bound(variant_name, domain);
      if (registry_bound == 0) continue;
      std::vector<void*> registry_pointers(registry_bound, nullptr);
      ::jit_registry_get_pointers(variant_name, domain, registry_pointers.data());
      for (void* ptr : registry_pointers) {
        if (!ptr) continue;
        collect_class_var(ptr);
        auto* obj = static_cast<mitsuba::Object*>(ptr);
        dr::traverse_1_fn_ro(*obj, &indices, collect_cb);
      }
    }
  }
  ::jit_set_flag(::JitFlag::EnableObjectTraversal, old_traversal);

  // Deduplicate collected indices using an order-preserving set
  std::vector<uint32_t> unique_indices;
  absl::flat_hash_set<uint32_t> seen_indices;
  unique_indices.reserve(indices.size());
  for (uint32_t idx : indices) {
    if (seen_indices.insert(idx).second) {
      unique_indices.push_back(idx);
    }
  }
  indices = std::move(unique_indices);
  return { indices, owned_class_vars };
}

// State maanger for JIT recording state.
template <typename Float, typename Spectrum>
class FrozenRender {
 public:
  using TensorXf = dr::Tensor<mitsuba::DynamicBuffer<Float>>;

  FrozenRender(JitBackend backend) : backend_(backend), render_count_(0) {}
  ~FrozenRender() { Clear(); }

  bool HasRecording() const { return recording_ != nullptr; }

  void Clear() {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Clear(): destroying recording %p\n", (void*)recording_);
    if (recording_) {
      jit_freeze_destroy(recording_);
      recording_ = nullptr;
    }
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Clear(): releasing %zu owned class variables\n", owned_class_vars_.size());
    for (uint32_t var : owned_class_vars_) {
      jit_var_dec_ref(var);
    }
    owned_class_vars_.clear();
    output_indices_.clear();
    output_shape_.clear();
    render_count_ = 0;             // Reset the recording warm-up count!
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Clear(): done\n");
  }

  bool CanReplay(const JITInputs<Float, Spectrum>& new_inputs) {
    if (!recording_) return false;
    return jit_freeze_dry_run(recording_, new_inputs.flat_indices.data()) != 0;
  }

  TensorXf Record(const JITInputs<Float, Spectrum>& inputs,
                      const std::function<TensorXf()>& render_fn) {
    if constexpr (dr::is_jit_v<Float>) {
      Clear();
      owned_class_vars_ = inputs.owned_class_vars; // Transfer ownership!
      // Force evaluation of all input JIT variables before recording.
      for (uint32_t idx : inputs.flat_indices) {
        jit_var_schedule(idx);
      }
      jit_eval();
      jit_freeze_start(backend_, inputs.flat_indices.data(), inputs.flat_indices.size());
      TensorXf result = render_fn();
      dr::eval(result.array());
      dr::sync_thread();
      output_shape_ = result.shape();
      output_indices_ = {result.array().index()};
      recording_ = jit_freeze_stop(backend_, output_indices_.data(),
                                   output_indices_.size());
      return result;
    } else {
      return render_fn();
    }
  }

  TensorXf Replay(const JITInputs<Float, Spectrum>& inputs) {
    if constexpr (dr::is_jit_v<Float>) {
      // Force evaluation of all new input JIT variables before replaying.
      for (uint32_t idx : inputs.flat_indices) {
        jit_var_schedule(idx);
      }
      jit_eval();
      std::vector<uint32_t> new_outputs(output_indices_.size(), 0);
      jit_freeze_replay(recording_, inputs.flat_indices.data(), new_outputs.data());
      // Decrement reference counts of the temporary class variables created for this replay!
      for (uint32_t var : inputs.owned_class_vars) {
        jit_var_dec_ref(var);
      }
      return TensorXf(std::move(Float::steal(new_outputs[0])), output_shape_);
    } else {
      throw std::runtime_error("FrozenRender::Replay() called on non-JIT variant");
    }
  }

  TensorXf Render(mitsuba::Scene<Float, Spectrum>* scene,
                      mitsuba::Sensor<Float, Spectrum>* sensor,
                      mitsuba::Integrator<Float, Spectrum>* integrator,
                      uint32_t sample_index,
                      uint32_t samples_to_render) {
    if constexpr (dr::is_jit_v<Float>) {
      if (render_count_ == 0) {
        TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("FrozenRender: Warm-up pass (Frame 1)\n");
        TensorXf result = integrator->render(scene, sensor, sample_index, samples_to_render, true, true);
        render_count_ = 1;
        return result;
      }

      using UInt32 = dr::uint32_array_t<Float>;
      UInt32 seed_var = dr::opaque<UInt32>(sample_index);
      auto inputs = GatherAllJitInputs(scene, sensor, integrator);
      inputs.flat_indices.push_back(seed_var.index());
      auto render_fn = [&]() {
        return integrator->render(scene, sensor, seed_var, samples_to_render, true, true);
      };

      if (render_count_ == 1) {
        TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("FrozenRender: Recording kernel (Frame 2)\n");
        TensorXf result = Record(inputs, render_fn);
        render_count_ = 2;
        return result;
      } else {
        if (CanReplay(inputs)) {
          TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("FrozenRender: Replaying frozen kernel (Frame 3+)\n");
          return Replay(inputs);
        } else {
          TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("FrozenRender: Replay dry-run failed, re-recording...\n");
          TensorXf result = Record(inputs, render_fn);
          render_count_ = 2;
          return result;
        }
      }
    } else {
      return integrator->render(scene, sensor, sample_index, samples_to_render, true, true);
    }
  }

 private:
  JitBackend backend_;
  Recording* recording_ = nullptr;
  std::vector<uint32_t> owned_class_vars_;
  std::vector<uint32_t> output_indices_;
  dr::vector<size_t> output_shape_;
  int render_count_ = 0;
};

PXR_NAMESPACE_CLOSE_SCOPE

#else // MI_ENABLE_JIT

#include <drjit/tensor.h>
#include <mitsuba/render/fwd.h>

PXR_NAMESPACE_OPEN_SCOPE

template <typename Float, typename Spectrum>
class FrozenRender {
 public:
  using TensorXf = dr::Tensor<mitsuba::DynamicBuffer<Float>>;

  template <typename T>
  FrozenRender(T&&) {}
  ~FrozenRender() {}

  void Clear() {}

  TensorXf Render(mitsuba::Scene<Float, Spectrum>* scene,
                      mitsuba::Sensor<Float, Spectrum>* sensor,
                      mitsuba::Integrator<Float, Spectrum>* integrator,
                      uint32_t sample_index,
                      uint32_t samples_to_render) {
    return integrator->render(scene, sensor, sample_index, samples_to_render, true, true);
  }
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif // MI_ENABLE_JIT
