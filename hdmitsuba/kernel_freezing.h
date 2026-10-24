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
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <drjit/array_traverse.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
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
  std::vector<uint8_t> layout;

  void AppendBytes(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    layout.insert(layout.end(), bytes, bytes + size);
  }

  template <typename T>
  void Append(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    AppendBytes(&value, sizeof(T));
  }
};

// Gathers all JIT variables from the scene, sensor, and integrator.
template <typename Float, typename Spectrum>
JITInputs<Float, Spectrum> GatherAllJitInputs(
    mitsuba::Scene<Float, Spectrum>* scene,
    mitsuba::Sensor<Float, Spectrum>* sensor,
    mitsuba::Integrator<Float, Spectrum>* integrator) {
  JITInputs<Float, Spectrum> inputs;

  if constexpr (dr::is_jit_v<Float>) {
    enum class Tag : uint8_t {
      kNullObject,
      kObjectRef,
      kObject,
      kState,
      kEmptyLeaf,
      kLiteralLeaf,
      kSlotLeaf,
    };

    struct CollectPayload {
      JITInputs<Float, Spectrum>* inputs;
      absl::flat_hash_map<uint32_t, uint32_t> slot_of_index;
      absl::flat_hash_map<dr::TraversableBase*, uint32_t> visited_objects;
      const dr::TraverseVisitor* visitor;
    };

    auto collect_cb = [](void* p, uint64_t index_combined, const char*,
                         const char*, const char*) -> uint64_t {
      auto* state = static_cast<CollectPayload*>(p);
      uint32_t index = static_cast<uint32_t>(index_combined);
      if (index == 0) {
        state->inputs->Append(Tag::kEmptyLeaf);
        return index_combined;
      }
      VarInfo info = ::jit_var_info(index);
      if (info.state == VarState::Literal ||
          info.state == VarState::Undefined) {
        state->inputs->Append(Tag::kLiteralLeaf);
        state->inputs->Append(info.type);
        state->inputs->Append(info.state);
        state->inputs->Append(info.size);
        state->inputs->Append(info.state == VarState::Literal ? info.literal
                                                              : uint64_t{0});
        return index_combined;
      }
      if (info.state != VarState::Evaluated) {
        ::jit_var_schedule(index);
      }
      auto [it, inserted] = state->slot_of_index.try_emplace(
          index, static_cast<uint32_t>(state->inputs->flat_indices.size()));
      if (inserted) {
        state->inputs->flat_indices.push_back(index);
      }
      state->inputs->Append(Tag::kSlotLeaf);
      state->inputs->Append(it->second);
      return index_combined;
    };

    auto child_cb = [](void* p, dr::TraversableBase* obj, const char*) {
      auto* state = static_cast<CollectPayload*>(p);
      if (!obj) {
        state->inputs->Append(Tag::kNullObject);
        return;
      }
      auto [it, inserted] = state->visited_objects.try_emplace(
          obj, static_cast<uint32_t>(state->visited_objects.size()));
      if (!inserted) {
        state->inputs->Append(Tag::kObjectRef);
        state->inputs->Append(it->second);
        return;
      }
      state->inputs->Append(Tag::kObject);
      state->inputs->Append(&typeid(*obj));
      obj->traverse_cb(p, *state->visitor);
    };

    auto state_cb = [](void* p, const void* value, size_t size, const char*) {
      auto* state = static_cast<CollectPayload*>(p);
      state->inputs->Append(Tag::kState);
      state->inputs->Append(size);
      state->inputs->AppendBytes(value, size);
    };

    dr::TraverseVisitor visitor{dr::TraverseRole::Freeze, collect_cb, child_cb,
                                state_cb};
    CollectPayload payload{&inputs, {}, {}, &visitor};

    dr::traverse_fn(scene, &payload, visitor);
    dr::traverse_fn(sensor, &payload, visitor);
    dr::traverse_fn(integrator, &payload, visitor);
    if constexpr (mitsuba::is_spectral_v<Spectrum>) {
      using Float32 = dr::float32_array_t<Float>;
      auto tables = mitsuba::detail::get_color_space_tables<Float32>();
      dr::traverse_fn(tables.xyz, &payload, visitor);
      dr::traverse_fn(tables.srgb, &payload, visitor);
      dr::traverse_fn(tables.d65, &payload, visitor);
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

    std::vector<void*> registry_pointers;
    for (const auto& domain : domains) {
      uint32_t registry_bound = ::jit_registry_id_bound(variant_name, domain);
      inputs.Append(registry_bound);
      if (registry_bound == 0) continue;
      registry_pointers.assign(registry_bound, nullptr);
      ::jit_registry_get_pointers(variant_name, domain, registry_pointers.data());
      for (void* ptr : registry_pointers) {
        child_cb(&payload, static_cast<mitsuba::Object*>(ptr), "");
      }
    }


    // Force evaluation of all input JIT variables before recording or replay.
    ::jit_eval();
    for (uint32_t idx : inputs.flat_indices) {
      VarInfo info = ::jit_var_info(idx);
      inputs.Append(info.type);
      inputs.Append(info.size);
      inputs.Append(info.unaligned);
    }
  }

  return inputs;
}

// State manager for JIT recording state.
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
    output_indices_.clear();
    output_shape_.clear();
    recorded_layout_.clear();
    render_count_ = 0;             // Reset the recording warm-up count!
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Clear(): done\n");
  }

  bool CanReplay(const JITInputs<Float, Spectrum>& new_inputs) {
    if (!recording_) return false;
    if (new_inputs.layout != recorded_layout_) {
      TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("CanReplay failed: input layout changed\n");
      return false;
    }
    return jit_freeze_dry_run(recording_, new_inputs.flat_indices.data()) != 0;
  }

  TensorXf Record(const JITInputs<Float, Spectrum>& inputs,
                      const std::function<TensorXf()>& render_fn) {
    if constexpr (dr::is_jit_v<Float>) {
      Clear();
      TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Record: flat_indices size = %zu\n", inputs.flat_indices.size());
      recorded_layout_ = inputs.layout;
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
      TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Replay: flat_indices size = %zu\n", inputs.flat_indices.size());
      std::vector<uint32_t> new_outputs(output_indices_.size(), 0);
      jit_freeze_replay(recording_, inputs.flat_indices.data(), new_outputs.data());
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
      if (recorded_samples_to_render_ != 0 &&
          recorded_samples_to_render_ != samples_to_render) {
        TF_DEBUG(HDMITSUBA_LIFECYCLE)
            .Msg(
                "FrozenRender: samples_to_render changed (%u -> %u), "
                "clearing recording\n",
                recorded_samples_to_render_, samples_to_render);
        Clear();
      }
      recorded_samples_to_render_ = samples_to_render;

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
  std::vector<uint32_t> output_indices_;
  dr::vector<size_t> output_shape_;
  std::vector<uint8_t> recorded_layout_;
  uint32_t recorded_samples_to_render_ = 0;
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
