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

#include "hdmitsuba/scene_manager.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/string_view.h>
#include <absl/synchronization/mutex.h>
#include <drjit-core/jit.h>
#include <drjit/array_router.h>
#include <drjit/array_traits.h>
#include <drjit/array_traverse.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <nanothread/nanothread.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/trace/trace.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#ifdef PXR_PYTHON_SUPPORT_ENABLED
#include <pxr/base/tf/pyInvoke.h>
#include <pxr/base/tf/pyUtils.h>
#endif

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/framebuffer.h"
#include "hdmitsuba/kernel_freezing.h"
#include "hdmitsuba/mesh/displacement.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/render_buffer.h"
#include "hdmitsuba/render_delegate.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/texture_cache.h"
#include "hdmitsuba/traversal.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

using mitsuba::Bitmap;
using mitsuba::ClassName;
using mitsuba::Color;
using mitsuba::Error;
using mitsuba::Mesh;
using mitsuba::MuellerMatrix;
using mitsuba::Object;
using mitsuba::PluginManager;
using mitsuba::Properties;
using mitsuba::ref;
using mitsuba::Scene;
using mitsuba::Spectrum;
using mitsuba::Thread;
namespace dr = drjit;

namespace {

constexpr size_t kDefaultSampleCount = 128;

// Sync the Python-side Mitsuba variant so Python-defined plugins
// (e.g. volprim_rf_basic) get registered for the delegate's variant.
void SyncMitsubaPythonVariant(const std::string& variant) {
#ifdef PXR_PYTHON_SUPPORT_ENABLED
  if (!TfPyIsInitialized()) {
    return;
  }
  if (!TfPyInvoke("mitsuba", "set_variant", variant)) {
    TF_WARN(
        "Failed to set Mitsuba Python variant '%s'; Python-defined plugins "
        "(e.g. volprim_rf_basic) will be unavailable.",
        variant.c_str());
  }
#else
  return;
#endif
}

constexpr std::string_view kProtoPrefix = "proto_";
constexpr std::string_view kProtoGroupPrefix = "proto_group_";
constexpr std::string_view kInstancePrefix = "instance_";

template <typename Float, typename Spectrum>
void SetTransform(
    mitsuba::Shape<Float, Spectrum>* instance,
    const mitsuba::Transform<mitsuba::Point<Float, 4>, true>& transform) {
  TraversalCallback cb;
  instance->traverse(&cb);
  using Transform4f = mitsuba::Transform<mitsuba::Point<Float, 4>, true>;
  cb.set<Transform4f>("to_world", transform);
  instance->parameters_changed({"to_world"});
}

// Runs the full post-subdivision mesh preparation pipeline on `spec` to produce
// triangle sub-meshes ready for `PrimTranslator::BuildMesh` /
// `UpdateMeshInPlace`:
//   1. Computes smooth vertex normals if missing and needed by subdivision or
//      displacement.
//   2. Applies per-material displacements (`displacements[i]` corresponds to
//      `spec.material_ids[i]`) in object space via `ApplyDisplacement`.
//   3. Transforms points and normals by `spec.transform`, then recomputes smooth
//      vertex normals if any displacement occurred.
//   4. Expands face-varying/uniform primvars into vertex-indexed buffers,
//      triangulates polygons, and splits/compacts the mesh into one
//      `SubMeshOutput` per assigned material.
template <typename Float, typename Spectrum>
std::vector<SubMeshOutput> RunGeometryPipeline(
    const MeshSpec& spec,
    const std::vector<MaterialDisplacement>& displacements) {
  TRACE_FUNCTION();
  PrimvarMap final_primvars = spec.primvars;
  if (final_primvars.find(HdTokens->normals) == final_primvars.end()) {
    bool has_displacement = false;
    for (const auto& displacement : displacements) {
      if (displacement.texture) {
        has_displacement = true;
        break;
      }
    }
    if (has_displacement || spec.is_subdivided) {
      GeometryProcessor::ComputeNormals(
          final_primvars, spec.face_vertex_indices, spec.face_vertex_counts);
    }
  }
  bool displaced = false;
  for (size_t i = 0; i < spec.material_ids.size(); ++i) {
    if (displacements[i].texture) {
      VtIntArray material_vertex_indices;
      VtIntArray material_face_counts;
      std::vector<int> global_face_indices;
      std::vector<int> global_corner_indices;
      int corner_index = 0;
      for (size_t face_index = 0; face_index < spec.face_vertex_counts.size();
           ++face_index) {
        int material_index = spec.face_material_indices[face_index];
        int vertex_count = spec.face_vertex_counts[face_index];
        if (static_cast<size_t>(material_index) == i) {
          global_face_indices.push_back(face_index);
          for (int j = 0; j < vertex_count; ++j) {
            global_corner_indices.push_back(corner_index + j);
            material_vertex_indices.push_back(
                spec.face_vertex_indices[corner_index + j]);
          }
          material_face_counts.push_back(vertex_count);
        }
        corner_index += vertex_count;
      }
      if (!material_vertex_indices.empty()) {
        ApplyDisplacement<Float, Spectrum>(
            spec.id, displacements[i], material_vertex_indices,
            material_face_counts, global_face_indices, global_corner_indices,
            final_primvars);
        displaced = true;
      }
    }
  }
  GeometryProcessor::TransformPrimvars(final_primvars, spec.transform);
  if (displaced) {
    GeometryProcessor::ComputeNormals(final_primvars, spec.face_vertex_indices,
                                      spec.face_vertex_counts);
  }

  // Expand primvar data, triangulate and split into submeshes.
  auto [face_indices, expanded_primvars] = GeometryProcessor::ExpandPrimData(
      spec.face_vertex_indices, spec.face_vertex_counts, final_primvars);
  auto [triangles, primitive_params] =
      GeometryProcessor::TriangulateWithFaceMapping(spec.face_vertex_counts,
                                                    face_indices);
  return GeometryProcessor::SplitAndCompactMeshes(
      spec.id, triangles, primitive_params, expanded_primvars,
      spec.material_ids, spec.face_material_indices);
}

}  // namespace

template <typename Float, typename Spectrum>
class SceneModel final : public SceneManager {
 public:
  MI_IMPORT_TYPES(BSDF, Emitter, Film, Integrator, Mesh, Scene, Sensor, Shape,
                  Texture);
  using PrimTranslator = PrimTranslator<Float, Spectrum>;

  struct CommittedMesh {
    std::vector<mitsuba::ref<Shape>> meshes;
    mitsuba::ref<Shape> shapegroup = nullptr;
    std::vector<mitsuba::ref<Shape>> instances;
  };

  struct MeshCommitWork {
    const MeshSpec* spec;
    std::vector<MaterialDisplacement> displacements;
  };

  SceneModel() {
    if constexpr (dr::is_jit_v<Float>) {
      jit_init(1u << static_cast<uint32_t>(dr::backend_v<Float>));
    }

    default_bsdf_ = mitsuba::PluginManager::instance()
                        ->create_object<mitsuba::BSDF<Float, Spectrum>>(
                            Properties("diffuse"));
  }

  ref<BSDF> DefaultBsdf() { return default_bsdf_; }

  ref<BSDF> ResolveBsdf(const SdfPath& material_id,
                        const PrimvarMap& primvars) {
    auto bsdf_it = bsdfs_.find(material_id.GetAsString());
    if (bsdf_it != bsdfs_.end()) {
      return bsdf_it->second;
    }
    // Fallback to the mesh's display color of specified.
    auto color_it = primvars.find(HdTokens->displayColor);
    if (color_it != primvars.end() && !color_it->second.value.IsEmpty()) {
      GfVec3f color(0.5f, 0.5f, 0.5f);
      if (color_it->second.value.IsHolding<VtVec3fArray>()) {
        const auto& colors = color_it->second.value.Get<VtVec3fArray>();
        if (!colors.empty()) {
          color = colors[0];
        }
      } else if (color_it->second.value.IsHolding<GfVec3f>()) {
        color = color_it->second.value.Get<GfVec3f>();
      }
      auto color_key = std::make_tuple(color[0], color[1], color[2]);

      {
        absl::MutexLock lock(color_bsdfs_mutex_);
        auto cached_it = color_bsdfs_.find(color_key);
        if (cached_it != color_bsdfs_.end()) {
          return cached_it->second;
        }
      }
      Properties props("diffuse");
      props.set("reflectance",
                mitsuba::Color<float, 3>(color[0], color[1], color[2]));
      auto bsdf = mitsuba::PluginManager::instance()
                      ->create_object<mitsuba::BSDF<Float, Spectrum>>(props);
      {
        absl::MutexLock lock(color_bsdfs_mutex_);
        auto [inserted_it, inserted] =
            color_bsdfs_.try_emplace(color_key, bsdf);
        return inserted_it->second;
      }
    }
    return DefaultBsdf();
  }

  // Returns the primvars that `material_id` reads as mesh attributes.
  const MeshAttributeRequests& ResolveMeshAttributes(
      const SdfPath& material_id) const {
    static const absl::NoDestructor<MeshAttributeRequests> kNone;
    auto it = mesh_attributes_.find(material_id.GetAsString());
    return it != mesh_attributes_.end() ? it->second : *kNone;
  }

  template <typename MapType, typename SpecType>
  static auto& UpsertSpec(MapType& map, SpecType&& spec) {
    auto it = map.find(spec.id);
    if (it == map.end()) {
      spec.needs_rebuild = true;
    } else {
      spec.FoldPendingFrom(it->second);
    }
    return map[spec.id] = std::forward<SpecType>(spec);
  }

  void SyncCamera(CameraSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncCamera: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    auto prev_it = camera_specs_.find(spec.id);
    std::optional<SdfPath> prev_target =
        prev_it == camera_specs_.end() ? std::nullopt
                                       : SurfaceSensorTarget(prev_it->second);
    const CameraSpec& stored = UpsertSpec(camera_specs_, std::move(spec));
    // Rebuild affected shapes if the surface sensor or its target changed.
    std::optional<SdfPath> target = SurfaceSensorTarget(stored);
    if (prev_target != target ||
        (target.has_value() && stored.needs_rebuild)) {
      if (prev_target.has_value()) {
        sensor_binding_dirty_.insert(*prev_target);
      }
      if (target.has_value()) {
        sensor_binding_dirty_.insert(*target);
      }
      shape_sensors_dirty_ = true;
    }
    reset_progressive_ = true;
  }

  void SyncMesh(MeshSpec spec) override {
    TRACE_FUNCTION();
    TF_DEBUG(HDMITSUBA_SYNC)
        .Msg("SyncMesh: %s (%zu materials, first: %s)\n", spec.id.GetText(),
             spec.material_ids.size(),
             spec.material_ids.empty() ? "none"
                                       : spec.material_ids[0].GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(mesh_specs_, std::move(spec));
    reset_progressive_ = true;
  }

  void SyncCurves(CurveSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncCurves: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    curve_specs_[spec.id] = std::move(spec);
    reset_progressive_ = true;
  }

  void SyncParticleField(ParticleFieldSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncParticleField: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(particle_field_specs_, std::move(spec));
    reset_progressive_ = true;
  }

  void SyncLight(LightSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncLight: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(light_specs_, std::move(spec));
    reset_progressive_ = true;
  }

  void SyncMaterial(MaterialSpec spec) override {
    TRACE_FUNCTION();
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncMaterial: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    material_specs_[spec.id] = std::move(spec);
    texture_cache_.MarkDirty();
    reset_progressive_ = true;
  }

  void RemoveShape(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveShape: %s\n", id.GetText());
    std::string id_str = id.GetAsString();
    absl::MutexLock lock(state_mutex_);
    mesh_specs_.erase(id);
    curve_specs_.erase(id);
    particle_field_specs_.erase(id);
    bool erased = false;
    if (shapes_.erase(id_str) > 0) {
      erased = true;
    }

    // Erase split sub-meshes
    if (EraseShapesWithPrefix(absl::StrCat(id_str, "/"))) {
      erased = true;
    }

    if (CleanUpInstancing(id)) {
      erased = true;
    }
    if (!erased) {
      TF_RUNTIME_ERROR("Could not remove shape: %s", id_str.c_str());
    }
    scene_dirty_ = true;
    reset_progressive_ = true;
  }

  void RemoveLight(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveLight: %s\n", id.GetText());
    std::string id_str = id.GetAsString();
    absl::MutexLock lock(state_mutex_);
    light_specs_.erase(id);
    shapes_.erase(id_str);
    emitters_.erase(id_str);
    scene_dirty_ = true;
    reset_progressive_ = true;
  }

  void RemoveMaterial(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveMaterial: %s\n", id.GetText());
    std::string id_str = id.GetAsString();
    absl::MutexLock lock(state_mutex_);
    material_specs_.erase(id);
    bsdfs_.erase(id_str);
    uint32_t dirty_flags = DirtyFlags::kMaterialUpdated;
    if (displacements_.erase(id_str) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    if (material_emitters_.erase(id_str) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    if (mesh_attributes_.erase(id_str) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    material_dirty_flags_[id] |= dirty_flags;
    texture_cache_.MarkDirty();
    scene_dirty_ = true;
    reset_progressive_ = true;
  }

  void SetAovBindings(
      const HdRenderPass* render_pass,
      const HdRenderPassAovBindingVector& aov_bindings) override {
    if (aov_bindings.empty()) {
      absl::MutexLock lock(aov_states_mutex_);
      pass_aov_states_.erase(render_pass);
      return;
    }

    RenderPassState pass_state;
    pass_state.aov_requests.reserve(aov_bindings.size());
    std::vector<std::string> aov_strings;
    aov_strings.reserve(aov_bindings.size());
    for (const auto& binding : aov_bindings) {
      auto* buf = static_cast<HdMitsubaRenderBuffer*>(binding.renderBuffer);
      const MitsubaAovSpec* spec = FindMitsubaAovSpec(binding.aovName);
      if (!TF_VERIFY(spec != nullptr, "Unsupported AOV: %s",
                     binding.aovName.GetText())) {
        return;
      }
      if (spec->mitsuba_aov.empty()) {
        pass_state.color_buffer = buf;
      } else {
        pass_state.aov_requests.push_back(
            {spec->mitsuba_aov, buf, spec->channels});
        aov_strings.push_back(spec->mitsuba_aov);
      }
    }
    pass_state.aov_integrator_keys = absl::StrJoin(aov_strings, ",");
    pass_state.aov_integrator = nullptr;
    reset_progressive_ = true;
    absl::MutexLock lock(aov_states_mutex_);
    pass_aov_states_[render_pass] = std::move(pass_state);
  }

  void Render(const HdRenderPass* render_pass,
              const HdCamera* camera,
              const std::optional<GfRect2i>& crop_window = std::nullopt) override {
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
    JitScopeGuard<Float> jit_guard;
    auto pass_it = pass_aov_states_.find(render_pass);
    if (pass_it == pass_aov_states_.end() ||
        (!pass_it->second.color_buffer &&
         pass_it->second.aov_requests.empty())) {
      TF_WARN("No color buffer or AOV requests for render pass.");
      return;
    }
    RenderPassState& pass_state = pass_it->second;

    // Get dimensions from the first buffer (assuming all match).
    HdMitsubaRenderBuffer* primary_buffer =
        pass_state.color_buffer ? pass_state.color_buffer
                                : pass_state.aov_requests[0].buffer;
    if (!primary_buffer) {
      TF_WARN("No valid primary buffer found.");
      return;
    }
    const unsigned int buffer_width = primary_buffer->GetWidth();
    const unsigned int buffer_height = primary_buffer->GetHeight();

    if (scene_->sensors().empty()) {
      TF_RUNTIME_ERROR("No sensor specified for Mitsuba scene.");
      return;
    }

    // 2. Setup Scene and Integrator.
    Sensor* sensor = nullptr;
    if (camera == nullptr) {
      // If no camera is specified, use the first sensor in the scene.
      sensor = scene_->sensors()[0];
    } else {
      // Otherwise, find the sensor with id matching the camera id.
      std::string camera_id = camera->GetId().GetAsString();
      for (auto& s : scene_->sensors()) {
        if (s->id() == camera_id) {
          sensor = s.get();
          break;
        }
      }
      if (!sensor) {
        TF_RUNTIME_ERROR("Camera not found in scene: %s", camera_id.c_str());
        return;
      }
    }

    if (sensor != last_sensor_) {
      last_sensor_ = sensor;
      reset_progressive_ = true;
      if constexpr (dr::is_jit_v<Float>) {
        if (frozen_render_) {
          frozen_render_->Clear();
        }
      }
    }

    Film* film = sensor->film();
    auto film_size = film->size();
    bool film_changed = false;
    // If necessary, resize film to match USD buffer size.
    if (film_size.x() != buffer_width || film_size.y() != buffer_height) {
      film->set_size(ScalarPoint2u(buffer_width, buffer_height));
      film_changed = true;
    }

    ScalarPoint2u new_crop_offset(0, 0);
    ScalarVector2u new_crop_size = film->size();
    if (crop_window.has_value()) {
      new_crop_offset = ScalarPoint2u(crop_window->GetMinX(), crop_window->GetMinY());
      new_crop_size = ScalarVector2u(crop_window->GetWidth(), crop_window->GetHeight());
    }

    if (dr::any(film->crop_offset() != new_crop_offset) || dr::any(film->crop_size() != new_crop_size)) {
      film->set_crop_window(new_crop_offset, new_crop_size);
      film_changed = true;
    }

    if (film_changed) {
      sensor->parameters_changed();
      if (frozen_render_) frozen_render_->Clear(); // Invalidate cache on resize
      reset_progressive_ = true;
    }

    if (!integrator_) {
      integrator_ = PluginManager::instance()->create_object<Integrator>(
          Properties("path"));
    }

    Integrator* render_integrator = integrator_.get();
    if (!pass_state.aov_requests.empty()) {
      if (!pass_state.aov_integrator) {
        Properties aov_props("aov");
        aov_props.set("aovs", pass_state.aov_integrator_keys);
        aov_props.set("integrator", integrator_.get());
        pass_state.aov_integrator =
            PluginManager::instance()->create_object<Integrator>(aov_props);
      }
      render_integrator = pass_state.aov_integrator.get();
    }

    if (reset_progressive_ || !progressive_rendering_) {
      accum_buffer_ = TensorXf();
      current_progressive_sample_ = 0;
      reset_progressive_ = false;
    }

    int samples_to_render = sample_count_;
    if (progressive_rendering_) {
      int remaining =
          static_cast<int>(sample_count_) - current_progressive_sample_;
      samples_to_render =
          std::max(0, std::min(interactive_samples_per_pass_, remaining));
    }
    TensorXf display_result;
    if (samples_to_render > 0) {
      // 3. Render a chunk of samples
      TensorXf result;
      uint32_t sample_index = progressive_rendering_
          ? static_cast<uint32_t>(current_progressive_sample_)
          : 0;
      bool run_frozen = false;
      if constexpr (dr::is_jit_v<Float>) {
        run_frozen = !has_instancing_ && frozen_render_;
      }
      if (run_frozen) {
        result = frozen_render_->Render(scene_.get(), sensor, render_integrator,
                                        sample_index, static_cast<uint32_t>(samples_to_render));
      } else {
        result = render_integrator->render(
            scene_.get(), sensor, sample_index,
            static_cast<uint32_t>(samples_to_render), true, true);
        if constexpr (dr::is_jit_v<Float>) {
          dr::sync_thread();
          if (frozen_render_) frozen_render_->Clear(); // Clear cache if freezing is disabled
        }
      }
      size_t expected_width = crop_window.has_value() ? crop_window->GetWidth() : buffer_width;
      size_t expected_height = crop_window.has_value() ? crop_window->GetHeight() : buffer_height;
      if (expected_width != result.shape()[1] ||
          expected_height != result.shape()[0]) {
        TF_RUNTIME_ERROR("Buffer size mismatch: %lu x %lu vs %lu x %lu",
                         expected_width, expected_height, result.shape()[1],
                         result.shape()[0]);
        return;
      }

      // 4. Accumulate and average
      current_progressive_sample_ += samples_to_render;
      if (progressive_rendering_) {
        // integrator->render() returns the per-pass mean over samples_to_render
        // samples; weight by samples_to_render so passes of different sizes
        // (e.g. 7 spp at 2 spp/pass -> 2, 2, 2, 1) contribute equally per sample.
        TensorXf weighted(
            result.array() * static_cast<float>(samples_to_render),
            result.shape());
        if (current_progressive_sample_ == samples_to_render ||
            accum_buffer_.empty()) {
          accum_buffer_ = std::move(weighted);
        } else {
          accum_buffer_.array() += weighted.array();
        }
        if constexpr (dr::is_jit_v<Float>) {
          dr::eval(accum_buffer_);  // keep the JIT graph bounded across passes
        }
        // Average for display
        display_result =
            TensorXf(accum_buffer_.array() /
                         static_cast<float>(current_progressive_sample_),
                     result.shape());
      } else {
        display_result = result;
      }
    } else {
      // Already converged, ensure buffers are marked as converged.
      bool converged = true;
      if (pass_state.color_buffer) {
        pass_state.color_buffer->SetConverged(converged);
      }
      for (auto& req : pass_state.aov_requests) {
        req.buffer->SetConverged(converged);
      }
      return;
    }

    // 5. Copy the data to the output render buffers.
    size_t base_channels = sensor->film()->base_channels().size();
    size_t total_channels = display_result.shape()[2];
    destinations_.clear();
    destinations_.reserve(1 + pass_state.aov_requests.size());

    if (pass_state.color_buffer) {
      destinations_.push_back({pass_state.color_buffer, 0, static_cast<int>(base_channels), false});
    }

    size_t current_offset = base_channels;
    for (const auto& req : pass_state.aov_requests) {
      if (!TF_VERIFY(req.buffer, "AOV buffer is null")) continue;
      if (!TF_VERIFY(current_offset + req.channel_count <= total_channels,
                     "AOV %s - out of bounds (offset %zu + %zu > %zu)",
                     req.mitsuba_name.c_str(), current_offset,
                     static_cast<size_t>(req.channel_count), total_channels)) {
        return;
      }
      bool is_int = absl::StrContains(req.mitsuba_name, "_index");
      destinations_.push_back({req.buffer, static_cast<int>(current_offset), req.channel_count, is_int});
      current_offset += req.channel_count;
    }
    PerformBatchedCopy<Float>(display_result, destinations_, crop_window);

    // 6. Set convergence status
    bool converged =
        !progressive_rendering_ ||
        (current_progressive_sample_ >= static_cast<int>(sample_count_));
    if (pass_state.color_buffer) {
      pass_state.color_buffer->SetConverged(converged);
    }
    for (auto& req : pass_state.aov_requests) {
      req.buffer->SetConverged(converged);
    }
  }

  bool IsConverged() const override {
    return !progressive_rendering_ ||
           (current_progressive_sample_ >= static_cast<int>(sample_count_));
  }

  int GetCurrentSampleCount() const override {
    return progressive_rendering_ ? current_progressive_sample_
                                  : static_cast<int>(sample_count_);
  }

  int GetTargetSampleCount() const override {
    return static_cast<int>(sample_count_);
  }

  void UpdateNamespacedSettings(
      const VtDictionary& namespaced_settings) override {
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
    {
      auto it =
          namespaced_settings.find("mitsuba:interactive_samples_per_pass");
      if (it != namespaced_settings.end()) {
        int new_samples_per_pass = std::max(1, it->second.Get<int>());
        if (new_samples_per_pass != interactive_samples_per_pass_) {
          interactive_samples_per_pass_ = new_samples_per_pass;
          reset_progressive_ = true;
        }
      }
    }
    {
      auto it = namespaced_settings.find("mitsuba:integrator:type");
      if (it != namespaced_settings.end()) {
        std::string integrator_type = it->second.Get<std::string>();
        if (integrator_type != integrator_type_ || !integrator_) {
          TF_DEBUG(HDMITSUBA_LIFECYCLE)
              .Msg("Creating integrator, type: %s (was: %s)\n",
                   integrator_type.c_str(), integrator_type_.c_str());
          Properties integrator_props(integrator_type);
          integrator_ = PluginManager::instance()->create_object<Integrator>(
              integrator_props);
          integrator_type_ = integrator_type;
          for (auto& [_, pass_state] : pass_aov_states_) {
            pass_state.aov_integrator = nullptr;
          }
          reset_progressive_ = true;
        }
      }
    }

    {
      auto it = namespaced_settings.find("mitsuba:sample_count");
      if (it != namespaced_settings.end()) {
        int new_sample_count = it->second.Get<int>();
        if (static_cast<size_t>(new_sample_count) != sample_count_) {
          sample_count_ = new_sample_count;
          reset_progressive_ = true;
        }
      }
    }
    {
      auto it = namespaced_settings.find("enableInteractive");
      if (it != namespaced_settings.end()) {
        bool progressive = it->second.Get<bool>();
        if (progressive != progressive_rendering_) {
          progressive_rendering_ = progressive;
          // A change in render mode requires rebuilding the sensors, since
          // this affects the used pixel filter type.
          for (auto& [_, camera_spec] : camera_specs_) {
            camera_spec.needs_rebuild = true;
            if (auto target = SurfaceSensorTarget(camera_spec)) {
              sensor_binding_dirty_.insert(*target);
              shape_sensors_dirty_ = true;
            }
          }
          reset_progressive_ = true;
        }
      }
    }
    {
      auto it = namespaced_settings.find("mitsuba:use_kernel_freezing");
      if (it != namespaced_settings.end()) {
        bool enabled = it->second.Get<bool>();
        TF_DEBUG(HDMITSUBA_LIFECYCLE)
            .Msg("Kernel freezing setting: %d\n", enabled);
        if constexpr (dr::is_jit_v<Float>) {
          if (enabled) {
            if (!frozen_render_) {
              frozen_render_ = std::make_unique<FrozenRender<Float, Spectrum>>(dr::backend_v<Float>);
            }
          } else {
            frozen_render_.reset(); // Immediately destroys it and frees all JIT GPU memory!
          }
        }
      }
    }
  }

  enum DirtyFlags : uint32_t {
    kClean = 0,
    kNeedsStructureRebuild = 1 << 0,
    kMaterialUpdated = 1 << 1
  };

  uint32_t UpdateMaterialStructure(absl::string_view id_str,
                                   TranslatedMaterial& trans) {
    uint32_t dirty_flags = 0;
    auto update = [&](auto& map, auto* val) {
      if (val) {
        auto it = map.find(id_str);
        if (it == map.end() || it->second != *val) {
          map[id_str] = std::move(*val);
          dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
        }
      } else if (map.erase(id_str) > 0) {
        dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
      }
    };
    update(displacements_,
           trans.displacement.texture ? &trans.displacement : nullptr);
    update(material_emitters_,
           trans.shape_emitter_props ? &*trans.shape_emitter_props : nullptr);
    update(mesh_attributes_,
           !trans.mesh_attributes.empty() ? &trans.mesh_attributes : nullptr);
    return dirty_flags;
  }

  template <typename MapType>
  auto GetPendingSpecs(MapType& specs_map) {
    using SpecType = typename MapType::mapped_type;
    std::vector<SpecType*> pending_specs;
    pending_specs.reserve(specs_map.size());
    auto apply_material_dirty = [&](const SdfPath& mat_id, BaseSpec& s,
                                    bool& bsdf_updated) {
      if (auto it = material_dirty_flags_.find(mat_id);
          it != material_dirty_flags_.end()) {
        if (it->second & DirtyFlags::kNeedsStructureRebuild) {
          s.needs_rebuild = true;
        }
        if (it->second & DirtyFlags::kMaterialUpdated) {
          bsdf_updated = true;
        }
      }
    };
    for (auto& [id, spec] : specs_map) {
      bool needs_bsdf_update = false;
      if constexpr (std::is_same_v<SpecType, MeshSpec>) {
        for (const auto& mat_id : spec.material_ids) {
          apply_material_dirty(mat_id, spec, needs_bsdf_update);
        }
        if (sensor_binding_dirty_.contains(spec.id)) {
          spec.needs_rebuild = true;
        }
      } else if constexpr (std::is_same_v<SpecType, CurveSpec>) {
        apply_material_dirty(spec.material_id, spec, needs_bsdf_update);
      }
      bool has_in_place_update = false;
      if constexpr (std::is_same_v<SpecType, ParticleFieldSpec>) {
        has_in_place_update = spec.geometry_dirty || spec.attributes_dirty();
      }
      if (spec.needs_rebuild || spec.dirty_bits != 0 || needs_bsdf_update ||
          has_in_place_update) {
        pending_specs.push_back(&spec);
      }
    }
    return pending_specs;
  }

  template <typename MapType, typename ValueType, typename WorkFn,
            typename MergeFn>
  bool ParallelCommit(MapType& specs_map, WorkFn&& work_fn,
                      MergeFn&& merge_fn) {
    auto pending_specs = GetPendingSpecs(specs_map);
    if (pending_specs.empty()) return false;

    std::vector<ValueType> results(pending_specs.size());

    jit_eval();  // Flush any pending side effects before multithreaded work.
    drjit::parallel_for(drjit::blocked_range<size_t>(0, pending_specs.size()),
                        [&](drjit::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            JitScopeGuard<Float> jit_guard;
                            work_fn(pending_specs[i], results[i]);
                          }
                          if constexpr (dr::is_metal_v<Float>) {
                            jit_flush_thread();
                          }
                        });

    JitScopeGuard<Float> jit_guard;
    bool rebuild = false;
    for (size_t i = 0; i < pending_specs.size(); ++i) {
      if (pending_specs[i]->needs_rebuild) {
        rebuild |= merge_fn(pending_specs[i], results[i]);
      }
      pending_specs[i]->MarkClean();
    }
    return rebuild;
  }

  void CommitMaterials() {
    auto pending_specs = GetPendingSpecs(material_specs_);

    // Pre-load all textures in parallel to support re-use.
    if (!pending_specs.empty()) {
      PreloadTextures(pending_specs);
    }

    ParallelCommit<decltype(material_specs_), TranslatedMaterial>(
        material_specs_,
        [&](MaterialSpec* spec, TranslatedMaterial& res) {
          res = PrimTranslator::BuildMaterial(*spec, texture_cache_);
        },
        [&](MaterialSpec* spec, TranslatedMaterial& trans) {
          std::string id_str = spec->id.GetAsString();
          if (trans.bsdf) {
            bsdfs_[id_str] = dynamic_cast<BSDF*>(trans.bsdf.get());
          }
          material_dirty_flags_[spec->id] |=
              UpdateMaterialStructure(id_str, trans) |
              DirtyFlags::kMaterialUpdated;
          return false;
        });
  }

  static std::optional<SdfPath> SurfaceSensorTarget(const CameraSpec& spec) {
    if (spec.sensor_type != "irradiancemeter") {
      return std::nullopt;
    }
    return spec.target_shape_id;
  }

  void RebuildShapeSensorMap() {
    shape_sensors_.clear();
    for (const auto& [camera_id, spec] : camera_specs_) {
      std::optional<SdfPath> target = SurfaceSensorTarget(spec);
      if (!target.has_value()) {
        continue;
      }
      if (!shape_sensors_.try_emplace(target->GetAsString(), camera_id)
               .second) {
        TF_WARN(
            "Shape %s is already measured by another sensor; ignoring sensor "
            "%s.",
            target->GetText(), camera_id.GetText());
      }
    }
  }

  bool CommitCameras() {
    bool rebuild = ParallelCommit<decltype(camera_specs_), mitsuba::ref<Sensor>>(
        camera_specs_,
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          // Surface sensors are built together with their target mesh.
          if (spec->sensor_type == "irradiancemeter") {
            return;
          }
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildSensor(*spec, progressive_rendering_);
          } else if (spec->dirty_bits != 0) {
            auto it = sensors_.find(spec->id.GetAsString());
            if (!TF_VERIFY(it != sensors_.end(), "Camera sensor not found: %s",
                           spec->id.GetText())) {
              return;
            }
            PrimTranslator::UpdateSensorInPlace(it->second.get(), *spec);
          }
        },
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          if (spec->sensor_type == "irradiancemeter") {
            return false;
          }
          sensors_[spec->id.GetAsString()] = res;
          return true;
        });
    if (shape_sensors_dirty_) {
      RebuildShapeSensorMap();
      shape_sensors_dirty_ = false;
    }
    return rebuild;
  }

  struct EmitterSensorPair {
    mitsuba::ref<Object> mesh_emitter = nullptr;
    Object* emitter_ptr = nullptr;
    mitsuba::ref<Object> mesh_sensor = nullptr;
    Object* sensor_ptr = nullptr;
  };

  EmitterSensorPair ResolveEmitterAndSensor(
      const std::optional<LightSpec>& emitter_spec, const SdfPath& material_id,
      const SdfPath& shape_id, bool double_sided) {
    EmitterSensorPair pair;
    if (emitter_spec.has_value()) {
      pair.mesh_emitter =
          PrimTranslator::CreateAreaEmitter(emitter_spec->emission);
      pair.emitter_ptr = pair.mesh_emitter.get();
    }
    if (pair.emitter_ptr == nullptr) {
      auto emitter_it = material_emitters_.find(material_id.GetAsString());
      if (emitter_it != material_emitters_.end()) {
        mitsuba::Properties emitter_props = emitter_it->second;
        if (double_sided) {
          emitter_props.set("twosided", true, false);
        }
        pair.mesh_emitter = mitsuba::PluginManager::instance()->create_object(
            emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
            mitsuba::Emitter<Float, Spectrum>::Type);
        pair.emitter_ptr = pair.mesh_emitter.get();
      }
    }
    // A Mitsuba sensor can only be attached to one shape, so like the area
    // emitter above, every (re)built mesh gets a fresh one.
    auto sens_it = shape_sensors_.find(shape_id.GetAsString());
    if (sens_it != shape_sensors_.end()) {
      pair.mesh_sensor = PrimTranslator::BuildSensor(
          camera_specs_.at(sens_it->second), progressive_rendering_);
      pair.sensor_ptr = pair.mesh_sensor.get();
    }
    return pair;
  }

  void UpdateSubMeshesInPlace(const MeshSpec& spec,
                              const std::vector<SubMeshOutput>& sub_meshes,
                              std::string_view key_prefix) {
    constexpr HdDirtyBits kMeshGeometryDirty =
        HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTransform |
        HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar |
        HdChangeTracker::DirtyTopology;
    const bool geometry_dirty = (spec.dirty_bits & kMeshGeometryDirty) != 0;
    for (const auto& sub_mesh : sub_meshes) {
      std::string key = absl::StrCat(key_prefix, sub_mesh.id.GetAsString());
      auto it = shapes_.find(key);
      if (!TF_VERIFY(it != shapes_.end(), "Sub-mesh not found: %s",
                     key.c_str())) {
        continue;
      }
      ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);
      if (geometry_dirty) {
        PrimTranslator::UpdateMeshInPlace(
            it->second.get(), sub_mesh.triangles, sub_mesh.primvars,
            ResolveMeshAttributes(sub_mesh.material_id), spec.dirty_bits);
      }
      if (key_prefix.empty() && spec.dirty_bits != 0 &&
          it->second->is_emitter() && spec.emitter_spec.has_value()) {
        // Update emissive mesh radiance in-place
        auto* emitter = it->second->emitter();
        TraversalCallback cb_emitter;
        emitter->traverse(&cb_emitter);
        using Color3f = mitsuba::Color<Float, 3>;
        cb_emitter.set<Color3f>("radiance.value",
                                Color3f(spec.emitter_spec->emission[0],
                                        spec.emitter_spec->emission[1],
                                        spec.emitter_spec->emission[2]));
        emitter->parameters_changed();
      }
      it->second->set_bsdf(bsdf.get());
    }
  }

  void CommitNonInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res) {
    const auto& spec = *(work->spec);
    auto sub_meshes =
        RunGeometryPipeline<Float, Spectrum>(spec, work->displacements);
    if (spec.needs_rebuild) {
      res.meshes.reserve(sub_meshes.size());
      for (const auto& sub_mesh : sub_meshes) {
        auto env = ResolveEmitterAndSensor(spec.emitter_spec,
                                           sub_mesh.material_id, spec.id,
                                           spec.double_sided);
        ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);
        auto mesh = PrimTranslator::BuildMesh(
            sub_mesh.id, sub_mesh.triangles, sub_mesh.primvars,
            ResolveMeshAttributes(sub_mesh.material_id), bsdf.get(),
            env.emitter_ptr, env.sensor_ptr);
        if (mesh) {
          res.meshes.push_back(mesh);
        }
      }
    } else {
      UpdateSubMeshesInPlace(spec, sub_meshes, "");
    }
  }

  void CommitInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res) {
    const auto& spec = *(work->spec);
    std::string id_str = spec.id.GetAsString();
    if (spec.needs_rebuild) {
      if (HasEmitter(spec)) {
        TF_WARN(
            "Mesh %s is instanced but has an emitter attached. Mitsuba "
            "does not support emitters on instances. Ignoring emitter.",
            spec.id.GetText());
      }
      if (shape_sensors_.contains(id_str)) {
        TF_WARN(
            "Mesh %s is instanced but has a sensor attached. Mitsuba does "
            "not support sensors on instances. Ignoring sensor.",
            spec.id.GetText());
      }

      // Run geometry pipeline
      auto sub_meshes =
          RunGeometryPipeline<Float, Spectrum>(spec, work->displacements);
      if (sub_meshes.empty()) return;

      // 1. Build prototype meshes
      std::vector<mitsuba::ref<Shape>> prototype_shapes;
      prototype_shapes.reserve(sub_meshes.size());
      for (const auto& sub_mesh : sub_meshes) {
        ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);

        auto mesh = PrimTranslator::BuildMesh(
            sub_mesh.id, sub_mesh.triangles, sub_mesh.primvars,
            ResolveMeshAttributes(sub_mesh.material_id), bsdf.get(), nullptr,
            nullptr);
        if (mesh) {
          prototype_shapes.push_back(mesh);
          res.meshes.push_back(mesh);
        }
      }
      if (prototype_shapes.empty()) return;

      // 2. Create ShapeGroup wrapping all prototype meshes
      mitsuba::Properties group_props("shapegroup");
      for (size_t i = 0; i < prototype_shapes.size(); ++i) {
        group_props.set("shape_" + std::to_string(i),
                        prototype_shapes[i].get());
      }
      res.shapegroup =
          mitsuba::PluginManager::instance()->create_object<Shape>(group_props);

      // 3. Create Instances
      res.instances.reserve(spec.instance_transforms.size());
      for (size_t i = 0; i < spec.instance_transforms.size(); ++i) {
        mitsuba::Properties inst_props("instance");
        inst_props.set("shapegroup", res.shapegroup.get());
        inst_props.set("to_world",
                       UsdToMitsubaTransform(spec.instance_transforms[i]));
        mitsuba::ref<Shape> inst =
            mitsuba::PluginManager::instance()->create_object<Shape>(
                inst_props);
        res.instances.push_back(inst);
      }
    } else {
      // Update in place
      auto sub_meshes =
          RunGeometryPipeline<Float, Spectrum>(spec, work->displacements);
      UpdateSubMeshesInPlace(spec, sub_meshes, kProtoPrefix);
      if (spec.dirty_bits & (HdChangeTracker::DirtyInstancer |
                             HdChangeTracker::DirtyInstanceIndex)) {
        for (size_t i = 0; i < spec.instance_transforms.size(); ++i) {
          auto inst_it =
              shapes_.find(absl::StrCat(kInstancePrefix, id_str, "_", i));
          if (TF_VERIFY(inst_it != shapes_.end())) {
            SetTransform(
                inst_it->second.get(),
                AffineTransform4f(
                    UsdToMitsubaTransform(spec.instance_transforms[i]).matrix));
          }
        }
      }
    }
  }

  bool MergeNonInstancedMesh(CommittedMesh& res,
                             const std::string& /*id_str*/) {
    for (auto& mesh : res.meshes) {
      shapes_[mesh->id()] = mesh;
    }
    return true;  // Rebuild scene
  }

  bool MergeInstancedMesh(CommittedMesh& res, const std::string& id_str) {
    for (auto& mesh : res.meshes) {
      shapes_[absl::StrCat(kProtoPrefix, mesh->id())] = mesh;
    }
    shapes_[absl::StrCat(kProtoGroupPrefix, id_str)] = res.shapegroup;
    for (size_t i = 0; i < res.instances.size(); ++i) {
      shapes_[absl::StrCat(kInstancePrefix, id_str, "_", i)] = res.instances[i];
    }
    return true;  // Rebuild scene
  }

  bool HasEmitter(const MeshSpec& spec) const {
    return spec.emitter_spec.has_value() ||
           std::any_of(spec.material_ids.begin(), spec.material_ids.end(),
                       [&](const SdfPath& material_id) {
                         return material_emitters_.contains(
                             material_id.GetAsString());
                       });
  }

  bool CommitMeshes() {
    // Dynamically determine if the scene contains any instanced meshes
    bool has_instancing = false;
    for (const auto& [id, spec] : mesh_specs_) {
      if (!spec.instance_transforms.empty()) {
        has_instancing = true;
        break;
      }
    }
    has_instancing_ = has_instancing;
    // 1. Get pending specs.
    auto pending_specs = GetPendingSpecs(mesh_specs_);
    if (pending_specs.empty()) return false;

    // 2. Prepare work on the main thread (resolve displacement textures
    // lock-free).
    std::vector<MeshCommitWork> work_items(pending_specs.size());
    for (size_t i = 0; i < pending_specs.size(); ++i) {
      const MeshSpec* spec = pending_specs[i];
      work_items[i].spec = spec;
      work_items[i].displacements.reserve(spec->material_ids.size());
      for (const auto& material_id : spec->material_ids) {
        auto disp_it = displacements_.find(material_id.GetAsString());
        if (disp_it != displacements_.end()) {
          work_items[i].displacements.push_back(disp_it->second);
        } else {
          work_items[i].displacements.emplace_back();
        }
      }
    }

    // 3. Run parallel commit on work items.
    std::vector<CommittedMesh> results(work_items.size());
    jit_eval();  // Flush any pending side effects before multithreaded work.
    drjit::parallel_for(
        drjit::blocked_range<size_t>(0, work_items.size()),
        [&](drjit::blocked_range<size_t> r) {
          JitScopeGuard<Float> jit_guard;
          bool needs_jit_sync = false;

          for (size_t i = r.begin(); i != r.end(); ++i) {
            const MeshSpec& spec = *work_items[i].spec;
            if (!spec.needs_rebuild || HasEmitter(spec) ||
                shape_sensors_.contains(spec.id.GetAsString()) ||
                std::any_of(work_items[i].displacements.begin(),
                            work_items[i].displacements.end(),
                            [](const MaterialDisplacement& d) {
                              return d.texture != nullptr;
                            })) {
              needs_jit_sync = true;
            }
            if (spec.instance_transforms.empty()) {
              CommitNonInstancedMeshWork(&work_items[i], results[i]);
            } else {
              CommitInstancedMeshWork(&work_items[i], results[i]);
            }
          }
          // Make sure any pending JIT compilations and scatter side-effects
          // (e.g. from UpdateMeshInPlace or displacement/emitter setup) are
          // flushed before exiting the worker scope.
          if constexpr (dr::is_jit_v<Float>) {
            if (needs_jit_sync) {
              dr::eval();
              dr::sync_thread();
            }
          }
          if constexpr (dr::is_metal_v<Float>) {
            jit_flush_thread();
          }
        });

    JitScopeGuard<Float> jit_guard;
    bool rebuild = false;
    // 4. Merge results on the main thread.
    for (size_t i = 0; i < pending_specs.size(); ++i) {
      const MeshSpec* spec = pending_specs[i];
      std::string id_str = spec->id.GetAsString();

      if (spec->needs_rebuild) {
        // Clean up old shapes
        CleanUpInstancing(spec->id);
        shapes_.erase(id_str);
        EraseShapesWithPrefix(absl::StrCat(id_str, "/"));
        if (!spec->instance_transforms.empty()) {
          rebuild |= MergeInstancedMesh(results[i], id_str);
        } else {
          rebuild |= MergeNonInstancedMesh(results[i], id_str);
        }
      }
      const_cast<MeshSpec*>(spec)->MarkClean();
    }
    return rebuild;
  }

  bool CommitCurves() {
    return ParallelCommit<decltype(curve_specs_), mitsuba::ref<Shape>>(
        curve_specs_,
        [&](CurveSpec* spec, mitsuba::ref<Shape>& res) {
          auto bsdf_it = bsdfs_.find(spec->material_id.GetAsString());
          ref<BSDF> bsdf =
              bsdf_it != bsdfs_.end() ? bsdf_it->second : DefaultBsdf();

          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildCurves(*spec, bsdf.get());
          } else {
            auto it = shapes_.find(spec->id.GetAsString());
            if (!TF_VERIFY(it != shapes_.end(), "Curve not found: %s",
                           spec->id.GetText())) {
              return;
            }
            it->second->set_bsdf(bsdf.get());
          }
        },
        [&](CurveSpec* spec, mitsuba::ref<Shape>& res) {
          if (res) {
            shapes_[spec->id.GetAsString()] = res;
          }
          return true;
        });
  }

  bool CommitParticleFields() {
    if (!particle_field_specs_.empty() &&
        integrator_type_ != "volprim_rf_basic") {
      TF_WARN(
          "Scene contains Gaussian splats (%zu), which require the "
          "'volprim_rf_basic' integrator for radiance field evaluation. "
          "Current integrator: '%s'.",
          particle_field_specs_.size(), integrator_type_.c_str());
    }
    return ParallelCommit<decltype(particle_field_specs_), mitsuba::ref<Shape>>(
        particle_field_specs_,
        [&](ParticleFieldSpec* spec, mitsuba::ref<Shape>& res) {
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildParticleField(*spec);
            return;
          }
          // Particle count and SH layout are unchanged, so the existing shape
          // can absorb the new values without being rebuilt (and without
          // forcing a scene rebuild - ParallelCommit only merges rebuilds).
          auto it = shapes_.find(spec->id.GetAsString());
          if (!TF_VERIFY(it != shapes_.end(), "Particle field not found: %s",
                         spec->id.GetText())) {
            return;
          }
          TF_DEBUG(HDMITSUBA_SYNC)
              .Msg("UpdateParticleFieldInPlace: %s (geometry: %d, "
                   "attributes: %d)\n",
                   spec->id.GetText(), spec->geometry_dirty,
                   spec->attributes_dirty());
          PrimTranslator::UpdateParticleFieldInPlace(it->second.get(), *spec);
        },
        [&](ParticleFieldSpec* spec, mitsuba::ref<Shape>& res) {
          if (res) {
            shapes_[spec->id.GetAsString()] = res;
          }
          return true;
        });
  }

  bool CommitLights() {
    return ParallelCommit<decltype(light_specs_),
                          typename PrimTranslator::TranslatedLight>(
        light_specs_,
        [&](LightSpec* spec, typename PrimTranslator::TranslatedLight& res) {
          std::string id_str = spec->id.GetAsString();
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildLight(*spec);
          } else if (spec->dirty_bits != 0) {
            if (spec->IsAreaLight()) {
              auto it = shapes_.find(id_str);
              if (TF_VERIFY(it != shapes_.end(), "Light shape not found: %s",
                            id_str.c_str())) {
                PrimTranslator::UpdateLightInPlace(it->second.get(), *spec);
              }
            } else {
              auto it = emitters_.find(id_str);
              if (TF_VERIFY(it != emitters_.end(),
                            "Light emitter not found: %s", id_str.c_str())) {
                PrimTranslator::UpdateLightInPlace(it->second.get(), *spec);
              }
            }
          }
        },
        [&](LightSpec* spec, typename PrimTranslator::TranslatedLight& trans) {
          std::string id_str = spec->id.GetAsString();
          if (trans.shape) {
            shapes_[id_str] = trans.shape;
            emitters_.erase(id_str);
          } else if (trans.emitter) {
            emitters_[id_str] = trans.emitter;
            shapes_.erase(id_str);
          }
          return true;
        });
  }

  void CommitResources() override {
    absl::MutexLock lock(state_mutex_);
    JitScopeGuard<Float> jit_guard;

    bool rebuild_scene = scene_dirty_;
    CommitMaterials();
    rebuild_scene |= CommitCameras();
    rebuild_scene |= CommitMeshes();
    rebuild_scene |= CommitCurves();
    rebuild_scene |= CommitParticleFields();
    rebuild_scene |= CommitLights();

    const bool materials_changed = !material_dirty_flags_.empty();
    if (!rebuild_scene && materials_changed &&
        NeedsRebuildForFilteredTextures()) {
      rebuild_scene = true;
    }

    if (rebuild_scene || materials_changed) {
      // In-place material updates swap BSDF/Texture objects whose JIT class
      // pointers may be baked into a frozen kernel, so it must be invalidated
      // even when the Scene itself is not rebuilt.
      if constexpr (dr::is_jit_v<Float>) {
        if (frozen_render_) {
          frozen_render_->Clear();
        }
      }
      reset_progressive_ = true;
    }

    if (rebuild_scene) {
      TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("Instantiating new scene\n");
      Properties props;
      for (auto& [id, shape] : shapes_) {
        bool is_proto_mesh = absl::StartsWith(id, kProtoPrefix) &&
                             !absl::StartsWith(id, kProtoGroupPrefix);
        if (is_proto_mesh) {
          continue;
        }
        props.set(id, shape.get());
      }
      for (auto& [id, sensor] : sensors_) props.set(id, sensor.get());
      for (auto& [id, emitter] : emitters_) props.set(id, emitter.get());
      scene_ = new Scene(props);
      scene_dirty_ = false;
    } else {
      scene_->parameters_changed();
    }

    // Run Garbage Collection after all resources are committed, ensuring
    // that all active shapes, mesh emitters, and sensors are instantiated.
    texture_cache_.GarbageCollect();

    material_dirty_flags_.clear();
    sensor_binding_dirty_.clear();
  }

  mitsuba::Object* GetScene() override { return scene_.get(); }

 private:
  struct AovRequest {
    std::string mitsuba_name;
    HdMitsubaRenderBuffer* buffer;
    int channel_count;
  };

  struct RenderPassState {
    ref<Integrator> aov_integrator = nullptr;
    std::string aov_integrator_keys;
    std::vector<AovRequest> aov_requests;
    HdMitsubaRenderBuffer* color_buffer = nullptr;
  };

  bool scene_dirty_ = true;
  ref<Scene> scene_;
  ref<Integrator> integrator_ = nullptr;
  std::string integrator_type_ = "path";
  absl::flat_hash_map<const HdRenderPass*, RenderPassState> pass_aov_states_;
  size_t sample_count_ = kDefaultSampleCount;
  int interactive_samples_per_pass_ = 1;
  bool has_instancing_ = false;

  // Progressive rendering state
  bool progressive_rendering_ = false;
  int current_progressive_sample_ = 0;
  TensorXf accum_buffer_;
  bool reset_progressive_ = true;
  const Sensor* last_sensor_ = nullptr;  // identity-only comparison

  absl::Mutex state_mutex_;
  absl::Mutex aov_states_mutex_;

  absl::flat_hash_map<SdfPath, uint32_t, SdfPath::Hash> material_dirty_flags_;
  absl::flat_hash_set<SdfPath, SdfPath::Hash> sensor_binding_dirty_;

  absl::flat_hash_map<std::string, ref<Sensor>> sensors_;
  absl::flat_hash_map<std::string, SdfPath> shape_sensors_;  // Shape -> camera
  absl::flat_hash_map<std::string, ref<Shape>> shapes_;
  absl::flat_hash_map<std::string, ref<Emitter>> emitters_;
  absl::flat_hash_map<std::string, ref<BSDF>> bsdfs_;
  absl::flat_hash_map<std::string, MaterialDisplacement> displacements_;
  absl::flat_hash_map<std::string, mitsuba::Properties> material_emitters_;
  absl::flat_hash_map<std::string, MeshAttributeRequests> mesh_attributes_;
  ref<BSDF> default_bsdf_ = nullptr;
  absl::flat_hash_map<std::tuple<float, float, float>, ref<BSDF>> color_bsdfs_;
  absl::Mutex color_bsdfs_mutex_;

  TextureCache<Float, Spectrum> texture_cache_;
  std::unique_ptr<FrozenRender<Float, Spectrum>> frozen_render_ = nullptr;

  void PreloadTextures(const std::vector<MaterialSpec*>& pending_specs) {
    absl::flat_hash_set<TextureKey> keys;
    for (const auto* spec : pending_specs) {
      DiscoverTextures(spec->network2,
                       [&](const TextureKey& key) { keys.insert(key); });
    }
    texture_cache_.Preload(keys);
  }

  // Mitsuba's Scene only scans for filtered (trilinear/anisotropic) textures
  // in its constructor, not in parameters_changed(). If an in-place material
  // update introduces the first such texture, the scene would keep stripping
  // ray footprints, so force a rebuild. The scan deliberately mirrors
  // Scene::update_filtered_textures() (textures reachable from shapes) so
  // that a rebuild always satisfies the condition; scanning the texture cache
  // instead would re-trigger a rebuild on every material change if a filtered
  // texture is cached but not reachable from any shape (e.g. displacement).
  bool NeedsRebuildForFilteredTextures() const {
    if (!scene_ || scene_->has_filtered_textures()) return false;
    struct Scan : public mitsuba::TraversalCallback {
      bool found = false;
      void put_object(std::string_view, mitsuba::Object* obj,
                      uint32_t) override {
        if (found || !obj) return;
        if (auto* texture = dynamic_cast<Texture*>(obj)) {
          found = texture->filtered();
        }
        if (!found) obj->traverse(this);
      }
      void put_value(std::string_view, void*, uint32_t,
                     const std::type_info&) override {}
    } scan;
    for (const auto& [id, shape] : shapes_) {
      // traverse() is not const even though this scan is read-only.
      const_cast<Shape*>(shape.get())->traverse(&scan);
      if (scan.found) return true;
    }
    return false;
  }

  bool EraseShapesWithPrefix(std::string_view prefix) {
    bool erased = false;
    for (auto it = shapes_.begin(); it != shapes_.end();) {
      if (absl::StartsWith(it->first, prefix)) {
        shapes_.erase(it++);
        erased = true;
      } else {
        ++it;
      }
    }
    return erased;
  }

  bool CleanUpInstancing(const SdfPath& id) {
    std::string id_str = id.GetAsString();
    bool erased = false;
    if (shapes_.erase(absl::StrCat(kProtoPrefix, id_str)) > 0) erased = true;
    // Erase prototype sub-meshes
    if (EraseShapesWithPrefix(absl::StrCat(kProtoPrefix, id_str, "/"))) {
      erased = true;
    }
    if (shapes_.erase(absl::StrCat(kProtoGroupPrefix, id_str)) > 0)
      erased = true;
    for (size_t i = 0;; ++i) {
      if (shapes_.erase(absl::StrCat(kInstancePrefix, id_str, "_", i)) == 0) {
        break;
      }
      erased = true;
    }
    return erased;
  }

  std::vector<CopyDestination> destinations_;
};

template <typename Float, typename Spectrum>
SceneManager* CreateSceneManagerImpl() {
  return new SceneModel<Float, Spectrum>();
}

SceneManager* SceneManager::CreateSceneManager(const std::string& variant) {
  SyncMitsubaPythonVariant(variant);
  return MI_INVOKE_VARIANT(variant, CreateSceneManagerImpl);
}

SceneManager::SceneManager() {
  absl::MutexLock lock(lifecycle_mutex_);

  // Try to determine if static initialization is already done or not.
  if (active_instances_ == 0 && mitsuba::logger() == nullptr &&
      mitsuba::Thread::thread() == nullptr) {
    owns_static_initialization_ = true;
    mitsuba::Thread::static_initialization();
    mitsuba::Logger::static_initialization();
    mitsuba::Bitmap::static_initialization();

    // Append the mitsuba directory to the FileResolver search path list
    mitsuba::ref<mitsuba::FileResolver> fr = mitsuba::file_resolver();
    mitsuba::fs::path base_path = mitsuba::util::library_path().parent_path();
    if (fr && !fr->contains(base_path)) {
      fr->append(base_path);
    }
#if defined(NDEBUG)
    mitsuba::logger()->set_log_level(mitsuba::LogLevel::Warn);
    jit_set_log_level_stderr(LogLevel::Disable);
#endif
  }
  active_instances_++;
}

SceneManager::~SceneManager() {
  absl::MutexLock lock(lifecycle_mutex_);
  active_instances_--;
  if (active_instances_ == 0 && owns_static_initialization_) {
    mitsuba::Bitmap::static_shutdown();
    mitsuba::Logger::static_shutdown();
    mitsuba::Thread::static_shutdown();
    owns_static_initialization_ = false;
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
