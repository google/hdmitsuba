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
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/no_destructor.h>
#include <absl/container/btree_map.h>
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
#include <mitsuba/core/random.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <nanothread/nanothread.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/tf/type.h>
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
#include "hdmitsuba/denoiser.h"
#include "hdmitsuba/framebuffer.h"
#include "hdmitsuba/ies_emitter.h"
#include "hdmitsuba/kernel_freezing.h"
#include "hdmitsuba/mesh/displacement.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/plugin_builder.h"
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

template <typename T>
std::optional<T> GetSetting(const VtDictionary& settings,
                            const std::string& key) {
  auto it = settings.find(key);
  if (it == settings.end() || it->second.IsEmpty()) {
    return std::nullopt;
  }
  if (it->second.IsHolding<T>()) {
    return it->second.UncheckedGet<T>();
  }
  if constexpr (std::is_same_v<T, std::string>) {
    if (it->second.IsHolding<TfToken>()) {
      return it->second.UncheckedGet<TfToken>().GetString();
    }
  }
  TF_WARN("Type mismatch for render setting '%s': expected %s, got %s",
          key.c_str(), TfType::Find<T>().GetTypeName().c_str(),
          it->second.GetTypeName().c_str());
  return std::nullopt;
}

std::optional<int64_t> ExtractIntValue(const VtValue& value) {
  if (value.IsHolding<int>()) {
    return static_cast<int64_t>(value.UncheckedGet<int>());
  }
  if (value.IsHolding<int64_t>()) {
    return value.UncheckedGet<int64_t>();
  }
  if (value.IsHolding<uint32_t>()) {
    return static_cast<int64_t>(value.UncheckedGet<uint32_t>());
  }
  if (value.IsHolding<uint64_t>()) {
    return static_cast<int64_t>(value.UncheckedGet<uint64_t>());
  }
  return std::nullopt;
}

std::optional<int64_t> GetIntSetting(const VtDictionary& settings,
                                     const std::string& key) {
  auto it = settings.find(key);
  if (it == settings.end() || it->second.IsEmpty()) {
    return std::nullopt;
  }
  if (std::optional<int64_t> parsed = ExtractIntValue(it->second)) {
    return parsed;
  }
  TF_WARN("Type mismatch for integer render setting '%s': got %s", key.c_str(),
          it->second.GetTypeName().c_str());
  return std::nullopt;
}

bool ExtractHideEmitters(const PluginSpecNode& spec) {
  auto it = spec.params.find("hide_emitters");
  if (it == spec.params.end()) {
    return false;
  }
  if (it->second.IsHolding<bool>()) {
    return it->second.UncheckedGet<bool>();
  }
  if (it->second.IsHolding<int>()) {
    return it->second.UncheckedGet<int>() != 0;
  }
  if (it->second.IsHolding<int64_t>()) {
    return it->second.UncheckedGet<int64_t>() != 0;
  }
  return false;
}

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
void SetTransform(mitsuba::Shape<Float, Spectrum>* instance,
                  const MotionTransform& transform) {
  TraversalCallback cb("", nullptr, /*recurse_objects=*/transform.IsAnimated());
  instance->traverse(&cb);
  PrimTranslator<Float, Spectrum>::SetTransformParameter(cb, transform);
  instance->parameters_changed({"to_world"});
}

struct ImplicitSurfacePart {
  const char* plugin;
  std::string id;
  ScalarAffineTransform4f to_world;
  bool flip_normals = false;
};

std::vector<ImplicitSurfacePart> GetImplicitSurfaceParts(
    const SdfPath& id, const ImplicitSurface& surface,
    const ScalarAffineTransform4f& to_world) {
  using Transform = ScalarAffineTransform4f;
  using Vector3f = mitsuba::Vector<float, 3>;
  const std::string& id_str = id.GetString();
  const float r = surface.radius, h = surface.height;
  if (surface.type == HdPrimTypeTokens->sphere) {
    return {{"sphere", id_str, to_world * Transform::scale(Vector3f(r))}};
  }
  // Mitsuba's `cylinder` spans z in [0, 1] and has no caps, whereas USD's is
  // centered and closed. Mitsuba's `disk` faces +z.
  const Transform top =
      to_world * Transform::translate(Vector3f(0.f, 0.f, 0.5f * h));
  const Transform bottom =
      to_world * Transform::translate(Vector3f(0.f, 0.f, -0.5f * h));
  const Transform cap_scale = Transform::scale(Vector3f(r, r, 1.f));
  return {
      {"cylinder", absl::StrCat(id_str, "#body"),
       bottom * Transform::scale(Vector3f(r, r, h))},
      {"disk", absl::StrCat(id_str, "#cap_top"), top * cap_scale},
      {"disk", absl::StrCat(id_str, "#cap_bottom"), bottom * cap_scale,
       /*flip_normals=*/true},
  };
}

}  // namespace

template <typename Float, typename Spectrum>
class SceneModel final : public SceneManager {
 public:
  MI_IMPORT_TYPES(BSDF, Emitter, Film, Integrator, Mesh, Scene, Sensor, Shape,
                  Texture);
  using PrimTranslator = PrimTranslator<Float, Spectrum>;

  struct CommittedMesh {
    std::vector<std::pair<std::string, mitsuba::ref<Shape>>> meshes;
    mitsuba::ref<Shape> shapegroup = nullptr;
    std::vector<mitsuba::ref<Shape>> instances;
  };

  struct MeshCommitWork {
    MeshSpec* spec;
    // Backing storage of `spec` for motion-wrapped meshes, see
    // `ResolveMotionWrapped`.
    std::optional<MeshSpec> resolved_spec;
    std::vector<MaterialDisplacement> displacements;
    // Primvars read by the materials of the mesh.
    absl::flat_hash_set<std::string> requested_attributes;
  };

  static std::vector<SubMeshOutput> ProcessGeometry(
      const MeshCommitWork& work) {
    const MeshSpec& spec = *work.spec;
    return RunGeometryPipeline<Float, Spectrum>(
        spec.id, spec.geometry, spec.EffectiveTransform(), work.displacements,
        work.requested_attributes);
  }

  SceneModel() {
    if constexpr (dr::is_jit_v<Float>) {
      jit_init(1u << static_cast<uint32_t>(dr::backend_v<Float>));
    }

    default_bsdf_ = mitsuba::PluginManager::instance()
                        ->create_object<mitsuba::BSDF<Float, Spectrum>>(
                            Properties("diffuse"));
    Properties emitter_bsdf_props("diffuse");
    emitter_bsdf_props.set("reflectance", 0.0f);
    emitter_default_bsdf_ =
        PluginManager::instance()->create_object<BSDF>(emitter_bsdf_props);
    null_bsdf_ =
        mitsuba::PluginManager::instance()
            ->create_object<mitsuba::BSDF<Float, Spectrum>>(Properties("null"));
  }

  ref<BSDF> DefaultBsdf() { return default_bsdf_; }

  static const VtValue& PrimvarValue(const VtValue& v) { return v; }
  static const VtValue& PrimvarValue(const PrimvarState& s) { return s.value; }

  static bool HasMedium(const MeshSpec& spec) {
    return (spec.shape_spec.children.count("interior") > 0) ||
           (spec.shape_spec.children.count("exterior") > 0);
  }

  template <typename Map>
  ref<BSDF> ResolveBsdf(const SdfPath& material_id, const Map& primvars,
                        bool has_medium = false, bool is_emitter = false) {
    auto bsdf_it = bsdfs_.find(material_id);
    if (bsdf_it != bsdfs_.end()) {
      return bsdf_it->second;
    }
    // Fallback to the mesh's display color of specified.
    auto color_it = primvars.find(HdTokens->displayColor);
    if (color_it != primvars.end() &&
        !PrimvarValue(color_it->second).IsEmpty()) {
      const VtValue& val = PrimvarValue(color_it->second);
      GfVec3f color(0.5f, 0.5f, 0.5f);
      if (val.IsHolding<VtVec3fArray>()) {
        const auto& colors = val.UncheckedGet<VtVec3fArray>();
        if (!colors.empty()) {
          color = colors[0];
        }
      } else if (val.IsHolding<GfVec3f>()) {
        color = val.UncheckedGet<GfVec3f>();
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
    if (has_medium) return null_bsdf_;
    return is_emitter ? emitter_default_bsdf_ : DefaultBsdf();
  }

  // Returns the primvars that `material_id` reads as mesh attributes.
  const MeshAttributeRequests& ResolveMeshAttributes(
      const SdfPath& material_id) const {
    static const absl::NoDestructor<MeshAttributeRequests> kNone;
    auto it = mesh_attributes_.find(material_id);
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
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void SyncMesh(MeshSpec spec) override {
    TRACE_FUNCTION();
    TF_DEBUG(HDMITSUBA_SYNC)
        .Msg("SyncMesh: %s (%zu materials, first: %s)\n", spec.id.GetText(),
             spec.geometry.material_ids.size(),
             spec.geometry.material_ids.empty()
                 ? "none"
                 : spec.geometry.material_ids[0].GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(mesh_specs_, std::move(spec));
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void SyncCurves(CurveSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncCurves: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    curve_specs_[spec.id] = std::move(spec);
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void SyncParticleField(ParticleFieldSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncParticleField: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(particle_field_specs_, std::move(spec));
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void SyncLight(LightSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncLight: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    UpsertSpec(light_specs_, std::move(spec));
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void SyncMaterial(MaterialSpec spec) override {
    TRACE_FUNCTION();
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncMaterial: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    material_specs_[spec.id] = std::move(spec);
    texture_cache_.MarkDirty();
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void RemoveShape(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveShape: %s\n", id.GetText());
    std::string id_str = id.GetAsString();
    absl::MutexLock lock(state_mutex_);
    bool erased = mesh_specs_.erase(id) > 0 || curve_specs_.erase(id) > 0 ||
                  particle_field_specs_.erase(id) > 0;
    if (shapes_.erase(id_str) > 0) {
      erased = true;
    }

    // Erase split sub-meshes
    if (EraseShapesWithPrefix(absl::StrCat(id_str, "#"))) {
      erased = true;
    }

    if (CleanUpInstancing(id)) {
      erased = true;
    }
    if (!erased) {
      TF_RUNTIME_ERROR("Could not remove shape: %s", id_str.c_str());
    }
    scene_dirty_ = true;
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void RemoveLight(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveLight: %s\n", id.GetText());
    absl::MutexLock lock(state_mutex_);
    light_specs_.erase(id);
    shapes_.erase(id.GetAsString());
    emitters_.erase(id);
    scene_dirty_ = true;
    has_pending_commits_ = true;
    reset_progressive_ = true;
  }

  void RemoveMaterial(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveMaterial: %s\n", id.GetText());
    absl::MutexLock lock(state_mutex_);
    material_specs_.erase(id);
    bsdfs_.erase(id);
    uint32_t dirty_flags = DirtyFlags::kMaterialUpdated;
    if (displacements_.erase(id) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    if (material_emitters_.erase(id) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    if (mesh_attributes_.erase(id) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    if (interior_media_.erase(id) > 0) {
      dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
    }
    material_dirty_flags_[id] |= dirty_flags;
    texture_cache_.MarkDirty();
    scene_dirty_ = true;
    has_pending_commits_ = true;
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
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
    reset_progressive_ = true;
    if (frozen_render_) {
      frozen_render_->Clear();
    }
    pass_aov_states_[render_pass] = std::move(pass_state);
  }

  void Render(const HdRenderPass* render_pass,
              const HdCamera* camera,
              const std::optional<GfRect2i>& crop_window = std::nullopt) override {
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
    const bool denoise_dirty = std::exchange(denoise_dirty_, false);
    NewJitScope<Float>();
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

    SdfPath sensor_id;
    if (camera == nullptr) {
      if (std::string sensor_path(sensor->id());
          SdfPath::IsValidPathString(sensor_path)) {
        sensor_id = SdfPath(sensor_path);
      }
    } else {
      sensor_id = camera->GetId();
    }
    auto active_camera_it = camera_specs_.find(sensor_id);
    size_t effective_sample_count = sample_count_;
    if (active_camera_it != camera_specs_.end()) {
      if (auto sampler_it =
              active_camera_it->second.sensor_spec.children.find("sampler");
          sampler_it != active_camera_it->second.sensor_spec.children.end()) {
        effective_sample_count = 4;
        if (auto spp_it = sampler_it->second.params.find("sample_count");
            spp_it != sampler_it->second.params.end()) {
          if (std::optional<int64_t> cam_spp =
                  ExtractIntValue(spp_it->second)) {
            effective_sample_count =
                static_cast<size_t>(std::max<int64_t>(1, *cam_spp));
          }
        }
      }
    }
    if (effective_sample_count != active_sample_count_) {
      active_sample_count_ = effective_sample_count;
      reset_progressive_ = true;
    }

    const bool is_signal_sensor =
        sensor->class_name() == "RadianceMeter" ||
        absl::StartsWith(sensor->class_name(), "DistantSensor");
    Film* film = sensor->film();
    bool film_changed = false;
    // Only output alpha if a 4-channel color buffer requests it, since the
    // extra film channel can make accumulation significantly more expensive.
    if constexpr (!mitsuba::is_monochromatic_v<Spectrum>) {
      const bool want_alpha =
          pass_state.color_buffer &&
          HdGetComponentCount(pass_state.color_buffer->GetFormat()) == 4;
      if (want_alpha != (film->base_channels().size() == 4) &&
          film->class_name() == "HDRFilm" && !film->base_channels().empty() &&
          film->base_channels()[0] == "R") {
        const PluginSpecNode* film_spec = nullptr;
        if (active_camera_it != camera_specs_.end()) {
          if (auto film_it =
                  active_camera_it->second.sensor_spec.children.find("film");
              film_it != active_camera_it->second.sensor_spec.children.end()) {
            film_spec = &film_it->second;
          }
        }
        bool supports_alpha_format = true;
        if (film_spec != nullptr) {
          if (film_spec->HasProperty("pixel_format")) {
            supports_alpha_format = false;
          } else if (auto fmt_it = film_spec->params.find("file_format");
                     fmt_it != film_spec->params.end()) {
            std::string file_format;
            if (fmt_it->second.IsHolding<std::string>()) {
              file_format = fmt_it->second.UncheckedGet<std::string>();
            } else if (fmt_it->second.IsHolding<TfToken>()) {
              file_format =
                  fmt_it->second.UncheckedGet<TfToken>().GetString();
            }
            if (!file_format.empty() && file_format != "openexr") {
              supports_alpha_format = false;
            }
          }
        }
        if (supports_alpha_format) {
          PluginSpecNode rebuilt_film =
              film_spec == nullptr ? PluginSpecNode{} : *film_spec;
          rebuilt_film.plugin_type = "hdrfilm";
          rebuilt_film.params.erase("crop_offset_x");
          rebuilt_film.params.erase("crop_offset_y");
          rebuilt_film.params.erase("crop_width");
          rebuilt_film.params.erase("crop_height");
          if (is_signal_sensor) {
            rebuilt_film.params["width"] = VtValue(1);
            rebuilt_film.params["height"] = VtValue(1);
          }
          rebuilt_film.params["pixel_format"] =
              VtValue(std::string(want_alpha ? "rgba" : "rgb"));
          if (progressive_rendering_ || denoise_) {
            rebuilt_film.children.erase("reconstruction_filter");
            rebuilt_film.children.erase("rfilter");
            rebuilt_film.children["pixel_filter"] =
                PluginSpecNode("box");
          }
          sensor->set_film(BuildPluginObject<Film>(rebuilt_film).get());
          film = sensor->film();
          film_changed = true;
        }
      }
    }
    auto film_size = film->size();
    // Signal sensors (`radiancemeter`, `distant`) only support 1x1 films; keep
    // their native 1x1 size and tile the output pixel across larger buffers.
    if (!is_signal_sensor &&
        (film_size.x() != buffer_width || film_size.y() != buffer_height)) {
      film->set_size(ScalarPoint2u(buffer_width, buffer_height));
      if (active_camera_it != camera_specs_.end() && buffer_height > 0) {
        PrimTranslator::UpdateSensorAspectRatio(
            sensor, active_camera_it->second,
            static_cast<double>(buffer_width) /
                static_cast<double>(buffer_height));
      }
      film_changed = true;
    }

    const bool has_hydra_crop =
        crop_window.has_value() &&
        (crop_window->GetMinX() != 0 || crop_window->GetMinY() != 0 ||
         crop_window->GetWidth() != static_cast<int>(buffer_width) ||
         crop_window->GetHeight() != static_cast<int>(buffer_height));
    std::optional<GfRect2i> effective_crop_window;
    if (!is_signal_sensor && has_hydra_crop) {
      effective_crop_window = crop_window;
    } else if (!is_signal_sensor && active_camera_it != camera_specs_.end() &&
               buffer_width > 0 && buffer_height > 0) {
      if (auto film_it =
              active_camera_it->second.sensor_spec.children.find("film");
          film_it != active_camera_it->second.sensor_spec.children.end()) {
        const auto& film_params = film_it->second.params;
        auto read_crop_int = [&](const char* key) -> std::optional<int64_t> {
          auto it = film_params.find(key);
          return it != film_params.end() ? ExtractIntValue(it->second)
                                         : std::nullopt;
        };
        const auto ox_opt = read_crop_int("crop_offset_x");
        const auto oy_opt = read_crop_int("crop_offset_y");
        const auto cw_opt = read_crop_int("crop_width");
        const auto ch_opt = read_crop_int("crop_height");
        if (ox_opt.has_value() || oy_opt.has_value() || cw_opt.has_value() ||
            ch_opt.has_value()) {
          const int64_t w = static_cast<int64_t>(buffer_width);
          const int64_t h = static_cast<int64_t>(buffer_height);
          const int ox = static_cast<int>(
              std::clamp<int64_t>(ox_opt.value_or(0), 0, w - 1));
          const int oy = static_cast<int>(
              std::clamp<int64_t>(oy_opt.value_or(0), 0, h - 1));
          const int cw = static_cast<int>(
              std::clamp<int64_t>(cw_opt.value_or(w - ox), 1, w - ox));
          const int ch = static_cast<int>(
              std::clamp<int64_t>(ch_opt.value_or(h - oy), 1, h - oy));
          if (ox != 0 || oy != 0 || cw != static_cast<int>(buffer_width) ||
              ch != static_cast<int>(buffer_height)) {
            effective_crop_window = GfRect2i(GfVec2i(ox, oy), cw, ch);
          }
        }
      }
    }

    ScalarPoint2u new_crop_offset(0, 0);
    ScalarVector2u new_crop_size = film->size();
    if (effective_crop_window.has_value()) {
      new_crop_offset = ScalarPoint2u(effective_crop_window->GetMinX(),
                                      effective_crop_window->GetMinY());
      new_crop_size = ScalarVector2u(effective_crop_window->GetWidth(),
                                     effective_crop_window->GetHeight());
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

    EnsureIntegrator();

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

    int samples_to_render = active_sample_count_;
    if (progressive_rendering_) {
      int remaining =
          static_cast<int>(active_sample_count_) - current_progressive_sample_;
      samples_to_render =
          std::max(0, std::min(interactive_samples_per_pass_, remaining));
    }
    TensorXf display_result;
    if (samples_to_render > 0) {
      // 3. Render a chunk of samples
      TensorXf result;
      uint32_t pass_seed =
          progressive_rendering_
              ? mitsuba::sample_tea_32(
                    seed_, static_cast<uint32_t>(current_progressive_sample_))
                    .first
              : seed_;
      if (frozen_render_) {
        result = frozen_render_->Render(
            scene_.get(), sensor, render_integrator, pass_seed,
            static_cast<uint32_t>(samples_to_render));
      } else {
        result = render_integrator->render(
            scene_.get(), sensor, pass_seed,
            static_cast<uint32_t>(samples_to_render), true, true);
      }
      size_t expected_width =
          is_signal_sensor ? 1
                           : (effective_crop_window.has_value()
                                  ? effective_crop_window->GetWidth()
                                  : buffer_width);
      size_t expected_height =
          is_signal_sensor ? 1
                           : (effective_crop_window.has_value()
                                  ? effective_crop_window->GetHeight()
                                  : buffer_height);
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
        display_result = std::move(result);
      }
    } else if (denoise_dirty && !accum_buffer_.empty() &&
               current_progressive_sample_ > 0) {
      display_result =
          TensorXf(accum_buffer_.array() /
                       static_cast<float>(current_progressive_sample_),
                   accum_buffer_.shape());
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
    size_t aov_start_channel =
        base_channels +
        (integrator_ ? integrator_->aov_names(sensor->film()).size() : 0);
    if (!is_signal_sensor && denoise_ && pass_state.color_buffer &&
        base_channels >= 3) {
      denoiser_.Denoise(display_result, sensor, pass_state.aov_requests,
                        base_channels, aov_start_channel);
    }
    size_t total_channels = display_result.shape()[2];
    if (is_signal_sensor && display_result.shape()[0] == 1 &&
        display_result.shape()[1] == 1 &&
        (buffer_width != 1 || buffer_height != 1)) {
      const size_t tiled_shape[3] = {buffer_height, buffer_width,
                                     total_channels};
      display_result = TensorXf(
          dr::tile(display_result.array(),
                   static_cast<size_t>(buffer_width) * buffer_height),
          3, tiled_shape);
    }
    destinations_.clear();
    destinations_.reserve(1 + pass_state.aov_requests.size());

    if (pass_state.color_buffer) {
      destinations_.push_back({pass_state.color_buffer, 0, static_cast<int>(base_channels), false});
    }

    size_t current_offset = aov_start_channel;
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
    PerformBatchedCopy<Float>(display_result, destinations_,
                              effective_crop_window);

    // 6. Set convergence status
    bool converged =
        !progressive_rendering_ ||
        (current_progressive_sample_ >= static_cast<int>(active_sample_count_));
    if (pass_state.color_buffer) {
      pass_state.color_buffer->SetConverged(converged);
    }
    for (auto& req : pass_state.aov_requests) {
      req.buffer->SetConverged(converged);
    }
  }

  bool IsConverged() const override {
    return !progressive_rendering_ ||
           (!denoise_dirty_ && (current_progressive_sample_ >=
                                static_cast<int>(active_sample_count_)));
  }

  int GetCurrentSampleCount() const override {
    return progressive_rendering_ ? current_progressive_sample_
                                  : static_cast<int>(active_sample_count_);
  }

  int GetTargetSampleCount() const override {
    return static_cast<int>(active_sample_count_);
  }

  void UpdateNamespacedSettings(
      const VtDictionary& namespaced_settings) override {
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
    const int new_samples_per_pass = static_cast<int>(std::max<int64_t>(
        1, GetIntSetting(namespaced_settings,
                         "mitsuba:interactive_samples_per_pass")
               .value_or(1)));
    if (new_samples_per_pass != interactive_samples_per_pass_) {
      interactive_samples_per_pass_ = new_samples_per_pass;
      reset_progressive_ = true;
    }
    PluginSpecNode integrator_spec = ParsePluginSpec(
        namespaced_settings, "mitsuba:integrator:", /*default_type=*/"path");
    if (integrator_spec != integrator_spec_) {
      const bool new_hide_emitters = ExtractHideEmitters(integrator_spec);
      if (new_hide_emitters != hide_emitters_) {
        if (hide_emitters_ && !new_hide_emitters) {
          MarkEmittersForRebuild();
        }
        hide_emitters_ = new_hide_emitters;
        scene_dirty_ = true;
      }
      integrator_spec_ = std::move(integrator_spec);
      integrator_ = nullptr;
      for (auto& [_, pass_state] : pass_aov_states_) {
        pass_state.aov_integrator = nullptr;
      }
      if (frozen_render_) frozen_render_->Clear();
      reset_progressive_ = true;
    }
    PluginSpecNode sampler_spec =
        ParsePluginSpec(namespaced_settings, "mitsuba:sampler:",
                        /*default_type=*/"independent");
    if (sampler_spec.plugin_type.empty()) {
      sampler_spec.plugin_type = "independent";
    }
    // sample_count is managed via sample_count_ and passed per-render pass,
    // so omit it from sampler_spec_ to avoid rebuilding sensors when only
    // mitsuba:sample_count changes.
    sampler_spec.params.erase("sample_count");
    if (sampler_spec != sampler_spec_) {
      TF_DEBUG(HDMITSUBA_LIFECYCLE)
          .Msg("Updating sampler spec, type: %s (was: %s)\n",
               sampler_spec.plugin_type.c_str(),
               sampler_spec_.plugin_type.c_str());
      sampler_spec_ = std::move(sampler_spec);
      MarkSensorsForRebuild();
      reset_progressive_ = true;
    }
    const size_t clamped_samples = static_cast<size_t>(std::max<int64_t>(
        1, GetIntSetting(namespaced_settings, "mitsuba:sample_count")
               .value_or(static_cast<int64_t>(kDefaultSampleCount))));
    if (clamped_samples != sample_count_) {
      sample_count_ = clamped_samples;
      active_sample_count_ = clamped_samples;
      reset_progressive_ = true;
    }
    const uint32_t new_seed = static_cast<uint32_t>(
        GetIntSetting(namespaced_settings, "mitsuba:seed").value_or(0));
    if (new_seed != seed_) {
      seed_ = new_seed;
      reset_progressive_ = true;
    }
    auto progressive =
        GetSetting<bool>(namespaced_settings, "enableInteractive");
    if (progressive.has_value()) {
      if (*progressive != progressive_rendering_) {
        progressive_rendering_ = *progressive;
        // A change in render mode requires rebuilding the sensors, since
        // this affects the used pixel filter type.
        MarkSensorsForRebuild();
        reset_progressive_ = true;
      }
    }
    auto enabled =
        GetSetting<bool>(namespaced_settings, "mitsuba:use_kernel_freezing");
    if (enabled.has_value()) {
      TF_DEBUG(HDMITSUBA_LIFECYCLE)
          .Msg("Kernel freezing setting: %d\n", *enabled);
      if constexpr (dr::is_jit_v<Float>) {
        if (*enabled) {
          if (!frozen_render_) {
            frozen_render_ = std::make_unique<FrozenRender<Float, Spectrum>>(
                dr::backend_v<Float>);
          }
        } else {
          frozen_render_.reset(); // Immediately destroys it and frees all JIT GPU memory!
        }
      }
    }
    const bool new_denoise =
        GetSetting<bool>(namespaced_settings, "mitsuba:denoise")
            .value_or(false);
    if (new_denoise != denoise_) {
      denoise_ = new_denoise;
      denoise_dirty_ = true;
      // Batch sensors switch to the box filter while denoising.
      if (!progressive_rendering_) {
        MarkSensorsForRebuild();
      }
    }
  }

  enum DirtyFlags : uint32_t {
    kClean = 0,
    kNeedsStructureRebuild = 1 << 0,
    kMaterialUpdated = 1 << 1
  };

  uint32_t UpdateMaterialStructure(const SdfPath& id,
                                   TranslatedMaterial& trans) {
    uint32_t dirty_flags = 0;
    auto update = [&](auto& map, auto* val) {
      if (val) {
        auto it = map.find(id);
        if (it == map.end() || it->second != *val) {
          map[id] = std::move(*val);
          dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
        }
      } else if (map.erase(id) > 0) {
        dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
      }
    };
    update(displacements_,
           trans.displacement.texture ? &trans.displacement : nullptr);
    update(material_emitters_,
           trans.shape_emitter_props ? &*trans.shape_emitter_props : nullptr);
    update(mesh_attributes_,
           !trans.mesh_attributes.empty() ? &trans.mesh_attributes : nullptr);
    update(interior_media_,
           trans.interior_medium ? &trans.interior_medium : nullptr);
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
        for (const auto& mat_id : spec.geometry.material_ids) {
          apply_material_dirty(mat_id, spec, needs_bsdf_update);
        }
        if (sensor_binding_dirty_.contains(spec.id)) {
          spec.needs_rebuild = true;
        }
      } else if constexpr (std::is_same_v<SpecType, CurveSpec>) {
        apply_material_dirty(spec.material_id, spec, needs_bsdf_update);
        if (sensor_binding_dirty_.contains(spec.id)) {
          spec.needs_rebuild = true;
        }
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
    JitParallelFor<Float>(pending_specs.size(), [&](size_t i) {
      work_fn(pending_specs[i], results[i]);
    });

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
          if (trans.bsdf) {
            bsdfs_[spec->id] = dynamic_cast<BSDF*>(trans.bsdf.get());
          } else {
            bsdfs_.erase(spec->id);
          }
          material_dirty_flags_[spec->id] |=
              UpdateMaterialStructure(spec->id, trans) |
              DirtyFlags::kMaterialUpdated;
          return false;
        });
  }

  static std::optional<SdfPath> SurfaceSensorTarget(const CameraSpec& spec) {
    return spec.target_shape_id;
  }

  void RebuildShapeSensorMap() {
    shape_sensors_.clear();
    for (const auto& [camera_id, spec] : camera_specs_) {
      std::optional<SdfPath> target = SurfaceSensorTarget(spec);
      if (!target.has_value()) {
        continue;
      }
      if (!shape_sensors_.try_emplace(*target, camera_id).second) {
        TF_WARN(
            "Shape %s is already measured by another sensor; ignoring sensor "
            "%s.",
            target->GetText(), camera_id.GetText());
      }
    }
  }

  bool CommitCameras(std::atomic<bool>* host_scalars_changed) {
    bool rebuild = ParallelCommit<decltype(camera_specs_), mitsuba::ref<Sensor>>(
        camera_specs_,
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          // Surface sensors are built together with their target shape.
          if (spec->target_shape_id.has_value() ||
              spec->sensor_type == "irradiancemeter") {
            if (spec->sensor_type == "irradiancemeter" &&
                !spec->target_shape_id.has_value()) {
              TF_WARN(
                  "Camera %s has mitsuba:sensor:type = 'irradiancemeter' "
                  "without mitsuba:sensor:shape; skipping.",
                  spec->id.GetText());
            }
            return;
          }
          auto build_sensor = [&]() {
            // Denoisers require the uncorrelated pixel noise of the box filter,
            // and progressive multi-pass accumulation requires `independent`
            // so structured samplers (`stratified`, `multijitter`,
            // `orthogonal`) do not collapse per-pass small sample counts into
            // biased grids.
            const bool use_box_filter = progressive_rendering_ || denoise_;
            const PluginSpecNode effective_sampler =
                progressive_rendering_
                    ? PluginSpecNode("independent")
                    : sampler_spec_;
            try {
              res = PrimTranslator::BuildSensor(
                  *spec, use_box_filter, effective_sampler,
                  /*force_independent_sampler=*/progressive_rendering_);
            } catch (const std::exception& e) {
              TF_RUNTIME_ERROR("Failed to create sensor '%s': %s",
                               spec->id.GetText(), e.what());
            }
          };
          if (spec->needs_rebuild) {
            build_sensor();
          } else if (spec->dirty_bits != 0) {
            auto it = sensors_.find(spec->id);
            if (!TF_VERIFY(it != sensors_.end(), "Camera sensor not found: %s",
                           spec->id.GetText())) {
              return;
            }
            bool wrote_host_scalar = false;
            if (!PrimTranslator::TryUpdateSensorInPlace(
                    it->second.get(), *spec, &wrote_host_scalar)) {
              spec->needs_rebuild = true;
              build_sensor();
            } else if (wrote_host_scalar) {
              host_scalars_changed->store(true, std::memory_order_relaxed);
            }
          }
        },
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          if (spec->target_shape_id.has_value() ||
              spec->sensor_type == "irradiancemeter" || !res) {
            return sensors_.erase(spec->id) > 0;
          }
          sensors_[spec->id] = res;
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

  template <typename SpecType>
  EmitterSensorPair ResolveEmitterAndSensor(const SpecType& spec,
                                            const SdfPath& material_id,
                                            bool attach_sensor = true) {
    EmitterSensorPair pair;
    if (spec.emitter_spec.has_value()) {
      try {
        pair.mesh_emitter = PrimTranslator::CreateAreaEmitter(
            *spec.emitter_spec, spec.mitsuba_emitter_spec);
        pair.emitter_ptr = pair.mesh_emitter.get();
      } catch (const std::exception& e) {
        TF_RUNTIME_ERROR("Failed to create area emitter for '%s': %s",
                         spec.id.GetText(), e.what());
      }
    }
    if (pair.emitter_ptr == nullptr) {
      auto emitter_it = material_emitters_.find(material_id);
      if (emitter_it != material_emitters_.end()) {
        try {
          mitsuba::Properties emitter_props = emitter_it->second;
          if (!spec.mitsuba_emitter_spec.plugin_type.empty() &&
              spec.mitsuba_emitter_spec.plugin_type !=
                  emitter_props.plugin_name()) {
            emitter_props =
                mitsuba::Properties(spec.mitsuba_emitter_spec.plugin_type);
          } else if (spec.double_sided) {
            emitter_props.set("twosided", true, false);
          }
          MergePluginProperties(&emitter_props, spec.mitsuba_emitter_spec,
                                mitsuba::Emitter<Float, Spectrum>::Variant);
          pair.mesh_emitter = mitsuba::PluginManager::instance()->create_object(
              emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
              mitsuba::Emitter<Float, Spectrum>::Type);
          pair.emitter_ptr = pair.mesh_emitter.get();
        } catch (const std::exception& e) {
          TF_RUNTIME_ERROR("Failed to create emitter for '%s': %s",
                           spec.id.GetText(), e.what());
        }
      } else if (!spec.mitsuba_emitter_spec.plugin_type.empty()) {
        try {
          pair.mesh_emitter =
              BuildPluginObject(spec.mitsuba_emitter_spec,
                                mitsuba::Emitter<Float, Spectrum>::Variant,
                                mitsuba::Emitter<Float, Spectrum>::Type);
          pair.emitter_ptr = pair.mesh_emitter.get();
        } catch (const std::exception& e) {
          TF_RUNTIME_ERROR("Failed to create emitter for '%s': %s",
                           spec.id.GetText(), e.what());
        }
      }
    }
    // A Mitsuba sensor can only be attached to one shape, so like the area
    // emitter above, every (re)built mesh gets a fresh one.
    if (attach_sensor) {
      auto sens_it = shape_sensors_.find(spec.id);
      if (sens_it != shape_sensors_.end()) {
        const PluginSpecNode effective_sampler =
            progressive_rendering_
                ? PluginSpecNode("independent")
                : sampler_spec_;
        try {
          pair.mesh_sensor = PrimTranslator::BuildSensor(
              camera_specs_.at(sens_it->second),
              progressive_rendering_ || denoise_, effective_sampler,
              /*force_independent_sampler=*/progressive_rendering_);
          pair.sensor_ptr = pair.mesh_sensor.get();
        } catch (const std::exception& e) {
          TF_RUNTIME_ERROR("Failed to create surface sensor '%s': %s",
                           sens_it->second.GetText(), e.what());
        }
      }
    }
    return pair;
  }

  static constexpr HdDirtyBits kMeshGeometryDirty =
      HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTransform |
      HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar |
      HdChangeTracker::DirtyTopology;

  static PluginSpecNode StripManagedShapeSpecFields(PluginSpecNode spec,
                                                    bool is_prototype,
                                                    size_t part_index = 0,
                                                    size_t part_count = 1) {
    spec.params.erase("to_world");
    spec.params.erase("flip_normals");
    if (is_prototype || part_count > 1) {
      spec.id.clear();
    }
    if (is_prototype) {
      spec.children.erase("emitter");
      spec.children.erase("sensor");
    } else if (part_index > 0) {
      spec.children.erase("sensor");
    }
    return spec;
  }

  template <typename Map>
  bool TryUpdateShapePluginSpecsAndBsdf(Shape* shape, const MeshSpec& spec,
                                        const SdfPath& material_id,
                                        const Map& primvars,
                                        std::string_view key_prefix,
                                        size_t part_index = 0,
                                        size_t part_count = 1,
                                        bool* wrote_host_scalar = nullptr) {
    if (!spec.committed_shape_spec.has_value() ||
        !spec.committed_mitsuba_emitter_spec.has_value()) {
      return false;
    }
    const bool is_prototype = !key_prefix.empty();
    const PluginSpecNode old_shape_spec = StripManagedShapeSpecFields(
        *spec.committed_shape_spec, is_prototype, part_index, part_count);
    const PluginSpecNode new_shape_spec = StripManagedShapeSpecFields(
        spec.shape_spec, is_prototype, part_index, part_count);
    if (!PrimTranslator::TryUpdatePluginInPlace(
            shape, old_shape_spec, new_shape_spec, wrote_host_scalar)) {
      return false;
    }
    if (!is_prototype && spec.dirty_bits != 0 &&
        spec.shape_spec.children.count("emitter") == 0) {
      if (spec.emitter_spec.has_value()) {
        if (!shape->is_emitter()) return false;
        if (!PrimTranslator::TryUpdateAreaEmitterInPlace(
                shape->emitter(), *spec.emitter_spec,
                *spec.committed_mitsuba_emitter_spec, spec.mitsuba_emitter_spec,
                wrote_host_scalar)) {
          return false;
        }
      } else if (*spec.committed_mitsuba_emitter_spec !=
                 spec.mitsuba_emitter_spec) {
        if (!shape->is_emitter()) return false;
        if (!PrimTranslator::TryUpdatePluginInPlace(
                shape->emitter(), *spec.committed_mitsuba_emitter_spec,
                spec.mitsuba_emitter_spec, wrote_host_scalar)) {
          return false;
        }
        shape->parameters_changed({"emitter"});
      }
    }
    Object* interior_medium = ResolveInteriorMedium(material_id);
    ref<BSDF> bsdf = ResolveBsdf(
        material_id, primvars, HasMedium(spec) || interior_medium != nullptr,
        shape->is_emitter());
    if (spec.shape_spec.children.count("bsdf") == 0) {
      shape->set_bsdf(bsdf.get());
    }
    return true;
  }

  bool TryUpdateSubMeshesInPlace(const MeshSpec& spec,
                                 const std::vector<SubMeshOutput>& sub_meshes,
                                 std::string_view key_prefix,
                                 bool* wrote_host_scalar) {
    const bool geometry_dirty = (spec.dirty_bits & kMeshGeometryDirty) != 0;
    if (geometry_dirty) {
      for (const auto& sub_mesh : sub_meshes) {
        std::string key = absl::StrCat(key_prefix, sub_mesh.id);
        auto it = shapes_.find(key);
        if (it == shapes_.end() ||
            PrimTranslator::HasDirectionalAreaEmitter(it->second.get())) {
          return false;
        }
      }
    }
    for (size_t i = 0; i < sub_meshes.size(); ++i) {
      const auto& sub_mesh = sub_meshes[i];
      std::string key = absl::StrCat(key_prefix, sub_mesh.id);
      auto it = shapes_.find(key);
      if (!TF_VERIFY(it != shapes_.end(), "Sub-mesh not found: %s",
                     key.c_str())) {
        return false;
      }
      if (!TryUpdateShapePluginSpecsAndBsdf(
              it->second.get(), spec, sub_mesh.material_id, sub_mesh.primvars,
              key_prefix, i, sub_meshes.size(), wrote_host_scalar)) {
        return false;
      }
    }
    if (geometry_dirty) {
      for (const auto& sub_mesh : sub_meshes) {
        std::string key = absl::StrCat(key_prefix, sub_mesh.id);
        PrimTranslator::UpdateMeshInPlace(
            shapes_.at(key).get(), sub_mesh.triangles, sub_mesh.primvars,
            ResolveMeshAttributes(sub_mesh.material_id), spec.dirty_bits);
      }
    }
    return true;
  }

  bool TryUpdateSubMeshesMetadataOnly(const MeshSpec& spec,
                                      std::string_view key_prefix,
                                      bool* wrote_host_scalar) {
    if (spec.geometry.material_ids.size() <= 1) {
      std::string key = absl::StrCat(key_prefix, spec.id.GetAsString());
      auto it = shapes_.find(key);
      if (it == shapes_.end()) {
        // Empty mesh (e.g. all faces marked as holes) was not emitted.
        return true;
      }
      const SdfPath& mat_id = spec.geometry.material_ids.empty()
                                  ? SdfPath()
                                  : spec.geometry.material_ids[0];
      return TryUpdateShapePluginSpecsAndBsdf(
          it->second.get(), spec, mat_id, spec.geometry.primvars, key_prefix, 0,
          1, wrote_host_scalar);
    }
    std::vector<std::pair<Shape*, const SdfPath*>> emitted;
    emitted.reserve(spec.geometry.material_ids.size());
    for (size_t i = 0; i < spec.geometry.material_ids.size(); ++i) {
      std::string key = absl::StrCat(
          key_prefix, GeometryProcessor::MakeSubMeshId(
                          spec.id, spec.geometry.material_ids[i], i));
      auto it = shapes_.find(key);
      if (it == shapes_.end()) {
        // Empty material subset was not emitted by SplitAndCompactMeshes.
        continue;
      }
      emitted.emplace_back(it->second.get(), &spec.geometry.material_ids[i]);
    }
    for (size_t i = 0; i < emitted.size(); ++i) {
      if (!TryUpdateShapePluginSpecsAndBsdf(
              emitted[i].first, spec, *emitted[i].second,
              spec.geometry.primvars, key_prefix, i, emitted.size(),
              wrote_host_scalar)) {
        return false;
      }
    }
    return true;
  }

  Object* ResolveInteriorMedium(const SdfPath& material_id) {
    auto it = interior_media_.find(material_id);
    return it != interior_media_.end() ? it->second.get() : nullptr;
  }

  // Builds the native shapes of the implicit surface of `spec` into
  // `res.meshes`, or updates the existing ones in place. Instance prototypes
  // are identified by a non-empty `key_prefix` and carry no emitter or sensor.
  void CommitImplicitSurfaceWork(MeshSpec& spec, std::string_view key_prefix,
                                 CommittedMesh& res,
                                 std::atomic<bool>* host_scalars_changed) {
    const ImplicitSurface& surface = *spec.implicit_surface;
    const ScalarAffineTransform4f to_world =
        UsdToMitsubaTransform(spec.transform);
    const SdfPath& material_id = spec.geometry.material_ids[0];
    Object* interior_medium = ResolveInteriorMedium(material_id);
    ref<BSDF> bsdf = ResolveBsdf(
        material_id, spec.geometry.primvars,
        HasMedium(spec) || interior_medium != nullptr,
        key_prefix.empty() && HasEmitter(spec));
    const std::vector<ImplicitSurfacePart> parts =
        GetImplicitSurfaceParts(spec.id, surface, to_world);
    if (!spec.needs_rebuild) {
      bool updated_all = true;
      if ((spec.dirty_bits & kMeshGeometryDirty) != 0) {
        for (const ImplicitSurfacePart& part : parts) {
          std::string key = absl::StrCat(key_prefix, part.id);
          auto it = shapes_.find(key);
          if (it == shapes_.end() ||
              PrimTranslator::HasDirectionalAreaEmitter(it->second.get())) {
            updated_all = false;
            break;
          }
        }
      }
      bool wrote_host_scalar = false;
      for (size_t i = 0; updated_all && i < parts.size(); ++i) {
        std::string key = absl::StrCat(key_prefix, parts[i].id);
        auto it = shapes_.find(key);
        if (!TF_VERIFY(it != shapes_.end(), "Implicit surface not found: %s",
                       key.c_str())) {
          updated_all = false;
          break;
        }
        if (!TryUpdateShapePluginSpecsAndBsdf(
                it->second.get(), spec, material_id, spec.geometry.primvars,
                key_prefix, i, parts.size(), &wrote_host_scalar)) {
          updated_all = false;
          break;
        }
      }
      if (updated_all) {
        if (wrote_host_scalar) {
          host_scalars_changed->store(true, std::memory_order_relaxed);
        }
        if ((spec.dirty_bits & kMeshGeometryDirty) != 0) {
          for (const ImplicitSurfacePart& part : parts) {
            std::string key = absl::StrCat(key_prefix, part.id);
            SetTransform(shapes_.at(key).get(),
                         MotionTransform::Static(part.to_world));
          }
        }
        return;
      }
      spec.needs_rebuild = true;
    }

    const bool is_prototype = !key_prefix.empty();
    if (displacements_.contains(material_id)) {
      TF_WARN(
          "Displacement on %s is not supported. Set "
          "HDMITSUBA_TESSELLATE_IMPLICIT_SURFACES=1 to tessellate it instead.",
          spec.id.GetText());
    }
    if (!is_prototype && !surface.SupportsTransform(to_world)) {
      TF_WARN(
          "%s has a transform with non-uniform scaling or shearing, which "
          "requires a Mitsuba instance, but also an emitter or surface sensor, "
          "which Mitsuba does not support on instances. It renders distorted.",
          spec.id.GetText());
    }
    if (!is_prototype && parts.size() > 1 &&
        shape_sensors_.contains(spec.id)) {
      TF_WARN("The surface sensor of %s only measures its side, not its caps.",
              spec.id.GetText());
    }
    PluginSpecNode part_shape_spec = spec.shape_spec;
    if (is_prototype || parts.size() > 1) {
      part_shape_spec.id.clear();
    }
    if (is_prototype) {
      part_shape_spec.children.erase("emitter");
      part_shape_spec.children.erase("sensor");
    }
    bool extra_flip_normals = false;
    if (auto flip_it = part_shape_spec.params.find("flip_normals");
        flip_it != part_shape_spec.params.end() &&
        flip_it->second.IsHolding<bool>()) {
      extra_flip_normals = flip_it->second.UncheckedGet<bool>();
      part_shape_spec.params.erase(flip_it);
    }
    part_shape_spec.params.erase("to_world");
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i == 1) {
        part_shape_spec.children.erase("sensor");
      }
      Properties props(parts[i].plugin);
      props.set("to_world", parts[i].to_world);
      props.set("flip_normals", parts[i].flip_normals != extra_flip_normals);
      if (interior_medium) props.set("interior", interior_medium);
      // Every part needs an emitter of its own, whereas the sensor can only be
      // attached to the first one.
      if (!is_prototype) {
        auto env = ResolveEmitterAndSensor(spec, material_id,
                                           /*attach_sensor=*/i == 0);
        if (env.emitter_ptr) props.set("emitter", env.emitter_ptr);
        if (env.sensor_ptr) props.set("sensor", env.sensor_ptr);
      }
      if (!part_shape_spec.Empty()) {
        try {
          MergePluginProperties(&props, part_shape_spec, Shape::Variant);
        } catch (const std::exception& e) {
          TF_RUNTIME_ERROR("Failed to apply shape properties for '%s': %s",
                           parts[i].id.c_str(), e.what());
        }
      }
      if (bsdf && !props.has_property("bsdf")) {
        props.set("bsdf", bsdf.get());
      }
      try {
        auto shape = PluginManager::instance()->create_object<Shape>(props);
        if (!shape) {
          continue;
        }
        shape->set_id(!part_shape_spec.id.empty() ? part_shape_spec.id
                                                  : parts[i].id);
        res.meshes.emplace_back(parts[i].id, std::move(shape));
      } catch (const std::exception& e) {
        TF_RUNTIME_ERROR("Failed to create shape '%s': %s",
                         parts[i].id.c_str(), e.what());
      }
    }
  }

  void CommitNonInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res,
                                  std::atomic<bool>* host_scalars_changed,
                                  std::string_view key_prefix = "") {
    MeshSpec& spec = *(work->spec);
    if (spec.implicit_surface.has_value()) {
      CommitImplicitSurfaceWork(spec, key_prefix, res, host_scalars_changed);
      return;
    }
    std::optional<std::vector<SubMeshOutput>> cached_sub_meshes;
    if (!spec.needs_rebuild) {
      bool wrote_host_scalar = false;
      if ((spec.dirty_bits & kMeshGeometryDirty) != 0) {
        cached_sub_meshes = ProcessGeometry(*work);
        if (TryUpdateSubMeshesInPlace(spec, *cached_sub_meshes, key_prefix,
                                      &wrote_host_scalar)) {
          if (wrote_host_scalar) {
            host_scalars_changed->store(true, std::memory_order_relaxed);
          }
          return;
        }
      } else if (TryUpdateSubMeshesMetadataOnly(spec, key_prefix,
                                                &wrote_host_scalar)) {
        if (wrote_host_scalar) {
          host_scalars_changed->store(true, std::memory_order_relaxed);
        }
        return;
      }
      spec.needs_rebuild = true;
    }
    auto sub_meshes = cached_sub_meshes.has_value()
                          ? std::move(*cached_sub_meshes)
                          : ProcessGeometry(*work);
    const bool is_prototype = !key_prefix.empty();
    if (!is_prototype && sub_meshes.size() > 1 &&
        shape_sensors_.contains(spec.id)) {
      TF_WARN(
          "The surface sensor of %s only measures its first material subset.",
          spec.id.GetText());
    }
    const bool has_medium = HasMedium(spec);
    PluginSpecNode sub_shape_spec = spec.shape_spec;
    if (is_prototype || sub_meshes.size() > 1) {
      sub_shape_spec.id.clear();
    }
    if (is_prototype) {
      sub_shape_spec.children.erase("emitter");
      sub_shape_spec.children.erase("sensor");
    }
    res.meshes.reserve(sub_meshes.size());
    for (size_t i = 0; i < sub_meshes.size(); ++i) {
      if (i == 1) {
        sub_shape_spec.children.erase("sensor");
      }
      const auto& sub_mesh = sub_meshes[i];
      EmitterSensorPair env;
      if (!is_prototype) {
        env = ResolveEmitterAndSensor(spec, sub_mesh.material_id,
                                      /*attach_sensor=*/i == 0);
      }
      Object* interior_medium = ResolveInteriorMedium(sub_mesh.material_id);
      ref<BSDF> bsdf = ResolveBsdf(
          sub_mesh.material_id, sub_mesh.primvars,
          has_medium || interior_medium != nullptr,
          env.emitter_ptr != nullptr ||
              sub_shape_spec.children.count("emitter") > 0);
      auto mesh = PrimTranslator::BuildMesh(
          sub_mesh.id, sub_mesh.triangles, sub_mesh.primvars,
          ResolveMeshAttributes(sub_mesh.material_id), bsdf.get(),
          env.emitter_ptr, env.sensor_ptr, interior_medium, sub_shape_spec);
      if (mesh) {
        res.meshes.emplace_back(sub_mesh.id, std::move(mesh));
      }
    }
  }

  void CommitInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res,
                               std::atomic<bool>* host_scalars_changed) {
    const auto& spec = *(work->spec);
    CommitNonInstancedMeshWork(work, res, host_scalars_changed, kProtoPrefix);
    if (spec.needs_rebuild) {
      if (HasEmitter(spec)) {
        TF_WARN(
            "Mesh %s is instanced but has an emitter attached. Mitsuba "
            "does not support emitters on instances. Ignoring emitter.",
            spec.id.GetText());
      }
      if (HasSensor(spec)) {
        TF_WARN(
            "Mesh %s is instanced but has a sensor attached. Mitsuba does "
            "not support sensors on instances. Ignoring sensor.",
            spec.id.GetText());
      }
      if (res.meshes.empty()) return;

      // Create ShapeGroup wrapping all prototype shapes.
      mitsuba::Properties group_props("shapegroup");
      for (size_t i = 0; i < res.meshes.size(); ++i) {
        group_props.set("shape_" + std::to_string(i),
                        res.meshes[i].second.get());
      }
      res.shapegroup =
          mitsuba::PluginManager::instance()->create_object<Shape>(group_props);
      // ShapeGroup's constructor leaves m_to_world literal until the first
      // parameters_changed() call makes it opaque.
      res.shapegroup->parameters_changed();

      // Create Instances.
      res.instances.reserve(spec.instance_transforms.size());
      for (const MotionTransform& xform : spec.instance_transforms) {
        mitsuba::Properties inst_props("instance");
        inst_props.set("shapegroup", res.shapegroup.get());
        PrimTranslator::SetTransformProperty(inst_props, xform);
        res.instances.push_back(
            mitsuba::PluginManager::instance()->create_object<Shape>(
                inst_props));
      }
    } else {
      if ((spec.dirty_bits & kMeshGeometryDirty) != 0) {
        auto group_it = shapes_.find(
            absl::StrCat(kProtoGroupPrefix, spec.id.GetAsString()));
        if (group_it != shapes_.end()) {
          group_it->second->parameters_changed();
        }
      }
      if (spec.dirty_bits & (HdChangeTracker::DirtyInstancer |
                             HdChangeTracker::DirtyInstanceIndex)) {
        std::string id_str = spec.id.GetAsString();
        for (size_t i = 0; i < spec.instance_transforms.size(); ++i) {
          auto inst_it =
              shapes_.find(absl::StrCat(kInstancePrefix, id_str, "_", i));
          if (inst_it != shapes_.end()) {
            SetTransform(inst_it->second.get(), spec.instance_transforms[i]);
          }
        }
      }
    }
  }

  bool MergeNonInstancedMesh(CommittedMesh& res,
                             const std::string& /*id_str*/) {
    for (auto& [key, mesh] : res.meshes) {
      shapes_[key] = mesh;
    }
    return true;  // Rebuild scene
  }

  bool MergeInstancedMesh(CommittedMesh& res, const std::string& id_str) {
    if (res.meshes.empty()) return true;
    for (auto& [key, mesh] : res.meshes) {
      shapes_[absl::StrCat(kProtoPrefix, key)] = mesh;
    }
    shapes_[absl::StrCat(kProtoGroupPrefix, id_str)] = res.shapegroup;
    for (size_t i = 0; i < res.instances.size(); ++i) {
      shapes_[absl::StrCat(kInstancePrefix, id_str, "_", i)] = res.instances[i];
    }
    return true;  // Rebuild scene
  }

  bool HasEmitter(const MeshSpec& spec) const {
    return spec.emitter_spec.has_value() ||
           !spec.mitsuba_emitter_spec.plugin_type.empty() ||
           (spec.shape_spec.children.count("emitter") > 0) ||
           std::any_of(spec.geometry.material_ids.begin(),
                       spec.geometry.material_ids.end(),
                       [&](const SdfPath& material_id) {
                         return material_emitters_.contains(material_id);
                       });
  }

  bool HasSensor(const MeshSpec& spec) const {
    return shape_sensors_.contains(spec.id) ||
           (spec.shape_spec.children.count("sensor") > 0);
  }

  // Whether `spec` is rendered through Mitsuba instances. A mesh wrapped in an
  // instance only for transform motion blur (`motion_wrapped`) is rendered as
  // a regular shape with its static transform at frame time if it carries an
  // emitter or a surface sensor, which Mitsuba does not support on instances
  // (the same policy as for area lights).
  bool IsInstanced(const MeshSpec& spec) const {
    return !spec.instance_transforms.empty() &&
           !(spec.motion_wrapped && (HasEmitter(spec) || HasSensor(spec)));
  }

  // Returns the spec to commit for `spec`: `motion_wrapped` meshes are resolved
  // into either an instanced prototype in object space (transform updates then
  // apply to the wrapping instance) or a plain static mesh.
  MeshSpec& ResolveMotionWrapped(MeshSpec& spec,
                                 std::optional<MeshSpec>& storage) const {
    if (!spec.motion_wrapped) return spec;
    MeshSpec& resolved = storage.emplace(spec);
    if (IsInstanced(spec)) {
      resolved.transform = GfMatrix4d(1.0);
      if (resolved.dirty_bits & HdChangeTracker::DirtyTransform) {
        resolved.dirty_bits =
            (resolved.dirty_bits & ~HdChangeTracker::DirtyTransform) |
            HdChangeTracker::DirtyInstancer;
      }
    } else {
      resolved.instance_transforms.clear();
    }
    return resolved;
  }

  bool CommitMeshes(std::atomic<bool>* host_scalars_changed) {
    // 1. Get pending specs.
    auto pending_specs = GetPendingSpecs(mesh_specs_);
    if (pending_specs.empty()) return false;

    // 2. Prepare work on the main thread (resolve motion-wrapped specs and
    // displacement textures lock-free).
    std::vector<MeshCommitWork> work_items(pending_specs.size());
    for (size_t i = 0; i < pending_specs.size(); ++i) {
      MeshSpec* spec = &ResolveMotionWrapped(*pending_specs[i],
                                             work_items[i].resolved_spec);
      work_items[i].spec = spec;
      work_items[i].displacements.reserve(spec->geometry.material_ids.size());
      for (const auto& material_id : spec->geometry.material_ids) {
        auto disp_it = displacements_.find(material_id);
        if (disp_it != displacements_.end()) {
          work_items[i].displacements.push_back(disp_it->second);
        } else {
          work_items[i].displacements.emplace_back();
        }
        for (const auto& [name, channels] :
             ResolveMeshAttributes(material_id)) {
          work_items[i].requested_attributes.insert(name);
        }
      }
    }

    // 3. Run parallel commit on work items.
    std::vector<CommittedMesh> results(work_items.size());
    JitParallelFor<Float>(work_items.size(), [&](size_t i) {
      if (work_items[i].spec->instance_transforms.empty()) {
        CommitNonInstancedMeshWork(&work_items[i], results[i],
                                   host_scalars_changed);
      } else {
        CommitInstancedMeshWork(&work_items[i], results[i],
                                host_scalars_changed);
      }
    });

    bool rebuild = false;
    // 4. Merge results on the main thread.
    for (size_t i = 0; i < pending_specs.size(); ++i) {
      const MeshSpec* spec = work_items[i].spec;

      if (spec->needs_rebuild) {
        std::string id_str = spec->id.GetAsString();
        // Clean up old shapes
        CleanUpInstancing(spec->id);
        shapes_.erase(id_str);
        EraseShapesWithPrefix(absl::StrCat(id_str, "#"));
        if (!spec->instance_transforms.empty()) {
          rebuild |= MergeInstancedMesh(results[i], id_str);
        } else {
          rebuild |= MergeNonInstancedMesh(results[i], id_str);
        }
      }
      pending_specs[i]->MarkClean();
    }
    return rebuild;
  }

  bool HasEmitter(const CurveSpec& spec) const {
    return spec.emitter_spec.has_value() ||
           !spec.mitsuba_emitter_spec.plugin_type.empty() ||
           spec.shape_spec.children.count("emitter") > 0 ||
           material_emitters_.contains(spec.material_id);
  }

  bool CommitCurves() {
    static const PrimvarMap kEmptyPrimvars;
    return ParallelCommit<decltype(curve_specs_), mitsuba::ref<Shape>>(
        curve_specs_,
        [&](CurveSpec* spec, mitsuba::ref<Shape>& res) {
          Object* interior_medium = ResolveInteriorMedium(spec->material_id);
          const bool has_medium =
              spec->shape_spec.children.count("interior") > 0 ||
              spec->shape_spec.children.count("exterior") > 0 ||
              interior_medium != nullptr;
          ref<BSDF> bsdf =
              ResolveBsdf(spec->material_id, kEmptyPrimvars, has_medium, HasEmitter(*spec));

          if (spec->needs_rebuild) {
            auto env = ResolveEmitterAndSensor(*spec, spec->material_id);
            res = PrimTranslator::BuildCurves(*spec, bsdf.get(),
                                              env.emitter_ptr, env.sensor_ptr,
                                              interior_medium);
          } else {
            auto it = shapes_.find(spec->id.GetAsString());
            if (!TF_VERIFY(it != shapes_.end(), "Curve not found: %s",
                           spec->id.GetText())) {
              return;
            }
            if (spec->shape_spec.children.count("bsdf") == 0) {
              it->second->set_bsdf(bsdf.get());
            }
          }
        },
        [&](CurveSpec* spec, mitsuba::ref<Shape>& res) {
          if (res) {
            shapes_[spec->id.GetAsString()] = res;
          }
          return true;
        });
  }

  static std::string ResolveLeafIntegratorType(const PluginSpecNode& spec) {
    if (spec.plugin_type == "aov" || spec.plugin_type == "moment") {
      for (const auto& [_, child] : spec.children) {
        if (!child.plugin_type.empty() && child.plugin_type != "rgb" &&
            child.plugin_type != "spectrum" && child.plugin_type != "vector") {
          return ResolveLeafIntegratorType(child);
        }
      }
    }
    return spec.plugin_type;
  }

  bool CommitParticleFields() {
    const std::string leaf_integrator =
        ResolveLeafIntegratorType(integrator_spec_);
    if (!particle_field_specs_.empty() &&
        leaf_integrator != "volprim_rf_basic") {
      TF_WARN(
          "Scene contains Gaussian splats (%zu), which require the "
          "'volprim_rf_basic' integrator for radiance field evaluation. "
          "Current integrator: '%s'.",
          particle_field_specs_.size(), leaf_integrator.c_str());
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

  bool CommitLights(std::atomic<bool>* host_scalars_changed) {
    return ParallelCommit<decltype(light_specs_),
                          typename PrimTranslator::TranslatedLight>(
        light_specs_,
        [&](LightSpec* spec, typename PrimTranslator::TranslatedLight& res) {
          auto build_light = [&]() {
            try {
              res = PrimTranslator::BuildLight(*spec);
            } catch (const std::exception& e) {
              TF_RUNTIME_ERROR("Failed to create light '%s': %s",
                               spec->id.GetText(), e.what());
            }
          };
          if (spec->needs_rebuild) {
            build_light();
          } else if (spec->dirty_bits != 0) {
            Object* light_obj = nullptr;
            if (spec->IsAreaLight()) {
              std::string id_str = spec->id.GetAsString();
              auto it = shapes_.find(id_str);
              if (TF_VERIFY(it != shapes_.end(), "Light shape not found: %s",
                            id_str.c_str())) {
                light_obj = it->second.get();
              }
            } else {
              auto it = emitters_.find(spec->id);
              if (TF_VERIFY(it != emitters_.end(),
                            "Light emitter not found: %s",
                            spec->id.GetText())) {
                light_obj = it->second.get();
              }
            }
            if (light_obj != nullptr) {
              bool wrote_host_scalar = false;
              if (!PrimTranslator::TryUpdateLightInPlace(
                      light_obj, *spec, &wrote_host_scalar)) {
                spec->needs_rebuild = true;
                build_light();
              } else if (wrote_host_scalar) {
                host_scalars_changed->store(true, std::memory_order_relaxed);
              }
            }
          }
        },
        [&](LightSpec* spec, typename PrimTranslator::TranslatedLight& trans) {
          std::string id_str = spec->id.GetAsString();
          if (trans.shape) {
            shapes_[id_str] = trans.shape;
            emitters_.erase(spec->id);
          } else if (trans.emitter) {
            emitters_[spec->id] = trans.emitter;
            shapes_.erase(id_str);
          } else {
            return shapes_.erase(id_str) > 0 || emitters_.erase(spec->id) > 0;
          }
          return true;
        });
  }

  void CommitResources() override {
    absl::MutexLock lock(state_mutex_);
    if (!scene_dirty_ && !has_pending_commits_) {
      return;
    }
    NewJitScope<Float>();

    bool rebuild_scene = scene_dirty_;
    std::atomic<bool> host_scalars_changed{false};
    CommitMaterials();
    rebuild_scene |= CommitCameras(&host_scalars_changed);
    rebuild_scene |= CommitMeshes(&host_scalars_changed);
    rebuild_scene |= CommitCurves();
    rebuild_scene |= CommitParticleFields();
    rebuild_scene |= CommitLights(&host_scalars_changed);

    const bool materials_changed = !material_dirty_flags_.empty();
    if (!rebuild_scene && materials_changed &&
        NeedsRebuildForFilteredTextures()) {
      rebuild_scene = true;
    }

    if (rebuild_scene || materials_changed ||
        host_scalars_changed.load(std::memory_order_relaxed)) {
      // In-place material updates swap BSDF/Texture objects whose JIT class
      // pointers may be baked into a frozen kernel, and C++ host scalar plugin
      // parameters are baked as trace-time constants, so both require clearing
      // the frozen kernel even when the Scene itself is not rebuilt.
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
      if (hide_emitters_) {
        EnsureIntegrator();
        if (integrator_) {
          props.set("integrator", integrator_.get());
        }
      }
      for (auto& [id, shape] : shapes_) {
        bool is_proto_mesh = absl::StartsWith(id, kProtoPrefix) &&
                             !absl::StartsWith(id, kProtoGroupPrefix);
        if (is_proto_mesh) {
          continue;
        }
        if (shape->is_shape_group()) {
          // The ray tracing backends only (re)build the acceleration structure
          // of a shape group while it is dirty. A group carried over from the
          // previous scene has its flag cleared, so re-mark it for the new one.
          shape->mark_dirty();
        }
        props.set(id, shape.get());
      }
      for (auto& [id, sensor] : sensors_) {
        props.set(id.GetAsString(), sensor.get());
      }
      // Mitsuba supports at most one environment emitter per scene. Of multiple
      // DomeLights (e.g., an illumination dome and a camera background dome
      // whose `lightLink` excludes the scene geometry), select the first one in
      // SdfPath order whose `lightLink` is unrestricted, or else the first one.
      const LightSpec* dome = nullptr;
      for (auto& [id, emitter] : emitters_) {
        const LightSpec& spec = light_specs_.at(id);
        if (spec.prim_type != HdPrimTypeTokens->domeLight) {
          props.set(id.GetAsString(), emitter.get());
        } else if (!dome || (spec.light_link_everything &&
                             !dome->light_link_everything)) {
          dome = &spec;
        }
      }
      if (dome) {
        props.set(dome->id.GetAsString(), emitters_.at(dome->id).get());
      }
      scene_ = new Scene(props);
      scene_dirty_ = false;
    } else if (scene_) {
      const auto previous_bbox = scene_->bbox();
      scene_->parameters_changed();
      if (scene_->bbox() != previous_bbox) {
        bool host_bsphere_updated = false;
        for (auto& sensor : scene_->sensors()) {
          sensor->set_scene(scene_.get());
          if (absl::StartsWith(sensor->class_name(), "DistantSensor")) {
            host_bsphere_updated = true;
          }
        }
        for (auto& emitter : scene_->emitters()) {
          if (!emitter->is_environment()) {
            emitter->set_scene(scene_.get());
            if (emitter->class_name() == "DirectionalEmitter") {
              host_bsphere_updated = true;
            }
          }
        }
        if (host_bsphere_updated) {
          if constexpr (dr::is_jit_v<Float>) {
            if (frozen_render_) {
              frozen_render_->Clear();
            }
          }
        }
      }
    }

    // Run Garbage Collection after all resources are committed, ensuring
    // that all active shapes, mesh emitters, and sensors are instantiated.
    texture_cache_.GarbageCollect();

    material_dirty_flags_.clear();
    sensor_binding_dirty_.clear();
    has_pending_commits_ = false;
  }

  mitsuba::Object* GetScene() override { return scene_.get(); }

 private:
  struct RenderPassState {
    ref<Integrator> aov_integrator = nullptr;
    std::string aov_integrator_keys;
    std::vector<AovRequest> aov_requests;
    HdMitsubaRenderBuffer* color_buffer = nullptr;
  };

  bool scene_dirty_ = true;
  ref<Scene> scene_;
  ref<Integrator> integrator_ = nullptr;
  PluginSpecNode integrator_spec_{"path"};
  bool hide_emitters_ = false;
  PluginSpecNode sampler_spec_{"independent"};
  absl::flat_hash_map<const HdRenderPass*, RenderPassState> pass_aov_states_;
  size_t sample_count_ = kDefaultSampleCount;
  size_t active_sample_count_ = kDefaultSampleCount;
  uint32_t seed_ = 0;
  int interactive_samples_per_pass_ = 1;

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

  // Sensors, shapes and emitters are ordered, so that the scene is assembled
  // deterministically. With hash maps, whose iteration order varies between
  // processes, e.g. the emitter order and thus the light sampling noise would
  // differ between runs.
  absl::btree_map<SdfPath, ref<Sensor>> sensors_;
  absl::flat_hash_map<SdfPath, SdfPath, SdfPath::Hash>
      shape_sensors_;  // Shape -> camera
  absl::btree_map<std::string, ref<Shape>> shapes_;
  absl::btree_map<SdfPath, ref<Emitter>> emitters_;
  absl::flat_hash_map<SdfPath, ref<BSDF>, SdfPath::Hash> bsdfs_;
  absl::flat_hash_map<SdfPath, MaterialDisplacement, SdfPath::Hash>
      displacements_;
  absl::flat_hash_map<SdfPath, mitsuba::Properties, SdfPath::Hash>
      material_emitters_;
  absl::flat_hash_map<SdfPath, MeshAttributeRequests, SdfPath::Hash>
      mesh_attributes_;
  absl::flat_hash_map<SdfPath, ref<Object>, SdfPath::Hash> interior_media_;
  ref<BSDF> default_bsdf_ = nullptr;
  ref<BSDF> emitter_default_bsdf_ = nullptr;
  ref<BSDF> null_bsdf_ = nullptr;
  absl::flat_hash_map<std::tuple<float, float, float>, ref<BSDF>> color_bsdfs_;
  absl::Mutex color_bsdfs_mutex_;

  TextureCache<Float, Spectrum> texture_cache_;
  std::unique_ptr<FrozenRender<Float, Spectrum>> frozen_render_ = nullptr;
  bool denoise_ = false;
  bool denoise_dirty_ = false;
  Denoiser<Float, Spectrum> denoiser_;

  void EnsureIntegrator() {
    if (integrator_) {
      return;
    }
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Creating integrator, type: %s\n",
             integrator_spec_.plugin_type.c_str());
    try {
      integrator_ = BuildPluginObject<Integrator>(integrator_spec_);
    } catch (const std::exception& e) {
      TF_WARN(
          "Failed to create integrator '%s': %s. Falling back to the default "
          "path tracer.",
          integrator_spec_.plugin_type.c_str(), e.what());
      integrator_ = PluginManager::instance()->create_object<Integrator>(
          Properties("path"));
    }
  }

  void MarkEmittersForRebuild() {
    for (auto& [_, light_spec] : light_specs_) {
      light_spec.needs_rebuild = true;
    }
    for (auto& [_, mesh_spec] : mesh_specs_) {
      if (HasEmitter(mesh_spec)) {
        mesh_spec.needs_rebuild = true;
      }
    }
    for (auto& [_, curve_spec] : curve_specs_) {
      if (HasEmitter(curve_spec)) {
        curve_spec.needs_rebuild = true;
      }
    }
    has_pending_commits_ = true;
  }

  void MarkSensorsForRebuild() {
    for (auto& [_, camera_spec] : camera_specs_) {
      camera_spec.needs_rebuild = true;
      if (auto target = SurfaceSensorTarget(camera_spec)) {
        sensor_binding_dirty_.insert(*target);
        shape_sensors_dirty_ = true;
      }
    }
    has_pending_commits_ = true;
  }

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
      // Shape groups and instances wrap underlying prototype meshes that are
      // already present in shapes_. Skipping them avoids duplicate traversals,
      // and calling Shape::traverse directly avoids Mesh::ensure_views
      // instantiating unused JIT mesh view buffers.
      if (shape->is_shape_group() || shape->is_instance()) {
        continue;
      }
      const_cast<Shape*>(shape.get())->Shape::traverse(&scan);
      if (scan.found) return true;
    }
    return false;
  }

  bool EraseShapesWithPrefix(std::string_view prefix) {
    bool erased = false;
    for (auto it = shapes_.begin(); it != shapes_.end();) {
      if (absl::StartsWith(it->first, prefix)) {
        it = shapes_.erase(it);
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
    if (EraseShapesWithPrefix(absl::StrCat(kProtoPrefix, id_str, "#"))) {
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

SceneManager* SceneManager::CreateSceneManager(const std::string& variant,
                                               bool sync_python_variant) {
  if (sync_python_variant) {
    SyncMitsubaPythonVariant(variant);
  }
  return MI_INVOKE_VARIANT(variant, CreateSceneManagerImpl);
}

void SceneManager::SyncPythonVariant(const std::string& variant) {
  SyncMitsubaPythonVariant(variant);
}

SceneManager::SceneManager() {
  RegisterIesEmitter();
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
