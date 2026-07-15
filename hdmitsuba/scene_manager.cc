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
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <nanothread/nanothread.h>
#include <pxr/base/gf/vec2f.h>
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

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/mesh.h"
#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/prim_translator.h"
#include "hdmitsuba/render_buffer.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/traversal.h"
#include "hdmitsuba/utils.h"
#include "hdmitsuba/kernel_freezing.h"

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

void DiscoverTextures(
    const HdMaterialNetwork2& network,
    const std::function<void(const mitsuba::Properties& tex_props)>& callback) {
  auto process_connection = [&](const TfToken& input_name,
                                const SdfPath& upstream_node_path) {
    auto upstream_it = network.nodes.find(upstream_node_path);
    if (upstream_it != network.nodes.end()) {
      const auto& node = upstream_it->second;
      if (node.nodeTypeId == TfToken("UsdUVTexture") ||
          node.nodeTypeId == TfToken("mitsuba_bitmap")) {
        if (auto props_opt = ExtractTextureProperties(
                node.parameters, node.nodeTypeId, input_name)) {
          callback(*props_opt);
        }
      }
    }
  };
  // 1. Traverse all connections between nodes
  for (const auto& [node_path, node] : network.nodes) {
    for (const auto& [input_name, connections] : node.inputConnections) {
      if (!connections.empty()) {
        process_connection(input_name, connections[0].upstreamNode);
      }
    }
  }
  // 2. Traverse all terminal connections (e.g., surface, displacement)
  for (const auto& [terminal_name, connection] : network.terminals) {
    if (!connection.upstreamNode.IsEmpty()) {
      process_connection(terminal_name, connection.upstreamNode);
    }
  }
}

constexpr std::string_view kProtoPrefix = "proto_";
constexpr std::string_view kProtoGroupPrefix = "proto_group_";
constexpr std::string_view kInstancePrefix = "instance_";

// Using Dr.Jit in a multithreaded environment requires explicitly creating
// JIT scopes on each thread. This RAII struct should be used in code blocks
// that may be executed on different threads, with dependencies crossing
// thread boundaries (e.g., accessing a texture that was initialized on a
// different thread).
template <typename Float>
struct JitScopeGuard {
  uint32_t backend = 0;
  uint32_t prev_scope = 0;

  JitScopeGuard() {
    if constexpr (dr::is_cuda_v<Float>) {
      backend = (uint32_t)JitBackend::CUDA;
    } else if constexpr (dr::is_llvm_v<Float>) {
      backend = (uint32_t)JitBackend::LLVM;
    } else if constexpr (dr::is_metal_v<Float>) {
      backend = (uint32_t)JitBackend::Metal;
    }
    if (backend) {
      prev_scope = jit_scope((JitBackend)backend);
      jit_new_scope((JitBackend)backend);
    }
  }

  ~JitScopeGuard() {
    if (backend) {
      jit_set_scope((JitBackend)backend, prev_scope);
    }
  }
};

// Copies data from a Mitsuba Tensor to a Hydra Render Buffer (Scalar/CPU path).
template <typename TensorT>
void ScalarCopyToRenderBuffer(HdMitsubaRenderBuffer* render_buffer,
                              const TensorT& tensor, int src_offset, int channels,
                              bool is_int,
                              const std::optional<GfRect2i>& crop_window = std::nullopt) {
  if (!TF_VERIFY(render_buffer, "Render buffer is null.")) {
    return;
  }
  int dst_channels = HdGetComponentCount(render_buffer->GetFormat());
  if (!TF_VERIFY(
          dst_channels == channels || (dst_channels == 4 && channels == 3),
          "Destination and source channel counts do not match (%d vs %d)",
          dst_channels, channels)) {
    return;
  }

  size_t dst_width = render_buffer->GetWidth();
  size_t dst_height = render_buffer->GetHeight();
  size_t src_channels = tensor.shape()[2];
  void* dst_ptr = render_buffer->Map();

  size_t crop_x = 0;
  size_t crop_y = 0;
  size_t crop_w = dst_width;
  size_t crop_h = dst_height;
  if (crop_window.has_value()) {
    crop_x = crop_window->GetMinX();
    crop_y = crop_window->GetMinY();
    crop_w = crop_window->GetWidth();
    crop_h = crop_window->GetHeight();
  }

  if (crop_w != tensor.shape()[1] || crop_h != tensor.shape()[0]) {
    TF_FATAL_ERROR(
        "Tensor dimensions do not match crop window: %lu x %lu vs %lu x %lu",
        crop_w, crop_h, tensor.shape()[1], tensor.shape()[0]);
  }

  const float* src_data = tensor.array().data();
  if (is_int) {
    int32_t* dst_int = static_cast<int32_t*>(dst_ptr);
    for (size_t src_y = 0; src_y < crop_h; ++src_y) {
      size_t dst_y = (dst_height - 1) - (crop_y + src_y);
      for (size_t src_x = 0; src_x < crop_w; ++src_x) {
        size_t src_idx = (src_y * crop_w + src_x) * src_channels + src_offset;
        size_t dst_idx = (dst_y * dst_width + (crop_x + src_x)) * dst_channels;
        for (int c = 0; c < channels && c < dst_channels; ++c) {
          dst_int[dst_idx + c] = static_cast<int32_t>(src_data[src_idx + c]);
        }
      }
    }
  } else {
    float* dst_float = static_cast<float*>(dst_ptr);
    for (size_t src_y = 0; src_y < crop_h; ++src_y) {
      size_t dst_y = (dst_height - 1) - (crop_y + src_y);
      for (size_t src_x = 0; src_x < crop_w; ++src_x) {
        size_t src_idx = (src_y * crop_w + src_x) * src_channels + src_offset;
        size_t dst_idx = (dst_y * dst_width + (crop_x + src_x)) * dst_channels;
        for (int c = 0; c < channels && c < dst_channels; ++c) {
          dst_float[dst_idx + c] = src_data[src_idx + c];
        }
        // Fill alpha if needed.
        if (dst_channels == 4 && channels == 3) {
          dst_float[dst_idx + 3] = 1.0f;
        }
      }
    }
  }
  render_buffer->SetConverged(true);
  render_buffer->Unmap();
}

struct CopyDestination {
  HdMitsubaRenderBuffer* buffer;
  int src_offset;
  int channels;
  bool is_int;
};

// Helper function to batch copy all output buffers / AOVs to hydra's render
// buffers.
template <typename Float, typename TensorT>
void PerformBatchedCopy(const TensorT& tensor,
                        const std::vector<CopyDestination>& destinations,
                        const std::optional<GfRect2i>& crop_window = std::nullopt) {
  if constexpr (dr::is_jit_v<Float>) {
    using Int32 = dr::int32_array_t<Float>;
    using MigratedFloat = std::decay_t<decltype(dr::migrate(
        std::declval<Float>(), JitBackend::None))>;
    using MigratedInt = std::decay_t<decltype(dr::migrate(std::declval<Int32>(),
                                                          JitBackend::None))>;

    std::vector<std::variant<Float, Int32>> gathered_vars;
    gathered_vars.reserve(destinations.size());
    for (const auto& dest : destinations) {
      int dst_channels = HdGetComponentCount(dest.buffer->GetFormat());
      using UInt32 = dr::uint32_array_t<Float>;
      size_t dst_width = dest.buffer->GetWidth();
      size_t dst_height = dest.buffer->GetHeight();
      size_t dst_pixel_count = dst_width * dst_height;

      size_t crop_x = 0;
      size_t crop_y = 0;
      size_t crop_w = dst_width;
      size_t crop_h = dst_height;
      if (crop_window.has_value()) {
        crop_x = crop_window->GetMinX();
        crop_y = crop_window->GetMinY();
        crop_w = crop_window->GetWidth();
        crop_h = crop_window->GetHeight();
      }

      UInt32 dst_pixel_idx = dr::arange<UInt32>(dst_pixel_count);
      UInt32 dst_x = dst_pixel_idx % dst_width;
      UInt32 dst_row = dst_pixel_idx / dst_width;
      UInt32 r_topdown = dst_height - 1 - dst_row;

      auto in_crop = (dst_x >= crop_x) && (dst_x < crop_x + crop_w) &&
                     (r_topdown >= crop_y) && (r_topdown < crop_y + crop_h);

      UInt32 src_x = dst_x - crop_x;
      UInt32 src_y_down = r_topdown - crop_y;
      UInt32 src_pixel_idx =
          dr::select(in_crop, src_y_down * crop_w + src_x, 0);

      UInt32 repeated_pixel_idx =
          dr::repeat(src_pixel_idx, dst_channels);
      UInt32 channel_offsets =
          dr::tile(dr::arange<UInt32>(dst_channels), dst_pixel_count);
      UInt32 final_idx = repeated_pixel_idx * tensor.shape()[2] +
                         dest.src_offset + channel_offsets;

      auto in_crop_channel = dr::repeat(in_crop, dst_channels);

      if (dest.is_int) {
        Int32 gathered_int = dr::select(
            in_crop_channel,
            Int32(dr::gather<Float>(tensor.array(), final_idx)),
            0);
        dr::schedule(gathered_int);
        gathered_vars.push_back(gathered_int);
      } else {
        Float gathered_float;
        if (dst_channels == 4 && dest.channels == 3) {
          gathered_float = dr::select(
              in_crop_channel,
              dr::select(channel_offsets < 3,
                         dr::gather<Float>(tensor.array(), final_idx), 1.0f),
              0.0f);
        } else {
          gathered_float = dr::select(
              in_crop_channel,
              dr::gather<Float>(tensor.array(), final_idx), 0.0f);
        }
        dr::schedule(gathered_float);
        gathered_vars.push_back(gathered_float);
      }
    }
    dr::eval();

    // Migrate result back to host memory.
    std::vector<std::variant<MigratedFloat, MigratedInt>> migrated_vars;
    migrated_vars.reserve(destinations.size());
    for (size_t i = 0; i < destinations.size(); ++i) {
      std::visit(
          [&](auto& var) {
            migrated_vars.push_back(dr::migrate(var, JitBackend::None));
          },
          gathered_vars[i]);
    }
    dr::sync_thread();

    for (size_t i = 0; i < destinations.size(); ++i) {
      const auto& dest = destinations[i];
      void* dst_ptr = dest.buffer->Map();
      int dst_channels = HdGetComponentCount(dest.buffer->GetFormat());
      size_t dst_width = dest.buffer->GetWidth();
      size_t dst_height = dest.buffer->GetHeight();
      size_t element_size = dest.is_int ? sizeof(int32_t) : sizeof(float);
      size_t size_bytes = dst_width * dst_height * dst_channels * element_size;

      std::visit(
          [&](auto& host_arr) {
            memcpy(dst_ptr, host_arr.data(), size_bytes);
          },
          migrated_vars[i]);
      dest.buffer->SetConverged(true);
      dest.buffer->Unmap();
    }
  } else {
    for (const auto& dest : destinations) {
      ScalarCopyToRenderBuffer(dest.buffer, tensor, dest.src_offset,
                               dest.channels, dest.is_int, crop_window);
    }
  }
}

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

template <typename Float, typename Spectrum>
void ApplyDisplacement(
    const SdfPath& mesh_id,
    const mitsuba::Texture<Float, Spectrum>* displacement_texture,
    const VtIntArray& vertex_indices, const VtIntArray& face_counts,
    const std::vector<int>& global_face_indices,
    const std::vector<int>& global_corner_indices, PrimvarMap& primvars) {
  TRACE_FUNCTION();
  if (!displacement_texture) return;

  using Vector2f = mitsuba::Vector<Float, 2>;
  using Vector3f = mitsuba::Vector<Float, 3>;
  using UInt32 = dr::uint32_array_t<Float>;

  // 1) Retrieve vector of UV coordinates Vec2f, normals and target vertex
  // index.
  VtVec2fArray uv_coords;
  VtVec3fArray normals;
  VtIntArray target_vertex_indices;
  absl::flat_hash_set<int> target_vertex_indices_set;
  auto uv_it = primvars.find(TfToken("st"));
  auto normal_it = primvars.find(HdTokens->normals);
  auto points_it = primvars.find(HdTokens->points);
  if (uv_it == primvars.end() || normal_it == primvars.end() ||
      points_it == primvars.end()) {
    TF_RUNTIME_ERROR("Missing required primvars for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  if (!uv_it->second.value.IsHolding<VtVec2fArray>() ||
      !normal_it->second.value.IsHolding<VtVec3fArray>() ||
      !points_it->second.value.IsHolding<VtVec3fArray>()) {
    TF_RUNTIME_ERROR("Invalid primvar types for displacement on %s",
                     mesh_id.GetText());
    return;
  }
  const auto& uv_primvar = uv_it->second;
  const auto& normal_primvar = normal_it->second;
  VtVec3fArray points = points_it->second.value.Get<VtVec3fArray>();

  auto uv_interpolator =
      GeometryProcessor::GetInterpolator(uv_primvar.value.Get<VtVec2fArray>(),
                                         uv_primvar.descriptor.interpolation);
  auto normal_interpolator = GeometryProcessor::GetInterpolator(
      normal_primvar.value.Get<VtVec3fArray>(),
      normal_primvar.descriptor.interpolation);

  size_t corner = 0;
  const float bias = 0.5f;
  for (size_t face = 0; face < face_counts.size(); ++face) {
    for (int v = 0; v < face_counts[face]; ++v) {
      int vertex_index = vertex_indices[corner];
      // Each vertex should only be displaced once, even if it is part of
      // multiple faces.
      if (target_vertex_indices_set.contains(vertex_index)) {
        corner++;
        continue;
      }
      uv_coords.push_back(uv_interpolator(global_face_indices[face], corner,
                                          global_corner_indices[corner],
                                          vertex_indices));
      normals.push_back(normal_interpolator(global_face_indices[face], corner,
                                            global_corner_indices[corner],
                                            vertex_indices));
      target_vertex_indices.push_back(vertex_index);
      target_vertex_indices_set.insert(vertex_index);
      corner++;
    }
  }

  // 2) Query displacement and scatter add on vertices at target index.
  using FloatStorage = typename mitsuba::Mesh<Float, Spectrum>::FloatStorage;
  if constexpr (!dr::is_dynamic_v<Float>) {
    for (size_t i = 0; i < target_vertex_indices.size(); ++i) {
      int vertex_index = target_vertex_indices[i];
      GfVec2f uv = uv_coords[i];
      GfVec3f normal = normals[i];
      mitsuba::SurfaceInteraction<Float, Spectrum> si;
      si.uv = {uv[0], 1.0f - uv[1]};
      GfVec3f displacement = (displacement_texture->eval_1(si) - bias) * normal;
      points[vertex_index] += displacement;
    }
  } else {
    size_t n_vertices = target_vertex_indices.size();
    FloatStorage uv = dr::load<FloatStorage>(uv_coords.data(), n_vertices * 2);
    FloatStorage normal =
        dr::load<FloatStorage>(normals.data(), n_vertices * 3);
    UInt32 indices = dr::arange<UInt32>(n_vertices);
    Vector2f uv_vec = dr::gather<Vector2f>(uv, indices);
    mitsuba::SurfaceInteraction<Float, Spectrum> si;
    si.uv = {uv_vec[0], 1.0f - uv_vec[1]};
    Vector3f displacement = (displacement_texture->eval_1(si) - bias) *
                            dr::gather<Vector3f>(normal, indices);
    Float displacement_flat = dr::ravel(displacement);
    dr::eval(displacement_flat);
    auto&& host_displacement = dr::migrate(displacement_flat, JitBackend::None);
    dr::sync_thread();
    for (size_t i = 0; i < n_vertices; ++i) {
      GfVec3f offset(host_displacement[3 * i + 0], host_displacement[3 * i + 1],
                     host_displacement[3 * i + 2]);
      points[target_vertex_indices[i]] += offset;
    }
  }
  primvars[HdTokens->points].value = VtValue(std::move(points));
}

template <typename Float, typename Spectrum>
std::vector<SubMeshOutput> RunGeometryPipeline(
    const MeshSpec& spec,
    const std::vector<const mitsuba::Texture<Float, Spectrum>*>&
        displacement_textures) {
  TRACE_FUNCTION();
  PrimvarMap final_primvars = spec.primvars;
  if (final_primvars.find(HdTokens->normals) == final_primvars.end()) {
    bool has_displacement = false;
    for (const auto* tex : displacement_textures) {
      if (tex != nullptr) {
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
    if (displacement_textures[i]) {
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
        ApplyDisplacement(spec.id, displacement_textures[i],
                          material_vertex_indices, material_face_counts,
                          global_face_indices, global_corner_indices,
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
    std::vector<const Texture*> displacement_textures;
  };

  SceneModel() {
    jit_init(1u << static_cast<uint32_t>(JitBackend::LLVM));
#if defined(MI_ENABLE_CUDA)
    jit_init(1u << static_cast<uint32_t>(JitBackend::CUDA));
#endif
#if defined(MI_ENABLE_METAL)
    jit_init(1u << static_cast<uint32_t>(JitBackend::Metal));
#endif

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

  void SyncCamera(CameraSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncCamera: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    auto prev_it = camera_specs_.find(spec.id);
    if (!TF_VERIFY(prev_it != camera_specs_.end() || spec.needs_rebuild,
                   "New camera spec %s must have needs_rebuild set.",
                   spec.id.GetText())) {
      spec.needs_rebuild = true;
    }
    if (prev_it != camera_specs_.end() && !spec.needs_rebuild) {
      spec.dirty_bits |= prev_it->second.dirty_bits;
    }
    camera_specs_[spec.id] = std::move(spec);
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
    auto prev_it = mesh_specs_.find(spec.id);
    if (!TF_VERIFY(prev_it != mesh_specs_.end() || spec.needs_rebuild,
                   "New mesh spec %s must have needs_rebuild set.",
                   spec.id.GetText())) {
      spec.needs_rebuild = true;
    }
    if (prev_it != mesh_specs_.end() && !spec.needs_rebuild) {
      spec.dirty_bits = 1;
    }
    mesh_specs_[spec.id] = std::move(spec);
    reset_progressive_ = true;
  }

  void SyncCurves(CurveSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncCurves: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    curve_specs_[spec.id] = std::move(spec);
    reset_progressive_ = true;
  }

  void SyncLight(LightSpec spec) override {
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncLight: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    auto prev_it = light_specs_.find(spec.id);
    if (!TF_VERIFY(prev_it != light_specs_.end() || spec.needs_rebuild,
                   "New light spec %s must have needs_rebuild set.",
                   spec.id.GetText())) {
      spec.needs_rebuild = true;
    }
    if (prev_it != light_specs_.end() && !spec.needs_rebuild) {
      spec.dirty_bits |= prev_it->second.dirty_bits;
    }
    light_specs_[spec.id] = std::move(spec);
    reset_progressive_ = true;
  }

  void SyncMaterial(MaterialSpec spec) override {
    TRACE_FUNCTION();
    TF_DEBUG(HDMITSUBA_SYNC).Msg("SyncMaterial: %s\n", spec.id.GetText());
    absl::MutexLock lock(state_mutex_);
    material_specs_[spec.id] = std::move(spec);
    reset_progressive_ = true;
  }

  void RemoveShape(const SdfPath& id) override {
    TF_DEBUG(HDMITSUBA_LIFECYCLE).Msg("RemoveShape: %s\n", id.GetText());
    std::string id_str = id.GetAsString();
    absl::MutexLock lock(state_mutex_);
    mesh_specs_.erase(id);
    curve_specs_.erase(id);
    bool erased = false;
    if (shapes_.erase(id_str) > 0) {
      erased = true;
    }

    // Erase split sub-meshes
    for (auto it = shapes_.begin(); it != shapes_.end();) {
      if (absl::StartsWith(it->first, id_str + "/")) {
        shapes_.erase(it++);
        erased = true;
      } else {
        ++it;
      }
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
    absl::MutexLock lock(state_mutex_);
    material_specs_.erase(id);
    bsdfs_.erase(id.GetAsString());
    material_emitters_.erase(id.GetAsString());
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

    static const auto* aov_map =
        new absl::flat_hash_map<TfToken, std::pair<std::string, int>,
                                TfToken::HashFunctor>{
            {HdAovTokens->depth, {"depth:depth", 1}},
            {HdAovTokens->normal, {"sh_normal:sh_normal", 3}},
            {HdAovTokens->primId, {"primId:shape_index", 1}},
            {TfToken("shape_index"), {"shape_index:shape_index", 1}},
            {HdAovTokens->elementId, {"elementId:prim_index", 1}},
            {TfToken("prim_index"), {"prim_index:prim_index", 1}},
            {HdAovTokens->instanceId, {"instanceId:shape_index", 1}},
            {TfToken("albedo"), {"albedo:albedo", 3}},
            {TfToken("position"), {"position:position", 3}},
            {TfToken("uv"), {"uv:uv", 2}},
            {TfToken("geo_normal"), {"geo_normal:geo_normal", 3}},
            {TfToken("dp_du"), {"dp_du:dp_du", 3}},
            {TfToken("dp_dv"), {"dp_dv:dp_dv", 3}},
            {TfToken("duv_dx"), {"duv_dx:duv_dx", 2}},
            {TfToken("duv_dy"), {"duv_dy:duv_dy", 2}},
        };

    RenderPassState pass_state;
    pass_state.aov_requests.reserve(aov_bindings.size());
    std::vector<std::string> aov_strings;
    aov_strings.reserve(aov_bindings.size());
    for (const auto& binding : aov_bindings) {
      auto* buf = static_cast<HdMitsubaRenderBuffer*>(binding.renderBuffer);
      if (binding.aovName == HdAovTokens->color ||
          binding.aovName == TfToken("raw")) {
        pass_state.color_buffer = buf;
      } else {
        auto it = aov_map->find(binding.aovName);
        if (!TF_VERIFY(it != aov_map->end(), "Unsupported AOV: %s",
                       binding.aovName.GetText())) {
          return;
        }
        const auto& [mitsuba_name, channels] = it->second;
        pass_state.aov_requests.push_back({mitsuba_name, buf, channels});
        aov_strings.push_back(mitsuba_name);
      }
    }
    pass_state.aov_integrator_keys = absl::StrJoin(aov_strings, ",");
    pass_state.aov_integrator = nullptr;
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

    if (reset_progressive_) {
      accum_buffer_ = TensorXf();
      current_progressive_sample_ = 0;
      reset_progressive_ = false;
    }
    int samples_to_render = sample_count_;
    if (progressive_rendering_) {
      samples_to_render =
          (current_progressive_sample_ < static_cast<int>(sample_count_)) ? 1
                                                                          : 0;
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
        if (current_progressive_sample_ == samples_to_render) {
          accum_buffer_ = result;
        } else {
          accum_buffer_.array() += result.array();
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
    size_t base_channels = sensor->film()->base_channels_count();
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

  bool IsConverged(const HdRenderPass* /*render_pass*/) const override {
    if (!progressive_rendering_) {
      return true;
    }
    return current_progressive_sample_ >= static_cast<int>(sample_count_);
  }

  void UpdateNamespacedSettings(
      const VtDictionary& namespaced_settings) override {
    absl::MutexLock state_lock(state_mutex_);
    absl::MutexLock aov_lock(aov_states_mutex_);
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
        sample_count_ = it->second.Get<int>();
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

  uint32_t UpdateMaterialStructure(
      absl::string_view id_str,
      typename PrimTranslator::TranslatedMaterial& trans) {
    uint32_t dirty_flags = 0;
    if (trans.displacement_texture) {
      auto disp_it = displacement_textures_.find(id_str);
      if (disp_it == displacement_textures_.end() ||
          disp_it->second != trans.displacement_texture.get()) {
        displacement_textures_[id_str] =
            static_cast<Texture*>(trans.displacement_texture.get());
      }
    } else {
      displacement_textures_.erase(id_str);
    }
    if (trans.shape_emitter_props.has_value()) {
      auto em_it = material_emitters_.find(id_str);
      if (em_it == material_emitters_.end() ||
          em_it->second != trans.shape_emitter_props.value()) {
        material_emitters_[id_str] = trans.shape_emitter_props.value();
        dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
      }
    } else {
      if (material_emitters_.erase(id_str) > 0) {
        dirty_flags |= DirtyFlags::kNeedsStructureRebuild;
      }
    }
    return dirty_flags;
  }

  template <typename MapType>
  auto GetPendingSpecs(MapType& specs_map) {
    using SpecType = typename MapType::mapped_type;
    std::vector<SpecType*> pending_specs;
    pending_specs.reserve(specs_map.size());
    for (auto& [id, spec] : specs_map) {
      bool needs_bsdf_update = false;
      if constexpr (std::is_same_v<SpecType, MeshSpec>) {
        for (const auto& mat_id : spec.material_ids) {
          uint32_t mat_flags = 0;
          if (auto mat_it = material_dirty_flags_.find(mat_id);
              mat_it != material_dirty_flags_.end()) {
            mat_flags = mat_it->second;
          }
          if (mat_flags & DirtyFlags::kNeedsStructureRebuild) {
            spec.needs_rebuild = true;
          }
          if (mat_flags & DirtyFlags::kMaterialUpdated) {
            needs_bsdf_update = true;
          }
        }
      } else if constexpr (std::is_same_v<SpecType, CurveSpec>) {
        uint32_t mat_flags = 0;
        if (auto mat_it = material_dirty_flags_.find(spec.material_id);
            mat_it != material_dirty_flags_.end()) {
          mat_flags = mat_it->second;
        }
        if (mat_flags & DirtyFlags::kNeedsStructureRebuild) {
          spec.needs_rebuild = true;
        }
        if (mat_flags & DirtyFlags::kMaterialUpdated) {
          needs_bsdf_update = true;
        }
      }
      if (spec.needs_rebuild || spec.dirty_bits != 0 || needs_bsdf_update) {
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
      } else if (pending_specs[i]->dirty_bits != 0) {
        if constexpr (std::is_same_v<typename MapType::mapped_type,
                                     MaterialSpec>) {
          std::string id_str = pending_specs[i]->id.GetAsString();
          if (bsdfs_.contains(id_str)) {
            material_dirty_flags_[pending_specs[i]->id] |=
                DirtyFlags::kMaterialUpdated;
          }
        }
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

    ParallelCommit<decltype(material_specs_),
                   typename PrimTranslator::TranslatedMaterial>(
        material_specs_,
        [&](MaterialSpec* spec,
            typename PrimTranslator::TranslatedMaterial& res) {
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildMaterial(*spec, texture_cache_);
          } else if (spec->dirty_bits != 0) {
            auto bsdf_it = bsdfs_.find(spec->id.GetAsString());
            if (!TF_VERIFY(bsdf_it != bsdfs_.end(), "Material not found: %s",
                           spec->id.GetText())) {
              return;
            }
            PrimTranslator::UpdateMaterialInPlace(bsdf_it->second.get(), *spec,
                                                  texture_cache_);
          }
        },
        [&](MaterialSpec* spec,
            typename PrimTranslator::TranslatedMaterial& trans) {
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

  bool CommitCameras() {
    return ParallelCommit<decltype(camera_specs_), mitsuba::ref<Sensor>>(
        camera_specs_,
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildSensor(*spec, progressive_rendering_);
          } else if (spec->dirty_bits != 0) {
            if (spec->sensor_type != "irradiancemeter") {
              auto it = sensors_.find(spec->id.GetAsString());
              if (!TF_VERIFY(it != sensors_.end(),
                             "Camera sensor not found: %s",
                             spec->id.GetText())) {
                return;
              }
              PrimTranslator::UpdateSensorInPlace(it->second.get(), *spec);
            }
          }
        },
        [&](CameraSpec* spec, mitsuba::ref<Sensor>& res) {
          if (spec->sensor_type == "irradiancemeter") {
            surface_sensors_[spec->id.GetAsString()] = res;
          } else {
            sensors_[spec->id.GetAsString()] = res;
          }
          return true;
        });
  }

  struct EmitterSensorPair {
    mitsuba::ref<Object> mesh_emitter = nullptr;
    Object* emitter_ptr = nullptr;
    Object* sensor_ptr = nullptr;
  };

  EmitterSensorPair ResolveEmitterAndSensor(
      const std::optional<LightSpec>& emitter_spec, const SdfPath& material_id,
      const std::optional<SdfPath>& attached_sensor_id) {
    EmitterSensorPair pair;
    if (emitter_spec.has_value()) {
      const LightSpec& l_spec = emitter_spec.value();
      mitsuba::Properties emitter_props("area");
      emitter_props.set("radiance", mitsuba::Color<float, 3>(
                                        l_spec.emission[0], l_spec.emission[1],
                                        l_spec.emission[2]));
      pair.mesh_emitter = mitsuba::PluginManager::instance()->create_object(
          emitter_props, mitsuba::Emitter<Float, Spectrum>::Variant,
          mitsuba::Emitter<Float, Spectrum>::Type);
      pair.emitter_ptr = pair.mesh_emitter.get();
    }
    if (pair.emitter_ptr == nullptr) {
      auto emitter_it = material_emitters_.find(material_id.GetAsString());
      if (emitter_it != material_emitters_.end()) {
        pair.mesh_emitter = mitsuba::PluginManager::instance()->create_object(
            emitter_it->second, mitsuba::Emitter<Float, Spectrum>::Variant,
            mitsuba::Emitter<Float, Spectrum>::Type);
        pair.emitter_ptr = pair.mesh_emitter.get();
      }
    }
    if (attached_sensor_id.has_value()) {
      auto sens_it =
          surface_sensors_.find(attached_sensor_id.value().GetAsString());
      if (sens_it != surface_sensors_.end()) {
        pair.sensor_ptr = sens_it->second.get();
      }
    }
    return pair;
  }

  void CommitNonInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res) {
    const auto& spec = *(work->spec);
    if (spec.needs_rebuild) {
      // Run geometry pipeline to get sub-meshes
      auto sub_meshes = RunGeometryPipeline(spec, work->displacement_textures);
      res.meshes.reserve(sub_meshes.size());
      for (const auto& sub_mesh : sub_meshes) {
        auto env = ResolveEmitterAndSensor(
            spec.emitter_spec, sub_mesh.material_id, spec.attached_sensor_id);
        ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);
        auto mesh = PrimTranslator::BuildMesh(sub_mesh.id, sub_mesh.triangles,
                                              sub_mesh.primvars, bsdf.get(),
                                              env.emitter_ptr, env.sensor_ptr);
        if (mesh) {
          res.meshes.push_back(mesh);
        }
      }
    } else {
      // Update in place.
      auto sub_meshes = RunGeometryPipeline(spec, work->displacement_textures);
      for (const auto& sub_mesh : sub_meshes) {
        std::string sub_mesh_id_str = sub_mesh.id.GetAsString();
        auto it = shapes_.find(sub_mesh_id_str);
        if (!TF_VERIFY(it != shapes_.end(), "Sub-mesh not found: %s",
                       sub_mesh_id_str.c_str())) {
          continue;
        }
        ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);
        if (spec.dirty_bits != 0) {
          PrimTranslator::UpdateMeshInPlace(
              it->second.get(), sub_mesh.triangles, sub_mesh.primvars);
          // Update emissive mesh radiance in-place
          if (it->second->is_emitter() && spec.emitter_spec.has_value()) {
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
        }
        it->second->set_bsdf(bsdf.get());
      }
    }
  }

  void CommitInstancedMeshWork(MeshCommitWork* work, CommittedMesh& res) {
    const auto& spec = *(work->spec);
    std::string id_str = spec.id.GetAsString();
    if (spec.needs_rebuild) {
      bool warned_emitter = false;
      if (spec.emitter_spec.has_value()) {
        TF_WARN(
            "Mesh %s is instanced but has an emitter attached. Mitsuba "
            "does not support emitters on instances. Ignoring emitter.",
            spec.id.GetText());
        warned_emitter = true;
      }

      // Run geometry pipeline
      auto sub_meshes = RunGeometryPipeline(spec, work->displacement_textures);
      if (sub_meshes.empty()) return;

      // 1. Build prototype meshes
      std::vector<mitsuba::ref<Shape>> prototype_shapes;
      prototype_shapes.reserve(sub_meshes.size());
      for (const auto& sub_mesh : sub_meshes) {
        auto env = ResolveEmitterAndSensor(std::nullopt, sub_mesh.material_id,
                                           spec.attached_sensor_id);
        if (env.emitter_ptr != nullptr && !warned_emitter) {
          TF_WARN(
              "Mesh %s is instanced but has an emitter attached. Mitsuba "
              "does not support emitters on instances. Ignoring emitter.",
              spec.id.GetText());
          warned_emitter = true;
        }
        ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);

        auto mesh = PrimTranslator::BuildMesh(sub_mesh.id, sub_mesh.triangles,
                                              sub_mesh.primvars, bsdf.get(),
                                              nullptr, env.sensor_ptr);
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
      auto sub_meshes = RunGeometryPipeline(spec, work->displacement_textures);
      for (const auto& sub_mesh : sub_meshes) {
        std::string sub_mesh_id_str = sub_mesh.id.GetAsString();
        auto it = shapes_.find(absl::StrCat(kProtoPrefix, sub_mesh_id_str));
        if (TF_VERIFY(it != shapes_.end())) {
          ref<BSDF> bsdf = ResolveBsdf(sub_mesh.material_id, sub_mesh.primvars);

          if (spec.dirty_bits != 0) {
            PrimTranslator::UpdateMeshInPlace(
                it->second.get(), sub_mesh.triangles, sub_mesh.primvars);
          }
          it->second->set_bsdf(bsdf.get());
        }
      }
      if (spec.transforms_dirty) {
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
      work_items[i].displacement_textures.reserve(spec->material_ids.size());
      for (const auto& material_id : spec->material_ids) {
        auto texture_it =
            displacement_textures_.find(material_id.GetAsString());
        if (texture_it != displacement_textures_.end()) {
          work_items[i].displacement_textures.push_back(texture_it->second);
        } else {
          work_items[i].displacement_textures.push_back(nullptr);
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
          for (size_t i = r.begin(); i != r.end(); ++i) {
            if (work_items[i].spec->instance_transforms.empty()) {
              CommitNonInstancedMeshWork(&work_items[i], results[i]);
            } else {
              CommitInstancedMeshWork(&work_items[i], results[i]);
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
        for (auto it = shapes_.begin(); it != shapes_.end();) {
          if (absl::StartsWith(it->first, id_str + "/")) {
            shapes_.erase(it++);
          } else {
            ++it;
          }
        }
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

  bool CommitLights() {
    return ParallelCommit<decltype(light_specs_),
                          typename PrimTranslator::TranslatedLight>(
        light_specs_,
        [&](LightSpec* spec, typename PrimTranslator::TranslatedLight& res) {
          std::string id_str = spec->id.GetAsString();
          if (spec->needs_rebuild) {
            res = PrimTranslator::BuildLight(*spec);
          } else if (spec->dirty_bits != 0) {
            bool is_shape = spec->prim_type == HdPrimTypeTokens->rectLight ||
                            spec->prim_type == HdPrimTypeTokens->diskLight ||
                            (spec->prim_type == HdPrimTypeTokens->sphereLight &&
                             !spec->treat_as_point);
            if (is_shape) {
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
    rebuild_scene |= CommitLights();

    if (rebuild_scene) {
      if constexpr (dr::is_jit_v<Float>) {
        if (frozen_render_) {
          frozen_render_->Clear();
        }
      }
    }
    if (rebuild_scene || !material_dirty_flags_.empty()) {
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
    GarbageCollectTextureCache();

    material_dirty_flags_.clear();
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
  bool has_instancing_ = false;

  // Progressive rendering state
  bool progressive_rendering_ = false;
  int current_progressive_sample_ = 0;
  TensorXf accum_buffer_;
  bool reset_progressive_ = true;

  absl::Mutex state_mutex_;
  absl::Mutex aov_states_mutex_;

  absl::flat_hash_map<SdfPath, MaterialSpec, SdfPath::Hash> material_specs_;
  absl::flat_hash_map<SdfPath, MeshSpec, SdfPath::Hash> mesh_specs_;
  absl::flat_hash_map<SdfPath, CurveSpec, SdfPath::Hash> curve_specs_;
  absl::flat_hash_map<SdfPath, LightSpec, SdfPath::Hash> light_specs_;
  absl::flat_hash_map<SdfPath, CameraSpec, SdfPath::Hash> camera_specs_;
  absl::flat_hash_map<SdfPath, uint32_t, SdfPath::Hash> material_dirty_flags_;

  absl::flat_hash_map<std::string, ref<Sensor>> sensors_;
  absl::flat_hash_map<std::string, mitsuba::ref<Object>> surface_sensors_;
  absl::flat_hash_map<std::string, ref<Shape>> shapes_;
  absl::flat_hash_map<std::string, ref<Emitter>> emitters_;
  absl::flat_hash_map<std::string, ref<BSDF>> bsdfs_;
  absl::flat_hash_map<std::string, ref<Texture>> displacement_textures_;
  absl::flat_hash_map<std::string, mitsuba::Properties> material_emitters_;
  ref<BSDF> default_bsdf_ = nullptr;
  absl::flat_hash_map<std::tuple<float, float, float>, ref<BSDF>> color_bsdfs_;
  absl::Mutex color_bsdfs_mutex_;

  TextureCache texture_cache_;
  std::unique_ptr<FrozenRender<Float, Spectrum>> frozen_render_ = nullptr;

  void PreloadTextures(const std::vector<MaterialSpec*>& pending_specs) {
    absl::flat_hash_set<mitsuba::Properties, PropertiesHash, PropertiesEqual>
        textures_to_load;
    for (const auto* spec : pending_specs) {
      const HdMaterialNetwork2& network2 = spec->network2;
      DiscoverTextures(network2, [&](const mitsuba::Properties& tex_props) {
        // Find any textures that still need to be loaded.
        if (!texture_cache_.contains(tex_props) &&
            !textures_to_load.contains(tex_props)) {
          textures_to_load.insert(tex_props);
        }
      });
    }
    if (textures_to_load.empty()) return;
    std::vector<mitsuba::Properties> texture_list(textures_to_load.begin(),
                                                  textures_to_load.end());
    std::vector<mitsuba::ref<mitsuba::Object>> loaded_textures(
        texture_list.size());
    drjit::parallel_for(drjit::blocked_range<size_t>(0, texture_list.size()),
                        [&](drjit::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            JitScopeGuard<Float> jit_guard;
                            loaded_textures[i] =
                                PrimTranslator::LoadTexture(texture_list[i]);
                          }
                        });
    for (size_t i = 0; i < texture_list.size(); ++i) {
      if (loaded_textures[i]) {
        texture_cache_[texture_list[i]] = loaded_textures[i];
      }
    }
  }

  void GarbageCollectTextureCache() {
    absl::flat_hash_set<mitsuba::Properties, PropertiesHash, PropertiesEqual>
        active_textures;
    for (const auto& [id, spec] : material_specs_) {
      const HdMaterialNetwork2& network2 = spec.network2;
      DiscoverTextures(network2, [&](const mitsuba::Properties& tex_props) {
        active_textures.insert(tex_props);
      });
    }
    for (auto it = texture_cache_.begin(); it != texture_cache_.end();) {
      if (!active_textures.contains(it->first)) {
        texture_cache_.erase(it++);
      } else {
        ++it;
      }
    }
  }

  bool CleanUpInstancing(const SdfPath& id) {
    std::string id_str = id.GetAsString();
    bool erased = false;
    if (shapes_.erase(absl::StrCat(kProtoPrefix, id_str)) > 0) erased = true;
    // Erase prototype sub-meshes
    std::string proto_prefix = absl::StrCat(kProtoPrefix, id_str, "/");
    for (auto it = shapes_.begin(); it != shapes_.end();) {
      if (absl::StartsWith(it->first, proto_prefix)) {
        shapes_.erase(it++);
        erased = true;
      } else {
        ++it;
      }
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
