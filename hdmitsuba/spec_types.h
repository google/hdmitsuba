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
#include <optional>
#include <string>
#include <tuple>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

struct BaseSpec {
  SdfPath id;
  bool needs_rebuild = false;
  HdDirtyBits dirty_bits = 0;

  void FoldPendingFrom(const BaseSpec& prev) {
    needs_rebuild |= prev.needs_rebuild;
    if (!needs_rebuild) {
      dirty_bits |= prev.dirty_bits;
    }
  }

  void MarkClean() {
    needs_rebuild = false;
    dirty_bits = 0;
  }
};

struct MaterialSpec : public BaseSpec {
  HdMaterialNetwork2 network2;
};

// UsdLuxShapingAPI IES profile of a light (`inputs:shaping:ies:*`).
struct IesSpec {
  std::string file_path;
  float angle_scale = 0.0f;
  bool normalize = false;

  bool operator==(const IesSpec& o) const {
    return std::tie(file_path, angle_scale, normalize) ==
           std::tie(o.file_path, o.angle_scale, o.normalize);
  }
  bool operator!=(const IesSpec& o) const { return !(*this == o); }
};

struct LightSpec : public BaseSpec {
  TfToken prim_type;
  // For mesh lights, which bake their transform into the vertices, this is the
  // light frame of the `ies` emitter (see `GetMeshEmitterSpec` in mesh.cc).
  MotionTransform transform;
  GfVec3f emission;
  float angle = 0.0f;
  float shaping_cone_angle = 0.0f;
  float shaping_cone_beam_width = 0.0f;
  bool treat_as_point = false;
  std::string texture_file_path;
  std::string visibility;
  // Whether the `lightLink` collection includes everything (the default).
  bool light_link_everything = true;
  IesSpec ies;

  // Area lights are represented as Mitsuba shapes with an attached area
  // emitter, rather than standalone emitter objects.
  bool IsAreaLight() const {
    return prim_type == HdPrimTypeTokens->rectLight ||
           prim_type == HdPrimTypeTokens->diskLight ||
           (prim_type == HdPrimTypeTokens->sphereLight && !treat_as_point);
  }
};

// An implicit surface rendered as a native Mitsuba shape (see
// HdMitsubaImplicitSurface).
struct ImplicitSurface {
  TfToken type;  // HdPrimTypeTokens->sphere or HdPrimTypeTokens->cylinder.
  float radius = 1.0f;
  float height = 2.0f;  // Cylinders only, which are centered and aligned with Z.

  // Returns whether the native Mitsuba shape can be placed by `transform`
  // without distortion. Spheres require a similarity transform, and cylinders
  // an orthogonal one with uniform scaling in X and Y.
  bool SupportsTransform(const ScalarAffineTransform4f& transform) const {
    using Vector3f = mitsuba::Vector<float, 3>;
    const Vector3f x = transform * Vector3f(1.f, 0.f, 0.f),
                   y = transform * Vector3f(0.f, 1.f, 0.f),
                   z = transform * Vector3f(0.f, 0.f, 1.f);
    const float lx = drjit::norm(x), ly = drjit::norm(y), lz = drjit::norm(z);
    // Same tolerance as Mitsuba's checks in the `sphere` and `cylinder` plugins.
    constexpr float kEpsilon = 1e-4f;
    return std::abs(drjit::dot(x, y)) <= kEpsilon * lx * ly &&
           std::abs(drjit::dot(x, z)) <= kEpsilon * lx * lz &&
           std::abs(drjit::dot(y, z)) <= kEpsilon * ly * lz &&
           std::abs(lx - ly) <= kEpsilon * lx &&
           (type == HdPrimTypeTokens->cylinder ||
            std::abs(lx - lz) <= kEpsilon * lx);
  }
};

struct MeshSpec : public BaseSpec {
  std::vector<SdfPath> material_ids;
  VtIntArray face_material_indices;
  VtIntArray face_vertex_counts;
  VtIntArray face_vertex_indices;
  PrimvarMap primvars;
  GfMatrix4d transform;
  std::optional<LightSpec> emitter_spec = std::nullopt;
  // If set, the shape is rendered as this native Mitsuba shape with a single
  // material, and the topology is unused.
  std::optional<ImplicitSurface> implicit_surface = std::nullopt;
  // Instance-to-world transforms when rendered through Mitsuba instances. For a
  // non-instanced mesh with an animated transform (or an implicit surface whose
  // transform is not supported by its native shape) this holds a single entry
  // and `motion_wrapped` is set; `transform` then still holds the static
  // transform at frame time so the SceneManager can fall back to it.
  std::vector<MotionTransform> instance_transforms;
  bool motion_wrapped = false;
  bool is_subdivided = false;
  bool double_sided = false;

  void FoldPendingFrom(const MeshSpec& prev) {
    if (emitter_spec.has_value() != prev.emitter_spec.has_value() ||
        (emitter_spec && emitter_spec->ies != prev.emitter_spec->ies) ||
        material_ids != prev.material_ids ||
        double_sided != prev.double_sided ||
        !std::equal(instance_transforms.begin(), instance_transforms.end(),
                    prev.instance_transforms.begin(),
                    prev.instance_transforms.end(),
                    [](const MotionTransform& a, const MotionTransform& b) {
                      return a.IsAnimated() == b.IsAnimated();
                    })) {
      needs_rebuild = true;
    }
    BaseSpec::FoldPendingFrom(prev);
  }
};

struct CurveSpec : public BaseSpec {
  SdfPath material_id;
  ScalarAffineTransform4f transform;
  std::string plugin_name;
  std::vector<float> control_points;  // Flat packed [x, y, z, r] control points
  std::vector<uint32_t> segment_indices;  // Precomputed segment indices
};

struct ParticleFieldSpec : public BaseSpec {
  GfMatrix4d transform;
  VtVec3fArray points;
  VtVec3fArray scales;
  VtQuatfArray orientations;
  VtFloatArray opacities;
  VtVec3fArray sh_coeffs;
  int sh_degree = 0;
  // Which parts of the plugin's parameters changed since the last sync. Only
  // consulted when `needs_rebuild` is false, i.e. when the particle count, SH
  // layout, and adaptive clamping setting are unchanged and the shape can be
  // updated in place.
  bool geometry_dirty = true;
  bool opacities_dirty = true;
  bool sh_dirty = true;

  bool attributes_dirty() const { return opacities_dirty || sh_dirty; }

  void FoldPendingFrom(const ParticleFieldSpec& prev) {
    needs_rebuild |= prev.needs_rebuild;
    geometry_dirty |= prev.geometry_dirty;
    opacities_dirty |= prev.opacities_dirty;
    sh_dirty |= prev.sh_dirty;
  }

  void MarkClean() {
    BaseSpec::MarkClean();
    geometry_dirty = false;
    opacities_dirty = false;
    sh_dirty = false;
  }
};

struct CameraSpec : public BaseSpec {
  std::string sensor_type = "perspective";
  MotionTransform transform;
  GfVec2f shutter{0.0f};
  float fov = 90.0f;
  float horizontal_aperture_offset = 0.0f;
  float vertical_aperture_offset = 0.0f;
  float near_clip = 0.01f;
  float far_clip = 1000.0f;
  std::string pixel_filter_type = "";
  std::optional<SdfPath> target_shape_id = std::nullopt;
};

PXR_NAMESPACE_CLOSE_SCOPE
