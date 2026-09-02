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

#include <optional>
#include <string>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/mesh/geometry_processor.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

struct BaseSpec {
  SdfPath id;
  bool needs_rebuild = false;
  HdDirtyBits dirty_bits = 0;

  void MarkClean() {
    needs_rebuild = false;
    dirty_bits = 0;
  }
};

struct MaterialSpec : public BaseSpec {
  HdMaterialNetwork2 network2;
};

struct LightSpec : public BaseSpec {
  TfToken prim_type;
  ScalarAffineTransform4f transform;
  GfVec3f emission;
  float shaping_cone_angle = 0.0f;
  float shaping_cone_beam_width = 0.0f;
  bool treat_as_point = false;
  std::string texture_file_path;
};

struct MeshSpec : public BaseSpec {
  std::vector<SdfPath> material_ids;
  VtIntArray face_material_indices;
  VtIntArray face_vertex_counts;
  VtIntArray face_vertex_indices;
  PrimvarMap primvars;
  GfMatrix4d transform;
  std::optional<SdfPath> attached_sensor_id = std::nullopt;
  std::optional<LightSpec> emitter_spec = std::nullopt;
  VtMatrix4dArray instance_transforms;
  bool transforms_dirty = false;
  bool is_subdivided = false;
};

struct CurveSpec : public BaseSpec {
  SdfPath material_id;
  ScalarAffineTransform4f transform;
  std::string plugin_name;
  std::vector<float> control_points;  // Flat packed [x, y, z, r] control points
  std::vector<uint32_t> segment_indices;  // Precomputed segment indices
  std::optional<SdfPath> attached_sensor_id = std::nullopt;
};

struct ParticleFieldSpec : public BaseSpec {
  GfMatrix4d transform;
  VtVec3fArray points;
  VtVec3fArray scales;
  VtQuatfArray orientations;
  VtFloatArray opacities;
  VtVec3fArray sh_coeffs;
  int sh_degree = 0;
  // Which half of the plugin's parameters changed since the last sync. Only
  // consulted when `needs_rebuild` is false, i.e. when the particle count and
  // SH layout are unchanged and the shape can be updated in place.
  bool geometry_dirty = true;
  bool attributes_dirty = true;

  void MarkClean() {
    BaseSpec::MarkClean();
    geometry_dirty = false;
    attributes_dirty = false;
  }
};

struct CameraSpec : public BaseSpec {
  std::string sensor_type = "perspective";
  ScalarAffineTransform4f transform;
  float fov = 90.0f;
  float horizontal_aperture_offset = 0.0f;
  float vertical_aperture_offset = 0.0f;
  float near_clip = 0.01f;
  float far_clip = 1000.0f;
  std::string pixel_filter_type = "";
};

PXR_NAMESPACE_CLOSE_SCOPE
