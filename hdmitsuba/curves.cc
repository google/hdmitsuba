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

#include "hdmitsuba/curves.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/basisCurves.h>
#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/materialBindingSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

float CalculateMeanWidth(const VtFloatArray& widths,
                         float default_radius = 0.01f) {
  if (widths.empty()) return default_radius;
  double sum = 0.0;
  for (float w : widths) {
    sum += w;
  }
  return static_cast<float>(sum / static_cast<double>(widths.size()));
}

template <typename WidthFn>
std::vector<float> PackControlPoints(const VtVec3fArray& points,
                                     WidthFn&& width_fn) {
  std::vector<float> control_points(points.size() * 4);
  for (size_t i = 0; i < points.size(); ++i) {
    const GfVec3f& point = points[i];
    control_points[4 * i + 0] = point[0];
    control_points[4 * i + 1] = point[1];
    control_points[4 * i + 2] = point[2];
    control_points[4 * i + 3] = width_fn(i);
  }
  return control_points;
}

// Converts Catmull-Rom strands into disjoint 4-point B-spline segments via
// M_bspline^-1 * M_catmullRom. With `wrap = pinned`, reflects phantom
// endpoint positions (2*P_0 - P_1, 2*P_{N-1} - P_{N-2}) while clamping
// endpoint radii so tapered tips do not extrapolate to negative radii.
std::vector<float> ConvertCatmullRomToBSpline(
    bool pinned, const VtIntArray& vertex_counts,
    const std::vector<float>& control_points) {
  static const GfMatrix4f kCatmullRomToBSpline(
      7.0f / 6.0f, -4.0f / 6.0f, 5.0f / 6.0f, -2.0f / 6.0f, -2.0f / 6.0f,
      11.0f / 6.0f, -4.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, -4.0f / 6.0f,
      11.0f / 6.0f, -2.0f / 6.0f, -2.0f / 6.0f, 5.0f / 6.0f, -4.0f / 6.0f,
      7.0f / 6.0f);
  int first = pinned ? -1 : 0;
  size_t num_segments = 0;
  for (int count : vertex_counts) {
    num_segments += std::max(0, count - 3 - 2 * first);
  }
  std::vector<float> bspline_points(num_segments * 16);
  float* out = bspline_points.data();
  int offset = 0;
  for (int count : vertex_counts) {
    for (int start = first; start + 3 <= count - 1 - first; ++start) {
      GfMatrix4f window;
      for (int k = 0; k < 4; ++k) {
        int local = start + k;
        int clamped = std::clamp(local, 0, count - 1);
        // For in-range indices, clamped == local, so 2*anchor - neighbor ==
        // anchor.
        GfVec4f anchor(&control_points[4 * (offset + clamped)]);
        GfVec4f neighbor(&control_points[4 * (offset + 2 * clamped - local)]);
        GfVec4f row = 2.0f * anchor - neighbor;
        row[3] = anchor[3];
        window.SetRow(k, row);
      }
      out = std::copy_n((kCatmullRomToBSpline * window).data(), 16, out);
    }
    offset += count;
  }
  return bspline_points;
}

}  // namespace

HdMitsubaCurves::HdMitsubaCurves(const SdfPath& id) : HdBasisCurves(id) {}

HdDirtyBits HdMitsubaCurves::GetInitialDirtyBitsMask() const {
  return HdChangeTracker::Clean | HdChangeTracker::DirtyPrimvar |
         HdChangeTracker::DirtyRepr | HdChangeTracker::DirtyTopology |
         HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
         HdChangeTracker::DirtyMaterialId;
}

void HdMitsubaCurves::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                           const TfToken& /*reprToken*/) {
  static const HdDataSourceLocator points_locator(
      HdPrimvarsSchema::GetSchemaToken(), HdTokens->points,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator widths_locator(
      HdPrimvarsSchema::GetSchemaToken(), HdTokens->widths,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator material_locator(
      HdMaterialBindingsSchema::GetSchemaToken(),
      HdMaterialBindingsSchemaTokens->allPurpose,
      HdMaterialBindingSchemaTokens->path);
  static const HdDataSourceLocator basis_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->basis);
  static const HdDataSourceLocator type_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->type);
  static const HdDataSourceLocator wrap_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->wrap);
  static const HdDataSourceLocator vertex_counts_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->curveVertexCounts);

  if (*dirtyBits == HdChangeTracker::Clean) return;

  const SdfPath& id = GetId();
  HdContainerDataSourceHandle data_source =
      GetPrimDataSource(sceneDelegate, id);

  // 1. Visibility
  bool visible = GetPrimVisible(data_source);
  if ((*dirtyBits & HdChangeTracker::DirtyVisibility) && visible) {
    *dirtyBits |= HdChangeTracker::AllDirty;
  }
  SceneManager* scene_manager =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  if (!visible) {
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 2. Transform
  GfMatrix4d transform = GetPrimTransform(data_source);

  // 3. Primvars (Points and Widths)
  if (points_.empty() || widths_.empty() ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points) ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->widths)) {
    points_ = GetParam<VtVec3fArray>(data_source, points_locator);
    if (points_.empty()) {
      TF_WARN("Curves %s: missing or invalid points primvar.", id.GetText());
    }
    widths_ = GetParam<VtFloatArray>(data_source, widths_locator);
    if (widths_.empty()) {
      TF_WARN("Curves %s: missing or invalid widths primvar.", id.GetText());
    }
  }

  // 4. Topology
  std::string plugin_name = "linearcurve";
  TfToken basis = GetParam<TfToken>(data_source, basis_locator);
  TfToken type = GetParam<TfToken>(data_source, type_locator);
  const bool is_catmull_rom =
      basis == HdTokens->catmullRom && type == HdTokens->cubic;
  if ((basis == HdTokens->bspline && type == HdTokens->cubic) ||
      is_catmull_rom) {
    plugin_name = "bsplinecurve";
  }
  const VtIntArray vertex_counts =
      GetParam<VtIntArray>(data_source, vertex_counts_locator);

  if (points_.empty() || widths_.empty()) {
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Curves %s has no points or widths. Removing from scene.\n",
             id.GetText());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  int min_count = 0;
  ptrdiff_t total_vertices = 0;
  for (int count : vertex_counts) {
    min_count = std::min(min_count, count);
    total_vertices += count;
  }
  if (min_count < 0 || static_cast<size_t>(total_vertices) != points_.size()) {
    TF_WARN(
        "Curves %s: curveVertexCounts sum to %td (min %d) but there are %zu "
        "points. Fix the curve topology in the USD asset. Removing from scene.",
        id.GetText(), total_vertices, min_count, points_.size());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 5. Material binding
  SdfPath material_id = GetParam<SdfPath>(data_source, material_locator);

  // 6. Convert data to CurveSpec and pass it to the scene manager.
  CurveSpec spec;
  spec.id = id;
  spec.transform = UsdToMitsubaTransform(transform);
  spec.material_id = std::move(material_id);
  spec.needs_rebuild = true;
  spec.plugin_name = std::move(plugin_name);

  // Pack control points [x, y, z, r] (lazy-evaluating widths based on layout)
  if (widths_.size() == points_.size()) {
    spec.control_points = PackControlPoints(
        points_, [&](size_t i) { return widths_.AsConst()[i] * 0.5f; });
  } else if (widths_.size() == 1) {
    float r = widths_.AsConst()[0] * 0.5f;
    spec.control_points =
        PackControlPoints(points_, [&](size_t /*i*/) { return r; });
  } else {
    float uniform_radius = CalculateMeanWidth(widths_) * 0.5f;
    spec.control_points = PackControlPoints(
        points_, [&](size_t /*i*/) { return uniform_radius; });
  }

  if (is_catmull_rom) {
    spec.control_points = ConvertCatmullRomToBSpline(
        GetParam<TfToken>(data_source, wrap_locator) == HdTokens->pinned,
        vertex_counts, spec.control_points);
  }

  // Generate segment indices
  int degree = (spec.plugin_name == "bsplinecurve") ? 3 : 1;
  int accumulated_vertices = 0;
  for (int count : is_catmull_rom
                       ? VtIntArray(spec.control_points.size() / 16, 4)
                       : vertex_counts) {
    for (int j = 0; j < count - degree; ++j) {
      spec.segment_indices.push_back(accumulated_vertices + j);
    }
    accumulated_vertices += count;
  }
  if (spec.segment_indices.empty()) {
    TF_WARN("Curves %s has no valid segments. Removing from scene.",
            id.GetText());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }
  scene_manager->SyncCurves(std::move(spec));
  in_scene_ = true;
  *dirtyBits = HdChangeTracker::Clean;
}

void HdMitsubaCurves::Finalize(HdRenderParam* renderParam) {
  RemoveFromScene(static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene());
}

void HdMitsubaCurves::RemoveFromScene(SceneManager* scene) {
  if (in_scene_) {
    scene->RemoveShape(GetId());
    in_scene_ = false;
  }
}

HdDirtyBits HdMitsubaCurves::_PropagateDirtyBits(HdDirtyBits bits) const {
  return bits;
}

void HdMitsubaCurves::_InitRepr(const TfToken& /*reprToken*/,
                                HdDirtyBits* /*dirtyBits*/) {}

PXR_NAMESPACE_CLOSE_SCOPE
