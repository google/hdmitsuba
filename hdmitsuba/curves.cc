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
#include "hdmitsuba/debug_codes.h"

#include <pxr/base/gf/matrix4d.h>
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

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

float CalculateMeanWidth(const VtFloatArray& widths,
                         float default_radius = 0.01f) {
  if (widths.empty()) return default_radius;
  double mean = 0.0;
  for (size_t i = 0; i < widths.size(); ++i) {
    mean += (widths[i] - mean) / (i + 1);
  }
  return static_cast<float>(mean);
}

template <typename WidthFn>
std::vector<float> PackControlPoints(const VtVec3fArray& points,
                                     WidthFn&& width_fn) {
  std::vector<float> control_points;
  control_points.resize(points.size() * 4);
  for (size_t i = 0; i < points.size(); ++i) {
    const GfVec3f& point = points[i];
    control_points[4 * i + 0] = point[0];
    control_points[4 * i + 1] = point[1];
    control_points[4 * i + 2] = point[2];
    control_points[4 * i + 3] = width_fn(i);
  }
  return control_points;
}

}  // namespace

HdMitsubaCurves::HdMitsubaCurves(const SdfPath& id) : HdBasisCurves(id) {}

HdDirtyBits HdMitsubaCurves::GetInitialDirtyBits() const {
  return GetInitialDirtyBitsMask();
}

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
  static const HdDataSourceLocator transform_locator(
      HdXformSchema::GetSchemaToken(), HdXformSchemaTokens->matrix);
  static const HdDataSourceLocator material_locator(
      HdMaterialBindingsSchema::GetSchemaToken(),
      HdMaterialBindingsSchemaTokens->allPurpose,
      HdMaterialBindingSchemaTokens->path);
  static const HdDataSourceLocator visibility_locator(
      HdVisibilitySchema::GetSchemaToken(),
      HdVisibilitySchemaTokens->visibility);
  static const HdDataSourceLocator basis_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->basis);
  static const HdDataSourceLocator type_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->type);
  static const HdDataSourceLocator vertex_counts_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->curveVertexCounts);

  if (*dirtyBits == HdChangeTracker::Clean) return;

  const SdfPath& id = GetId();
  HdSceneIndexBaseRefPtr scene_index =
      sceneDelegate->GetRenderIndex().GetTerminalSceneIndex();
  if (!TF_VERIFY(scene_index)) {
    return;
  }
  HdContainerDataSourceHandle data_source = scene_index->GetPrim(id).dataSource;

  // 1. Visibility
  bool visible = GetParam<bool>(data_source, visibility_locator, true);
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
  GfMatrix4d transform =
      GetParam<GfMatrix4d>(data_source, transform_locator, GfMatrix4d(1.0));

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
  if (basis == TfToken("bspline") && type == TfToken("cubic")) {
    plugin_name = "bsplinecurve";
  }
  VtIntArray vertex_counts =
      GetParam<VtIntArray>(data_source, vertex_counts_locator);

  if (points_.empty() || widths_.empty()) {
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Curves %s has no points or widths. Removing from scene.\n",
             id.GetText());
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
  spec.material_id = material_id;
  spec.needs_rebuild = true;
  spec.plugin_name = plugin_name;

  // Pack control points [x, y, z, r] (lazy-evaluating widths based on layout)
  if (widths_.size() == points_.size()) {
    spec.control_points =
        PackControlPoints(points_, [&](size_t i) { return widths_[i]; });
  } else if (widths_.size() == 1) {
    float r = widths_[0];
    spec.control_points =
        PackControlPoints(points_, [&](size_t /*i*/) { return r; });
  } else {
    float uniform_radius = CalculateMeanWidth(widths_);
    spec.control_points = PackControlPoints(
        points_, [&](size_t /*i*/) { return uniform_radius; });
  }

  // Generate segment indices
  int degree = (plugin_name == "bsplinecurve") ? 3 : 1;
  int n_indices = 0;
  for (int count : vertex_counts) {
    n_indices += std::max(0, count - degree);
  }
  spec.segment_indices.resize(n_indices);
  int index = 0;
  int accumulated_vertices = 0;
  for (int count : vertex_counts) {
    int num_segments = count - degree;
    for (int j = 0; j < num_segments; ++j, ++index) {
      spec.segment_indices[index] = accumulated_vertices + j;
    }
    accumulated_vertices += count;
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
