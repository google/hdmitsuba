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

#include <utility>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/basisCurves.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"

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
  if (*dirtyBits == HdChangeTracker::Clean) return;

  const SdfPath& id = GetId();
  bool visible = sceneDelegate->GetVisible(id);
  if ((*dirtyBits & HdChangeTracker::DirtyVisibility) && visible) {
    *dirtyBits |= HdChangeTracker::AllDirty;
  }

  SceneManager* sceneManager =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  if (!visible) {
    RemoveFromScene(sceneManager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  if (points_.empty() || widths_.empty() ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points) ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->widths)) {
    points_ = sceneDelegate->Get(id, HdTokens->points)
                  .GetWithDefault<VtVec3fArray>(VtVec3fArray());
    if (points_.empty()) {
      TF_WARN("Curves %s: missing or invalid points primvar.", id.GetText());
    }
    widths_ = sceneDelegate->Get(id, HdTokens->widths)
                  .GetWithDefault<VtFloatArray>(VtFloatArray());
    if (widths_.empty()) {
      TF_WARN("Curves %s: missing or invalid widths primvar.", id.GetText());
    }
  }

  if (points_.empty() || widths_.empty()) {
    TF_WARN("Curves %s has no points or widths. Removing from scene.",
            id.GetText());
    RemoveFromScene(sceneManager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  CurveSpec spec;
  spec.id = id;
  spec.transform = UsdToMitsubaTransform(sceneDelegate->GetTransform(id));
  spec.material_id = sceneDelegate->GetMaterialId(id);
  spec.needs_rebuild = true;

  // 1. Determine plugin name
  HdBasisCurvesTopology topology = GetBasisCurvesTopology(sceneDelegate);
  std::string plugin_name = "linearcurve";
  if (topology.GetCurveBasis() == TfToken("bspline") &&
      topology.GetCurveType() == TfToken("cubic")) {
    plugin_name = "bsplinecurve";
  }
  spec.plugin_name = plugin_name;

  // 2. Pack control points [x, y, z, r] (lazy-evaluating widths based on
  // layout)
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

  // 4. Generate segment indices
  int degree = (plugin_name == "bsplinecurve") ? 3 : 1;
  const auto& vertex_counts = topology.GetCurveVertexCounts();
  int n_indices = 0;
  for (size_t i = 0; i < vertex_counts.size(); ++i) {
    n_indices += vertex_counts[i] - degree;
  }
  spec.segment_indices.resize(n_indices);
  int index = 0;
  for (size_t i = 0; i < vertex_counts.size(); ++i) {
    int offset = i * degree;
    for (int j = 0; j < vertex_counts[i] - degree; ++j, ++index) {
      spec.segment_indices[index] = index + offset;
    }
  }

  sceneManager->SyncCurves(std::move(spec));
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
