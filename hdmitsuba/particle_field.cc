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

#include "hdmitsuba/particle_field.h"

#include <utility>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdVol/tokens.h>

#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaParticleField::HdMitsubaParticleField(const SdfPath& id)
    : HdRprim(id) {}

HdDirtyBits HdMitsubaParticleField::GetInitialDirtyBits() const {
  return GetInitialDirtyBitsMask();
}

HdDirtyBits HdMitsubaParticleField::GetInitialDirtyBitsMask() const {
  return HdChangeTracker::Clean | HdChangeTracker::DirtyPrimvar |
         HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility;
}

TfTokenVector const& HdMitsubaParticleField::GetBuiltinPrimvarNames() const {
  static const TfTokenVector primvarNames = {
      UsdVolTokens->positions,
      UsdVolTokens->scales,
      UsdVolTokens->orientations,
      UsdVolTokens->opacities,
      UsdVolTokens->radianceSphericalHarmonicsCoefficients,
      UsdVolTokens->radianceSphericalHarmonicsDegree};
  return primvarNames;
}

void HdMitsubaParticleField::Sync(HdSceneDelegate* sceneDelegate,
                                  HdRenderParam* renderParam,
                                  HdDirtyBits* dirtyBits,
                                  const TfToken& /*reprToken*/) {
  if (*dirtyBits == HdChangeTracker::Clean) return;

  const SdfPath& id = GetId();
  HdSceneIndexBaseRefPtr scene_index =
      sceneDelegate->GetRenderIndex().GetTerminalSceneIndex();
  if (!TF_VERIFY(scene_index)) return;

  HdContainerDataSourceHandle prim_source = scene_index->GetPrim(id).dataSource;

  // 1. Visibility & Transform (using Pixar schema wrappers directly)
  if (auto vis_schema = HdVisibilitySchema::GetFromParent(prim_source)) {
    if (vis_schema.GetVisibility() &&
        !vis_schema.GetVisibility()->GetValue(0.0f).Get<bool>()) {
      auto* mitsubaRenderParam =
          static_cast<HdMitsubaRenderParam*>(renderParam);
      RemoveFromScene(mitsubaRenderParam->GetScene());
      *dirtyBits = HdChangeTracker::Clean;
      return;
    }
  }

  const size_t previous_count = points_.size();
  const size_t previous_sh_size = sh_coeffs_.size();
  // Only refetch everything when we hold no data yet; a prim that was removed
  // for being invisible still has valid cached primvars.
  const bool refetch_all = points_.empty();

  // "geometry" is what the plugin packs into its "data" parameter; the rest
  // are plain attributes it can swap without re-deriving its proxy mesh.
  bool geometry_dirty = false;
  bool attributes_dirty = false;

  if (HdChangeTracker::IsTransformDirty(*dirtyBits, id)) {
    if (auto xform_schema = HdXformSchema::GetFromParent(prim_source)) {
      if (xform_schema.GetMatrix()) {
        transform_ = xform_schema.GetMatrix()->GetValue(0.0f).Get<GfMatrix4d>();
      }
    }
    geometry_dirty = true;
  }

  // 2. Primvars — Open the /primvars container ONCE
  if (auto primvars_schema = HdPrimvarsSchema::GetFromParent(prim_source)) {
    // Hydra raises the coarse DirtyPrimvar bit, so every primvar reads as
    // dirty whenever any single one of them changes. Comparing against the
    // cached copy is what actually tells geometry edits apart from attribute
    // edits - and that distinction is worth an array compare here, because it
    // decides whether the plugin has to re-derive its proxy mesh.
    auto fetch_primvar = [&](const TfToken& name, auto& target,
                             bool& dirty_flag) {
      using T = std::decay_t<decltype(target)>;
      if (!refetch_all &&
          !HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, name)) {
        return;
      }
      auto pv = primvars_schema.GetPrimvar(name);
      if (!pv || !pv.GetPrimvarValue()) {
        return;
      }
      T value = pv.GetPrimvarValue()->GetValue(0.0f).Get<T>();
      if (!refetch_all && value == target) {
        return;
      }
      target = std::move(value);
      dirty_flag = true;
    };

    fetch_primvar(UsdVolTokens->positions, points_, geometry_dirty);
    fetch_primvar(UsdVolTokens->scales, scales_, geometry_dirty);
    fetch_primvar(UsdVolTokens->orientations, orientations_, geometry_dirty);
    fetch_primvar(UsdVolTokens->opacities, opacities_, attributes_dirty);
    fetch_primvar(UsdVolTokens->radianceSphericalHarmonicsCoefficients,
                  sh_coeffs_, attributes_dirty);

    if (refetch_all ||
        HdChangeTracker::IsPrimvarDirty(
            *dirtyBits, id, UsdVolTokens->radianceSphericalHarmonicsDegree)) {
      if (auto pv = primvars_schema.GetPrimvar(
              UsdVolTokens->radianceSphericalHarmonicsDegree)) {
        if (pv.GetPrimvarValue()) {
          const int degree = pv.GetPrimvarValue()->GetValue(0.0f).Get<int>();
          if (refetch_all || degree != sh_degree_) {
            sh_degree_ = degree;
            attributes_dirty = true;
          }
        }
      }
    }
  }

  if (in_scene_ && !geometry_dirty && !attributes_dirty) {
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // The proxy mesh and the attribute buffers are sized from the particle count
  // and the SH layout, so a change to either has to go through a full rebuild.
  // Anything else can be pushed into the existing shape.
  const bool needs_rebuild = !in_scene_ ||
                             points_.size() != previous_count ||
                             sh_coeffs_.size() != previous_sh_size;

  // 3. Scene Sync
  auto* mitsubaRenderParam = static_cast<HdMitsubaRenderParam*>(renderParam);
  SceneManager* sceneManager = mitsubaRenderParam->GetScene();

  ParticleFieldSpec spec;
  spec.id = id;
  spec.transform = transform_;
  spec.points = points_;
  spec.scales = scales_;
  spec.orientations = orientations_;
  spec.opacities = opacities_;
  spec.sh_coeffs = sh_coeffs_;
  spec.sh_degree = sh_degree_;
  spec.needs_rebuild = needs_rebuild;
  spec.geometry_dirty = geometry_dirty;
  spec.attributes_dirty = attributes_dirty;
  sceneManager->SyncParticleField(std::move(spec));
  in_scene_ = true;

  *dirtyBits = HdChangeTracker::Clean;
}

void HdMitsubaParticleField::Finalize(HdRenderParam* renderParam) {
  auto* mitsuba_render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  auto* scene = mitsuba_render_param->GetScene();
  RemoveFromScene(scene);
}

void HdMitsubaParticleField::RemoveFromScene(SceneManager* scene) {
  scene->RemoveShape(GetId());
  in_scene_ = false;
}

HdDirtyBits HdMitsubaParticleField::_PropagateDirtyBits(
    HdDirtyBits bits) const {
  return bits;
}

void HdMitsubaParticleField::_InitRepr(const TfToken& /*reprToken*/,
                                       HdDirtyBits* /*dirtyBits*/) {}

PXR_NAMESPACE_CLOSE_SCOPE
