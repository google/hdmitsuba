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

#include "hdmitsuba/material.h"

#include <utility>

#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaMaterial::HdMitsubaMaterial(const SdfPath& id) : HdMaterial(id) {}

void HdMitsubaMaterial::Sync(HdSceneDelegate* scene_delegate,
                             HdRenderParam* render_param,
                             HdDirtyBits* dirty_bits) {
  const SdfPath& id = GetId();
  TF_DEBUG(HDMITSUBA_SYNC).Msg("HdMitsubaMaterial::Sync: %s\n", id.GetText());
  if ((*dirty_bits & DirtyBits::DirtyParams) ||
      (*dirty_bits & DirtyBits::DirtyResource)) {
    auto* mitsuba_render_param =
        static_cast<HdMitsubaRenderParam*>(render_param);
    VtValue material_resource = scene_delegate->GetMaterialResource(id);
    if (material_resource.IsHolding<HdMaterialNetworkMap>()) {
      MaterialSpec spec;
      spec.id = id;
      spec.network2 = HdConvertToHdMaterialNetwork2(
          material_resource.Get<HdMaterialNetworkMap>(), nullptr);
      spec.needs_rebuild = true;
      mitsuba_render_param->GetScene()->SyncMaterial(std::move(spec));
    }
  }
  *dirty_bits = HdChangeTracker::Clean;
}

HdDirtyBits HdMitsubaMaterial::GetInitialDirtyBitsMask() const {
  return DirtyBits::DirtyResource | DirtyBits::DirtyParams;
}

void HdMitsubaMaterial::Finalize(HdRenderParam* renderParam) {
  auto* mitsuba_render_param = static_cast<HdMitsubaRenderParam*>(renderParam);
  mitsuba_render_param->GetScene()->RemoveMaterial(GetId());
  HdMaterial::Finalize(renderParam);
}

PXR_NAMESPACE_CLOSE_SCOPE
