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

#include "hdmitsuba/render_settings.h"

#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderSettings.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/render_param.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaRenderSettings::HdMitsubaRenderSettings(SdfPath const& id)
    : HdRenderSettings(id) {}

void HdMitsubaRenderSettings::_Sync(HdSceneDelegate* /*sceneDelegate*/,
                                    HdRenderParam* renderParam,
                                    const HdDirtyBits* dirtyBits) {
  TF_DEBUG(HDMITSUBA_SYNC).Msg("HdMitsubaRenderSettings::_Sync\n");
  if (*dirtyBits & HdRenderSettings::DirtyNamespacedSettings) {
    static_cast<HdMitsubaRenderParam*>(renderParam)
        ->GetScene()
        ->UpdateNamespacedSettings(GetNamespacedSettings());
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
