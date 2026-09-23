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


// Scene-index plugins for HdMitsuba.
//
// Every value hdMitsuba reads off a prim is either a schema property or a USD
// primvar, both of which UsdImaging surfaces natively -- renderer-specific
// per-mesh controls are authored as `primvars:mitsuba:*`. Nothing here
// contributes data sources; this file only fixes *invalidation*.
//
// UsdImagingStageSceneIndex forwards a property change only if some adapter
// returns a non-empty locator set for it: _ComputeDirtiedEntries has no resync
// fallback, so an unclaimed edit is silently dropped. A few properties
// hdMitsuba cares about are claimed by nobody, hence the keyless API-schema
// adapter below.
//
// This file therefore contains:
//   * a scene index that turns implicit surfaces into meshes, and
//   * a keyless UsdImaging API-schema adapter that maps the unclaimed property
//     changes onto the right data-source locators.
//

#include <string_view>

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/cameraSchema.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/dirtyBitsTranslator.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPlugin.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdsi/implicitSurfaceSceneIndex.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/tokens.h>
#include <pxr/usdImaging/usdImaging/apiSchemaAdapter.h>
#include <pxr/usdImaging/usdImaging/types.h>

#include "hdmitsuba/camera.h"

#if PXR_VERSION < 2605
#error "hdmitsuba requires OpenUSD 26.05 or newer."
#endif

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
    HdMitsubaPluginTokens,
    ((sceneIndexPluginName, "HdMitsuba_ImplicitSurfaceSceneIndexPlugin"))
    ((rendererDisplayName, "Mitsuba")));

class HdMitsuba_ImplicitSurfaceSceneIndexPlugin : public HdSceneIndexPlugin {
 protected:
  HdSceneIndexBaseRefPtr _AppendSceneIndex(
      const HdSceneIndexBaseRefPtr& input_scene,
      const HdContainerDataSourceHandle& /*input_args*/) override {
    const HdDataSourceBaseHandle to_mesh =
        HdRetainedTypedSampledDataSource<TfToken>::New(
            HdsiImplicitSurfaceSceneIndexTokens->toMesh);
    return HdsiImplicitSurfaceSceneIndex::New(
        input_scene,
        HdRetainedContainerDataSource::New(
            HdPrimTypeTokens->sphere, to_mesh, HdPrimTypeTokens->cube, to_mesh,
            HdPrimTypeTokens->cone, to_mesh, HdPrimTypeTokens->cylinder,
            to_mesh, HdPrimTypeTokens->capsule, to_mesh,
            HdPrimTypeTokens->plane, to_mesh));
  }
};

bool StartsWith(std::string_view s, std::string_view prefix) {
  return s.substr(0, prefix.size()) == prefix;
}

// Keyless UsdImaging adapter filling property-invalidation gaps for custom
// `mitsuba:sensor:*` attributes, UsdLux shaping flags, and material output
// connections.
class HdMitsuba_APISchemaAdapter : public UsdImagingAPISchemaAdapter {
 public:
  HdDataSourceLocatorSet InvalidateImagingSubprim(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name, const TfTokenVector& properties,
      const UsdImagingPropertyInvalidationType /*invalidation_type*/) override {
    if (!subprim.IsEmpty() || !applied_instance_name.IsEmpty()) {
      return {};
    }

    HdDataSourceLocatorSet result;
    for (const TfToken& prop : properties) {
      // Check prefix quickly to possibly skip immediately
      const std::string_view prop_name = prop.GetString();
      if (StartsWith(prop_name, kMitsubaSensorNamespace) &&
          prim.IsA<UsdGeomCamera>()) {
        result.insert(HdCameraSchema::GetDefaultLocator());
      } else if ((prop == UsdLuxTokens->treatAsPoint ||
                  prop == UsdLuxTokens->treatAsLine) &&
                 prim.HasAPI<UsdLuxLightAPI>()) {
        result.insert(HdLightSchema::GetDefaultLocator());
      } else if (StartsWith(prop_name, UsdShadeTokens->outputs.GetString()) &&
                 prim.IsA<UsdShadeMaterial>()) {
        result.insert(HdMaterialSchema::GetDefaultLocator());
      }
    }
    return result;
  }
};

TF_REGISTRY_FUNCTION(TfType) {
  HdSceneIndexPluginRegistry::Define<
      HdMitsuba_ImplicitSurfaceSceneIndexPlugin>();
  TfType::Define<HdMitsuba_APISchemaAdapter,
                 TfType::Bases<UsdImagingAPISchemaAdapter>>()
      .SetFactory<
          UsdImagingAPISchemaAdapterFactory<HdMitsuba_APISchemaAdapter>>();

  // Mesh lights are `mesh` rprims that carry a `light` data source. The
  // built-in rprim translator has no mapping for that locator, and custom
  // translators are consulted *in addition to* the built-in ones, so this adds
  // the missing edge without disturbing the standard mesh dirty bits.
  HdDirtyBitsTranslator::RegisterTranslatorsForCustomRprimType(
      HdPrimTypeTokens->mesh,
      [](const HdDataSourceLocatorSet& set, HdDirtyBits* bits) {
        if (set.Intersects(HdLightSchema::GetDefaultLocator())) {
          *bits |= HdChangeTracker::DirtyParams;
        }
      },
      [](const HdDirtyBits, HdDataSourceLocatorSet*) {});
}

TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
  HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
      HdMitsubaPluginTokens->rendererDisplayName.GetString(),
      HdMitsubaPluginTokens->sceneIndexPluginName,
      /* inputArgs = */ nullptr, /* insertionPhase = */ 0,
      HdSceneIndexPluginRegistry::InsertionOrderAtStart);
}

PXR_NAMESPACE_CLOSE_SCOPE
