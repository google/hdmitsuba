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
// Nothing here contributes data sources -- every value hdMitsuba reads is a
// schema property or a primvar UsdImaging already surfaces. This file only
// fixes *invalidation*: UsdImagingStageSceneIndex drops a property change
// unless some adapter claims it (_ComputeDirtiedEntries has no resync
// fallback), and a few properties hdMitsuba cares about are claimed by nobody.
//
// Contents: a scene index converting implicit surfaces to meshes, and a
// keyless API-schema adapter covering those unclaimed properties.
//
// TODO: keyless adapters cannot be scoped to a renderer. Constructing them
// loads this library -- and with it Mitsuba and Dr.Jit -- into every
// UsdImaging session, and pushes every prim onto the multi-adapter path in
// UsdImagingStageSceneIndex. A separate plugin library without the Mitsuba
// dependency would avoid both.

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
#include "hdmitsuba/mesh.h"

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

// Helpers only: the registered classes must stay at namespace scope so their
// demangled type names match the plugInfo.json keys.
namespace {

bool StartsWith(std::string_view s, std::string_view prefix) {
  return s.substr(0, prefix.size()) == prefix;
}

}  // namespace

// Covers invalidation gaps for `mitsuba:sensor:*`, treatAsPoint/treatAsLine,
// and material outputs. Being keyless, this runs for every prim of every stage
// in the process -- see the TODO at the top of the file.
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
      // Name test first: cheap, and keeps the schema lookups off the path
      // taken by the vast majority of prims, which match nothing here.
      const std::string_view prop_name = prop.GetString();
      if (StartsWith(prop_name, kMitsubaSensorNamespace) &&
          prim.IsA<UsdGeomCamera>()) {
        // The camera adapter only maps UsdGeomCamera schema attributes.
        result.insert(HdCameraSchema::GetDefaultLocator());
      } else if ((prop == UsdLuxTokens->treatAsPoint ||
                  prop == UsdLuxTokens->treatAsLine) &&
                 prim.HasAPI<UsdLuxLightAPI>()) {
        // The LightAPI adapter only claims `inputs:*` and `light:*`.
        result.insert(HdLightSchema::GetDefaultLocator());
      } else if (StartsWith(prop_name, UsdShadeTokens->outputs.GetString()) &&
                 prim.IsA<UsdShadeMaterial>()) {
        // The material adapter claims any output it can still see via
        // GetOutputs(); this only adds the case of one being *removed*.
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

  // Mesh lights are `mesh` rprims carrying a `light` data source, which the
  // built-in rprim translator has no mapping for. Custom translators run in
  // addition to the built-ins, so only the forward direction needs filling in.
  HdDirtyBitsTranslator::RegisterTranslatorsForCustomRprimType(
      HdPrimTypeTokens->mesh,
      [](const HdDataSourceLocatorSet& set, HdDirtyBits* bits) {
        if (set.Intersects(HdLightSchema::GetDefaultLocator())) {
          *bits |= HdMitsubaMesh::DirtyLight;
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
