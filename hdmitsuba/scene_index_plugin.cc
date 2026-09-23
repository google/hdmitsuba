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
// TODO: unlike the scene index above, the keyless adapter cannot be scoped to
// a renderer -- UsdImaging_AdapterManager constructs keyless adapters in its
// constructor, which loads this library (and with it Mitsuba and Dr.Jit) into
// *every* UsdImaging session, and makes every prim take the multi-adapter path
// in UsdImagingStageSceneIndex. Moving the adapter into a small plugin library
// without the Mitsuba dependency would avoid that.
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

// Note: only the helper is given internal linkage. The two classes below must
// stay at namespace scope -- `TfType::Define` registers them under their
// demangled type name, which has to match the keys in plugInfo.json.
namespace {

bool StartsWith(std::string_view s, std::string_view prefix) {
  return s.substr(0, prefix.size()) == prefix;
}

}  // namespace

// Keyless UsdImaging adapter filling property-invalidation gaps for custom
// `mitsuba:sensor:*` attributes, the UsdLuxSphereLight/UsdLuxCylinderLight
// `treatAsPoint`/`treatAsLine` attributes, and material output connections.
//
// Being keyless means this runs for every prim of every stage in the process,
// including sessions driving another renderer -- see the TODO in the file
// header.
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
      // The name test comes first in each branch: it is a cheap compare, and
      // it keeps the schema lookups off the path taken by the vast majority of
      // prims, which match nothing here.
      const std::string_view prop_name = prop.GetString();
      if (StartsWith(prop_name, kMitsubaSensorNamespace) &&
          prim.IsA<UsdGeomCamera>()) {
        // UsdImagingDataSourceCameraPrim::Invalidate only maps names returned
        // by UsdGeomCamera::GetSchemaAttributeNames().
        result.insert(HdCameraSchema::GetDefaultLocator());
      } else if ((prop == UsdLuxTokens->treatAsPoint ||
                  prop == UsdLuxTokens->treatAsLine) &&
                 prim.HasAPI<UsdLuxLightAPI>()) {
        // UsdImagingLightAPIAdapter only claims `inputs:*` and `light:*`.
        result.insert(HdLightSchema::GetDefaultLocator());
      } else if (StartsWith(prop_name, UsdShadeTokens->outputs.GetString()) &&
                 prim.IsA<UsdShadeMaterial>()) {
        // Narrow gap: UsdImagingMaterialAdapter already claims edits to any
        // output it can still see via UsdShadeMaterial::GetOutputs(). What it
        // cannot see is an output that was just *removed*, which is the case
        // this branch exists for.
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
          *bits |= HdMitsubaMesh::DirtyLight;
        }
      },
      // The reverse direction needs nothing: the built-in translator already
      // produces the locators for every bit Hydra itself sets on a mesh.
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
