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

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/cameraSchema.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/dirtyBitsTranslator.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPlugin.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdsi/implicitSurfaceSceneIndex.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usdImaging/usdImaging/apiSchemaAdapter.h>
#include <pxr/usdImaging/usdImaging/dataSourceAttribute.h>

#include "hdmitsuba/mesh.h"

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

namespace {

// Exposes `mitsuba:subdivision_level` under `primvars` for HdSceneIndexAdapter
// lookups while returning empty GetNames() so it is not treated as a geometric
// primvar.
class HdMitsuba_MeshSubdivDataSource : public HdContainerDataSource {
 public:
  HD_DECLARE_DATASOURCE(HdMitsuba_MeshSubdivDataSource);

  TfTokenVector GetNames() override { return {}; }

  HdDataSourceBaseHandle Get(const TfToken& name) override {
    if (name == HdPrimvarsSchema::GetSchemaToken()) {
      return HdContainerDataSourceHandle(this);
    }
    if (name == HdMitsubaMeshTokens->subdivision_level) {
      UsdAttribute attr =
          prim_.GetAttribute(HdMitsubaMeshTokens->subdivision_level);
      if (attr && attr.HasValue()) {
        return HdPrimvarSchema::Builder()
            .SetPrimvarValue(UsdImagingDataSourceAttribute<int>::New(
                attr, stage_globals_, prim_.GetPath(),
                HdPrimvarsSchema::GetDefaultLocator().Append(
                    HdMitsubaMeshTokens->subdivision_level)))
            .Build();
      }
    }
    return nullptr;
  }

 private:
  HdMitsuba_MeshSubdivDataSource(
      const UsdPrim& prim,
      const UsdImagingDataSourceStageGlobals& stage_globals)
      : prim_(prim), stage_globals_(stage_globals) {}

  UsdPrim prim_;
  const UsdImagingDataSourceStageGlobals& stage_globals_;
};

}  // namespace

// Keyless UsdImaging adapter bridging data-source and property-invalidation
// gaps for custom `mitsuba:*` attributes, UsdLux shaping flags, and material
// output connections.
class HdMitsuba_APISchemaAdapter : public UsdImagingAPISchemaAdapter {
 public:
  HdContainerDataSourceHandle GetImagingSubprimData(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name,
      const UsdImagingDataSourceStageGlobals& stage_globals) override {
    if (subprim.IsEmpty() && applied_instance_name.IsEmpty() &&
        prim.IsA<UsdGeomMesh>()) {
      return HdMitsuba_MeshSubdivDataSource::New(prim, stage_globals);
    }
    return nullptr;
  }

  HdDataSourceLocatorSet InvalidateImagingSubprim(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name, const TfTokenVector& properties,
      const UsdImagingPropertyInvalidationType /*invalidation_type*/) override {
    if (!subprim.IsEmpty() || !applied_instance_name.IsEmpty()) {
      return {};
    }

    HdDataSourceLocatorSet result;
    for (const TfToken& prop : properties) {
      if (TfStringStartsWith(prop.GetString(), "mitsuba:sensor:") &&
          prim.IsA<UsdGeomCamera>()) {
        result.insert(HdCameraSchema::GetDefaultLocator());
      } else if ((prop == UsdLuxTokens->treatAsPoint ||
                  prop == UsdLuxTokens->treatAsLine) &&
                 prim.HasAPI<UsdLuxLightAPI>()) {
        result.insert(HdLightSchema::GetDefaultLocator());
      } else if (TfStringStartsWith(prop.GetString(), "outputs:") &&
                 prim.IsA<UsdShadeMaterial>()) {
        result.insert(HdMaterialSchema::GetDefaultLocator());
      } else if (prop == HdMitsubaMeshTokens->subdivision_level &&
                 prim.IsA<UsdGeomMesh>()) {
        result.insert(HdPrimvarsSchema::GetDefaultLocator().Append(
            HdMitsubaMeshTokens->subdivision_level));
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

  // Map mesh-light schema changes to DirtyParams on mesh rprims.
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
