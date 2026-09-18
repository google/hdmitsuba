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

#include "hdmitsuba/renderer_plugin.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPlugin.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdsi/implicitSurfaceSceneIndex.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usdImaging/usdImaging/apiSchemaAdapter.h>
#include <pxr/usdImaging/usdImaging/dataSourceAttribute.h>

#include "hdmitsuba/mesh.h"
#include "hdmitsuba/render_delegate.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(HdMitsubaPluginTokens,
                         ((sceneIndexPluginName,
                           "HdMitsuba_ImplicitSurfaceSceneIndexPlugin")));

class HdMitsuba_ImplicitSurfaceSceneIndexPlugin : public HdSceneIndexPlugin {
 public:
  HdMitsuba_ImplicitSurfaceSceneIndexPlugin() = default;
  ~HdMitsuba_ImplicitSurfaceSceneIndexPlugin() override = default;

 protected:
  HdSceneIndexBaseRefPtr _AppendSceneIndex(
      const HdSceneIndexBaseRefPtr& input_scene,
      const HdContainerDataSourceHandle& input_args) override {
    TF_UNUSED(input_args);
    const HdDataSourceBaseHandle to_mesh_src =
        HdRetainedTypedSampledDataSource<TfToken>::New(
            HdsiImplicitSurfaceSceneIndexTokens->toMesh);

    const HdContainerDataSourceHandle local_input_args =
        HdRetainedContainerDataSource::New(
            HdPrimTypeTokens->sphere, to_mesh_src, HdPrimTypeTokens->cube,
            to_mesh_src, HdPrimTypeTokens->cone, to_mesh_src,
            HdPrimTypeTokens->cylinder, to_mesh_src, HdPrimTypeTokens->capsule,
            to_mesh_src, HdPrimTypeTokens->plane, to_mesh_src);

    return HdsiImplicitSurfaceSceneIndex::New(input_scene, local_input_args);
  }
};

class HdMitsuba_LightAPISchemaAdapter : public UsdImagingAPISchemaAdapter {
 public:
  HdContainerDataSourceHandle GetImagingSubprimData(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name,
      const UsdImagingDataSourceStageGlobals& stage_globals) override {
    if (!subprim.IsEmpty() || !applied_instance_name.IsEmpty()) {
      return nullptr;
    }
    if (UsdAttribute sensor_attr =
            prim.GetAttribute(HdMitsubaMeshTokens->sensor)) {
      return HdRetainedContainerDataSource::New(
          HdMitsubaMeshTokens->sensor,
          UsdImagingDataSourceAttributeNew(sensor_attr, stage_globals));
    }
    return nullptr;
  }

  HdDataSourceLocatorSet InvalidateImagingSubprim(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name, const TfTokenVector& properties,
      const UsdImagingPropertyInvalidationType /*invalidation_type*/) override {
    if (!subprim.IsEmpty() || !applied_instance_name.IsEmpty()) {
      return HdDataSourceLocatorSet();
    }
    static const HdDataSourceLocator kSensorLocator(
        HdMitsubaMeshTokens->sensor);
    HdDataSourceLocatorSet result;
    for (const TfToken& property_name : properties) {
      if (property_name == HdMitsubaMeshTokens->sensor) {
        result.insert(kSensorLocator);
      }
      if ((property_name == UsdLuxTokens->treatAsPoint ||
           property_name == UsdLuxTokens->treatAsLine) &&
          prim.HasAPI<UsdLuxLightAPI>()) {
        result.insert(HdLightSchema::GetDefaultLocator());
      }
      if (TfStringStartsWith(property_name.GetString(), "outputs:") &&
          prim.IsA<UsdShadeMaterial>()) {
        result.insert(HdMaterialSchema::GetDefaultLocator());
      }
    }
    return result;
  }
};

// Register this plugin with the TfType system.
TF_REGISTRY_FUNCTION(TfType) {
  HdRendererPluginRegistry::Define<HdMitsubaRendererPlugin>();
  HdSceneIndexPluginRegistry::Define<
      HdMitsuba_ImplicitSurfaceSceneIndexPlugin>();
  TfType t = TfType::Define<HdMitsuba_LightAPISchemaAdapter,
                            TfType::Bases<UsdImagingAPISchemaAdapter>>();
  t.SetFactory<
      UsdImagingAPISchemaAdapterFactory<HdMitsuba_LightAPISchemaAdapter>>();
}

TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
  constexpr HdSceneIndexPluginRegistry::InsertionPhase kInsertionPhase = 0;
  HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
      "Mitsuba", HdMitsubaPluginTokens->sceneIndexPluginName,
      /* inputArgs = */ nullptr, kInsertionPhase,
      HdSceneIndexPluginRegistry::InsertionOrderAtStart);
}

HdRenderDelegate* HdMitsubaRendererPlugin::CreateRenderDelegate() {
  return new HdMitsubaRenderDelegate();
}

HdRenderDelegate* HdMitsubaRendererPlugin::CreateRenderDelegate(
    const HdRenderSettingsMap& settingsMap) {
  return new HdMitsubaRenderDelegate(settingsMap);
}

void HdMitsubaRendererPlugin::DeleteRenderDelegate(
    HdRenderDelegate* renderDelegate) {
  delete renderDelegate;
}

#if HD_API_VERSION < 83
bool HdMitsubaRendererPlugin::IsSupported(bool /*gpuEnabled*/) const {
  return true;
}
#else
bool HdMitsubaRendererPlugin::IsSupported(
    const HdRendererCreateArgs& /*rendererCreateArgs*/,
    std::string* /*reasonWhyNot*/) const {
  return true;
}
#endif

PXR_NAMESPACE_CLOSE_SCOPE
