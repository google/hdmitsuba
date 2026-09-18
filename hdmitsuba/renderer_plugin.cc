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
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/dirtyBitsTranslator.h>
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

TF_DEFINE_PRIVATE_TOKENS(
    HdMitsubaPluginTokens,
    ((sceneIndexPluginName, "HdMitsuba_ImplicitSurfaceSceneIndexPlugin"))
    // Keep in sync with `displayName` in plugInfo.json.in.
    ((rendererDisplayName, "Mitsuba")));

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

// UsdImaging adapter covering the gaps between what hdMitsuba reads off a prim
// and what the stock usdImaging adapters expose / invalidate.
//
// Registered *keyless* (empty "apiSchemaName" in plugInfo.json), which means
// UsdImaging_AdapterManager applies it to every prim of every UsdImaging scene
// index in the process, ahead of the built-in adapters. That is unavoidable for
// the two invalidation rules below, which must observe prims this renderer does
// not own; keep the per-prim work here as cheap as possible.
//
// Each rule is a workaround for a specific upstream gap and should be deleted
// if/when USD closes it. Rules and their regression tests:
//   1. `mitsuba:sensor`  - custom attribute, no stock adapter surfaces it.
//                          Covered by camera_test.py::test_irradiancemeter_render.
//   2. treatAsPoint/Line - UsdLux adapters do not invalidate these non-`inputs:`
//                          attributes. Covered by
//                          light_test.py::test_update_treat_as_point.
//   3. `outputs:*`       - material terminal (re)connections do not invalidate
//                          the material network. Covered by
//                          fuzzing/test_179_nodegraph_disconnected_shading.
class HdMitsuba_APISchemaAdapter : public UsdImagingAPISchemaAdapter {
 public:
  HdContainerDataSourceHandle GetImagingSubprimData(
      const UsdPrim& prim, const TfToken& subprim,
      const TfToken& applied_instance_name,
      const UsdImagingDataSourceStageGlobals& stage_globals) override {
    if (!subprim.IsEmpty() || !applied_instance_name.IsEmpty()) {
      return nullptr;
    }
    // Rule 1.
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
    // Resolve the prim type once rather than per changed property: this runs
    // for every prim on the stage on every property change.
    const bool is_light = prim.HasAPI<UsdLuxLightAPI>();
    const bool is_material = prim.IsA<UsdShadeMaterial>();

    static const HdDataSourceLocator kSensorLocator(
        HdMitsubaMeshTokens->sensor);
    HdDataSourceLocatorSet result;
    for (const TfToken& property_name : properties) {
      // Rule 1.
      if (property_name == HdMitsubaMeshTokens->sensor) {
        result.insert(kSensorLocator);
      }
      // Rule 2.
      if (is_light && (property_name == UsdLuxTokens->treatAsPoint ||
                       property_name == UsdLuxTokens->treatAsLine)) {
        result.insert(HdLightSchema::GetDefaultLocator());
      }
      // Rule 3.
      if (is_material &&
          TfStringStartsWith(property_name.GetString(), "outputs:")) {
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
  TfType t = TfType::Define<HdMitsuba_APISchemaAdapter,
                            TfType::Bases<UsdImagingAPISchemaAdapter>>();
  t.SetFactory<UsdImagingAPISchemaAdapterFactory<HdMitsuba_APISchemaAdapter>>();

  // Teach Hydra how to translate the data source locators this renderer cares
  // about into rprim dirty bits. Two things to know before editing:
  //  - This registers against the *built-in* `mesh` prim type, so it applies to
  //    every renderer's meshes in the process. It only ever adds a bit.
  //  - For rprims the custom function is *additive* (called after the standard
  //    translation), unlike RegisterTranslatorsForCustomSprimType, where it
  //    replaces it. Do not assume the sprim semantics here.
  HdDirtyBitsTranslator::RegisterTranslatorsForCustomRprimType(
      HdPrimTypeTokens->mesh,
      [](const HdDataSourceLocatorSet& set, HdDirtyBits* bits) {
        static const HdDataSourceLocator kSensorLocator(
            HdMitsubaMeshTokens->sensor);
        if (set.Intersects(HdLightSchema::GetDefaultLocator()) ||
            set.Intersects(kSensorLocator)) {
          *bits |= HdChangeTracker::DirtyParams;
        }
      },
      // Bits -> locators is unused: nothing marks these bits through the
      // legacy change tracker path.
      [](const HdDirtyBits /*bits*/, HdDataSourceLocatorSet* /*set*/) {});
}

TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
  constexpr HdSceneIndexPluginRegistry::InsertionPhase kInsertionPhase = 0;
  // Must match the `displayName` of HdMitsubaRendererPlugin in plugInfo.json:
  // HdRenderIndex looks plugins up by the render delegate's display name, and a
  // mismatch silently disables the whole chain.
  HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
      HdMitsubaPluginTokens->rendererDisplayName.GetString(),
      HdMitsubaPluginTokens->sceneIndexPluginName,
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
