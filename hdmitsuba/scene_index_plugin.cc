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
// Contents:
// - A scene index converting implicit surfaces to meshes.
// - A scene index publishing the motion interval (the union of all camera
//   shutter intervals) on the absolute root prim, and invalidating light
//   transforms when it changes. This is the only data source contributed here.
// - A keyless API-schema adapter fixing *invalidation*: every other value
//   hdMitsuba reads is a schema property or a primvar UsdImaging already
//   surfaces, but UsdImagingStageSceneIndex drops a property change unless
//   some adapter claims it (_ComputeDirtiedEntries has no resync fallback), and
//   a few properties hdMitsuba cares about are claimed by nobody.
//
// TODO: keyless adapters cannot be scoped to a renderer. Constructing them
// loads this library -- and with it Mitsuba and Dr.Jit -- into every
// UsdImaging session, and pushes every prim onto the multi-adapter path in
// UsdImagingStageSceneIndex. A separate plugin library without the Mitsuba
// dependency would avoid both.

#include <algorithm>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/cameraSchema.h>
#include <pxr/imaging/hd/dirtyBitsTranslator.h>
#include <pxr/imaging/hd/filteringSceneIndex.h>
#include <pxr/imaging/hd/lightSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/overlayContainerDataSource.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPlugin.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/sceneIndexPrimView.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/xformSchema.h>
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
#include "hdmitsuba/motion.h"

#if PXR_VERSION < 2605
#error "hdmitsuba requires OpenUSD 26.05 or newer."
#endif

PXR_NAMESPACE_OPEN_SCOPE

// Publishes the motion interval, i.e., the union of the shutter intervals of
// all cameras, on the absolute root prim (see motion.h), and dirties shape,
// instancer, and light transforms whenever it changes. This never happens for
// scenes without motion blur, where all shutter intervals are empty.
class HdMitsuba_MotionIntervalSceneIndex final
    : public HdSingleInputFilteringSceneIndexBase {
 public:
  static HdSceneIndexBaseRefPtr New(const HdSceneIndexBaseRefPtr& input_scene) {
    return TfCreateRefPtr(new HdMitsuba_MotionIntervalSceneIndex(input_scene));
  }

  HdSceneIndexPrim GetPrim(const SdfPath& prim_path) const override {
    HdSceneIndexPrim prim = _GetInputSceneIndex()->GetPrim(prim_path);
    if (prim_path.IsAbsoluteRootPath()) {
      prim.dataSource =
          HdOverlayContainerDataSource::OverlayedContainerDataSources(
              HdRetainedContainerDataSource::New(
                  HdMitsubaMotionTokens->motion_interval,
                  HdRetainedTypedSampledDataSource<GfVec2f>::New(
                      motion_interval_)),
              prim.dataSource);
    }
    return prim;
  }

  SdfPathVector GetChildPrimPaths(const SdfPath& prim_path) const override {
    return _GetInputSceneIndex()->GetChildPrimPaths(prim_path);
  }

 protected:
  explicit HdMitsuba_MotionIntervalSceneIndex(
      const HdSceneIndexBaseRefPtr& input_scene)
      : HdSingleInputFilteringSceneIndexBase(input_scene) {
    for (const SdfPath& path : HdSceneIndexPrimView(input_scene)) {
      TrackPrim(path, input_scene->GetPrim(path).primType);
    }
    UpdateMotionInterval(/*notify=*/false);
  }

  void _PrimsAdded(
      const HdSceneIndexBase& /*sender*/,
      const HdSceneIndexObserver::AddedPrimEntries& entries) override {
    for (const auto& entry : entries) {
      TrackPrim(entry.primPath, entry.primType);
    }
    _SendPrimsAdded(entries);
    UpdateMotionInterval();
  }

  void _PrimsRemoved(
      const HdSceneIndexBase& /*sender*/,
      const HdSceneIndexObserver::RemovedPrimEntries& entries) override {
    for (const auto& entry : entries) {
      absl::erase_if(shutters_, [&](const auto& kv) {
        return kv.first.HasPrefix(entry.primPath);
      });
    }
    _SendPrimsRemoved(entries);
    UpdateMotionInterval();
  }

  void _PrimsDirtied(
      const HdSceneIndexBase& /*sender*/,
      const HdSceneIndexObserver::DirtiedPrimEntries& entries) override {
    static const HdDataSourceLocatorSet shutter_locators{
        HdCameraSchema::GetShutterOpenLocator(),
        HdCameraSchema::GetShutterCloseLocator()};
    for (const auto& entry : entries) {
      if (auto it = shutters_.find(entry.primPath);
          it != shutters_.end() &&
          entry.dirtyLocators.Intersects(shutter_locators)) {
        it->second = GetShutter(entry.primPath);
      }
    }
    _SendPrimsDirtied(entries);
    UpdateMotionInterval();
  }

 private:
  void TrackPrim(const SdfPath& path, const TfToken& prim_type) {
    if (prim_type == HdPrimTypeTokens->camera) {
      shutters_[path] = GetShutter(path);
    } else {
      shutters_.erase(path);
    }
  }

  GfVec2f GetShutter(const SdfPath& camera_path) const {
    HdCameraSchema camera = HdCameraSchema::GetFromParent(
        _GetInputSceneIndex()->GetPrim(camera_path).dataSource);
    HdDoubleDataSourceHandle open = camera.GetShutterOpen();
    HdDoubleDataSourceHandle close = camera.GetShutterClose();
    return GfVec2f(open ? open->GetTypedValue(0.0f) : 0.0,
                   close ? close->GetTypedValue(0.0f) : 0.0);
  }

  void UpdateMotionInterval(bool notify = true) {
    GfVec2f interval(0.0f);
    for (const auto& [_, s] : shutters_) {
      if (s[1] > s[0]) {
        interval = interval[1] > interval[0]
                       ? GfVec2f(std::min(interval[0], s[0]),
                                 std::max(interval[1], s[1]))
                       : s;
      }
    }
    if (interval == motion_interval_) return;
    motion_interval_ = interval;
    if (!notify) return;

    HdSceneIndexObserver::DirtiedPrimEntries dirtied = {
        {SdfPath::AbsoluteRootPath(),
        HdDataSourceLocator(HdMitsubaMotionTokens->motion_interval)}};
    const HdSceneIndexBaseRefPtr& input = _GetInputSceneIndex();
    for (const SdfPath& path : HdSceneIndexPrimView(input)) {
      const TfToken& type = input->GetPrim(path).primType;
      if (HdPrimTypeIsLight(type) || HdPrimTypeIsGprim(type) ||
          type == HdPrimTypeTokens->instancer) {
        dirtied.emplace_back(path, HdXformSchema::GetDefaultLocator());
      }
    }
    _SendPrimsDirtied(dirtied);
  }

  absl::flat_hash_map<SdfPath, GfVec2f, SdfPath::Hash> shutters_;
  GfVec2f motion_interval_{0.0f};
};

// Converts implicit surfaces to meshes and publishes the motion interval.
class HdMitsuba_SceneIndexPlugin : public HdSceneIndexPlugin {
 protected:
  HdSceneIndexBaseRefPtr _AppendSceneIndex(
      const HdSceneIndexBaseRefPtr& input_scene,
      const HdContainerDataSourceHandle& /*input_args*/) override {
    const HdDataSourceBaseHandle to_mesh =
        HdRetainedTypedSampledDataSource<TfToken>::New(
            HdsiImplicitSurfaceSceneIndexTokens->toMesh);
    return HdMitsuba_MotionIntervalSceneIndex::New(
        HdsiImplicitSurfaceSceneIndex::New(
            input_scene,
            HdRetainedContainerDataSource::New(
                HdPrimTypeTokens->sphere, to_mesh, HdPrimTypeTokens->cube,
                to_mesh, HdPrimTypeTokens->cone, to_mesh,
                HdPrimTypeTokens->cylinder, to_mesh, HdPrimTypeTokens->capsule,
                to_mesh, HdPrimTypeTokens->plane, to_mesh)));
  }
};

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
      if (absl::StartsWith(prop.GetString(), kMitsubaSensorNamespace) &&
          prim.IsA<UsdGeomCamera>()) {
        // The camera adapter only maps UsdGeomCamera schema attributes.
        result.insert(HdCameraSchema::GetDefaultLocator());
      } else if ((prop == UsdLuxTokens->treatAsPoint ||
                  prop == UsdLuxTokens->treatAsLine) &&
                 prim.HasAPI<UsdLuxLightAPI>()) {
        // The LightAPI adapter only claims `inputs:*` and `light:*`.
        result.insert(HdLightSchema::GetDefaultLocator());
      } else if (absl::StartsWith(prop.GetString(),
                                  UsdShadeTokens->outputs.GetString()) &&
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
  HdSceneIndexPluginRegistry::Define<HdMitsuba_SceneIndexPlugin>();
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
      "Mitsuba", TfToken("HdMitsuba_SceneIndexPlugin"),
      /* inputArgs = */ nullptr, /* insertionPhase = */ 0,
      HdSceneIndexPluginRegistry::InsertionOrderAtStart);
}

PXR_NAMESPACE_CLOSE_SCOPE
