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

#include "hdmitsuba/implicit_surface.h"

#include <utility>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/trace/trace.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/cylinderSchema.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/sphereSchema.h>
#include <pxr/imaging/hd/tokens.h>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/instancer.h"
#include "hdmitsuba/motion.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

float GetValue(const HdDoubleDataSourceHandle& source, float default_value) {
  return source ? static_cast<float>(source->GetTypedValue(0.0f))
                : default_value;
}

}  // namespace

HdMitsubaImplicitSurface::HdMitsubaImplicitSurface(const SdfPath& id,
                                                   const TfToken& type)
    : HdRprim(id), type_(type) {}

HdDirtyBits HdMitsubaImplicitSurface::GetInitialDirtyBitsMask() const {
  return HdChangeTracker::AllDirty;
}

const TfTokenVector& HdMitsubaImplicitSurface::GetBuiltinPrimvarNames() const {
  static const TfTokenVector kEmpty;
  return kEmpty;
}

HdDirtyBits HdMitsubaImplicitSurface::_PropagateDirtyBits(
    HdDirtyBits bits) const {
  // HdsiImplicitSurfaceSceneIndex folds `cylinder/axis` into `xform`, whereas
  // HdDirtyBitsTranslator maps the cylinder schema to DirtyPrimvar.
  if (type_ == HdPrimTypeTokens->cylinder &&
      (bits & HdChangeTracker::DirtyPrimvar)) {
    bits |= HdChangeTracker::DirtyTransform;
  }
  return bits;
}

void HdMitsubaImplicitSurface::_InitRepr(const TfToken& /*reprToken*/,
                                         HdDirtyBits* /*dirtyBits*/) {}

void HdMitsubaImplicitSurface::Sync(HdSceneDelegate* sceneDelegate,
                                    HdRenderParam* renderParam,
                                    HdDirtyBits* dirtyBits,
                                    const TfToken& /*reprToken*/) {
  TRACE_FUNCTION();
  if (*dirtyBits == HdChangeTracker::Clean) {
    return;
  }
  _UpdateInstancer(sceneDelegate, dirtyBits);
  HdInstancer::_SyncInstancerAndParents(sceneDelegate->GetRenderIndex(),
                                        GetInstancerId());
  const SdfPath& id = GetId();
  TF_DEBUG(HDMITSUBA_SYNC)
      .Msg("HdMitsubaImplicitSurface::Sync: %s\n", id.GetText());

  SceneManager* scene =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();
  if (!sceneDelegate->GetVisible(id)) {
    RemoveFromScene(scene);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  const HdContainerDataSourceHandle prim_source =
      GetPrimDataSource(sceneDelegate, id);
  ImplicitSurface surface{type_};
  if (type_ == HdPrimTypeTokens->sphere) {
    surface.radius = GetValue(
        HdSphereSchema::GetFromParent(prim_source).GetRadius(), 1.0f);
  } else {
    const HdCylinderSchema cylinder =
        HdCylinderSchema::GetFromParent(prim_source);
    surface.height = GetValue(cylinder.GetHeight(), 2.0f);
    // UsdGeomCylinder's `radius` is deprecated in favor of the radii of
    // UsdGeomCylinder_1, which UsdImaging also populates for the former.
    surface.radius =
        GetValue(cylinder.GetRadiusBottom(), GetValue(cylinder.GetRadius(), 1.0f));
    if (GetValue(cylinder.GetRadiusTop(), surface.radius) != surface.radius &&
        !std::exchange(warned_tapered_, true)) {
      TF_WARN(
          "Cylinder %s has differing top and bottom radii, which is not "
          "supported. Rendering it with its bottom radius. Set "
          "HDMITSUBA_TESSELLATE_IMPLICIT_SURFACES=1 to tessellate it instead.",
          id.GetText());
    }
  }

  MeshSpec spec;
  spec.id = id;
  spec.implicit_surface = surface;
  spec.material_ids = {sceneDelegate->GetMaterialId(id)};
  spec.double_sided = sceneDelegate->GetDoubleSided(id);
  for (const HdPrimvarDescriptor& descriptor :
       GetPrimvarDescriptors(sceneDelegate, HdInterpolationConstant)) {
    spec.primvars[descriptor.name] = {GetPrimvar(sceneDelegate, descriptor.name),
                                      descriptor};
  }

  const SdfPath& instancer_id = GetInstancerId();
  if (!instancer_id.IsEmpty()) {
    // Build the prototype in object space, and fold its transform into the
    // instance transforms instead: these may contain non-uniform scaling,
    // which only instances support.
    auto* instancer = static_cast<HdMitsubaInstancer*>(
        sceneDelegate->GetRenderIndex().GetInstancer(instancer_id));
    const ScalarAffineTransform4f prototype_transform =
        UsdToMitsubaTransform(sceneDelegate->GetTransform(id));
    for (const MotionTransform& transform :
         instancer->ComputeInstanceTransforms(id)) {
      spec.instance_transforms.push_back(
          transform.Map([&](const ScalarAffineTransform4f& m) {
            return m * prototype_transform;
          }));
    }
    spec.transform = GfMatrix4d(1.0);
    if (*dirtyBits & HdChangeTracker::DirtyTransform) {
      *dirtyBits = (*dirtyBits & ~HdChangeTracker::DirtyTransform) |
                   HdChangeTracker::DirtyInstancer;
    }
  } else {
    MotionTransform transform =
        SampleTransform(prim_source, GetMotionInterval(sceneDelegate),
                        /*uniform=*/true)
            .Map([](const GfMatrix4d& m) { return UsdToMitsubaTransform(m); });
    spec.transform = sceneDelegate->GetTransform(id);
    // Animated transforms, as well as transforms that the native shape does
    // not support, require wrapping the shape in a Mitsuba instance.
    if (transform.IsAnimated() || !surface.SupportsTransform(transform.First())) {
      spec.motion_wrapped = true;
      spec.instance_transforms.push_back(std::move(transform));
    }
  }
  spec.dirty_bits = *dirtyBits;

  scene->SyncMesh(std::move(spec));
  in_scene_ = true;
  *dirtyBits = HdChangeTracker::Clean;
}

void HdMitsubaImplicitSurface::Finalize(HdRenderParam* renderParam) {
  RemoveFromScene(static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene());
}

void HdMitsubaImplicitSurface::RemoveFromScene(SceneManager* scene) {
  if (std::exchange(in_scene_, false)) {
    scene->RemoveShape(GetId());
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
