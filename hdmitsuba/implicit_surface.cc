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
#include "hdmitsuba/light.h"
#include "hdmitsuba/mesh.h"
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

// Matches `HdsiImplicitSurfaceSceneIndex`'s `axisToTransform` adjustment matrix
// so object-space `mitsuba:shape:to_world` overrides compose after the spine
// axis alignment rather than in canonical Z-axis space.
GfMatrix4d CylinderAxisMatrix(const HdCylinderSchema& cylinder) {
  const TfToken axis =
      cylinder.GetAxis() ? cylinder.GetAxis()->GetTypedValue(0.0f) : TfToken();
  if (axis == HdCylinderSchemaTokens->X) {
    return GfMatrix4d(0.0, 1.0, 0.0, 0.0,  //
                      0.0, 0.0, 1.0, 0.0,  //
                      1.0, 0.0, 0.0, 0.0,  //
                      0.0, 0.0, 0.0, 1.0);
  }
  if (axis == HdCylinderSchemaTokens->Y) {
    return GfMatrix4d(0.0, 0.0, 1.0, 0.0,  //
                      1.0, 0.0, 0.0, 0.0,  //
                      0.0, 1.0, 0.0, 0.0,  //
                      0.0, 0.0, 0.0, 1.0);
  }
  return GfMatrix4d(1.0);
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
  GfMatrix4d axis_matrix(1.0);
  if (type_ == HdPrimTypeTokens->sphere) {
    surface.radius = GetValue(
        HdSphereSchema::GetFromParent(prim_source).GetRadius(), 1.0f);
  } else {
    const HdCylinderSchema cylinder =
        HdCylinderSchema::GetFromParent(prim_source);
    axis_matrix = CylinderAxisMatrix(cylinder);
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
  spec.emitter_spec = GetMeshEmitterSpec(sceneDelegate, id);
  if (!in_scene_ || (*dirtyBits & HdMitsubaMesh::DirtyMitsubaParams) != 0) {
    ScopedResolverCache resolver_cache(
        static_cast<HdMitsubaRenderParam*>(renderParam)->ResolverCacheData());
    HdContainerDataSourceHandle mitsuba_source =
        GetMitsubaDataSource(prim_source);
    PluginSpecNode parsed_shape_spec =
        ParsePluginSpec(mitsuba_source, "mitsuba:shape:");
    PluginSpecNode parsed_emitter_spec =
        ParsePluginSpec(mitsuba_source, "mitsuba:emitter:");
    if (parsed_shape_spec != shape_spec_ ||
        parsed_emitter_spec != mitsuba_emitter_spec_) {
      auto param_changed = [](const PluginSpecNode& lhs,
                              const PluginSpecNode& rhs,
                              const std::string& key) {
        auto lhs_it = lhs.params.find(key);
        auto rhs_it = rhs.params.find(key);
        if (lhs_it == lhs.params.end() && rhs_it == rhs.params.end()) {
          return false;
        }
        if (lhs_it == lhs.params.end() || rhs_it == rhs.params.end()) {
          return true;
        }
        return lhs_it->second != rhs_it->second;
      };
      if (!in_scene_ ||
          !CanUpdatePluginStructureInPlace(shape_spec_, parsed_shape_spec) ||
          !CanUpdatePluginStructureInPlace(mitsuba_emitter_spec_,
                                           parsed_emitter_spec) ||
          param_changed(shape_spec_, parsed_shape_spec, "flip_normals")) {
        spec.needs_rebuild = true;
      }
      if (param_changed(shape_spec_, parsed_shape_spec, "to_world")) {
        *dirtyBits |= HdChangeTracker::DirtyTransform;
      }
      shape_spec_ = std::move(parsed_shape_spec);
      mitsuba_emitter_spec_ = std::move(parsed_emitter_spec);
    }
  }
  if ((*dirtyBits &
       (HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyPrimvar)) !=
      0) {
    for (const char* geom_key : {"center", "radius", "p0", "p1"}) {
      if (shape_spec_.params.count(geom_key) > 0) {
        spec.needs_rebuild = true;
        break;
      }
    }
  }
  spec.shape_spec = shape_spec_;
  spec.mitsuba_emitter_spec = mitsuba_emitter_spec_;
  GfMatrix4d local_extra(1.0);
  if (auto xform_it = spec.shape_spec.params.find("to_world");
      xform_it != spec.shape_spec.params.end()) {
    if (const auto extra = ExtractMatrix4d(xform_it->second)) {
      local_extra = axis_matrix * (*extra) * axis_matrix.GetTranspose();
    }
    spec.shape_spec.params.erase(xform_it);
  }
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
        UsdToMitsubaTransform(local_extra * sceneDelegate->GetTransform(id));
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
            .Map([&](const GfMatrix4d& m) {
              return UsdToMitsubaTransform(local_extra * m);
            });
    spec.transform = local_extra * sceneDelegate->GetTransform(id);
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
    shape_spec_ = {};
    mitsuba_emitter_spec_ = {};
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
