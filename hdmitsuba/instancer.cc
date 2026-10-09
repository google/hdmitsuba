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

#include "hdmitsuba/instancer.h"

#include <vector>

#include <absl/synchronization/mutex.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/instancerTopologySchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaInstancer::HdMitsubaInstancer(HdSceneDelegate* delegate,
                                       const SdfPath& id)
    : HdInstancer(delegate, id) {}

void HdMitsubaInstancer::Sync(HdSceneDelegate* scene_delegate,
                              HdRenderParam* /* render_param */,
                              HdDirtyBits* dirty_bits) {
  _UpdateInstancer(scene_delegate, dirty_bits);

  if (HdChangeTracker::IsAnyPrimvarDirty(*dirty_bits, GetId()) ||
      HdChangeTracker::IsTransformDirty(*dirty_bits, GetId())) {
    absl::MutexLock lock(cache_mutex_);
    cached_transforms_.clear();
  }
}

VtMatrix4dArray HdMitsubaInstancer::ComputeInstanceTransforms(
    const SdfPath& prototype_id) {
  static const HdDataSourceLocator transforms_locator(
      HdInstancerTokens->instanceTransforms,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator translations_locator(
      HdInstancerTokens->instanceTranslations,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator scales_locator(
      HdInstancerTokens->instanceScales, HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator rotations_locator(
      HdInstancerTokens->instanceRotations,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator transform_locator(
      HdXformSchema::GetSchemaToken(), HdXformSchemaTokens->matrix);

  {
    absl::MutexLock lock(cache_mutex_);
    auto it = cached_transforms_.find(prototype_id);
    if (it != cached_transforms_.end()) {
      return it->second;
    }
  }

  const SdfPath& instancer_id = GetId();
  HdRenderIndex& render_index = GetDelegate()->GetRenderIndex();
  HdSceneIndexBaseRefPtr scene_index = render_index.GetTerminalSceneIndex();
  if (!TF_VERIFY(scene_index)) {
    return {};
  }
  HdSceneIndexPrim prim = scene_index->GetPrim(instancer_id);
  HdInstancerTopologySchema topology_schema =
      HdInstancerTopologySchema::GetFromParent(prim.dataSource);
  VtIntArray instance_indices;
  if (topology_schema.IsDefined()) {
    instance_indices =
        topology_schema.ComputeInstanceIndicesForProto(prototype_id);
  }

  VtMatrix4dArray transforms;
  if (instance_indices.empty()) {
    absl::MutexLock lock(cache_mutex_);
    cached_transforms_[prototype_id] = transforms;
    return transforms;
  }

  // Fetch various primvars outside of the loop over instances.
  VtMatrix4dArray instancer_transforms;
  VtVec3fArray instancer_translations;
  VtVec3fArray instancer_scales;
  VtVec4fArray instancer_rotations;
  VtQuathArray instancer_rotations_q;
  VtQuatfArray instancer_rotations_qf;

  HdPrimvarsSchema primvars_schema =
      HdPrimvarsSchema::GetFromParent(prim.dataSource);
  if (primvars_schema.IsDefined()) {
    HdContainerDataSourceHandle primvars_container =
        primvars_schema.GetContainer();

    instancer_transforms =
        GetParam(primvars_container, transforms_locator, instancer_transforms);
    instancer_translations = GetParam(primvars_container, translations_locator,
                                      instancer_translations);
    instancer_scales =
        GetParam(primvars_container, scales_locator, instancer_scales);

    // Rotations can have different types, try them one by one.
    if (auto data_source = HdSampledDataSource::Cast(HdContainerDataSource::Get(
            primvars_container, rotations_locator))) {
      VtValue value = data_source->GetValue(0.0f);
      if (value.IsHolding<VtVec4fArray>()) {
        instancer_rotations = value.UncheckedGet<VtVec4fArray>();
      } else if (value.IsHolding<VtQuathArray>()) {
        instancer_rotations_q = value.UncheckedGet<VtQuathArray>();
      } else if (value.IsHolding<VtQuatfArray>()) {
        instancer_rotations_qf = value.UncheckedGet<VtQuatfArray>();
      } else if (!value.IsEmpty()) {
        TF_WARN("Unexpected type for instanceRotations: %s",
                value.GetTypeName().c_str());
      }
    }
  }

  GfMatrix4d instancer_transform =
      GetParam<GfMatrix4d>(prim.dataSource, transform_locator, GfMatrix4d(1.0));

  const size_t num_scales = instancer_scales.size();
  const size_t num_rotations = instancer_rotations.size();
  const size_t num_rotations_q = instancer_rotations_q.size();
  const size_t num_rotations_qf = instancer_rotations_qf.size();
  const size_t num_translations = instancer_translations.size();
  const size_t num_xforms = instancer_transforms.size();

  const bool has_instancer_xform = (instancer_transform != GfMatrix4d(1.0));
  const size_t num_instances = instance_indices.size();
  transforms.resize(num_instances);
  GfMatrix4d* out_ptr = transforms.data();

  for (size_t i = 0; i < num_instances; ++i) {
    const int index = instance_indices.AsConst()[i];
    if (index < 0) {
      out_ptr[i] = instancer_transform;
      continue;
    }

    const size_t uindex = static_cast<size_t>(index);
    GfMatrix4d transform(1.0);
    if (num_rotations > 0) {
      const GfVec4f& r =
          instancer_rotations.AsConst()[std::min(uindex, num_rotations - 1)];
      transform.SetRotate(GfQuatd(r[0], r[1], r[2], r[3]));
    } else if (num_rotations_q > 0) {
      transform.SetRotate(GfQuatd(instancer_rotations_q.AsConst()[std::min(
          uindex, num_rotations_q - 1)]));
    } else if (num_rotations_qf > 0) {
      transform.SetRotate(GfQuatd(instancer_rotations_qf.AsConst()[std::min(
          uindex, num_rotations_qf - 1)]));
    }
    if (num_scales > 0) {
      const GfVec3f& s =
          instancer_scales.AsConst()[std::min(uindex, num_scales - 1)];
      for (int row = 0; row < 3; ++row) {
        transform[row][0] *= s[row];
        transform[row][1] *= s[row];
        transform[row][2] *= s[row];
      }
    }
    if (num_translations > 0) {
      transform.SetTranslateOnly(
          GfVec3d(instancer_translations
                      .AsConst()[std::min(uindex, num_translations - 1)]));
    }
    if (num_xforms > 0) {
      transform *=
          instancer_transforms.AsConst()[std::min(uindex, num_xforms - 1)];
    }
    out_ptr[i] =
        has_instancer_xform ? (transform * instancer_transform) : transform;
  }

  // Flatten nested instancing transforms.
  const SdfPath parent_instancer_id = GetParentId();
  if (!parent_instancer_id.IsEmpty()) {
    if (HdMitsubaInstancer* parent_instancer = static_cast<HdMitsubaInstancer*>(
            render_index.GetInstancer(parent_instancer_id))) {
      const VtMatrix4dArray parent_transforms =
          parent_instancer->ComputeInstanceTransforms(GetId());
      if (!parent_transforms.empty()) {
        const size_t num_parent = parent_transforms.size();
        VtMatrix4dArray new_transforms(num_instances * num_parent);
        GfMatrix4d* dst_ptr = new_transforms.data();
        for (size_t p = 0; p < num_parent; ++p) {
          const GfMatrix4d& pt = parent_transforms[p];
          for (size_t c = 0; c < num_instances; ++c) {
            dst_ptr[p * num_instances + c] = transforms.AsConst()[c] * pt;
          }
        }
        transforms = std::move(new_transforms);
      }
    }
  }
  {
    absl::MutexLock lock(cache_mutex_);
    cached_transforms_[prototype_id] = transforms;
  }
  return transforms;
}

PXR_NAMESPACE_CLOSE_SCOPE
