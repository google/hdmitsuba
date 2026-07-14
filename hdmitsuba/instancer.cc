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

  const bool has_scales = !instancer_scales.empty();
  const bool has_rotations = !instancer_rotations.empty();
  const bool has_rotations_q = !instancer_rotations_q.empty();
  const bool has_rotations_qf = !instancer_rotations_qf.empty();
  const bool has_translations = !instancer_translations.empty();
  const bool has_transforms = !instancer_transforms.empty();
  const size_t num_instances = instance_indices.size();
  transforms.resize(num_instances);

  for (size_t i = 0; i < num_instances; ++i) {
    const int index = instance_indices[i];
    if (index < 0) {
      transforms[i] = instancer_transform;
      continue;
    }

    const size_t uindex = static_cast<size_t>(index);
    GfMatrix4d transform(1.0);
    if (has_scales) {
      const size_t clamped_index =
          std::min(uindex, instancer_scales.size() - 1);
      transform.SetScale(GfVec3d(instancer_scales[clamped_index]));
    }
    if (has_rotations) {
      const size_t clamped_index =
          std::min(uindex, instancer_rotations.size() - 1);
      const GfVec4f& r = instancer_rotations[clamped_index];
      transform *= GfMatrix4d().SetRotate(GfQuatd(r[0], r[1], r[2], r[3]));
    } else if (has_rotations_q) {
      const size_t clamped_index =
          std::min(uindex, instancer_rotations_q.size() - 1);
      transform *=
          GfMatrix4d().SetRotate(GfQuatd(instancer_rotations_q[clamped_index]));
    } else if (has_rotations_qf) {
      const size_t clamped_index =
          std::min(uindex, instancer_rotations_qf.size() - 1);
      transform *= GfMatrix4d().SetRotate(
          GfQuatd(instancer_rotations_qf[clamped_index]));
    }
    if (has_translations) {
      const size_t clamped_index =
          std::min(uindex, instancer_translations.size() - 1);
      transform *= GfMatrix4d().SetTranslate(
          GfVec3d(instancer_translations[clamped_index]));
    }
    if (has_transforms) {
      const size_t clamped_index =
          std::min(uindex, instancer_transforms.size() - 1);
      transform *= instancer_transforms[clamped_index];
    }
    transforms[i] = transform * instancer_transform;
  }

  // Flatten nested instancing transforms.
  const SdfPath parent_instancer_id = GetParentId();
  if (!parent_instancer_id.IsEmpty()) {
    if (HdMitsubaInstancer* parent_instancer = static_cast<HdMitsubaInstancer*>(
            render_index.GetInstancer(parent_instancer_id))) {
      const VtMatrix4dArray parent_transforms =
          parent_instancer->ComputeInstanceTransforms(GetId());
      if (!parent_transforms.empty()) {
        VtMatrix4dArray new_transforms;
        new_transforms.reserve(transforms.size() * parent_transforms.size());
        for (const auto& pt : parent_transforms) {
          for (const auto& t : transforms) {
            new_transforms.push_back(t * pt);
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
