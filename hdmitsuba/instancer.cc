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
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

template <typename T>
T GetPrimvarArray(HdSceneDelegate* delegate, const SdfPath& id,
                  const TfToken& token) {
  if (delegate) {
    VtValue value = delegate->Get(id, token);
    if (value.IsHolding<T>()) {
      return value.UncheckedGet<T>();
    }
  }
  return T();
}

}  // namespace

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
  {
    absl::MutexLock lock(cache_mutex_);
    auto it = cached_transforms_.find(prototype_id);
    if (it != cached_transforms_.end()) {
      return it->second;
    }
  }

  HdSceneDelegate* delegate = GetDelegate();
  const SdfPath& instancer_id = GetId();
  const VtIntArray instance_indices =
      delegate->GetInstanceIndices(instancer_id, prototype_id);

  VtMatrix4dArray transforms;
  if (instance_indices.empty()) {
    absl::MutexLock lock(cache_mutex_);
    cached_transforms_[prototype_id] = transforms;
    return transforms;
  }

  // Fetch various primvars outside of the loop over instances.
  auto instancer_transforms = GetPrimvarArray<VtMatrix4dArray>(
      delegate, instancer_id, HdInstancerTokens->instanceTransforms);
  auto instancer_translations = GetPrimvarArray<VtVec3fArray>(
      delegate, instancer_id, HdInstancerTokens->instanceTranslations);
  auto instancer_scales = GetPrimvarArray<VtVec3fArray>(
      delegate, instancer_id, HdInstancerTokens->instanceScales);
  VtVec4fArray instancer_rotations;
  VtQuathArray instancer_rotations_q;
  VtQuatfArray instancer_rotations_qf;
  VtValue rotations_val =
      delegate->Get(instancer_id, HdInstancerTokens->instanceRotations);
  if (rotations_val.IsHolding<VtVec4fArray>()) {
    instancer_rotations = rotations_val.UncheckedGet<VtVec4fArray>();
  } else if (rotations_val.IsHolding<VtQuathArray>()) {
    instancer_rotations_q = rotations_val.UncheckedGet<VtQuathArray>();
  } else if (rotations_val.IsHolding<VtQuatfArray>()) {
    instancer_rotations_qf = rotations_val.UncheckedGet<VtQuatfArray>();
  }

  const GfMatrix4d instancer_transform =
      delegate->GetInstancerTransform(instancer_id);

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
            delegate->GetRenderIndex().GetInstancer(parent_instancer_id))) {
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
