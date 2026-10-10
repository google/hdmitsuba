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

namespace {

const HdDataSourceLocator& TransformsLocator() {
  static const HdDataSourceLocator locator(
      HdInstancerTokens->instanceTransforms,
      HdPrimvarSchemaTokens->primvarValue);
  return locator;
}

const HdDataSourceLocator& TranslationsLocator() {
  static const HdDataSourceLocator locator(
      HdInstancerTokens->instanceTranslations,
      HdPrimvarSchemaTokens->primvarValue);
  return locator;
}

const HdDataSourceLocator& ScalesLocator() {
  static const HdDataSourceLocator locator(HdInstancerTokens->instanceScales,
                                           HdPrimvarSchemaTokens->primvarValue);
  return locator;
}

const HdDataSourceLocator& RotationsLocator() {
  static const HdDataSourceLocator locator(HdInstancerTokens->instanceRotations,
                                           HdPrimvarSchemaTokens->primvarValue);
  return locator;
}

const HdDataSourceLocator& TransformLocator() {
  static const HdDataSourceLocator locator(HdXformSchema::GetSchemaToken(),
                                           HdXformSchemaTokens->matrix);
  return locator;
}

}  // namespace

std::vector<MotionTransform> HdMitsubaInstancer::ComputeInstanceTransforms(
    const SdfPath& prototype_id) {
  {
    absl::MutexLock lock(cache_mutex_);
    auto it = cached_transforms_.find(prototype_id);
    if (it != cached_transforms_.end()) {
      return it->second;
    }
  }

  VtMatrix4dArray t0_transforms =
      ComputeInstanceTransformsAtTime(prototype_id, 0.0f);
  std::vector<MotionTransform> result;
  result.reserve(t0_transforms.size());
  if (!t0_transforms.empty()) {
    const GfVec2f interval = GetMotionInterval(GetDelegate());
    std::vector<float> sample_times;
    if (interval[1] <= interval[0] ||
        !GetContributingSampleTimesForInterval(interval, &sample_times)) {
      for (const GfMatrix4d& xform : t0_transforms) {
        result.push_back(
            MotionTransform::Static(UsdToMitsubaTransform(xform)));
      }
    } else {
      absl::flat_hash_map<float, VtMatrix4dArray> transforms_at_time;
      const size_t num_instances = t0_transforms.size();
      transforms_at_time.emplace(0.0f, std::move(t0_transforms));
      for (size_t i = 0; i < num_instances; ++i) {
        result.push_back(
            SampleTransformOverInterval(
                [&](float t) {
                  auto it = transforms_at_time.find(t);
                  if (it == transforms_at_time.end()) {
                    it = transforms_at_time
                             .emplace(
                                 t, ComputeInstanceTransformsAtTime(prototype_id,
                                                                    t))
                             .first;
                  }
                  return it->second.AsConst()[i];
                },
                interval, sample_times, /*uniform=*/true)
                .Map([](const GfMatrix4d& m) {
                  return UsdToMitsubaTransform(m);
                }));
      }
    }
  }

  {
    absl::MutexLock lock(cache_mutex_);
    cached_transforms_[prototype_id] = result;
  }
  return result;
}

bool HdMitsubaInstancer::GetContributingSampleTimesForInterval(
    const GfVec2f& interval, std::vector<float>* times) {
  HdRenderIndex& render_index = GetDelegate()->GetRenderIndex();
  HdSceneIndexBaseRefPtr scene_index = render_index.GetTerminalSceneIndex();
  if (!TF_VERIFY(scene_index)) {
    return false;
  }
  HdSceneIndexPrim prim = scene_index->GetPrim(GetId());
  bool varies = false;
  auto collect = [&](const HdContainerDataSourceHandle& container,
                     const HdDataSourceLocator& locator) {
    if (!container) return;
    if (auto ds = HdSampledDataSource::Cast(
            HdContainerDataSource::Get(container, locator))) {
      std::vector<float> ds_times;
      if (ds->GetContributingSampleTimesForInterval(interval[0], interval[1],
                                                    &ds_times)) {
        times->insert(times->end(), ds_times.begin(), ds_times.end());
        varies = true;
      }
    }
  };
  collect(prim.dataSource, TransformLocator());
  if (HdPrimvarsSchema primvars =
          HdPrimvarsSchema::GetFromParent(prim.dataSource);
      primvars.IsDefined()) {
    HdContainerDataSourceHandle container = primvars.GetContainer();
    collect(container, TransformsLocator());
    collect(container, TranslationsLocator());
    collect(container, ScalesLocator());
    collect(container, RotationsLocator());
  }
  const SdfPath parent_instancer_id = GetParentId();
  if (!parent_instancer_id.IsEmpty()) {
    if (auto* parent_instancer = static_cast<HdMitsubaInstancer*>(
            render_index.GetInstancer(parent_instancer_id))) {
      if (parent_instancer->GetContributingSampleTimesForInterval(interval,
                                                                  times)) {
        varies = true;
      }
    }
  }
  return varies;
}

VtMatrix4dArray HdMitsubaInstancer::ComputeInstanceTransformsAtTime(
    const SdfPath& prototype_id, float time) {
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

    instancer_transforms = GetParam(primvars_container, TransformsLocator(),
                                    instancer_transforms, time);
    instancer_translations = GetParam(primvars_container, TranslationsLocator(),
                                      instancer_translations, time);
    instancer_scales =
        GetParam(primvars_container, ScalesLocator(), instancer_scales, time);

    // Rotations can have different types, try them one by one.
    if (auto data_source = HdSampledDataSource::Cast(
            HdContainerDataSource::Get(primvars_container, RotationsLocator()))) {
      VtValue value = data_source->GetValue(time);
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

  GfMatrix4d instancer_transform = GetParam<GfMatrix4d>(
      prim.dataSource, TransformLocator(), GfMatrix4d(1.0), time);

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
          parent_instancer->ComputeInstanceTransformsAtTime(GetId(), time);
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
  return transforms;
}

PXR_NAMESPACE_CLOSE_SCOPE
