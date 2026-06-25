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

#pragma once

#include <map>

#include <absl/synchronization/mutex.h>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaInstancer final : public HdInstancer {
 public:
  HdMitsubaInstancer(HdSceneDelegate* delegate, const SdfPath& id);
  ~HdMitsubaInstancer() override = default;

  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

  VtMatrix4dArray ComputeInstanceTransforms(const SdfPath& prototype_id);

 private:
  std::map<SdfPath, VtMatrix4dArray> cached_transforms_;
  absl::Mutex cache_mutex_;
};

PXR_NAMESPACE_CLOSE_SCOPE
