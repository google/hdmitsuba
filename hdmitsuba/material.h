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

#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/sprim.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaMaterial final : public HdMaterial {
 public:
  explicit HdMitsubaMaterial(const SdfPath& id);
  ~HdMitsubaMaterial() override = default;

  void Sync(HdSceneDelegate* scene_delegate, HdRenderParam* render_param,
            HdDirtyBits* dirty_bits) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

  void Finalize(HdRenderParam* renderParam) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
