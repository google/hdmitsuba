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

#include <atomic>
#include <cstdint>
#include <vector>

#include <pxr/base/gf/declare.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdMitsubaRenderBuffer final : public HdRenderBuffer {
 public:
  explicit HdMitsubaRenderBuffer(const SdfPath& id);
  ~HdMitsubaRenderBuffer() override;

  bool Allocate(const GfVec3i& dimensions, HdFormat format,
                bool multiSampled) override;

  unsigned int GetWidth() const override { return width_; }
  unsigned int GetHeight() const override { return height_; }
  unsigned int GetDepth() const override { return 1; }
  HdFormat GetFormat() const override { return format_; }
  bool IsMultiSampled() const override { return false; }

  void* Map() override;
  void Unmap() override;
  bool IsMapped() const override { return mapped_; }
  void Resolve() override;
  bool IsConverged() const override;
  void SetConverged(bool converged) { converged_ = converged; }

 protected:
  void _Deallocate() override;

 private:
  unsigned int width_ = 0;
  unsigned int height_ = 0;
  HdFormat format_ = HdFormatInvalid;
  std::atomic<bool> mapped_{false};
  std::atomic<bool> converged_{false};
  std::vector<uint8_t> buffer_;
};

PXR_NAMESPACE_CLOSE_SCOPE
