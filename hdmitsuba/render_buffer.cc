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

#include "hdmitsuba/render_buffer.h"

#include <cstddef>

#include <pxr/base/gf/vec3i.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/imaging/hd/perfLog.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/pxr.h>

#include "hdmitsuba/debug_codes.h"

PXR_NAMESPACE_OPEN_SCOPE

HdMitsubaRenderBuffer::HdMitsubaRenderBuffer(const SdfPath& id)
    : HdRenderBuffer(id) {}

HdMitsubaRenderBuffer::~HdMitsubaRenderBuffer() { _Deallocate(); }

bool HdMitsubaRenderBuffer::Allocate(const GfVec3i& dimensions, HdFormat format,
                                     bool multiSampled) {
  TF_DEBUG(HDMITSUBA_LIFECYCLE)
      .Msg("Allocate: %d x %d x %d format: %d (multisampled: %d)\n",
           dimensions[0], dimensions[1], dimensions[2], format, multiSampled);

  if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] != 1) {
    TF_CODING_ERROR("Invalid dimensions %s for render buffer %s",
                    TfStringify(dimensions).c_str(), GetId().GetText());
    return false;
  }

  size_t format_size = HdDataSizeOfFormat(format);
  if (format_size == 0) {
    TF_CODING_ERROR("Invalid format %s for render buffer %s",
                    TfStringify(format).c_str(), GetId().GetText());
    return false;
  }

  if (multiSampled) {
    TF_WARN("Multisampled render buffers are not supported.");
  }

  _Deallocate();

  width_ = dimensions[0];
  height_ = dimensions[1];
  format_ = format;
  converged_ = false;
  size_t byte_size = width_ * height_ * format_size;
  buffer_.resize(byte_size);
  return true;
}

void* HdMitsubaRenderBuffer::Map() {
  mapped_ = true;
  return buffer_.data();
}

void HdMitsubaRenderBuffer::Unmap() { mapped_ = false; }

void HdMitsubaRenderBuffer::Resolve() {
  // No-op for non-multisampled buffers.
}

bool HdMitsubaRenderBuffer::IsConverged() const { return converged_; }

void HdMitsubaRenderBuffer::_Deallocate() {
  buffer_.clear();
  buffer_.shrink_to_fit();
  width_ = 0;
  height_ = 0;
  format_ = HdFormatInvalid;
  mapped_ = false;
  converged_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
