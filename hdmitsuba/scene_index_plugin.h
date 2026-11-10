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

#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

// Dirties the display style of all meshes in `terminal_scene`, which makes them
// re-resolve their refine level. Does nothing if the hdMitsuba scene index
// plugin is not part of the scene index chain. Must not be called during
// HdRenderIndex::SyncAll.
void DirtyMeshDisplayStyles(const HdSceneIndexBasePtr& terminal_scene);

PXR_NAMESPACE_CLOSE_SCOPE
