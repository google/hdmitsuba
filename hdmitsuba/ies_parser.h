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

#include <pxr/pxr.h>

#include <string_view>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

// Parsed 1D or 2D Type C IES luminous intensity profile.
//
// Currently supported subset of IESNA LM-63 (1991 / 2002):
// - Type C photometry (`photometric_type == 1`), supporting both 1D radially
//   symmetric profiles and full 2D azimuthally varying profiles with standard
//   LM-63 symmetry domains (`[0, 90]`, `[0, 180]`, `[90, 270]`, and
//   `[0, 360]`).
// - Raw angle-space tables in radians over vertical (`theta`) and horizontal
//   (`phi`) knots, without resampling.
// - `TILT=NONE`, `num_lamps == 1`, and `candela_multiplier == 1.0`.
//
// Features not currently supported (documented for future reference):
// - Back-hemisphere (`theta > 90 deg`, `cos(theta) < 0`) emission.
// - Lamp-to-luminaire tilt data (`TILT != NONE`).
// - `num_lamps != 1`, `candela_multiplier != 1.0`, `photometric_type != 1`
//   (Type A / Type B photometry), `ballast_factor`, and `lumens_per_lamp`
//   flux normalization.
struct IesProfile {
  // Vertical (`theta`) knot angles in radians over `[0, pi]`. There are at
  // least two knots, and they are strictly increasing: duplicate knots (step
  // discontinuities) are nudged apart by a tiny epsilon, and a single knot
  // (constant profile) is duplicated, along with its candela values.
  std::vector<double> vertical_angles;
  // Horizontal (`phi`) knot angles in radians over `[0, 2 * pi]` after Type C
  // symmetry expansion, or empty for 1D radially symmetric profiles
  // (`num_horizontal == 1`).
  std::vector<double> horizontal_angles;
  // Row-major (`[num_horizontal, num_vertical]`, with `num_horizontal =
  // max(horizontal_angles.size(), 1)`) candela values matching the raw (or
  // horizontally symmetry-expanded) IES table.
  std::vector<double> values;
  // Average luminous intensity (`power / (4 * pi)` for full-sphere profiles,
  // or `power / (2 * pi)` for hemispherical profiles), used when
  // `shaping:ies:normalize` (`normalize = true`) is enabled.
  double power = 0.0;
};

// Parses an IES profile from an in-memory string. Throws on invalid or
// unsupported input.
IesProfile ParseIesString(std::string_view ies_content);

// Loads and parses an IES profile from an `ArResolver` asset path or a
// filesystem path. Throws if the file cannot be opened or contains an invalid
// profile.
IesProfile ParseIesFile(std::string_view path);

PXR_NAMESPACE_CLOSE_SCOPE
