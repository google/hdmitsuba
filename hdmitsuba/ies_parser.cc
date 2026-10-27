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

#include "hdmitsuba/ies_parser.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <drjit/math.h>
#include <drjit/sphere.h>
#include <mitsuba/core/logger.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using mitsuba::ClassName;
using mitsuba::Error;

// Tolerance for comparing angles in degrees (as OpenUSD's `pxrIES`).
constexpr double kAngleTolerance = 1e-4;
// Minimum step in radians between consecutive vertical knots.
constexpr double kMinKnotStepRad = 1e-6;
constexpr size_t kIesHeaderTokenCount = 13;
constexpr int kPhotometricTypeC = 1;

bool ParseDouble(std::string_view token, double* out) {
  return absl::SimpleAtod(token, out) && std::isfinite(*out);
}

bool AngleClose(double a, double b) {
  return std::abs(a - b) < kAngleTolerance;
}

// Finds the `TILT=...` line, validates `TILT=NONE`, and returns the remaining
// content strictly after that line.
std::string_view FindPostTiltContent(std::string_view content) {
  while (!content.empty()) {
    size_t nl = content.find('\n');
    std::string_view line = absl::StripAsciiWhitespace(content.substr(0, nl));
    content = (nl == std::string_view::npos) ? std::string_view()
                                             : content.substr(nl + 1);
    if (absl::StartsWith(line, "TILT=")) {
      if (line != "TILT=NONE") {
        Throw(
            "Unsupported IES tilt specification '%s'; only 'TILT=NONE' is "
            "supported.",
            line);
      }
      return content;
    }
  }
  Throw("Unsupported IES profile: missing 'TILT=NONE' line.");
}

// Checks whether `h_angles` forms a standard LM-63 Type C 2D symmetry domain
// (`[0, 90]`, `[0, 180]`, `[90, 270]`, or `[0, 360]`) and, if so, expands
// `h_angles` and `h_slices` to cover `[0, 360]` (matching Blender Cycles'
// `IESFile::process_type_c` and OpenUSD's `pxrIES`).
bool ExpandTypeCHorizontalSymmetry(std::vector<double>& h_angles,
                                   std::vector<std::vector<double>>& h_slices) {
  if (h_angles.size() < 2) {
    return false;
  }
  if (AngleClose(h_angles.front(), 90.0) &&
      AngleClose(h_angles.back(), 270.0)) {
    for (double& angle : h_angles) {
      angle -= 90.0;
    }
  }
  if (!AngleClose(h_angles.front(), 0.0)) {
    return false;
  }
  for (double target_max : {180.0, 360.0}) {
    if (AngleClose(h_angles.back(), 0.5 * target_max)) {
      for (size_t i = h_angles.size() - 1; i-- > 0;) {
        h_angles.push_back(target_max - h_angles[i]);
        h_slices.push_back(h_slices[i]);
      }
    }
  }
  if (AngleClose(h_angles.back(), 360.0)) {
    return true;
  }
  // Some full-circle files omit the 360 deg slice, which equals the 0 deg one.
  // Restore it if the gap to 360 deg matches the first or last angle step.
  size_t n = h_angles.size();
  double gap = 360.0 - h_angles.back();
  if (!AngleClose(h_angles[n - 1] - h_angles[n - 2], gap) &&
      !AngleClose(h_angles[1] - h_angles[0], gap)) {
    return false;
  }
  h_angles.push_back(360.0);
  h_slices.push_back(h_slices.front());
  return true;
}

// Integrates the profile's average luminous intensity over the unit sphere or
// front hemisphere (matching OpenUSD hdEmbree's `PxrIESFile::load`).
double ComputeIesPower(const IesProfile& profile) {
  const std::vector<double>& v_knots = profile.vertical_angles;
  const std::vector<double>& h_knots = profile.horizontal_angles;
  size_t v_count = v_knots.size();
  if (v_count < 2) {
    return profile.values.empty() ? 0.0 : profile.values.front();
  }
  double power = 0.0;
  for (size_t v = 0; v + 1 < v_count; ++v) {
    double dv = v_knots[v + 1] - v_knots[v];
    double sin_mid = std::sin(v_knots[v] + 0.5 * dv);
    if (h_knots.empty()) {
      double avg = 0.5 * (profile.values[v] + profile.values[v + 1]);
      power += drjit::TwoPi<double> * dv * sin_mid * avg;
    } else {
      for (size_t h = 0; h + 1 < h_knots.size(); ++h) {
        double dh = h_knots[h + 1] - h_knots[h];
        double i0 = 0.5 * (profile.values[h * v_count + v] +
                           profile.values[h * v_count + v + 1]);
        double i1 = 0.5 * (profile.values[(h + 1) * v_count + v] +
                           profile.values[(h + 1) * v_count + v + 1]);
        power += dh * dv * sin_mid * 0.5 * (i0 + i1);
      }
    }
  }
  bool is_sphere =
      (v_knots.back() - v_knots.front()) > (0.5 * drjit::Pi<double> + 0.1);
  return power / (drjit::Pi<double> * (is_sphere ? 4.0 : 2.0));
}

// Makes the vertical knots of `profile` strictly increasing by nudging
// duplicate knots apart by `kMinKnotStepRad`, and duplicates a single knot
// (along with its candela values) so that there are at least two knots.
void MakeVerticalKnotsStrictlyIncreasing(IesProfile& profile) {
  std::vector<double>& v_knots = profile.vertical_angles;
  if (v_knots.size() == 1) {
    v_knots.push_back(v_knots.front());
    std::vector<double> values;
    values.reserve(2 * profile.values.size());
    for (double value : profile.values) {
      values.insert(values.end(), 2, value);
    }
    profile.values = std::move(values);
  }
  for (size_t i = 1; i < v_knots.size(); ++i) {
    v_knots[i] = std::max(v_knots[i], v_knots[i - 1] + kMinKnotStepRad);
  }
}

}  // namespace

IesProfile ParseIesString(std::string_view ies_content) {
  std::string_view post_tilt = FindPostTiltContent(ies_content);

  std::vector<std::string_view> tokens =
      absl::StrSplit(post_tilt, absl::ByAnyChar(" \t\r\n"), absl::SkipEmpty());
  if (tokens.size() < kIesHeaderTokenCount) {
    Throw(
        "Invalid IES profile: expected at least %zu header tokens after "
        "TILT=NONE, got %zu.",
        kIesHeaderTokenCount, tokens.size());
  }

  // Lumens, units, opening dimensions and electrical data are unused.
  int num_lamps = 0, n_vertical = 0, n_horizontal = 0, photometric_type = 0;
  double candela_mult = 0.0;
  if (!absl::SimpleAtoi(tokens[0], &num_lamps) ||
      !ParseDouble(tokens[2], &candela_mult) ||
      !absl::SimpleAtoi(tokens[3], &n_vertical) ||
      !absl::SimpleAtoi(tokens[4], &n_horizontal) ||
      !absl::SimpleAtoi(tokens[5], &photometric_type)) {
    Throw("Failed to parse IES header tokens.");
  }
  if (num_lamps != 1 || candela_mult != 1.0 || n_vertical <= 0 ||
      n_horizontal <= 0 || photometric_type != kPhotometricTypeC) {
    Throw(
        "Unsupported IES header parameters (requires num_lamps=1, "
        "candela_multiplier=1, and Type C photometry).");
  }

  size_t v_count = static_cast<size_t>(n_vertical);
  size_t h_count = static_cast<size_t>(n_horizontal);
  size_t expected_total_tokens =
      kIesHeaderTokenCount + v_count + h_count + h_count * v_count;
  if (tokens.size() != expected_total_tokens) {
    Throw("Invalid IES profile: expected %zu tokens after TILT=NONE, got %zu.",
          expected_total_tokens, tokens.size());
  }

  size_t cursor = kIesHeaderTokenCount;
  IesProfile profile;
  profile.vertical_angles.resize(v_count);
  double prev_v_deg = 0.0;
  for (size_t v = 0; v < v_count; ++v) {
    double angle = 0.0;
    if (!ParseDouble(tokens[cursor++], &angle) || angle < 0.0 ||
        angle > 180.0 || (v > 0 && angle < prev_v_deg)) {
      Throw("IES vertical angles must be non-decreasing floats in [0, 180].");
    }
    prev_v_deg = angle;
    profile.vertical_angles[v] = drjit::deg_to_rad(angle);
  }

  std::vector<double> horizontal_angles(h_count);
  for (size_t h = 0; h < h_count; ++h) {
    double angle = 0.0;
    if (!ParseDouble(tokens[cursor++], &angle) || angle < 0.0 ||
        angle > 360.0 || (h > 0 && angle <= horizontal_angles[h - 1])) {
      Throw(
          "IES horizontal angles must be strictly increasing floats in "
          "[0, 360].");
    }
    horizontal_angles[h] = angle;
  }

  std::vector<std::vector<double>> h_slices(h_count,
                                            std::vector<double>(v_count));
  for (auto& slice : h_slices) {
    for (auto& candela : slice) {
      if (!ParseDouble(tokens[cursor++], &candela) || candela < 0.0) {
        Throw("IES candela values must be non-negative floats.");
      }
    }
  }

  if (h_count == 1) {
    profile.values = std::move(h_slices.front());
  } else {
    if (!ExpandTypeCHorizontalSymmetry(horizontal_angles, h_slices)) {
      Throw(
          "IES horizontal angles [%g, %g] do not form a valid LM-63 Type C 2D "
          "symmetry domain ([0, 90], [0, 180], [90, 270], or [0, 360]).",
          horizontal_angles.front(), horizontal_angles.back());
    }
    for (auto& angle : horizontal_angles) {
      angle = drjit::deg_to_rad(angle);
    }
    profile.horizontal_angles = std::move(horizontal_angles);
    profile.values.reserve(h_slices.size() * v_count);
    for (const auto& slice : h_slices) {
      profile.values.insert(profile.values.end(), slice.begin(), slice.end());
    }
  }
  profile.power = ComputeIesPower(profile);
  MakeVerticalKnotsStrictlyIncreasing(profile);
  return profile;
}

IesProfile ParseIesFile(std::string_view path) {
  if (path.empty()) {
    Throw("IES file path is empty.");
  }
  const std::string path_str(path);
  // Read filesystem paths directly, so that the `ies` plugin also works where
  // no USD plugins are registered (e.g. in plain Mitsuba scenes).
  if (std::ifstream file(path_str, std::ios::binary); file) {
    std::string content((std::istreambuf_iterator<char>(file)), {});
    return ParseIesString(content);
  }
  if (std::shared_ptr<ArAsset> asset =
          ArGetResolver().OpenAsset(ArResolvedPath(path_str))) {
    if (std::shared_ptr<const char> buf = asset->GetBuffer()) {
      return ParseIesString({buf.get(), asset->GetSize()});
    }
  }
  Throw("Failed to open IES file '%s'.", path);
}

PXR_NAMESPACE_CLOSE_SCOPE
