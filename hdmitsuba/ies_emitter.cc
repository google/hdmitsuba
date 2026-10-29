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

#include "hdmitsuba/ies_emitter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/ascii.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <drjit/math.h>
#include <drjit/tensor.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/records.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace dr = drjit;
using mitsuba::ClassName;
using mitsuba::Emitter;
using mitsuba::Error;
using mitsuba::ScopedPhase;

// Type C IES luminous intensity profile on its raw knots.
struct IesProfile {
  // Vertical (`theta`) knots in radians, non-decreasing (duplicates encode
  // steps), at least two.
  std::vector<double> vertical_angles;
  // Horizontal (`phi`) knots in radians over `[0, 2 * pi]`, at least two.
  std::vector<double> horizontal_angles;
  // Row-major `[horizontal, vertical]` candela values.
  std::vector<double> values;
};

// Compares two angles in degrees.
bool AngleClose(double a, double b) { return std::abs(a - b) < 1e-4; }

// Expands the horizontal angles `h` (in degrees) and the matching rows of
// `values` (`v_count` values each) over `[0, 360]` deg with the LM-63 Type C
// symmetries.
void ExpandTypeCSymmetry(std::vector<double>& h, std::vector<double>& values,
                         size_t v_count) {
  auto append_row = [&](size_t row) {
    for (size_t i = 0; i < v_count; ++i) {
      values.push_back(values[row * v_count + i]);
    }
  };
  if (h.size() == 1) {
    h = {0.0, 360.0};
    append_row(0);
    return;
  }
  // Rotate `[90, 270]` profiles to `[0, 180]`.
  if (AngleClose(h.front(), 90.0)) {
    for (double& angle : h) {
      angle -= 90.0;
    }
  }
  if (!AngleClose(h.front(), 0.0)) {
    Throw("IES horizontal angles must start at 0 or 90 deg, got %g.",
          h.front());
  }
  // Mirror `[0, 90]` to `[0, 180]`, and `[0, 180]` to `[0, 360]`.
  for (double end : {180.0, 360.0}) {
    if (AngleClose(h.back(), 0.5 * end)) {
      for (size_t i = h.size() - 1; i-- > 0;) {
        h.push_back(end - h[i]);
        append_row(i);
      }
    }
  }
  if (!AngleClose(h.back(), 360.0)) {
    // Some full-circle files omit the 360 deg slice, which equals the 0 deg
    // one. Restore it if the gap to 360 deg matches the first or last step.
    size_t n = h.size();
    double gap = 360.0 - h.back();
    if (!AngleClose(h[n - 1] - h[n - 2], gap) &&
        !AngleClose(h[1] - h[0], gap)) {
      Throw("IES horizontal angles [%g, %g] are not a Type C symmetry domain.",
            h.front(), h.back());
    }
    h.push_back(360.0);
    append_row(0);
  }
}

// Integrates the average intensity of `profile` over the sphere (or hemisphere
// for profiles spanning at most ~90 deg vertically), as OpenUSD's
// `PxrIESFile::pxr_extra_process`.
double ComputeAverageIntensity(const IesProfile& profile) {
  const std::vector<double>& v = profile.vertical_angles;
  const std::vector<double>& h = profile.horizontal_angles;
  auto at = [&](size_t i, size_t j) {
    return profile.values[i * v.size() + j];
  };
  double power = 0.0;
  for (size_t i = 0; i + 1 < h.size(); ++i) {
    for (size_t j = 0; j + 1 < v.size(); ++j) {
      double dh = h[i + 1] - h[i];
      double dv = v[j + 1] - v[j];
      power += dh * dv * std::sin(v[j] + 0.5 * dv) * 0.25 *
               (at(i, j) + at(i, j + 1) + at(i + 1, j) + at(i + 1, j + 1));
    }
  }
  bool is_sphere = v.back() - v.front() > 0.5 * drjit::Pi<double> + 0.1;
  return power / (drjit::Pi<double> * (is_sphere ? 4.0 : 2.0));
}

// Loads the LM-63 Type C IES profile at `path`. Candela values include the
// candela multiplier and ballast factors, horizontal angles are expanded over
// `[0, 360]` deg, and `TILT=INCLUDE` data is skipped.
IesProfile LoadIesProfile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    Throw("Failed to open IES file '%s'.", path);
  }
  const std::string ies_content(std::istreambuf_iterator<char>(file), {});
  // Skip the keyword header up to and including the `TILT=` line.
  size_t tilt_pos = ies_content.find("TILT=");
  if (tilt_pos == std::string::npos) {
    Throw("Invalid IES profile: missing 'TILT=' line.");
  }
  std::string_view content = std::string_view(ies_content).substr(tilt_pos);
  size_t eol = content.find('\n');
  std::string_view tilt = absl::StripAsciiWhitespace(content.substr(0, eol));
  content = eol == std::string_view::npos ? "" : content.substr(eol + 1);

  std::vector<std::string_view> tokens =
      absl::StrSplit(content, absl::ByAnyChar(" \t\r\n,"), absl::SkipEmpty());
  size_t cursor = 0;
  auto next = [&]() {
    double value = 0.0;
    if (cursor >= tokens.size() ||
        !absl::SimpleAtod(tokens[cursor++], &value)) {
      Throw("Invalid IES profile: truncated or non-numeric data.");
    }
    return value;
  };
  auto next_vector = [&](size_t count, double scale) {
    std::vector<double> result(count);
    for (double& value : result) {
      value = next() * scale;
    }
    return result;
  };

  if (tilt == "TILT=INCLUDE") {
    next();  // Lamp-to-luminaire geometry.
    // Tilt angles and factors.
    cursor += 2 * static_cast<size_t>(std::max(next(), 0.0));
  } else if (tilt != "TILT=NONE") {
    Throw("Unsupported IES tilt specification '%s'.", tilt);
  }

  next();  // Number of lamps.
  next();  // Lumens per lamp.
  double multiplier = next();
  double v_count_value = next(), h_count_value = next();
  if (next() != 1.0) {
    Throw("Unsupported IES photometric type: only Type C is supported.");
  }
  cursor += 4;           // Units, width, length, height.
  multiplier *= next();  // Ballast factor.
  multiplier *= next();  // Ballast-lamp photometric factor.
  next();                // Input watts.
  if (v_count_value < 2.0 || h_count_value < 1.0) {
    Throw(
        "Invalid IES profile: needs at least 2 vertical and 1 horizontal "
        "angle.");
  }
  size_t v_count = static_cast<size_t>(v_count_value);
  size_t h_count = static_cast<size_t>(h_count_value);
  if (tokens.size() != cursor + v_count + h_count + v_count * h_count) {
    Throw("Invalid IES profile: expected %zu angles and values, got %zu.",
          v_count + h_count + v_count * h_count, tokens.size() - cursor);
  }

  IesProfile profile;
  profile.vertical_angles = next_vector(v_count, drjit::deg_to_rad(1.0));
  profile.horizontal_angles = next_vector(h_count, 1.0);
  profile.values = next_vector(v_count * h_count, multiplier);
  ExpandTypeCSymmetry(profile.horizontal_angles, profile.values, v_count);
  for (double& angle : profile.horizontal_angles) {
    angle = drjit::deg_to_rad(angle);
  }
  return profile;
}

/**!

.. _emitter-ies:

IES light profile (:monosp:`ies`)
---------------------------------

.. pluginparameters::

 * - (Nested plugin)
   - :paramtype:`emitter`
   - The emitter whose emission is modulated by the profile: :monosp:`area`
     when attached to a shape, or :monosp:`point` or :monosp:`spot` for a
     point light.
   - |exposed|, |differentiable|

 * - filename
   - |string|
   - Filename of the LM-63 Type C IES profile.

 * - angle_scale
   - |float|
   - Scales the vertical angles of the profile, as
     ``inputs:shaping:ies:angleScale`` of ``UsdLuxShapingAPI``. (Default: 0,
     i.e. no scaling)

 * - normalize
   - |bool|
   - Divide the profile by its average intensity, as
     ``inputs:shaping:ies:normalize`` of ``UsdLuxShapingAPI``. (Default:
     |false|)

 * - to_world
   - |transform| or |animation|
   - Specifies the rigid light frame of the profile. (Default: none, i.e.
     light space = world space)
   - |exposed|

 * - ies_profile
   - |tensor|
   - The :math:`(H, V)` candela values of the profile at its horizontal
     (:math:`\phi`) and vertical (:math:`\theta`) knots.
   - |exposed|, |differentiable|

This plugin modulates the emission of a nested emitter by a Type C IES
luminous intensity profile, bilinearly interpolated on the raw knots of the
profile. In its local coordinate system, :math:`\theta = 0` points along the
positive Z direction, and :math:`\phi = 0` along the negative X direction,
increasing towards positive Y. This frame is independent of the nested emitter
and its shape. There is no emission outside the vertical angle range of the
profile.

A non-positive ``sampling_weight`` turns the emitter off: the scene never
samples it, and it evaluates to 0 when it is hit. This is a performance
optimization, which switches lights off through ``traverse`` without
rebuilding the scene.

.. tabs::
    .. code-tab:: xml
        :name: ies-light

        <shape type="disk">
            <emitter type="ies">
                <string name="filename" value="profile.ies"/>
                <emitter type="area">
                    <rgb name="radiance" value="1.0"/>
                </emitter>
            </emitter>
        </shape>

    .. code-tab:: python

        'type': 'disk',
        'emitter': {
            'type': 'ies',
            'filename': 'profile.ies',
            'nested_emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 1.0,
                }
            }
        }

 */
template <typename Float, typename Spectrum>
class IesEmitter final : public Emitter<Float, Spectrum> {
 public:
  MI_IMPORT_BASE(Emitter, m_flags, m_sampling_weight, world_transform,
                 traverse_world_transform)
  MI_IMPORT_TYPES(Shape)
  MI_DECLARE_CLASS(IesEmitter)

  using FloatStorage = mitsuba::DynamicBuffer<Float>;

  explicit IesEmitter(const mitsuba::Properties& props) : Base(props) {
    for (auto& prop : props.objects()) {
      if (Base* emitter = prop.try_get<Base>()) {
        m_nested = emitter;
      }
    }
    if (!m_nested) {
      Throw("IesEmitter requires a nested emitter (e.g. 'area' or 'point').");
    }
    m_angle_scale = props.get<ScalarFloat>("angle_scale", 0.f);
    IesProfile profile = LoadIesProfile(props.get<std::string>("filename"));
    if (props.get<bool>("normalize", false)) {
      double average_intensity = ComputeAverageIntensity(profile);
      for (double& value : profile.values) {
        value /= average_intensity;
      }
    }
    m_theta = MakeAxis(profile.vertical_angles);
    m_phi = MakeAxis(profile.horizontal_angles);
    m_theta_knots = ToStorage(profile.vertical_angles);
    m_phi_knots = ToStorage(profile.horizontal_angles);
    size_t shape[] = {m_phi.size, m_theta.size};
    m_ies_profile = TensorXf(ToStorage(profile.values), 2, shape);
    m_flags = m_nested->flags();
  }

  void set_shape(Shape* shape) override {
    Base::set_shape(shape);
    m_nested->set_shape(shape);
    m_flags = m_nested->flags();
  }

  void traverse(mitsuba::TraversalCallback* cb) override {
    Base::traverse(cb);
    traverse_world_transform(cb);
    cb->put("nested_emitter", m_nested, mitsuba::ParamFlags::Differentiable);
    cb->put("ies_profile", m_ies_profile, mitsuba::ParamFlags::Differentiable);
  }

  void parameters_changed(const std::vector<std::string>& keys) override {
    m_nested->parameters_changed();
    Base::parameters_changed(keys);
  }

  Spectrum eval(const SurfaceInteraction3f& si, Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    if (m_sampling_weight <= 0.f) {
      return 0.f;
    }
    return m_nested->eval(si, active) *
           eval_profile(si.to_world(si.wi), si.time, active);
  }

  std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                        const Point2f& sample2,
                                        const Point2f& sample3,
                                        Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointSampleRay, active);
    auto [ray, weight] =
        m_nested->sample_ray(time, wavelength_sample, sample2, sample3, active);
    return {ray, weight * eval_profile(ray.d, time, active)};
  }

  std::pair<DirectionSample3f, Spectrum> sample_direction(
      const Interaction3f& it, const Point2f& sample,
      Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointSampleDirection, active);
    auto [ds, weight] = m_nested->sample_direction(it, sample, active);
    ds.emitter = this;
    return {ds, weight * eval_profile(-ds.d, it.time, active)};
  }

  Float pdf_direction(const Interaction3f& it, const DirectionSample3f& ds,
                      Mask active) const override {
    return m_nested->pdf_direction(it, ds, active);
  }

  Spectrum eval_direction(const Interaction3f& it, const DirectionSample3f& ds,
                          Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    if (m_sampling_weight <= 0.f) {
      return 0.f;
    }
    return m_nested->eval_direction(it, ds, active) *
           eval_profile(-ds.d, it.time, active);
  }

  std::pair<PositionSample3f, Float> sample_position(
      Float time, const Point2f& sample, Mask active) const override {
    return m_nested->sample_position(time, sample, active);
  }

  Float pdf_position(const PositionSample3f& ps, Mask active) const override {
    return m_nested->pdf_position(ps, active);
  }

  std::pair<Wavelength, Spectrum> sample_wavelengths(
      const SurfaceInteraction3f& si, Float sample,
      Mask active) const override {
    return m_nested->sample_wavelengths(si, sample, active);
  }

  ScalarBoundingBox3f bbox() const override { return m_nested->bbox(); }

  std::string to_string() const override {
    std::ostringstream oss;
    oss << "IesEmitter[\n"
        << "  ies_knots = [" << m_phi.size << ", " << m_theta.size << "],\n"
        << "  nested_emitter = " << mitsuba::string::indent(m_nested) << "\n]";
    return oss.str();
  }

 private:
  // Scalar description of the knots of an angle axis.
  struct Axis {
    ScalarFloat front = 0.f;
    ScalarFloat back = 0.f;
    // `1 / step` if the knots are uniformly spaced, and 0 otherwise.
    ScalarFloat inv_step = 0.f;
    uint32_t size = 0;
  };

  static Axis MakeAxis(const std::vector<double>& knots) {
    double step = (knots.back() - knots.front()) / (knots.size() - 1);
    bool uniform = step > 0.0;
    for (size_t i = 1; uniform && i < knots.size(); ++i) {
      uniform = std::abs(knots[i] - (knots.front() + i * step)) < 1e-6;
    }
    return {static_cast<ScalarFloat>(knots.front()),
            static_cast<ScalarFloat>(knots.back()),
            uniform ? static_cast<ScalarFloat>(1.0 / step) : 0.f,
            static_cast<uint32_t>(knots.size())};
  }

  static FloatStorage ToStorage(const std::vector<double>& data) {
    std::vector<ScalarFloat> values(data.begin(), data.end());
    return dr::load<FloatStorage>(values.data(), values.size());
  }

  // Returns the knot interval `[i, i + 1]` of `axis` containing `x` and the
  // interpolation weight of knot `i + 1`, clamped to the first and last knots.
  // Duplicate knots (steps) are never selected from the inside, so steps stay
  // sharp.
  std::pair<UInt32, Float> lookup(const Axis& axis, const FloatStorage& knots,
                                  const Float& x) const {
    if (axis.inv_step > 0.f) {
      Float t = (x - axis.front) * axis.inv_step;
      UInt32 i =
          UInt32(dr::clip(dr::floor(t), 0.f, ScalarFloat(axis.size - 2)));
      return {i, dr::clip(t - Float(i), 0.f, 1.f)};
    }
    UInt32 i = mitsuba::math::find_interval<UInt32>(
        axis.size,
        [&](const UInt32& j) { return dr::gather<Float>(knots, j) <= x; });
    Float x0 = dr::gather<Float>(knots, i);
    Float x1 = dr::gather<Float>(knots, i + 1u);
    return {i,
            dr::select(x1 > x0, dr::clip((x - x0) / (x1 - x0), 0.f, 1.f), 1.f)};
  }

  // Evaluates the profile in the world-space emission direction `w`.
  Float eval_profile(const Vector3f& w, const Float& time, Mask active) const {
    Vector3f local = world_transform(time).inverse() * w;
    Float theta = dr::safe_acos(local.z());
    // As OpenUSD's `PxrIESFile::eval`.
    if (m_angle_scale > 0.f) {
      theta /= m_angle_scale;
    } else if (m_angle_scale < 0.f) {
      theta = (dr::Pi<Float> - theta) / m_angle_scale + dr::Pi<Float>;
    }
    active &= (theta >= m_theta.front) && (theta <= m_theta.back);
    Float phi = dr::atan2(local.y(), -local.x());
    phi = dr::select(phi < 0.f, phi + dr::TwoPi<Float>, phi);

    auto [v_idx, v_weight] = lookup(m_theta, m_theta_knots, theta);
    auto [h_idx, h_weight] = lookup(m_phi, m_phi_knots, phi);
    UInt32 i0 = h_idx * m_theta.size + v_idx;
    UInt32 i1 = i0 + m_theta.size;
    auto value = [&](const UInt32& i) {
      return dr::gather<Float>(m_ies_profile.array(), i, active);
    };
    return dr::lerp(dr::lerp(value(i0), value(i0 + 1u), v_weight),
                    dr::lerp(value(i1), value(i1 + 1u), v_weight), h_weight);
  }

  mitsuba::ref<Base> m_nested;
  TensorXf m_ies_profile;
  FloatStorage m_theta_knots;
  FloatStorage m_phi_knots;
  Axis m_theta;
  Axis m_phi;
  ScalarFloat m_angle_scale = 0.f;

  MI_TRAVERSE_CB(Base, m_nested, m_ies_profile, m_theta_knots, m_phi_knots)
};

}  // namespace

void RegisterIesEmitter() {
  static const bool registered = [] {
    using namespace mitsuba;  // For `MI_REGISTER_PLUGIN`.
    auto register_fn = [](auto... args) {
      PluginManager::instance()->register_plugin(args..., nullptr, nullptr);
    };
    MI_REGISTER_PLUGIN(register_fn, "ies", IesEmitter)
    return true;
  }();
  (void)registered;
}

PXR_NAMESPACE_CLOSE_SCOPE
