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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <drjit/array.h>
#include <drjit/math.h>
#include <drjit/tensor.h>
#include <mitsuba/core/bbox.h>
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
#include <mitsuba/render/shape.h>
#include <pxr/pxr.h>

#include "hdmitsuba/ies_parser.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

namespace dr = drjit;
using mitsuba::Emitter;
using mitsuba::Error;
using mitsuba::ScopedPhase;

// Photometric wrapper emitter (`'ies'`) that modulates an existing child
// emitter (`'point'` or `'area'`) by a 1D or 2D Type C IESNA LM-63 luminous
// intensity profile:
// - `filename`: Type C IES file, linearly interpolated between its `(theta,
//   phi)` knots.
// - `ies_profile` (traversed, differentiable): the candela values at those
//   knots, a `(V,)` or `(H, V)` `TensorXf`.
template <typename Float, typename Spectrum>
class IesEmitter final : public Emitter<Float, Spectrum> {
 public:
  MI_IMPORT_BASE(Emitter, set_shape, m_flags, m_shape, m_sampling_weight,
                 world_transform, traverse_world_transform)
  MI_IMPORT_TYPES(Scene, Shape)
  MI_DECLARE_CLASS(IesEmitter)

  using DoubleStorage =
      mitsuba::DynamicBuffer<dr::replace_scalar_t<Float, double>>;
  using FloatStorage = mitsuba::DynamicBuffer<Float>;

  explicit IesEmitter(const mitsuba::Properties& props) : Base(props) {
    for (auto& prop : props.objects()) {
      if (Base* emitter = prop.try_get<Base>()) {
        if (m_nested) {
          Throw("IesEmitter: only a single nested emitter can be specified.");
        }
        m_nested = emitter;
      }
    }
    if (!m_nested) {
      Throw(
          "IesEmitter requires a nested emitter plugin (e.g. 'area' or "
          "'point').");
    }

    m_angle_scale = props.get<ScalarFloat>("angle_scale", 0.f);
    bool normalize = props.get<bool>("normalize", false);

    IesProfile profile = ParseIesFile(props.get<std::string>("filename"));
    if (normalize && profile.power > 0.0) {
      double inv_power = 1.0 / profile.power;
      for (double& v : profile.values) {
        v *= inv_power;
      }
    }
    auto to_storage = [](const std::vector<double>& data) {
      return FloatStorage(dr::load<DoubleStorage>(data.data(), data.size()));
    };
    const std::vector<double>& v_knots = profile.vertical_angles;
    const std::vector<double>& h_knots = profile.horizontal_angles;
    m_v_max = static_cast<ScalarFloat>(v_knots.back());
    m_v_nodes = to_storage(v_knots);
    if (!h_knots.empty()) {
      m_h_nodes = to_storage(h_knots);
    }
    size_t shape[] = {h_knots.size(), v_knots.size()};
    size_t ndim = h_knots.empty() ? 1 : 2;
    m_ies_profile =
        TensorXf(to_storage(profile.values), ndim, shape + 2 - ndim);
    m_flags = m_nested->flags();
  }

  void set_shape(Shape* shape) override {
    if (shape && mitsuba::has_flag(m_nested->flags(),
                                   mitsuba::EmitterFlags::DeltaPosition)) {
      Throw("IesEmitter: cannot attach a positional delta emitter to a shape.");
    }
    if (m_shape != shape) {
      Base::set_shape(shape);
    }
    if (m_nested->shape() != shape) {
      m_nested->set_shape(shape);
    }
    m_flags = m_nested->flags();
  }

  void traverse(mitsuba::TraversalCallback* cb) override {
    Base::traverse(cb);
    traverse_world_transform(cb);
    cb->put("nested_emitter", m_nested, mitsuba::ParamFlags::Differentiable);
    cb->put("ies_profile", m_ies_profile, mitsuba::ParamFlags::Differentiable);
  }

  void parameters_changed(const std::vector<std::string>& keys) override {
    m_nested->parameters_changed(keys);
    if (keys.empty() || mitsuba::string::contains(keys, "ies_profile")) {
      check_profile_shape();
    }
    m_flags = m_nested->flags();
    Base::parameters_changed(keys);
  }

  Spectrum eval(const SurfaceInteraction3f& si, Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    if (m_sampling_weight <= 0.f) {
      return 0.f;
    }
    return m_nested->eval(si, active) *
           eval_ies_attenuation(si.to_world(si.wi), si.time, active);
  }

  std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                        const Point2f& sample2,
                                        const Point2f& sample3,
                                        Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointSampleRay, active);
    if (m_sampling_weight <= 0.f) {
      return {dr::zeros<Ray3f>(), 0.f};
    }
    auto [ray, weight] =
        m_nested->sample_ray(time, wavelength_sample, sample2, sample3, active);
    return {ray, weight * eval_ies_attenuation(ray.d, time, active)};
  }

  std::pair<DirectionSample3f, Spectrum> sample_direction(
      const Interaction3f& it, const Point2f& sample,
      Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointSampleDirection, active);
    if (m_sampling_weight <= 0.f) {
      return {dr::zeros<DirectionSample3f>(), 0.f};
    }
    auto [ds, spec] = m_nested->sample_direction(it, sample, active);
    ds.emitter = this;
    return {ds, spec * eval_ies_attenuation(-ds.d, it.time, active)};
  }

  Float pdf_direction(const Interaction3f& it, const DirectionSample3f& ds,
                      Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    return m_sampling_weight <= 0.f ? Float(0.f)
                                    : m_nested->pdf_direction(it, ds, active);
  }

  Spectrum eval_direction(const Interaction3f& it, const DirectionSample3f& ds,
                          Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    if (m_sampling_weight <= 0.f) {
      return 0.f;
    }
    return m_nested->eval_direction(it, ds, active) *
           eval_ies_attenuation(-ds.d, it.time, active);
  }

  std::pair<PositionSample3f, Float> sample_position(
      Float time, const Point2f& sample, Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointSamplePosition, active);
    if (m_sampling_weight <= 0.f) {
      return {dr::zeros<PositionSample3f>(), 0.f};
    }
    return m_nested->sample_position(time, sample, active);
  }

  Float pdf_position(const PositionSample3f& ps, Mask active) const override {
    MI_MASKED_FUNCTION(mitsuba::ProfilerPhase::EndpointEvaluate, active);
    return m_sampling_weight <= 0.f ? Float(0.f)
                                    : m_nested->pdf_position(ps, active);
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
        << "  ies_knots = [" << std::max<size_t>(dr::width(m_h_nodes), 1)
        << ", " << dr::width(m_v_nodes) << "],\n"
        << "  nested_emitter = " << mitsuba::string::indent(m_nested) << "\n]";
    return oss.str();
  }

 private:
  std::pair<Vector3f, Vector3f> get_uv_axes(const Float& time) const {
    AffineTransform4f to_world =
        m_nested->world_transform(time) * world_transform(time);
    if (m_shape) {
      to_world = m_shape->to_world() * to_world;
    }
    return {dr::normalize(to_world * Vector3f(1.f, 0.f, 0.f)),
            dr::normalize(to_world * Vector3f(0.f, 1.f, 0.f))};
  }

  // Checks that `ies_profile` still has one value per `(phi, theta)` knot.
  void check_profile_shape() const {
    size_t ndim = m_h_nodes.empty() ? 1 : 2;
    if (m_ies_profile.ndim() != ndim ||
        m_ies_profile.shape(ndim - 1) != dr::width(m_v_nodes) ||
        (ndim == 2 && m_ies_profile.shape(0) != dr::width(m_h_nodes))) {
      Throw("IesEmitter: 'ies_profile' must keep the shape of the IES table.");
    }
  }

  // Returns the knot interval `[i, i + 1]` of the strictly increasing
  // `knots` that contains `x`, and the interpolation weight of knot `i + 1`,
  // clamped to the first and last knots.
  std::pair<UInt32, Float> find_knot_interval(const FloatStorage& knots,
                                              const Float& x) const {
    UInt32 i = mitsuba::math::find_interval<UInt32>(
        static_cast<uint32_t>(dr::width(knots)),
        [&](UInt32 idx) { return dr::gather<Float>(knots, idx) <= x; });
    Float x0 = dr::gather<Float>(knots, i);
    Float x1 = dr::gather<Float>(knots, i + 1u);
    return {i, dr::clip((x - x0) / (x1 - x0), 0.f, 1.f)};
  }

  Float eval_ies_attenuation(const Vector3f& w, const Float& time,
                             Mask active) const {
    auto [u_axis, v_axis] = get_uv_axes(time);
    Float cos_theta = dr::clip(
        dr::dot(w, dr::normalize(dr::cross(u_axis, v_axis))), -1.f, 1.f);
    Float theta = dr::safe_acos(cos_theta);
    if (m_angle_scale > 0.f) {
      theta /= m_angle_scale;
    } else if (m_angle_scale < 0.f) {
      theta = (dr::Pi<Float> - theta) / m_angle_scale + dr::Pi<Float>;
    }
    active &= (theta >= 0.f) && (theta <= m_v_max);

    // Bilinear interpolation of the `(H, V)` table, where `H = 1` for 1D
    // profiles.
    const FloatStorage& values = m_ies_profile.array();
    UInt32 v_idx;
    Float v_weight;
    std::tie(v_idx, v_weight) = find_knot_interval(m_v_nodes, theta);
    auto eval_row = [&](const UInt32& row_offset) {
      return dr::lerp(
          dr::gather<Float>(values, row_offset + v_idx, active),
          dr::gather<Float>(values, row_offset + v_idx + 1u, active), v_weight);
    };
    if (m_h_nodes.empty()) {
      return eval_row(0u);
    }
    Float phi = dr::atan2(dr::select(active, dr::dot(w, v_axis), 0.f),
                          dr::select(active, -dr::dot(w, u_axis), 1.f));
    phi = dr::select(phi < 0.f, phi + dr::TwoPi<Float>, phi);
    auto [h_idx, h_weight] = find_knot_interval(m_h_nodes, phi);
    auto v_count = static_cast<uint32_t>(dr::width(m_v_nodes));
    return dr::lerp(eval_row(h_idx * v_count), eval_row((h_idx + 1u) * v_count),
                    h_weight);
  }

  mitsuba::ref<Base> m_nested;
  // Candela values at the `(phi, theta)` knots `m_h_nodes` (empty for 1D
  // profiles) and `m_v_nodes`, in radians.
  TensorXf m_ies_profile;
  FloatStorage m_v_nodes;
  FloatStorage m_h_nodes;
  ScalarFloat m_v_max = 0.f;
  ScalarFloat m_angle_scale = 0.f;

  MI_TRAVERSE_CB(Base, m_nested, m_ies_profile, m_v_nodes, m_h_nodes)
};

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE

NAMESPACE_BEGIN(mitsuba)
MI_EXPORT_PLUGIN(PXR_NS::IesEmitter)
NAMESPACE_END(mitsuba)
