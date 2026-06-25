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
#include <cstdint>

#include <nanobind/nanobind.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/external/boost/python.hpp>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/timeCode.h>

namespace bp = PXR_BOOST_PYTHON_NAMESPACE;

namespace nanobind {
namespace detail {

template <typename T>
struct boost_caster_traits {
  static constexpr bool value = false;
  static constexpr auto Name = nanobind::detail::const_name("");
};

}  // namespace detail
}  // namespace nanobind

#define NANOBIND_BOOST_CASTER(Type, NameStr)                            \
  template <>                                                           \
  struct nanobind::detail::boost_caster_traits<Type> {                  \
    static constexpr bool value = true;                                 \
    static constexpr auto Name = nanobind::detail::const_name(NameStr); \
  }

namespace nanobind {
namespace detail {

template <typename T>
struct type_caster<T, enable_if_t<boost_caster_traits<T>::value>> {
  NB_TYPE_CASTER(T, boost_caster_traits<T>::Name)

  bool from_python(handle src, uint8_t /*flags*/,
                   cleanup_list* /*cleanup*/) noexcept {
    bp::extract<Value> extractor(reinterpret_cast<PyObject*>(src.ptr()));
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    if (!extractor.check()) {
      return false;
    }
    try {
      value = extractor();
    } catch (const bp::error_already_set& e) {
      PyErr_Clear();
      return false;
    }
    return true;
  }

  template <typename T_>
  static handle from_cpp(T_&& src, rv_policy /*policy*/,
                         cleanup_list* /*cleanup*/) noexcept {
    bp::object bp_obj(std::forward<T_>(src));
    return handle(bp_obj.ptr()).inc_ref();
  }

  template <typename T_>
  static handle from_cpp(T_* src, rv_policy policy, cleanup_list* cleanup) {
    if (!src) return none().release();
    return from_cpp(*src, policy, cleanup);
  }
};

}  // namespace detail
}  // namespace nanobind

NANOBIND_BOOST_CASTER(pxr::SdfPath, "Sdf.Path");
NANOBIND_BOOST_CASTER(pxr::VtValue, "Vt.VtValue");
NANOBIND_BOOST_CASTER(pxr::UsdStagePtr, "Usd.Stage");
NANOBIND_BOOST_CASTER(pxr::UsdTimeCode, "Usd.TimeCode");
NANOBIND_BOOST_CASTER(pxr::TfToken, "Tf.Token");
