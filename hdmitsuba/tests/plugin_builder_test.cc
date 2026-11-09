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

#include "hdmitsuba/plugin_builder.h"

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/scene.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/pxr.h>

#include <stdexcept>
#include <string>

#include "hdmitsuba/tests/test_util.h"
#include <gtest/gtest.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

using Float = float;
using Spectrum = mitsuba::Color<Float, 3>;
using Scene = mitsuba::Scene<Float, Spectrum>;
using SamplingIntegrator = mitsuba::SamplingIntegrator<Float, Spectrum>;
using Emitter = mitsuba::Emitter<Float, Spectrum>;

static MitsubaStaticState<Scene> global_static_state;

TEST(PluginBuilderTest, ParseVtDictionaryAndInvalidSpecialKeys) {
  VtDictionary dict;
  dict["mitsuba:integrator:type"] = VtValue(std::string("aov"));
  dict["mitsuba:integrator:aovs"] = VtValue(std::string("dd.y:depth"));
  dict["mitsuba:integrator:sub:type"] = VtValue(std::string("path"));
  dict["mitsuba:integrator:sub:max_depth"] = VtValue(4);
  // Invalid non-string or empty `:type` / `:id` must not create ghost children.
  dict["mitsuba:integrator:ghost_int:type"] = VtValue(123);
  dict["mitsuba:integrator:ghost_empty:id"] = VtValue(std::string(""));
  dict["other:namespace:value"] = VtValue(99);

  PluginSpecNode spec = ParsePluginSpec(dict, "mitsuba:integrator:", "path");
  EXPECT_EQ(spec.plugin_type, "aov");
  EXPECT_TRUE(spec.HasProperty("aovs"));
  EXPECT_TRUE(spec.HasProperty("sub"));
  EXPECT_FALSE(spec.HasProperty("ghost_int"));
  EXPECT_FALSE(spec.HasProperty("ghost_empty"));
  ASSERT_EQ(spec.children.count("sub"), 1u);
  EXPECT_EQ(spec.children.at("sub").plugin_type, "path");
  EXPECT_EQ(spec.children.at("sub").params.at("max_depth"), VtValue(4));
}

TEST(PluginBuilderTest, ParseContainerDataSourceWithRoleTaggedVector) {
  HdDataSourceBaseHandle vector_container = HdRetainedContainerDataSource::New(
      TfToken("type"),
      HdRetainedTypedSampledDataSource<TfToken>::New(TfToken("vector")),
      TfToken("value"),
      HdRetainedTypedSampledDataSource<GfVec3f>::New(
          GfVec3f(1.0f, 2.0f, 3.0f)));
  HdContainerDataSourceHandle container = HdRetainedContainerDataSource::New(
      TfToken("mitsuba:sensor:type"),
      HdRetainedTypedSampledDataSource<std::string>::New("radiancemeter"),
      TfToken("mitsuba:sensor:origin"), vector_container,
      TfToken("mitsuba:sensor:color"),
      HdRetainedTypedSampledDataSource<GfVec3f>::New(
          GfVec3f(0.2f, 0.4f, 0.6f)));

  PluginSpecNode spec = ParsePluginSpec(container, "mitsuba:sensor:");
  EXPECT_EQ(spec.plugin_type, "radiancemeter");
  EXPECT_TRUE(spec.HasProperty("origin"));
  EXPECT_TRUE(spec.HasProperty("color"));

  mitsuba::Properties props = BuildPluginProperties(spec, "scalar_rgb");
  EXPECT_EQ(props.type("origin"), mitsuba::Properties::Type::Vector);
  const auto origin = props.get<mitsuba::Vector<double, 3>>("origin");
  EXPECT_DOUBLE_EQ(origin.x(), 1.0);
  EXPECT_DOUBLE_EQ(origin.y(), 2.0);
  EXPECT_DOUBLE_EQ(origin.z(), 3.0);
  EXPECT_EQ(props.type("color"), mitsuba::Properties::Type::Color);
}

TEST(PluginBuilderTest, SpecialChildrenAndPointEmitterNotHijacked) {
  VtDictionary dict;
  dict["mitsuba:emitter:type"] = VtValue(std::string("spot"));
  dict["mitsuba:emitter:rgb_val:type"] = VtValue(std::string("rgb"));
  dict["mitsuba:emitter:rgb_val:value"] = VtValue(GfVec3f(0.5f, 1.0f, 1.5f));
  dict["mitsuba:emitter:spec_int:type"] = VtValue(std::string("spectrum"));
  dict["mitsuba:emitter:spec_int:value"] = VtValue(2);
  dict["mitsuba:emitter:spec_pairs:type"] = VtValue(std::string("spectrum"));
  dict["mitsuba:emitter:spec_pairs:value"] = VtValue(
      VtArray<GfVec2f>{GfVec2f(400.0f, 1.0f), GfVec2f(700.0f, 2.0f)});
  dict["mitsuba:emitter:weights"] = VtValue(VtFloatArray{0.25f, 0.75f});
  // `type = "point"` is a Mitsuba emitter plugin, not a special value wrapper.
  dict["mitsuba:emitter:nested_light:type"] = VtValue(std::string("point"));

  PluginSpecNode spec = ParsePluginSpec(dict, "mitsuba:emitter:");
  mitsuba::Properties props = BuildPluginProperties(spec, "scalar_rgb");
  EXPECT_EQ(props.type("rgb_val"), mitsuba::Properties::Type::Color);
  EXPECT_EQ(props.type("spec_int"), mitsuba::Properties::Type::Spectrum);
  EXPECT_EQ(props.type("spec_pairs"), mitsuba::Properties::Type::Spectrum);
  EXPECT_EQ(props.get<std::string_view>("weights"), "0.25,0.75");
  EXPECT_EQ(props.type("nested_light"), mitsuba::Properties::Type::Object);
  mitsuba::ref<mitsuba::Object> nested =
      props.get<mitsuba::ref<mitsuba::Object>>("nested_light");
  EXPECT_NE(dynamic_cast<Emitter*>(nested.get()), nullptr);
}

TEST(PluginBuilderTest, BuildPluginObjectCheckedDynamicCast) {
  PluginSpecNode path_spec;
  path_spec.plugin_type = "path";
  mitsuba::ref<SamplingIntegrator> path_integrator =
      BuildPluginObject<SamplingIntegrator>(path_spec);
  EXPECT_NE(path_integrator.get(), nullptr);

  // `ptracer` is a valid Mitsuba Integrator (`AdjointIntegrator`), not a
  // `SamplingIntegrator`; `BuildPluginObject<SamplingIntegrator>` must throw.
  PluginSpecNode ptracer_spec;
  ptracer_spec.plugin_type = "ptracer";
  EXPECT_THROW(BuildPluginObject<SamplingIntegrator>(ptracer_spec),
               std::runtime_error);
}

}  // namespace

PXR_NAMESPACE_CLOSE_SCOPE
