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

#include <string>
#include <vector>

#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdr/declare.h>
#include <pxr/usd/sdr/discoveryPlugin.h>
#include <pxr/usd/sdr/shaderNodeDiscoveryResult.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (shader)(mitsuba));

class SdrMitsubaDiscoveryPlugin : public SdrDiscoveryPlugin {
 public:
  SdrMitsubaDiscoveryPlugin() = default;
  ~SdrMitsubaDiscoveryPlugin() override = default;

  SdrShaderNodeDiscoveryResultVec DiscoverShaderNodes(
      const SdrDiscoveryPluginContext& /*context*/) override {
    TfToken filename("<built-in>");
    SdrShaderNodeDiscoveryResultVec ret;

    // TODO: Dynamically discover all Mitsuba plugins.
    static const std::vector<std::string>* nodes = new std::vector<std::string>{
        "blendbsdf",
        "bumpmap",
        "conductor",
        "dielectric",
        "diffuse",
        "hair",
        "mask",
        "measured",
        "normalmap",
        "null",
        "plastic",
        "roughconductor",
        "roughdielectric",
        "roughplastic",
        "thindielectric",
        "twosided",
        "polarizer",
        "retarder",
        "circular",
        "measured_polarized",
        "pplastic",
        "principled",
        "principledthin",
        "blackbody",
        "uniform",
        "regular",
        "irregular",
        "d65",
        "rawconstant",
        "srgb",
        "bitmap",
        "checkerboard",
        "mesh_attribute",
        "volume",
    };
    ret.reserve(nodes->size());
    for (const auto& node : *nodes) {
      std::string mitsuba_id = "mitsuba_" + node;
      ret.push_back(SdrShaderNodeDiscoveryResult(
          SdrIdentifier(mitsuba_id), SdrVersion(1, 0), node, _tokens->shader,
          _tokens->mitsuba, _tokens->mitsuba, filename, filename));
    }
    return ret;
  }

  const SdrStringVec& GetSearchURIs() const override {
    static const SdrStringVec* result = new SdrStringVec{"<built-in>"};
    return *result;
  }
};

SDR_REGISTER_DISCOVERY_PLUGIN(SdrMitsubaDiscoveryPlugin)

PXR_NAMESPACE_CLOSE_SCOPE
