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

#include <memory>
#include <utility>

#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdr/declare.h>
#include <pxr/usd/sdr/parserPlugin.h>
#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/sdr/shaderNodeDiscoveryResult.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (mitsuba)(shader));

class SdrMitsubaParserPlugin : public SdrParserPlugin {
 public:
  SdrMitsubaParserPlugin() = default;
  ~SdrMitsubaParserPlugin() override = default;

  SdrShaderNodeUniquePtr ParseShaderNode(
      const SdrShaderNodeDiscoveryResult& discoveryResult) override {
    SdrShaderPropertyUniquePtrVec properties;
    return std::make_unique<SdrShaderNode>(
        discoveryResult.identifier, discoveryResult.version,
        discoveryResult.name, discoveryResult.family, TfToken(),
        discoveryResult.sourceType, discoveryResult.uri, discoveryResult.uri,
        std::move(properties));
  }

  const SdrTokenVec& GetDiscoveryTypes() const override {
    static const SdrTokenVec* ret = new SdrTokenVec{_tokens->mitsuba};
    return *ret;
  }

  const TfToken& GetSourceType() const override { return _tokens->mitsuba; }
};

SDR_REGISTER_PARSER_PLUGIN(SdrMitsubaParserPlugin)

PXR_NAMESPACE_CLOSE_SCOPE
