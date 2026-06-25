# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Static library for the engine wrapper
add_library(engine_lib STATIC
  engine.cc
)
target_include_directories(engine_lib PUBLIC
  ${HDMITSUBA_COMMON_INCLUDES}
)
target_link_libraries(engine_lib PUBLIC
  absl::flat_hash_map
  absl::log
  absl::check
  usd
  usdImaging
  usdRender
  hd
  hdx
)
# Make sure we build with PIC since it will be loaded by Python extension
set_target_properties(engine_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Nanobind module for Python bindings
nanobind_add_module(usd_render
  engine_ext.cc
)
target_link_libraries(usd_render PRIVATE
  engine_lib
)
set_target_properties(usd_render PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  INSTALL_RPATH "${PLATFORM_ORIGIN}/../../lib"
)

# --- Installation Rules ---
install(TARGETS usd_render
  LIBRARY DESTINATION lib/python
  ARCHIVE DESTINATION lib/python
  RUNTIME DESTINATION lib/python
)

