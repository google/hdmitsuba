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

# hdmitsuba/CMakeLists.txt

# Define the Hydra delegate target
add_library(hdMitsuba SHARED
  material.cc
  camera.cc
  config.cc
  debug_codes.cc
  curves.cc
  light.cc
  instancer.cc
  mesh.cc
  mesh/geometry_processor.cc
  mesh/subdivision.cc
  sdr_discovery.cc
  sdr_parser.cc
  render_buffer.cc
  render_delegate.cc
  render_pass.cc
  render_settings.cc
  renderer_plugin.cc
  scene_manager.cc
  traversal.cc
  prim_translator.cc
)

target_include_directories(hdMitsuba PUBLIC
  ${HDMITSUBA_COMMON_INCLUDES}
)

# Link dependencies
# We try to use targets, fallback to variables if needed (especially for OpenSubdiv)
target_link_libraries(hdMitsuba
  PUBLIC
    absl::base
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
    absl::strings
    absl::time
    arch
    gf
    hd
    hdx
    pxOsd
    sdf
    tf
    usdGeom
    vt
  PRIVATE
    absl::log
    absl::check
    absl::synchronization
    sdr
    ${MITSUBA_LIBRARY}
    ${DRJIT_CORE_LIBRARY}
    ${NANOTHREAD_LIBRARY}
)

if(MITSUBA_HAS_AD)
  target_link_libraries(hdMitsuba PRIVATE ${DRJIT_EXTRA_LIBRARY})
endif()

# OpenSubdiv handling
# Try to use standard CMake targets if they exist (usually defined by modern FindOpenSubdiv)
if(TARGET OpenSubdiv::osdCPU)
  target_link_libraries(hdMitsuba PRIVATE OpenSubdiv::osdCPU)
elseif(OPENSUBDIV_LIBRARIES)
  target_link_libraries(hdMitsuba PRIVATE ${OPENSUBDIV_LIBRARIES})
  if(OPENSUBDIV_INCLUDE_DIR)
    target_include_directories(hdMitsuba PRIVATE ${OPENSUBDIV_INCLUDE_DIR})
  endif()
else()
  # Fallback: assume it might be linked via USD or try to link it directly if target is just 'opensubdiv'
  find_library(OPENSUBDIV_FAR_LIB NAMES osdCPU Far PATHS /usr/lib /usr/local/lib)
  if(OPENSUBDIV_FAR_LIB)
    target_link_libraries(hdMitsuba PRIVATE ${OPENSUBDIV_FAR_LIB})
  else()
    message(WARNING "OpenSubdiv library not found explicitly, relying on transitive linking or system paths.")
  endif()
endif()

# We need -rdynamic for mitsuba plugins to resolve symbols from the host binary
if(NOT MSVC)
  target_link_options(hdMitsuba PRIVATE "-rdynamic")
endif()
set_target_properties(hdMitsuba PROPERTIES
  INSTALL_RPATH "${PLATFORM_ORIGIN}/../../lib"
)


# Enable CUDA if needed
option(HDMITSUBA_ENABLE_CUDA "Enable CUDA support for Mitsuba" OFF)
if(HDMITSUBA_ENABLE_CUDA)
  target_compile_definitions(hdMitsuba PRIVATE MI_ENABLE_CUDA)
endif()

# Copy plugInfo.json and resources to the build directory for runtime loading
set(PLUG_INFO_LIBRARY_PATH "libhdMitsuba${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(plugInfo.json.in plugInfo.json @ONLY)
file(COPY resources/curve.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resources)

# --- Geometry Nanobind Module ---
nanobind_add_module(geometry_ext
  mesh/geometry_ext.cc
)
target_include_directories(geometry_ext PRIVATE
  ${HDMITSUBA_COMMON_INCLUDES}
)
target_link_libraries(geometry_ext PRIVATE
  hdMitsuba
  usd
  usdSkel
)
set_target_properties(geometry_ext PROPERTIES
  INSTALL_RPATH "${PLATFORM_ORIGIN}/../../../plugin/usd;${PLATFORM_ORIGIN}/../../../lib"
)

# --- C++ Unit Tests ---
if(HDMITSUBA_BUILD_TESTS)
  # Helper function to define a test
  function(add_hdmitsuba_test Name Source)
    add_executable(${Name} ${Source})
    target_include_directories(${Name} PRIVATE
      ${HDMITSUBA_COMMON_INCLUDES}
    )
    target_link_libraries(${Name} PRIVATE
      hdMitsuba
      gtest_main
      gmock
      usdImaging
      usd
      ${MITSUBA_LIBRARY}
      ${DRJIT_CORE_LIBRARY}
      ${NANOTHREAD_LIBRARY}
    )
    if(MITSUBA_HAS_AD)
      target_link_libraries(${Name} PRIVATE ${DRJIT_EXTRA_LIBRARY})
    endif()
    add_test(NAME ${Name} COMMAND ${Name})
    set_tests_properties(${Name} PROPERTIES ENVIRONMENT "HDMITSUBA_VARIANT=scalar_rgb")
  endfunction()

  add_hdmitsuba_test(camera_test tests/camera_test.cc)
  add_hdmitsuba_test(mesh_test tests/mesh_test.cc)
  add_hdmitsuba_test(geometry_processor_test mesh/tests/geometry_processor_test.cc)
  add_hdmitsuba_test(subdivision_test mesh/tests/subdivision_test.cc)
endif()

# --- Installation Rules ---
install(TARGETS hdMitsuba
  LIBRARY DESTINATION plugin/usd
  ARCHIVE DESTINATION plugin/usd
  RUNTIME DESTINATION plugin/usd
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json
  DESTINATION plugin/usd
)

install(DIRECTORY resources
  DESTINATION plugin/usd
)

install(TARGETS geometry_ext
  LIBRARY DESTINATION lib/python/hdmitsuba
  ARCHIVE DESTINATION lib/python/hdmitsuba
  RUNTIME DESTINATION lib/python/hdmitsuba
)

