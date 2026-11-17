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

# Configures Python, OpenUSD, OpenSubdiv, and OpenImageDenoise targets when
# CMAKE_PREFIX_PATH points to a SideFX Houdini installation (Houdini 22.0+).

foreach(_prefix ${CMAKE_PREFIX_PATH})
  find_package(Houdini QUIET NO_DEFAULT_PATH PATHS
    "${_prefix}"
    "${_prefix}/toolkit/cmake"
    "${_prefix}/Frameworks/Houdini.framework/Resources/toolkit/cmake")
  if(Houdini_FOUND)
    break()
  endif()
endforeach()

if(Houdini_FOUND)
  message(STATUS "hdmitsuba: Building against Houdini ${Houdini_VERSION} at '${_houdini_hfs_root}'")
  set(Python_EXECUTABLE "${_python_binary}" CACHE FILEPATH "" FORCE)
  set(Python_INCLUDE_DIR "${_python_include_dir}" CACHE PATH "" FORCE)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

macro(hdmitsuba_setup_houdini)
  if(APPLE)
    get_filename_component(HOUDINI_LIB_DIR "${_houdini_hfs_root}/../Libraries" REALPATH)
    list(APPEND CMAKE_BUILD_RPATH
      "${_houdini_install_root}/Frameworks"
      "${_houdini_install_root}/Frameworks/Python.framework/Versions/${_houdini_python_version}")
  else()
    set(HOUDINI_LIB_DIR "${_houdini_hfs_root}/dsolib")
    list(APPEND CMAKE_BUILD_RPATH "${_houdini_install_root}/python/lib")
  endif()

  file(READ "${_houdini_include_dir}/pxr/pxr.h" _pxr_h)
  foreach(_v PXR_MAJOR_VERSION PXR_MINOR_VERSION PXR_PATCH_VERSION PXR_VERSION)
    if(_pxr_h MATCHES "#define[ \t]+${_v}[ \t]+([0-9]+)")
      set(${_v} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(PXR_CMAKE_DIR "${_houdini_hfs_root}/toolkit/cmake")
  set(pxr_DIR "${_houdini_hfs_root}")
  set(PXR_PYTHON_DIR "${Python_SITELIB}")

  find_library(HOUDINI_TBB_LIBRARY NAMES tbb PATHS "${HOUDINI_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(HOUDINI_TCMALLOC_LIBRARY NAMES tcmalloc_minimal PATHS "${HOUDINI_LIB_DIR}" NO_DEFAULT_PATH)

  foreach(_lib ar arch cameraUtil gf hd hdsi hdx pxOsd python sdf sdr tf trace
               usd usdGeom usdImaging usdLux usdRender usdShade usdSkel usdVol vt)
    find_library(_pxr_${_lib} NAMES "pxr_${_lib}" PATHS "${HOUDINI_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    add_library(${_lib} SHARED IMPORTED)
    set_target_properties(${_lib} PROPERTIES
      IMPORTED_LOCATION "${_pxr_${_lib}}"
      INTERFACE_INCLUDE_DIRECTORIES "${_houdini_include_dir};${_python_include_dir}"
      INTERFACE_COMPILE_DEFINITIONS "TBB_USE_DEBUG=0;$<$<PLATFORM_ID:Darwin>:MBSD>")
  endforeach()
  set_property(TARGET tf APPEND PROPERTY INTERFACE_LINK_LIBRARIES arch python "${HOUDINI_TBB_LIBRARY}")
  if(APPLE)
    set_property(TARGET tf APPEND PROPERTY INTERFACE_LINK_OPTIONS "LINKER:SHELL:-undefined dynamic_lookup")
  endif()
  set_property(TARGET hd APPEND PROPERTY INTERFACE_LINK_LIBRARIES cameraUtil gf tf trace vt "${HOUDINI_TBB_LIBRARY}")

  find_library(_osd_lib NAMES osdCPU PATHS "${HOUDINI_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  add_library(OpenSubdiv::osdCPU SHARED IMPORTED)
  set_target_properties(OpenSubdiv::osdCPU PROPERTIES
    IMPORTED_LOCATION "${_osd_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_houdini_include_dir}")

  find_library(_oidn_lib NAMES OpenImageDenoise PATHS "${HOUDINI_LIB_DIR}" NO_DEFAULT_PATH)
  if(_oidn_lib AND EXISTS "${_houdini_include_dir}/OpenImageDenoise/oidn.hpp")
    add_library(OpenImageDenoise SHARED IMPORTED)
    set_target_properties(OpenImageDenoise PROPERTIES
      IMPORTED_LOCATION "${_oidn_lib}"
      INTERFACE_INCLUDE_DIRECTORIES "${_houdini_include_dir}")
    set(OpenImageDenoise_FOUND TRUE)
  endif()

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/houdini/hdmitsuba.json.in
    ${CMAKE_BINARY_DIR}/hdmitsuba.json
    @ONLY)
endmacro()

macro(hdmitsuba_setup_houdini_nanobind)
  # Build nanobind_backend for Houdini's bundled Python so split-mode Mitsuba
  # Python bindings can be imported without modifying Houdini's site-packages.
  set(_nb_init "${MITSUBA_DIR}/ext/nanobind/nanobind-backend/src/nanobind_backend/__init__.py")
  if(COMMAND nanobind_add_backend AND EXISTS "${_nb_init}")
    nanobind_add_backend(_nb_backend_v1 NB_SUPPRESS_WARNINGS)
    set_target_properties(_nb_backend_v1 PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nanobind_backend")
    file(COPY "${_nb_init}" DESTINATION "${CMAKE_BINARY_DIR}/nanobind_backend")
  endif()
endmacro()
