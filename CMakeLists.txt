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

cmake_minimum_required(VERSION 3.15)
project(hdmitsuba VERSION 0.1.0 LANGUAGES CXX)

# --- Mitsuba 3 Directory Config ---
set(MITSUBA_DIR "" CACHE PATH "Path to Mitsuba 3 source directory")
set(MITSUBA_BUILD_DIR "" CACHE PATH "Path to Mitsuba 3 build directory (defaults to MITSUBA_DIR/build)")
set(EMBREE_LIB_DIR "" CACHE PATH "Path to Embree library directory (if needed at runtime)")

if(NOT MITSUBA_DIR)
  message(FATAL_ERROR "Please set MITSUBA_DIR to your Mitsuba 3 source directory.")
endif()

if(NOT MITSUBA_BUILD_DIR)
  set(MITSUBA_BUILD_DIR "${MITSUBA_DIR}/build")
endif()

# --- Default Build Configurations ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/inst" CACHE PATH "Install path prefix" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
  set(PLATFORM_ORIGIN "@loader_path")
else()
  set(PLATFORM_ORIGIN "$ORIGIN")
endif()

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Werror=unused-variable -Werror=unused-but-set-variable -Wno-deprecated)
endif()

if(NOT EXISTS "${MITSUBA_DIR}/ext/drjit/ext/drjit-core/ext/nanothread/ext/cmake-defaults/CMakeLists.txt")
  message(FATAL_ERROR "cmake-defaults not found in Mitsuba's nanothread. Please ensure Mitsuba submodules are updated.")
endif()
include(${MITSUBA_DIR}/ext/drjit/ext/drjit-core/ext/nanothread/ext/cmake-defaults/CMakeLists.txt)
set(CMAKE_CXX_VISIBILITY_PRESET default)

# Enable AddressSanitizer for debugging (disabled by default)
option(HDMITSUBA_ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(HDMITSUBA_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# --- Dependencies Config ---

# 1. Abseil (built from submodule as shared libraries to prevent dynamic symbol duplication)
message(STATUS "hdmitsuba: Building Abseil from submodule as shared libraries...")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp/CMakeLists.txt")
  message(FATAL_ERROR "Abseil submodule is missing. Please run: git submodule update --init --recursive")
endif()
set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "Use project C++ standard")
set(ABSL_ENABLE_INSTALL ON CACHE INTERNAL "Enable install rule")

set(BUILD_SHARED_LIBS ON)
add_subdirectory(third_party/abseil-cpp)
unset(BUILD_SHARED_LIBS)

# 2. OpenUSD (PXR)
find_package(OpenGL REQUIRED)
find_package(pxr REQUIRED)
# Note: User must set CMAKE_PREFIX_PATH to their USD build directory.

# 3. OpenSubdiv
find_package(OpenSubdiv REQUIRED)

# 4. Mitsuba 3 & DrJit

# Find Mitsuba 3 libraries in the build directory
find_library(MITSUBA_LIBRARY
  NAMES mitsuba
  PATHS ${MITSUBA_BUILD_DIR}
  PATH_SUFFIXES "" "dist"
  NO_DEFAULT_PATH
)
if(NOT MITSUBA_LIBRARY)
  message(FATAL_ERROR "Could not find libmitsuba in ${MITSUBA_BUILD_DIR}")
endif()

find_library(DRJIT_CORE_LIBRARY
  NAMES drjit-core
  PATHS ${MITSUBA_BUILD_DIR}
  PATH_SUFFIXES "" "dist" "ext/drjit" "ext/drjit/drjit-core" "ext_build/drjit" "ext_build/drjit/drjit-core"
  NO_DEFAULT_PATH
)
if(NOT DRJIT_CORE_LIBRARY)
  message(FATAL_ERROR "Could not find libdrjit-core in ${MITSUBA_BUILD_DIR}")
endif()

find_library(DRJIT_EXTRA_LIBRARY
  NAMES drjit-extra
  PATHS ${MITSUBA_BUILD_DIR}
  PATH_SUFFIXES "" "dist" "ext/drjit" "ext/drjit/drjit-extra" "ext_build/drjit" "ext_build/drjit/drjit-extra"
  NO_DEFAULT_PATH
)


find_library(NANOTHREAD_LIBRARY
  NAMES nanothread
  PATHS ${MITSUBA_BUILD_DIR}
  PATH_SUFFIXES "" "dist" "ext/drjit" "ext/drjit/nanothread" "ext_build/drjit" "ext_build/drjit/nanothread"
  NO_DEFAULT_PATH
)
if(NOT NANOTHREAD_LIBRARY)
  message(FATAL_ERROR "Could not find libnanothread in ${MITSUBA_BUILD_DIR}")
endif()

# 5. Python & Nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Match mitsuba3 build configuration to ensure ABI compatibility by querying mitsuba via Python
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${MITSUBA_BUILD_DIR}/python"
          "LD_LIBRARY_PATH=${MITSUBA_BUILD_DIR}"
          "DYLD_LIBRARY_PATH=${MITSUBA_BUILD_DIR}"
          ${Python_EXECUTABLE} -c "import mitsuba; print(';'.join(mitsuba.variants()))"
  OUTPUT_VARIABLE MITSUBA_VARIANTS_LIST
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE PYTHON_ERRORS
  RESULT_VARIABLE PYTHON_RESULT
)

if(NOT PYTHON_RESULT EQUAL 0)
  message(WARNING "Could not query mitsuba variants using Python (Result: ${PYTHON_RESULT}, Error: ${PYTHON_ERRORS}). Using fallback compile definitions.")
  add_compile_definitions(
    MI_ENABLE_LLVM=1
    MI_ENABLE_JIT=1
    MI_ENABLE_EMBREE=1
  )
  if(APPLE)
    add_compile_definitions(MI_ENABLE_METAL=1)
  endif()
else()
  set(HAS_JIT_VARIANT FALSE)
  if(MITSUBA_VARIANTS_LIST MATCHES "cuda_")
    message(STATUS "hdmitsuba: Mitsuba 3 has CUDA enabled, enabling CUDA.")
    add_compile_definitions(MI_ENABLE_CUDA=1)
    set(HAS_JIT_VARIANT TRUE)
  endif()
  if(MITSUBA_VARIANTS_LIST MATCHES "llvm_")
    message(STATUS "hdmitsuba: Mitsuba 3 has LLVM enabled, enabling LLVM.")
    add_compile_definitions(MI_ENABLE_LLVM=1)
    set(HAS_JIT_VARIANT TRUE)
  endif()
  if(MITSUBA_VARIANTS_LIST MATCHES "metal_")
    message(STATUS "hdmitsuba: Mitsuba 3 has Metal enabled, enabling Metal.")
    add_compile_definitions(MI_ENABLE_METAL=1)
    set(HAS_JIT_VARIANT TRUE)
  endif()
  if(HAS_JIT_VARIANT)
    add_compile_definitions(MI_ENABLE_JIT=1)
  endif()
  # Embree is typically enabled in mitsuba3
  add_compile_definitions(MI_ENABLE_EMBREE=1)
endif()

# Determine if Mitsuba has AD variants enabled
set(MITSUBA_HAS_AD FALSE)
if(PYTHON_RESULT EQUAL 0)
  if(MITSUBA_VARIANTS_LIST MATCHES "ad_")
    set(MITSUBA_HAS_AD TRUE)
  endif()
else()
  # Fallback: assume AD is enabled to be safe
  set(MITSUBA_HAS_AD TRUE)
endif()

if(MITSUBA_HAS_AD)
  if(NOT DRJIT_EXTRA_LIBRARY)
    message(FATAL_ERROR "Mitsuba has AD variants enabled (or fallback active), but libdrjit-extra could not be found in ${MITSUBA_BUILD_DIR}")
  endif()
  message(STATUS "hdmitsuba: Mitsuba has AD variants enabled, linking drjit-extra.")
endif()


# Add Nanobind (reusing the one from Mitsuba 3 to ensure ABI compatibility)
if(NOT EXISTS "${MITSUBA_DIR}/ext/nanobind/CMakeLists.txt")
  message(FATAL_ERROR "Nanobind not found in Mitsuba directory: ${MITSUBA_DIR}/ext/nanobind. Please ensure MITSUBA_DIR is correct.")
endif()
add_subdirectory(${MITSUBA_DIR}/ext/nanobind ${CMAKE_CURRENT_BINARY_DIR}/nanobind)

# 6. GoogleTest (for C++ unit tests)
option(HDMITSUBA_BUILD_TESTS "Build C++ unit tests" ON)
if(HDMITSUBA_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# --- Include Directories ---
# We define a common set of includes for our targets
set(HDMITSUBA_COMMON_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR} # For nanobind/usd.h, engine.h
  ${MITSUBA_DIR}/include
  ${MITSUBA_BUILD_DIR}/include
  ${MITSUBA_DIR}/ext/nanobind/include
  ${MITSUBA_DIR}/ext/nanobind/ext/robin_map/include
  ${MITSUBA_DIR}/ext/tinyformat
  ${MITSUBA_DIR}/ext/asmjit/src
  ${MITSUBA_DIR}/ext/drjit/include
  ${MITSUBA_DIR}/ext/drjit/ext/drjit-core/include
  ${MITSUBA_DIR}/ext/drjit/ext/drjit-core/ext/nanothread/include
  ${MITSUBA_DIR}/ext/embree/include
  ${MITSUBA_DIR}/ext/struct-jit/include
)

# --- Subdirectories ---
add_subdirectory(hdmitsuba)

# --- Main Engine Library & Python Extension (engine) ---
add_subdirectory(render_engine)

# Generate setpath.sh for easy environment setup
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/setpath.sh.in
  ${CMAKE_BINARY_DIR}/setpath.sh
  @ONLY
)

# Generate pyrefly.toml for local type checking
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/pyrefly.toml.in
  ${CMAKE_CURRENT_SOURCE_DIR}/pyrefly.toml
  @ONLY
)

# --- Installation Rules ---
install(DIRECTORY usd_mitsuba
  DESTINATION lib/python
  PATTERN "*.py"
)

install(PROGRAMS render_engine/render_scene_cli.py
  DESTINATION bin
  RENAME render_scene_cli
)
