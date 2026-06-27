# hdMitsuba - USD Hydra Delegate for Mitsuba 3

[![Linux CI](https://github.com/google/hdmitsuba/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/google/hdmitsuba/actions/workflows/ci-linux.yml)
[![macOS CI](https://github.com/google/hdmitsuba/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/google/hdmitsuba/actions/workflows/ci-macos.yml)
[![Windows CI](https://github.com/google/hdmitsuba/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/google/hdmitsuba/actions/workflows/ci-windows.yml)

`hdMitsuba` is a USD Hydra render delegate that enables [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) to render scenes using [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3). It allows integrating Mitsuba rendering into USD-compatible applications (e.g., usdview, Houdini, ...), or using it programmatically via Hydra.

<p align="center">
  <img src="docs/usdview_example.gif" alt="Mitsuba in usdview" width="640" />
  <br>
  <p align="center"><i>The video shows interactive rendering using Mitsuba in usdview on MacOS (using Metal GPU acceleration, and with experimental kernel freezing enabled).</i></p>
</p>

## Motivation

This Hydra delegate bridges the gap between rendering research and production pipelines by bringing Mitsuba 3 into the Pixar OpenUSD ecosystem. This allows to use the wide range of existing USD tools for scene assembly, while benefitting from various Mitsuba capabilities. USD provides a clean separation between renderer-agnostic scene assembly and final image rendering. Some of the key benefits of `hdMitsuba` are:

*   **Direct USD rendering**: Render USD stages directly using Mitsuba 3 without (inefficient) manual scene translation. This allows harnessing both USD-compatible tools as well as USD's APIs for authoring scenes that are then rendered in Mitsuba. Moreover, Mitsuba can be used as a viewport renderer in USD-native applications.
*   **Efficient incremental updates**: Leverage Hydra's *incremental* update mechanism. Only modified scene elements (e.g., camera motion, material edits) are passed to Mitsuba, enabling efficient interactive and sequence rendering.
*   **Advanced research features**: Exposes Mitsuba's unique capabilities, such as **spectral** and **polarized** rendering, and ease of research prototyping, to the USD ecosystem.
*   **Extensibility**: Mitsuba's modular design allows to easily prototype custom BSDFs or camera models (often in Python), and immediately use them within USD pipelines.


## Components

This repository provies the following components:

*   **`hdmitsuba`**: C++ implementation of the Hydra render delegate and its unit tests.
*   **`usd_mitsuba`**: A Python package for offline USD-to-Mitsuba scene translation. This code provides a reference implementation converting a USD scene to a Mitsuba scene dictionary. This is much slower than the Hydra delegate, but offers a very hackable and easy-to-understand code path. Morever, we use this code as a reference to test the hydra delegate.
*   **`render_engine`**: A simple USD render engine with Python bindings. This allows rendering using Hydra delegates and directly obtaining the result as NumPy arrays.
*   **`nanobind/usd.h`**: A generic Nanobind caster for USD types. This allows writing Nanobind-wrapped functions that interact with USD's Python bindings.


## Features & Documentation

`hdMitsuba` supports many features of both USD (subdivision surfaces, displacement maps, skeletal animation, instances, ...) and Mitsuba (native BSDFs, integrators, ...). For a detailed guide on supported features, custom render settings (variants, sample counts), and scene authoring examples, please refer to the [hdMitsuba User Guide](docs/user_guide.md). Please also see the extensive unit tests in `hdmitsuba/tests/` for usage examples.

## Building

### Prerequisites

1.  **Toolchain**: A modern C++17 compiler (e.g., Clang, GCC)
    *   **macOS**: Clang is required (Xcode toolchain).
    *   **Linux**: Either GCC or Clang work. If you install OpenUSD using `conda`, you need to use GCC for ABI compatibility.
2.  **Dependencies**:
    *   **OpenUSD**: For example [built locally](https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/BUILDING.md) or installed via `conda`. Due to the size and complexity of this dependency, we currently do not include it as a git submodule.
    *   **Mitsuba 3**: Locally built from source. Similarly to USD, it is not included as a submodule. For compatibility, it is recommended to use commit [`3f00b723`](https://github.com/mitsuba-renderer/mitsuba3/commit/3f00b72372a24d0811a56186f137a817c9174f1f) (which is the version validated in our CI configuration). The `pip`-installed version of Mitsuba currently is not supported.
    *   **Additional dependencies**: Additional C++ dependencies are included as git submoduls. When cloning this repository, use `git clone --recursive` to initialize submodules. Otherwise, they can also be initialized using `git submodule update --init --recursive`.
    *   **Python dependencies**: `numpy`, `pytest`.

### Build Steps

The build configuration has been tested on Linux and MacOS.
Configure and build using CMake and Ninja from the repository root:

```bash
cmake -G Ninja \
  -DCMAKE_PREFIX_PATH="/path/to/usd/installation/prefix" \
  -DMITSUBA_DIR="/path/to/mitsuba3/source" \
  -B build
cmake --build build
```

## Running Tests

To run the tests, you must configure the environment so that USD can find the `hdMitsuba` plugin, and Python can find the built modules and dependencies.

### Environment Setup

CMake generates a `setpath.sh` script in your build directory that automatically configures all required environment variables (`PYTHONPATH`, `LD_LIBRARY_PATH`, etc.) based on your build configuration.

Before running any tests, source this script from the repository root:

```bash
source build/setpath.sh
```

### Running Test Suites

#### 1. Python Tests (pytest)
We use `pytest` to run all Python-based tests (including high-level integration tests, Hydra delegate tests, and offline translator tests). Ensure you have sourced `setpath.sh` first, and then run:

```bash
# Run all Python tests
pytest

# Run the more extensive suite of state update tests (not enabled by default):
pytest hdmitsuba/tests/fuzzing
```

#### 2. C++ Unit Tests (ctest)
We use `ctest` (part of CMake) to run the C++ unit tests. These tests verify some of the internal Hydra delegate classes and geometry processing helpers.

Sourcing `setpath.sh` is generally not required for C++ tests as CMake configures build-tree RPATHs.

```bash
ctest --test-dir build
```

#### 3. Running with usdview
You can visualize USD stages interactively in Pixar's `usdview` using the Mitsuba delegate:

```bash
# Launch usdview with Mitsuba pre-selected as the renderer
usdview -r Mitsuba /path/to/scene.usd
```

Or, launch `usdview` normally and select **Mitsuba** from the **Renderer** menu in the top menu bar.

#### 4. Running the Render CLI
You can use the provided command-line script to render a USD stage using `hdMitsuba` programmatically:

```bash
# Render a scene using the default Mitsuba variant
python render_engine/render_scene_cli.py --scene /path/to/scene.usd

# Render with a specific Mitsuba variant, sample count, and output path
python render_engine/render_scene_cli.py --scene /path/to/scene.usd --variant llvm_ad_rgb --spp 32 --output output.png
```

**Key Features of the CLI:**
*   **Automatic Camera Framing**: If your USD stage has no authored cameras, the CLI automatically computes the scene's bounding box and creates a default camera (`/default_cam`) that frames the entire scene.
*   **Performance Profiling**: Pass the `--profile` flag to collect a detailed Chrome tracing JSON profile (saved as `<output_path>.trace.json`). You can load this file into [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing` to inspect and profile execution times (JIT tracing, scene sync, rendering).
*   **Dr.Jit Logging**: Pass `--drjit_log info` to print Dr.Jit's own log statements.

## Static Type Checking (Pyrefly)

We use `pyrefly` for static type checking of the Python code. A local `pyrefly.toml` configuration file is automatically generated by CMake in your project root.

To run the type checker, ensure your environment is set up and execute:

```bash
source build/setpath.sh
pyrefly check
```

*Note: Currently Pyrefly is configured to not check USD or Mitsuba types. It's a useful sanity check, but not a guarantee of overall code correctness. The rigor of the type checking may be expanded in the future.*


## Limitations and Future Work

While `hdMitsuba` already supports many Mitsuba & USD features, it currently has a number of known limitations. Some areas of future work are:

*   **Flexible plugin discovery**: The list of discoverable Mitsuba plugins is currently hardcoded in `hdmitsuba/sdr_discovery.cc`. Custom Mitsuba plugins will not be automatically discovered by USD applications without adding them to this list and rebuilding the delegate.
*   **Custom Emitter Support**: Currently, we only support standard USD emitters. In the near future, we will add support to specify custom Mitsuba textures on emitters (e.g., to set up spectral emission).
*   **Incremental Material Updates**: Materials are currently re-instantiated when modified. It would be useful to extend material support to also support efficient incremental updates.
*   **Performance**: There is room to further reduce overheads and improve performance. Note that `hdMitsuba` also inherits Mitsuba's performance characteristics (e.g., high JIT costs for scenes containing a very large number of materials).
*   **Volume rendering**: Volume rendering is not yet supported.

## Contributing
We welcome community contributions and bug reports. To contribute code, please see the instructions in [CONTRIBUTING](CONTRIBUTING).

## Citation

This project was created by [Delio Vicini](https://dvicini.github.io/), and significant improvements and fixes were contributed by [Philippe Weier](https://weiphil.github.io/portfolio/), [Alyssa Cheng](https://github.com/squarefish711), [Oscar Lin](https://github.com/dongclin) and [Rus Maxham](https://github.com/rrrus). We also acknowledge [Matt Estela](https://tokeru.com/cgwiki/), [Luca Prasso](https://www.lucaprasso.com), [Iker J. de los Mozos](linkedin.com/in/ikerj), [Stephan Garbin](http://stephangarbin.com/), [Bernd Bickel](https://www.berndbickel.com), [Vincent Huang](https://github.com/katsura0), and [Thabo Beeler](https://thabobeeler.com) for helpful discussions and support. If you use `hdMitsuba` in your research, please consider citing it using the following BibTeX entry:

```bibtex
@software{hdmitsuba,
	title   = {hdMitsuba: A Hydra Render Delegate for Mitsuba 3},
	author  = {Delio Vicini, Philippe Weier, Alyssa Cheng, Oscar Lin, Rus Maxham},
	year    = 2026,
	note    = {https://github.com/google/hdmitsuba},
	version = {0.1.0}
}
```

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
