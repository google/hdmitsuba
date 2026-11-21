# hdMitsuba agent guidelines

Guidelines for AI coding agents (and a summary of design conventions for
contributors). See [README.md](README.md) for build and test instructions and
[docs/user_guide.md](docs/user_guide.md) for the supported features.

## Design

*   **Parity is a hard requirement.** The Hydra delegate (`hdmitsuba/`) and the
    offline translator (`usd_mitsuba/`) must produce the same result: every
    feature or fix lands in both.
*   **Follow the USD specification** (e.g., UsdPreviewSurface) rather than
    heuristics. Matching the specification takes precedence over preserving the
    previous appearance.
*   **Check for existing mechanisms first.** Before adding a `mitsuba:*` render
    setting or primvar, check whether USD or Hydra already provides one (e.g.,
    display style, complexity, standard schemas) and what other renderers do.
    Do not add ad-hoc thresholds or policies.
*   Read prim data through scene index schemas and data sources (e.g.,
    `HdPrimvarsSchema`) instead of legacy `HdSceneDelegate` calls.
*   No process-global state.
*   Avoid defensive programming: only handle conditions that can actually
    occur.
*   **Match Mitsuba and Dr.Jit conventions in custom plugins.** Follow Mitsuba's
    built-in plugin style (`UnpolarizedSpectrum` internally with
    `depolarizer<Spectrum>` at the boundary, `snake_case` locals, plain float
    literals like `1.0f`), use `dr::any_or`/`dr::none_or` instead of Dr.Jit
    internals, and reuse Dr.Jit/Mitsuba math, frame, and epsilon helpers
    (`dr::Epsilon<Float>`, `Frame3f`).

## Tests

*   Prefer an on-disk `.usda` scene under `hdmitsuba/test_assets/`, added to
    the scene list of an existing `test_render` test (`shape_test.py`,
    `material_test.py`, `light_test.py`), which compares the Hydra render to
    the offline translator.
*   Write in-memory tests only for dynamic behavior that a static scene cannot
    express (e.g., changing a setting between renders), and keep them short.
*   Use tolerances when comparing floating point values.
*   For custom BSDF tests, reuse `mi.chi2.ChiSquareTest` and Mitsuba/Dr.Jit
    helpers (`dr.allclose`, `mi.has_flag`, `mi.PCG32`).
*   Check that a new test fails without the change it covers.

## Workflow

*   Build with `cmake --build build`, then `source build/setpath.sh`.
*   Before declaring a change done, run all of the following (CI enforces
    them):
    *   `pytest`
    *   `ctest --test-dir build`
    *   `pyrefly check` (must report 0 errors)
    *   `pyink --check .` and `isort --check-only .` (format Python code with
        `pyink .` and `isort .`)
    *   `git ls-files '*.cc' '*.h' | xargs clang-format --dry-run --Werror`
        (format C++ code with `clang-format -i`)
*   Keep diffs minimal: do not touch unrelated code or comments.
*   Keep the history of a branch clean: no "fix up" commits, amend follow-up
    changes into the commit they belong to.
*   Update `docs/user_guide.md` for user-visible changes.
