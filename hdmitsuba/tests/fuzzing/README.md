# Fuzzing tests

This folder contains AI-generated tests to *fuzz test* the Hydra delegate. The primary purpose of these tests is to catch Hydra state management bugs. For this, all these tests follow the same pattern:

1. Set up an initial USD test scene.
2. Render that scene using `hdMitsuba` and the `usd_mitsuba` Python translation, assert that the results match.
3. Modify the scene.
4. Render again using both code paths and assert the rendered images match. In this case, `hdMitsuba` will have perform *incremental* state updates, while `usd_mitsuba` will perform a clean translation.

The reason these tests were AI generated is to very broadly cover different cases of state transitions.

These tests are expensive to run and do not run by default when executing `pytest` at the project root. You can run them explicitly using `pytest hdmitsuba/tests/fuzzing/`.
