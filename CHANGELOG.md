# Changelog

This file documents relevant project changes for public releases.

## [0.1.0] - YYYY-MM-DD

### Added

- Packaging structure under `src/`.
- Consolidated public API for variance testing, normalization, battery execution, simulation, and visuals.
- Support for `input_kind="log_prices"` and `input_kind="returns"` in the public variance-ratio workflow.
- Common input normalization utilities shared by test orchestration paths.
- Weak-form battery v1 with variance ratio multi-q, Holm family summary, Ljung-Box variants, runs test on signs, and ARCH LM.
- Offline examples under `examples/`.
- Minimal CI coverage for core quality checks.

### Changed

- Variance ratio implementation was hardened for robustness across edge cases.
- Non-computable scenarios are handled explicitly with structured outcomes and warnings instead of silent failures.
- Rolling support was introduced partially for variance ratio multi-q, Ljung-Box returns, and Ljung-Box squared returns.

### Notes

- Rolling is intentionally not applied in this version to Holm summary, runs test, or ARCH LM.
- Release `0.1.0` is documented as ready for maintainer-driven publication workflows.
