# Changelog

This file documents relevant project changes for public releases.

## [0.2.0] - 2026-04-09

### Added

- Opt-in robust variance-ratio family for weak-form battery execution.
- Automatic variance-ratio test (`avr`) with AR(1) plug-in horizon selection.
- Wild-bootstrap automatic variance-ratio test (`wbavr`) with deterministic seed support.
- Chow-Denning-style max test (`chow_denning`) using Sidak-normal familywise calibration.
- Internal diagnostics refactor under `variance_test.diagnostics` for extensible battery internals.
- New robust test suite coverage and robust example script.

### Changed

- `BatteryConfig` now accepts optional `robust_vr=RobustVRConfig(...)`.
- Robust diagnostics are disabled by default and run only when explicitly enabled.
- Full-sample v1 battery behavior remains unchanged when robust diagnostics are not enabled.

### Notes

- Rolling scope is unchanged: robust tests are full-sample only in this release.
- Structural-break and long-memory diagnostics are not included yet.
- `chow_denning` in this release is Sidak-normal calibrated and is not exact SMM calibration.

## [0.1.0] - 2026-03-15

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
