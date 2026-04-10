"""Weak-form efficiency battery orchestration utilities."""

from __future__ import annotations

from .data import normalize_series
from .diagnostics import (
    _apply_holm_bonferroni,
    _run_arch_lm,
    _run_ljung_box,
    _run_runs_test,
    _run_variance_ratio_family,
    make_non_computable_robust_outcome,
    run_avr,
    run_chow_denning,
    run_wbavr,
)
from .models import BatteryConfig, BatteryOutcome, TestOutcome
from .rolling import _build_rolling_results


def _validate_battery_compatibility(normalized, config: BatteryConfig) -> None:
    """Validate that configured horizons/lags are compatible with the input length."""
    n_log_prices = int(normalized.log_prices.size)
    for q in config.q_list:
        if q >= n_log_prices:
            raise ValueError("Each q must satisfy q < normalized.log_prices.size.")

    max_lag = max(config.ljung_box_lags)
    if max_lag >= normalized.n_returns:
        raise ValueError("max(config.ljung_box_lags) must satisfy < normalized.n_returns.")

    if config.arch_lm_lags >= normalized.n_returns:
        raise ValueError("config.arch_lm_lags must satisfy < normalized.n_returns.")


def _build_battery_summary(tests: dict[str, TestOutcome]) -> dict[str, object]:
    """Build the final battery-level summary payload."""
    rejected_tests = [name for name, outcome in tests.items() if outcome.reject_null is True]

    runs_outcome = tests.get("runs_test_signs")
    sign_rejected = bool(runs_outcome.reject_null is True) if runs_outcome is not None else False

    mean_rejected = bool(
        tests["variance_ratio_holm"].reject_null or tests["ljung_box_returns"].reject_null
    )
    volatility_rejected = bool(
        tests["ljung_box_squared_returns"].reject_null or tests["arch_lm"].reject_null
    )

    weak_form_evidence_against_null = bool(
        mean_rejected or sign_rejected or volatility_rejected
    )

    return {
        "rejected_tests": rejected_tests,
        "n_rejections": len(rejected_tests),
        "mean_dependence_rejected": mean_rejected,
        "sign_dependence_rejected": sign_rejected,
        "volatility_dependence_rejected": volatility_rejected,
        "weak_form_evidence_against_null": weak_form_evidence_against_null,
    }


def _append_robust_vr_tests(
    tests: dict[str, TestOutcome],
    normalized,
    config: BatteryConfig,
) -> dict[str, object]:
    """Run robust VR tests if enabled and return robust summary payload."""
    robust = config.robust_vr
    if robust is None or robust.enabled is False:
        return {}

    robust_test_names: list[str] = []

    robust_specs: tuple[tuple[bool, str, str, callable], ...] = (
        (
            robust.enable_avr,
            "avr",
            "The return process is compatible with the automatic variance-ratio random walk null.",
            run_avr,
        ),
        (
            robust.enable_wbavr,
            "wbavr",
            "The return process is compatible with the wild-bootstrap automatic variance-ratio random walk null.",
            run_wbavr,
        ),
        (
            robust.enable_chow_denning,
            "chow_denning",
            "All configured variance-ratio horizons are jointly compatible with the random walk null.",
            run_chow_denning,
        ),
    )

    for enabled, name, null_hypothesis, runner in robust_specs:
        if not enabled:
            continue
        robust_test_names.append(name)
        try:
            tests[name] = runner(normalized, config)
        except ValueError as exc:
            tests[name] = make_non_computable_robust_outcome(
                name=name,
                null_hypothesis=null_hypothesis,
                alpha=config.alpha,
                reason=f"{name} not computable: {exc}",
            )

    rejected = [name for name in robust_test_names if tests[name].reject_null is True]
    return {
        "tests_included": robust_test_names,
        "rejected_tests": rejected,
        "n_rejections": len(rejected),
        "any_reject": bool(rejected),
    }


def run_weak_form_battery(
    series,
    config: BatteryConfig | None = None,
) -> BatteryOutcome:
    """Run the weak-form efficiency battery and return structured outcomes."""
    if config is None:
        config = BatteryConfig()

    normalized = normalize_series(series, input_kind=config.input_kind)
    _validate_battery_compatibility(normalized, config)

    if config.rolling_window is not None and config.rolling_window > normalized.n_raw:
        raise ValueError("config.rolling_window must satisfy <= normalized.n_raw.")

    tests: dict[str, TestOutcome] = {}

    vr_family = _run_variance_ratio_family(normalized, config)
    tests.update(vr_family)

    vr_list = [tests[f"variance_ratio_q{q}"] for q in config.q_list]
    holm_summary = _apply_holm_bonferroni(vr_list, config.alpha)
    tests["variance_ratio_holm"] = TestOutcome(
        name="variance_ratio_holm",
        null_hypothesis="No variance-ratio horizon rejects after Holm-Bonferroni correction.",
        statistic=None,
        p_value=None,
        alpha=config.alpha,
        reject_null=bool(holm_summary["any_reject"]),
        metadata=holm_summary,
        warnings=[],
    )

    tests["ljung_box_returns"] = _run_ljung_box(
        series=normalized.returns,
        lags=config.ljung_box_lags,
        alpha=config.alpha,
        name="ljung_box_returns",
        null_hypothesis="Returns are serially independent up to the tested lags.",
    )

    tests["ljung_box_squared_returns"] = _run_ljung_box(
        series=normalized.squared_returns,
        lags=config.ljung_box_lags,
        alpha=config.alpha,
        name="ljung_box_squared_returns",
        null_hypothesis="Squared returns are serially independent up to the tested lags.",
    )

    if config.runs_test is True:
        tests["runs_test_signs"] = _run_runs_test(normalized=normalized, alpha=config.alpha)

    tests["arch_lm"] = _run_arch_lm(
        returns=normalized.returns,
        nlags=config.arch_lm_lags,
        alpha=config.alpha,
    )

    robust_summary = _append_robust_vr_tests(tests=tests, normalized=normalized, config=config)

    multiple_testing = {
        "variance_ratio_holm": holm_summary,
        "battery_summary": _build_battery_summary(tests),
    }
    if robust_summary:
        multiple_testing["robust_vr_summary"] = robust_summary

    warnings: list[str] = []
    for outcome in tests.values():
        warnings.extend(outcome.warnings)

    rolling = None
    if config.rolling_window is not None:
        rolling = _build_rolling_results(normalized=normalized, config=config)

    return BatteryOutcome(
        input_kind=config.input_kind,
        n_obs=normalized.n_raw,
        returns_n_obs=normalized.n_returns,
        tests=tests,
        multiple_testing=multiple_testing,
        rolling=rolling,
        warnings=warnings,
    )


__all__ = ["run_weak_form_battery"]
