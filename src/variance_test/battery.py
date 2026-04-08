"""Weak-form efficiency battery orchestration utilities."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats import diagnostic

from .core import EMH
from .data import normalize_series
from .models import BatteryConfig, BatteryOutcome, TestOutcome


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


def _run_variance_ratio_family(normalized, config: BatteryConfig) -> dict[str, TestOutcome]:
    """Run multi-horizon variance ratio tests using EMH.vrt."""
    emh = EMH()
    outcomes: dict[str, TestOutcome] = {}

    for q in config.q_list:
        z_score, p_value = emh.vrt(
            X=normalized.log_prices,
            q=q,
            heteroskedastic=True,
            centered=True,
            unbiased=True,
            annualize=False,
            input_kind="log_prices",
            alternative="two-sided",
        )
        name = f"variance_ratio_q{q}"
        outcomes[name] = TestOutcome(
            name=name,
            null_hypothesis="Variance ratio equals 1 under the random walk null.",
            statistic=float(z_score),
            p_value=float(p_value),
            alpha=config.alpha,
            reject_null=bool(p_value < config.alpha),
            metadata={
                "q": q,
                "heteroskedastic": True,
                "centered": True,
                "unbiased": True,
                "annualize": False,
                "input_kind_used": "log_prices",
                "alternative": "two-sided",
            },
            warnings=[],
        )

    return outcomes


def _apply_holm_bonferroni(vr_outcomes: list[TestOutcome], alpha: float) -> dict[str, object]:
    """Apply Holm-Bonferroni correction to the variance-ratio family."""
    family = [outcome.name for outcome in vr_outcomes]
    raw_p_values = {outcome.name: float(outcome.p_value) for outcome in vr_outcomes}

    sorted_outcomes = sorted(vr_outcomes, key=lambda item: (float(item.p_value), item.name))
    m = len(sorted_outcomes)

    thresholds: dict[str, float] = {}
    rejections: dict[str, bool] = {name: False for name in family}

    stop = False
    for i, outcome in enumerate(sorted_outcomes):
        threshold = alpha / (m - i)
        thresholds[outcome.name] = float(threshold)

        if stop:
            rejections[outcome.name] = False
            continue

        if float(outcome.p_value) <= threshold:
            rejections[outcome.name] = True
        else:
            rejections[outcome.name] = False
            stop = True

    return {
        "method": "holm-bonferroni",
        "family": family,
        "raw_p_values": raw_p_values,
        "sorted_tests": [outcome.name for outcome in sorted_outcomes],
        "thresholds": thresholds,
        "rejections": rejections,
        "any_reject": any(rejections.values()),
    }


def _run_ljung_box(series: np.ndarray, lags: tuple[int, ...], alpha: float, name: str, null_hypothesis: str) -> TestOutcome:
    """Run Ljung-Box test and select the lag with minimum p-value."""
    lb_stat, lb_pvalue = diagnostic.acorr_ljungbox(series, lags=list(lags), return_df=False)

    per_lag: dict[int, dict[str, float | bool]] = {}
    selected_lag = int(lags[0])
    selected_p = float(lb_pvalue[0])

    for lag, stat_value, p_value in zip(lags, lb_stat, lb_pvalue):
        lag_int = int(lag)
        stat_float = float(stat_value)
        p_float = float(p_value)
        per_lag[lag_int] = {
            "statistic": stat_float,
            "p_value": p_float,
            "reject_null": bool(p_float < alpha),
        }

        if p_float < selected_p or (np.isclose(p_float, selected_p) and lag_int < selected_lag):
            selected_lag = lag_int
            selected_p = p_float

    selected_stat = float(per_lag[selected_lag]["statistic"])

    return TestOutcome(
        name=name,
        null_hypothesis=null_hypothesis,
        statistic=selected_stat,
        p_value=selected_p,
        alpha=alpha,
        reject_null=bool(selected_p < alpha),
        metadata={
            "tested_lags": list(lags),
            "selected_lag": selected_lag,
            "per_lag": per_lag,
        },
        warnings=[],
    )


def _run_runs_test(normalized, alpha: float) -> TestOutcome:
    """Run bilateral Wald-Wolfowitz runs test over non-zero return signs."""
    non_zero_returns = normalized.returns[normalized.returns != 0.0]
    signs = np.sign(non_zero_returns)

    n_effective = int(signs.size)
    n_positive = int(np.sum(signs > 0))
    n_negative = int(np.sum(signs < 0))

    if n_effective < 2:
        return TestOutcome(
            name="runs_test_signs",
            null_hypothesis="Signs of non-zero returns are random.",
            statistic=None,
            p_value=None,
            alpha=alpha,
            reject_null=None,
            metadata={
                "n_effective": n_effective,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "reason": "effective length is less than 2",
            },
            warnings=["Runs test not computable: effective length is less than 2."],
        )

    if n_positive == 0 or n_negative == 0:
        return TestOutcome(
            name="runs_test_signs",
            null_hypothesis="Signs of non-zero returns are random.",
            statistic=None,
            p_value=None,
            alpha=alpha,
            reject_null=None,
            metadata={
                "n_effective": n_effective,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "reason": "both positive and negative signs are required",
            },
            warnings=["Runs test not computable: both sign classes are required."],
        )

    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    expected_runs = 1 + (2 * n_positive * n_negative) / (n_positive + n_negative)
    variance_runs = (
        2
        * n_positive
        * n_negative
        * (2 * n_positive * n_negative - n_positive - n_negative)
        / (((n_positive + n_negative) ** 2) * (n_positive + n_negative - 1))
    )

    if variance_runs <= 0:
        return TestOutcome(
            name="runs_test_signs",
            null_hypothesis="Signs of non-zero returns are random.",
            statistic=None,
            p_value=None,
            alpha=alpha,
            reject_null=None,
            metadata={
                "n_effective": n_effective,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "reason": "theoretical runs variance is non-positive",
            },
            warnings=["Runs test not computable: theoretical variance is non-positive."],
        )

    z_score = (runs - expected_runs) / np.sqrt(variance_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return TestOutcome(
        name="runs_test_signs",
        null_hypothesis="Signs of non-zero returns are random.",
        statistic=float(z_score),
        p_value=float(p_value),
        alpha=alpha,
        reject_null=bool(p_value < alpha),
        metadata={
            "n_effective": n_effective,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "runs": runs,
            "expected_runs": float(expected_runs),
            "variance_runs": float(variance_runs),
        },
        warnings=[],
    )


def _run_arch_lm(returns: np.ndarray, nlags: int, alpha: float) -> TestOutcome:
    """Run ARCH LM test and return a standardized TestOutcome."""
    lm_stat, lm_p_value, f_stat, f_p_value = diagnostic.het_arch(returns, nlags=nlags)

    return TestOutcome(
        name="arch_lm",
        null_hypothesis="No ARCH effects are present up to the tested lag order.",
        statistic=float(lm_stat),
        p_value=float(lm_p_value),
        alpha=alpha,
        reject_null=bool(lm_p_value < alpha),
        metadata={
            "nlags": nlags,
            "lm_stat": float(lm_stat),
            "lm_p_value": float(lm_p_value),
            "f_stat": float(f_stat),
            "f_p_value": float(f_p_value),
        },
        warnings=[],
    )


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


def run_weak_form_battery(series, config: BatteryConfig | None = None) -> BatteryOutcome:
    """Run the weak-form efficiency battery v1 and return structured outcomes."""
    if config is None:
        config = BatteryConfig()

    normalized = normalize_series(series, input_kind=config.input_kind)
    _validate_battery_compatibility(normalized, config)

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

    multiple_testing = {
        "variance_ratio_holm": holm_summary,
        "battery_summary": _build_battery_summary(tests),
    }

    warnings: list[str] = []
    for outcome in tests.values():
        warnings.extend(outcome.warnings)

    return BatteryOutcome(
        input_kind=config.input_kind,
        n_obs=normalized.n_raw,
        returns_n_obs=normalized.n_returns,
        tests=tests,
        multiple_testing=multiple_testing,
        rolling=None,
        warnings=warnings,
    )


__all__ = ["run_weak_form_battery"]
