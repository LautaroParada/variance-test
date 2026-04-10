"""Ljung-Box diagnostics used by the weak-form battery."""

from __future__ import annotations

import numpy as np
from statsmodels.stats import diagnostic

from variance_test.models import TestOutcome


def _run_ljung_box(
    series: np.ndarray,
    lags: tuple[int, ...],
    alpha: float,
    name: str,
    null_hypothesis: str,
) -> TestOutcome:
    """Run Ljung-Box test and select the lag with minimum p-value."""
    result = diagnostic.acorr_ljungbox(series, lags=list(lags), return_df=True)

    lb_stat = result["lb_stat"].to_numpy(dtype=float)
    lb_pvalue = result["lb_pvalue"].to_numpy(dtype=float)

    per_lag: dict[int, dict[str, float | bool]] = {}

    finite_pairs = [
        (int(lag), float(stat_value), float(p_value))
        for lag, stat_value, p_value in zip(lags, lb_stat, lb_pvalue)
        if np.isfinite(stat_value) and np.isfinite(p_value)
    ]

    if not finite_pairs:
        return TestOutcome(
            name=name,
            null_hypothesis=null_hypothesis,
            statistic=None,
            p_value=None,
            alpha=alpha,
            reject_null=None,
            metadata={
                "tested_lags": list(lags),
                "selected_lag": None,
                "per_lag": {},
                "reason": "Ljung-Box statistics are not finite for this series.",
            },
            warnings=[f"{name} not computable: Ljung-Box statistics are not finite."],
        )

    selected_lag, selected_stat, selected_p = min(finite_pairs, key=lambda x: (x[2], x[0]))

    for lag, stat_value, p_value in finite_pairs:
        per_lag[int(lag)] = {
            "statistic": float(stat_value),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
        }

    return TestOutcome(
        name=name,
        null_hypothesis=null_hypothesis,
        statistic=float(selected_stat),
        p_value=float(selected_p),
        alpha=alpha,
        reject_null=bool(selected_p < alpha),
        metadata={
            "tested_lags": list(lags),
            "selected_lag": int(selected_lag),
            "per_lag": per_lag,
        },
        warnings=[],
    )
