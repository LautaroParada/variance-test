"""Runs-test diagnostics used by the weak-form battery."""

from __future__ import annotations

import numpy as np
from scipy import stats

from variance_test.models import TestOutcome


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
