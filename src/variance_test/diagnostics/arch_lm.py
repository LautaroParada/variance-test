"""ARCH LM diagnostics used by the weak-form battery."""

from __future__ import annotations

import numpy as np
from statsmodels.stats import diagnostic

from variance_test.models import TestOutcome


def _run_arch_lm(returns: np.ndarray, nlags: int, alpha: float) -> TestOutcome:
    """Run ARCH LM test on returns."""
    lm_stat, lm_p_value, f_stat, f_p_value = diagnostic.het_arch(returns, nlags=nlags)

    values = [lm_stat, lm_p_value, f_stat, f_p_value]
    if not all(np.isfinite(value) for value in values):
        return TestOutcome(
            name="arch_lm",
            null_hypothesis="No ARCH effects are present up to the tested lag order.",
            statistic=None,
            p_value=None,
            alpha=alpha,
            reject_null=None,
            metadata={
                "nlags": nlags,
                "lm_stat": None,
                "lm_p_value": None,
                "f_stat": None,
                "f_p_value": None,
                "reason": "ARCH LM statistics are not finite for this series.",
            },
            warnings=["arch_lm not computable: ARCH LM statistics are not finite."],
        )

    lm_stat = float(lm_stat)
    lm_p_value = float(lm_p_value)
    f_stat = float(f_stat)
    f_p_value = float(f_p_value)

    return TestOutcome(
        name="arch_lm",
        null_hypothesis="No ARCH effects are present up to the tested lag order.",
        statistic=lm_stat,
        p_value=lm_p_value,
        alpha=alpha,
        reject_null=bool(lm_p_value < alpha),
        metadata={
            "nlags": nlags,
            "lm_stat": lm_stat,
            "lm_p_value": lm_p_value,
            "f_stat": f_stat,
            "f_p_value": f_p_value,
        },
        warnings=[],
    )
