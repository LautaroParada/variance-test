"""Classic variance-ratio family diagnostics for battery orchestration."""

from __future__ import annotations

from variance_test.core import EMH
from variance_test.models import BatteryConfig, TestOutcome


def _run_variance_ratio_family(normalized, config: BatteryConfig) -> dict[str, TestOutcome]:
    """Run multi-horizon variance-ratio tests with classic heteroskedastic Z2."""
    emh = EMH()
    outcomes: dict[str, TestOutcome] = {}

    for q in config.q_list:
        name = f"variance_ratio_q{q}"

        try:
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

        except ValueError as exc:
            outcomes[name] = TestOutcome(
                name=name,
                null_hypothesis="Variance ratio equals 1 under the random walk null.",
                statistic=None,
                p_value=None,
                alpha=config.alpha,
                reject_null=None,
                metadata={
                    "q": q,
                    "heteroskedastic": True,
                    "centered": True,
                    "unbiased": True,
                    "annualize": False,
                    "input_kind_used": "log_prices",
                    "alternative": "two-sided",
                    "reason": str(exc),
                },
                warnings=[f"Variance ratio not computable for q={q}: {exc}"],
            )

    return outcomes


def _apply_holm_bonferroni(vr_outcomes: list[TestOutcome], alpha: float) -> dict[str, object]:
    """Apply Holm-Bonferroni correction to a variance-ratio test family."""
    family = [outcome.name for outcome in vr_outcomes]
    raw_p_values = {outcome.name: outcome.p_value for outcome in vr_outcomes}

    computable = [outcome for outcome in vr_outcomes if outcome.p_value is not None]
    sorted_outcomes = sorted(computable, key=lambda item: (float(item.p_value), item.name))
    m = len(sorted_outcomes)

    thresholds: dict[str, float | None] = {name: None for name in family}
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
