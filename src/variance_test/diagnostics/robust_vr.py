"""Robust variance-ratio diagnostics for weak-form battery v0.2.0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from variance_test.models import BatteryConfig, RobustVRConfig, TestOutcome


def _compute_vr_value_from_log_prices(log_prices: np.ndarray, q: int) -> float:
    """Compute centered variance-ratio value at horizon ``q`` from log-prices."""
    prices = np.asarray(log_prices, dtype=float)
    n_obs = prices.size
    if q <= 0:
        raise ValueError("Aggregation horizon q must be a positive integer.")
    if n_obs <= q:
        raise ValueError("Not enough observations for the requested aggregation horizon.")

    mu_est = (prices[-1] - prices[0]) / n_obs

    one_period_diffs = prices[1:] - prices[:-1] - mu_est
    n_one = one_period_diffs.size
    denom_b = n_one - 1
    if denom_b <= 0:
        raise ValueError("Degrees of freedom for one-period variance must be positive.")

    sigma_b = float(np.dot(one_period_diffs, one_period_diffs)) / denom_b

    q_period_diffs = prices[q:] - prices[:-q] - (q * mu_est)
    n_q = q_period_diffs.size
    denom_a = n_q - 1
    if denom_a <= 0:
        raise ValueError("Degrees of freedom for q-period variance must be positive.")

    sigma_a = float(np.dot(q_period_diffs, q_period_diffs)) / (denom_a * q)

    if sigma_a <= 0 or sigma_b <= 0:
        raise ValueError("Variance estimates must be positive.")

    return (sigma_a / sigma_b) - 1.0


def _compute_heteroskedastic_z2_from_log_prices(log_prices: np.ndarray, q: int) -> float:
    """Compute heteroskedastic Lo-MacKinlay Z2 at horizon ``q`` from log-prices."""
    prices = np.asarray(log_prices, dtype=float)
    if q < 2:
        raise ValueError("q must be at least 2 for the heteroskedastic variance estimator.")
    if prices.ndim != 1:
        raise ValueError("log_prices must be one-dimensional.")

    n = int(np.floor(prices.size / q))
    if n <= 0:
        raise ValueError("Not enough observations for the requested aggregation horizon.")

    mr = _compute_vr_value_from_log_prices(prices, q=q)

    upper_bound = n * q
    if upper_bound <= q:
        raise ValueError("Not enough observations to evaluate heteroskedastic variance.")

    mu_est = (prices[-1] - prices[0]) / prices.size
    diffs = prices[1:upper_bound] - prices[: upper_bound - 1] - mu_est
    denominator = float(np.dot(diffs, diffs))
    if denominator <= 0:
        raise ValueError("Second moment of the differenced series is non-positive.")

    v_hat = 0.0
    for j in range(1, q):
        lead = diffs[j:]
        lagged = diffs[:-j]
        numerator = float(np.dot(lead ** 2, lagged ** 2))
        delta = (upper_bound * numerator) / (denominator ** 2)
        weight = (2 * (q - j) / q) ** 2
        v_hat += weight * delta

    if v_hat <= 0:
        raise ValueError("Asymptotic variance estimate must be positive.")

    z2 = np.sqrt(n) * mr
    return float(z2 / np.sqrt(v_hat))


@dataclass(frozen=True)
class _AutomaticQResult:
    """Internal automatic-horizon selection payload for AVR-family tests."""

    selected_q: int
    q_raw: int
    rho_hat: float
    alpha_hat: float
    warnings: list[str]


def _select_automatic_q(returns: np.ndarray, max_automatic_q: int | None) -> _AutomaticQResult:
    """Select automatic horizon using AR(1) plug-in bandwidth rule."""
    r = np.asarray(returns, dtype=float)
    n_returns = int(r.size)
    if n_returns < 2:
        raise ValueError("At least 2 returns are required for automatic horizon selection.")

    r_demeaned = r - float(np.mean(r))
    x = r_demeaned[:-1]
    y = r_demeaned[1:]
    denom = float(np.dot(x, x))
    if denom <= 0:
        rho_hat = 0.0
    else:
        rho_hat = float(np.dot(x, y) / denom)

    if not np.isfinite(rho_hat):
        rho_hat = 0.0

    if abs(rho_hat) >= 0.97:
        rho_hat = 0.97 if rho_hat >= 0 else -0.97

    denom_alpha = (1 - rho_hat ** 2) ** 2
    alpha_hat = (4 * rho_hat ** 2) / denom_alpha if denom_alpha > 0 else np.inf

    if (not np.isfinite(alpha_hat)) or alpha_hat <= 0:
        q_raw = 2
    else:
        q_raw = int(np.ceil(1.3221 * (alpha_hat * n_returns) ** (1 / 3)))

    upper_candidate = max_automatic_q if max_automatic_q is not None else (n_returns // 2)
    upper_bound = min(upper_candidate, n_returns - 1)
    lower_bound = 2

    if upper_bound < lower_bound:
        raise ValueError(
            "Sample too short to select automatic horizon with bounds requiring q >= 2."
        )

    selected_q = int(min(max(q_raw, lower_bound), upper_bound))

    warnings: list[str] = []
    if selected_q != q_raw:
        warnings.append(
            f"Automatic q_raw={q_raw} was clamped to selected_q={selected_q} within "
            f"[{lower_bound}, {upper_bound}]."
        )
    if upper_bound == lower_bound:
        warnings.append(
            "Automatic horizon selection is dominated by sample-size upper-bound clamping."
        )

    return _AutomaticQResult(
        selected_q=selected_q,
        q_raw=int(q_raw),
        rho_hat=float(rho_hat),
        alpha_hat=float(alpha_hat),
        warnings=warnings,
    )


def _resolve_chow_denning_q_list(config: BatteryConfig) -> tuple[int, ...]:
    """Resolve q-family for Chow-Denning-style test from robust config or battery config."""
    robust = config.robust_vr
    if robust is None:
        raise ValueError("robust_vr config is required for robust diagnostics.")
    q_list = robust.q_list if robust.q_list is not None else config.q_list
    if not q_list:
        raise ValueError("Effective q_list cannot be empty.")
    if any((not isinstance(q, int)) or q <= 0 for q in q_list):
        raise ValueError("All effective q_list values must be positive integers.")
    if any(curr <= prev for prev, curr in zip(q_list, q_list[1:])):
        raise ValueError("Effective q_list must be strictly increasing.")
    return tuple(q_list)


def run_avr(normalized, config: BatteryConfig) -> TestOutcome:
    """Run automatic variance-ratio (AVR) diagnostic."""
    robust = _require_enabled_robust_config(config)
    automatic = _select_automatic_q(normalized.returns, robust.max_automatic_q)

    z_stat = _compute_heteroskedastic_z2_from_log_prices(
        normalized.log_prices,
        automatic.selected_q,
    )
    vr_value = _compute_vr_value_from_log_prices(normalized.log_prices, automatic.selected_q)
    p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

    return TestOutcome(
        name="avr",
        null_hypothesis=(
            "The return process is compatible with the automatic variance-ratio random walk null."
        ),
        statistic=float(z_stat),
        p_value=p_value,
        alpha=config.alpha,
        reject_null=bool(p_value < config.alpha),
        metadata={
            "selected_q": automatic.selected_q,
            "q_raw": automatic.q_raw,
            "rho_hat": automatic.rho_hat,
            "alpha_hat": automatic.alpha_hat,
            "vr_value": float(vr_value),
            "input_kind_used": "log_prices",
            "automatic_rule": "ar1_plugin_1.3221",
            "max_automatic_q": robust.max_automatic_q,
        },
        warnings=list(automatic.warnings),
    )


def run_wbavr(normalized, config: BatteryConfig) -> TestOutcome:
    """Run wild-bootstrap automatic variance-ratio (WBAVR) diagnostic."""
    robust = _require_enabled_robust_config(config)
    automatic = _select_automatic_q(normalized.returns, robust.max_automatic_q)

    selected_q = automatic.selected_q
    observed_z = _compute_heteroskedastic_z2_from_log_prices(normalized.log_prices, selected_q)
    vr_value = _compute_vr_value_from_log_prices(normalized.log_prices, selected_q)

    centered_returns = normalized.returns - float(np.mean(normalized.returns))
    rng = np.random.default_rng(robust.bootstrap_seed)
    finite_bootstrap_stats: list[float] = []

    for _ in range(robust.bootstrap_reps):
        eta = rng.choice(np.array([-1.0, 1.0]), size=centered_returns.size)
        bootstrap_returns = eta * centered_returns
        bootstrap_log_prices = np.concatenate(([0.0], np.cumsum(bootstrap_returns)))

        try:
            z_star = _compute_heteroskedastic_z2_from_log_prices(bootstrap_log_prices, selected_q)
        except ValueError:
            continue

        if np.isfinite(z_star):
            finite_bootstrap_stats.append(float(z_star))

    abs_obs = abs(float(observed_z))
    exceedances = sum(abs(stat) >= abs_obs for stat in finite_bootstrap_stats)
    p_value = float((1 + exceedances) / (robust.bootstrap_reps + 1))

    quantiles = {
        "q90": float(np.quantile(finite_bootstrap_stats, 0.90)) if finite_bootstrap_stats else None,
        "q95": float(np.quantile(finite_bootstrap_stats, 0.95)) if finite_bootstrap_stats else None,
        "q99": float(np.quantile(finite_bootstrap_stats, 0.99)) if finite_bootstrap_stats else None,
    }

    return TestOutcome(
        name="wbavr",
        null_hypothesis=(
            "The return process is compatible with the wild-bootstrap automatic "
            "variance-ratio random walk null."
        ),
        statistic=float(observed_z),
        p_value=p_value,
        alpha=config.alpha,
        reject_null=bool(p_value < config.alpha),
        metadata={
            "selected_q": selected_q,
            "bootstrap_reps": robust.bootstrap_reps,
            "bootstrap_seed": robust.bootstrap_seed,
            "wild_weights": robust.wild_weights,
            "vr_value": float(vr_value),
            "input_kind_used": "log_prices",
            "bootstrap_quantiles": quantiles,
            "n_bootstrap_effective": len(finite_bootstrap_stats),
        },
        warnings=list(automatic.warnings),
    )


def run_chow_denning(normalized, config: BatteryConfig) -> TestOutcome:
    """Run Chow-Denning-style max test with Sidak-normal familywise calibration."""
    robust = _require_enabled_robust_config(config)
    q_list = _resolve_chow_denning_q_list(config)

    per_q_statistics: dict[int, float] = {}
    per_q_p_values: dict[int, float] = {}

    for q in q_list:
        z_q = _compute_heteroskedastic_z2_from_log_prices(normalized.log_prices, q)
        p_q = float(2 * (1 - norm.cdf(abs(z_q))))
        per_q_statistics[int(q)] = float(z_q)
        per_q_p_values[int(q)] = p_q

    max_abs_z = float(max(abs(z) for z in per_q_statistics.values()))
    m = len(q_list)
    p_single_max = float(2 * (1 - norm.cdf(max_abs_z)))
    p_family = float(1 - (1 - p_single_max) ** m)

    return TestOutcome(
        name="chow_denning",
        null_hypothesis=(
            "All configured variance-ratio horizons are jointly compatible with the random walk null."
        ),
        statistic=max_abs_z,
        p_value=p_family,
        alpha=config.alpha,
        reject_null=bool(p_family < config.alpha),
        metadata={
            "q_list": list(q_list),
            "per_q_statistics": per_q_statistics,
            "per_q_p_values": per_q_p_values,
            "max_abs_z": max_abs_z,
            "familywise_p_value": p_family,
            "calibration": robust.chow_denning_calibration,
            "input_kind_used": "log_prices",
        },
        warnings=[],
    )


def make_non_computable_robust_outcome(
    name: str,
    null_hypothesis: str,
    alpha: float,
    reason: str,
) -> TestOutcome:
    """Build standardized non-computable robust diagnostic outcome."""
    return TestOutcome(
        name=name,
        null_hypothesis=null_hypothesis,
        statistic=None,
        p_value=None,
        alpha=alpha,
        reject_null=None,
        metadata={"reason": reason},
        warnings=[reason],
    )


def _require_enabled_robust_config(config: BatteryConfig) -> RobustVRConfig:
    """Validate and return robust config when robust diagnostics are enabled."""
    robust = config.robust_vr
    if robust is None:
        raise ValueError("robust_vr must be provided to run robust diagnostics.")
    if not isinstance(robust, RobustVRConfig):
        raise ValueError("robust_vr must be a RobustVRConfig instance.")
    if robust.enabled is not True:
        raise ValueError("robust_vr.enabled must be True to run robust diagnostics.")
    return robust
