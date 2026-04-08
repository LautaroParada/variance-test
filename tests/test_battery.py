"""Tests for weak-form battery orchestration (Sprint 3)."""

from __future__ import annotations

import numpy as np
import pytest

from variance_test import BatteryConfig, BatteryOutcome, run_weak_form_battery


def _iid_returns(seed: int, size: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=size)


def _ar1_returns(seed: int, size: int, phi: float, sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=size)
    series = np.zeros(size, dtype=float)
    for idx in range(1, size):
        series[idx] = phi * series[idx - 1] + eps[idx]
    return series


def _arch_like_returns(seed: int, size: int, omega: float = 0.00001, alpha1: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, size=size)
    returns = np.zeros(size, dtype=float)
    sigma2 = np.zeros(size, dtype=float)
    sigma2[0] = omega / (1 - alpha1)

    for idx in range(1, size):
        sigma2[idx] = omega + alpha1 * (returns[idx - 1] ** 2)
        returns[idx] = shocks[idx] * np.sqrt(sigma2[idx])

    return returns


def test_run_weak_form_battery_importable_from_variance_test() -> None:
    """run_weak_form_battery must be available from package root."""
    assert run_weak_form_battery is not None


def test_minimal_execution_returns_battery_outcome() -> None:
    """A minimal execution on IID returns should produce BatteryOutcome."""
    returns = _iid_returns(seed=100, size=800)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4), ljung_box_lags=(5, 10), arch_lm_lags=5)

    outcome = run_weak_form_battery(returns, config=config)

    assert isinstance(outcome, BatteryOutcome)


def test_battery_tests_keys_exact_for_runs_enabled() -> None:
    """The tests mapping must contain exactly the required keys when runs_test is True."""
    returns = _iid_returns(seed=101, size=900)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4, 8),
        ljung_box_lags=(5, 10),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert list(outcome.tests.keys()) == [
        "variance_ratio_q2",
        "variance_ratio_q4",
        "variance_ratio_q8",
        "variance_ratio_holm",
        "ljung_box_returns",
        "ljung_box_squared_returns",
        "runs_test_signs",
        "arch_lm",
    ]


def test_returns_and_log_prices_equivalence_for_core_outcomes() -> None:
    """Equivalent returns and synthetic log-prices must produce equal core outcomes."""
    returns = _iid_returns(seed=102, size=1200)
    log_prices = np.concatenate(([0.0], np.cumsum(returns)))

    config_returns = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
    )
    config_log_prices = BatteryConfig(
        input_kind="log_prices",
        q_list=(2, 4),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
    )

    out_returns = run_weak_form_battery(returns, config=config_returns)
    out_log_prices = run_weak_form_battery(log_prices, config=config_log_prices)

    comparable = [
        "variance_ratio_q2",
        "variance_ratio_q4",
        "ljung_box_returns",
        "ljung_box_squared_returns",
        "runs_test_signs",
        "arch_lm",
    ]
    for name in comparable:
        left = out_returns.tests[name]
        right = out_log_prices.tests[name]

        if left.statistic is None or right.statistic is None:
            assert left.statistic is right.statistic
        else:
            assert np.isclose(left.statistic, right.statistic, atol=1e-10, rtol=1e-8)

        if left.p_value is None or right.p_value is None:
            assert left.p_value is right.p_value
        else:
            assert np.isclose(left.p_value, right.p_value, atol=1e-10, rtol=1e-8)

        assert left.reject_null == right.reject_null

    assert (
        out_returns.tests["variance_ratio_holm"].reject_null
        == out_log_prices.tests["variance_ratio_holm"].reject_null
    )


def test_iid_gaussian_alpha_001_non_rejection_for_selected_tests() -> None:
    """IID Gaussian data should not reject selected tests at alpha=0.01."""
    returns = _iid_returns(seed=12345, size=5000)
    config = BatteryConfig(
        input_kind="returns",
        alpha=0.01,
        q_list=(2, 4, 8),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert outcome.tests["variance_ratio_holm"].reject_null is False
    assert outcome.tests["ljung_box_returns"].reject_null is False
    assert outcome.tests["runs_test_signs"].reject_null is False
    assert outcome.tests["arch_lm"].reject_null is False


def test_ar1_detects_serial_dependence() -> None:
    """AR(1) data should reject return dependence and VR family summary."""
    returns = _ar1_returns(seed=456, size=4000, phi=0.3)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4, 8),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert outcome.tests["ljung_box_returns"].reject_null is True
    vr_rejections = [
        outcome.tests[f"variance_ratio_q{q}"].reject_null for q in config.q_list
    ]
    assert any(vr_rejections)
    assert outcome.tests["variance_ratio_holm"].reject_null is True


def test_arch_like_detects_volatility_dependence() -> None:
    """ARCH-like data should reject squared-return dependence and ARCH LM."""
    returns = _arch_like_returns(seed=789, size=5000)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4, 8),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert outcome.tests["ljung_box_squared_returns"].reject_null is True
    assert outcome.tests["arch_lm"].reject_null is True


def test_alternating_signs_reject_runs_test() -> None:
    """Alternating signs should reject runs test randomness."""
    returns = np.tile(np.array([0.01, -0.01]), 800)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4),
        ljung_box_lags=(5, 10),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert outcome.tests["runs_test_signs"].reject_null is True


def test_all_zero_returns_make_runs_test_not_computable() -> None:
    """All-zero returns should produce non-computable runs test outcome."""
    returns = np.zeros(500, dtype=float)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4),
        ljung_box_lags=(5, 10),
        runs_test=True,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)
    runs_test = outcome.tests["runs_test_signs"]

    assert runs_test.statistic is None
    assert runs_test.p_value is None
    assert runs_test.reject_null is None
    assert runs_test.warnings


def test_raises_for_incompatible_q_list() -> None:
    """Incompatible q_list values must raise ValueError."""
    returns = _iid_returns(seed=200, size=5)
    config = BatteryConfig(input_kind="returns", q_list=(2, 6), ljung_box_lags=(1,), arch_lm_lags=1)

    with pytest.raises(ValueError):
        run_weak_form_battery(returns, config=config)


def test_raises_for_incompatible_ljung_box_lags() -> None:
    """Incompatible Ljung-Box lags must raise ValueError."""
    returns = _iid_returns(seed=201, size=10)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4), ljung_box_lags=(5, 10), arch_lm_lags=3)

    with pytest.raises(ValueError):
        run_weak_form_battery(returns, config=config)


def test_raises_for_incompatible_arch_lm_lags() -> None:
    """Incompatible ARCH LM lags must raise ValueError."""
    returns = _iid_returns(seed=202, size=8)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4), ljung_box_lags=(2, 3), arch_lm_lags=8)

    with pytest.raises(ValueError):
        run_weak_form_battery(returns, config=config)


def test_runs_key_absent_when_disabled() -> None:
    """runs_test_signs must be absent when runs_test=False."""
    returns = _iid_returns(seed=300, size=1000)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(2, 4, 8),
        ljung_box_lags=(5, 10),
        runs_test=False,
        arch_lm_lags=5,
    )

    outcome = run_weak_form_battery(returns, config=config)

    assert "runs_test_signs" not in outcome.tests


def test_multiple_testing_keys_exact() -> None:
    """multiple_testing must contain exactly the required keys."""
    returns = _iid_returns(seed=301, size=1000)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4), ljung_box_lags=(5, 10), arch_lm_lags=5)

    outcome = run_weak_form_battery(returns, config=config)

    assert set(outcome.multiple_testing.keys()) == {"variance_ratio_holm", "battery_summary"}


def test_battery_summary_keys_exact() -> None:
    """battery_summary must contain exactly the required keys."""
    returns = _iid_returns(seed=302, size=1000)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4), ljung_box_lags=(5, 10), arch_lm_lags=5)

    outcome = run_weak_form_battery(returns, config=config)
    summary = outcome.multiple_testing["battery_summary"]

    assert set(summary.keys()) == {
        "rejected_tests",
        "n_rejections",
        "mean_dependence_rejected",
        "sign_dependence_rejected",
        "volatility_dependence_rejected",
        "weak_form_evidence_against_null",
    }


def test_all_computable_p_values_in_unit_interval() -> None:
    """All computable p-values must lie in [0, 1]."""
    returns = _iid_returns(seed=303, size=1200)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4, 8), ljung_box_lags=(5, 10), arch_lm_lags=5)

    outcome = run_weak_form_battery(returns, config=config)

    for test_outcome in outcome.tests.values():
        if test_outcome.p_value is not None:
            assert 0.0 <= test_outcome.p_value <= 1.0


def test_reject_null_is_consistent_with_p_value_and_alpha() -> None:
    """Each computable reject_null must be equivalent to p_value < alpha."""
    returns = _iid_returns(seed=304, size=1200)
    config = BatteryConfig(input_kind="returns", q_list=(2, 4, 8), ljung_box_lags=(5, 10), arch_lm_lags=5)

    outcome = run_weak_form_battery(returns, config=config)

    for test_outcome in outcome.tests.values():
        if test_outcome.p_value is not None:
            assert test_outcome.reject_null == (test_outcome.p_value < test_outcome.alpha)
