"""Tests for Sprint 2 data contract models."""

from __future__ import annotations

import pytest

from variance_test import BatteryConfig, BatteryOutcome, TestOutcome


def test_battery_config_defaults_construct() -> None:
    """BatteryConfig must construct with default values."""
    config = BatteryConfig()
    assert config.input_kind == "log_prices"


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_battery_config_rejects_invalid_alpha(alpha: float) -> None:
    """alpha must satisfy strict open interval constraints."""
    with pytest.raises(ValueError):
        BatteryConfig(alpha=alpha)


def test_battery_config_rejects_empty_q_list() -> None:
    """q_list cannot be empty."""
    with pytest.raises(ValueError):
        BatteryConfig(q_list=())


@pytest.mark.parametrize("q_list", [(4, 2, 8), (2, 2, 8)])
def test_battery_config_requires_strictly_increasing_q_list(q_list: tuple[int, ...]) -> None:
    """q_list must be strictly increasing."""
    with pytest.raises(ValueError):
        BatteryConfig(q_list=q_list)


def test_battery_config_rejects_empty_ljung_box_lags() -> None:
    """ljung_box_lags cannot be empty."""
    with pytest.raises(ValueError):
        BatteryConfig(ljung_box_lags=())


def test_battery_config_rejects_arch_lm_lags_less_than_one() -> None:
    """arch_lm_lags must be at least 1."""
    with pytest.raises(ValueError):
        BatteryConfig(arch_lm_lags=0)


def test_test_outcome_valid_consistent_rejection_constructs() -> None:
    """A consistent reject_null value must be accepted."""
    outcome = TestOutcome(
        name="vrt",
        null_hypothesis="random walk",
        statistic=2.0,
        p_value=0.01,
        alpha=0.05,
        reject_null=True,
        metadata={},
        warnings=[],
    )
    assert outcome.reject_null is True


def test_test_outcome_reject_null_consistency_is_enforced() -> None:
    """An inconsistent reject_null value must raise ValueError."""
    with pytest.raises(ValueError):
        TestOutcome(
            name="vrt",
            null_hypothesis="random walk",
            statistic=2.0,
            p_value=0.01,
            alpha=0.05,
            reject_null=False,
            metadata={},
            warnings=[],
        )


def test_battery_outcome_accepts_empty_tests() -> None:
    """BatteryOutcome may contain an empty tests mapping in this sprint."""
    outcome = BatteryOutcome(
        input_kind="log_prices",
        n_obs=10,
        returns_n_obs=9,
        tests={},
        multiple_testing={},
        rolling=None,
        warnings=[],
    )
    assert outcome.tests == {}


def test_battery_outcome_rejects_non_test_outcome_values() -> None:
    """tests values must be instances of TestOutcome."""
    with pytest.raises(ValueError):
        BatteryOutcome(
            input_kind="log_prices",
            n_obs=10,
            returns_n_obs=9,
            tests={"vrt": "invalid"},
            multiple_testing={},
            rolling=None,
            warnings=[],
        )
