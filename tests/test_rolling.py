"""Rolling-window tests for weak-form battery (Sprint 4)."""

from __future__ import annotations

import numpy as np
import pytest

from variance_test import BatteryConfig, run_weak_form_battery


def _log_prices(seed: int, n_obs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n_obs - 1)
    return np.concatenate(([0.0], np.cumsum(returns)))


def _returns(seed: int, n_obs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n_obs)


def _rolling_config(**kwargs) -> BatteryConfig:
    base = {
        "input_kind": "log_prices",
        "q_list": (2, 4),
        "ljung_box_lags": (5, 10),
        "arch_lm_lags": 5,
        "runs_test": True,
        "rolling_window": 120,
        "rolling_step": 1,
    }
    base.update(kwargs)
    return BatteryConfig(**base)


def test_01_rolling_none_when_disabled() -> None:
    series = _log_prices(seed=1, n_obs=400)
    config = _rolling_config(rolling_window=None)
    outcome = run_weak_form_battery(series, config=config)
    assert outcome.rolling is None


def test_02_rolling_present_when_enabled() -> None:
    series = _log_prices(seed=2, n_obs=400)
    outcome = run_weak_form_battery(series, config=_rolling_config(rolling_window=100, rolling_step=1))
    assert outcome.rolling is not None


def test_03_rolling_top_level_keys_exact() -> None:
    series = _log_prices(seed=3, n_obs=400)
    rolling = run_weak_form_battery(series, config=_rolling_config()).rolling
    assert rolling is not None
    assert set(rolling.keys()) == {"window", "step", "n_windows", "tests"}


def test_04_rolling_test_keys_exact() -> None:
    series = _log_prices(seed=4, n_obs=400)
    rolling = run_weak_form_battery(series, config=_rolling_config(q_list=(2, 4))).rolling
    assert rolling is not None
    assert set(rolling["tests"].keys()) == {
        "variance_ratio_q2",
        "variance_ratio_q4",
        "ljung_box_returns",
        "ljung_box_squared_returns",
    }


def test_05_n_windows_formula_matches() -> None:
    n_obs = 501
    rolling_window = 100
    rolling_step = 7
    series = _log_prices(seed=5, n_obs=n_obs)
    rolling = run_weak_form_battery(
        series,
        config=_rolling_config(rolling_window=rolling_window, rolling_step=rolling_step),
    ).rolling
    assert rolling is not None
    expected = ((n_obs - rolling_window) // rolling_step) + 1
    assert rolling["n_windows"] == expected


def test_06_each_rolling_list_matches_n_windows() -> None:
    series = _log_prices(seed=6, n_obs=500)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=120, rolling_step=10)).rolling
    assert rolling is not None
    for values in rolling["tests"].values():
        assert len(values) == rolling["n_windows"]


def test_07_each_rolling_entry_keys_exact() -> None:
    series = _log_prices(seed=7, n_obs=420)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=100, rolling_step=13)).rolling
    assert rolling is not None
    for values in rolling["tests"].values():
        for item in values:
            assert set(item.keys()) == {
                "start",
                "end",
                "statistic",
                "p_value",
                "reject_null",
                "warning",
            }


def test_08_window_indices_monotonic_and_correct() -> None:
    n_obs = 501
    rolling_window = 100
    rolling_step = 50
    series = _log_prices(seed=8, n_obs=n_obs)
    rolling = run_weak_form_battery(
        series,
        config=_rolling_config(rolling_window=rolling_window, rolling_step=rolling_step),
    ).rolling
    assert rolling is not None
    starts = [entry["start"] for entry in rolling["tests"]["variance_ratio_q2"]]
    ends = [entry["end"] for entry in rolling["tests"]["variance_ratio_q2"]]
    assert starts == sorted(starts)
    assert all(curr > prev for prev, curr in zip(starts, starts[1:]))
    assert all(end - start == rolling_window for start, end in zip(starts, ends))
    assert all(curr - prev == rolling_step for prev, curr in zip(starts, starts[1:]))


def test_09_n_windows_501_100_1() -> None:
    series = _log_prices(seed=9, n_obs=501)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=100, rolling_step=1)).rolling
    assert rolling is not None
    assert rolling["n_windows"] == 402


def test_10_n_windows_501_100_50() -> None:
    series = _log_prices(seed=10, n_obs=501)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=100, rolling_step=50)).rolling
    assert rolling is not None
    assert rolling["n_windows"] == 9


def test_11_rolling_window_greater_than_n_obs_raises() -> None:
    series = _log_prices(seed=11, n_obs=100)
    with pytest.raises(ValueError):
        run_weak_form_battery(series, config=_rolling_config(rolling_window=101))


def test_12_rolling_window_equal_n_obs_has_one_window() -> None:
    n_obs = 250
    series = _log_prices(seed=12, n_obs=n_obs)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=n_obs, rolling_step=1)).rolling
    assert rolling is not None
    assert rolling["n_windows"] == 1


def test_13_small_window_for_q_is_non_computable_not_global_error() -> None:
    series = _log_prices(seed=13, n_obs=400)
    rolling = run_weak_form_battery(
        series,
        config=_rolling_config(q_list=(2, 50), rolling_window=20, rolling_step=5),
    ).rolling
    assert rolling is not None
    entries = rolling["tests"]["variance_ratio_q50"]
    assert entries
    assert any(item["statistic"] is None for item in entries)
    assert any(item["warning"] for item in entries)


def test_14_small_window_for_ljung_box_is_non_computable_not_global_error() -> None:
    series = _log_prices(seed=14, n_obs=500)
    rolling = run_weak_form_battery(
        series,
        config=_rolling_config(ljung_box_lags=(5, 10), rolling_window=8, rolling_step=2),
    ).rolling
    assert rolling is not None
    entries = rolling["tests"]["ljung_box_returns"]
    assert entries
    assert all(item["statistic"] is None for item in entries)
    assert all(item["warning"] for item in entries)


def test_15_full_sample_tests_still_present_when_rolling_active() -> None:
    series = _log_prices(seed=15, n_obs=450)
    outcome = run_weak_form_battery(series, config=_rolling_config(rolling_window=120))
    assert outcome.tests
    assert "variance_ratio_holm" in outcome.tests
    assert "arch_lm" in outcome.tests


def test_16_multiple_testing_keys_unchanged() -> None:
    series = _log_prices(seed=16, n_obs=450)
    outcome = run_weak_form_battery(series, config=_rolling_config(rolling_window=100))
    assert set(outcome.multiple_testing.keys()) == {"variance_ratio_holm", "battery_summary"}


def test_17_rolling_only_contains_allowed_entries() -> None:
    series = _log_prices(seed=17, n_obs=450)
    rolling = run_weak_form_battery(series, config=_rolling_config()).rolling
    assert rolling is not None
    assert "variance_ratio_holm" not in rolling["tests"]
    assert "runs_test_signs" not in rolling["tests"]
    assert "arch_lm" not in rolling["tests"]


def test_18_returns_and_log_prices_rolling_first_vr_window_matches() -> None:
    returns = _returns(seed=18, n_obs=600)
    log_prices = np.concatenate(([0.0], np.cumsum(returns)))

    out_returns = run_weak_form_battery(
        returns,
        config=_rolling_config(input_kind="returns", rolling_window=100, q_list=(2,)),
    )
    out_log = run_weak_form_battery(
        log_prices,
        config=_rolling_config(input_kind="log_prices", rolling_window=101, q_list=(2,)),
    )

    assert out_returns.rolling is not None
    assert out_log.rolling is not None
    assert out_returns.rolling["n_windows"] == out_log.rolling["n_windows"]

    first_returns = out_returns.rolling["tests"]["variance_ratio_q2"][0]
    first_log = out_log.rolling["tests"]["variance_ratio_q2"][0]
    assert first_returns["statistic"] is not None
    assert first_log["statistic"] is not None
    assert np.isclose(first_returns["statistic"], first_log["statistic"], atol=1e-10, rtol=1e-8)


def test_19_full_sample_results_unchanged_with_rolling_active() -> None:
    series = _log_prices(seed=19, n_obs=500)
    base = run_weak_form_battery(series, config=_rolling_config(rolling_window=None))
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=120, rolling_step=3))

    assert rolling.rolling is not None
    assert set(base.tests.keys()) == set(rolling.tests.keys())
    for name in base.tests:
        assert base.tests[name].statistic == rolling.tests[name].statistic
        assert base.tests[name].p_value == rolling.tests[name].p_value
        assert base.tests[name].reject_null == rolling.tests[name].reject_null

    assert base.multiple_testing == rolling.multiple_testing


def test_20_all_computable_p_values_are_in_unit_interval() -> None:
    series = _log_prices(seed=20, n_obs=550)
    rolling = run_weak_form_battery(series, config=_rolling_config(rolling_window=120, rolling_step=4)).rolling
    assert rolling is not None
    for values in rolling["tests"].values():
        for item in values:
            if item["p_value"] is not None:
                assert 0.0 <= item["p_value"] <= 1.0


def test_21_all_computable_reject_null_are_consistent_with_alpha() -> None:
    series = _log_prices(seed=21, n_obs=550)
    config = _rolling_config(rolling_window=120, rolling_step=4, alpha=0.05)
    rolling = run_weak_form_battery(series, config=config).rolling
    assert rolling is not None
    for values in rolling["tests"].values():
        for item in values:
            if item["p_value"] is not None:
                assert item["reject_null"] == (item["p_value"] < config.alpha)
