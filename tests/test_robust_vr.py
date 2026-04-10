"""Tests for robust variance-ratio diagnostics introduced in v0.2.0."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from variance_test import BatteryConfig, EMH, RobustVRConfig, run_weak_form_battery
from variance_test.diagnostics.robust_vr import _compute_heteroskedastic_z2_from_log_prices


def _iid_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.01, size=5000)


def _ar1_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    size = 5000
    eps = rng.normal(0.0, 0.01, size=size)
    values = np.zeros(size, dtype=float)
    phi = 0.3
    for idx in range(1, size):
        values[idx] = phi * values[idx - 1] + eps[idx]
    return values


def _as_log_prices(returns: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(returns)))


def _base_config(input_kind: str, alpha: float = 0.05, robust: RobustVRConfig | None = None) -> BatteryConfig:
    return BatteryConfig(
        input_kind=input_kind,
        alpha=alpha,
        q_list=(2, 4, 8, 16),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
        robust_vr=robust,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bootstrap_reps": 198},
        {"wild_weights": "normal"},
        {"chow_denning_calibration": "smm"},
        {"q_list": (2, 2, 4)},
        {"max_automatic_q": 1},
    ],
)
def test_robust_config_validation_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RobustVRConfig(**kwargs)


def test_robust_config_validation_valid() -> None:
    assert BatteryConfig(robust_vr=None)
    assert BatteryConfig(robust_vr=RobustVRConfig())


def test_pure_math_helper_matches_emh_vrt_heteroskedastic_z2() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=200)
    log_prices = _as_log_prices(returns)

    emh = EMH()
    for q in (2, 4, 8, 16):
        helper = _compute_heteroskedastic_z2_from_log_prices(log_prices, q=q)
        expected, _ = emh.vrt(
            X=log_prices,
            q=q,
            heteroskedastic=True,
            centered=True,
            unbiased=True,
            annualize=False,
            input_kind="log_prices",
            alternative="two-sided",
        )
        assert np.isclose(helper, expected, atol=1e-10, rtol=1e-8)


def test_robust_disabled_preserves_v1_keys_and_multiple_testing() -> None:
    returns = _iid_returns()
    outcome = run_weak_form_battery(returns, config=_base_config(input_kind="returns"))

    assert "avr" not in outcome.tests
    assert "wbavr" not in outcome.tests
    assert "chow_denning" not in outcome.tests
    assert tuple(outcome.multiple_testing.keys()) == ("variance_ratio_holm", "battery_summary")


def test_enabling_robust_keeps_v1_statistics_identical() -> None:
    returns = _iid_returns()
    base = run_weak_form_battery(returns, config=_base_config(input_kind="returns"))
    robust = run_weak_form_battery(
        returns,
        config=_base_config(
            input_kind="returns",
            robust=RobustVRConfig(enabled=True, bootstrap_reps=199, bootstrap_seed=42),
        ),
    )

    comparable = [
        *(f"variance_ratio_q{q}" for q in (2, 4, 8, 16)),
        "variance_ratio_holm",
        "ljung_box_returns",
        "ljung_box_squared_returns",
        "runs_test_signs",
        "arch_lm",
    ]
    for name in comparable:
        assert base.tests[name].statistic == robust.tests[name].statistic
        assert base.tests[name].p_value == robust.tests[name].p_value
        assert base.tests[name].reject_null == robust.tests[name].reject_null


def test_robust_enabled_output_shape_and_order() -> None:
    returns = _iid_returns()
    outcome = run_weak_form_battery(
        returns,
        config=_base_config(
            input_kind="returns",
            robust=RobustVRConfig(enabled=True, bootstrap_reps=199, bootstrap_seed=42),
        ),
    )

    assert list(outcome.tests.keys()) == [
        "variance_ratio_q2",
        "variance_ratio_q4",
        "variance_ratio_q8",
        "variance_ratio_q16",
        "variance_ratio_holm",
        "ljung_box_returns",
        "ljung_box_squared_returns",
        "runs_test_signs",
        "arch_lm",
        "avr",
        "wbavr",
        "chow_denning",
    ]
    assert tuple(outcome.multiple_testing.keys()) == (
        "variance_ratio_holm",
        "battery_summary",
        "robust_vr_summary",
    )
    assert tuple(outcome.multiple_testing["robust_vr_summary"].keys()) == (
        "tests_included",
        "rejected_tests",
        "n_rejections",
        "any_reject",
    )


def test_robust_selective_enable_rules() -> None:
    returns = _iid_returns()
    for robust in (
        RobustVRConfig(enabled=True, enable_wbavr=False, enable_chow_denning=False, bootstrap_reps=199),
        RobustVRConfig(enabled=True, enable_avr=False, enable_chow_denning=False, bootstrap_reps=199),
        RobustVRConfig(enabled=True, enable_avr=False, enable_wbavr=False, bootstrap_reps=199),
    ):
        outcome = run_weak_form_battery(returns, config=_base_config(input_kind="returns", robust=robust))
        included = outcome.multiple_testing["robust_vr_summary"]["tests_included"]
        assert len(included) == 1
        assert included[0] in {"avr", "wbavr", "chow_denning"}


def test_returns_vs_log_prices_equivalence_for_robust_tests() -> None:
    returns = _iid_returns()
    log_prices = _as_log_prices(returns)
    robust = RobustVRConfig(enabled=True, bootstrap_reps=199, bootstrap_seed=42)

    out_ret = run_weak_form_battery(returns, config=_base_config("returns", robust=robust))
    out_log = run_weak_form_battery(log_prices, config=_base_config("log_prices", robust=robust))

    for name in ("avr", "wbavr", "chow_denning"):
        assert np.isclose(out_ret.tests[name].statistic, out_log.tests[name].statistic, atol=1e-10, rtol=1e-8)
        assert np.isclose(out_ret.tests[name].p_value, out_log.tests[name].p_value, atol=1e-10, rtol=1e-8)
        assert out_ret.tests[name].reject_null == out_log.tests[name].reject_null


def test_avr_contract_and_behavior() -> None:
    iid = _iid_returns()
    config_iid = _base_config(
        "returns",
        alpha=0.01,
        robust=RobustVRConfig(enabled=True, enable_wbavr=False, enable_chow_denning=False, bootstrap_reps=199),
    )
    out_iid = run_weak_form_battery(iid, config=config_iid)
    avr = out_iid.tests["avr"]

    assert np.isfinite(avr.statistic)
    assert np.isfinite(avr.p_value)
    for key in ("selected_q", "q_raw", "rho_hat", "alpha_hat", "vr_value", "input_kind_used"):
        assert key in avr.metadata
    assert avr.metadata["selected_q"] >= 2
    assert avr.reject_null is False

    out_iid_2 = run_weak_form_battery(iid, config=config_iid)
    assert avr.statistic == out_iid_2.tests["avr"].statistic
    assert avr.p_value == out_iid_2.tests["avr"].p_value

    ar1 = _ar1_returns()
    config_ar1 = _base_config(
        "returns",
        alpha=0.05,
        robust=RobustVRConfig(enabled=True, enable_wbavr=False, enable_chow_denning=False, bootstrap_reps=199),
    )
    out_ar1 = run_weak_form_battery(ar1, config=config_ar1)
    assert out_ar1.tests["avr"].reject_null is True


def test_wbavr_contract_and_behavior() -> None:
    iid = _iid_returns()
    config_iid = _base_config(
        "returns",
        alpha=0.01,
        robust=RobustVRConfig(enabled=True, enable_avr=False, enable_chow_denning=False, bootstrap_reps=199, bootstrap_seed=42),
    )
    out_iid = run_weak_form_battery(iid, config=config_iid)
    wbavr = out_iid.tests["wbavr"]

    assert np.isfinite(wbavr.statistic)
    assert np.isfinite(wbavr.p_value)
    for key in (
        "selected_q",
        "bootstrap_reps",
        "bootstrap_seed",
        "wild_weights",
        "bootstrap_quantiles",
        "n_bootstrap_effective",
    ):
        assert key in wbavr.metadata
    assert set(wbavr.metadata["bootstrap_quantiles"].keys()) == {"q90", "q95", "q99"}
    assert wbavr.reject_null is False

    out_iid_2 = run_weak_form_battery(iid, config=config_iid)
    assert wbavr.statistic == out_iid_2.tests["wbavr"].statistic
    assert wbavr.p_value == out_iid_2.tests["wbavr"].p_value

    ar1 = _ar1_returns()
    config_ar1 = _base_config(
        "returns",
        alpha=0.05,
        robust=RobustVRConfig(enabled=True, enable_avr=False, enable_chow_denning=False, bootstrap_reps=199, bootstrap_seed=42),
    )
    out_ar1 = run_weak_form_battery(ar1, config=config_ar1)
    assert out_ar1.tests["wbavr"].reject_null is True


def test_chow_denning_contract_and_behavior() -> None:
    iid = _iid_returns()
    config_iid = _base_config(
        "returns",
        alpha=0.01,
        robust=RobustVRConfig(enabled=True, enable_avr=False, enable_wbavr=False, bootstrap_reps=199),
    )
    out_iid = run_weak_form_battery(iid, config=config_iid)
    cd = out_iid.tests["chow_denning"]

    per_q = cd.metadata["per_q_statistics"]
    assert np.isclose(cd.statistic, max(abs(v) for v in per_q.values()))
    assert cd.metadata["familywise_p_value"] == cd.p_value
    assert cd.metadata["calibration"] == "sidak_normal"

    for q, z_val in per_q.items():
        expected = 2 * (1 - norm.cdf(abs(z_val)))
        assert np.isclose(cd.metadata["per_q_p_values"][q], expected, atol=1e-10, rtol=1e-8)

    assert cd.reject_null is False

    ar1 = _ar1_returns()
    config_ar1 = _base_config(
        "returns",
        alpha=0.05,
        robust=RobustVRConfig(enabled=True, enable_avr=False, enable_wbavr=False, bootstrap_reps=199),
    )
    out_ar1 = run_weak_form_battery(ar1, config=config_ar1)
    assert out_ar1.tests["chow_denning"].reject_null is True


def test_short_series_non_computable_robust_without_crash() -> None:
    short_returns = np.array([0.01, -0.01], dtype=float)
    config = BatteryConfig(
        input_kind="returns",
        q_list=(1,),
        ljung_box_lags=(1,),
        arch_lm_lags=1,
        runs_test=False,
        robust_vr=RobustVRConfig(enabled=True, bootstrap_reps=199),
    )
    outcome = run_weak_form_battery(short_returns, config=config)

    for name in ("avr", "wbavr", "chow_denning"):
        robust_test = outcome.tests[name]
        assert robust_test.statistic is None
        assert robust_test.p_value is None
        assert robust_test.reject_null is None
        assert robust_test.warnings


def test_computable_robust_tests_p_value_and_reject_consistency() -> None:
    returns = _iid_returns()
    config = _base_config(
        "returns",
        alpha=0.05,
        robust=RobustVRConfig(enabled=True, bootstrap_reps=199, bootstrap_seed=42),
    )
    outcome = run_weak_form_battery(returns, config=config)

    for name in ("avr", "wbavr", "chow_denning"):
        test = outcome.tests[name]
        assert test.p_value is not None
        assert 0.0 <= test.p_value <= 1.0
        assert test.reject_null == (test.p_value < config.alpha)


def test_robust_enable_does_not_change_rolling_payload_shape() -> None:
    returns = _iid_returns()
    config = _base_config(
        "returns",
        robust=RobustVRConfig(enabled=True, bootstrap_reps=199, bootstrap_seed=42),
    )
    config.rolling_window = 120
    config.rolling_step = 10

    outcome = run_weak_form_battery(returns, config=config)
    assert outcome.rolling is not None
    assert set(outcome.rolling["tests"].keys()) == {
        "variance_ratio_q2",
        "variance_ratio_q4",
        "variance_ratio_q8",
        "variance_ratio_q16",
        "ljung_box_returns",
        "ljung_box_squared_returns",
    }
