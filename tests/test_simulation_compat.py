"""Compatibility checks for simulation and core VRT behavior."""

from __future__ import annotations

import numpy as np

from variance_test import (
    EMH,
    SimulationConfig,
    SimulationResults,
    run_simulation,
    simulate_price_processes,
)


def test_run_simulation_returns_simulation_results() -> None:
    """A minimal run_simulation call should still return SimulationResults."""
    config = SimulationConfig(
        num_series=2,
        horizon=25,
        aggregation_horizon=3,
        seed=7,
    )

    results = run_simulation(config)
    assert isinstance(results, SimulationResults)


def test_simulate_price_processes_returns_expected_keys_and_arrays() -> None:
    """Price process simulation should keep expected process keys and array outputs."""
    config = SimulationConfig(
        num_series=2,
        horizon=20,
        aggregation_horizon=2,
        seed=21,
    )

    processes, elapsed = simulate_price_processes(config)
    assert set(processes.keys()) == {"gbm", "merton", "heston"}
    assert elapsed >= 0

    for values in processes.values():
        assert isinstance(values, np.ndarray)
        assert values.shape[0] == config.horizon


def test_emh_vrt_minimal_call_returns_numeric_tuple() -> None:
    """EMH.vrt must return numeric z-score and p-value after migration."""
    emh = EMH()
    rng = np.random.default_rng(123)
    log_prices = np.log(np.cumsum(rng.normal(0.0, 0.5, size=64)) + 100.0)

    z_score, p_value = emh.vrt(
        X=log_prices,
        q=3,
        heteroskedastic=True,
        centered=True,
        unbiased=True,
        annualize=False,
    )

    assert isinstance(float(z_score), float)
    assert isinstance(float(p_value), float)
