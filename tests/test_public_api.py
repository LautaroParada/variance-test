"""Tests for the stable public API surface."""

from __future__ import annotations

from variance_test import (
    BatteryConfig,
    BatteryOutcome,
    EMH,
    NormalizedSeries,
    PricePaths,
    RobustVRConfig,
    SimulationConfig,
    SimulationResults,
    TestOutcome,
    compute_vrt_statistics,
    normalize_series,
    run_simulation,
    run_weak_form_battery,
    simulate_price_processes,
)
import variance_test


EXPECTED_PUBLIC_API = {
    "EMH",
    "PricePaths",
    "VRTVisuals",
    "SimulationConfig",
    "SimulationResults",
    "simulate_price_processes",
    "compute_vrt_statistics",
    "run_simulation",
    "NormalizedSeries",
    "normalize_series",
    "BatteryConfig",
    "RobustVRConfig",
    "TestOutcome",
    "BatteryOutcome",
    "run_weak_form_battery",
}


def test_emh_exists_and_has_vrt_method() -> None:
    """EMH must be available and expose the public vrt method."""
    emh = EMH()
    assert hasattr(emh, "vrt")


def test_price_paths_exists() -> None:
    """PricePaths class must remain importable."""
    assert PricePaths is not None


def test_simulation_entities_importable_from_root() -> None:
    """Simulation API entities must be importable from package root."""
    assert SimulationConfig is not None
    assert SimulationResults is not None
    assert simulate_price_processes is not None
    assert compute_vrt_statistics is not None
    assert run_simulation is not None


def test_new_data_contract_api_importable_from_root() -> None:
    """Data/model contracts must be importable from package root."""
    assert NormalizedSeries is not None
    assert normalize_series is not None
    assert BatteryConfig is not None
    assert RobustVRConfig is not None
    assert TestOutcome is not None
    assert BatteryOutcome is not None


def test_run_weak_form_battery_importable_from_root() -> None:
    """run_weak_form_battery must be importable from package root."""
    assert run_weak_form_battery is not None


def test_all_contains_exact_public_api() -> None:
    """__all__ must match the required minimal API exactly."""
    assert hasattr(variance_test, "__all__")
    assert set(variance_test.__all__) == EXPECTED_PUBLIC_API
