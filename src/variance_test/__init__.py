"""Public package interface for variance ratio testing."""

from .battery import run_weak_form_battery
from .core import EMH
from .data import NormalizedSeries, normalize_series
from .models import BatteryConfig, BatteryOutcome, TestOutcome
from .price_paths import PricePaths
from .simulation import (
    SimulationConfig,
    SimulationResults,
    compute_vrt_statistics,
    run_simulation,
    simulate_price_processes,
)
from .visuals import VRTVisuals

__all__ = [
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
    "TestOutcome",
    "BatteryOutcome",
    "run_weak_form_battery",
]
