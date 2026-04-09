#!/usr/bin/env python3
"""Offline rolling-window example for the weak-form battery."""

from __future__ import annotations

import numpy as np

from variance_test import BatteryConfig, run_weak_form_battery


rng = np.random.default_rng(123)
returns = rng.normal(0.0, 0.01, size=1200)

config = BatteryConfig(
    input_kind="returns",
    q_list=(2, 4),
    ljung_box_lags=(5, 10),
    rolling_window=120,
    rolling_step=20,
)
outcome = run_weak_form_battery(returns, config=config)

rolling = outcome.rolling or {}
vr_q2 = rolling.get("tests", {}).get("variance_ratio_q2", [])

print(f"n_windows={rolling.get('n_windows')}")
print("first_two_variance_ratio_q2_entries=")
print(vr_q2[:2])
