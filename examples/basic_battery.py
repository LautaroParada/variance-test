#!/usr/bin/env python3
"""Basic offline example for the weak-form battery without rolling windows."""

from __future__ import annotations

import numpy as np

from variance_test import BatteryConfig, run_weak_form_battery


rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, size=1500)
config = BatteryConfig(input_kind="returns", q_list=(2, 4, 8), ljung_box_lags=(5, 10, 20))
outcome = run_weak_form_battery(returns, config=config)

print("Battery summary:")
print(outcome.multiple_testing["battery_summary"])
