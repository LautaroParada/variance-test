#!/usr/bin/env python3
"""Minimal EMH.vrt usage example for log-prices and returns inputs."""

from __future__ import annotations

import numpy as np

from variance_test import EMH


rng = np.random.default_rng(7)
returns = rng.normal(0.0, 0.01, size=1000)
log_prices = np.concatenate(([0.0], np.cumsum(returns)))

emh = EMH()

z_log, p_log = emh.vrt(X=log_prices, q=4, input_kind="log_prices")
z_ret, p_ret = emh.vrt(X=returns, q=4, input_kind="returns")

print(f"log_prices mode -> z={z_log:.6f}, p={p_log:.6f}")
print(f"returns mode    -> z={z_ret:.6f}, p={p_ret:.6f}")
