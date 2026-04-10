"""Offline example for robust weak-form battery diagnostics."""

from __future__ import annotations

import numpy as np

from variance_test import BatteryConfig, RobustVRConfig, run_weak_form_battery


def main() -> None:
    """Run robust battery on synthetic data and print robust outputs."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=5000)

    config = BatteryConfig(
        input_kind="returns",
        alpha=0.05,
        q_list=(2, 4, 8, 16),
        ljung_box_lags=(5, 10, 20),
        runs_test=True,
        arch_lm_lags=5,
        robust_vr=RobustVRConfig(
            enabled=True,
            enable_avr=True,
            enable_wbavr=True,
            enable_chow_denning=True,
            bootstrap_reps=999,
            bootstrap_seed=42,
        ),
    )

    outcome = run_weak_form_battery(returns, config=config)

    print("AVR:", outcome.tests["avr"])
    print("WBAVR:", outcome.tests["wbavr"])
    print("CHOW_DENNING:", outcome.tests["chow_denning"])
    print("robust_vr_summary:", outcome.multiple_testing["robust_vr_summary"])


if __name__ == "__main__":
    main()
