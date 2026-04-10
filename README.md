# variance-test ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[![PyPI version](https://img.shields.io/pypi/v/variance-test?style=flat)](https://pypi.org/project/variance-test/)
[![Python versions](https://img.shields.io/pypi/pyversions/variance-test?style=flat)](https://pypi.org/project/variance-test/)
[![GitHub stars](https://img.shields.io/github/stars/LautaroParada/variance-test?style=flat)](https://github.com/LautaroParada/variance-test/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/LautaroParada/variance-test?style=flat)](https://github.com/LautaroParada/variance-test/commits/master)
[![License](https://img.shields.io/github/license/LautaroParada/variance-test?style=flat)](https://github.com/LautaroParada/variance-test/blob/master/LICENSE)

`variance-test` is an offline Python package for weak-form efficiency diagnostics on return series. It provides classic variance-ratio testing and a structured weak-form battery with optional robust variance-ratio tools.

```python
from variance_test import BatteryConfig, run_weak_form_battery

outcome = run_weak_form_battery(returns, config=BatteryConfig(input_kind="returns"))
print(outcome.multiple_testing["battery_summary"])
```

## Installation

```bash
pip install variance-test
```

## Quick start

### Variance ratio test only

```python
import numpy as np
from variance_test import EMH

rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, size=2000)

emh = EMH()
z_score, p_value = emh.vrt(
    X=returns,
    q=4,
    input_kind="returns",
    heteroskedastic=True,
    centered=True,
    unbiased=True,
    annualize=False,
    alternative="two-sided",
)

print("z_score:", z_score)
print("p_value:", p_value)
```

### Full weak-form battery (classic)

```python
import numpy as np
from variance_test import BatteryConfig, run_weak_form_battery

rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, size=2000)

config = BatteryConfig(
    input_kind="returns",
    alpha=0.05,
    q_list=(2, 4, 8),
    ljung_box_lags=(5, 10, 20),
    runs_test=True,
    arch_lm_lags=5,
)

outcome = run_weak_form_battery(returns, config=config)
print("VR family rejects:", outcome.tests["variance_ratio_holm"].reject_null)
print(outcome.multiple_testing["battery_summary"])
```

### Battery with rolling windows

```python
import numpy as np
from variance_test import BatteryConfig, run_weak_form_battery

rng = np.random.default_rng(123)
returns = rng.normal(0.0, 0.01, size=1500)

config = BatteryConfig(
    input_kind="returns",
    q_list=(2, 4),
    ljung_box_lags=(5, 10),
    runs_test=True,
    arch_lm_lags=5,
    rolling_window=120,
    rolling_step=20,
)

outcome = run_weak_form_battery(returns, config=config)
print("n_windows:", outcome.rolling["n_windows"])
print("first_vr_windows:", outcome.rolling["tests"]["variance_ratio_q2"][:2])
```

## Robust variance-ratio tools (v0.2.0)

Release `0.2.0` adds an **opt-in** robust VR family via `RobustVRConfig`:

- `avr` (automatic variance-ratio)
- `wbavr` (wild-bootstrap automatic variance-ratio)
- `chow_denning` (Chow-Denning-style max test)

Robust tests are disabled by default. Enable them explicitly:

```python
from variance_test import BatteryConfig, RobustVRConfig, run_weak_form_battery

config = BatteryConfig(
    input_kind="returns",
    q_list=(2, 4, 8, 16),
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
print(outcome.multiple_testing["robust_vr_summary"])
```

Important calibration note for this release:

- `chow_denning` uses **Sidak-normal** familywise calibration (`"sidak_normal"`).
- This is **not exact SMM calibration**.
- Sidak-normal treats familywise tails as if built from independent normal tails.
- Since per-q VR Z statistics come from overlapping data and are not independent, this is an approximation and should be described as conservative rather than exact.

Rolling support for robust tests (`avr`, `wbavr`, `chow_denning`) is **not included** in `0.2.0`.

## What the battery includes

| Test | Diagnostic target | Rolling support |
|---|---|---|
| Variance ratio multi-q | Mean dependence at multiple horizons | Yes |
| Holm-Bonferroni correction | Family-wise error control for VR tests | No |
| Ljung-Box on returns | Serial dependence in returns | Yes |
| Ljung-Box on squared returns | Serial dependence in volatility | Yes |
| Runs test on signs | Non-random sign patterns | No |
| ARCH LM | Conditional heteroskedasticity | No |
| AVR | Automatic variance-ratio diagnostic | No |
| WBAVR | Wild-bootstrap automatic variance-ratio diagnostic | No |
| chow_denning | Chow-Denning-style max family diagnostic | No |

The battery provides statistical evidence under its implemented diagnostics. Rejection does not prove market inefficiency.

## Input formats

- `input_kind="returns"` for one-period log-returns.
- `input_kind="log_prices"` for cumulative log-prices.

The package normalizes both pathways consistently.

## References

- Lo, A.W. and MacKinlay, A.C. (1988). *Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test.*
- Lo, A.W. and MacKinlay, A.C. (1989). *The Size and Power of the Variance Ratio Test in Finite Samples: A Monte Carlo Investigation.*
- Choi, I. (1999). *Testing the random walk hypothesis for real exchange rates.* (Automatic variance-ratio framing.)
- Kim, J.H. (2006). *Wild bootstrapping variance ratio tests.*
- Chow, K.V. and Denning, K.C. (1993). *A simple multiple variance ratio test.* (This release uses a Chow-Denning-style framing with Sidak-normal approximation.)
