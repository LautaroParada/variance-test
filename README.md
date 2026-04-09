# variance-test

## Short description

`variance-test` provides a focused toolkit for weak-form market efficiency diagnostics. The current package includes a variance ratio test, common series normalization utilities, a weak-form battery, rolling-window support for a subset of battery tests, and offline simulation utilities with runnable examples.

## Installation

This repository is documented for the `0.1.0` release target.

Install from PyPI (target command for the public package):

```bash
pip install variance-test
```

Install locally in editable mode:

```bash
pip install -e .
```

## Current scope

The current public scope includes:

- `EMH().vrt(...)` for variance ratio testing.
- `normalize_series(...)` for shared input normalization.
- `run_weak_form_battery(...)` for the weak-form battery v1.
- Rolling support for:
  - variance ratio multi-q
  - Ljung-Box returns
  - Ljung-Box squared returns
- Offline examples under `examples/`.

## Weak-form battery

The weak-form battery v1 currently includes:

- variance ratio multi-q
- Holm summary for the VR family
- Ljung-Box returns
- Ljung-Box squared returns
- runs test on signs
- ARCH LM

Rolling support in this version does not apply to:

- Holm summary
- runs test
- ARCH LM

## Usage

### Variance ratio with `EMH().vrt(...)`

```python
import numpy as np
from variance_test import EMH

emh = EMH()

log_prices = np.log(np.cumsum(np.random.normal(0.0, 1.0, 600)) + 5000.0)
z_score, p_value = emh.vrt(
    X=log_prices,
    q=4,
    input_kind="log_prices",
)
print(z_score, p_value)
```

### Weak-form battery with `run_weak_form_battery(...)`

```python
import numpy as np
from variance_test import BatteryConfig, run_weak_form_battery

rng = np.random.default_rng(42)
returns = rng.normal(0.0, 0.01, size=2000)

config = BatteryConfig(
    input_kind="returns",
    q_list=(2, 4, 8),
    ljung_box_lags=(5, 10, 20),
)
outcome = run_weak_form_battery(returns, config=config)
print(outcome.multiple_testing["battery_summary"])
```

### Weak-form battery with rolling

```python
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
print(outcome.rolling["n_windows"])
```

## Examples

Available offline scripts:

- `examples/basic_battery.py`
- `examples/rolling_battery.py`
- `examples/variance_ratio_only.py`

## Interpretation notes

The battery provides evidence against compatibility with weak-form efficiency under this battery design. It is not definitive proof of market inefficiency.

## Release status

This documentation corresponds to release `0.1.0`. The repository is prepared for a `0.1.0` publication workflow. Effective publication depends on the maintainer environment and credentials.

## Local release process

Manual maintainer commands for a local release flow:

```bash
python -m build
twine check dist/*
twine upload dist/*
```

These are manual maintainer steps and are not executed automatically by this repository.

## Development notes

- The test suite should pass before release.
- Examples and documentation should remain aligned with the public API.
