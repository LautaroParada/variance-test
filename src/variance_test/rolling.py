"""Internal rolling-window helpers for weak-form battery tests."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from statsmodels.stats import diagnostic

from .core import EMH
from .data import normalize_series


def _iter_rolling_windows(n_raw: int, window: int, step: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` index pairs for rolling windows over raw input."""
    for start in range(0, n_raw - window + 1, step):
        yield start, start + window


def _non_computable_result(start: int, end: int, warning: str) -> dict[str, object]:
    """Build a standardized rolling record for non-computable windows."""
    return {
        "start": start,
        "end": end,
        "statistic": None,
        "p_value": None,
        "reject_null": None,
        "warning": warning,
    }


def _compute_rolling_variance_ratio(normalized, config) -> dict[str, list[dict[str, object]]]:
    """Compute rolling variance ratio outcomes for every q in the config list."""
    emh = EMH()
    per_test: dict[str, list[dict[str, object]]] = {
        f"variance_ratio_q{q}": [] for q in config.q_list
    }

    for start, end in _iter_rolling_windows(
        n_raw=normalized.n_raw,
        window=config.rolling_window,
        step=config.rolling_step,
    ):
        window_raw = normalized.raw[start:end]
        window_normalized = normalize_series(window_raw, input_kind=config.input_kind)

        for q in config.q_list:
            key = f"variance_ratio_q{q}"
            if q >= window_normalized.log_prices.size:
                per_test[key].append(
                    _non_computable_result(
                        start=start,
                        end=end,
                        warning="Variance ratio not computable for this window: q must be smaller than window log-price length.",
                    )
                )
                continue

            try:
                z_score, p_value = emh.vrt(
                    X=window_normalized.log_prices,
                    q=q,
                    heteroskedastic=False,
                    centered=True,
                    unbiased=True,
                    annualize=False,
                    input_kind="log_prices",
                    alternative="two-sided",
                )
                p_float = float(p_value)
                per_test[key].append(
                    {
                        "start": start,
                        "end": end,
                        "statistic": float(z_score),
                        "p_value": p_float,
                        "reject_null": bool(p_float < config.alpha),
                        "warning": None,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive by contract
                per_test[key].append(
                    _non_computable_result(
                        start=start,
                        end=end,
                        warning=f"Variance ratio failed for this window: {exc}",
                    )
                )

    return per_test


def _compute_rolling_ljung_box(
    normalized,
    config,
    *,
    field: str,
    name: str,
) -> list[dict[str, object]]:
    """Compute rolling Ljung-Box outcomes for one normalized series field."""
    outcomes: list[dict[str, object]] = []
    max_lag = max(config.ljung_box_lags)

    for start, end in _iter_rolling_windows(
        n_raw=normalized.n_raw,
        window=config.rolling_window,
        step=config.rolling_step,
    ):
        window_raw = normalized.raw[start:end]
        window_normalized = normalize_series(window_raw, input_kind=config.input_kind)
        series = getattr(window_normalized, field)

        if max_lag >= window_normalized.n_returns:
            outcomes.append(
                _non_computable_result(
                    start=start,
                    end=end,
                    warning=(
                        f"{name} not computable for this window: "
                        "max(config.ljung_box_lags) must be smaller than window n_returns."
                    ),
                )
            )
            continue

        try:
            result = diagnostic.acorr_ljungbox(
                series,
                lags=list(config.ljung_box_lags),
                return_df=True,
            )

            lb_stat = result["lb_stat"].to_numpy(dtype=float)
            lb_pvalue = result["lb_pvalue"].to_numpy(dtype=float)

            finite_pairs = [
                (int(lag), float(stat_value), float(p_value))
                for lag, stat_value, p_value in zip(config.ljung_box_lags, lb_stat, lb_pvalue)
                if np.isfinite(stat_value) and np.isfinite(p_value)
            ]

            if not finite_pairs:
                outcomes.append(
                    _non_computable_result(
                        start=start,
                        end=end,
                        warning=f"{name} not computable for this window: Ljung-Box statistics are not finite.",
                    )
                )
                continue

            selected_lag, selected_stat, selected_p = min(
                finite_pairs,
                key=lambda x: (x[2], x[0]),
            )

            outcomes.append(
                {
                    "start": start,
                    "end": end,
                    "statistic": float(selected_stat),
                    "p_value": float(selected_p),
                    "reject_null": bool(selected_p < config.alpha),
                    "warning": None,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive by contract
            outcomes.append(
                _non_computable_result(
                    start=start,
                    end=end,
                    warning=f"{name} failed for this window: {exc}",
                )
            )

    return outcomes


def _build_rolling_results(normalized, config) -> dict[str, object]:
    """Build rolling payload for supported subtests in battery v1."""
    vr_results = _compute_rolling_variance_ratio(normalized=normalized, config=config)
    lb_returns = _compute_rolling_ljung_box(
        normalized=normalized,
        config=config,
        field="returns",
        name="ljung_box_returns",
    )
    lb_sq_returns = _compute_rolling_ljung_box(
        normalized=normalized,
        config=config,
        field="squared_returns",
        name="ljung_box_squared_returns",
    )

    tests: dict[str, list[dict[str, object]]] = {
        **vr_results,
        "ljung_box_returns": lb_returns,
        "ljung_box_squared_returns": lb_sq_returns,
    }

    n_windows = len(next(iter(tests.values()))) if tests else 0

    return {
        "window": config.rolling_window,
        "step": config.rolling_step,
        "n_windows": n_windows,
        "tests": tests,
    }
