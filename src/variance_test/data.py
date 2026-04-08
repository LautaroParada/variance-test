"""Data normalization utilities for variance-test inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedSeries:
    """Canonical normalized representation for input series."""

    input_kind: str
    raw: np.ndarray
    returns: np.ndarray
    log_prices: np.ndarray
    squared_returns: np.ndarray
    signs: np.ndarray
    n_raw: int
    n_returns: int


def normalize_series(series, input_kind: str = "log_prices") -> NormalizedSeries:
    """Validate and normalize an input series as log-prices or returns.

    Args:
        series: One-dimensional array-like numeric sequence.
        input_kind: Semantic type of the input series. Must be
            ``"log_prices"`` or ``"returns"``.

    Returns:
        A :class:`NormalizedSeries` instance with aligned derived arrays.

    Raises:
        ValueError: If ``input_kind`` is invalid, the data is not one-dimensional,
            contains non-finite values, or is too short for the selected kind.
    """

    if input_kind not in {"log_prices", "returns"}:
        raise ValueError("input_kind must be 'log_prices' or 'returns'.")

    raw = np.asarray(series, dtype=float)
    if raw.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    if raw.size == 0:
        raise ValueError("Input series cannot be empty.")
    if not np.isfinite(raw).all():
        raise ValueError("Input series must contain only finite values.")

    if input_kind == "log_prices":
        if raw.size < 2:
            raise ValueError("log_prices input requires at least 2 observations.")

        log_prices = raw
        returns = np.diff(log_prices)
        n_raw = int(log_prices.size)
        n_returns = int(returns.size)
    else:
        if raw.size < 1:
            raise ValueError("returns input requires at least 1 observation.")

        returns = raw
        log_prices = np.concatenate(([0.0], np.cumsum(returns)))
        n_raw = int(returns.size)
        n_returns = int(returns.size)

    squared_returns = returns ** 2
    signs = np.sign(returns)

    return NormalizedSeries(
        input_kind=input_kind,
        raw=raw,
        returns=returns,
        log_prices=log_prices,
        squared_returns=squared_returns,
        signs=signs,
        n_raw=n_raw,
        n_returns=n_returns,
    )
