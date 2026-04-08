"""Tests for series normalization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from variance_test import normalize_series


def test_normalize_log_prices_outputs_expected_derived_arrays() -> None:
    """log_prices input must produce consistent derived return arrays."""
    log_prices = np.array([0.0, 0.1, 0.15, 0.12], dtype=float)

    ns = normalize_series(log_prices, input_kind="log_prices")

    expected_returns = np.diff(log_prices)
    np.testing.assert_allclose(ns.returns, expected_returns)
    np.testing.assert_allclose(ns.squared_returns, expected_returns**2)
    np.testing.assert_allclose(ns.signs, np.sign(expected_returns))
    assert ns.n_raw == len(log_prices)
    assert ns.n_returns == len(log_prices) - 1


def test_normalize_returns_reconstructs_synthetic_log_prices() -> None:
    """returns input must reconstruct log-prices with origin 0.0."""
    returns = np.array([0.01, -0.02, 0.03], dtype=float)

    ns = normalize_series(returns, input_kind="returns")

    np.testing.assert_allclose(ns.returns, returns)
    assert ns.log_prices[0] == 0.0
    np.testing.assert_allclose(ns.log_prices[1:], np.cumsum(returns))
    assert ns.n_raw == len(returns)
    assert ns.n_returns == len(returns)


def test_normalize_series_rejects_invalid_input_kind() -> None:
    """An invalid input_kind must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_series([0.0, 0.1], input_kind="invalid")


def test_normalize_series_rejects_empty_array() -> None:
    """An empty series must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_series([], input_kind="returns")


def test_normalize_series_rejects_nan_and_inf() -> None:
    """Series containing NaN or inf must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_series([0.0, np.nan], input_kind="log_prices")
    with pytest.raises(ValueError):
        normalize_series([0.0, np.inf], input_kind="log_prices")


def test_normalize_series_rejects_two_dimensional_input() -> None:
    """A two-dimensional input must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_series(np.array([[0.0, 0.1], [0.2, 0.3]]), input_kind="log_prices")


def test_normalize_series_rejects_too_short_log_prices() -> None:
    """log_prices input with one observation must raise ValueError."""
    with pytest.raises(ValueError):
        normalize_series([0.0], input_kind="log_prices")
