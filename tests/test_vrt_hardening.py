"""Tests for VRT input hardening and p-value alternatives."""

from __future__ import annotations

import numpy as np
import scipy.stats as st
import pytest

from variance_test import EMH


def _sample_log_prices() -> np.ndarray:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=200)
    return np.concatenate([[0.0], np.cumsum(returns)])


def test_vrt_legacy_signature_returns_two_numeric_values() -> None:
    """Legacy call must still return (z_score, p_value)."""
    log_prices = _sample_log_prices()

    result = EMH().vrt(X=log_prices, q=3)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])


def test_vrt_default_two_sided_pvalue_matches_formula() -> None:
    """Default p-value must be two-sided and based on abs(z)."""
    log_prices = _sample_log_prices()
    z_score, p_value = EMH().vrt(X=log_prices, q=3)

    expected = 2 * (1 - st.norm.cdf(abs(z_score)))
    np.testing.assert_allclose(p_value, expected, atol=1e-12)


def test_vrt_one_sided_alternatives_match_tail_probabilities() -> None:
    """greater and less alternatives must match one-sided normal tails."""
    log_prices = _sample_log_prices()
    z_score, _ = EMH().vrt(X=log_prices, q=3, alternative="two-sided")
    _, p_greater = EMH().vrt(X=log_prices, q=3, alternative="greater")
    _, p_less = EMH().vrt(X=log_prices, q=3, alternative="less")

    np.testing.assert_allclose(p_greater, 1 - st.norm.cdf(z_score), atol=1e-12)
    np.testing.assert_allclose(p_less, st.norm.cdf(z_score), atol=1e-12)


def test_vrt_rejects_invalid_alternative() -> None:
    """Invalid alternative must raise ValueError."""
    log_prices = _sample_log_prices()
    with pytest.raises(ValueError):
        EMH().vrt(X=log_prices, q=3, alternative="invalid")


def test_vrt_accepts_returns_input_and_returns_finite_values() -> None:
    """returns input_kind must be accepted with finite output values."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.01, size=120)

    z_score, p_value = EMH().vrt(X=returns, q=2, input_kind="returns")

    assert np.isfinite(z_score)
    assert np.isfinite(p_value)


def test_vrt_returns_mode_matches_synthetic_log_prices_mode() -> None:
    """returns mode must match equivalent synthetic log-prices mode."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=100)
    log_prices_synthetic = np.concatenate([[0.0], np.cumsum(returns)])

    z1, p1 = EMH().vrt(X=returns, q=4, input_kind="returns", annualize=False)
    z2, p2 = EMH().vrt(X=log_prices_synthetic, q=4, input_kind="log_prices", annualize=False)

    assert abs(z1 - z2) < 1e-10
    assert abs(p1 - p2) < 1e-10


def test_vrt_rejects_invalid_input_kind() -> None:
    """Invalid input_kind must raise ValueError."""
    log_prices = _sample_log_prices()
    with pytest.raises(ValueError):
        EMH().vrt(X=log_prices, q=3, input_kind="invalid")
