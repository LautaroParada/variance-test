"""Package import smoke tests."""

from __future__ import annotations


def test_import_variance_test_package() -> None:
    """The root package must be importable."""
    import variance_test

    assert variance_test is not None


def test_import_public_symbols_from_root_package() -> None:
    """All required public symbols should be importable from package root."""
    from variance_test import (  # noqa: F401
        EMH,
        PricePaths,
        SimulationConfig,
        SimulationResults,
        VRTVisuals,
        compute_vrt_statistics,
        run_simulation,
        simulate_price_processes,
    )


def test_root_package_does_not_expose_empirical_application() -> None:
    """The empirical example must not be part of the root package API."""
    import variance_test

    assert not hasattr(variance_test, "empirical_application")
