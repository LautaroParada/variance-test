"""Tests for build metadata and src-layout configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_exists() -> None:
    """The project must define pyproject.toml."""
    assert Path("pyproject.toml").exists()


def test_project_name_is_variance_test() -> None:
    """The distribution name must match the required value."""
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "variance-test"


def test_build_system_contains_setuptools_and_wheel() -> None:
    """Build requirements must include setuptools and wheel."""
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    requires = data["build-system"]["requires"]

    assert any(req.startswith("setuptools") for req in requires)
    assert any(req.startswith("wheel") for req in requires)


def test_setuptools_find_where_is_src() -> None:
    """Setuptools package discovery must be configured for src layout."""
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    where = data["tool"]["setuptools"]["packages"]["find"]["where"]

    assert where == ["src"]
