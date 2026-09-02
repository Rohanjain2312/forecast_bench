"""Scaffold smoke test: the package imports and reports a version.

This is deliberately trivial. It exists so that the test suite is non-empty from the first
commit, which means CI is genuinely green rather than vacuously green.
"""

import forecast_bench
from forecast_bench.version import __version__


def test_package_exposes_version() -> None:
    """The package re-exports the version string from ``version.py``."""
    assert forecast_bench.__version__ == __version__


def test_version_is_semver_like() -> None:
    """The version is three dot-separated numeric components."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
