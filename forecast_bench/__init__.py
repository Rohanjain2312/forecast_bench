"""forecast_bench — a leakage-safe forecasting benchmark.

Classical statistical models and time-series foundation models are compared on real
financial data, with every model scored through identical code. See ``PREREGISTRATION.md``
for the decision rules, which were committed before any result existed.
"""

from forecast_bench.version import __version__

__all__ = ["__version__"]
