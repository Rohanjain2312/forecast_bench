"""Shared machinery for every model: fold bookkeeping and quantiles from residuals.

Two things live here because getting them wrong in one model and right in another would
make the panel incomparable:

1. **Fold bookkeeping.** :class:`BaseForecaster` records the origin it was fitted on, which
   ``tests/test_no_leakage.py`` asserts against the fold. Subclasses implement ``_fit`` and
   ``_quantile_paths`` and never touch that bookkeeping.
2. **Quantile construction.** A model that emits a point forecast plus a normal
   approximation is not comparable to one that emits an empirical predictive distribution.
   The helpers here are the two sanctioned ways to turn a point model into a quantile
   model, and both are recomputed per fold.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from forecast_bench.backtest.protocol import QuantileForecast
from forecast_bench.config import QUANTILE_GRID

logger = logging.getLogger(__name__)


def enforce_monotonic(quantiles: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sort quantile values across levels at each step, removing any crossing.

    Args:
        quantiles: Mapping of level to a path.

    Returns:
        A mapping with the same keys whose values are non-decreasing in the level at every
        step.

    Note:
        Crossings arise legitimately from finite-sample quantile estimation. Sorting is the
        standard repair and changes nothing when there is no crossing. A crossing that
        survives into a :class:`QuantileForecast` is a bug, so it is fixed here rather than
        tolerated downstream.
    """
    levels = sorted(quantiles)
    stacked = np.vstack([np.asarray(quantiles[level], dtype=float) for level in levels])
    stacked.sort(axis=0)
    return {level: stacked[position] for position, level in enumerate(levels)}


def empirical_change_quantiles(
    values: np.ndarray,
    horizon: int,
    levels: list[float] = QUANTILE_GRID,
) -> dict[float, np.ndarray]:
    """Quantiles of h-step changes, measured in the training window.

    For each step ``h``, the empirical distribution of ``y[t + h] - y[t]`` over the training
    window. This is what turns a random walk into an honest probabilistic forecast: the
    spread widens with the horizon because the data says it does, not because a normal
    assumption says it should.

    Args:
        values: Training values, in observation order.
        horizon: Number of steps to produce.
        levels: Quantile levels.

    Returns:
        Mapping of level to a path of h-step change quantiles, length ``horizon``.

    Raises:
        ValueError: If the window is too short to measure a change at every step.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= horizon:
        raise ValueError(
            f"Need more than {horizon} observations to measure h-step changes; got "
            f"{len(values)}."
        )

    paths = {level: np.empty(horizon) for level in levels}
    for step in range(1, horizon + 1):
        changes = values[step:] - values[:-step]
        for level in levels:
            paths[level][step - 1] = float(np.quantile(changes, level))
    return paths


def scaled_residual_quantiles(
    residuals: np.ndarray,
    horizon: int,
    levels: list[float] = QUANTILE_GRID,
) -> dict[float, np.ndarray]:
    """One-step residual quantiles widened by ``sqrt(h)``.

    The standard way to give an OLS-style point model a predictive distribution when the
    residuals are approximately serially uncorrelated: the h-step forecast error variance
    accumulates roughly linearly, so the standard deviation grows as ``sqrt(h)``.

    Args:
        residuals: In-sample one-step residuals from the training window.
        horizon: Number of steps to produce.
        levels: Quantile levels.

    Returns:
        Mapping of level to an offset path, length ``horizon``.

    Note:
        Recomputed per fold. A residual quantile cached across folds is a fitted object
        that crossed a fold boundary, which is exactly what
        ``tests/test_no_leakage.py`` check 4 looks for.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("No finite residuals to build quantiles from.")

    scale = np.sqrt(np.arange(1, horizon + 1, dtype=float))
    return {level: float(np.quantile(residuals, level)) * scale for level in levels}


class BaseForecaster(ABC):
    """Common base for every model in the panel.

    Handles target-column resolution and fold bookkeeping so that no subclass can get
    them subtly different.

    Attributes:
        model_id: Stable identifier used as the results-table key.
        target_column: Column holding the target. ``None`` means "the first column",
            which is the convention the merge layer guarantees.
        fitted_on_origin: Origin of the fold this instance was fitted on.
    """

    model_id: str = "BaseForecaster"

    def __init__(self, target_column: str | None = None) -> None:
        """Initialise an unfitted model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
        """
        self.target_column = target_column
        self.fitted_on_origin: pd.Timestamp | None = None
        self._series: pd.Series | None = None

    def fit(self, train: pd.DataFrame, origin: pd.Timestamp) -> None:
        """Fit on data at or before ``origin``.

        Args:
            train: Training frame for this fold. Its index never exceeds ``origin``.
            origin: The last timestamp this model is allowed to have seen.

        Raises:
            ValueError: If the resolved target column is empty after dropping NaNs.
        """
        name = self.target_column or str(train.columns[0])
        series = train[name].dropna()
        if series.empty:
            raise ValueError(f"{self.model_id}: no observations in column {name!r}")

        self.fitted_on_origin = origin
        self._series = series
        self._fit(train, series, origin)

    def predict(self, horizon: int, index: pd.DatetimeIndex) -> QuantileForecast:
        """Produce a quantile forecast over the supplied dates.

        Args:
            horizon: Number of steps to forecast.
            index: The dates being forecast.

        Returns:
            A validated, non-crossing forecast.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.fitted_on_origin is None:
            raise RuntimeError(f"{self.model_id}: predict() called before fit()")

        quantiles = enforce_monotonic(self._quantile_paths(horizon))
        forecast = QuantileForecast(
            origin=self.fitted_on_origin,
            index=pd.DatetimeIndex(index)[:horizon],
            quantiles=quantiles,
            model_id=self.model_id,
        )
        forecast.assert_monotonic()
        return forecast

    @abstractmethod
    def _fit(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Fit the model. Called by :meth:`fit` after bookkeeping.

        Args:
            train: The full training frame, including any covariates.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """

    @abstractmethod
    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Return one path per quantile level.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to an array of length ``horizon``.
        """
