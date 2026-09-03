"""Shared machinery for every model: fold bookkeeping and quantiles from residuals.

Two things live here because getting them wrong in one model and right in another would
make the panel incomparable:

1. **Fold bookkeeping.** :class:`BaseForecaster` records the origin it was fitted on, which
   ``tests/test_no_leakage.py`` asserts against the fold. Subclasses implement ``_fit`` and
   ``_quantile_paths`` and never touch that bookkeeping.
2. **Quantile construction.** A model that emits a point forecast plus a normal
   approximation is not comparable to one that emits an empirical predictive distribution.
   The two helpers here are the sanctioned ways to turn a point model into a quantile
   model, and both are recomputed per fold. Both *measure* the spread from data rather
   than assuming a shape for it — see :func:`stepwise_residual_quantiles` for what went
   wrong when one of them assumed instead.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from forecast_bench.backtest.protocol import QuantileForecast
from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON, QUANTILE_GRID

logger = logging.getLogger(__name__)


#: Trading days of *distinct fine-tuning or training material* per sample-efficiency
#: slice (DECISIONS.md D9) -- how many separate forecast origins a slice gives a model to
#: learn from, not a raw observation count. See :func:`sample_efficiency_window_size`.
SAMPLE_EFFICIENCY_DAYS: dict[str, int | None] = {
    "1y": 252,
    "3y": 756,
    "10y": 2520,
    "full": None,
}


def sample_efficiency_window_size(
    label: str,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = MAX_HORIZON,
) -> int | None:
    """Convert a sample-efficiency label into a raw observation count.

    Args:
        label: Key into :data:`SAMPLE_EFFICIENCY_DAYS`.
        context_length: Context length every model in the sweep is fixed to.
        horizon: Forecast horizon every model in the sweep is fixed to.

    Returns:
        Raw observations to keep, ending at the fold's origin. ``None`` for ``"full"``,
        meaning no truncation.

    Raises:
        KeyError: If ``label`` is not a recognised slice.

    Note:
        A "1 year" slice does **not** mean literally 252 raw trading days. Every model in
        the sweep — Chronos-2/Bolt fine-tuning, N-BEATS, the DeepAR-class LSTM — has its
        context length fixed at :data:`~forecast_bench.config.CONTEXT_LENGTH` (512), so
        that context length is not a confound across model classes (IMPLEMENTATION_PLAN.md
        §4c). A window of only 252 raw observations is *shorter than the context window
        itself* and cannot supply even one ``(context, target)`` training example,
        regardless of which model receives it.

        "1y" instead means 252 distinct forecast origins' worth of material *beyond* the
        one context-plus-horizon window every slice needs at minimum:
        ``context_length + horizon + days - 1`` raw observations. This was found live, on
        the first Colab run of the sample-efficiency sweep: see
        docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    if label not in SAMPLE_EFFICIENCY_DAYS:
        raise KeyError(
            f"Unknown training window {label!r}; expected one of "
            f"{sorted(SAMPLE_EFFICIENCY_DAYS)}"
        )
    days = SAMPLE_EFFICIENCY_DAYS[label]
    if days is None:
        return None
    return context_length + horizon + days - 1


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


def stepwise_residual_quantiles(
    residuals: np.ndarray,
    levels: list[float] = QUANTILE_GRID,
) -> dict[float, np.ndarray]:
    """Quantiles of a model's own h-step-ahead residuals, measured per step.

    Args:
        residuals: Shape ``(horizon, n_origins)``. Row ``h - 1`` holds the model's
            h-step-ahead errors across training-window origins. NaNs are ignored.
        levels: Quantile levels.

    Returns:
        Mapping of level to an offset path of length ``horizon``.

    Raises:
        ValueError: If the input is not a matrix, or a step has no finite residuals.

    Note:
        This replaced a ``sqrt(h)`` widening of one-step residuals, which assumes forecast
        error variance grows linearly in the horizon. That holds for an *integrated*
        process and fails badly for a mean-reverting one. Measured on SPY log realized
        variance over 2000-2014, the spread of h-step changes grows 1.23x by h=21 where
        ``sqrt(h)`` assumes 4.58x, so HAR's 21-step intervals came out roughly 3.7 times
        too wide and covered 100% of actuals. See docs/planning/PROGRESS_NOTES.md Step 14.

        Measuring the spread instead of assuming it also captures asymmetry, which matters
        for :class:`~forecast_bench.models.classical.har.HAR`, whose residuals live in
        right-skewed variance space.

        Recomputed per fold. A residual quantile cached across folds is a fitted object
        that crossed a fold boundary, which ``tests/test_no_leakage.py`` check 4 looks for.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError(
            f"Expected a (horizon, n_origins) matrix, got shape {residuals.shape}"
        )

    horizon = residuals.shape[0]
    paths = {level: np.empty(horizon) for level in levels}
    for step in range(horizon):
        finite = residuals[step][np.isfinite(residuals[step])]
        if finite.size == 0:
            raise ValueError(f"No finite residuals at step {step + 1}")
        for level in levels:
            paths[level][step] = float(np.quantile(finite, level))
    return paths


class BaseForecaster(ABC):
    """Common base for every model in the panel.

    Handles target-column resolution and fold bookkeeping so that no subclass can get
    them subtly different.

    Attributes:
        model_id: Stable identifier used as the results-table key.
        target_column: Column holding the target. ``None`` means "the first column",
            which is the convention the merge layer guarantees.
        fitted_on_origin: Origin of the fold this instance last conditioned on. Always
            the current fold's origin.
        parameters_fitted_on_origin: Origin at which the parameters were last estimated.
            Under a block cadence this lags ``fitted_on_origin`` within a block, which is
            exactly the intended difference between the two cadences.
    """

    model_id: str = "BaseForecaster"

    def __init__(self, target_column: str | None = None) -> None:
        """Initialise an unfitted model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
        """
        self.target_column = target_column
        self.fitted_on_origin: pd.Timestamp | None = None
        self.parameters_fitted_on_origin: pd.Timestamp | None = None
        self._parameters_fitted = False
        self._series: pd.Series | None = None

    def fit(
        self,
        train: pd.DataFrame,
        origin: pd.Timestamp,
        refit_parameters: bool = True,
    ) -> None:
        """Fit on data at or before ``origin``.

        Called on every fold. Splits into two steps so that the refit cadence can govern
        parameters without ever freezing conditioning data.

        Args:
            train: Training frame for this fold. Its index never exceeds ``origin``.
            origin: The last timestamp this model is allowed to have seen.
            refit_parameters: Re-estimate parameters when ``True``; otherwise keep them
                and only refresh the conditioning state.

        Raises:
            ValueError: If the resolved target column is empty after dropping NaNs.
        """
        name = self.target_column or str(train.columns[0])
        series = train[name].dropna()
        if series.empty:
            raise ValueError(f"{self.model_id}: no observations in column {name!r}")

        self.fitted_on_origin = origin
        self._series = series

        if refit_parameters or not self._parameters_fitted:
            self._estimate_parameters(train, series, origin)
            self._parameters_fitted = True
            self.parameters_fitted_on_origin = origin

        self._update_state(train, series, origin)

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
    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Re-estimate parameters from this fold's training window.

        Everything learned from data belongs here: regression coefficients, ARIMA orders,
        residual quantiles, MASE denominators, scalers. The refit cadence decides how often
        this runs.

        Args:
            train: The full training frame, including any covariates.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """

    @abstractmethod
    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Refresh the data the forecast is conditioned on.

        Runs on **every** fold, whatever the cadence. Only the values a forecast is
        conditioned on belong here — the last observation, the recent history, the context
        window — never anything estimated from data.

        Both halves are abstract rather than one defaulting to the other, so that adding a
        model forces an explicit decision about which of its attributes are parameters and
        which are state. Getting that split wrong silently is what produced the stale
        baseline described in docs/planning/PROGRESS_NOTES.md, Step 14.

        Args:
            train: The full training frame, including any covariates.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
