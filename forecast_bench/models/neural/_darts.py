"""Shared darts plumbing for the from-scratch neural baselines.

Private to :mod:`forecast_bench.models.neural`. Both neural models are darts estimators
wrapped behind the study's ``Forecaster`` protocol, so the runner cannot tell them apart
from ARIMA or Chronos-2.

The harness never calls ``darts.historical_forecasts``. Outsourcing the backtest would mean
the foundation models and the neural models no longer provably traverse identical code,
which is the one claim the harness exists to support.
"""

import logging

import numpy as np
import pandas as pd

from forecast_bench.config import (
    CONTEXT_LENGTH,
    MAX_HORIZON,
    QUANTILE_GRID,
    RANDOM_SEED,
)
from forecast_bench.models.base import BaseForecaster

logger = logging.getLogger(__name__)

#: Validation windows carved from the end of a training block for early stopping.
VALIDATION_WINDOWS = 252

#: Early-stopping patience, in epochs.
PATIENCE = 5


def to_timeseries(values: np.ndarray):
    """Wrap a value array as a darts ``TimeSeries``.

    Args:
        values: Observations in order.

    Returns:
        A univariate ``TimeSeries`` on a synthetic integer index. The real calendar is
        supplied to ``predict`` by the runner, so darts never needs it.

    Note:
        Cast to ``float32``. Torch's MPS backend cannot take ``float64`` at all, and
        float32 is the standard dtype for neural training on every backend, so this is the
        right cast rather than a platform workaround.
    """
    from darts import TimeSeries

    return TimeSeries.from_values(np.asarray(values, dtype=np.float32).reshape(-1, 1))


class DartsQuantileForecaster(BaseForecaster):
    """Base for darts models that emit quantiles through a quantile likelihood.

    Attributes:
        model_id: Results-table key.
        input_chunk_length: Context fed to the network, matched to the foundation models
            so that context length is not a confound across model classes.
        output_chunk_length: Steps predicted per forward pass.
        n_epochs: Maximum training epochs per parameter refit.
        training_window: Sample-efficiency slice name, recorded in results.
    """

    model_id = "DartsQuantile"

    def __init__(
        self,
        target_column: str | None = None,
        input_chunk_length: int = CONTEXT_LENGTH,
        output_chunk_length: int = MAX_HORIZON,
        n_epochs: int = 50,
        training_window_days: int | None = None,
        random_state: int = RANDOM_SEED,
        device: str = "auto",
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            input_chunk_length: Context length.
            output_chunk_length: Steps per forward pass.
            n_epochs: Maximum epochs.
            training_window_days: Most recent observations to train on, for the
                sample-efficiency sweep. ``None`` uses the whole window.
            random_state: Seed.
            device: Torch accelerator passed to darts.
        """
        super().__init__(target_column=target_column)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.training_window_days = training_window_days
        self.random_state = random_state
        self.device = device
        self._model = None
        self._context: np.ndarray | None = None

    def _build(self):
        """Construct the underlying darts model.

        Returns:
            An unfitted darts estimator.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def _trainer_kwargs(self) -> dict:
        """Lightning trainer settings shared by both neural models.

        Returns:
            Keyword arguments for darts' ``pl_trainer_kwargs``.
        """
        return {
            "accelerator": self.device,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        }

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Train the network on this fold's window, with fold-local early stopping.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.

        Note:
            The validation slice is carved from the **end** of the training window, so
            nothing after the origin is read. Early stopping is therefore a fold-local
            decision like every other calibration in the study.
        """
        values = series.to_numpy(dtype=float)
        if self.training_window_days is not None:
            values = values[-self.training_window_days :]

        split = max(len(values) - VALIDATION_WINDOWS, self.input_chunk_length + 1)
        train_values, validation_values = values[:split], values[split:]

        self._model = self._build()
        fit_kwargs = {"series": to_timeseries(train_values)}
        if len(validation_values) > self.input_chunk_length + self.output_chunk_length:
            fit_kwargs["val_series"] = to_timeseries(
                values[split - self.input_chunk_length :]
            )
        self._model.fit(**fit_kwargs)

    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Refresh the context the forecast is conditioned on.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        self._context = series.to_numpy(dtype=float)[-self.input_chunk_length :]

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Sample the fitted network and read the study's quantiles off the samples.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path.
        """
        prediction = self._model.predict(
            n=horizon,
            series=to_timeseries(self._context),
            num_samples=500,
        )
        samples = np.asarray(prediction.all_values())  # (time, component, sample)
        samples = samples[:, 0, :]
        return {level: np.quantile(samples, level, axis=1) for level in QUANTILE_GRID}
