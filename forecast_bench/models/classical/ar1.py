"""AR(1) in levels — the standard macro-forecasting benchmark on the rates track.

Recent vintage-consistent macro literature benchmarks foundation models against exactly
this, which is why it earns a place in the panel alongside ARIMA rather than being
subsumed by it: ARIMA selects its own order per fold and may not choose (1, 0, 0).
"""

import warnings

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster


class AR1(BaseForecaster):
    """A fixed AR(1) with an intercept, fitted per fold.

    Attributes:
        model_id: ``"AR1"``.
    """

    model_id = "AR1"

    def __init__(self, target_column: str | None = None, max_train: int = 2000) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            max_train: Most recent observations used for fitting.
        """
        super().__init__(target_column=target_column)
        self.max_train = max_train
        self._results = None

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Fit AR(1) on this fold's training window.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        from statsmodels.tsa.arima import model as sm_arima

        values = series.to_numpy(dtype=float)[-self.max_train :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = sm_arima.ARIMA(values, order=(1, 0, 0)).fit()

    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Re-apply the existing parameters to data running to this fold's origin.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.

        Note:
            ``results.apply(..., refit=False)`` keeps the estimated coefficients and
            recomputes the filtered state on the new sample, which is exactly the
            parameters/conditioning split the refit cadence is meant to express.
        """
        values = series.to_numpy(dtype=float)[-self.max_train :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = self._results.apply(values, refit=False)

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Extract quantiles from the fitted model's forecast distribution.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path.
        """
        forecast = self._results.get_forecast(horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=float)

        paths: dict[float, np.ndarray] = {}
        for level in QUANTILE_GRID:
            if np.isclose(level, 0.5):
                paths[level] = mean
                continue
            alpha = 2.0 * level if level < 0.5 else 2.0 * (1.0 - level)
            bounds = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
            paths[level] = bounds[:, 0] if level < 0.5 else bounds[:, 1]
        return paths
