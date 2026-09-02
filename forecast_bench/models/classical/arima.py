"""ARIMA with per-fold AIC order selection.

Prediction intervals are extracted from the fitted statsmodels results via
``get_forecast().conf_int(alpha)`` at each quantile level, not approximated from a normal
assumption after the fact. Evaluating a probabilistic model through a point-forecast lens
and then bolting on a normal interval is the shortcut that makes a benchmark unpersuasive.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster

logger = logging.getLogger(__name__)

#: Order grid searched by AIC on every fold. Bounded deliberately: the grid is a decision
#: fixed once on the training span, and PREREGISTRATION.md section 5 commits to not
#: re-tuning it after seeing results.
DEFAULT_ORDER_GRID = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]


class ARIMA(BaseForecaster):
    """ARIMA whose order is selected by AIC inside each fold.

    Attributes:
        model_id: ``"ARIMA"``.
        order_grid: Candidate ``(p, d, q)`` orders.
        selected_order: The order chosen on the current fold.
    """

    model_id = "ARIMA"

    def __init__(
        self,
        target_column: str | None = None,
        order_grid: list[tuple[int, int, int]] | None = None,
        max_train: int = 2000,
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            order_grid: Candidate orders. Defaults to :data:`DEFAULT_ORDER_GRID`.
            max_train: Most recent observations used for fitting. Bounds the cost of
                refitting on an expanding window without changing the model class.
        """
        super().__init__(target_column=target_column)
        self.order_grid = order_grid or DEFAULT_ORDER_GRID
        self.max_train = max_train
        self.selected_order: tuple[int, int, int] | None = None
        self._results = None

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Select an order by AIC on this fold's data only, then fit it.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.

        Raises:
            RuntimeError: If no candidate order fits.

        Note:
            Order selection is part of fitting, so it happens inside the fold. Selecting an
            order once on the full series and reusing it would be a fitted object crossing
            every fold boundary in the study.
        """
        from statsmodels.tsa.arima import model as sm_arima

        values = series.to_numpy(dtype=float)[-self.max_train :]

        best_aic = np.inf
        best_order = None
        best_results = None

        with warnings.catch_warnings():
            # Non-converging candidates are expected in a grid search; the AIC comparison
            # is what decides, and a warning per candidate would drown the run's log.
            warnings.simplefilter("ignore")
            for order in self.order_grid:
                try:
                    results = sm_arima.ARIMA(values, order=order).fit()
                except (
                    Exception
                ):  # noqa: BLE001 - a candidate that will not fit is data
                    continue
                aic = float(results.aic)
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order, best_results = aic, order, results

        if best_results is None:
            raise RuntimeError(
                f"{self.model_id}: no candidate order in the grid fitted at origin {origin}"
            )

        self.selected_order = best_order
        self._results = best_results

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

        Note:
            ``conf_int(alpha)`` returns the ``[alpha/2, 1 - alpha/2]`` bounds, so the
            level-``q`` quantile is the lower bound at ``alpha = 2q`` for ``q < 0.5`` and
            the upper bound at ``alpha = 2(1 - q)`` for ``q > 0.5``.
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
