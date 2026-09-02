"""SARIMAX — the classical covariate-informed model, used in Arm B only.

Exogenous regressors are lagged before they enter, so that a forecast for ``t + h`` uses
only covariate values observed at or before the origin. An unlagged exogenous regressor
would require knowing the covariate's future path, which is the same look-ahead problem
that ``docs/data_protocol.md`` exists to prevent, arriving through a different door.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class SARIMAX(BaseForecaster):
    """Seasonal ARIMA with lagged exogenous regressors.

    Attributes:
        model_id: ``"SARIMAX"``.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``. ``s = 5`` is a business week.
        exog_columns: Covariate columns to use. ``None`` means every non-target column.
        exog_lag: How far the covariates are lagged before entering.
    """

    model_id = "SARIMAX"

    def __init__(
        self,
        target_column: str | None = None,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_columns: list[str] | None = None,
        exog_lag: int = 1,
        max_train: int = 2000,
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            order: Non-seasonal ARIMA order.
            seasonal_order: Seasonal order.
            exog_columns: Covariates to use, or ``None`` for all non-target columns.
            exog_lag: Lag applied to every covariate. Must be at least 1.
            max_train: Most recent observations used for fitting.

        Raises:
            ValueError: If ``exog_lag`` is less than 1.
        """
        super().__init__(target_column=target_column)
        if exog_lag < 1:
            raise ValueError(
                "exog_lag must be at least 1. An unlagged exogenous regressor requires "
                "the covariate's future path, which is look-ahead bias."
            )
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = exog_columns
        self.exog_lag = exog_lag
        self.max_train = max_train
        self._results = None
        self._future_exog: np.ndarray | None = None

    def _fit(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Fit SARIMAX with lagged covariates on this fold.

        Args:
            train: Training frame for this fold, including covariates.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        from statsmodels.tsa.statespace import sarimax as sm_sarimax

        name = self.target_column or str(train.columns[0])
        columns = self.exog_columns or [
            column for column in train.columns if column != name
        ]

        aligned = train[[name, *columns]].copy()
        for column in columns:
            aligned[column] = aligned[column].shift(self.exog_lag)
        aligned = aligned.ffill().dropna().tail(self.max_train)

        endog = aligned[name].to_numpy(dtype=float)
        exog = aligned[columns].to_numpy(dtype=float) if columns else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = sm_sarimax.SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

        # The forecast needs exogenous values for future dates. Only values known at the
        # origin may be used, so the last observed row is held constant across the path.
        # This is a modelling assumption, not a data source: it uses nothing unobserved.
        self._future_exog = (
            np.tile(exog[-1], (1, 1)) if exog is not None and len(exog) else None
        )

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Extract quantiles from the fitted model's forecast distribution.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path.
        """
        exog = (
            np.repeat(self._future_exog, horizon, axis=0)
            if self._future_exog is not None
            else None
        )
        forecast = self._results.get_forecast(horizon, exog=exog)
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
