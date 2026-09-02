"""Naive baselines: the random walk and the seasonal naive.

The random walk is the reference every skill score is measured against, so its quantiles
matter as much as any learned model's. Scoring a point-only naive baseline against
quantile-emitting models on weighted quantile loss would hand the comparison to whoever
emits intervals, regardless of forecast quality.
"""

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster, empirical_change_quantiles

#: Business-day seasonal period used by :class:`SeasonalNaive`.
WEEKLY_PERIOD = 5


class RandomWalk(BaseForecaster):
    """Last value carried forward, with an empirical predictive distribution.

    The median forecast is the last observed value. Quantiles come from the empirical
    distribution of h-step changes measured in the training window, recomputed on every
    fold.

    Attributes:
        model_id: ``"RandomWalk"``.
    """

    model_id = "RandomWalk"

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Record the sample the h-step change distribution is measured from.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        self._values = series.to_numpy(dtype=float)

    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Refresh the last observed value, which is the forecast itself.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.

        Note:
            The last value is *state*, not a parameter. A random walk forecasting from a
            value four months old is not a random walk, and since it is the baseline every
            skill score is quoted against, freezing it inflated the whole study.
        """
        self._last_value = float(series.iloc[-1])

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Centre the empirical h-step change distribution on the last observed value.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path whose median is exactly the last observed value.

        Note:
            The change quantiles are **centred** by subtracting the median change, so the
            median path is the last value and only the spread comes from the data.

            Without centring, a training window with drift produces a random walk *with
            drift*, which is a different and generally stronger model. That matters more
            here than anywhere else in the panel: this is the baseline every skill score
            in the study is measured against, and PREREGISTRATION.md fixes the primary
            metric as WQL skill versus the random walk. A baseline that quietly absorbs
            in-sample drift moves the reference point every other number is quoted
            against.
        """
        changes = empirical_change_quantiles(self._values, horizon, QUANTILE_GRID)
        median_change = changes[0.5]
        return {
            level: self._last_value + (path - median_change)
            for level, path in changes.items()
        }


class SeasonalNaive(BaseForecaster):
    """Repeat the value from the same weekday of the previous week.

    A sanity baseline: on a series with genuine weekly structure it should beat the random
    walk, and on one without it should lose. Either outcome is informative.

    Attributes:
        model_id: ``"SeasonalNaive"``.
        period: Seasonal period in observations. Five for business days.
    """

    model_id = "SeasonalNaive"

    def __init__(
        self, target_column: str | None = None, period: int = WEEKLY_PERIOD
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            period: Seasonal period in observations.
        """
        super().__init__(target_column=target_column)
        self.period = period

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Measure the in-sample one-season-ahead errors, this model's spread.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.

        Raises:
            ValueError: If the training window is shorter than one season.
        """
        values = series.to_numpy(dtype=float)
        if len(values) <= self.period:
            raise ValueError(
                f"{self.model_id}: need more than {self.period} observations, got "
                f"{len(values)}"
            )
        self._errors = values[self.period :] - values[: -self.period]

    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Refresh the final season, which is what gets repeated forward.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        self._last_season = series.to_numpy(dtype=float)[-self.period :]

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Repeat the last season and widen by the centred seasonal error spread.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path whose median repeats the final season exactly.

        Note:
            The error quantiles are centred on their median for the same reason as in
            :class:`RandomWalk`: a seasonal naive that absorbs in-sample drift is no
            longer the naive baseline it is being used as.
        """
        repeats = int(np.ceil(horizon / self.period))
        path = np.tile(self._last_season, repeats)[:horizon]

        # Errors accumulate with the number of seasons ahead, not with the raw step count.
        seasons_ahead = np.ceil(np.arange(1, horizon + 1) / self.period)
        scale = np.sqrt(seasons_ahead)
        median_error = float(np.quantile(self._errors, 0.5))

        return {
            level: path
            + (float(np.quantile(self._errors, level)) - median_error) * scale
            for level in QUANTILE_GRID
        }
