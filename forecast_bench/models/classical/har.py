"""HAR and LogHAR — the realized-volatility benchmark.

HAR-RV is *the* benchmark in the realized-volatility literature and the model a referee
would ask about first. A volatility study whose classical arm is only ARIMA has a hole in
it that a quant interviewer finds in thirty seconds. It is also the model that beat
foundation models in the 2026 benchmark cited in DECISIONS.md D1, so leaving it out would
have quietly removed the strongest classical competitor.

Both variants regress on daily, weekly and monthly averages of past realized variance. They
differ in the space they are fitted in:

- :class:`LogHAR` fits on the log target directly, which is what the study models.
- :class:`HAR` fits in variance space and transforms the forecast back, which is the
  classical formulation.

Both emit quantiles in **log space**, so they are scored against the same target as every
other model in the panel.
"""

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster, scaled_residual_quantiles

#: Averaging windows: daily, weekly, monthly. The 22-day component is why the study's
#: longest horizon is 21 trading days.
HAR_LAGS = (1, 5, 22)


def har_design_matrix(
    values: np.ndarray, lags: tuple[int, ...] = HAR_LAGS
) -> tuple[np.ndarray, np.ndarray]:
    """Build the HAR regression design matrix and target vector.

    Row ``t`` predicts ``values[t]`` from the trailing means of ``values`` over each lag
    window, all ending at ``t - 1``.

    Args:
        values: The series being modelled, in observation order.
        lags: Averaging windows.

    Returns:
        A ``(design, target)`` pair. ``design`` has an intercept column first.

    Raises:
        ValueError: If the series is shorter than the longest lag window.
    """
    longest = max(lags)
    if len(values) <= longest:
        raise ValueError(
            f"HAR needs more than {longest} observations, got {len(values)}"
        )

    frame = pd.Series(values)
    columns = [frame.shift(1).rolling(lag).mean().to_numpy(dtype=float) for lag in lags]
    design = np.column_stack([np.ones(len(values)), *columns])
    target = np.asarray(values, dtype=float)

    usable = np.isfinite(design).all(axis=1) & np.isfinite(target)
    return design[usable], target[usable]


class LogHAR(BaseForecaster):
    """HAR fitted directly on the log target.

    Attributes:
        model_id: ``"LogHAR"``.
        lags: Averaging windows.
    """

    model_id = "LogHAR"
    _fits_in_log_space = True

    def __init__(
        self, target_column: str | None = None, lags: tuple[int, ...] = HAR_LAGS
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            lags: Averaging windows.
        """
        super().__init__(target_column=target_column)
        self.lags = lags
        self._coefficients: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._history: np.ndarray | None = None

    def _model_space(self, series: pd.Series) -> np.ndarray:
        """Return the series in the space this variant regresses in.

        Args:
            series: The target column.

        Returns:
            Values in model space. Identity for :class:`LogHAR`.
        """
        return series.to_numpy(dtype=float)

    def _fit(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Fit the HAR regression by ordinary least squares on this fold.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        values = self._model_space(series)
        design, target = har_design_matrix(values, self.lags)

        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        self._coefficients = coefficients
        self._residuals = target - design @ coefficients
        self._history = values

    def _iterate(self, horizon: int) -> np.ndarray:
        """Roll the fitted regression forward, feeding each forecast back in.

        Args:
            horizon: Number of steps to produce.

        Returns:
            The point forecast path in model space.
        """
        history = list(self._history)
        path = np.empty(horizon)
        for step in range(horizon):
            features = [1.0] + [float(np.mean(history[-lag:])) for lag in self.lags]
            prediction = float(np.dot(self._coefficients, features))
            path[step] = prediction
            history.append(prediction)
        return path

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Widen the iterated point path by ``sqrt(h)``-scaled residual quantiles.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path, in log space.
        """
        path = self._iterate(horizon)
        offsets = scaled_residual_quantiles(self._residuals, horizon, QUANTILE_GRID)
        return {level: path + offsets[level] for level in QUANTILE_GRID}


class HAR(LogHAR):
    """Classical HAR, fitted in variance space and reported in log space.

    Attributes:
        model_id: ``"HAR"``.

    Note:
        Quantiles are formed in variance space and then log-transformed. That is valid
        because ``log`` is strictly increasing, so it maps the q-quantile of variance to
        the q-quantile of log-variance exactly — no Jensen correction is needed, which
        would not be true if the *mean* were being transformed.
    """

    model_id = "HAR"
    _fits_in_log_space = False

    def _model_space(self, series: pd.Series) -> np.ndarray:
        """Exponentiate the log target back into variance space.

        Args:
            series: The log target column.

        Returns:
            Values in variance space.
        """
        return np.exp(series.to_numpy(dtype=float))

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Build variance-space quantiles, floor them, and return their logs.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path, in log space.
        """
        path = self._iterate(horizon)
        offsets = scaled_residual_quantiles(self._residuals, horizon, QUANTILE_GRID)

        # An additive residual can push a variance forecast non-positive, and log() of
        # that is -inf. Floor at the smallest positive variance seen in this fold's
        # training window, which is a fold-local quantity like every other calibration.
        positive = self._history[self._history > 0]
        floor = float(positive.min()) if positive.size else 1e-12

        return {
            level: np.log(np.maximum(path + offsets[level], floor))
            for level in QUANTILE_GRID
        }
