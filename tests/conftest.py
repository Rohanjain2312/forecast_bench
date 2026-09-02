"""Shared fixtures: synthetic series, tiny fold specifications, and a stub forecaster.

Everything here is deterministic and offline. A leakage guard that depends on a network
pull is a leakage guard that gets skipped in CI, and a skipped guard is no guard.

These fixtures deliberately do **not** import the backtest harness. They are written
before it exists, so that the harness is built against them rather than the other way
round. Tests that need the harness import it inside the test body.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from forecast_bench.backtest.protocol import QuantileForecast
from forecast_bench.config import MAX_HORIZON, QUANTILE_GRID, RANDOM_SEED

#: Length of the synthetic series, long enough to carve a realistic fold structure from.
SYNTHETIC_DAYS = 3000

#: Start of the synthetic series.
SYNTHETIC_START = "2000-01-03"


@pytest.fixture
def synthetic_series() -> pd.Series:
    """A deterministic 3000-day AR(1) series on a business-day index.

    Persistent enough that a naive forecast is decent and a leak is obvious, which is what
    the canary needs.

    Returns:
        A float series named ``target``.
    """
    index = pd.date_range(SYNTHETIC_START, periods=SYNTHETIC_DAYS, freq="B")
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.empty(SYNTHETIC_DAYS)
    values[0] = 0.0
    for position in range(1, SYNTHETIC_DAYS):
        values[position] = 0.97 * values[position - 1] + rng.standard_normal()
    return pd.Series(values, index=index, name="target")


@pytest.fixture
def synthetic_frame(synthetic_series: pd.Series) -> pd.DataFrame:
    """The synthetic target plus one benign, causal covariate.

    The covariate is a lagged transform of the target, so it carries information but no
    information from the future.

    Returns:
        A frame with columns ``target`` and ``benign_covariate``.
    """
    frame = synthetic_series.to_frame()
    frame["benign_covariate"] = synthetic_series.shift(1).rolling(5).mean()
    return frame.dropna()


@pytest.fixture
def leaky_frame(synthetic_series: pd.Series) -> pd.DataFrame:
    """The synthetic target with a column that is a perfect copy of the future target.

    This is the canary. At row ``t`` the ``future_leak`` column holds the target's value at
    ``t + MAX_HORIZON``. Any model allowed to read it can forecast the 21-step-ahead value
    exactly, so the error must collapse to approximately zero.

    A guard nobody has watched fail is a guard nobody knows works.

    Returns:
        A frame with columns ``target`` and ``future_leak``.
    """
    frame = synthetic_series.to_frame()
    frame["future_leak"] = synthetic_series.shift(-MAX_HORIZON)
    return frame.dropna()


@pytest.fixture
def tiny_fold_spec() -> dict[str, object]:
    """Parameters producing a small fold set, for tests that assert on fold structure.

    Sized so that the resulting fold count is small enough to enumerate in an assertion
    message but large enough to exercise block boundaries across two calendar years.

    Returns:
        Keyword arguments for ``expanding_origin_folds``.
    """
    return {
        "train_start": SYNTHETIC_START,
        "test_start": "2004-01-01",
        "test_end": "2004-07-01",
        "stride": MAX_HORIZON,
        "horizon": MAX_HORIZON,
    }


@dataclass
class StubForecaster:
    """A constant forecaster that records what it was fitted on.

    Implements ``backtest.protocol.Forecaster``.

    Attributes:
        model_id: Identifier used as the results-table key.
        constant: Value returned at every quantile and every step.
        fitted_on_origin: Origin passed to the most recent :meth:`fit` call. The leakage
            suite asserts this equals the fold's origin for every model in the panel.
        seen_max_index: Latest timestamp observed in training data, used to prove that no
            model saw past its origin.
        fit_calls: Number of parameter re-estimations, used by the cadence tests.
        parameters_fitted_on_origin: Origin at which parameters were last re-estimated.
    """

    model_id: str = "StubForecaster"
    constant: float = 0.0
    fitted_on_origin: pd.Timestamp | None = None
    seen_max_index: pd.Timestamp | None = None
    parameters_fitted_on_origin: pd.Timestamp | None = None
    fit_calls: int = 0
    _last_index: pd.DatetimeIndex | None = field(default=None, repr=False)

    def fit(
        self,
        train: pd.DataFrame,
        origin: pd.Timestamp,
        refit_parameters: bool = True,
    ) -> None:
        """Record the fold's origin, and re-estimate only when asked.

        Args:
            train: Training frame for this fold.
            origin: The last timestamp this model is allowed to have seen.
            refit_parameters: Whether the cadence wants parameters re-estimated.
        """
        self.fitted_on_origin = origin
        self.seen_max_index = train.index.max()
        self._last_index = train.index
        if refit_parameters:
            self.fit_calls += 1
            self.parameters_fitted_on_origin = origin
        # State always refreshes, whatever the cadence.
        self.constant = float(train.iloc[:, 0].iloc[-1])

    def predict(self, horizon: int, index: pd.DatetimeIndex) -> QuantileForecast:
        """Return a flat quantile path over the given dates.

        Args:
            horizon: Number of steps to forecast.
            index: The dates being forecast.

        Returns:
            A forecast whose quantiles are flat and ordered.
        """
        return QuantileForecast(
            origin=self.fitted_on_origin,
            index=index[:horizon],
            quantiles={
                level: np.full(horizon, self.constant + (level - 0.5))
                for level in QUANTILE_GRID
            },
            model_id=self.model_id,
        )


@dataclass
class CheatingForecaster(StubForecaster):
    """A forecaster that reads a leaked future column, used only by the canary.

    Exists to demonstrate that the error-collapse detector has power: with the leak its
    error goes to approximately zero, and without it the same model is unremarkable.

    Attributes:
        leak_column: Column holding the future value.
    """

    model_id: str = "CheatingForecaster"
    leak_column: str = "future_leak"
    _leak_tail: np.ndarray | None = field(default=None, repr=False)

    def fit(
        self,
        train: pd.DataFrame,
        origin: pd.Timestamp,
        refit_parameters: bool = True,
    ) -> None:
        """Read the future path straight out of the leaked column.

        Because ``future_leak[t] == target[t + MAX_HORIZON]``, the column's final
        ``MAX_HORIZON`` rows hold ``target[origin + 1 .. origin + MAX_HORIZON]`` — the
        exact path being forecast. Every one of those rows is dated at or before the
        origin, so no date comparison would object.

        Args:
            train: Training frame for this fold.
            origin: The last timestamp this model is allowed to have seen.
            refit_parameters: Whether the cadence wants parameters re-estimated.
        """
        super().fit(train, origin, refit_parameters)
        if self.leak_column in train.columns:
            self._leak_tail = train[self.leak_column].to_numpy(dtype=float)

    def predict(self, horizon: int, index: pd.DatetimeIndex) -> QuantileForecast:
        """Return the leaked future path, or fall back to the stub behaviour.

        Args:
            horizon: Number of steps to forecast.
            index: The dates being forecast.

        Returns:
            A forecast that is exactly right when the leak is available.
        """
        if self._leak_tail is None or len(self._leak_tail) < horizon:
            return super().predict(horizon, index)

        path = self._leak_tail[-horizon:]
        return QuantileForecast(
            origin=self.fitted_on_origin,
            index=index[:horizon],
            quantiles={level: path + (level - 0.5) * 1e-9 for level in QUANTILE_GRID},
            model_id=self.model_id,
        )


@pytest.fixture
def stub_forecaster() -> StubForecaster:
    """A fresh :class:`StubForecaster`."""
    return StubForecaster()


@pytest.fixture
def cheating_forecaster() -> CheatingForecaster:
    """A fresh :class:`CheatingForecaster` for the canary test."""
    return CheatingForecaster()
