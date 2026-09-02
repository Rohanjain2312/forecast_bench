"""The interface every model in the study implements, and nothing else.

Seven model families — naive, classical, neural, foundation — reduce to two definitions
here. ARIMA and Chronos-2 are indistinguishable to ``runner.py``. That is the whole design,
and it is what lets the study claim that every model traversed identical code.

If a model needs special handling inside the runner, the abstraction is wrong. Fix the
abstraction, not the runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from forecast_bench.config import QUANTILE_GRID


@dataclass(frozen=True)
class QuantileForecast:
    """A model's h-step-ahead quantile forecast from one origin.

    Attributes:
        origin: The last timestamp the model was allowed to see.
        index: Forecast timestamps, length h. Every entry must be strictly after
            ``origin``.
        quantiles: Mapping of quantile level to an array of length h.
        model_id: Stable identifier used as the results-table key.

    Raises:
        ValueError: If the index is not strictly after the origin, if any quantile array
            has the wrong length, or if the quantile levels are not exactly
            :data:`~forecast_bench.config.QUANTILE_GRID`.
    """

    origin: pd.Timestamp
    index: pd.DatetimeIndex
    quantiles: Mapping[float, np.ndarray]
    model_id: str

    def __post_init__(self) -> None:
        """Validate the forecast at construction, where the offending model is still on the stack."""
        if len(self.index) == 0:
            raise ValueError(f"{self.model_id}: forecast index is empty")

        if self.index.min() <= self.origin:
            raise ValueError(
                f"{self.model_id}: forecast index starts at {self.index.min()}, which is "
                f"not strictly after the origin {self.origin}. A forecast that includes "
                "its own origin is reading a value it was asked to predict."
            )

        expected_levels = set(QUANTILE_GRID)
        actual_levels = set(self.quantiles)
        if actual_levels != expected_levels:
            missing = sorted(expected_levels - actual_levels)
            extra = sorted(actual_levels - expected_levels)
            raise ValueError(
                f"{self.model_id}: quantile levels do not match the study grid. "
                f"Missing {missing}, unexpected {extra}. Every model emits the full grid "
                "so that weighted quantile loss is comparable across the panel."
            )

        for level, values in self.quantiles.items():
            if len(values) != len(self.index):
                raise ValueError(
                    f"{self.model_id}: quantile {level} has {len(values)} values for "
                    f"{len(self.index)} forecast steps"
                )

    @property
    def horizon(self) -> int:
        """Number of steps in this forecast path."""
        return len(self.index)

    def median(self) -> np.ndarray:
        """The 0.5-quantile path, used as the point forecast."""
        return np.asarray(self.quantiles[0.5])

    def step(self, step: int) -> dict[float, float]:
        """Read one step off the path.

        Horizons 1, 5 and 21 are read off steps 1, 5 and 21 of a single 21-step path, so
        all three share identical folds and identical model fits.

        Args:
            step: One-based step number.

        Returns:
            Mapping of quantile level to that step's value.

        Raises:
            IndexError: If ``step`` is outside the path.
        """
        if not 1 <= step <= self.horizon:
            raise IndexError(f"step {step} outside a {self.horizon}-step path")
        return {
            level: float(values[step - 1]) for level, values in self.quantiles.items()
        }

    def assert_monotonic(self, tolerance: float = 1e-9) -> None:
        """Assert that quantiles do not cross at any step.

        Args:
            tolerance: Slack allowed for floating-point noise.

        Raises:
            AssertionError: If a higher quantile falls below a lower one. A crossing
                quantile is a bug: it describes a distribution that does not exist.
        """
        levels = sorted(self.quantiles)
        stacked = np.vstack([np.asarray(self.quantiles[level]) for level in levels])
        differences = np.diff(stacked, axis=0)
        if (differences < -tolerance).any():
            worst = int(np.argmin(differences.min(axis=0)))
            raise AssertionError(
                f"{self.model_id}: quantiles cross at step {worst + 1}. A higher quantile "
                "below a lower one describes a distribution that does not exist."
            )


@runtime_checkable
class Forecaster(Protocol):
    """Every model in the study implements exactly this.

    ``fit()`` may only read data at or before ``origin``. Implementations must not close
    over any object fitted outside the current fold — no scaler, no ARIMA order selection,
    no residual quantile, no MASE denominator, no regime threshold derived from data the
    fold cannot see.

    Violating this produces leakage that no metric will reveal. The backtest will simply
    look better. ``tests/test_no_leakage.py`` is what checks it, and it must never be
    weakened to make a run pass.

    Implementations should record the origin they were fitted on as
    ``fitted_on_origin``, which the leakage suite asserts against the fold.
    """

    model_id: str

    def fit(self, train: pd.DataFrame, origin: pd.Timestamp) -> None:
        """Fit on data at or before ``origin``.

        Args:
            train: Training frame for this fold. Its index never exceeds ``origin``.
            origin: The last timestamp this model is allowed to have seen.
        """
        ...

    def predict(self, horizon: int) -> QuantileForecast:
        """Produce a quantile forecast path.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            The forecast, whose index must start strictly after the fitted origin.
        """
        ...
