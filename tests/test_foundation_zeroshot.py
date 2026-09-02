"""Tests for the zero-shot foundation models.

Marked ``slow`` because they download and run real checkpoints. Run them with::

    poetry run pytest tests/test_foundation_zeroshot.py -m slow
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import MAX_HORIZON, QUANTILE_GRID
from forecast_bench.models.foundation.chronos2 import Chronos2ZeroShot
from forecast_bench.models.foundation.chronos_bolt import (
    BOLT_TRAINED_QUANTILES,
    ChronosBoltZeroShot,
)
from forecast_bench.models.registry import foundation_model_ids

pytestmark = pytest.mark.slow


@pytest.fixture
def training_frame(synthetic_series) -> pd.DataFrame:
    """A single-column frame long enough to fill the context window."""
    return synthetic_series.to_frame(name="target")


@pytest.fixture
def forecast_index(training_frame) -> pd.DatetimeIndex:
    """Dates immediately after the training frame."""
    last = training_frame.index[-1]
    return pd.date_range(
        last + pd.tseries.offsets.BDay(1), periods=MAX_HORIZON, freq="B"
    )


def _forecast(model, frame, index):
    """Fit and forecast, returning the QuantileForecast."""
    model.fit(frame, origin=frame.index[-1])
    return model.predict(horizon=MAX_HORIZON, index=index)


def test_chronos2_emits_genuine_tail_quantiles(training_frame, forecast_index) -> None:
    """Chronos-2's trained range covers the study grid, so no level is clamped."""
    model = Chronos2ZeroShot(target_column="target")
    forecast = _forecast(model, training_frame, forecast_index)

    forecast.assert_monotonic()
    assert not np.allclose(forecast.quantiles[0.025], forecast.quantiles[0.1])
    assert not np.allclose(forecast.quantiles[0.975], forecast.quantiles[0.9])
    assert min(model.trained_quantiles) <= min(QUANTILE_GRID)
    assert max(model.trained_quantiles) >= max(QUANTILE_GRID)


def test_chronos_bolt_tails_are_clamped_to_its_trained_range(
    training_frame, forecast_index
) -> None:
    """Bolt cannot produce the study's tail levels, and the study records that as a fact.

    Asserted rather than described so that if a future checkpoint gains real tail
    predictions, this test fails and the limitation text gets revisited instead of
    silently becoming wrong.
    """
    model = ChronosBoltZeroShot(target_column="target")
    forecast = _forecast(model, training_frame, forecast_index)

    assert np.allclose(forecast.quantiles[0.025], forecast.quantiles[0.1])
    assert np.allclose(forecast.quantiles[0.975], forecast.quantiles[0.9])
    assert min(BOLT_TRAINED_QUANTILES) > min(QUANTILE_GRID)


def test_zero_shot_models_learn_nothing_from_our_data(
    training_frame, forecast_index
) -> None:
    """Two instances fitted on different windows forecast identically given equal context.

    Zero-shot means no parameters are estimated from this study's data. Only the context
    window differs between folds, so identical context must give identical output.
    """
    model_a = Chronos2ZeroShot(target_column="target")
    model_b = Chronos2ZeroShot(target_column="target")

    first = _forecast(model_a, training_frame, forecast_index)
    # A different training history, then the same final context restored.
    model_b.fit(training_frame.iloc[:1500], origin=training_frame.index[1499])
    second = _forecast(model_b, training_frame, forecast_index)

    assert np.allclose(first.median(), second.median())


def test_context_window_refreshes_between_folds(training_frame, forecast_index) -> None:
    """Conditioning on a later origin changes the forecast, as it must."""
    model = Chronos2ZeroShot(target_column="target")

    early = training_frame.iloc[:2000]
    model.fit(early, origin=early.index[-1])
    first = model.predict(horizon=MAX_HORIZON, index=forecast_index)

    model.fit(training_frame, origin=training_frame.index[-1], refit_parameters=False)
    second = model.predict(horizon=MAX_HORIZON, index=forecast_index)

    assert not np.allclose(first.median(), second.median())


def test_both_foundation_models_are_registered() -> None:
    """The registry exposes exactly the two zero-shot models the study runs."""
    assert foundation_model_ids() == ["Chronos2-ZeroShot", "ChronosBolt-ZeroShot"]
