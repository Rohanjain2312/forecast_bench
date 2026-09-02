"""Every registered model satisfies the Forecaster protocol and emits usable quantiles.

Parametrised over ``registry.all_registered_model_classes()`` rather than a hand-written
list, so a model added to the panel is covered here automatically instead of being
forgotten.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.backtest.protocol import Forecaster, QuantileForecast
from forecast_bench.config import MAX_HORIZON, QUANTILE_GRID
from forecast_bench.models.registry import (
    BASELINE_MODEL_ID,
    all_registered_model_classes,
    classical_panel,
)

MODEL_CLASSES = all_registered_model_classes()
MODEL_IDS = sorted(MODEL_CLASSES)


@pytest.fixture
def training_frame(synthetic_series) -> pd.DataFrame:
    """A target plus one covariate, long enough for every model's lag structure."""
    frame = synthetic_series.to_frame(name="target")
    frame["benign_covariate"] = synthetic_series.shift(1).rolling(5).mean()
    return frame.dropna()


@pytest.fixture
def forecast_index(training_frame) -> pd.DatetimeIndex:
    """Dates immediately after the training frame, as the runner would supply."""
    last = training_frame.index[-1]
    return pd.date_range(
        last + pd.tseries.offsets.BDay(1), periods=MAX_HORIZON, freq="B"
    )


def _fit_and_predict(model_id: str, frame: pd.DataFrame, index) -> QuantileForecast:
    """Fit a registered model on the frame and forecast over the index."""
    model = MODEL_CLASSES[model_id](target_column="target")
    origin = frame.index[-1]
    model.fit(frame, origin=origin)
    return model.predict(horizon=MAX_HORIZON, index=index)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_satisfies_the_forecaster_protocol(model_id) -> None:
    """Every registered class structurally implements Forecaster."""
    model = MODEL_CLASSES[model_id](target_column="target")
    assert isinstance(model, Forecaster)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert isinstance(model.model_id, str) and model.model_id


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_returns_the_full_quantile_grid(
    model_id, training_frame, forecast_index
) -> None:
    """Every model emits all eleven levels, so WQL is comparable across the panel."""
    forecast = _fit_and_predict(model_id, training_frame, forecast_index)
    assert sorted(forecast.quantiles) == sorted(QUANTILE_GRID)
    for level in QUANTILE_GRID:
        assert len(forecast.quantiles[level]) == MAX_HORIZON


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_quantiles_do_not_cross(model_id, training_frame, forecast_index) -> None:
    """Quantiles are non-decreasing in the level at every step.

    A crossing quantile describes a distribution that does not exist.
    """
    forecast = _fit_and_predict(model_id, training_frame, forecast_index)
    forecast.assert_monotonic()

    levels = sorted(forecast.quantiles)
    stacked = np.vstack([forecast.quantiles[level] for level in levels])
    assert (np.diff(stacked, axis=0) >= -1e-9).all()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_produces_finite_forecasts(
    model_id, training_frame, forecast_index
) -> None:
    """No NaN or infinity reaches the results table."""
    forecast = _fit_and_predict(model_id, training_frame, forecast_index)
    for level, path in forecast.quantiles.items():
        assert np.isfinite(
            path
        ).all(), f"{model_id} produced non-finite values at {level}"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_records_the_origin_it_was_fitted_on(
    model_id, training_frame, forecast_index
) -> None:
    """Fold provenance is recorded, which the leakage suite asserts against."""
    model = MODEL_CLASSES[model_id](target_column="target")
    origin = training_frame.index[-1]
    model.fit(training_frame, origin=origin)
    assert model.fitted_on_origin == origin


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_forecast_index_starts_after_the_origin(
    model_id, training_frame, forecast_index
) -> None:
    """The forecast never includes a date the model was allowed to see."""
    forecast = _fit_and_predict(model_id, training_frame, forecast_index)
    assert forecast.index[0] > forecast.origin


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_predict_before_fit_is_an_error(model_id, forecast_index) -> None:
    """Predicting from an unfitted model fails loudly rather than returning nonsense."""
    model = MODEL_CLASSES[model_id](target_column="target")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(horizon=MAX_HORIZON, index=forecast_index)


def test_random_walk_median_is_the_last_observed_value(
    training_frame, forecast_index
) -> None:
    """The random walk's median path is flat at the last value, by definition."""
    forecast = _fit_and_predict("RandomWalk", training_frame, forecast_index)
    last = float(training_frame["target"].iloc[-1])
    assert np.allclose(forecast.median(), last, atol=1e-9)


def test_random_walk_intervals_widen_with_the_horizon(
    training_frame, forecast_index
) -> None:
    """Uncertainty grows with the horizon because the empirical changes say it does."""
    forecast = _fit_and_predict("RandomWalk", training_frame, forecast_index)
    widths = forecast.quantiles[0.975] - forecast.quantiles[0.025]
    assert widths[-1] > widths[0]


def test_seasonal_naive_repeats_the_previous_week(
    training_frame, forecast_index
) -> None:
    """The median path repeats the final five observations."""
    forecast = _fit_and_predict("SeasonalNaive", training_frame, forecast_index)
    expected = training_frame["target"].to_numpy()[-5:]
    assert np.allclose(forecast.median()[:5], expected, atol=1e-9)


def test_arima_selects_an_order_inside_the_fold(training_frame) -> None:
    """Order selection happens during fit, so it is a fold-local decision."""
    model = MODEL_CLASSES["ARIMA"](target_column="target")
    assert model.selected_order is None
    model.fit(training_frame, origin=training_frame.index[-1])
    assert model.selected_order is not None
    assert len(model.selected_order) == 3


def test_har_and_loghar_are_different_models(training_frame, forecast_index) -> None:
    """HAR fits in variance space and LogHAR in log space, so they must not coincide."""
    har = _fit_and_predict("HAR", training_frame, forecast_index)
    log_har = _fit_and_predict("LogHAR", training_frame, forecast_index)
    assert not np.allclose(har.median(), log_har.median())


def test_sarimax_refuses_an_unlagged_exogenous_regressor() -> None:
    """An unlagged covariate needs its own future path, which is look-ahead bias."""
    with pytest.raises(ValueError, match="look-ahead"):
        MODEL_CLASSES["SARIMAX"](exog_lag=0)


def test_registry_builders_return_fresh_instances() -> None:
    """Each call builds a new object, so no fitted state survives a fold boundary."""
    panel = classical_panel("spy_logrv")
    for build in panel.values():
        first, second = build(), build()
        assert first is not second
        assert first.fitted_on_origin is None


def test_baseline_is_registered_for_every_series() -> None:
    """The skill-score baseline exists on both tracks, or skill scores are undefined."""
    for series in ("spy_logrv", "dgs10"):
        assert BASELINE_MODEL_ID in classical_panel(series)
