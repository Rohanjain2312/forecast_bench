"""Tests for the LoRA-adapted foundation models.

Marked ``slow``: these download real checkpoints and adapters from the Hub. Run with::

    poetry run pytest tests/test_foundation_finetuned.py -m slow
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import MAX_HORIZON, QUANTILE_GRID
from forecast_bench.models.foundation.chronos2 import (
    Chronos2FineTuned,
    Chronos2ZeroShot,
)
from forecast_bench.models.foundation.chronos_bolt import ChronosBoltFineTuned
from forecast_bench.models.foundation.hub import revision_tag
from forecast_bench.models.registry import classical_panel, finetuned_model_ids

pytestmark = pytest.mark.slow

#: An origin inside the fine-tuned span, so a real adapter exists for its block.
COVERED_ORIGIN = pd.Timestamp("2020-06-30")


@pytest.fixture
def frame() -> pd.DataFrame:
    """A log-RV-like series ending at a date whose block has an adapter."""
    index = pd.bdate_range(end=COVERED_ORIGIN, periods=1200)
    rng = np.random.default_rng(0)
    values = np.empty(1200)
    values[0] = -10.0
    for i in range(1, 1200):
        values[i] = 0.97 * values[i - 1] + 0.03 * (-10) + rng.standard_normal() * 0.3
    return pd.DataFrame({"spy_logrv": values}, index=index)


@pytest.fixture
def forecast_index() -> pd.DatetimeIndex:
    """Business days immediately after the covered origin."""
    return pd.bdate_range(
        start=COVERED_ORIGIN + pd.Timedelta(days=1), periods=MAX_HORIZON
    )


def test_finetuned_loads_the_adapter_for_its_fold_block(frame, forecast_index) -> None:
    """The revision loaded is the one fine-tuned on the block containing the origin."""
    model = Chronos2FineTuned(target_column="spy_logrv", series="spy_logrv", arm="A")
    model.fit(frame, origin=COVERED_ORIGIN)

    assert model.loaded_revision == revision_tag(
        "spy_logrv", "A", COVERED_ORIGIN.year, "full", model="chronos2"
    )
    assert str(COVERED_ORIGIN.year) in model.loaded_revision


@pytest.mark.parametrize("cls", [Chronos2FineTuned, ChronosBoltFineTuned])
def test_finetuned_emits_the_full_quantile_grid(cls, frame, forecast_index) -> None:
    """Both fine-tuned models satisfy the same output contract as everything else."""
    model = cls(target_column="spy_logrv", series="spy_logrv", arm="A")
    model.fit(frame, origin=COVERED_ORIGIN)
    forecast = model.predict(horizon=MAX_HORIZON, index=forecast_index)

    assert sorted(forecast.quantiles) == sorted(QUANTILE_GRID)
    forecast.assert_monotonic()
    for path in forecast.quantiles.values():
        assert len(path) == MAX_HORIZON
        assert np.isfinite(path).all()


def test_finetuned_differs_from_zero_shot(frame, forecast_index) -> None:
    """The adaptation actually changes the forecast, or it is not doing anything."""
    zero = Chronos2ZeroShot(target_column="spy_logrv")
    zero.fit(frame, origin=COVERED_ORIGIN)
    zero_forecast = zero.predict(horizon=MAX_HORIZON, index=forecast_index)

    tuned = Chronos2FineTuned(target_column="spy_logrv", series="spy_logrv", arm="A")
    tuned.fit(frame, origin=COVERED_ORIGIN)
    tuned_forecast = tuned.predict(horizon=MAX_HORIZON, index=forecast_index)

    assert not np.allclose(zero_forecast.median(), tuned_forecast.median())


def test_loading_an_adapter_does_not_contaminate_the_zero_shot_model(
    frame, forecast_index
) -> None:
    """A fine-tuned model must not mutate the cached base pipeline.

    ``PeftModel.from_pretrained`` wraps the model object it is handed. If the fine-tuned
    path reused the shared zero-shot pipeline, every zero-shot forecast in the process
    would silently become a fine-tuned one — the study's central comparison, quietly
    collapsed, with nothing raised anywhere.
    """
    zero = Chronos2ZeroShot(target_column="spy_logrv")
    zero.fit(frame, origin=COVERED_ORIGIN)
    before = zero.predict(horizon=MAX_HORIZON, index=forecast_index).median()

    tuned = Chronos2FineTuned(target_column="spy_logrv", series="spy_logrv", arm="A")
    tuned.fit(frame, origin=COVERED_ORIGIN)
    tuned.predict(horizon=MAX_HORIZON, index=forecast_index)

    fresh = Chronos2ZeroShot(target_column="spy_logrv")
    fresh.fit(frame, origin=COVERED_ORIGIN)
    after = fresh.predict(horizon=MAX_HORIZON, index=forecast_index).median()

    assert np.allclose(before, after), "the zero-shot model was contaminated"


def test_missing_adapter_fails_with_the_revision_named(frame, forecast_index) -> None:
    """An absent adapter names the tag and how to produce it, rather than 404-ing raw."""
    model = Chronos2FineTuned(
        target_column="spy_logrv", series="spy_logrv", arm="A", training_window="99y"
    )
    with pytest.raises((FileNotFoundError, KeyError)) as excinfo:
        model.fit(frame, origin=COVERED_ORIGIN)
    assert "99y" in str(excinfo.value) or "training window" in str(excinfo.value)


def test_finetuned_registered_only_where_adapters_exist() -> None:
    """Bolt was fine-tuned on the volatility track only; the panel says so."""
    assert finetuned_model_ids("spy_logrv") == [
        "Chronos2-FineTuned",
        "ChronosBolt-FineTuned",
    ]
    assert finetuned_model_ids("dgs10") == ["Chronos2-FineTuned"]

    spy = classical_panel("spy_logrv", include_finetuned=True)
    rates = classical_panel("dgs10", include_finetuned=True)
    assert "ChronosBolt-FineTuned" in spy
    assert "ChronosBolt-FineTuned" not in rates


def test_registry_builder_passes_series_to_finetuned_models() -> None:
    """A fine-tuned model built from the registry knows which adapter family it needs."""
    panel = classical_panel("dgs10", include_finetuned=True)
    model = panel["Chronos2-FineTuned"]()

    assert model.series == "dgs10"
    assert model.arm == "A"
