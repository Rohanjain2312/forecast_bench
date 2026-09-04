"""Tests for merging forecast frames computed in different places.

The panel is produced in two environments: neural models train on a Colab GPU and arrive
through the Hub, everything else runs locally. Merging them is therefore a real operation
in the pipeline, and the failure it must prevent is double-counting a model.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.backtest.writer import SCHEMA, merge_forecasts


def _frame(
    model_id: str, series: str = "spy_logrv", cadence: str = "block_ys", n: int = 4
):
    """A minimal frame in the tidy schema."""
    return pd.DataFrame(
        {
            "origin": pd.to_datetime(["2015-01-02"] * n),
            "target_date": pd.to_datetime(["2015-01-05"] * n),
            "step": np.arange(1, n + 1),
            "model_id": model_id,
            "quantile": 0.5,
            "value": np.linspace(0.0, 1.0, n),
            "actual": np.linspace(0.0, 1.0, n),
            "regime": "calm",
            "block_id": 2015,
            "series": series,
            "arm": "A",
            "cadence": cadence,
        }
    )[SCHEMA]


def test_merging_distinct_models_keeps_every_row() -> None:
    """Two frames covering different models combine without loss."""
    merged = merge_forecasts(_frame("ARIMA"), _frame("N-BEATS"))

    assert set(merged.model_id.unique()) == {"ARIMA", "N-BEATS"}
    assert len(merged) == 8
    assert list(merged.columns) == SCHEMA


def test_merging_the_same_model_twice_is_refused() -> None:
    """The failure this exists to prevent: a model counted twice.

    Blind concatenation would average every metric for that model over duplicated rows
    and silently inflate its origin count, with nothing raised.
    """
    with pytest.raises(ValueError, match="appears in both"):
        merge_forecasts(_frame("ARIMA"), _frame("ARIMA"))


def test_merging_across_configurations_is_refused() -> None:
    """Frames from different series, arms or cadences are not the same panel."""
    with pytest.raises(ValueError, match="multiple series"):
        merge_forecasts(_frame("ARIMA"), _frame("N-BEATS", series="dgs10"))

    with pytest.raises(ValueError, match="multiple cadence"):
        merge_forecasts(_frame("ARIMA"), _frame("N-BEATS", cadence="every_fold"))


def test_empty_and_missing_frames_are_ignored() -> None:
    """A configuration with nothing to merge still returns the local result."""
    merged = merge_forecasts(_frame("ARIMA"), None, pd.DataFrame(columns=SCHEMA))

    assert set(merged.model_id.unique()) == {"ARIMA"}


def test_merging_nothing_is_an_error() -> None:
    """Silently returning an empty frame would produce empty results tables."""
    with pytest.raises(ValueError, match="No non-empty"):
        merge_forecasts(None, pd.DataFrame(columns=SCHEMA))


def test_merged_output_is_deterministically_ordered() -> None:
    """Sorted output keeps the parquet diff-friendly across re-runs."""
    forward = merge_forecasts(_frame("ARIMA"), _frame("N-BEATS"))
    reverse = merge_forecasts(_frame("N-BEATS"), _frame("ARIMA"))

    pd.testing.assert_frame_equal(forward, reverse)
