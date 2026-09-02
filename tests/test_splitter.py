"""Tests for fold generation: non-overlap, expansion, boundaries, and fold count."""

import pandas as pd
import pytest

from forecast_bench.backtest.splitter import Fold, expanding_origin_folds
from forecast_bench.config import MAX_HORIZON, STRIDE, TEST_END, TEST_START, TRAIN_START


@pytest.fixture
def trading_index() -> pd.DatetimeIndex:
    """A business-day index spanning the full configured study period."""
    return pd.date_range(TRAIN_START, "2026-09-01", freq="B")


@pytest.fixture
def folds(trading_index) -> list[Fold]:
    """Every fold for the configured span."""
    return list(expanding_origin_folds(trading_index))


def test_forecast_windows_never_overlap(folds) -> None:
    """No observation appears in two forecast windows.

    This is what stride == horizon buys, and what the Diebold-Mariano test depends on.
    """
    seen: set[pd.Timestamp] = set()
    for fold in folds:
        dates = set(fold.forecast_index)
        assert not (dates & seen), f"fold {fold.fold_index} reuses a target date"
        seen |= dates
    assert len(seen) == sum(len(fold.forecast_index) for fold in folds)


def test_training_window_expands_monotonically(folds) -> None:
    """Each fold's training window ends later than the previous one's, and starts identically."""
    for earlier, later in zip(folds, folds[1:]):
        assert later.origin > earlier.origin
        assert later.train_slice.stop == later.origin
        assert later.train_slice.start == earlier.train_slice.start


def test_every_origin_precedes_its_forecast_window(folds) -> None:
    """``origin < forecast_index[0]`` for every fold, with no exception."""
    for fold in folds:
        assert fold.origin < fold.forecast_index[0]
        assert fold.train_slice.stop == fold.origin


def test_block_ids_are_calendar_years(folds) -> None:
    """Block identifiers are the calendar year of the origin, non-decreasing over time."""
    for fold in folds:
        assert fold.block_id == fold.origin.year
    block_ids = [fold.block_id for fold in folds]
    assert block_ids == sorted(block_ids)
    assert len(set(block_ids)) > 1, "the span should cross at least one block boundary"


def test_fold_count_is_about_137_for_the_configured_span(folds) -> None:
    """The configured span yields roughly 137 folds, as recorded in DECISIONS.md D6."""
    assert 130 <= len(folds) <= 145, f"got {len(folds)} folds"


def test_every_fold_has_the_full_horizon(folds) -> None:
    """A partial final fold is dropped rather than scored on fewer steps."""
    for fold in folds:
        assert fold.horizon == MAX_HORIZON
        assert len(fold.forecast_index) == MAX_HORIZON


def test_no_fold_forecasts_beyond_the_test_end(folds) -> None:
    """Every forecast date lies within the configured test span."""
    for fold in folds:
        assert fold.forecast_index[0] >= pd.Timestamp(TEST_START)
        assert fold.forecast_index[-1] <= pd.Timestamp(TEST_END)


def test_origins_are_one_stride_apart(folds, trading_index) -> None:
    """Consecutive origins are exactly ``stride`` observations apart in the index."""
    positions = [trading_index.get_loc(fold.origin) for fold in folds]
    gaps = {later - earlier for earlier, later in zip(positions, positions[1:])}
    assert gaps == {STRIDE}


def test_unequal_stride_and_horizon_is_refused(trading_index) -> None:
    """A stride shorter than the horizon would overlap windows, so it is rejected."""
    with pytest.raises(ValueError, match="Diebold-Mariano"):
        list(expanding_origin_folds(trading_index, stride=5, horizon=21))


def test_unsorted_index_is_refused() -> None:
    """An unsorted index would silently produce nonsense slices."""
    index = pd.DatetimeIndex(["2020-01-03", "2020-01-02"])
    with pytest.raises(ValueError, match="sorted"):
        list(expanding_origin_folds(index))


def test_duplicate_index_is_refused() -> None:
    """A duplicated timestamp would double-count an observation in every fold."""
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError, match="duplicate"):
        list(expanding_origin_folds(index))


def test_span_with_no_training_data_is_refused() -> None:
    """A test start at or before the training start leaves nothing to train on."""
    index = pd.date_range("2015-01-01", periods=500, freq="B")
    with pytest.raises(ValueError, match="nothing"):
        list(expanding_origin_folds(index, train_start="2015-01-01"))


def test_span_too_short_for_one_fold_is_refused() -> None:
    """A test span shorter than the horizon yields no fold, and says so."""
    index = pd.date_range("2014-01-01", periods=400, freq="B")
    with pytest.raises(ValueError, match="No complete fold"):
        list(
            expanding_origin_folds(
                index,
                train_start="2014-01-01",
                test_start="2015-06-01",
                test_end="2015-06-20",
            )
        )
