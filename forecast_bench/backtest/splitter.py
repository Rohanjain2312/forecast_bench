"""Expanding-origin walk-forward fold generation.

One design choice here carries the study's statistical validity, so it is stated in three
places and enforced in one: **stride == horizon**, which makes forecast windows
non-overlapping.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from forecast_bench.config import MAX_HORIZON, STRIDE, TEST_END, TEST_START, TRAIN_START

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Attributes:
        origin: The last timestamp any model in this fold may see.
        train_slice: ``slice`` usable with ``DataFrame.loc`` to select the training window.
            Expands with each fold; always ends at ``origin``.
        forecast_index: The timestamps being forecast, length ``horizon``, all strictly
            after ``origin``.
        block_id: Calendar year of the origin. The block cadence refits at the first fold
            of each block and reuses the fit within it.
        fold_index: Zero-based position in the fold sequence.
        regime: Volatility regime label at the origin. Assigned downstream once VIX is
            available; ``None`` here because the splitter is deliberately data-agnostic.

    Note:
        Frozen but **not hashable**: it carries a ``slice`` and a ``DatetimeIndex``, so it
        cannot be used as a dictionary key. Keep collections of folds as lists, or key on
        ``fold_index``.
    """

    origin: pd.Timestamp
    train_slice: slice
    forecast_index: pd.DatetimeIndex
    block_id: int
    fold_index: int
    regime: str | None = None

    @property
    def horizon(self) -> int:
        """Number of steps forecast from this origin."""
        return len(self.forecast_index)


def expanding_origin_folds(
    index: pd.DatetimeIndex,
    train_start: str = TRAIN_START,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
    stride: int = STRIDE,
    horizon: int = MAX_HORIZON,
) -> Iterator[Fold]:
    """Yield non-overlapping expanding-window folds.

    Non-overlapping is the point: ``stride == horizon`` means no two forecast windows
    share an observation, which is what makes the Diebold-Mariano test in
    ``evaluation/stats.py`` defensible. Overlapping windows leave the loss-differential
    series autocorrelated by construction and inflate significance. Most published
    backtests quietly overlap; this one does not, and changing the stride away from the
    horizon silently invalidates every p-value in the study.

    Args:
        index: The full observation index of the series being backtested.
        train_start: First date available for training.
        test_start: First date that may be forecast.
        test_end: Last date that may be forecast. A fold is emitted only if its entire
            forecast window fits at or before this date.
        stride: Observations between consecutive origins.
        horizon: Length of each forecast path.

    Yields:
        :class:`Fold` objects in chronological order.

    Raises:
        ValueError: If ``stride != horizon``, if the index is unsorted or has duplicates,
            or if the span leaves no room for a single fold.

    Note:
        No embargo gap is used, by design. Embargoes exist to stop feature-engineering
        lookbacks from straddling the train/test boundary; every feature here is a causal
        context window ending at the origin, so there is nothing to embargo. The real
        invariant — ``max(train_index) <= origin < min(forecast_index)`` — is enforced by
        construction here and asserted in ``tests/test_no_leakage.py``.
    """
    if stride != horizon:
        raise ValueError(
            f"stride ({stride}) must equal horizon ({horizon}). Unequal values produce "
            "overlapping forecast windows, which violates the independence assumption "
            "behind the Diebold-Mariano test and inflates significance. If you genuinely "
            "want overlapping windows, evaluation/stats.py must change first."
        )

    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending.")
    if index.has_duplicates:
        raise ValueError("Index has duplicate timestamps.")

    usable = index[index >= pd.Timestamp(train_start)]
    first_forecast_position = int(usable.searchsorted(pd.Timestamp(test_start), "left"))
    if first_forecast_position == 0:
        raise ValueError(
            f"No observations between {train_start} and {test_start}; there is nothing "
            "to train the first fold on."
        )

    last_forecast_date = pd.Timestamp(test_end)
    origin_position = first_forecast_position - 1
    fold_index = 0

    while origin_position + horizon < len(usable):
        forecast_index = usable[origin_position + 1 : origin_position + 1 + horizon]
        if len(forecast_index) < horizon or forecast_index[-1] > last_forecast_date:
            break

        origin = usable[origin_position]
        yield Fold(
            origin=origin,
            train_slice=slice(pd.Timestamp(train_start), origin),
            forecast_index=forecast_index,
            block_id=int(origin.year),
            fold_index=fold_index,
        )
        fold_index += 1
        origin_position += stride

    if fold_index == 0:
        raise ValueError(
            f"No complete fold fits between {test_start} and {test_end} at horizon "
            f"{horizon}. The span is too short or the index too sparse."
        )
    logger.info(
        "Generated %d folds, stride %d, horizon %d, %s to %s",
        fold_index,
        stride,
        horizon,
        test_start,
        test_end,
    )
