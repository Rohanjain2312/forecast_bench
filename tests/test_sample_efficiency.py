"""Tests for the sample-efficiency window sizing shared by fine-tuning and neural training.

Regression coverage for a bug hit live on the first Colab run: ``"1y"`` was defined as a
literal 252 raw trading days, shorter than the study's 512-step context window, so it could
not supply even one training example — on either the Chronos-2 fine-tuning path or the
darts neural path, since both fix context length at 512. See
docs/planning/PROGRESS_NOTES.md, Step 16.
"""

import pytest

from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON, TRAIN_START
from forecast_bench.models.base import (
    SAMPLE_EFFICIENCY_DAYS,
    sample_efficiency_window_size,
)
from forecast_bench.models.foundation.finetune import slice_training_window

MINIMUM_EXAMPLE_SIZE = CONTEXT_LENGTH + MAX_HORIZON


@pytest.mark.parametrize("label", ["1y", "3y", "10y"])
def test_every_finite_window_supports_at_least_one_training_example(label) -> None:
    """No sample-efficiency slice can be shorter than one context-plus-horizon window.

    This is the exact property that was violated: "1y" resolved to 252, which is less
    than CONTEXT_LENGTH + MAX_HORIZON (533), so fine-tuning could not draw a single
    (context, target) pair from it.
    """
    size = sample_efficiency_window_size(label)
    assert size is not None
    assert size >= MINIMUM_EXAMPLE_SIZE


def test_full_window_means_no_truncation() -> None:
    """'full' resolves to None, the signal to use every available observation."""
    assert sample_efficiency_window_size("full") is None


def test_window_sizes_are_strictly_increasing() -> None:
    """1y < 3y < 10y, so the sweep actually sweeps something."""
    sizes = [sample_efficiency_window_size(label) for label in ("1y", "3y", "10y")]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == 3


def test_unknown_label_is_refused() -> None:
    """A typo'd label fails loudly rather than silently using the full window."""
    with pytest.raises(KeyError, match="Unknown training window"):
        sample_efficiency_window_size("2y")


def test_context_length_is_actually_reflected_in_the_minimum() -> None:
    """Sanity check that the function is reading the real study constants, not a copy."""
    bespoke = sample_efficiency_window_size("1y", context_length=100, horizon=10)
    assert bespoke == 100 + 10 + SAMPLE_EFFICIENCY_DAYS["1y"] - 1


def test_slice_training_window_reproduces_the_first_colab_block(
    synthetic_series,
) -> None:
    """The exact failing case: the study's very first block, '1y' window.

    synthetic_series spans TRAIN_START onward for ~12 years, comparable to the real
    2000-01-01 -> 2014-12-31 span the first block trains on.
    """
    origin = synthetic_series.index[synthetic_series.index <= "2011-01-01"][-1]

    values = slice_training_window(synthetic_series, origin, "1y")

    assert len(values) >= MINIMUM_EXAMPLE_SIZE


def test_slice_training_window_rejects_a_window_shorter_than_the_train_span() -> None:
    """A block too close to TRAIN_START to fill even the minimum window still fails loudly.

    The fix widens what '1y' means; it does not remove the underlying guard for a
    genuinely too-short history, which real data should never hit given the ~15-year
    initial span but which the function must still refuse rather than silently proceed.
    """
    import numpy as np
    import pandas as pd

    short_index = pd.date_range(TRAIN_START, periods=100, freq="B")
    short_series = pd.Series(np.arange(100, dtype=float), index=short_index)

    with pytest.raises(ValueError, match="fewer than"):
        slice_training_window(short_series, short_index[-1], "1y")
