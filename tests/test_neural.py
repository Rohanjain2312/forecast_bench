"""Tests for the from-scratch neural baselines: window sizing and early stopping.

Both regressions here were found on real GPU runs, not by reading the code.
See docs/planning/PROGRESS_NOTES.md, Step 16.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON
from forecast_bench.models.base import sample_efficiency_window_size
from forecast_bench.models.neural._darts import PATIENCE, VALIDATION_WINDOWS
from forecast_bench.models.neural.nbeats import NBEATS

pytestmark = pytest.mark.slow

MINIMUM_SAMPLE = CONTEXT_LENGTH + MAX_HORIZON


def _series(n: int, name: str = "target") -> pd.DataFrame:
    """A deterministic mean-reverting series of length ``n``."""
    index = pd.date_range("2000-01-03", periods=n, freq="B")
    rng = np.random.default_rng(0)
    values = np.empty(n)
    values[0] = -10.0
    for i in range(1, n):
        values[i] = 0.97 * values[i - 1] + 0.03 * (-10) + rng.standard_normal() * 0.3
    return pd.Series(values, index=index, name=name).to_frame()


@pytest.mark.parametrize("window", ["1y", "3y", "10y"])
def test_every_sweep_window_leaves_a_trainable_series(window) -> None:
    """The validation reserve never eats the window below one usable sample.

    Regression test: a fixed 252-observation reserve left the ``1y`` slice with 532
    observations against a 533 minimum, and darts raised "The input `series` are too short
    to extract even a single sample" on a real GPU run. The reserve is now capped at half
    the surplus above the minimum.
    """
    raw = sample_efficiency_window_size(window)
    surplus = raw - MINIMUM_SAMPLE
    reserve = min(VALIDATION_WINDOWS, surplus // 2)

    assert raw - reserve >= MINIMUM_SAMPLE


def test_full_window_reserve_is_unchanged_by_the_cap() -> None:
    """The cap must not alter the headline run's split, only rescue short windows."""
    raw = 6707  # the real spy_logrv length
    surplus = raw - MINIMUM_SAMPLE
    assert min(VALIDATION_WINDOWS, surplus // 2) == VALIDATION_WINDOWS


def test_early_stopping_is_actually_attached_and_fires() -> None:
    """A validation series is passed and EarlyStopping stops training before n_epochs.

    Regression test: the guard was ``len(validation_values) > input_chunk + horizon``,
    comparing a reserve that is at most 252 against a minimum of 533 — never true. So
    ``val_series`` was silently never passed, no callback was configured, and every
    neural fit ran all 50 epochs with no validation at all, contradicting
    IMPLEMENTATION_PLAN.md section 4b.
    """
    frame = _series(1600)
    model = NBEATS(
        target_column="target", n_epochs=30, input_chunk_length=64, device="cpu"
    )
    model.fit(frame, origin=frame.index[-1])

    trainer = model._model.trainer
    callbacks = [type(c).__name__ for c in trainer.callbacks]

    assert "EarlyStopping" in callbacks
    assert trainer.current_epoch < 30, "early stopping did not fire"

    stopper = next(c for c in trainer.callbacks if type(c).__name__ == "EarlyStopping")
    assert stopper.patience == PATIENCE
    assert stopper.monitor == "val_loss"


def test_a_minimal_window_trains_without_validation_rather_than_crashing() -> None:
    """With no room to reserve validation, the model trains anyway and says so."""
    frame = _series(MINIMUM_SAMPLE + 4)
    model = NBEATS(
        target_column="target",
        n_epochs=1,
        input_chunk_length=CONTEXT_LENGTH,
        device="cpu",
    )

    model.fit(frame, origin=frame.index[-1])

    callbacks = [type(c).__name__ for c in model._model.trainer.callbacks]
    assert "EarlyStopping" not in callbacks


def test_a_window_below_one_sample_is_refused_clearly() -> None:
    """Genuinely too little data raises a message naming the shortfall."""
    frame = _series(MINIMUM_SAMPLE - 10)
    model = NBEATS(
        target_column="target",
        n_epochs=1,
        input_chunk_length=CONTEXT_LENGTH,
        device="cpu",
    )

    with pytest.raises(ValueError, match="fewer than the 533 needed"):
        model.fit(frame, origin=frame.index[-1])
