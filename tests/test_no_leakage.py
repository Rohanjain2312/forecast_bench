"""The hard constraint, made executable.

Five checks, from IMPLEMENTATION_PLAN.md section 3.5. Written **before** the harness so
that the harness is built against them rather than around them.

All five checks are live and enforced as of build Step 13. Each was written before the
module it guards and carried ``xfail(strict=True)`` until that module landed, so every one
of them has been watched turning from failing to passing.

**This file must never be weakened or skipped to make a run pass.** If a guard fails, the
run is wrong, not the guard.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import MAX_HORIZON

# --- Check 1: fold boundaries ----------------------------------------------------------


def test_every_fold_trains_only_at_or_before_its_origin(
    synthetic_series, tiny_fold_spec
) -> None:
    """For every fold: ``max(train_index) <= origin < min(forecast_index)``.

    This is the invariant the whole study rests on. An embargo gap is deliberately not
    used — the real constraint is enforced here rather than approximated by a cosmetic
    buffer.
    """
    from forecast_bench.backtest.splitter import expanding_origin_folds

    folds = list(expanding_origin_folds(synthetic_series.index, **tiny_fold_spec))
    assert folds, "splitter produced no folds"

    for fold in folds:
        train_index = synthetic_series.loc[fold.train_slice].index
        assert train_index.max() <= fold.origin
        assert fold.forecast_index.min() > fold.origin
        assert len(fold.forecast_index) == MAX_HORIZON


def test_forecast_windows_never_overlap(synthetic_series, tiny_fold_spec) -> None:
    """No observation appears in two forecast windows.

    ``stride == horizon`` is what makes this true, and it is what makes the
    Diebold-Mariano test in ``evaluation/stats.py`` defensible. Changing the stride
    invalidates that test, so the property is pinned here rather than left as a comment.
    """
    from forecast_bench.backtest.splitter import expanding_origin_folds

    folds = list(expanding_origin_folds(synthetic_series.index, **tiny_fold_spec))
    seen: set[pd.Timestamp] = set()
    for fold in folds:
        dates = set(fold.forecast_index)
        assert not (
            dates & seen
        ), f"fold at {fold.origin} reuses an earlier target date"
        seen |= dates


# --- Check 2: fitted objects are fold-local --------------------------------------------


def test_every_fitted_model_records_its_fold_origin(
    synthetic_frame, tiny_fold_spec
) -> None:
    """Every model fitted during a fold carries that fold's origin.

    A fitted object whose recorded origin differs from the fold it is used in has crossed
    a fold boundary: a scaler, an ARIMA order, or a residual quantile computed somewhere
    it should not have been.
    """
    from forecast_bench.backtest.runner import run_backtest
    from forecast_bench.backtest.splitter import expanding_origin_folds
    from tests.conftest import StubForecaster

    folds = list(expanding_origin_folds(synthetic_frame.index, **tiny_fold_spec))
    fitted = run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"stub": StubForecaster},
        folds=folds,
        return_fitted=True,
    )

    for fold, models in fitted:
        for model in models.values():
            assert model.fitted_on_origin == fold.origin
            assert model.seen_max_index <= fold.origin


def test_runner_rejects_a_forecast_that_starts_at_or_before_the_origin(
    synthetic_frame, tiny_fold_spec
) -> None:
    """The runner's cheap runtime guard fires when a forecast index starts too early."""
    from forecast_bench.backtest.runner import assert_forecast_is_after_origin
    from forecast_bench.backtest.splitter import expanding_origin_folds

    fold = next(iter(expanding_origin_folds(synthetic_frame.index, **tiny_fold_spec)))
    bad_index = pd.DatetimeIndex([fold.origin])

    with pytest.raises(AssertionError):
        assert_forecast_is_after_origin(bad_index, fold.origin)


# --- Check 3: the canary ---------------------------------------------------------------


def test_canary_error_collapses_when_the_future_is_injected(
    leaky_frame, synthetic_frame, tiny_fold_spec
) -> None:
    """A model reading a perfect copy of the future scores near-zero error; a clean one does not.

    This is the test that proves the detector has power. Without it, "our error is not
    suspiciously low" is an assertion nobody has watched fail.
    """
    from forecast_bench.backtest.runner import run_backtest
    from forecast_bench.backtest.splitter import expanding_origin_folds
    from tests.conftest import CheatingForecaster

    leaky_folds = list(expanding_origin_folds(leaky_frame.index, **tiny_fold_spec))
    leaked = run_backtest(
        data=leaky_frame,
        target="target",
        panel={"cheat": CheatingForecaster},
        folds=leaky_folds,
        # Deliberately off: this test measures what leakage does to the error. That the
        # guard would have stopped it is the next test's job.
        check_leakage=False,
    )

    clean_folds = list(expanding_origin_folds(synthetic_frame.index, **tiny_fold_spec))
    clean = run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"cheat": CheatingForecaster},
        folds=clean_folds,
    )

    leaked_error = _median_absolute_error(leaked)
    clean_error = _median_absolute_error(clean)

    assert leaked_error < 1e-6, "injected future did not collapse the error"
    assert (
        clean_error > 100 * leaked_error
    ), "clean run is suspiciously close to the leak"


def test_canary_leak_is_detected_by_the_fold_guard(leaky_frame, tiny_fold_spec) -> None:
    """The leakage guard flags a training frame containing a shifted copy of the target.

    Index-level checks alone cannot catch this: every row of the leaked frame is dated at
    or before the origin. The *values* are what came from the future. This is why the
    guard inspects column content, and why ``docs/data_protocol.md`` restricts inputs
    structurally rather than trusting a date comparison.
    """
    from forecast_bench.backtest.runner import assert_fold_is_clean
    from forecast_bench.backtest.splitter import expanding_origin_folds

    fold = next(iter(expanding_origin_folds(leaky_frame.index, **tiny_fold_spec)))
    train = leaky_frame.loc[fold.train_slice]

    # The index-level invariant holds even though the frame is leaking.
    assert train.index.max() <= fold.origin

    with pytest.raises(AssertionError, match="future"):
        assert_fold_is_clean(train, target="target", origin=fold.origin)


# --- Check 4: per-fold recomputation ---------------------------------------------------


def test_mase_denominator_and_rw_quantiles_are_recomputed_each_fold(
    synthetic_frame, tiny_fold_spec
) -> None:
    """Fold-local quantities actually vary across folds.

    A constant value across folds means someone computed it once on the whole series and
    cached it — the exact mistake that makes MASE look better than it is.
    """
    from forecast_bench.backtest.splitter import expanding_origin_folds
    from forecast_bench.evaluation.metrics import seasonal_naive_denominator
    from forecast_bench.models.naive import RandomWalk

    folds = list(expanding_origin_folds(synthetic_frame.index, **tiny_fold_spec))
    denominators: list[float] = []
    spreads: list[float] = []

    for fold in folds:
        train = synthetic_frame.loc[fold.train_slice]
        denominators.append(seasonal_naive_denominator(train["target"]))

        model = RandomWalk()
        model.fit(train, origin=fold.origin)
        forecast = model.predict(horizon=MAX_HORIZON, index=fold.forecast_index)
        spreads.append(
            float(forecast.quantiles[0.975][-1] - forecast.quantiles[0.025][-1])
        )

    assert len(set(np.round(denominators, 10))) > 1, "MASE denominator is constant"
    assert len(set(np.round(spreads, 10))) > 1, "RW quantile spread is constant"


# --- Check 5: frozen regime thresholds -------------------------------------------------


def test_regime_thresholds_match_the_frozen_config() -> None:
    """The loaded thresholds equal the committed values, asserted at import time.

    Recomputing them on the full sample would define "stressed" using knowledge that 2020
    and 2022 happened. That is leakage in the reporting layer, and it is easier to miss
    than leakage in the modelling layer.
    """
    from forecast_bench.evaluation.regimes import CALM_UPPER, NORMAL_UPPER

    assert CALM_UPPER == 15.9
    assert NORMAL_UPPER == 22.5582


# --- helpers ---------------------------------------------------------------------------


def _median_absolute_error(results: pd.DataFrame) -> float:
    """Median absolute error of the median forecast in a tidy results frame.

    Args:
        results: Tidy long-format forecast frame from the runner.

    Returns:
        The median absolute error at the 0.5 quantile.
    """
    median = results[np.isclose(results["quantile"], 0.5)]
    return float((median["value"] - median["actual"]).abs().median())
