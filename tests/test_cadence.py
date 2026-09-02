"""Tests for the refit cadence policies: triggers fire exactly when expected."""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.backtest.cadence import (
    BlockCadence,
    EveryFoldCadence,
    build_cadence,
)
from forecast_bench.backtest.runner import run_backtest
from forecast_bench.backtest.splitter import expanding_origin_folds
from forecast_bench.config import TRAIN_START
from tests.conftest import StubForecaster


@pytest.fixture
def multi_year_folds():
    """Folds spanning several calendar years, so block boundaries are exercised."""
    index = pd.date_range(TRAIN_START, "2020-01-01", freq="B")
    return list(
        expanding_origin_folds(
            index,
            test_start="2015-01-01",
            test_end="2019-12-31",
        )
    )


def test_every_fold_cadence_refits_at_every_fold(multi_year_folds) -> None:
    """The native policy refits unconditionally."""
    policy = EveryFoldCadence()
    assert all(policy.should_refit("m", fold) for fold in multi_year_folds)


def test_block_cadence_refits_once_per_calendar_year(multi_year_folds) -> None:
    """The matched policy refits at the first fold of each block and not within it."""
    policy = BlockCadence(freq="YS")
    refits = [fold for fold in multi_year_folds if policy.should_refit("m", fold)]

    years = [fold.block_id for fold in refits]
    assert years == sorted(set(fold.block_id for fold in multi_year_folds))
    assert len(years) == len(set(years)), "refit fired twice within one block"


def test_block_cadence_tracks_models_independently(multi_year_folds) -> None:
    """One model's refit does not suppress another's."""
    policy = BlockCadence()
    first = multi_year_folds[0]

    assert policy.should_refit("model_a", first)
    assert policy.should_refit("model_b", first), "model_b was suppressed by model_a"
    assert not policy.should_refit("model_a", first)


def test_block_cadence_reset_clears_state(multi_year_folds) -> None:
    """A reset policy behaves as if it had never been used, so it can drive a second run."""
    policy = BlockCadence()
    first = multi_year_folds[0]

    assert policy.should_refit("m", first)
    assert not policy.should_refit("m", first)

    policy.reset()
    assert policy.should_refit("m", first)


def test_unknown_block_frequency_is_refused() -> None:
    """A frequency the policy cannot key on is an error, not a silent every-fold refit."""
    with pytest.raises(ValueError, match="Unsupported block frequency"):
        BlockCadence(freq="W")


def test_build_cadence_resolves_the_two_reported_configurations() -> None:
    """The names used on the command line map to the two policies."""
    assert isinstance(build_cadence("matched"), BlockCadence)
    assert isinstance(build_cadence("native"), EveryFoldCadence)
    with pytest.raises(KeyError, match="Unknown cadence"):
        build_cadence("weekly")


def test_cadence_changes_how_often_the_runner_fits(synthetic_frame) -> None:
    """The runner honours the policy: matched fits far less often than native."""
    folds = list(
        expanding_origin_folds(
            synthetic_frame.index,
            test_start="2004-01-01",
            test_end="2010-01-01",
        )
    )

    native = run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"stub": StubForecaster},
        folds=folds,
        cadence=EveryFoldCadence(),
        return_fitted=True,
    )
    matched = run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"stub": StubForecaster},
        folds=folds,
        cadence=BlockCadence(),
        return_fitted=True,
    )

    native_origins = {models["stub"].fitted_on_origin for _, models in native}
    matched_origins = {models["stub"].fitted_on_origin for _, models in matched}

    assert len(native_origins) == len(folds)
    assert len(matched_origins) == len({fold.block_id for fold in folds})
    assert len(matched_origins) < len(native_origins)


def test_matched_cadence_reuses_a_fit_within_a_block(synthetic_frame) -> None:
    """Within a block the same fitted object is reused, not silently refitted.

    Note the first fold's origin is the last observation *before* the test start, so it
    falls in the previous calendar year and forms a block of its own. That is correct: the
    origin is what dates a fold, and the boundary has to fall somewhere.
    """
    folds = list(
        expanding_origin_folds(
            synthetic_frame.index,
            test_start="2004-01-01",
            test_end="2005-12-31",
        )
    )
    fitted = run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"stub": StubForecaster},
        folds=folds,
        cadence=BlockCadence(),
        return_fitted=True,
    )

    by_block: dict[int, list[StubForecaster]] = {}
    for fold, models in fitted:
        by_block.setdefault(fold.block_id, []).append(models["stub"])

    assert len(by_block) > 1, "the span should cross a block boundary"
    for block, models in by_block.items():
        assert all(
            model is models[0] for model in models
        ), f"block {block} refitted mid-block"

    # One fit per block, and the objects across blocks are genuinely distinct.
    firsts = [models[0] for models in by_block.values()]
    assert len({id(model) for model in firsts}) == len(by_block)


def test_state_refreshes_every_fold_even_when_parameters_do_not(
    synthetic_frame,
) -> None:
    """The cadence governs parameters only; conditioning data always runs to the origin.

    Regression test for the bug in docs/planning/PROGRESS_NOTES.md Step 14, where caching
    the fitted object froze its conditioning data too and left 124 of 137 random-walk
    forecasts running from a value a median of 84 trading days old.

    Provenance is recorded *at fit time*. Inspecting the returned models afterwards would
    not work: the block cadence reuses one object across a block, so every reference to it
    shows the final fold's state.
    """
    recorded: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []

    class RecordingForecaster(StubForecaster):
        def fit(self, train, origin, refit_parameters=True):
            super().fit(train, origin, refit_parameters)
            recorded.append((origin, train.index.max(), refit_parameters))

    folds = list(
        expanding_origin_folds(
            synthetic_frame.index,
            test_start="2004-01-01",
            test_end="2006-12-31",
        )
    )
    run_backtest(
        data=synthetic_frame,
        target="target",
        panel={"stub": RecordingForecaster},
        folds=folds,
        cadence=BlockCadence(),
    )

    assert len(recorded) == len(folds), "fit was not called on every fold"
    for fold, (origin, seen_max, _) in zip(folds, recorded):
        assert origin == fold.origin
        assert seen_max <= fold.origin

    refits = sum(1 for _, _, refit in recorded if refit)
    assert refits == len({fold.block_id for fold in folds})
    assert refits < len(folds), "block cadence never reused a parameter fit"


def test_random_walk_forecasts_from_the_origin_under_every_cadence(
    synthetic_frame,
) -> None:
    """The random walk's median is the value at the origin, by definition.

    This property is what exposed the stale-conditioning bug. It is the baseline every
    skill score in the study is quoted against, so if it drifts, every number drifts.
    """
    from forecast_bench.models.naive import RandomWalk

    folds = list(
        expanding_origin_folds(
            synthetic_frame.index,
            test_start="2004-01-01",
            test_end="2006-12-31",
        )
    )

    for policy in (EveryFoldCadence(), BlockCadence()):
        results = run_backtest(
            data=synthetic_frame,
            target="target",
            panel={"RandomWalk": lambda: RandomWalk(target_column="target")},
            folds=folds,
            cadence=policy,
        )
        medians = results[
            (results["step"] == 1) & (np.isclose(results["quantile"], 0.5))
        ].set_index("origin")["value"]
        expected = synthetic_frame["target"].reindex(medians.index)

        assert np.allclose(
            medians.to_numpy(), expected.to_numpy()
        ), f"under {policy.name} the random walk is not forecasting from its origin"
