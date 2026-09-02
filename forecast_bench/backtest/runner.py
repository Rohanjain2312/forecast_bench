"""The harness. The most important file in the repository.

Every model in the study — random walk, ARIMA, HAR, N-BEATS, Chronos-2 — is driven
fold-by-fold by this loop. Nothing here branches on which model it is holding. If a model
ever needs special handling in this file, the abstraction is wrong and
``backtest/protocol.py`` should change, not this.

The harness deliberately does not call ``darts.historical_forecasts``. Outsourcing the
backtest would mean the foundation models and the classical models no longer provably
traverse identical code, which is the single claim this file exists to support.
"""

import logging
from collections.abc import Callable, Mapping

import pandas as pd

from forecast_bench.backtest.cadence import EveryFoldCadence, RefitCadence
from forecast_bench.backtest.protocol import Forecaster
from forecast_bench.backtest.splitter import Fold
from forecast_bench.backtest.writer import ForecastWriter
from forecast_bench.config import MAX_HORIZON

logger = logging.getLogger(__name__)

#: Absolute correlation above which a training column is treated as a copy of the future.
#:
#: Deliberately near 1.0. This detector catches a *near-perfect copy* of a future target
#: value, which is what an accidental shift or a mis-joined release date produces. It is
#: not a test for "suspiciously informative" — a genuinely useful covariate on a
#: persistent financial series correlates highly with the future target and is not a bug.
#: Structural prevention of that class of problem is the allowlist in
#: ``data/fred_client.py``; see ``docs/data_protocol.md``.
LEAK_CORRELATION_THRESHOLD = 0.999


def assert_forecast_is_after_origin(
    index: pd.DatetimeIndex, origin: pd.Timestamp
) -> None:
    """Assert that a forecast index begins strictly after the fold origin.

    The cheap runtime guard from IMPLEMENTATION_PLAN.md section 3.4, run on every forecast
    of every fold.

    Args:
        index: The forecast's timestamps.
        origin: The fold's origin.

    Raises:
        AssertionError: If the index is empty or starts at or before the origin.
    """
    if len(index) == 0:
        raise AssertionError(f"Empty forecast index at origin {origin}")
    if index[0] <= origin:
        raise AssertionError(
            f"Forecast starts at {index[0]}, which is not strictly after the origin "
            f"{origin}. The model is being scored on a date it was allowed to see."
        )


def assert_fold_is_clean(
    train: pd.DataFrame,
    target: str,
    origin: pd.Timestamp,
    *,
    max_lead: int = MAX_HORIZON,
    threshold: float = LEAK_CORRELATION_THRESHOLD,
) -> None:
    """Assert that a fold's training frame contains nothing from after the origin.

    Two separate checks, because they catch different failures:

    1. **Index-level.** No training row is dated after the origin.
    2. **Value-level.** No column is a near-perfect copy of a future target value.

    The second exists because the first cannot catch it. A column built as
    ``target.shift(-21)`` sits on rows dated at or before the origin — every date
    comparison passes — while the *values* came from the future. That is exactly the shape
    of the FRED release-lag bug in ``docs/data_protocol.md``, and it is why the canary in
    ``tests/test_no_leakage.py`` exists.

    Args:
        train: The fold's training frame.
        target: Name of the target column.
        origin: The fold's origin.
        max_lead: How many steps ahead to test for a copied future value.
        threshold: Absolute correlation above which a column is treated as a copy.

    Raises:
        AssertionError: If either check fails.
        KeyError: If ``target`` is not a column of ``train``.
    """
    if target not in train.columns:
        raise KeyError(f"Target column {target!r} not in training frame")

    if len(train) and train.index.max() > origin:
        raise AssertionError(
            f"Training window ends at {train.index.max()}, after the origin {origin}."
        )

    actual = train[target]
    for column in train.columns:
        if column == target:
            continue
        values = train[column]
        if values.notna().sum() < 3 or values.std(skipna=True) == 0:
            continue
        for lead in range(1, max_lead + 1):
            correlation = values.corr(actual.shift(-lead))
            if pd.notna(correlation) and abs(correlation) > threshold:
                raise AssertionError(
                    f"Column {column!r} correlates {correlation:.6f} with the target "
                    f"{lead} steps in the future, which means it contains values from "
                    "the future. Every row is dated at or before the origin, so no date "
                    "comparison would have caught this. See docs/data_protocol.md."
                )


def run_backtest(
    data: pd.DataFrame,
    target: str,
    panel: Mapping[str, Callable[[], Forecaster]],
    folds: list[Fold],
    *,
    cadence: RefitCadence | None = None,
    series: str | None = None,
    arm: str = "A",
    horizon: int = MAX_HORIZON,
    check_leakage: bool = True,
    return_fitted: bool = False,
) -> pd.DataFrame | list[tuple[Fold, dict[str, Forecaster]]]:
    """Run every model in the panel across every fold.

    Args:
        data: Frame containing the target and any covariates, indexed by date.
        target: Name of the target column.
        panel: Mapping of model key to a zero-argument builder. A builder is called afresh
            on every refit, so no fitted state can survive a fold boundary by accident.
        folds: Folds to evaluate, from ``splitter.expanding_origin_folds``.
        cadence: Refit policy. Defaults to :class:`~forecast_bench.backtest.cadence.EveryFoldCadence`.
        series: Series label written into the results. Defaults to ``target``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        horizon: Steps to forecast per fold.
        check_leakage: Whether to run :func:`assert_fold_is_clean` on every fold.
        return_fitted: Return the fitted models per fold instead of the results frame.
            Used by the leakage suite, which needs to inspect provenance.

    Returns:
        The tidy long-format results frame, or — when ``return_fitted`` is set — a list
        of ``(fold, models)`` pairs. A list rather than a mapping because :class:`Fold`
        carries a ``slice`` and a ``DatetimeIndex`` and so is not hashable.

    Raises:
        AssertionError: If any leakage guard fires.
        KeyError: If ``target`` is not a column of ``data``.
    """
    if target not in data.columns:
        raise KeyError(f"Target column {target!r} not in data")

    policy = cadence if cadence is not None else EveryFoldCadence()
    policy.reset()

    actuals = data[target]
    writer = ForecastWriter(
        series=series or target, arm=arm, cadence=getattr(policy, "name", "unknown")
    )
    cache: dict[str, Forecaster] = {}
    fitted: list[tuple[Fold, dict[str, Forecaster]]] = []
    refits = 0

    for fold in folds:
        train = data.loc[fold.train_slice]
        if check_leakage:
            assert_fold_is_clean(train, target=target, origin=fold.origin)

        for model_key, build in panel.items():
            if policy.should_refit(model_key, fold) or model_key not in cache:
                model = build()
                model.fit(train, origin=fold.origin)
                cache[model_key] = model
                refits += 1

            forecast = cache[model_key].predict(
                horizon=horizon, index=fold.forecast_index
            )
            assert_forecast_is_after_origin(forecast.index, fold.origin)
            writer.append(forecast, fold, actuals)

        if return_fitted:
            fitted.append((fold, dict(cache)))

    logger.info(
        "Ran %d folds x %d models under cadence %r (%d fits)",
        len(folds),
        len(panel),
        getattr(policy, "name", "unknown"),
        refits,
    )

    if return_fitted:
        return fitted
    return writer.to_frame()


def attach_regimes(folds: list[Fold], regimes: pd.Series) -> list[Fold]:
    """Return copies of ``folds`` labelled with the volatility regime at each origin.

    Kept separate from fold generation so the splitter stays data-agnostic.

    Args:
        folds: Folds to label.
        regimes: Regime label per date, from ``evaluation/regimes.py``.

    Returns:
        New folds carrying a ``regime``. The label is read at the origin, which is known
        at time ``t``, so stratification introduces no look-ahead.
    """
    import dataclasses

    labelled = []
    for fold in folds:
        label = regimes.get(fold.origin)
        labelled.append(
            dataclasses.replace(
                fold, regime=None if label is None or pd.isna(label) else str(label)
            )
        )
    return labelled
