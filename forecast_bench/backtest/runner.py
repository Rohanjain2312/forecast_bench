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
from forecast_bench.backtest.splitter import Fold, expanding_origin_folds
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


def assert_conditioned_on_origin(model: Forecaster, origin: pd.Timestamp) -> None:
    """Assert a model has just conditioned on data running to this fold's origin.

    Args:
        model: The model that was just fitted.
        origin: The fold's origin.

    Raises:
        AssertionError: If the model reports conditioning on a different origin.

    Note:
        The cheap runtime counterpart to :func:`assert_forecast_is_after_origin`, and the
        guard that would have caught the stale-conditioning bug in PROGRESS_NOTES.md Step
        14 on the first fold rather than in the results table. Models that do not expose
        ``fitted_on_origin`` are skipped rather than rejected, since the protocol only
        recommends the attribute.
    """
    recorded = getattr(model, "fitted_on_origin", None)
    if recorded is not None and recorded != origin:
        raise AssertionError(
            f"{getattr(model, 'model_id', model)} reports conditioning on {recorded} at a "
            f"fold whose origin is {origin}. The refit cadence governs parameters only; "
            "conditioning data must always run to the fold's origin."
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
            on every parameter refit, so no fitted parameter can survive a block boundary
            by accident. Conditioning data is refreshed on every fold regardless.
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
            # The cadence governs parameters only. Every model is handed data running to
            # this fold's origin on every fold, so no model can forecast from stale
            # conditioning data -- see the Step 14 entry in PROGRESS_NOTES.md for what
            # happened when it could.
            refit = policy.should_refit(model_key, fold) or model_key not in cache
            if refit:
                cache[model_key] = build()
                refits += 1
            cache[model_key].fit(train, origin=fold.origin, refit_parameters=refit)
            assert_conditioned_on_origin(cache[model_key], fold.origin)

            forecast = cache[model_key].predict(
                horizon=horizon, index=fold.forecast_index
            )
            assert_forecast_is_after_origin(forecast.index, fold.origin)
            writer.append(forecast, fold, actuals)

        if return_fitted:
            fitted.append((fold, dict(cache)))

    logger.info(
        "Ran %d folds x %d models under cadence %r (%d parameter refits)",
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


def run_series_backtest(
    series: str,
    frame: pd.DataFrame | None = None,
    cadence: str = "matched",
    arm: str = "A",
    horizon: int = MAX_HORIZON,
    include_foundation: bool = False,
    include_neural: bool = False,
    include_finetuned: bool = False,
    only_models: list[str] | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """Run one full backtest configuration for a series.

    Lives in the package rather than in ``scripts/`` so that Colab and the Space, which
    ``pip install`` this repository, can call exactly the code the CLI calls. A notebook
    that reimplemented this would be a notebook that drifts from the repository.

    Args:
        series: Target series name.
        frame: Processed frame. Loaded from the Hub when ``None``.
        cadence: ``"matched"`` or ``"native"``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        horizon: Steps forecast per fold.
        include_foundation: Add the zero-shot foundation models.
        include_neural: Add the from-scratch neural baselines, which need a GPU.
        include_finetuned: Add the LoRA-adapted foundation models, which load their
            per-block adapters from the Hub.
        only_models: Restrict the panel to these model ids. Used by the
            sample-efficiency sweep, where only models with an adapter at every
            window can take part.
        **model_kwargs: Extra keyword arguments passed to every model builder, e.g.
            ``device`` or ``training_window_days`` for the sample-efficiency sweep.

    Returns:
        The tidy long-format results frame.

    Note:
        In Arm A every model is handed the target column only. Passing the full frame and
        trusting each model to ignore covariates would make the univariate claim depend on
        model discipline rather than on what the model was given.
    """
    from forecast_bench.backtest.cadence import build_cadence
    from forecast_bench.evaluation.regimes import regime_series
    from forecast_bench.models.registry import classical_panel

    if frame is None:
        from forecast_bench.data.hub import load_processed

        frame = load_processed(series)

    folds = list(expanding_origin_folds(frame.index))
    if "vixcls" in frame.columns:
        folds = attach_regimes(folds, regime_series(frame["vixcls"]))

    data = frame[[series]] if arm == "A" else frame
    panel = classical_panel(
        series,
        arm=arm,
        target_column=series,
        include_foundation=include_foundation,
        include_neural=include_neural,
        include_finetuned=include_finetuned,
    )
    if only_models is not None:
        missing = set(only_models) - set(panel)
        if missing:
            raise KeyError(
                f"Requested models not in the {series} panel: {sorted(missing)}. "
                f"Available: {sorted(panel)}"
            )
        panel = {k: v for k, v in panel.items() if k in set(only_models)}

    if model_kwargs:
        panel = {
            model_id: (lambda b=build, k=model_kwargs: _build_with(b, k))
            for model_id, build in panel.items()
        }

    logger.info(
        "Backtesting %s | arm %s | cadence %s | %d folds | models: %s",
        series,
        arm,
        cadence,
        len(folds),
        ", ".join(sorted(panel)),
    )
    return run_backtest(
        data=data,
        target=series,
        panel=panel,
        folds=folds,
        cadence=build_cadence(cadence),
        series=series,
        arm=arm,
        horizon=horizon,
    )


def _build_with(build, kwargs: dict):
    """Apply extra keyword arguments to a model builder where the model accepts them.

    Args:
        build: A zero-argument builder from the registry.
        kwargs: Extra keyword arguments to apply.

    Returns:
        A model instance, with any argument its constructor does not accept dropped.
    """
    import inspect

    model = build()
    accepted = inspect.signature(type(model).__init__).parameters
    for name, value in kwargs.items():
        if name in accepted:
            setattr(model, name, value)
    return model
