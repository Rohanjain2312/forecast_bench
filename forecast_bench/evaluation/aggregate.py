"""Turn the tidy forecast parquet into the six results tables.

Reads the one schema defined in ``backtest/writer.py`` and imports every metric from
``evaluation/metrics.py``. Nothing here recomputes a metric locally, which is what keeps
``docs/benchmark_results.md``, the README, and the Space showing the same numbers.
"""

import logging

import numpy as np
import pandas as pd

from forecast_bench.config import HORIZONS, QUANTILE_GRID
from forecast_bench.evaluation import metrics as m
from forecast_bench.models.registry import BASELINE_MODEL_ID

logger = logging.getLogger(__name__)

#: Interval levels reported as (coverage, width) pairs.
INTERVAL_LEVELS = {80: (0.1, 0.9), 95: (0.025, 0.975)}

#: Metrics for which a skill score against the baseline is reported.
SKILL_METRICS = ["wql", "mae", "rmse"]


def mase_denominators(
    target: pd.Series, origins: pd.Series, season: int = m.DEFAULT_SEASON
) -> dict[pd.Timestamp, float]:
    """Per-fold seasonal-naive denominators, each from that fold's training window.

    Args:
        target: The full target series, indexed by date.
        origins: Fold origins to compute a denominator for.
        season: Seasonal lag.

    Returns:
        Mapping of origin to denominator.

    Note:
        Each denominator uses only observations at or before its origin, so it is the same
        quantity the fold's models were fitted against. Computing one denominator on the
        full series is the most common way MASE is reported wrongly.
    """
    denominators: dict[pd.Timestamp, float] = {}
    for origin in pd.DatetimeIndex(pd.unique(origins)):
        window = target.loc[target.index <= origin]
        try:
            denominators[origin] = m.seasonal_naive_denominator(
                window.to_numpy(dtype=float), season=season
            )
        except ValueError:
            denominators[origin] = float("nan")
    return denominators


def _pivot_quantiles(block: pd.DataFrame) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Split one model-horizon block into actuals and a level-to-forecast mapping.

    Args:
        block: Rows sharing a model and step, one row per (origin, quantile).

    Returns:
        A ``(actuals, quantile_forecasts)`` pair aligned by origin.
    """
    wide = block.pivot_table(
        index="origin", columns="quantile", values="value", aggfunc="first"
    )
    actual = block.groupby("origin")["actual"].first().reindex(wide.index)
    forecasts = {
        level: wide[level].to_numpy(dtype=float)
        for level in QUANTILE_GRID
        if level in wide.columns
    }
    return actual.to_numpy(dtype=float), forecasts


def evaluate(
    results: pd.DataFrame,
    target: pd.Series | None = None,
    group_cols: tuple[str, ...] = ("model_id",),
    horizons: list[int] = HORIZONS,
    baseline: str = BASELINE_MODEL_ID,
) -> pd.DataFrame:
    """Score a tidy results frame, one row per group and horizon.

    Args:
        results: Tidy long-format forecasts from the runner.
        target: The full target series, used for per-fold MASE denominators. MASE is
            omitted when this is ``None``.
        group_cols: Columns defining a reported group. Always includes ``model_id``.
        horizons: Steps to report.
        baseline: Model id used as the skill-score reference.

    Returns:
        A long metrics table with skill scores attached.

    Raises:
        ValueError: If the frame is empty or lacks ``model_id``.
    """
    if results.empty:
        raise ValueError("No forecast rows to evaluate")
    if "model_id" not in results.columns:
        raise ValueError("Results frame has no model_id column")

    group_cols = tuple(dict.fromkeys(("model_id", *group_cols)))
    denominators = (
        mase_denominators(target, results["origin"]) if target is not None else {}
    )

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        at_horizon = results[results["step"] == horizon]
        if at_horizon.empty:
            logger.warning("No rows at step %d; skipping that horizon", horizon)
            continue

        for keys, block in at_horizon.groupby(list(group_cols), dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            actual, forecasts = _pivot_quantiles(block)
            if 0.5 not in forecasts:
                continue
            median = forecasts[0.5]

            row: dict[str, object] = dict(zip(group_cols, keys))
            row["horizon"] = horizon
            row["n_origins"] = int(len(actual))
            row["mae"] = m.mae(actual, median)
            row["rmse"] = m.rmse(actual, median)
            row["smape"] = m.smape(actual, median)
            row["wql"] = m.weighted_quantile_loss(
                actual, forecasts, [q for q in QUANTILE_GRID if q in forecasts]
            )

            if denominators:
                origins = pd.DatetimeIndex(block.groupby("origin").first().index)
                scale = np.nanmean([denominators.get(o, np.nan) for o in origins])
                row["mase"] = (
                    row["mae"] / scale if scale and np.isfinite(scale) else np.nan
                )

            origin_values = (
                target.reindex(pd.DatetimeIndex(block["origin"].unique())).to_numpy()
                if target is not None
                else np.zeros_like(actual)
            )
            row["directional_accuracy"] = m.directional_accuracy(
                actual, median, origin_values
            )

            for label, (low, high) in INTERVAL_LEVELS.items():
                if low in forecasts and high in forecasts:
                    coverage, width = m.interval_coverage_and_width(
                        actual, forecasts[low], forecasts[high]
                    )
                    row[f"coverage_{label}"] = coverage
                    row[f"width_{label}"] = width

            rows.append(row)

    table = pd.DataFrame(rows)
    return _attach_skill_scores(table, group_cols, baseline)


def _attach_skill_scores(
    table: pd.DataFrame, group_cols: tuple[str, ...], baseline: str
) -> pd.DataFrame:
    """Add ``skill_<metric>`` columns relative to the baseline model.

    Args:
        table: Metrics table from :func:`evaluate`.
        group_cols: Columns defining a reported group.
        baseline: Model id used as the reference.

    Returns:
        The table with skill columns appended.
    """
    if table.empty or baseline not in set(table["model_id"]):
        logger.warning(
            "Baseline %r absent from the results; skill scores omitted.", baseline
        )
        return table

    # Skill is computed within a group that is identical except for the model.
    context = [column for column in group_cols if column != "model_id"] + ["horizon"]
    reference = (
        table[table["model_id"] == baseline].set_index(context)
        if context
        else table[table["model_id"] == baseline]
    )

    for metric in SKILL_METRICS:
        if metric not in table.columns:
            continue
        aligned = (
            table.set_index(context)[metric].index.map(
                lambda key, ref=reference, mt=metric: ref[mt].get(key, np.nan)
            )
            if context
            else np.repeat(float(reference[metric].iloc[0]), len(table))
        )
        baseline_values = np.asarray(aligned, dtype=float)
        table[f"skill_{metric}"] = np.where(
            baseline_values != 0, 1.0 - table[metric] / baseline_values, np.nan
        )
    return table


def headline_table(
    results: pd.DataFrame, target: pd.Series | None = None
) -> pd.DataFrame:
    """Table 1 — model x series x horizon, matched cadence, Arm A.

    Args:
        results: Tidy forecasts, possibly spanning several arms and cadences.
        target: The full target series, for MASE.

    Returns:
        The registered headline cut. Everything else in this module is secondary and is
        labelled as such wherever it appears.
    """
    subset = results
    if "arm" in results.columns:
        subset = subset[subset["arm"] == "A"]
    if "cadence" in results.columns and (subset["cadence"] == "block_ys").any():
        subset = subset[subset["cadence"] == "block_ys"]
    return evaluate(subset, target=target, group_cols=("model_id", "series"))


def cadence_comparison(
    results: pd.DataFrame, target: pd.Series | None = None
) -> pd.DataFrame:
    """Table 2 — matched versus native refit cadence.

    Args:
        results: Tidy forecasts spanning both cadences.
        target: The full target series, for MASE.

    Returns:
        Metrics grouped by cadence. The gap measures how much of the classical arm's
        performance comes from refitting frequently rather than from the model.
    """
    return evaluate(
        results, target=target, group_cols=("model_id", "series", "cadence")
    )


def covariate_comparison(
    results: pd.DataFrame, target: pd.Series | None = None
) -> pd.DataFrame:
    """Table 3 — Arm A versus Arm B.

    Args:
        results: Tidy forecasts spanning both arms.
        target: The full target series, for MASE.

    Returns:
        Metrics grouped by arm.
    """
    return evaluate(results, target=target, group_cols=("model_id", "series", "arm"))


def regime_table(
    results: pd.DataFrame, target: pd.Series | None = None
) -> pd.DataFrame:
    """Table 4 — model x regime x horizon.

    Args:
        results: Tidy forecasts carrying a ``regime`` column.
        target: The full target series, for MASE.

    Returns:
        Regime-stratified metrics.
    """
    return evaluate(results, target=target, group_cols=("model_id", "series", "regime"))


def sample_efficiency_table(
    results: pd.DataFrame, target: pd.Series | None = None
) -> pd.DataFrame:
    """Table 5 — skill against training-window size.

    Args:
        results: Tidy forecasts carrying a ``training_window`` column.
        target: The full target series, for MASE.

    Returns:
        Metrics grouped by training-window size.

    Raises:
        ValueError: If the results carry no ``training_window`` column.
    """
    if "training_window" not in results.columns:
        raise ValueError(
            "Results carry no training_window column; run the sample-efficiency sweep first."
        )
    return evaluate(
        results, target=target, group_cols=("model_id", "series", "training_window")
    )


def contamination_free_table(
    results: pd.DataFrame,
    target: pd.Series | None = None,
    release_date: str = "2025-11-01",
) -> pd.DataFrame:
    """Table 6 — origins after a model's release date only.

    Args:
        results: Tidy forecasts.
        target: The full target series, for MASE.
        release_date: Earliest origin considered uncontaminated. Chronos-2 was released
            October 2025.

    Returns:
        Metrics restricted to post-release origins, with ``n_origins`` carried in the
        table itself.

    Note:
        The sample is roughly eight folds — far too few for significance, and the count is
        reported **in the table** rather than only in the prose, so a reader cannot take a
        number from it without seeing what it rests on. This is the only genuinely clean
        zero-shot read available; running it with its inadequate sample size stated is more
        honest than not running it.
    """
    subset = results[pd.DatetimeIndex(results["origin"]) >= pd.Timestamp(release_date)]
    if subset.empty:
        raise ValueError(f"No forecast origins on or after {release_date}")
    table = evaluate(subset, target=target, group_cols=("model_id", "series"))
    table["contamination_free"] = True
    return table
