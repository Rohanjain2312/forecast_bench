"""Run the walk-forward backtest for one series, arm, and cadence.

Run with::

    poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A

Writes one tidy parquet per (series, arm, cadence) to ``experiments/results/forecasts/``.
Everything downstream reads that schema and only that schema.
"""

import argparse
import logging
import time

import pandas as pd

from forecast_bench.backtest.cadence import build_cadence
from forecast_bench.backtest.runner import attach_regimes, run_backtest
from forecast_bench.backtest.splitter import expanding_origin_folds
from forecast_bench.backtest.writer import write_results
from forecast_bench.config import MAX_HORIZON, get_config, setup_logging
from forecast_bench.evaluation.regimes import regime_series
from forecast_bench.models.registry import classical_panel

logger = logging.getLogger(__name__)


def load_series(name: str) -> pd.DataFrame:
    """Load a processed series from ``data/processed/``.

    Args:
        name: ``"spy_logrv"`` or ``"dgs10"``.

    Returns:
        The merged target-and-covariates frame.

    Raises:
        FileNotFoundError: If the processed file has not been built yet.
    """
    path = get_config().processed_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"    poetry run python -m scripts.fetch_data --config {name}"
        )
    return pd.read_parquet(path)


def run(
    series: str,
    cadence: str = "matched",
    arm: str = "A",
    horizon: int = MAX_HORIZON,
    include_foundation: bool = False,
) -> pd.DataFrame:
    """Run one backtest configuration end to end.

    Args:
        series: Target series name.
        cadence: ``"matched"`` or ``"native"``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        horizon: Steps forecast per fold.
        include_foundation: Add the zero-shot foundation models to the panel.

    Returns:
        The tidy results frame.

    Note:
        In Arm A every model is handed the target column only. Passing the full frame and
        trusting models to ignore covariates would make the univariate claim depend on
        each model's discipline rather than on what it was given.
    """
    frame = load_series(series)
    folds = list(expanding_origin_folds(frame.index))

    if "vixcls" in frame.columns:
        folds = attach_regimes(folds, regime_series(frame["vixcls"]))

    data = frame[[series]] if arm == "A" else frame
    panel = classical_panel(
        series, arm=arm, target_column=series, include_foundation=include_foundation
    )
    policy = build_cadence(cadence)

    logger.info(
        "Backtesting %s | arm %s | cadence %s | %d folds | models: %s",
        series,
        arm,
        cadence,
        len(folds),
        ", ".join(sorted(panel)),
    )

    started = time.perf_counter()
    results = run_backtest(
        data=data,
        target=series,
        panel=panel,
        folds=folds,
        cadence=policy,
        series=series,
        arm=arm,
        horizon=horizon,
    )
    logger.info(
        "Finished %s in %.1fs (%d rows)",
        series,
        time.perf_counter() - started,
        len(results),
    )
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=["spy_logrv", "dgs10"])
    parser.add_argument("--cadence", default="matched", choices=["matched", "native"])
    parser.add_argument("--arm", default="A", choices=["A", "B"])
    parser.add_argument(
        "--with-foundation",
        action="store_true",
        help="Add the zero-shot foundation models to the panel.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested configuration and write its parquet.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging()
    config = get_config()
    config.ensure_dirs()

    results = run(
        args.config,
        cadence=args.cadence,
        arm=args.arm,
        include_foundation=args.with_foundation,
    )

    path = write_results(results, config.forecasts_dir)

    print(f"Wrote {len(results)} rows to {path}")
    print(f"  models  : {sorted(results['model_id'].unique())}")
    print(f"  origins : {results['origin'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
