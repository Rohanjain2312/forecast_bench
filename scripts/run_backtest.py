"""Run the walk-forward backtest for one series, arm, and cadence.

Run with::

    poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A

Writes one tidy parquet per (series, arm, cadence) to ``experiments/results/forecasts/``.
Everything downstream reads that schema and only that schema.
"""

import argparse
import logging

import pandas as pd

from forecast_bench.backtest.runner import run_series_backtest
from forecast_bench.backtest.writer import write_results
from forecast_bench.config import get_config, setup_logging

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

    results = run_series_backtest(
        args.config,
        frame=load_series(args.config),
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
