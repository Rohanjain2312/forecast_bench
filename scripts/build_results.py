"""Turn the forecast parquets into results tables.

Run with::

    poetry run python -m scripts.build_results

Reads every parquet in ``experiments/results/forecasts/``, scores it through
``evaluation/aggregate.py``, and writes the tables to ``experiments/results/metrics/``.
Every number a reader ever sees comes from here.
"""

import argparse
import logging

import pandas as pd

from forecast_bench.config import get_config, setup_logging
from forecast_bench.evaluation import aggregate as agg

logger = logging.getLogger(__name__)


def load_forecasts() -> pd.DataFrame:
    """Concatenate every forecast parquet on disk.

    Returns:
        One tidy frame spanning every (series, arm, cadence) that has been run.

    Raises:
        FileNotFoundError: If no forecast parquet exists yet.
    """
    directory = get_config().forecasts_dir
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No forecast parquets in {directory}. Run scripts.run_backtest first."
        )
    logger.info("Loading %d forecast files", len(paths))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_targets() -> dict[str, pd.Series]:
    """Load each processed target series, for per-fold MASE denominators.

    Returns:
        Mapping of series name to its target column. Series that have not been built are
        omitted rather than raising, so partial runs still produce tables.
    """
    processed = get_config().processed_dir
    targets: dict[str, pd.Series] = {}
    for name in ("spy_logrv", "dgs10"):
        path = processed / f"{name}.parquet"
        if path.is_file():
            targets[name] = pd.read_parquet(path)[name]
    return targets


def build_tables(
    forecasts: pd.DataFrame, targets: dict[str, pd.Series]
) -> dict[str, pd.DataFrame]:
    """Build every results table the data on disk supports.

    Args:
        forecasts: All tidy forecast rows.
        targets: Target series keyed by name.

    Returns:
        Mapping of table name to table. Tables whose inputs are not present yet are
        skipped with a log line rather than failing the whole build.
    """
    tables: dict[str, pd.DataFrame] = {}

    per_series = []
    for series, block in forecasts.groupby("series"):
        per_series.append(
            agg.evaluate(
                block, target=targets.get(series), group_cols=("model_id", "series")
            )
        )
    tables["headline"] = pd.concat(per_series, ignore_index=True)

    optional = {
        "cadence_comparison": agg.cadence_comparison,
        "covariate_comparison": agg.covariate_comparison,
        "regime_stratified": agg.regime_table,
        "sample_efficiency": agg.sample_efficiency_table,
        "contamination_free": agg.contamination_free_table,
    }
    for name, builder in optional.items():
        try:
            frames = [
                builder(block, target=targets.get(series))
                for series, block in forecasts.groupby("series")
            ]
            tables[name] = pd.concat(frames, ignore_index=True)
        except (ValueError, KeyError) as error:
            logger.info("Skipping %s: %s", name, error)
    return tables


def main() -> int:
    """Build and write every available results table.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    setup_logging()
    config = get_config()
    config.ensure_dirs()

    tables = build_tables(load_forecasts(), load_targets())
    for name, table in tables.items():
        path = config.metrics_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"Wrote {name}: {len(table)} rows -> {path}")

    headline = tables["headline"]
    columns = [
        c
        for c in [
            "series",
            "model_id",
            "horizon",
            "wql",
            "skill_wql",
            "mae",
            "skill_mae",
            "mase",
        ]
        if c in headline.columns
    ]
    print("\nHEADLINE — Arm A, matched cadence, WQL skill vs RandomWalk\n")
    print(
        headline.sort_values(["series", "horizon", "wql"])[columns].to_string(
            index=False, float_format=lambda v: f"{v: .4f}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
