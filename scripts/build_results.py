"""Turn the forecast parquets into results tables.

Run with::

    poetry run python -m scripts.build_results

Reads every parquet in ``experiments/results/forecasts/``, scores it through
``evaluation/aggregate.py``, and writes the tables to ``experiments/results/metrics/``.
Every number a reader ever sees comes from here.
"""

import argparse
import logging
from pathlib import Path

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

    Note:
        Deliberately **not** recursive. The sample-efficiency sweep lives in
        ``forecasts/sample_efficiency/`` and its ``full`` slice repeats the headline run
        exactly, so globbing recursively here would double-count those rows into every
        table: inflated ``n_origins``, and metrics silently averaged over duplicates.
        :func:`load_sample_efficiency` reads that directory separately.
    """
    directory = get_config().forecasts_dir
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No forecast parquets in {directory}. Run scripts.run_backtest first."
        )
    logger.info("Loading %d forecast files", len(paths))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_sample_efficiency() -> pd.DataFrame | None:
    """Load the sample-efficiency sweep, if it has been run.

    Returns:
        The sweep frame, or ``None`` when the sweep directory is empty. Returning ``None``
        rather than raising lets a partial run still build every other table.
    """
    directory = get_config().sample_efficiency_dir
    paths = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not paths:
        return None
    logger.info("Loading %d sample-efficiency files", len(paths))
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

    # The sweep is scored from its own directory, never from `forecasts`, so that its
    # duplicated `full` slice cannot leak into the tables above.
    sweep = load_sample_efficiency()
    if sweep is None:
        logger.info("Skipping sample_efficiency: no sweep files on disk")
    else:
        try:
            tables["sample_efficiency"] = pd.concat(
                [
                    agg.sample_efficiency_table(block, target=targets.get(series))
                    for series, block in sweep.groupby("series")
                ],
                ignore_index=True,
            )
        except (ValueError, KeyError) as error:
            logger.info("Skipping sample_efficiency: %s", error)
    return tables


#: Columns shown in the markdown report, in order.
REPORT_COLUMNS = [
    "model_id",
    "horizon",
    "wql",
    "skill_wql",
    "mae",
    "skill_mae",
    "mase",
    "coverage_80",
    "width_80",
    "coverage_95",
    "width_95",
    "directional_accuracy",
    "n_origins",
]


def _markdown_table(frame: pd.DataFrame, group_cols: list[str]) -> str:
    """Render one results table as markdown at full precision.

    Args:
        frame: A metrics table from ``evaluation.aggregate``.
        group_cols: Columns identifying the cut, shown before the metrics.

    Returns:
        A markdown table.
    """
    columns = group_cols + [c for c in REPORT_COLUMNS if c in frame.columns]
    ordered = frame[columns].sort_values(
        [c for c in ("series", "horizon", "wql") if c in columns]
    )
    return ordered.to_markdown(index=False, floatfmt=".4f")


def write_markdown_report(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    """Write every results table to one markdown document.

    Args:
        tables: Results tables keyed by name.
        path: Destination file.

    Returns:
        The path written.

    Note:
        Generated from the parquet rather than typed by hand, so a number cannot appear
        here that the pipeline did not produce. ``docs/`` and the Space both read from
        this, which is what makes "no number in the demo that is not in the docs"
        enforceable rather than aspirational.
    """
    sections = [
        ("headline", "1. Headline — Arm A, matched cadence", ["series"]),
        (
            "cadence_comparison",
            "2. Cadence comparison — matched vs native",
            ["series", "cadence"],
        ),
        ("covariate_comparison", "3. Covariates — Arm A vs Arm B", ["series", "arm"]),
        ("regime_stratified", "4. Regime-stratified", ["series", "regime"]),
        ("sample_efficiency", "5. Sample efficiency", ["series", "training_window"]),
        ("contamination_free", "6. Contamination-free subset", ["series"]),
    ]

    lines = [
        "# Benchmark Results",
        "",
        "Generated by `scripts/build_results.py` from the forecast parquets. Every number",
        "here comes from `evaluation/metrics.py`; nothing is typed by hand and nothing is",
        "recomputed anywhere else.",
        "",
        "The **headline** cut is the one registered in `PREREGISTRATION.md` §2: Arm A,",
        "matched cadence, WQL skill score against the random walk. Every other table is",
        "secondary and labelled as such.",
        "",
    ]

    for key, title, group_cols in sections:
        table = tables.get(key)
        lines.append(f"## {title}")
        lines.append("")
        if table is None or table.empty:
            lines += ["*Not available in this run.*", ""]
            continue
        present = [c for c in group_cols if c in table.columns]
        if key == "contamination_free":
            origins = int(table["n_origins"].max()) if "n_origins" in table else 0
            lines += [
                f"**Sample size: {origins} forecast origins.** Far too few for",
                "significance. Stated here rather than in a footnote so no number can be",
                "quoted from this table without it.",
                "",
            ]
        lines += [_markdown_table(table, present), ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path.resolve())
    return path


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

    write_markdown_report(tables, Path("docs/benchmark_results.md"))

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
