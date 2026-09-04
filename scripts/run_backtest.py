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
from forecast_bench.backtest.writer import merge_forecasts, write_results
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


def _merge_published_models(
    results: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Add models from the published forecasts that this run did not compute.

    Args:
        results: Locally computed forecasts.
        args: Parsed command-line arguments, for the file name to look up.

    Returns:
        The merged frame, or the local one unchanged if nothing could be pulled.

    Note:
        The neural models are trained on a GPU in Colab and published from there. Rather
        than retraining them locally — which is not viable on CPU — their forecasts are
        merged back in by model id, so the headline panel is complete while every model
        is still scored by identical code downstream.
    """
    from forecast_bench.data.hub import load_forecast_file

    cadence_label = results["cadence"].iloc[0]
    filename = f"{args.config}_arm{args.arm}_{cadence_label}.parquet"
    try:
        published = load_forecast_file(filename)
    except Exception as error:  # noqa: BLE001 - merging is best-effort by design
        logger.warning(
            "Could not pull %s from the Hub (%s); continuing with local models only.",
            filename,
            type(error).__name__,
        )
        return results

    local_models = set(results["model_id"].unique())
    extra = published[~published["model_id"].isin(local_models)]
    if extra.empty:
        logger.info("Published forecasts add no models beyond the local panel.")
        return results

    logger.info(
        "Merging %s from the published forecasts", sorted(extra["model_id"].unique())
    )
    return merge_forecasts(results, extra)


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
    parser.add_argument(
        "--with-finetuned",
        action="store_true",
        help="Add the LoRA-adapted foundation models, loaded from the Hub by revision.",
    )
    parser.add_argument(
        "--sample-efficiency-window",
        choices=["1y", "3y", "10y", "full"],
        help=(
            "Run one point of the D9 sample-efficiency sweep. Selects the matching "
            "fine-tuned adapter, truncates the neural training window to the same size, "
            "labels the output, and writes it under forecasts/sample_efficiency/."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Restrict the panel to these model ids.",
    )
    parser.add_argument(
        "--merge-hub",
        action="store_true",
        help=(
            "Merge in models present in the published forecasts but not computed here. "
            "This is how the GPU-trained neural models join a locally-run panel."
        ),
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

    window = args.sample_efficiency_window
    model_kwargs: dict[str, object] = {}
    if window:
        from forecast_bench.models.base import sample_efficiency_window_size

        # Two different knobs for the same idea: the fine-tuned models select an adapter
        # trained on that slice, the neural models truncate their own training window.
        model_kwargs["training_window"] = window
        model_kwargs["training_window_days"] = sample_efficiency_window_size(window)

    results = run_series_backtest(
        args.config,
        frame=load_series(args.config),
        cadence=args.cadence,
        arm=args.arm,
        include_foundation=args.with_foundation,
        include_finetuned=args.with_finetuned,
        only_models=args.models,
        **model_kwargs,
    )

    if args.merge_hub:
        results = _merge_published_models(results, args)

    if window:
        results["training_window"] = window
        destination = config.sample_efficiency_dir
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{args.config}_arm{args.arm}_{window}.parquet"
        results.to_parquet(path, index=False)
        logger.info("Wrote %d sweep rows to %s", len(results), path.resolve())
    else:
        path = write_results(results, config.forecasts_dir)

    print(f"Wrote {len(results)} rows to {path}")
    print(f"  models  : {sorted(results['model_id'].unique())}")
    print(f"  origins : {results['origin'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
