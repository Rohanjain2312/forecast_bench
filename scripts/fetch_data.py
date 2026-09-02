"""Build a processed modelling series and write it to ``data/processed/``.

Run with::

    poetry run python -m scripts.fetch_data --config spy_logrv
    poetry run python -m scripts.fetch_data --config dgs10

Raw pulls are cached under ``data/raw/`` with provenance sidecars, so re-running is cheap
and offline once the cache is warm. Pass ``--force-refresh`` to refetch from source.
"""

import argparse
import logging

import pandas as pd

from forecast_bench.config import get_config, setup_logging
from forecast_bench.data.covariates import build_covariates
from forecast_bench.data.merge import align_to_target
from forecast_bench.data.targets import build_dgs10, build_spy_logrv

logger = logging.getLogger(__name__)

#: Target builders, keyed by the ``--config`` value.
TARGET_BUILDERS = {
    "spy_logrv": build_spy_logrv,
    "dgs10": build_dgs10,
}


def build(target: str, with_covariates: bool = True) -> pd.DataFrame:
    """Build one processed series with its covariates aligned onto the target's index.

    Args:
        target: ``"spy_logrv"`` or ``"dgs10"``.
        with_covariates: Whether to attach the Arm B covariate columns. Arm A ignores
            them, but they are stored once so both arms read the same file.

    Returns:
        The merged frame, target column first.

    Raises:
        KeyError: If ``target`` is not a known configuration.
    """
    if target not in TARGET_BUILDERS:
        raise KeyError(f"Unknown config {target!r}; expected {sorted(TARGET_BUILDERS)}")

    series = TARGET_BUILDERS[target]()
    covariates = (
        build_covariates(target, index=series.index) if with_covariates else None
    )
    return align_to_target(series, covariates)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(TARGET_BUILDERS),
        help="Which target series to build.",
    )
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Build the target alone, without the Arm B covariate columns.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the requested series and write it to ``data/processed/``.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging()
    config = get_config()
    config.ensure_dirs()

    frame = build(args.config, with_covariates=not args.no_covariates)
    destination = config.processed_dir / f"{args.config}.parquet"
    frame.to_parquet(destination)

    print(f"Wrote {len(frame)} rows x {len(frame.columns)} cols to {destination}")
    print(f"  span    : {frame.index.min().date()} -> {frame.index.max().date()}")
    print(f"  columns : {list(frame.columns)}")
    print(frame.describe().T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
