"""Publish the processed series and results to the Hugging Face dataset repo.

``data/`` is gitignored, so ``forecastbench-data`` is the durable copy of everything the
study is computed from. The Space and the Colab notebooks read from there, which is what
keeps three environments showing the same numbers.
"""

import logging
from pathlib import Path

import pandas as pd

from forecast_bench.config import get_config

logger = logging.getLogger(__name__)

#: Dataset card written alongside the files. The YAML front-matter is what makes the
#: license show on the Hub page rather than only in prose.
DATASET_CARD = """---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - finance
  - realized-volatility
  - treasury-yields
  - forecasting-benchmark
---

# forecastbench-data

Processed inputs and results for [forecast_bench](https://github.com/Rohanjain2312/forecast_bench),
a leakage-safe benchmark of classical statistical models against time-series foundation
models on financial data.

## Files

| File | Contents |
|---|---|
| `processed/spy_logrv.parquet` | SPY log realized variance (Garman-Klass, daily OHLC) plus Arm B covariates |
| `processed/dgs10.parquet` | 10-year Treasury yield in levels plus Arm B covariates |
| `results/*.parquet` | Tidy long-format forecasts and metric tables, when a run has been published |

## Point-in-time guarantee

Only daily, market-observed FRED series that FRED does not revise are used: `DGS10`,
`DGS3MO`, `T10Y2Y`, `VIXCLS`, `DFF`. Revised series such as `CPIAUCSL` are excluded from
every model, because FRED indexes them by reference period rather than release date, so
reading them at a forecast origin reads the future. The claim "every model saw only data
available at the time" therefore holds by construction rather than by careful bookkeeping.

Targets are never forward-filled. `DGS10` carries NaN on market holidays and those rows are
dropped, because forward-filling a target manufactures an observation that never existed.

Full detail: [`docs/data_protocol.md`](https://github.com/Rohanjain2312/forecast_bench/blob/main/docs/data_protocol.md).

## Sources

- SPY OHLC from Yahoo Finance
- All macro and rates series from FRED (St. Louis Fed)

Redistributed here as derived research artifacts. Consult the original providers for their
terms.

## License

MIT, matching the source repository.
"""


def push_processed_series(
    series: list[str] | None = None,
    repo_id: str | None = None,
    private: bool = False,
) -> list[str]:
    """Upload the processed parquet files and the dataset card.

    Args:
        series: Series names to publish. Defaults to both targets.
        repo_id: Destination dataset repo. Defaults to the configured one.
        private: Whether to create the repo private if it does not exist.

    Returns:
        Paths uploaded, relative to the repo root.

    Raises:
        FileNotFoundError: If a requested series has not been built yet.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_dataset_repo
    names = series or ["spy_logrv", "dgs10"]

    api.create_repo(destination, repo_type="dataset", private=private, exist_ok=True)

    uploaded: list[str] = []
    for name in names:
        path = config.processed_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Build it with:\n"
                f"    poetry run python -m scripts.fetch_data --config {name}"
            )
        target = f"processed/{name}.parquet"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=destination,
            repo_type="dataset",
        )
        uploaded.append(target)
        logger.info("Uploaded %s to %s", target, destination)

    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=destination,
        repo_type="dataset",
    )
    uploaded.append("README.md")
    return uploaded


def push_results(
    directory: Path | None = None, repo_id: str | None = None
) -> list[str]:
    """Upload the metric tables produced by ``scripts.build_results``.

    Args:
        directory: Directory holding the tables. Defaults to the configured metrics dir.
        repo_id: Destination dataset repo. Defaults to the configured one.

    Returns:
        Paths uploaded, relative to the repo root.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_dataset_repo
    source = directory or config.metrics_dir

    uploaded: list[str] = []
    for path in sorted(source.glob("*.parquet")):
        target = f"results/{path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=destination,
            repo_type="dataset",
        )
        uploaded.append(target)
    logger.info("Uploaded %d result tables to %s", len(uploaded), destination)
    return uploaded


def push_forecasts(
    directory: Path | None = None, repo_id: str | None = None
) -> list[str]:
    """Upload the tidy forecast parquets written by the backtest runner.

    Args:
        directory: Directory holding the parquets. Defaults to the configured forecasts dir.
        repo_id: Destination dataset repo. Defaults to the configured one.

    Returns:
        Paths uploaded, relative to the repo root.

    Note:
        Used by the Colab notebooks: the GPU work happens there, but the forecasts have to
        come back so the local machine can score them through the same
        ``evaluation/aggregate.py`` as everything else.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_dataset_repo
    source = directory or config.forecasts_dir

    uploaded: list[str] = []
    # rglob, not glob: the sample-efficiency sweep lives in a subdirectory so that
    # load_forecasts() cannot double-count it, but it still has to be published.
    for path in sorted(source.rglob("*.parquet")):
        target = f"forecasts/{path.relative_to(source).as_posix()}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=destination,
            repo_type="dataset",
        )
        uploaded.append(target)
    logger.info("Uploaded %d forecast files to %s", len(uploaded), destination)
    return uploaded


def load_processed(name: str, repo_id: str | None = None) -> pd.DataFrame:
    """Read a processed series from the Hub.

    Used by Colab and the Space, which have no local ``data/`` directory.

    Args:
        name: Series name.
        repo_id: Source dataset repo. Defaults to the configured one.

    Returns:
        The processed frame.
    """
    from huggingface_hub import hf_hub_download

    config = get_config()
    path = hf_hub_download(
        repo_id=repo_id or config.hf_dataset_repo,
        filename=f"processed/{name}.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)
