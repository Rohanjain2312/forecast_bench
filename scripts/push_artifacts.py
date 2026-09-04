"""Publish the project's artifacts to Hugging Face.

Run with::

    poetry run python -m scripts.push_artifacts --target space
    poetry run python -m scripts.push_artifacts --target dataset
    poetry run python -m scripts.push_artifacts --target all

``space/`` lives in the repository rather than only on the Hub so that the demo is
reviewable, diffable and CI-checked like everything else. This mirrors it up; the
repository is the source of truth and the Space is a deployment of it.
"""

import argparse
import logging

from forecast_bench.config import PROJECT_ROOT, get_config, setup_logging

logger = logging.getLogger(__name__)

#: Files mirrored to the Space. Anything else in space/ is not deployed.
SPACE_FILES = ["app.py", "model_cards.py", "requirements.txt", "README.md"]


def push_space(repo_id: str | None = None) -> list[str]:
    """Mirror ``space/`` to the Hugging Face Space.

    Args:
        repo_id: Destination Space. Defaults to the configured one.

    Returns:
        The file names uploaded.

    Raises:
        FileNotFoundError: If any expected file is missing, rather than deploying a
            partial Space that fails to build for a reason the logs will not explain.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_space_repo
    source = PROJECT_ROOT / "space"

    missing = [name for name in SPACE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"space/ is missing {missing}; refusing to deploy.")

    for name in SPACE_FILES:
        api.upload_file(
            path_or_fileobj=str(source / name),
            path_in_repo=name,
            repo_id=destination,
            repo_type="space",
            commit_message=f"forecast_bench: deploy {name}",
        )
        logger.info("Uploaded %s", name)

    print(f"Space updated: https://huggingface.co/spaces/{destination}")
    print("A rebuild starts automatically; the build log is on the Space page.")
    return list(SPACE_FILES)


def push_dataset() -> list[str]:
    """Publish the processed series, forecasts and results tables.

    Returns:
        Paths uploaded, relative to the dataset repo root.
    """
    from forecast_bench.data.hub import (
        push_forecasts,
        push_processed_series,
        push_results,
    )

    uploaded = push_processed_series() + push_forecasts() + push_results()
    print(f"Dataset updated with {len(uploaded)} files.")
    return uploaded


def main() -> int:
    """Push the requested artifacts.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["space", "dataset", "all"],
        default="all",
        help="Which artifacts to publish.",
    )
    args = parser.parse_args()
    setup_logging()

    if args.target in {"dataset", "all"}:
        push_dataset()
    if args.target in {"space", "all"}:
        push_space()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
