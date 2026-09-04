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


def assert_github_is_current() -> None:
    """Refuse to deploy while local commits are unpushed.

    Raises:
        RuntimeError: If the working tree is dirty or HEAD is ahead of ``origin/main``.

    Note:
        The Space installs ``forecast_bench`` from GitHub, so deploying before pushing
        builds the Space against *older* code than the one just written. That is exactly
        how the first deploy failed: ``viz/`` existed locally, the Space installed a commit
        without it, and the app died on import with a ModuleNotFoundError that looks like a
        packaging bug rather than an ordering mistake.
    """
    import subprocess

    def git(*args: str) -> str:
        """Run a git command and return its trimmed stdout.

        Args:
            *args: Arguments passed to ``git``.

        Returns:
            Standard output with surrounding whitespace removed.
        """
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("status", "--porcelain"):
        raise RuntimeError(
            "Working tree has uncommitted changes. The Space installs from GitHub, so "
            "commit and push before deploying or it will build against older code."
        )

    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], check=False)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise RuntimeError(
            "Local HEAD differs from origin/main. Push first: the Space installs the "
            "package from GitHub and would deploy against the older commit."
        )
    logger.info("GitHub is current; safe to deploy.")


def push_space(repo_id: str | None = None, check_github: bool = True) -> list[str]:
    """Mirror ``space/`` to the Hugging Face Space.

    Args:
        repo_id: Destination Space. Defaults to the configured one.
        check_github: Verify GitHub is current before deploying.

    Returns:
        The file names uploaded.

    Raises:
        FileNotFoundError: If any expected file is missing, rather than deploying a
            partial Space that fails to build for a reason the logs will not explain.
    """
    from huggingface_hub import HfApi

    if check_github:
        assert_github_is_current()

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
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full Space rebuild so it reinstalls the package from GitHub.",
    )
    parser.add_argument(
        "--skip-github-check",
        action="store_true",
        help="Deploy even if GitHub is behind. Almost always the wrong thing to do.",
    )
    args = parser.parse_args()
    setup_logging()

    if args.target in {"dataset", "all"}:
        push_dataset()
    if args.target in {"space", "all"}:
        push_space(check_github=not args.skip_github_check)
        if args.rebuild:
            from huggingface_hub import HfApi

            config = get_config()
            HfApi(token=config.require_secret("hf_token")).restart_space(
                config.hf_space_repo, factory_reboot=True
            )
            print("Factory rebuild triggered: the Space will reinstall from GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
