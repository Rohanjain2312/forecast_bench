"""Single source of truth for every setting in the study.

Nothing downstream of this module reads ``os.environ`` directly. Secrets come from a
gitignored ``.env`` via :mod:`pydantic_settings`; study constants are hardcoded here
because they are decisions rather than configuration, and are cross-checked against
``experiments/configs/base.yaml`` at import time.

Typical use::

    from forecast_bench.config import HORIZONS, get_config, setup_logging

    setup_logging()
    cfg = get_config()
    frame = pd.read_parquet(cfg.processed_dir / "spy_logrv.parquet")
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Study constants
#
# Hardcoded on purpose. These are pre-registered decisions (PREREGISTRATION.md §2 and §5),
# not knobs — making them environment-driven would mean a stray shell variable could
# silently change what the benchmark measures.
# --------------------------------------------------------------------------------------

#: Quantile levels every model must emit (DECISIONS.md D4).
QUANTILE_GRID: Final[list[float]] = [
    0.025,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.975,
]

#: Reported horizons in trading days, read off one 21-step path (DECISIONS.md D2).
HORIZONS: Final[list[int]] = [1, 5, 21]

#: Length of the single forecast path produced per fold.
MAX_HORIZON: Final[int] = 21

#: Fold stride. ``STRIDE == MAX_HORIZON`` makes forecast windows non-overlapping, which is
#: what makes the Diebold-Mariano test defensible. Changing it invalidates that test.
STRIDE: Final[int] = 21

#: Context fed to foundation models, and the input chunk for the neural models, so that
#: context length is not a confound across model classes.
CONTEXT_LENGTH: Final[int] = 512

TRAIN_START: Final[str] = "2000-01-01"
TEST_START: Final[str] = "2015-01-01"
TEST_END: Final[str] = "2026-06-30"

RANDOM_SEED: Final[int] = 42

#: Daily, market-observed FRED series that FRED does not revise.
#:
#: Nothing outside this set may enter a model. Revised series (``CPIAUCSL``, ``UNRATE``,
#: ``FEDFUNDS``, monthly ``GS10``/``GS3M``) are indexed by reference period rather than
#: release date, so reading them at time ``t`` reads data published weeks later. This is
#: the study's central point-in-time claim, enforced in ``data/fred_client.py``.
NON_REVISED_FRED_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"DGS10", "DGS3MO", "T10Y2Y", "VIXCLS", "DFF"}
)

#: Repository root, resolved from this file's location.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The YAML mirror of the constants above, cross-checked at import time.
BASE_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "experiments" / "configs" / "base.yaml"


class ConfigMismatchError(RuntimeError):
    """Raised when ``base.yaml`` and the Python study constants disagree.

    This is a hard failure rather than a warning. A silent divergence between the two
    would produce two different studies sharing one name, and no metric would reveal it.
    """


def _verify_base_yaml_agrees(path: Path = BASE_CONFIG_PATH) -> None:
    """Assert that ``experiments/configs/base.yaml`` matches the constants in this module.

    Args:
        path: Location of the YAML mirror. Defaults to :data:`BASE_CONFIG_PATH`.

    Raises:
        ConfigMismatchError: If any value in the YAML differs from its Python counterpart.

    Note:
        A *missing* file is not an error. When the package is pip-installed from GitHub
        (which is how Colab and the Hugging Face Space consume it) only ``forecast_bench/``
        ships, so ``experiments/`` is absent. Absence is logged and skipped; disagreement
        is fatal.
    """
    if not path.is_file():
        logger.debug(
            "base.yaml not found at %s; skipping the constants cross-check. This is "
            "expected when the package is installed rather than used from a clone.",
            path,
        )
        return

    with path.open("r", encoding="utf-8") as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)

    expected: dict[str, Any] = {
        "quantile_grid": QUANTILE_GRID,
        "horizons": HORIZONS,
        "max_horizon": MAX_HORIZON,
        "stride": STRIDE,
        "context_length": CONTEXT_LENGTH,
        "train_start": TRAIN_START,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "random_seed": RANDOM_SEED,
        # Compared as a set: the YAML is a list for readability, but order is meaningless.
        "non_revised_fred_allowlist": NON_REVISED_FRED_ALLOWLIST,
    }

    mismatches: list[str] = []
    for key, python_value in expected.items():
        if key not in loaded:
            mismatches.append(f"{key}: missing from {path.name}")
            continue
        yaml_value = loaded[key]
        if isinstance(python_value, frozenset):
            yaml_value = frozenset(yaml_value)
        if yaml_value != python_value:
            mismatches.append(f"{key}: yaml={yaml_value!r} != python={python_value!r}")

    if mismatches:
        raise ConfigMismatchError(
            f"{path} disagrees with forecast_bench.config:\n  "
            + "\n  ".join(mismatches)
            + "\n\nFix one of the two so they agree. They are duplicated deliberately, "
            "and a divergence means two different studies are running under one name."
        )


class Config(BaseSettings):
    """Runtime settings: secrets, repository identifiers, and filesystem paths.

    Study constants are *not* here — they are module-level constants above, because they
    must not be overridable from the environment.

    Secrets are :class:`~pydantic.SecretStr`, so they are masked in ``repr()``, in
    ``model_dump()``, and in any traceback that renders the settings object. Read the
    underlying value explicitly with :meth:`require_secret`, which also fails loudly when
    the value is absent instead of sending an empty string to an API.

    Attributes:
        fred_api_key: FRED API key. Required by ``data/fred_client.py``.
        hf_token: Hugging Face write token. Required to push checkpoints, the dataset,
            and the Space.
        wandb_api_key: Weights & Biases key. Optional; tracking is skipped without it.
        wandb_project: W&B project name for fine-tuning and neural training runs.
        hf_model_repo: Hub repo for the fine-tuned Chronos checkpoints.
        hf_dataset_repo: Hub repo for the processed series and results tables.
        hf_space_repo: Hub repo for the Gradio demo.
        data_dir: Root of the local data cache. Gitignored.
        results_dir: Root of the backtest outputs.
        log_level: Level applied by :func:`setup_logging`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    fred_api_key: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")
    wandb_api_key: SecretStr = SecretStr("")
    wandb_project: str = "forecast-bench"

    hf_model_repo: str = "rohanjain2312/forecastbench-chronos"
    hf_dataset_repo: str = "rohanjain2312/forecastbench-data"
    hf_space_repo: str = "rohanjain2312/forecastbench-demo"

    data_dir: Path = Path("./data")
    results_dir: Path = Path("./experiments/results")

    log_level: str = "INFO"

    # --- Path helpers -----------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Cache of unmodified source pulls, one parquet plus a ``.meta.json`` per series."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Model-ready series: targets joined to covariates, business-day indexed."""
        return self.data_dir / "processed"

    @property
    def forecasts_dir(self) -> Path:
        """Tidy long-format forecast parquets written by ``backtest/writer.py``."""
        return self.results_dir / "forecasts"

    @property
    def sample_efficiency_dir(self) -> Path:
        """Sweep forecasts, kept out of ``forecasts/`` itself.

        The sweep's ``full`` slice is the same computation as the headline run, so its rows
        duplicate that run exactly. ``load_forecasts()`` globs ``forecasts/*.parquet``
        non-recursively, so keeping the sweep one level down means the headline table
        cannot silently double-count it.
        """
        return self.forecasts_dir / "sample_efficiency"

    @property
    def metrics_dir(self) -> Path:
        """Aggregated results tables produced by ``evaluation/aggregate.py``."""
        return self.results_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        """Figures shared by the docs, the Medium writeup, and the Space."""
        return self.results_dir / "figures"

    def ensure_dirs(self) -> None:
        """Create every output directory this project writes to, if it does not exist."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.forecasts_dir,
            self.sample_efficiency_dir,
            self.metrics_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        # DATA_DIR and RESULTS_DIR are relative by default, so where output actually
        # lands depends on the working directory. Log it once, resolved, so a later
        # "the file is not there" is a one-line diagnosis rather than a hunt.
        logger.info(
            "Output directories ready under %s and %s",
            self.data_dir.resolve(),
            self.results_dir.resolve(),
        )

    # --- Secret access ----------------------------------------------------------------

    def require_secret(self, name: str) -> str:
        """Return a secret's plaintext value, raising if it was never set.

        Args:
            name: Field name, e.g. ``"fred_api_key"``.

        Returns:
            The secret's plaintext value.

        Raises:
            ValueError: If the field is unset or empty. Failing here is deliberate — an
                empty string sent to an API produces a confusing 401 far from the cause.
            AttributeError: If ``name`` is not a field on this object.
        """
        value: SecretStr = getattr(self, name)
        plaintext = value.get_secret_value()
        if not plaintext:
            raise ValueError(
                f"{name.upper()} is not set. Copy .env.example to .env and fill it in "
                "(see docs/planning/MANUAL_TASKS.md step 4)."
            )
        return plaintext

    @property
    def has_wandb(self) -> bool:
        """Whether a Weights & Biases key is present. Tracking is skipped when it is not."""
        return bool(self.wandb_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide :class:`Config` singleton.

    Cached so that ``.env`` is read once and every caller observes identical settings.

    Returns:
        The singleton configuration object.
    """
    return Config()


def enable_tensor_cores(precision: str = "high") -> str:
    """Let matmuls use the GPU's Tensor Cores, trading a little precision for speed.

    Args:
        precision: ``"high"`` (TF32, ~10-bit mantissa), ``"medium"`` (bfloat16, fastest),
            or ``"highest"`` (full fp32, PyTorch's default).

    Returns:
        The precision actually in effect. ``"highest"`` when no CUDA device is present,
        since the setting only affects Tensor Core hardware.

    Raises:
        ValueError: If ``precision`` is not one of the three accepted values.

    Note:
        Off by default, and deliberately so. This changes numerics, so it must be applied
        **uniformly across everything being compared** or the comparison is not one. The
        sample-efficiency sweep in particular retrains the same models on nested windows
        and expects its ``full`` point to reproduce the headline run; mixing precisions
        between them would quietly break that.

        ``"high"`` keeps a 10-bit mantissa, which is far below the noise floor of a
        financial forecasting task and is the standard recommendation for neural training.
        It does not affect the classical models, which run on CPU, or the Chronos
        fine-tuning in notebook 04, which was run without it.
    """
    permitted = {"highest", "high", "medium"}
    if precision not in permitted:
        raise ValueError(
            f"precision must be one of {sorted(permitted)}, got {precision!r}"
        )

    import torch

    if not torch.cuda.is_available():
        logger.info(
            "No CUDA device; leaving matmul precision at 'highest'. This setting only "
            "affects Tensor Core hardware."
        )
        return "highest"

    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = precision != "highest"
    torch.backends.cudnn.allow_tf32 = precision != "highest"

    logger.info(
        "Matmul precision set to %r on %s. Every model trained in this process now uses "
        "it; anything compared against these results must too.",
        precision,
        torch.cuda.get_device_name(0),
    )
    return precision


def setup_logging(level: str | None = None) -> None:
    """Configure root logging for scripts and notebooks.

    Library modules only ever call ``logging.getLogger(__name__)``; this function is the
    one place that decides where those records go, and it is called by entry points
    rather than at import time.

    Args:
        level: Level name such as ``"DEBUG"``. Defaults to the configured ``log_level``.
    """
    resolved = level or get_config().log_level
    logging.basicConfig(
        level=getattr(logging, resolved.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# Fail at import rather than mid-run if the YAML mirror has drifted from the constants.
_verify_base_yaml_agrees()
