"""Volatility regime assignment from frozen VIX tercile thresholds.

The thresholds are loaded from ``experiments/configs/regimes.yaml`` and asserted against
the committed values **at import time**. They were computed once, on 2000-2014 only, and
must never be recomputed — see ``DECISIONS.md`` D8 and ``PREREGISTRATION.md`` §5.

Recomputing them over the full sample would define "stressed" using the knowledge that
2020 and 2022 happened. That is leakage in the reporting layer rather than the modelling
layer, which makes it easier to miss and no less invalidating.
"""

import logging
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from forecast_bench.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Location of the frozen thresholds.
REGIMES_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "experiments" / "configs" / "regimes.yaml"
)

#: The committed values, duplicated here so a change to the YAML fails loudly at import
#: rather than silently restratifying every result table.
EXPECTED_CALM_UPPER: Final[float] = 15.9
EXPECTED_NORMAL_UPPER: Final[float] = 22.5582

#: Regime labels, in increasing order of volatility.
REGIME_LABELS: Final[list[str]] = ["calm", "normal", "stressed"]


class FrozenThresholdError(RuntimeError):
    """Raised when the loaded regime thresholds differ from the committed values."""


def _load_thresholds(path: Path = REGIMES_CONFIG_PATH) -> tuple[float, float]:
    """Load and verify the frozen thresholds.

    Args:
        path: Location of ``regimes.yaml``.

    Returns:
        A ``(calm_upper, normal_upper)`` pair.

    Raises:
        FrozenThresholdError: If the file is malformed, or its values differ from the
            committed constants.

    Note:
        A *missing* file is not an error, for the same reason as ``base.yaml`` in
        :mod:`forecast_bench.config`: when the package is pip-installed from GitHub — how
        Colab and the Hugging Face Space consume it — only ``forecast_bench/`` ships, so
        ``experiments/`` does not exist at all.

        This is safe because :data:`EXPECTED_CALM_UPPER` and :data:`EXPECTED_NORMAL_UPPER`
        **are** the frozen values, duplicated into this module deliberately. The YAML is a
        cross-check that makes any change to them show up in a diff, not the source of
        truth. Absence falls back to the constants; disagreement stays fatal.
    """
    if not path.is_file():
        logger.debug(
            "regimes.yaml not found at %s; falling back to the committed constants "
            "(%.4f, %.4f). Expected when the package is installed rather than used from "
            "a clone.",
            path,
            EXPECTED_CALM_UPPER,
            EXPECTED_NORMAL_UPPER,
        )
        return EXPECTED_CALM_UPPER, EXPECTED_NORMAL_UPPER

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        calm_upper = float(loaded["thresholds"]["calm_upper"])
        normal_upper = float(loaded["thresholds"]["normal_upper"])
    except (KeyError, TypeError, ValueError) as error:
        raise FrozenThresholdError(f"{path} is malformed: {error}") from error

    if (calm_upper, normal_upper) != (EXPECTED_CALM_UPPER, EXPECTED_NORMAL_UPPER):
        raise FrozenThresholdError(
            f"{path} holds ({calm_upper}, {normal_upper}) but the committed values are "
            f"({EXPECTED_CALM_UPPER}, {EXPECTED_NORMAL_UPPER}). These thresholds were "
            "computed once on pre-2015 data and PREREGISTRATION.md section 5 commits to "
            "not recomputing them. If this change is deliberate, it needs an amendment "
            "entry, not an edit."
        )
    return calm_upper, normal_upper


#: Upper bound of the calm regime, verified against the committed value at import.
CALM_UPPER: Final[float]
#: Upper bound of the normal regime, verified against the committed value at import.
NORMAL_UPPER: Final[float]
CALM_UPPER, NORMAL_UPPER = _load_thresholds()


def assign_regime(vix_level: float) -> str | None:
    """Label one VIX level.

    Args:
        vix_level: VIX close at the forecast origin.

    Returns:
        ``"calm"``, ``"normal"``, ``"stressed"``, or ``None`` if the level is missing.
    """
    if vix_level is None or pd.isna(vix_level):
        return None
    if vix_level <= CALM_UPPER:
        return "calm"
    if vix_level <= NORMAL_UPPER:
        return "normal"
    return "stressed"


def regime_series(vix: pd.Series) -> pd.Series:
    """Label a whole VIX series.

    Args:
        vix: VIX closes indexed by date.

    Returns:
        Regime labels on the same index.

    Note:
        Assignment uses the VIX level **at the forecast origin**, which is known at time
        ``t``, so stratification introduces no look-ahead. Using VIX for stratification is
        reporting, not modelling, so it does not contaminate Arm A even though VIX is also
        an Arm B covariate.
    """
    return vix.map(assign_regime).rename("regime")
