"""Arm B covariate sets, restricted to series that FRED does not revise.

Only non-revised daily FRED series are permitted here. Adding a revised series
(``CPIAUCSL``, ``UNRATE``, ``FEDFUNDS``, monthly ``GS10``/``GS3M``) will silently
introduce look-ahead bias, because FRED indexes those by reference period, not release
date. The allowlist is enforced twice — once when the specification is validated here, and
again inside :mod:`forecast_bench.data.fred_client` when the series is fetched — because
this is the study's central claim and one guard is not enough.

Arm A, the headline arm, uses none of this: every model sees only the target's own
history. See DECISIONS.md D3.
"""

import logging

import pandas as pd

from forecast_bench.config import NON_REVISED_FRED_ALLOWLIST
from forecast_bench.data.fred_client import assert_non_revised, fetch_fred_series

logger = logging.getLogger(__name__)

#: FRED series used as covariates for each target, per DECISIONS.md D3.
COVARIATE_SERIES: dict[str, list[str]] = {
    "spy_logrv": ["VIXCLS", "DGS10"],
    "dgs10": ["T10Y2Y", "DGS3MO", "VIXCLS", "DFF"],
}

#: Targets that additionally receive a calendar day-of-week feature.
DAY_OF_WEEK_TARGETS = frozenset({"spy_logrv"})


def covariate_series_for(target: str) -> list[str]:
    """Return the FRED series used as covariates for a target.

    Args:
        target: Target name, ``"spy_logrv"`` or ``"dgs10"``.

    Returns:
        Series identifiers, all of which are on the non-revised allowlist.

    Raises:
        KeyError: If the target is unknown.
        RevisedSeriesError: If the configured set has drifted off the allowlist.
    """
    if target not in COVARIATE_SERIES:
        raise KeyError(
            f"Unknown target {target!r}; expected one of {sorted(COVARIATE_SERIES)}"
        )
    series_ids = COVARIATE_SERIES[target]
    for series_id in series_ids:
        assert_non_revised(series_id)
    return list(series_ids)


def build_covariates(
    target: str, index: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Build the covariate frame for a target.

    Args:
        target: Target name, ``"spy_logrv"`` or ``"dgs10"``.
        index: Trading-day index to align onto. When ``None``, the union of the
            covariates' own indices is used.

    Returns:
        A frame whose columns are the lower-cased series identifiers, plus one-hot
        day-of-week columns for targets in :data:`DAY_OF_WEEK_TARGETS`.

    Note:
        Values are aligned by date and **not** forward-filled here. A covariate gap is a
        real absence, and how a model handles it is the model's decision, made inside the
        fold. Filling at build time would push that decision outside every leakage guard.
    """
    series_ids = covariate_series_for(target)
    columns = {
        series_id.lower(): fetch_fred_series(series_id) for series_id in series_ids
    }
    frame = pd.DataFrame(columns)

    if index is not None:
        frame = frame.reindex(index)

    if target in DAY_OF_WEEK_TARGETS:
        frame = frame.join(day_of_week_features(frame.index))

    validate_covariates(frame)
    return frame


def day_of_week_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build one-hot day-of-week indicators.

    Args:
        index: Dates to encode.

    Returns:
        A frame with columns ``dow_mon`` through ``dow_fri``. Known at time ``t`` by
        construction — a calendar carries no information about the future.
    """
    names = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
    data = {
        name: (index.dayofweek == position).astype(float)
        for position, name in enumerate(names)
    }
    return pd.DataFrame(data, index=index)


def validate_covariates(frame: pd.DataFrame) -> None:
    """Assert that a covariate frame contains only permitted columns.

    Args:
        frame: Covariate frame to check.

    Raises:
        ValueError: If any column is neither an allowlisted FRED series nor a calendar
            feature.
    """
    permitted = {series_id.lower() for series_id in NON_REVISED_FRED_ALLOWLIST}
    permitted |= set(day_of_week_features(pd.DatetimeIndex([])).columns)

    offending = [column for column in frame.columns if column not in permitted]
    if offending:
        raise ValueError(
            f"Covariate frame contains non-allowlisted columns: {offending}. "
            f"Permitted: {sorted(permitted)}. Revised FRED series are indexed by "
            "reference period rather than release date and would introduce look-ahead "
            "bias. See docs/data_protocol.md."
        )
