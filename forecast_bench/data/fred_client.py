"""Cached FRED pulls, restricted to series that FRED does not revise.

The allowlist in this module is the study's point-in-time protocol expressed as code. It
is not a style preference and not a convenience: it is the reason the claim "every model
saw only data available at the time" holds by construction rather than by careful
bookkeeping.

Only non-revised daily FRED series are permitted here. Adding a revised series
(``CPIAUCSL``, ``UNRATE``, ``FEDFUNDS``, monthly ``GS10``/``GS3M``) will silently
introduce look-ahead bias, because FRED indexes those by reference period, not release
date. March CPI carries a ``2024-03-01`` index but is not published until mid-April, so a
model reading it on ``2024-03-15`` is reading the future. See ``docs/data_protocol.md``.
"""

import logging
from pathlib import Path

import pandas as pd

from forecast_bench.config import NON_REVISED_FRED_ALLOWLIST, get_config
from forecast_bench.data._cache import read_cached, write_cache

logger = logging.getLogger(__name__)

#: Revised series that are commonly reached for, mapped to why they are refused. Used to
#: make the error message specific rather than generic.
_KNOWN_REVISED = {
    "CPIAUCSL": "monthly CPI, indexed by reference month but released weeks later",
    "UNRATE": "monthly unemployment rate, indexed by reference month",
    "FEDFUNDS": "monthly average fed funds rate, indexed by reference month",
    "GS10": "monthly 10-year yield; use the daily DGS10 instead",
    "GS3M": "monthly 3-month yield; use the daily DGS3MO instead",
    "GDP": "quarterly and heavily revised",
    "PAYEMS": "monthly payrolls, revised for two months after first release",
}


class RevisedSeriesError(ValueError):
    """Raised when a series outside the non-revised allowlist is requested.

    Subclasses :class:`ValueError` so that ``pytest.raises(ValueError)`` catches it and
    callers need not import this class to handle it.
    """


def assert_non_revised(series_id: str) -> None:
    """Raise unless ``series_id`` is on the non-revised allowlist.

    Args:
        series_id: FRED series identifier, e.g. ``"DGS10"``.

    Raises:
        RevisedSeriesError: If the series is not on
            :data:`~forecast_bench.config.NON_REVISED_FRED_ALLOWLIST`.
    """
    if series_id in NON_REVISED_FRED_ALLOWLIST:
        return

    reason = _KNOWN_REVISED.get(series_id)
    specific = f" {series_id} is {reason}." if reason else ""
    raise RevisedSeriesError(
        f"{series_id!r} is not on the non-revised FRED allowlist "
        f"{sorted(NON_REVISED_FRED_ALLOWLIST)}.{specific}\n\n"
        "FRED indexes revised series by their reference period, not their release date. "
        "A value stamped 2024-03-01 may not have been published until mid-April, so "
        "reading it at a 2024-03-15 forecast origin reads the future and introduces "
        "look-ahead bias that no metric in this study would reveal.\n\n"
        "If this series is genuinely needed, it belongs in the writeup's discussion, not "
        "in a model input. See docs/data_protocol.md and DECISIONS.md D10-G1."
    )


def fetch_fred_series(
    series_id: str,
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Fetch a non-revised daily FRED series, using the local cache when possible.

    Args:
        series_id: FRED series identifier. Must be on the non-revised allowlist.
        cache_dir: Directory for raw pulls. Defaults to the configured ``data/raw``.
        force_refresh: Bypass the cache and refetch from FRED.

    Returns:
        The series, named ``series_id``, with a :class:`~pandas.DatetimeIndex`. NaNs on
        market holidays are preserved rather than filled — forward-filling a target is a
        subtle leak, and the decision of what to do with a gap belongs to the caller.

    Raises:
        RevisedSeriesError: If ``series_id`` is not on the allowlist.
    """
    assert_non_revised(series_id)

    config = get_config()
    directory = cache_dir if cache_dir is not None else config.raw_dir
    stem = f"fred_{series_id}"

    if not force_refresh:
        cached = read_cached(directory, stem)
        if cached is not None:
            return cached[series_id]

    logger.info("Fetching %s from FRED", series_id)
    frame = _download(series_id).to_frame(name=series_id)
    parquet_path = write_cache(
        frame,
        directory,
        stem,
        source="FRED",
        params={"series_id": series_id},
    )
    # Read back rather than returning the in-memory frame, so a cold fetch and a cache hit
    # return byte-identical objects. A parquet round-trip drops DatetimeIndex.freq, and a
    # caller that behaved differently on the first run than the second would be a
    # miserable bug to find. Dropping freq is right on its own terms too: these series have
    # market holidays, so a freq attribute claiming regular spacing is a false promise.
    return pd.read_parquet(parquet_path)[series_id]


def _download(series_id: str) -> pd.Series:
    """Pull a series from the FRED API.

    Separated from :func:`fetch_fred_series` so that tests can count network calls
    without stubbing the whole caching path.

    Args:
        series_id: FRED series identifier.

    Returns:
        The raw series as returned by ``fredapi``, with a DatetimeIndex.
    """
    from fredapi import Fred

    client = Fred(api_key=get_config().require_secret("fred_api_key"))
    series = client.get_series(series_id)
    series.index = pd.DatetimeIndex(series.index)
    return series
