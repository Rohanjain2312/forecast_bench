"""Cached OHLC pulls from Yahoo Finance, with yfinance's shape quirks handled in one place.

Two yfinance behaviours would otherwise leak into every caller:

1. A multi-ticker download returns a :class:`~pandas.MultiIndex` column frame, and even a
   single-ticker download does so in some versions.
2. The adjusted-close column is named ``Adj Close`` when ``auto_adjust=False`` and folded
   into ``Close`` when ``auto_adjust=True``, which differs by version and by call.

:func:`normalize_ohlc_columns` is the only place either is handled. Do not bypass this
module by calling ``yfinance`` directly.

Assumes SPY OHLC is split- and dividend-adjusted consistently across the full span; a
mid-series change in adjustment convention would produce a spurious volatility jump in the
Garman-Klass estimator built on top of it.
"""

import logging
from pathlib import Path

import pandas as pd

from forecast_bench.config import TRAIN_START, get_config
from forecast_bench.data._cache import read_cached, write_cache

logger = logging.getLogger(__name__)

#: Canonical column names produced by this module, in order.
OHLC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

#: The four columns the Garman-Klass estimator requires.
REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def normalize_ohlc_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's column shapes into lower-case snake-case names.

    Handles both the MultiIndex frame returned for multi-ticker downloads and the varying
    adjusted-close column name.

    Args:
        frame: Raw frame as returned by ``yfinance.download``.
        ticker: Ticker whose columns to select when the frame is MultiIndexed.

    Returns:
        A frame with a subset of :data:`OHLC_COLUMNS`, in that order.

    Raises:
        KeyError: If any of :data:`REQUIRED_COLUMNS` is missing after normalisation. The
            Garman-Klass estimator needs all four, so a partial frame is a hard error
            rather than something to paper over.
    """
    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):
        # yfinance orders the levels (field, ticker) for downloads and (ticker, field) for
        # some grouped calls. Pick whichever level actually contains the ticker.
        levels_with_ticker = [
            position
            for position in range(result.columns.nlevels)
            if ticker in result.columns.get_level_values(position)
        ]
        if levels_with_ticker:
            result = result.xs(ticker, axis=1, level=levels_with_ticker[-1])
        else:
            # Single-ticker download that still came back MultiIndexed: the ticker level
            # is redundant, so drop every level but the field names.
            result.columns = result.columns.get_level_values(0)

    renamed = {
        str(column): str(column).lower().replace(" ", "_") for column in result.columns
    }
    result = result.rename(columns=renamed)

    if "adj_close" not in result.columns and "close" in result.columns:
        # auto_adjust=True folds the adjustment into `close` and drops `Adj Close`.
        result["adj_close"] = result["close"]

    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise KeyError(
            f"OHLC frame for {ticker!r} is missing {missing}; got "
            f"{sorted(result.columns)}. The Garman-Klass estimator needs open, high, low "
            "and close."
        )

    ordered = [column for column in OHLC_COLUMNS if column in result.columns]
    result = result[ordered]
    result.columns.name = None  # yfinance leaves "Price" on the flattened column index
    result.index = pd.DatetimeIndex(result.index).tz_localize(None)
    result.index.name = "date"
    return result


def fetch_ohlc(
    ticker: str = "SPY",
    *,
    start: str = TRAIN_START,
    end: str | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLC bars for one ticker, using the local cache when possible.

    Args:
        ticker: Yahoo Finance ticker.
        start: First date to request, ``YYYY-MM-DD``.
        end: Last date to request, exclusive. ``None`` means up to today.
        cache_dir: Directory for raw pulls. Defaults to the configured ``data/raw``.
        force_refresh: Bypass the cache and refetch from Yahoo.

    Returns:
        A frame indexed by tz-naive date with columns from :data:`OHLC_COLUMNS`.

    Raises:
        ValueError: If Yahoo returns no rows for the requested span.
    """
    config = get_config()
    directory = cache_dir if cache_dir is not None else config.raw_dir
    stem = f"yfinance_{ticker}_ohlc"

    if not force_refresh:
        cached = read_cached(directory, stem)
        if cached is not None:
            return cached

    logger.info("Fetching %s OHLC from Yahoo Finance (%s to %s)", ticker, start, end)
    raw = _download(ticker, start=start, end=end)
    if raw is None or raw.empty:
        raise ValueError(
            f"Yahoo Finance returned no rows for {ticker!r} between {start} and {end}."
        )

    frame = normalize_ohlc_columns(raw, ticker)
    parquet_path = write_cache(
        frame,
        directory,
        stem,
        source="yfinance",
        params={"ticker": ticker, "start": start, "end": end},
    )
    # Read back so that a cold fetch and a cache hit return byte-identical objects; see
    # the same note in fred_client.fetch_fred_series.
    return pd.read_parquet(parquet_path)


def _download(ticker: str, *, start: str, end: str | None) -> pd.DataFrame:
    """Call ``yfinance.download``.

    Separated from :func:`fetch_ohlc` so that tests can count network calls without
    stubbing the whole caching path.

    Args:
        ticker: Yahoo Finance ticker.
        start: First date to request.
        end: Last date to request, exclusive.

    Returns:
        The raw frame as returned by yfinance, shape quirks intact.
    """
    import yfinance as yf

    # auto_adjust=False keeps `Close` and `Adj Close` distinct. The Garman-Klass estimator
    # is a ratio of same-day prices, so it wants the unadjusted bar; `adj_close` is kept
    # alongside for reference.
    return yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
