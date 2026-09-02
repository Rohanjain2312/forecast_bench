"""Tests for the data clients: caching, the non-revised allowlist, and yfinance shapes.

None of these tests touch the network. The download functions are monkeypatched, which is
also how the cache-hit test counts calls.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import NON_REVISED_FRED_ALLOWLIST
from forecast_bench.data import fred_client, yahoo_client
from forecast_bench.data._cache import read_meta


@pytest.fixture
def fake_fred_series() -> pd.Series:
    """A deterministic stand-in for a FRED daily series."""
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    return pd.Series(np.linspace(1.0, 2.0, len(index)), index=index)


@pytest.fixture
def fake_ohlc_multiindex() -> pd.DataFrame:
    """A synthetic yfinance frame with the MultiIndex column shape it returns."""
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["SPY"]])
    values = np.arange(len(index) * len(fields), dtype=float).reshape(
        len(index), len(fields)
    )
    return pd.DataFrame(values, index=index, columns=columns)


# --- Allowlist -------------------------------------------------------------------------


def test_allowlist_rejects_cpiaucsl() -> None:
    """CPIAUCSL is refused, and the error explains the release-lag reason."""
    with pytest.raises(ValueError) as excinfo:
        fred_client.assert_non_revised("CPIAUCSL")

    message = str(excinfo.value)
    assert "CPIAUCSL" in message
    assert "reference period" in message
    assert "release date" in message
    assert "look-ahead" in message


@pytest.mark.parametrize("series_id", ["UNRATE", "FEDFUNDS", "GS10", "GS3M", "GDP"])
def test_allowlist_rejects_other_revised_series(series_id: str) -> None:
    """Every revised series a modeller might reach for is refused."""
    with pytest.raises(ValueError):
        fred_client.assert_non_revised(series_id)


@pytest.mark.parametrize("series_id", sorted(NON_REVISED_FRED_ALLOWLIST))
def test_allowlist_accepts_every_permitted_series(series_id: str) -> None:
    """Every series on the allowlist passes."""
    fred_client.assert_non_revised(series_id)


def test_fetch_refuses_revised_series_before_touching_the_network(monkeypatch) -> None:
    """The allowlist check happens before any download is attempted."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("network call attempted for a revised series")

    monkeypatch.setattr(fred_client, "_download", explode)
    with pytest.raises(ValueError):
        fred_client.fetch_fred_series("CPIAUCSL")


# --- Caching ---------------------------------------------------------------------------


def test_fred_cache_hit_avoids_a_second_network_call(
    tmp_path, monkeypatch, fake_fred_series
) -> None:
    """A second fetch of the same series is served from disk."""
    calls = {"count": 0}

    def counting_download(series_id: str) -> pd.Series:
        calls["count"] += 1
        return fake_fred_series

    monkeypatch.setattr(fred_client, "_download", counting_download)

    first = fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)
    second = fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)

    assert calls["count"] == 1
    pd.testing.assert_series_equal(first, second)
    assert (tmp_path / "fred_DGS10.parquet").is_file()


def test_force_refresh_bypasses_the_cache(
    tmp_path, monkeypatch, fake_fred_series
) -> None:
    """force_refresh=True refetches even when a cache entry exists."""
    calls = {"count": 0}

    def counting_download(series_id: str) -> pd.Series:
        calls["count"] += 1
        return fake_fred_series

    monkeypatch.setattr(fred_client, "_download", counting_download)

    fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)
    fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path, force_refresh=True)

    assert calls["count"] == 2


def test_cache_writes_a_provenance_sidecar(
    tmp_path, monkeypatch, fake_fred_series
) -> None:
    """The .meta.json sidecar records fetch timestamp, source, and checksum."""
    monkeypatch.setattr(fred_client, "_download", lambda series_id: fake_fred_series)
    fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)

    meta = read_meta(tmp_path, "fred_DGS10")
    assert meta is not None
    assert meta["source"] == "FRED"
    assert meta["params"]["series_id"] == "DGS10"
    assert meta["rows"] == len(fake_fred_series)
    assert len(meta["sha256"]) == 64
    assert meta["fetched_at"].endswith("+00:00")


def test_parquet_without_a_sidecar_is_not_trusted(
    tmp_path, monkeypatch, fake_fred_series
) -> None:
    """A cache entry missing its sidecar is refetched, not silently used."""
    calls = {"count": 0}

    def counting_download(series_id: str) -> pd.Series:
        calls["count"] += 1
        return fake_fred_series

    monkeypatch.setattr(fred_client, "_download", counting_download)

    fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)
    (tmp_path / "fred_DGS10.meta.json").unlink()
    fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)

    assert calls["count"] == 2


def test_fred_preserves_holiday_nans(tmp_path, monkeypatch) -> None:
    """NaNs are returned as NaNs. Forward-filling a target is a subtle leak."""
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], index=index)
    monkeypatch.setattr(fred_client, "_download", lambda series_id: series)

    result = fred_client.fetch_fred_series("DGS10", cache_dir=tmp_path)

    assert result.isna().sum() == 2


# --- yfinance shape repair -------------------------------------------------------------


def test_multiindex_repair_on_a_synthetic_frame(fake_ohlc_multiindex) -> None:
    """A MultiIndex column frame is flattened to canonical snake-case names."""
    result = yahoo_client.normalize_ohlc_columns(fake_ohlc_multiindex, "SPY")

    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert not isinstance(result.columns, pd.MultiIndex)
    assert len(result) == len(fake_ohlc_multiindex)


def test_multiindex_repair_when_ticker_is_the_outer_level(fake_ohlc_multiindex) -> None:
    """The ticker level is found whichever side of the MultiIndex it sits on."""
    swapped = fake_ohlc_multiindex.copy()
    swapped.columns = swapped.columns.swaplevel(0, 1)
    swapped = swapped.sort_index(axis=1)

    result = yahoo_client.normalize_ohlc_columns(swapped, "SPY")

    assert set(yahoo_client.REQUIRED_COLUMNS).issubset(result.columns)


def test_flat_columns_pass_through() -> None:
    """A plain single-level frame is normalised without special handling."""
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Adj Close": [1.5, 2.5, 3.5],
            "Volume": [100, 200, 300],
        },
        index=index,
    )

    result = yahoo_client.normalize_ohlc_columns(frame, "SPY")

    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]


def test_missing_adj_close_is_filled_from_close() -> None:
    """auto_adjust=True drops Adj Close; close stands in for it."""
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
        },
        index=index,
    )

    result = yahoo_client.normalize_ohlc_columns(frame, "SPY")

    pd.testing.assert_series_equal(
        result["adj_close"], result["close"], check_names=False
    )


def test_missing_required_column_is_a_hard_error() -> None:
    """A frame without a Low column cannot produce a Garman-Klass estimate."""
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [2.0, 3.0, 4.0], "Close": [1.5, 2.5, 3.5]},
        index=index,
    )

    with pytest.raises(KeyError, match="low"):
        yahoo_client.normalize_ohlc_columns(frame, "SPY")


def test_yahoo_cache_hit_avoids_a_second_network_call(
    tmp_path, monkeypatch, fake_ohlc_multiindex
) -> None:
    """A second OHLC fetch is served from disk."""
    calls = {"count": 0}

    def counting_download(ticker: str, *, start: str, end: str | None) -> pd.DataFrame:
        calls["count"] += 1
        return fake_ohlc_multiindex

    monkeypatch.setattr(yahoo_client, "_download", counting_download)

    first = yahoo_client.fetch_ohlc("SPY", cache_dir=tmp_path)
    second = yahoo_client.fetch_ohlc("SPY", cache_dir=tmp_path)

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_empty_yahoo_response_raises(tmp_path, monkeypatch) -> None:
    """An empty frame is an error, not an empty result silently passed downstream."""
    monkeypatch.setattr(
        yahoo_client,
        "_download",
        lambda ticker, *, start, end: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="no rows"):
        yahoo_client.fetch_ohlc("SPY", cache_dir=tmp_path)
