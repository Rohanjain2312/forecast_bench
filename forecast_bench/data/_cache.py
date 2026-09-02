"""Parquet cache with provenance sidecars, shared by the data clients.

Every raw pull is written as ``<stem>.parquet`` alongside ``<stem>.meta.json`` recording
when it was fetched, what it was fetched from, and a checksum of the parquet bytes. The
sidecar is what makes a cached file auditable months later: without it, a stale parquet is
indistinguishable from a fresh one.

This module is private to :mod:`forecast_bench.data`. It exists so that the FRED and Yahoo
clients cannot drift into two different caching behaviours — the same reason
``evaluation/metrics.py`` is the only place a metric is defined.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Sidecar schema version, bumped if the metadata layout ever changes.
META_VERSION = 1


def cache_paths(cache_dir: Path, stem: str) -> tuple[Path, Path]:
    """Return the parquet and sidecar paths for a cache entry.

    Args:
        cache_dir: Directory holding raw pulls, normally ``data/raw``.
        stem: File stem, e.g. ``"fred_DGS10"``.

    Returns:
        A ``(parquet_path, meta_path)`` pair.
    """
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.meta.json"


def checksum_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cached(cache_dir: Path, stem: str) -> pd.DataFrame | None:
    """Return a cached frame, or ``None`` if there is no usable cache entry.

    A parquet without its sidecar is treated as absent rather than trusted: provenance is
    part of the artifact, not an optional extra.

    Args:
        cache_dir: Directory holding raw pulls.
        stem: File stem to look up.

    Returns:
        The cached frame, or ``None``.
    """
    parquet_path, meta_path = cache_paths(cache_dir, stem)
    if not parquet_path.is_file():
        return None
    if not meta_path.is_file():
        logger.warning(
            "%s exists but its .meta.json sidecar does not; ignoring the cache and "
            "refetching so provenance is recorded.",
            parquet_path,
        )
        return None
    logger.info("Cache hit: %s", parquet_path)
    return pd.read_parquet(parquet_path)


def write_cache(
    frame: pd.DataFrame,
    cache_dir: Path,
    stem: str,
    *,
    source: str,
    params: dict[str, Any] | None = None,
) -> Path:
    """Write a frame to the cache with a provenance sidecar.

    Args:
        frame: Data to cache. Its index is preserved.
        cache_dir: Directory holding raw pulls. Created if absent.
        stem: File stem, e.g. ``"fred_DGS10"``.
        source: Human-readable origin, e.g. ``"FRED"`` or ``"yfinance"``.
        params: Request parameters worth recording, e.g. ticker and date range.

    Returns:
        Path to the written parquet file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, meta_path = cache_paths(cache_dir, stem)
    frame.to_parquet(parquet_path)

    index = frame.index
    meta = {
        "meta_version": META_VERSION,
        "source": source,
        "params": params or {},
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "index_start": str(index.min()) if len(index) else None,
        "index_end": str(index.max()) if len(index) else None,
        "sha256": checksum_file(parquet_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Cached %d rows to %s", len(frame), parquet_path)
    return parquet_path


def read_meta(cache_dir: Path, stem: str) -> dict[str, Any] | None:
    """Return a cache entry's sidecar metadata, or ``None`` if absent.

    Args:
        cache_dir: Directory holding raw pulls.
        stem: File stem to look up.

    Returns:
        The parsed sidecar, or ``None``.
    """
    _, meta_path = cache_paths(cache_dir, stem)
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
