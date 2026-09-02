"""Construction of the two modelling targets.

Primary: log realized variance of SPY, from the Garman-Klass range estimator on daily
OHLC. Contrast: the 10-year Treasury yield in levels.

Assumes SPY OHLC is split- and dividend-adjusted consistently across the full span; a
mid-series adjustment change would produce a spurious volatility jump that the estimator
would faithfully report as a volatility event. The estimator uses same-day price ratios,
so it reads the unadjusted bar deliberately — ``adj_close`` is carried alongside for
reference but never enters the target.
"""

import logging

import numpy as np
import pandas as pd

from forecast_bench.config import TEST_START, TRAIN_START
from forecast_bench.data.fred_client import fetch_fred_series
from forecast_bench.data.yahoo_client import fetch_ohlc

logger = logging.getLogger(__name__)

#: Percentile of the strictly-positive training-window estimates used as the variance
#: floor. Low-range days can drive the Garman-Klass estimate to zero or below, and ln()
#: would produce NaN or -inf.
FLOOR_PERCENTILE = 0.1

#: Constant from Garman & Klass (1980): ``2 * ln(2) - 1``.
GK_CLOSE_OPEN_COEFFICIENT = 2.0 * np.log(2.0) - 1.0


def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Compute the Garman-Klass daily variance estimate from OHLC bars.

    ``sigma2 = 0.5 * ln(H/L)**2 - (2*ln(2) - 1) * ln(C/O)**2``

    Roughly seven times more efficient than close-to-close squared returns, which is why
    it is the standard free substitute for intraday realized variance.

    Args:
        ohlc: Frame with ``open``, ``high``, ``low`` and ``close`` columns.

    Returns:
        The variance estimate per bar. **May be non-positive**, so the caller must floor it
        before taking a log. See :func:`floor_variance`.

    Note:
        Two distinct cases produce a non-positive estimate, and they are not the same
        problem. A *well-formed* bar can only reach exactly zero, on a fully flat day
        (H = L = O = C): going negative would require ``|ln(C/O)| > 1.138 * ln(H/L)``,
        which is unreachable when the range contains the open and the close. A *malformed*
        bar — a bad tick with the close outside the high-low range — can go genuinely
        negative. ``ln()`` of either is ``-inf`` or ``NaN``, so both are floored, but only
        the second indicates a data-quality problem worth investigating.

    Raises:
        KeyError: If any required column is absent.
    """
    missing = [
        column
        for column in ("open", "high", "low", "close")
        if column not in ohlc.columns
    ]
    if missing:
        raise KeyError(f"OHLC frame is missing {missing}")

    log_hl = np.log(ohlc["high"] / ohlc["low"])
    log_co = np.log(ohlc["close"] / ohlc["open"])
    variance = 0.5 * log_hl**2 - GK_CLOSE_OPEN_COEFFICIENT * log_co**2
    variance.name = "gk_variance"
    return variance


def parkinson_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Compute the Parkinson daily variance estimate, a high-low-only cross-check.

    ``sigma2 = ln(H/L)**2 / (4 * ln(2))``

    Always non-negative, since it uses only the range. Used to sanity-check
    :func:`garman_klass_variance` and as the fallback if OHLC quality is poor on a span.

    Args:
        ohlc: Frame with ``high`` and ``low`` columns.

    Returns:
        The variance estimate per bar.

    Raises:
        KeyError: If ``high`` or ``low`` is absent.
    """
    missing = [column for column in ("high", "low") if column not in ohlc.columns]
    if missing:
        raise KeyError(f"OHLC frame is missing {missing}")

    log_hl = np.log(ohlc["high"] / ohlc["low"])
    variance = log_hl**2 / (4.0 * np.log(2.0))
    variance.name = "parkinson_variance"
    return variance


def training_variance_floor(
    variance: pd.Series,
    train_end: str = TEST_START,
    percentile: float = FLOOR_PERCENTILE,
) -> float:
    """Compute the variance floor from the training window only.

    Args:
        variance: Raw variance estimates, possibly containing non-positive values.
        train_end: Exclusive upper bound of the window the floor may be computed on.
            Defaults to the start of the test span.
        percentile: Percentile of strictly-positive estimates to use, in percent.

    Returns:
        The floor value.

    Raises:
        ValueError: If the training window contains no strictly-positive estimates.

    Note:
        The floor is deliberately computed on data strictly before ``train_end`` and never
        on the full series. Taking the percentile over the whole sample would let the test
        period's volatility distribution set a constant that shapes the training target —
        a leak that no metric in this study would reveal.

        The floor is a single constant rather than a per-fold quantity. That is sound
        because it is derived only from data available before the first forecast origin,
        so every fold may see it, in the same way as the frozen VIX regime thresholds.
    """
    window = variance.loc[variance.index < pd.Timestamp(train_end)]
    positive = window[window > 0]
    if positive.empty:
        raise ValueError(
            f"No strictly-positive Garman-Klass estimates before {train_end}; "
            "cannot compute a variance floor."
        )
    return float(np.percentile(positive, percentile))


def floor_variance(variance: pd.Series, floor: float) -> pd.Series:
    """Replace non-positive variance estimates with the floor, logging how often it fires.

    Args:
        variance: Raw variance estimates.
        floor: Replacement value, from :func:`training_variance_floor`.

    Returns:
        The series with non-positive entries replaced.

    Note:
        Only **non-positive** estimates are replaced. A small but positive estimate is a
        real observation of a quiet day and is left alone: clipping everything below the
        floor would winsorize the low tail of the target, silently altering the actuals
        that every model is scored against. On SPY 2000-2026 that distinction is the
        difference between touching 0 bars and touching 36.

        The count is logged at WARNING rather than applied silently. Flooring changes the
        target, so how often it fires belongs in the writeup, not only in a log.
    """
    non_positive = variance <= 0.0
    count = int(non_positive.sum())
    if count:
        logger.warning(
            "Garman-Klass estimate was non-positive on %d of %d bars (%.2f%%) and was "
            "replaced with the training-window floor %.3e; ln() of a non-positive "
            "variance would be -inf or NaN.",
            count,
            len(variance),
            100.0 * count / len(variance),
            floor,
        )
    return variance.where(~non_positive, floor)


def build_spy_logrv(
    ohlc: pd.DataFrame | None = None,
    train_end: str = TEST_START,
    start: str = TRAIN_START,
) -> pd.Series:
    """Build the primary target: log Garman-Klass realized variance for SPY.

    Args:
        ohlc: Pre-fetched OHLC frame. Fetched via
            :func:`~forecast_bench.data.yahoo_client.fetch_ohlc` when ``None``.
        train_end: Exclusive upper bound of the window the variance floor is computed on.
        start: First date retained, clipping the series to the study span.

    Returns:
        Log variance, named ``spy_logrv``, indexed by trading day with no missing values.

    Note:
        Bars where any OHLC field is missing are dropped rather than interpolated. The
        index is the set of days SPY actually traded, not a synthetic business-day grid,
        so no holiday is invented.
    """
    frame = ohlc if ohlc is not None else fetch_ohlc("SPY")
    frame = frame.loc[frame.index >= pd.Timestamp(start)]
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    variance = garman_klass_variance(frame)
    floor = training_variance_floor(variance, train_end=train_end)
    target = np.log(floor_variance(variance, floor))
    target.name = "spy_logrv"

    if target.isna().any():
        raise ValueError(
            f"spy_logrv contains {int(target.isna().sum())} NaNs after flooring; the "
            "floor should have made every value strictly positive."
        )
    return target


def build_dgs10(
    series: pd.Series | None = None,
    start: str = TRAIN_START,
) -> pd.Series:
    """Build the contrast target: the 10-year Treasury yield in levels.

    Args:
        series: Pre-fetched ``DGS10`` series. Fetched from FRED when ``None``.
        start: First date retained, clipping the series to the study span.

    Returns:
        The yield in percent, named ``dgs10``, indexed by observation date.

    Note:
        Deliberately **not** differenced. The honest finding on a near-unit-root series is
        that the random walk is hard to beat, and differencing away the persistence would
        hide the very thing the study is measuring. ARIMA selects its own ``d`` per fold;
        that is the model's decision, not ours.

        Market holidays arrive from FRED as NaN and are **dropped, never forward-filled**.
        Forward-filling a target manufactures an observation that did not exist and lets a
        model score against a value it was effectively handed — a subtle leak.
    """
    raw = series if series is not None else fetch_fred_series("DGS10")
    # FRED serves DGS10 back to 1962. The study trains from TRAIN_START, and carrying four
    # extra decades into the modelling artifact makes covariate-coverage statistics
    # meaningless (VIXCLS does not exist before 1990) without adding a usable observation.
    # The full history stays in data/raw/ either way.
    target = raw.loc[raw.index >= pd.Timestamp(start)].dropna()
    target.name = "dgs10"
    return target
