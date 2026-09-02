"""Tests for target construction: the estimators, the variance floor, and no forward-fill.

The Garman-Klass check is against a value computed by hand rather than against the
implementation, so a sign error or a dropped constant cannot pass by agreeing with itself.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.data import covariates as covariates_module
from forecast_bench.data.targets import (
    build_dgs10,
    build_spy_logrv,
    floor_variance,
    garman_klass_variance,
    parkinson_variance,
    training_variance_floor,
)

# One bar, chosen so the arithmetic is checkable on paper:
#   O=100, H=110, L=90, C=105
#   ln(H/L) = ln(1.2222...)      = 0.20067069546215124
#   0.5 * ln(H/L)^2              = 0.020134364008631726
#   ln(C/O) = ln(1.05)           = 0.04879016416943205
#   (2*ln2 - 1)                  = 0.3862943611198906
#   (2*ln2 - 1) * ln(C/O)^2      = 0.0009195660469904367
#   GK = 0.020134364 - 0.000919566 = 0.01921479796164129
KNOWN_BAR = {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0}
KNOWN_BAR_GK = 0.01921479796164129
KNOWN_BAR_PARKINSON = 0.014523873553353111


@pytest.fixture
def one_bar() -> pd.DataFrame:
    """A single OHLC bar with hand-computable estimator values."""
    return pd.DataFrame(KNOWN_BAR, index=pd.DatetimeIndex(["2020-01-02"]))


# --- Estimators ------------------------------------------------------------------------


def test_garman_klass_matches_a_hand_computed_bar(one_bar) -> None:
    """The estimator reproduces a value computed by hand from the formula."""
    result = garman_klass_variance(one_bar)
    assert result.iloc[0] == pytest.approx(KNOWN_BAR_GK, rel=1e-12)


def test_parkinson_matches_a_hand_computed_bar(one_bar) -> None:
    """The high-low cross-check estimator reproduces its hand-computed value."""
    result = parkinson_variance(one_bar)
    assert result.iloc[0] == pytest.approx(KNOWN_BAR_PARKINSON, rel=1e-12)


def test_garman_klass_is_zero_for_a_flat_bar() -> None:
    """A bar with no range and no move has zero estimated variance."""
    flat = pd.DataFrame(
        {"open": [50.0], "high": [50.0], "low": [50.0], "close": [50.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    assert garman_klass_variance(flat).iloc[0] == pytest.approx(0.0, abs=1e-15)


def test_garman_klass_is_non_negative_for_well_formed_bars() -> None:
    """A bar with H >= max(O, C) and L <= min(O, C) can never produce a negative estimate.

    Going negative needs ``|ln(C/O)| > 1.138 * ln(H/L)``, which a well-formed bar cannot
    satisfy, since the range always contains the open and the close. The reachable failure
    is exactly zero, on a flat bar. Recorded as a test because the floor's justification
    depends on which case actually occurs.
    """
    rng = np.random.default_rng(7)
    opens = 100.0 + rng.standard_normal(500)
    closes = opens + rng.standard_normal(500)
    highs = np.maximum(opens, closes) + rng.random(500)
    lows = np.minimum(opens, closes) - rng.random(500)
    bars = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.date_range("2020-01-01", periods=500, freq="B"),
    )

    assert (garman_klass_variance(bars) >= 0).all()


def test_garman_klass_goes_negative_on_a_malformed_bar() -> None:
    """A bad tick whose close sits outside the high-low range drives the estimate below zero.

    This is the case the floor exists for in practice, alongside the flat bar. ``ln()`` of
    a non-positive variance is ``-inf`` or ``NaN``, either of which would silently poison
    the target.
    """
    malformed = pd.DataFrame(
        {"open": [100.0], "high": [100.5], "low": [99.5], "close": [110.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    assert garman_klass_variance(malformed).iloc[0] < 0


def test_estimators_require_their_columns() -> None:
    """A frame missing a price column is a hard error, not a silent NaN."""
    frame = pd.DataFrame({"open": [1.0], "high": [2.0]})
    with pytest.raises(KeyError):
        garman_klass_variance(frame)
    with pytest.raises(KeyError):
        parkinson_variance(pd.DataFrame({"high": [2.0]}))


# --- The variance floor ----------------------------------------------------------------


def test_floor_is_computed_on_the_training_window_only() -> None:
    """Test-period values cannot influence the floor.

    The series is tiny in the training window and enormous afterwards. A floor computed
    over the full sample would differ; one computed on the training window must not.
    """
    index = pd.date_range("2000-01-03", periods=400, freq="B")
    variance = pd.Series(np.full(len(index), 1e-3), index=index)
    post = index >= pd.Timestamp("2001-01-01")
    variance[post] = 1e-9  # a far quieter test period than anything in training

    floor_train = training_variance_floor(variance, train_end="2001-01-01")
    floor_full = float(np.percentile(variance[variance > 0], 0.1))

    assert floor_train == pytest.approx(1e-3)
    assert floor_full == pytest.approx(1e-9)
    # Using the full sample would drag the floor six orders of magnitude down, shaped by
    # data the model is not allowed to have seen.
    assert floor_train > floor_full * 1e5


def test_floor_ignores_non_positive_values() -> None:
    """The percentile is taken over strictly-positive estimates only."""
    index = pd.date_range("2000-01-03", periods=100, freq="B")
    variance = pd.Series(np.linspace(1e-6, 1e-3, len(index)), index=index)
    variance.iloc[:10] = -1.0

    floor = training_variance_floor(variance, train_end="2001-01-01")

    assert floor > 0


def test_floor_raises_without_positive_training_values() -> None:
    """A training window with nothing positive in it is an error, not a silent zero."""
    index = pd.date_range("2000-01-03", periods=10, freq="B")
    variance = pd.Series(np.full(len(index), -1.0), index=index)
    with pytest.raises(ValueError, match="No strictly-positive"):
        training_variance_floor(variance, train_end="2001-01-01")


def test_floor_variance_clips_and_warns(caplog) -> None:
    """Flooring is applied and its frequency is logged at WARNING."""
    variance = pd.Series([-1.0, 0.0, 1.0, 2.0])
    with caplog.at_level("WARNING"):
        result = floor_variance(variance, floor=0.5)

    assert result.tolist() == [0.5, 0.5, 1.0, 2.0]
    assert "floor" in caplog.text.lower()
    assert "2 of 4" in caplog.text


def test_floor_leaves_small_positive_values_untouched() -> None:
    """A quiet-but-valid day is not winsorized.

    Clipping everything below the floor would alter the actuals that models are scored
    against. Only non-positive estimates, which ln() cannot handle, may be replaced.
    """
    variance = pd.Series([-1.0, 1e-12, 0.5, 2.0])

    result = floor_variance(variance, floor=0.25)

    assert result.iloc[0] == 0.25  # non-positive, replaced
    assert result.iloc[1] == 1e-12  # positive but tiny, left alone
    assert result.iloc[2] == 0.5
    assert result.iloc[3] == 2.0


def test_logrv_has_no_nans_after_flooring() -> None:
    """The floor guarantees ln() is finite everywhere, including on negative bars."""
    index = pd.date_range("2000-01-03", periods=600, freq="B")
    rng = np.random.default_rng(0)
    base = 100.0 + rng.standard_normal(len(index)).cumsum()
    ohlc = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
        },
        index=index,
    )
    # Force a non-positive estimate: a near-zero range with a large close-open move.
    ohlc.iloc[100] = [base[100], base[100] + 0.001, base[100] - 0.001, base[100] + 2.0]

    target = build_spy_logrv(ohlc, train_end="2002-01-01")

    assert not target.isna().any()
    assert np.isfinite(target).all()
    assert target.name == "spy_logrv"


# --- No forward-fill -------------------------------------------------------------------


def test_dgs10_drops_holidays_and_never_forward_fills() -> None:
    """NaN observations are removed, not carried forward from the previous value."""
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    raw = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0], index=index)

    target = build_dgs10(raw)

    assert len(target) == 4
    assert target.tolist() == [1.0, 3.0, 5.0, 6.0]
    # A forward-fill would have produced [1, 1, 3, 3, 5, 6] and kept the length at 6.
    assert index[1] not in target.index
    assert index[3] not in target.index


def test_dgs10_is_not_differenced() -> None:
    """The contrast target is levels, not changes."""
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    raw = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0], index=index)

    target = build_dgs10(raw)

    pd.testing.assert_series_equal(
        target, raw.rename("dgs10"), check_freq=False, check_names=True
    )


def test_spy_logrv_does_not_invent_calendar_days() -> None:
    """The index is the days SPY actually traded, not a synthetic business-day grid."""
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-07"])
    ohlc = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=index,
    )

    target = build_spy_logrv(ohlc, train_end="2021-01-01")

    assert list(target.index) == list(index)
    assert pd.Timestamp("2020-01-06") not in target.index


# --- Covariate allowlist ---------------------------------------------------------------


def test_configured_covariates_are_all_allowlisted() -> None:
    """Every series in the shipped covariate configuration passes the allowlist."""
    for target in covariates_module.COVARIATE_SERIES:
        covariates_module.covariate_series_for(target)


def test_validate_covariates_rejects_a_revised_series() -> None:
    """A frame carrying a revised series is refused."""
    frame = pd.DataFrame({"vixcls": [1.0], "cpiaucsl": [2.0]})
    with pytest.raises(ValueError, match="non-allowlisted"):
        covariates_module.validate_covariates(frame)


def test_validate_covariates_accepts_allowlisted_and_calendar_columns() -> None:
    """Allowlisted series and day-of-week indicators are both permitted."""
    frame = pd.DataFrame({"vixcls": [1.0], "dgs10": [2.0], "dow_mon": [1.0]})
    covariates_module.validate_covariates(frame)


def test_covariate_sets_match_the_registered_design() -> None:
    """The covariate sets are the ones registered in DECISIONS.md D3."""
    assert covariates_module.COVARIATE_SERIES["spy_logrv"] == ["VIXCLS", "DGS10"]
    assert covariates_module.COVARIATE_SERIES["dgs10"] == [
        "T10Y2Y",
        "DGS3MO",
        "VIXCLS",
        "DFF",
    ]
