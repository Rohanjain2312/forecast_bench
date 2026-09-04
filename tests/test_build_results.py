"""Tests for results assembly, focused on which rows reach which table.

The headline is a *registered* cut, not "everything on disk". Getting that wrong does not
raise — it produces a plausible table averaged over things that should never have been
combined.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast_bench.backtest.writer import SCHEMA  # noqa: E402
from scripts.build_results import build_tables  # noqa: E402


def _forecasts(
    cadence: str, model_id: str = "ARIMA", value: float = 1.0
) -> pd.DataFrame:
    """Tidy rows for one model under one cadence, at every reported horizon."""
    rows = []
    for origin in pd.date_range("2015-01-02", periods=40, freq="21D"):
        for step in (1, 5, 21):
            for level in (0.1, 0.5, 0.9):
                rows.append(
                    {
                        "origin": origin,
                        "target_date": origin + pd.Timedelta(days=step),
                        "step": step,
                        "model_id": model_id,
                        "quantile": level,
                        "value": value + (level - 0.5),
                        "actual": 1.0,
                        "regime": "calm",
                        "block_id": origin.year,
                        "series": "spy_logrv",
                        "arm": "A",
                        "cadence": cadence,
                    }
                )
    return pd.DataFrame(rows)[SCHEMA]


def test_headline_uses_only_the_matched_cadence() -> None:
    """Native-cadence rows must not reach the headline table.

    Regression test: ``build_tables`` grouped the whole forecast set by series alone, so
    the moment a native run existed its rows were averaged into the headline together with
    the matched ones. Nothing raised — the table simply became a blend of two cadences
    that PREREGISTRATION.md §2 defines as separate cuts, one headline and one secondary.
    """
    matched = _forecasts("block_ys", value=1.0)
    native = _forecasts("every_fold", value=5.0)
    forecasts = pd.concat([matched, native], ignore_index=True)

    tables = build_tables(forecasts, targets={})
    headline = tables["headline"]

    matched_only = build_tables(matched, targets={})["headline"]

    # The headline computed from both cadences must equal the one from matched alone.
    pd.testing.assert_frame_equal(
        headline.sort_values(["model_id", "horizon"]).reset_index(drop=True),
        matched_only.sort_values(["model_id", "horizon"]).reset_index(drop=True),
    )


def test_cadence_comparison_keeps_both_cadences() -> None:
    """The secondary table is where the two cadences are meant to appear together."""
    forecasts = pd.concat(
        [_forecasts("block_ys", value=1.0), _forecasts("every_fold", value=5.0)],
        ignore_index=True,
    )

    comparison = build_tables(forecasts, targets={})["cadence_comparison"]

    assert set(comparison["cadence"].unique()) == {"block_ys", "every_fold"}


def test_headline_survives_a_matched_only_run() -> None:
    """Before any native run exists, the headline still builds."""
    tables = build_tables(_forecasts("block_ys"), targets={})

    assert not tables["headline"].empty
    assert set(tables["headline"]["horizon"]) == {1, 5, 21}


def test_empty_forecasts_are_refused() -> None:
    """An empty input should fail loudly rather than write empty tables."""
    with pytest.raises(ValueError):
        build_tables(pd.DataFrame(columns=SCHEMA), targets={})
