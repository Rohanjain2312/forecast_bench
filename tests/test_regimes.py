"""Regime thresholds match the frozen config, and assignment is boundary-correct."""

import pandas as pd
import pytest
import yaml

from forecast_bench.evaluation.regimes import (
    CALM_UPPER,
    EXPECTED_CALM_UPPER,
    EXPECTED_NORMAL_UPPER,
    NORMAL_UPPER,
    REGIME_LABELS,
    REGIMES_CONFIG_PATH,
    FrozenThresholdError,
    _load_thresholds,
    assign_regime,
    regime_series,
)


def test_loaded_thresholds_match_the_committed_values() -> None:
    """The module-level constants equal the values frozen in 2000-2014."""
    assert CALM_UPPER == EXPECTED_CALM_UPPER == 15.9
    assert NORMAL_UPPER == EXPECTED_NORMAL_UPPER == 22.5582


def test_yaml_file_matches_the_module_constants() -> None:
    """The committed YAML and the Python constants agree."""
    loaded = yaml.safe_load(REGIMES_CONFIG_PATH.read_text(encoding="utf-8"))
    assert loaded["thresholds"]["calm_upper"] == CALM_UPPER
    assert loaded["thresholds"]["normal_upper"] == NORMAL_UPPER
    assert loaded["labels"] == REGIME_LABELS


def test_config_records_that_it_was_computed_on_pre_2015_data() -> None:
    """The derivation window is recorded in the file, so the numbers are auditable."""
    loaded = yaml.safe_load(REGIMES_CONFIG_PATH.read_text(encoding="utf-8"))
    assert loaded["computed_on"]["start"] == "2000-01-01"
    assert loaded["computed_on"]["end"] == "2014-12-31"
    assert loaded["source_series"] == "VIXCLS"


def test_config_carries_the_do_not_recompute_warning() -> None:
    """The warning is in the file itself, where anyone editing it will see it."""
    text = REGIMES_CONFIG_PATH.read_text(encoding="utf-8")
    assert "NEVER RECOMPUTE" in text


def test_altered_thresholds_raise_at_load(tmp_path) -> None:
    """A changed threshold fails loudly instead of silently restratifying every table."""
    altered = tmp_path / "regimes.yaml"
    altered.write_text(
        yaml.safe_dump({"thresholds": {"calm_upper": 14.0, "normal_upper": 22.5582}}),
        encoding="utf-8",
    )
    with pytest.raises(FrozenThresholdError, match="committed values"):
        _load_thresholds(altered)


def test_missing_config_falls_back_to_the_committed_constants(tmp_path) -> None:
    """An absent file yields the frozen values rather than raising.

    Regression test: this used to raise, which broke every pip-installed consumer. Only
    ``forecast_bench/`` ships in the package, so ``experiments/configs/regimes.yaml``
    does not exist on Colab or in the Hugging Face Space at all. It failed live in
    notebook 05.

    This is safe precisely because the constants in the module *are* the frozen values —
    the YAML is a cross-check that surfaces changes in a diff, not the source of truth.
    A missing file cannot silently recompute anything; only a present-and-different file
    could, and that still raises.
    """
    calm, normal = _load_thresholds(tmp_path / "absent.yaml")

    assert calm == EXPECTED_CALM_UPPER
    assert normal == EXPECTED_NORMAL_UPPER


def test_altered_thresholds_still_raise_even_though_absence_does_not(tmp_path) -> None:
    """Tolerating absence must not weaken the guard against a *changed* file."""
    altered = tmp_path / "regimes.yaml"
    altered.write_text(
        yaml.safe_dump({"thresholds": {"calm_upper": 15.9, "normal_upper": 99.0}}),
        encoding="utf-8",
    )
    with pytest.raises(FrozenThresholdError, match="committed values"):
        _load_thresholds(altered)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (5.0, "calm"),
        (15.89, "calm"),
        (15.9, "calm"),
        (15.91, "normal"),
        (22.5582, "normal"),
        (22.56, "stressed"),
        (80.0, "stressed"),
    ],
)
def test_assignment_is_boundary_correct(level, expected) -> None:
    """Boundaries are inclusive at the top of each band."""
    assert assign_regime(level) == expected


def test_missing_vix_yields_no_label() -> None:
    """A missing VIX print gives None rather than a default regime."""
    assert assign_regime(None) is None
    assert assign_regime(float("nan")) is None


def test_regime_series_labels_a_whole_series() -> None:
    """A VIX series maps to labels on the same index."""
    vix = pd.Series(
        [10.0, 18.0, 40.0, float("nan")],
        index=pd.date_range("2020-01-01", periods=4, freq="B"),
    )
    labels = regime_series(vix)

    assert list(labels[:3]) == ["calm", "normal", "stressed"]
    # pandas normalises a mapped None to NaN; both mean 'no label'.
    assert pd.isna(labels.iloc[3])
    assert labels.index.equals(vix.index)
