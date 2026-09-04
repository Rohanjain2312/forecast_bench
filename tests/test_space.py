"""Tests for the demo layer: model cards, plotting, and the Space's own constraints.

The rule these enforce is the one that keeps the demo trustworthy: no number appears in
the Space that is not also in the published results, and no model appears without a
plain-language description.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import PROJECT_ROOT, QUANTILE_GRID
from forecast_bench.models.registry import all_registered_model_classes
from forecast_bench.viz.forecast_plots import comparison_chart, fan_chart
from forecast_bench.viz.results_plots import (
    coverage_width_scatter,
    regime_heatmap,
    sample_efficiency_curve,
    skill_bar_chart,
)

sys.path.insert(0, str(PROJECT_ROOT / "space"))

import model_cards  # noqa: E402

SPACE_DIR = PROJECT_ROOT / "space"


# --- Model cards ------------------------------------------------------------------------


def test_every_registered_model_has_a_plain_language_description() -> None:
    """A model in the panel with no gloss would appear in the demo unexplained."""
    missing = set(all_registered_model_classes()) - set(model_cards.MODEL_CARDS)
    assert not missing, f"models with no description: {sorted(missing)}"


def test_every_model_has_a_family_label() -> None:
    """The interface groups models by family, so every one needs a label."""
    missing = set(model_cards.MODEL_CARDS) - set(model_cards.MODEL_FAMILIES)
    assert not missing


def test_descriptions_avoid_formulas_and_jargon_markers() -> None:
    """The glosses are for someone who has never heard of ARIMA."""
    for model_id, text in model_cards.MODEL_CARDS.items():
        assert "$" not in text, f"{model_id} contains a formula"
        assert len(text) > 40, f"{model_id} description is too thin"
        assert text.strip().endswith("."), f"{model_id} description is not a sentence"


def test_headline_finding_states_the_loss_plainly() -> None:
    """PREREGISTRATION.md commits to saying 'lost', not 'showed mixed results'."""
    text = model_cards.HEADLINE_FINDING.lower()
    assert "lost" in text
    assert "mixed results" not in text
    # The counter-evidence must be there too, or the demo is telling half the story.
    assert "85%" in model_cards.HEADLINE_FINDING


def test_markdown_table_covers_every_model() -> None:
    """The explanation tab lists all of them."""
    table = model_cards.markdown_table()
    for model_id in model_cards.MODEL_CARDS:
        assert model_id in table


# --- Space files ------------------------------------------------------------------------


def test_space_has_the_files_the_push_script_requires() -> None:
    """A partial Space fails to build for reasons the logs do not explain."""
    from scripts.push_artifacts import SPACE_FILES

    for name in SPACE_FILES:
        assert (SPACE_DIR / name).is_file(), f"space/{name} is missing"


def test_space_card_declares_cpu_basic_and_gradio() -> None:
    """DECISIONS.md D12 chose CPU Basic deliberately; the card must not drift from it."""
    card = (SPACE_DIR / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---")
    assert "sdk: gradio" in card
    assert "app_file: app.py" in card
    assert "CPU Basic" in card
    # HF rejects a longer one, which fails the deploy rather than the build.
    short = next(
        line for line in card.splitlines() if line.startswith("short_description:")
    )
    assert len(short.split(": ", 1)[1]) <= 60


def test_space_requirements_install_the_package_from_github() -> None:
    """The Space must run the same code the benchmark ran, not a copy."""
    requirements = (SPACE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "github.com/Rohanjain2312/forecast_bench" in requirements
    assert "peft" in requirements, "loading fine-tuned adapters needs peft"


def test_space_app_recomputes_no_metrics() -> None:
    """Every number shown must come from the published tables.

    The Space importing anything from ``evaluation.metrics`` would mean it could compute a
    number the documentation does not contain, which is the drift this rule prevents.
    """
    source = (SPACE_DIR / "app.py").read_text(encoding="utf-8")
    assert "evaluation.metrics" not in source
    assert "evaluation import metrics" not in source


# --- Plots ------------------------------------------------------------------------------


@pytest.fixture
def history() -> pd.Series:
    """A short series standing in for observed history."""
    index = pd.bdate_range("2020-01-01", periods=200)
    return pd.Series(np.linspace(-10, -9, 200), index=index)


@pytest.fixture
def quantiles() -> dict[float, np.ndarray]:
    """A forecast covering the full study grid."""
    return {level: np.full(21, -9.5 + (level - 0.5)) for level in QUANTILE_GRID}


def test_fan_chart_labels_axes_in_words(history, quantiles) -> None:
    """No symbols on the axes: the demo is read by people who have not seen log-variance."""
    index = pd.bdate_range(start=history.index[-1] + pd.Timedelta(days=1), periods=21)
    figure = fan_chart(history, quantiles, index, series="spy_logrv", model_name="Test")

    y_label = figure.layout.yaxis.title.text
    assert "volatility" in y_label.lower()
    assert "sigma" not in y_label.lower() and "σ" not in y_label
    assert figure.layout.xaxis.title.text == "Date"


def test_comparison_chart_draws_one_line_per_model(history, quantiles) -> None:
    """Medians only, plus the history line."""
    index = pd.bdate_range(start=history.index[-1] + pd.Timedelta(days=1), periods=21)
    figure = comparison_chart(
        history, {"A": quantiles, "B": quantiles}, index, series="spy_logrv"
    )
    assert len(figure.data) >= 3


def test_results_plots_render_from_a_headline_table() -> None:
    """The chart functions accept exactly what aggregate.py produces."""
    headline = pd.DataFrame(
        {
            "series": ["spy_logrv"] * 3,
            "model_id": ["LogHAR", "ARIMA", "RandomWalk"],
            "horizon": [1, 1, 1],
            "skill_wql": [0.17, 0.16, 0.0],
            "coverage_80": [0.72, 0.77, 0.72],
            "width_80": [1.98, 2.11, 2.44],
        }
    )
    assert skill_bar_chart(headline, "spy_logrv", 1) is not None
    assert coverage_width_scatter(headline, "spy_logrv", 1) is not None

    regimes = pd.DataFrame(
        {
            "series": ["spy_logrv"] * 4,
            "model_id": ["LogHAR", "LogHAR", "ARIMA", "ARIMA"],
            "horizon": [1] * 4,
            "regime": ["calm", "stressed", "calm", "stressed"],
            "skill_wql": [0.2, 0.1, 0.18, 0.05],
        }
    )
    assert regime_heatmap(regimes, "spy_logrv", 1) is not None

    sweep = pd.DataFrame(
        {
            "series": ["spy_logrv"] * 4,
            "model_id": ["N-BEATS"] * 4,
            "horizon": [1] * 4,
            "training_window": ["1y", "3y", "10y", "full"],
            "skill_wql": [-0.5, -0.2, 0.04, 0.06],
        }
    )
    assert sample_efficiency_curve(sweep) is not None
