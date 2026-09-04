"""The forecast_bench demo: five tabs over a leakage-safe forecasting benchmark.

Design constraints, all from DECISIONS.md D12 and the two-minute rule:

- Something real is on screen before the visitor clicks anything.
- Every axis is labelled in words, never symbols.
- Every model name carries a plain-language gloss from ``model_cards.py``.
- Limitations are a tab, not a link.
- **No number appears here that is not also in docs/benchmark_results.md.** Every table is
  read from the published results rather than recomputed, which is what makes that
  enforceable rather than aspirational.

Live inference runs on CPU. Measured at 0.85 s per forecast for Chronos-2 with a 512-step
context, which is why this is a live demo rather than a grid of pre-computed pictures.
"""

import logging
import os

import gradio as gr
import pandas as pd
from model_cards import (
    HEADLINE_FINDING,
    PROJECT_EXPLANATION,
    SERIES_CARDS,
    describe,
    markdown_table,
)

from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON, setup_logging
from forecast_bench.viz.forecast_plots import SERIES_LABELS, comparison_chart, fan_chart
from forecast_bench.viz.results_plots import (
    coverage_width_scatter,
    regime_heatmap,
    sample_efficiency_curve,
    skill_bar_chart,
)

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

DATASET_REPO = os.getenv("HF_DATASET_REPO", "rohanjain2312/forecastbench-data")
MODEL_REPO = os.getenv("HF_MODEL_REPO", "rohanjain2312/forecastbench-chronos")
REPO_URL = "https://github.com/Rohanjain2312/forecast_bench"

#: Models offered for live forecasting. Kept small so the landing tab stays fast.
LIVE_MODELS = ["Chronos-2 (adapted)", "Chronos-2 (untouched)", "ARIMA", "Random walk"]

_CACHE: dict[str, object] = {}


def _hub_parquet(path: str) -> pd.DataFrame:
    """Read a parquet from the dataset repo, caching it for the session."""
    from huggingface_hub import hf_hub_download

    if path not in _CACHE:
        local = hf_hub_download(DATASET_REPO, path, repo_type="dataset")
        _CACHE[path] = pd.read_parquet(local)
    return _CACHE[path]


def results(name: str) -> pd.DataFrame:
    """Load one published results table.

    Args:
        name: Table name, e.g. ``"headline"``.

    Returns:
        The table as published, or an empty frame if it is not available.
    """
    try:
        return _hub_parquet(f"results/{name}.parquet")
    except Exception as error:  # noqa: BLE001 - a missing table must not break the app
        logger.warning("Could not load %s: %s", name, error)
        return pd.DataFrame()


def series_data(series: str) -> pd.DataFrame:
    """Load a processed series from the dataset repo."""
    return _hub_parquet(f"processed/{series}.parquet")


def _forecast_one(model_label: str, history: pd.Series, series: str, origin) -> dict:
    """Produce one model's quantile forecast for the live tab.

    Args:
        model_label: Human-facing model name from :data:`LIVE_MODELS`.
        history: Observations up to and including the origin.
        series: Series name.
        origin: Forecast origin.

    Returns:
        Mapping of quantile level to a forecast path.
    """
    frame = history.to_frame(name=series)

    if model_label == "Random walk":
        from forecast_bench.models.naive import RandomWalk

        model = RandomWalk(target_column=series)
    elif model_label == "ARIMA":
        from forecast_bench.models.classical.arima import ARIMA

        model = ARIMA(target_column=series, max_train=1000)
    elif model_label == "Chronos-2 (untouched)":
        from forecast_bench.models.foundation.chronos2 import Chronos2ZeroShot

        model = Chronos2ZeroShot(target_column=series)
    else:
        from forecast_bench.models.foundation.chronos2 import Chronos2FineTuned

        model = Chronos2FineTuned(
            target_column=series, series=series, arm="A", repo_id=MODEL_REPO
        )

    model.fit(frame, origin=origin)
    index = pd.bdate_range(start=origin + pd.Timedelta(days=1), periods=MAX_HORIZON)
    return model.predict(horizon=MAX_HORIZON, index=index).quantiles, index


def run_forecast(series: str, date_text: str, model_label: str):
    """Produce the live forecast shown on the landing tab.

    Args:
        series: Which series to forecast.
        date_text: Forecast origin, ``YYYY-MM-DD``.
        model_label: Which model to run.

    Returns:
        A ``(figure, caption)`` pair.
    """
    try:
        frame = series_data(series)
        target = frame[series].dropna()
        origin = pd.Timestamp(date_text)
        available = target.index[target.index <= origin]
        if len(available) < CONTEXT_LENGTH:
            return None, (
                f"Not enough history before {origin.date()}. "
                f"Pick a date after {target.index[CONTEXT_LENGTH].date()}."
            )
        origin = available[-1]
        history = target.loc[:origin]

        quantiles, index = _forecast_one(model_label, history, series, origin)
        actuals = target.loc[(target.index > origin) & (target.index <= index[-1])]

        figure = fan_chart(
            history=history,
            quantiles=quantiles,
            forecast_index=index,
            series=series,
            model_name=model_label,
            actuals=actuals if not actuals.empty else None,
        )
        caption = (
            f"**{model_label}** forecasting {SERIES_LABELS.get(series, series)} for the 21 "
            f"trading days after **{origin.date()}**. The shaded bands are the model's own "
            "uncertainty: the dark band is where it thinks the value lands 80% of the time. "
            "The red dashed line is what actually happened."
        )
        return figure, caption
    except Exception as error:  # noqa: BLE001 - surface the problem, never a blank page
        logger.exception("Live forecast failed")
        return None, f"Could not produce that forecast: {error}"


def run_comparison(series: str, date_text: str):
    """Compare several models' central forecasts from one origin.

    Args:
        series: Which series to forecast.
        date_text: Forecast origin.

    Returns:
        A ``(figure, caption)`` pair.
    """
    try:
        frame = series_data(series)
        target = frame[series].dropna()
        origin = pd.Timestamp(date_text)
        available = target.index[target.index <= origin]
        if len(available) < CONTEXT_LENGTH:
            return None, "Not enough history before that date."
        origin = available[-1]
        history = target.loc[:origin]

        forecasts, index = {}, None
        for label in LIVE_MODELS:
            quantiles, index = _forecast_one(label, history, series, origin)
            forecasts[label] = quantiles

        actuals = target.loc[(target.index > origin) & (target.index <= index[-1])]
        figure = comparison_chart(
            history=history,
            forecasts=forecasts,
            forecast_index=index,
            series=series,
            actuals=actuals if not actuals.empty else None,
        )
        return figure, (
            f"All four models forecasting from **{origin.date()}**. Only their central "
            "guesses are shown — overlaying four uncertainty fans is unreadable."
        )
    except Exception as error:  # noqa: BLE001
        logger.exception("Comparison failed")
        return None, f"Could not produce that comparison: {error}"


def headline_table(series: str, horizon: int) -> pd.DataFrame:
    """Format the headline results for display.

    Args:
        series: Series to show.
        horizon: Forecast horizon.

    Returns:
        A display-ready frame with plain-language column names.
    """
    table = results("headline")
    if table.empty:
        return pd.DataFrame({"note": ["Results are not published yet."]})

    rows = table[(table.series == series) & (table.horizon == horizon)].copy()
    rows = rows.sort_values("skill_wql", ascending=False)
    out = pd.DataFrame(
        {
            "Model": rows["model_id"],
            "What it is": [describe(m).split(".")[0] + "." for m in rows["model_id"]],
            "Better than random walk": [f"{v:+.1%}" for v in rows["skill_wql"]],
            "Typical error (MASE)": [f"{v:.3f}" for v in rows["mase"]],
            "80% interval covers": [f"{v:.0%}" for v in rows["coverage_80"]],
            "Forecast dates": rows["n_origins"],
        }
    )
    return out.reset_index(drop=True)


def build_app() -> gr.Blocks:
    """Assemble the five-tab demo.

    Returns:
        The Gradio application.
    """
    default_series = "spy_logrv"
    default_date = "2024-06-28"

    with gr.Blocks(title="forecast_bench", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Can AI forecast the market better than 1970s statistics?\n"
            "**Short answer: not here — but it needs far less data to try.** "
            f"[Code and full results on GitHub]({REPO_URL})"
        )

        with gr.Tab("1. Try a forecast"):
            gr.Markdown(
                "Pick a date. Each model sees only what was known *on that date* and "
                "forecasts the next 21 trading days. The red dashed line is what actually "
                "happened, so you can judge for yourself."
            )
            with gr.Row():
                series_pick = gr.Dropdown(
                    choices=[
                        ("S&P 500 volatility", "spy_logrv"),
                        ("10-year Treasury yield", "dgs10"),
                    ],
                    value=default_series,
                    label="What to forecast",
                )
                date_pick = gr.Textbox(
                    value=default_date, label="Forecast from this date (YYYY-MM-DD)"
                )
                model_pick = gr.Dropdown(
                    choices=LIVE_MODELS, value=LIVE_MODELS[0], label="Model"
                )
            run_button = gr.Button("Forecast", variant="primary")
            plot = gr.Plot()
            caption = gr.Markdown()
            gr.Markdown("---\n### Or compare all four models at once")
            compare_button = gr.Button("Compare models")
            compare_plot = gr.Plot()
            compare_caption = gr.Markdown()

            run_button.click(
                run_forecast, [series_pick, date_pick, model_pick], [plot, caption]
            )
            compare_button.click(
                run_comparison,
                [series_pick, date_pick],
                [compare_plot, compare_caption],
            )
            demo.load(
                run_forecast, [series_pick, date_pick, model_pick], [plot, caption]
            )

        with gr.Tab("2. Full results"):
            gr.Markdown(
                "Every model, scored through identical code on the same 137 forecast "
                "dates. **Better than random walk** is the headline number: 0% means no "
                "better than assuming tomorrow looks like today.\n\n"
                "*These are the pre-registered headline numbers — Arm A, matched refit "
                "cadence — identical to `docs/benchmark_results.md`.*"
            )
            with gr.Row():
                res_series = gr.Dropdown(
                    choices=[
                        ("S&P 500 volatility", "spy_logrv"),
                        ("10-year Treasury yield", "dgs10"),
                    ],
                    value=default_series,
                    label="Series",
                )
                res_horizon = gr.Radio(
                    choices=[1, 5, 21], value=1, label="Trading days ahead"
                )
            res_table = gr.Dataframe(wrap=True)
            res_plot = gr.Plot()

            def _refresh(series, horizon):
                table = results("headline")
                figure = (
                    skill_bar_chart(table, series, horizon) if not table.empty else None
                )
                return headline_table(series, horizon), figure

            res_series.change(
                _refresh, [res_series, res_horizon], [res_table, res_plot]
            )
            res_horizon.change(
                _refresh, [res_series, res_horizon], [res_table, res_plot]
            )
            demo.load(_refresh, [res_series, res_horizon], [res_table, res_plot])

        with gr.Tab("3. Where each model wins"):
            gr.Markdown(
                "The same models, split by how volatile markets were on the forecast date. "
                "Blue means better than a random walk, red means worse.\n\n"
                "*Calm / normal / stressed are VIX thirds fixed on pre-2015 data and never "
                "recomputed — so no model can look good because the definition moved.*"
            )
            with gr.Row():
                reg_series = gr.Dropdown(
                    choices=[
                        ("S&P 500 volatility", "spy_logrv"),
                        ("10-year Treasury yield", "dgs10"),
                    ],
                    value=default_series,
                    label="Series",
                )
                reg_horizon = gr.Radio(
                    choices=[1, 5, 21], value=1, label="Trading days ahead"
                )
            reg_plot = gr.Plot()
            cov_plot = gr.Plot()

            def _refresh_regime(series, horizon):
                regimes, head = results("regime_stratified"), results("headline")
                return (
                    (
                        regime_heatmap(regimes, series, horizon)
                        if not regimes.empty
                        else None
                    ),
                    (
                        coverage_width_scatter(head, series, horizon)
                        if not head.empty
                        else None
                    ),
                )

            reg_series.change(
                _refresh_regime, [reg_series, reg_horizon], [reg_plot, cov_plot]
            )
            reg_horizon.change(
                _refresh_regime, [reg_series, reg_horizon], [reg_plot, cov_plot]
            )
            demo.load(_refresh_regime, [reg_series, reg_horizon], [reg_plot, cov_plot])

        with gr.Tab("4. How much data do you need?"):
            gr.Markdown(
                "**This is the most interesting chart in the project.** The same models "
                "trained on 1 year, 3 years, 10 years and the full history.\n\n"
                "The pretrained model starts near its ceiling with one year of data. The "
                "from-scratch networks are *worse than a coin flip* at that size. "
                "Pretraining did not buy accuracy here — it bought not needing much data."
            )
            eff_plot = gr.Plot()
            demo.load(
                lambda: (
                    sample_efficiency_curve(results("sample_efficiency"))
                    if not results("sample_efficiency").empty
                    else None
                ),
                None,
                eff_plot,
            )

        with gr.Tab("5. What am I looking at?"):
            gr.Markdown(PROJECT_EXPLANATION)
            gr.Markdown(HEADLINE_FINDING)
            gr.Markdown(
                "### The two things being forecast\n\n"
                + "\n\n".join(SERIES_CARDS.values())
            )
            gr.Markdown("### The models\n\n" + markdown_table())
            gr.Markdown(
                "### What this does **not** show\n\n"
                "- **The foundation models may have seen this data already.** Chronos-2 was "
                "released in October 2025; most of the test period predates it, and its "
                "training data is not public. Untouched-model results on the early period "
                "may not be a fair test at all. This is unfixable, so it is stated instead "
                "of hidden.\n"
                "- **137 forecast dates is a small sample.** Most of these models cannot be "
                "told apart with statistical confidence. The foundation model lost by the "
                "rule set in advance, and is *not* demonstrably worse than what beat it.\n"
                "- **One asset class, two series, one 11-year window.** This is a fact about "
                "this comparison, not about foundation models in general.\n"
                "- **The older Chronos model cannot produce extreme quantiles**, so its "
                "95% and 80% intervals are identical, which costs it on the headline metric.\n"
                "- **The covariate arm is incomplete** — the foundation models never got "
                "covariate support, so that comparison was not run.\n\n"
                f"Full detail: [`docs/limitations.md`]({REPO_URL}/blob/main/docs/limitations.md)"
            )
            gr.Markdown(
                "### Why trust any of this?\n\n"
                f"The [pre-registration]({REPO_URL}/blob/main/PREREGISTRATION.md) was "
                "committed to git *before any model ran*. It states what would count as the "
                "AI model losing. The git timestamp is the evidence that the goalposts did "
                "not move.\n\n"
                "Every model goes through the same backtest code, and a test suite fails the "
                "build if any model can see data from after its forecast date."
            )

    return demo


def warm_up() -> None:
    """Run one forecast at startup so the first visitor does not pay the cold start.

    Loading Chronos-2 takes roughly ten seconds the first time; forecasting afterwards
    takes about one. Paying that once at boot rather than on someone's first click is the
    whole difference between a demo that feels instant and one that looks broken.

    Failures are logged and swallowed: a warm-up that cannot reach the Hub must not stop
    the Space from starting, since every tab handles its own missing data.
    """
    try:
        results("headline")
        run_forecast("spy_logrv", "2024-06-28", LIVE_MODELS[0])
        logger.info("Warm-up complete; first visitor gets a fast response.")
    except Exception as error:  # noqa: BLE001 - never block startup on a warm-up
        logger.warning("Warm-up failed (%s); the Space will still start.", error)


if __name__ == "__main__":
    warm_up()
    build_app().launch()
