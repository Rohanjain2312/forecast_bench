"""Fan charts, shared by the repository figures and the Hugging Face Space.

One implementation, so a chart in the demo cannot look different from the same chart in
the docs. Every axis is labelled in words rather than symbols: the demo is read by people
who have never heard of realized variance, and "Predicted volatility (log variance)" costs
nothing that "log sigma^2" buys.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Interval bands drawn on a fan chart, outermost first.
FAN_BANDS = [(0.025, 0.975, "95% range"), (0.1, 0.9, "80% range")]

#: Plain-language axis labels, keyed by series.
AXIS_LABELS = {
    "spy_logrv": "Stock market volatility (log variance)",
    "dgs10": "10-year Treasury yield (percent)",
}

#: Human-readable series names.
SERIES_LABELS = {
    "spy_logrv": "S&P 500 (SPY) volatility",
    "dgs10": "10-year Treasury yield",
}


def fan_chart(
    history: pd.Series,
    quantiles: dict[float, np.ndarray],
    forecast_index: pd.DatetimeIndex,
    series: str = "spy_logrv",
    model_name: str = "Model",
    actuals: pd.Series | None = None,
    history_days: int = 120,
):
    """Draw a forecast fan over recent history.

    Args:
        history: Observed values up to and including the forecast origin.
        quantiles: Mapping of quantile level to a forecast path.
        forecast_index: Dates being forecast.
        series: Series name, used for axis labels.
        model_name: Name shown in the title.
        actuals: What actually happened, drawn dashed when available.
        history_days: How much history to show before the origin.

    Returns:
        A Plotly figure.

    Note:
        The bands are the model's own quantiles, not a normal approximation. A wide fan is
        a model saying it does not know — which is information, and the reason coverage
        and width are always reported as a pair rather than coverage alone.
    """
    import plotly.graph_objects as go

    recent = history.iloc[-history_days:]
    figure = go.Figure()

    for low, high, label in FAN_BANDS:
        if low not in quantiles or high not in quantiles:
            continue
        figure.add_trace(
            go.Scatter(
                x=list(forecast_index) + list(forecast_index[::-1]),
                y=list(quantiles[high]) + list(quantiles[low][::-1]),
                fill="toself",
                fillcolor=(
                    "rgba(31,119,180,0.30)"
                    if label.startswith("80")
                    else "rgba(31,119,180,0.15)"
                ),
                line={"color": "rgba(0,0,0,0)"},
                name=label,
                hoverinfo="skip",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.to_numpy(),
            mode="lines",
            name="What actually happened (past)",
            line={"color": "#333333", "width": 2},
        )
    )

    if 0.5 in quantiles:
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=quantiles[0.5],
                mode="lines",
                name=f"{model_name} best guess",
                line={"color": "#1f77b4", "width": 3},
            )
        )

    if actuals is not None and not actuals.empty:
        figure.add_trace(
            go.Scatter(
                x=actuals.index,
                y=actuals.to_numpy(),
                mode="lines",
                name="What actually happened (after)",
                line={"color": "#d62728", "width": 2, "dash": "dash"},
            )
        )

    figure.add_vline(
        x=recent.index[-1],
        line_dash="dot",
        line_color="#888888",
        annotation_text="forecast starts here",
        annotation_position="top left",
    )

    figure.update_layout(
        title=f"{model_name} — {SERIES_LABELS.get(series, series)}",
        xaxis_title="Date",
        yaxis_title=AXIS_LABELS.get(series, series),
        hovermode="x unified",
        template="plotly_white",
        height=460,
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )
    return figure


def comparison_chart(
    history: pd.Series,
    forecasts: dict[str, dict[float, np.ndarray]],
    forecast_index: pd.DatetimeIndex,
    series: str = "spy_logrv",
    actuals: pd.Series | None = None,
    history_days: int = 120,
):
    """Draw several models' median forecasts on one axis.

    Args:
        history: Observed values up to the forecast origin.
        forecasts: Mapping of model name to its quantile paths.
        forecast_index: Dates being forecast.
        series: Series name, used for axis labels.
        actuals: What actually happened, drawn dashed when available.
        history_days: How much history to show.

    Returns:
        A Plotly figure comparing the models' central forecasts.

    Note:
        Medians only. Overlaying several fans produces a chart nobody can read; the fan
        belongs on the single-model view where its width is the point.
    """
    import plotly.graph_objects as go

    recent = history.iloc[-history_days:]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.to_numpy(),
            mode="lines",
            name="What actually happened (past)",
            line={"color": "#333333", "width": 2},
        )
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for position, (name, quantiles) in enumerate(forecasts.items()):
        if 0.5 not in quantiles:
            continue
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=quantiles[0.5],
                mode="lines",
                name=name,
                line={"color": palette[position % len(palette)], "width": 2.5},
            )
        )

    if actuals is not None and not actuals.empty:
        figure.add_trace(
            go.Scatter(
                x=actuals.index,
                y=actuals.to_numpy(),
                mode="lines",
                name="What actually happened (after)",
                line={"color": "#d62728", "width": 2.5, "dash": "dash"},
            )
        )

    figure.add_vline(x=recent.index[-1], line_dash="dot", line_color="#888888")
    figure.update_layout(
        title=f"Model comparison — {SERIES_LABELS.get(series, series)}",
        xaxis_title="Date",
        yaxis_title=AXIS_LABELS.get(series, series),
        hovermode="x unified",
        template="plotly_white",
        height=460,
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )
    return figure
