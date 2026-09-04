"""Fan charts, shared by the repository figures and the Hugging Face Space.

One implementation, so a chart in the demo cannot look different from the same chart in
the docs. Every axis is labelled in words rather than symbols: the demo is read by people
who have never heard of realized variance, and "Predicted volatility (log variance)" costs
nothing that "log sigma^2" buys.
"""

import logging
from typing import Any

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


def annualised_vol_ticks(low: float, high: float) -> tuple[list[float], list[str]]:
    """Y-axis ticks for log realized variance, annotated with what they mean.

    Args:
        low: Lowest value on the axis.
        high: Highest value on the axis.

    Returns:
        A ``(positions, labels)`` pair.

    Note:
        "log variance = -11" means nothing to a first-time reader; "about 7% a year" does.
        Converting via ``sqrt(exp(v)) * sqrt(252)`` puts a familiar number beside the one
        the models actually forecast, without changing the scale being plotted.
    """
    positions = [v for v in range(int(np.floor(low)), int(np.ceil(high)) + 1)]
    labels = [
        f"{v}<br><span style='font-size:0.8em;color:#777'>"
        f"≈{np.sqrt(np.exp(v)) * np.sqrt(252) * 100:.0f}% a year</span>"
        for v in positions
    ]
    return positions, labels


def fan_chart(
    history: pd.Series,
    quantiles: dict[float, np.ndarray],
    forecast_index: pd.DatetimeIndex,
    series: str = "spy_logrv",
    model_name: str = "Model",
    actuals: pd.Series | None = None,
    history_days: int = 63,
    smooth_window: int = 5,
) -> Any:
    """Draw a forecast fan over recent history.

    Args:
        history: Observed values up to and including the forecast origin.
        quantiles: Mapping of quantile level to a forecast path.
        forecast_index: Dates being forecast.
        series: Series name, used for axis labels.
        model_name: Name shown in the title.
        actuals: What actually happened, drawn dashed when available.
        history_days: How much history to show. Three times the horizon by default, so
            the forecast occupies about a quarter of the width rather than a sliver.
        smooth_window: Trading days in the readability average. Zero disables it.

    Returns:
        A Plotly figure.

    Note:
        Three deliberate choices, all about a first-time reader:

        The forecast region is **shaded**, so the eye finds the prediction before reading
        the legend. Daily realized variance is extremely spiky, so the raw series is drawn
        faint and a short moving average is drawn over it — the average is a *reading aid*
        and is labelled as one; the models forecast the raw series, not the smoothed one.
        And the bands are the model's own quantiles, never a normal approximation: a wide
        fan is a model saying it does not know, which is information.
    """
    import plotly.graph_objects as go

    recent = history.iloc[-history_days:]
    figure = go.Figure()

    # Shade the forecast region first so everything else draws on top of it.
    figure.add_vrect(
        x0=recent.index[-1],
        x1=forecast_index[-1],
        fillcolor="#1f77b4",
        opacity=0.06,
        line_width=0,
        layer="below",
        annotation_text="  forecast",
        annotation_position="top left",
        annotation={"font": {"size": 13, "color": "#1f77b4"}},
    )

    for low, high, label in FAN_BANDS:
        if low not in quantiles or high not in quantiles:
            continue
        wide = label.startswith("95")
        figure.add_trace(
            go.Scatter(
                x=list(forecast_index) + list(forecast_index[::-1]),
                y=list(quantiles[high]) + list(quantiles[low][::-1]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.14)" if wide else "rgba(31,119,180,0.30)",
                line={"color": "rgba(0,0,0,0)"},
                name=f"{label} the model expects",
                hoverinfo="skip",
            )
        )

    # Raw history, kept faint: it is the truth, but it is too noisy to read directly.
    figure.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.to_numpy(),
            mode="lines",
            name="Actual, day by day",
            line={"color": "#9aa5b1", "width": 1},
            opacity=0.75,
        )
    )

    if smooth_window > 1 and len(recent) > smooth_window:
        smoothed = recent.rolling(smooth_window, min_periods=1).mean()
        figure.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.to_numpy(),
                mode="lines",
                name=f"Actual, {smooth_window}-day average",
                line={"color": "#1a1a1a", "width": 2.5},
            )
        )

    if 0.5 in quantiles:
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=quantiles[0.5],
                mode="lines",
                name=f"{model_name} forecast",
                line={"color": "#1f77b4", "width": 3.5},
            )
        )

    if actuals is not None and not actuals.empty:
        figure.add_trace(
            go.Scatter(
                x=actuals.index,
                y=actuals.to_numpy(),
                mode="lines+markers",
                name="What actually happened",
                line={"color": "#d62728", "width": 2, "dash": "dot"},
                marker={"size": 4},
            )
        )

    figure.add_vline(x=recent.index[-1], line_dash="dot", line_color="#555555")

    axis = {"title": AXIS_LABELS.get(series, series)}
    if series == "spy_logrv":
        values = list(recent.to_numpy())
        for path in quantiles.values():
            values.extend(path)
        positions, labels = annualised_vol_ticks(min(values), max(values))
        axis.update(tickmode="array", tickvals=positions, ticktext=labels)

    figure.update_layout(
        title={
            "text": f"{model_name} — {SERIES_LABELS.get(series, series)}",
            "font": {"size": 18},
        },
        xaxis_title="Date",
        yaxis=axis,
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend={"orientation": "h", "y": -0.18, "font": {"size": 12}},
        margin={"l": 90, "r": 30, "t": 70, "b": 90},
    )
    return figure


def comparison_chart(
    history: pd.Series,
    forecasts: dict[str, dict[float, np.ndarray]],
    forecast_index: pd.DatetimeIndex,
    series: str = "spy_logrv",
    actuals: pd.Series | None = None,
    history_days: int = 63,
) -> Any:
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
    figure.add_vrect(
        x0=recent.index[-1],
        x1=forecast_index[-1],
        fillcolor="#1f77b4",
        opacity=0.06,
        line_width=0,
        layer="below",
        annotation_text="  forecast",
        annotation_position="top left",
        annotation={"font": {"size": 13, "color": "#1f77b4"}},
    )
    figure.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.to_numpy(),
            mode="lines",
            name="Actual, day by day",
            line={"color": "#9aa5b1", "width": 1},
            opacity=0.75,
        )
    )
    smoothed = recent.rolling(5, min_periods=1).mean()
    figure.add_trace(
        go.Scatter(
            x=smoothed.index,
            y=smoothed.to_numpy(),
            mode="lines",
            name="Actual, 5-day average",
            line={"color": "#1a1a1a", "width": 2.5},
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

    figure.add_vline(x=recent.index[-1], line_dash="dot", line_color="#555555")

    axis = {"title": AXIS_LABELS.get(series, series)}
    if series == "spy_logrv":
        values = list(recent.to_numpy())
        for quantiles in forecasts.values():
            if 0.5 in quantiles:
                values.extend(quantiles[0.5])
        positions, labels = annualised_vol_ticks(min(values), max(values))
        axis.update(tickmode="array", tickvals=positions, ticktext=labels)

    figure.update_layout(
        title={
            "text": f"All models compared — {SERIES_LABELS.get(series, series)}",
            "font": {"size": 18},
        },
        xaxis_title="Date",
        yaxis=axis,
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend={"orientation": "h", "y": -0.18, "font": {"size": 12}},
        margin={"l": 90, "r": 30, "t": 70, "b": 90},
    )
    return figure
