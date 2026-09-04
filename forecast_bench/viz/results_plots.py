"""Charts built from the results tables: skill bars, regime heatmap, sample-efficiency curve.

Every function takes a table produced by ``evaluation/aggregate.py`` and reads the numbers
straight off it. Nothing here recomputes a metric, so a chart cannot disagree with the
table it was drawn from.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Diverging scale for skill scores: red is worse than the baseline, blue is better.
SKILL_SCALE = [[0.0, "#b2182b"], [0.5, "#f7f7f7"], [1.0, "#2166ac"]]


def skill_bar_chart(headline: pd.DataFrame, series: str, horizon: int = 1):
    """Rank models by skill against the random walk at one horizon.

    Args:
        headline: The headline metrics table.
        series: Series to show.
        horizon: Forecast horizon in trading days.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    rows = headline[
        (headline.series == series) & (headline.horizon == horizon)
    ].sort_values("skill_wql")

    colours = ["#b2182b" if v < 0 else "#2166ac" for v in rows["skill_wql"]]
    figure = go.Figure(
        go.Bar(
            x=rows["skill_wql"],
            y=rows["model_id"],
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.1%}" for v in rows["skill_wql"]],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:+.2%} vs random walk<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#333333", line_width=2)
    figure.update_layout(
        title=(
            f"How much better than a random walk? — {series}, "
            f"{horizon} trading day{'s' if horizon > 1 else ''} ahead"
        ),
        xaxis_title="Better than a coin-flip forecast  →",
        yaxis_title="",
        xaxis_tickformat=".0%",
        template="plotly_white",
        height=460,
        margin={"l": 170, "r": 40, "t": 70, "b": 60},
    )
    return figure


def regime_heatmap(regimes: pd.DataFrame, series: str, horizon: int = 1):
    """Skill by model and volatility regime.

    Args:
        regimes: The regime-stratified metrics table.
        series: Series to show.
        horizon: Forecast horizon.

    Returns:
        A Plotly figure.

    Note:
        Regimes are VIX terciles frozen on pre-2015 data. They are never recomputed, so a
        model cannot look good because the definition of "stressed" moved to suit it.
    """
    import plotly.graph_objects as go

    rows = regimes[(regimes.series == series) & (regimes.horizon == horizon)]
    order = ["calm", "normal", "stressed"]
    table = rows.pivot_table(index="model_id", columns="regime", values="skill_wql")
    table = table[[c for c in order if c in table.columns]]

    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=[c.title() for c in table.columns],
            y=table.index,
            colorscale=SKILL_SCALE,
            zmid=0.0,
            text=[
                [f"{v:+.1%}" if pd.notna(v) else "" for v in row]
                for row in table.to_numpy()
            ],
            texttemplate="%{text}",
            colorbar={"title": "Better than<br>random walk"},
            hovertemplate="%{y} in %{x} markets<br>%{z:+.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Where each model wins — {series}, {horizon} day ahead",
        xaxis_title="Market conditions at the time of the forecast",
        yaxis_title="",
        template="plotly_white",
        height=520,
        margin={"l": 170, "r": 40, "t": 70, "b": 60},
    )
    return figure


def sample_efficiency_curve(
    sweep: pd.DataFrame, series: str = "spy_logrv", horizon: int = 1
):
    """Skill against the amount of training data, one line per model.

    Args:
        sweep: The sample-efficiency metrics table.
        series: Series to show.
        horizon: Forecast horizon.

    Returns:
        A Plotly figure.

    Note:
        The most legible chart in the project: one axis is how much data, the other is how
        good, and the shapes tell the story without a caption. A pretrained model that
        starts high and stays flat has bought data efficiency; a from-scratch model that
        starts far below zero has not.
    """
    import plotly.graph_objects as go

    order = ["1y", "3y", "10y", "full"]
    labels = {
        "1y": "1 year",
        "3y": "3 years",
        "10y": "10 years",
        "full": "All 15 years",
    }
    rows = sweep[(sweep.series == series) & (sweep.horizon == horizon)]

    figure = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]
    for position, (model, block) in enumerate(rows.groupby("model_id")):
        indexed = block.set_index("training_window")["skill_wql"]
        present = [w for w in order if w in indexed.index]
        if len(present) < 2:
            continue
        figure.add_trace(
            go.Scatter(
                x=[labels[w] for w in present],
                y=[indexed[w] for w in present],
                mode="lines+markers",
                name=model,
                line={"width": 3, "color": palette[position % len(palette)]},
                marker={"size": 9},
                hovertemplate="%{fullData.name}<br>%{x}: %{y:+.2%}<extra></extra>",
            )
        )

    figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="#333333",
        annotation_text="no better than a random walk",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title="Does pretraining reduce how much data you need?",
        xaxis_title="How much history the model was trained on",
        yaxis_title="Better than a random walk  →",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=480,
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 70, "r": 40, "t": 70, "b": 80},
    )
    return figure


def coverage_width_scatter(headline: pd.DataFrame, series: str, horizon: int = 1):
    """Interval coverage against interval width, with the nominal target marked.

    Args:
        headline: The headline metrics table.
        series: Series to show.
        horizon: Forecast horizon.

    Returns:
        A Plotly figure.

    Note:
        The pair exists precisely so a model cannot buy coverage with uselessly wide
        intervals. Plotting them together makes that visible: the good corner is close to
        the dashed line and far to the left.
    """
    import plotly.graph_objects as go

    rows = headline[(headline.series == series) & (headline.horizon == horizon)]
    figure = go.Figure(
        go.Scatter(
            x=rows["width_80"],
            y=rows["coverage_80"],
            mode="markers+text",
            text=rows["model_id"],
            textposition="top center",
            marker={"size": 12, "color": "#1f77b4"},
            hovertemplate="%{text}<br>covers %{y:.1%} of outcomes<br>width %{x:.2f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="what an 80% interval should cover",
    )
    figure.update_layout(
        title=f"Honest uncertainty? — {series}, {horizon} day ahead",
        xaxis_title="How wide the interval is (narrower is better)",
        yaxis_title="How often the truth fell inside it",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=460,
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return figure
