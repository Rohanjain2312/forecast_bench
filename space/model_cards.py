"""Plain-language descriptions of every model, and the project's own explanation.

Imported by ``space/app.py`` and by the documentation, so the demo and the repository
cannot describe the same model differently.

The writing constraint: a reader who has never heard of ARIMA should finish each gloss
knowing what the model actually does and why it is in the panel. No formulas.
"""

#: One-sentence gloss per model, keyed by the results-table id.
MODEL_CARDS: dict[str, str] = {
    "RandomWalk": (
        "Predicts that tomorrow looks exactly like today. The baseline everything else "
        "is measured against — on financial data it is much harder to beat than it sounds."
    ),
    "SeasonalNaive": (
        "Predicts that today looks like the same weekday last week. A sanity check: if a "
        "model cannot beat this, it has found nothing."
    ),
    "ARIMA": (
        "The classic statistical forecaster, in use since the 1970s. Learns how strongly "
        "recent values predict the next one, and picks its own complexity on each "
        "refit."
    ),
    "AR1": (
        "The simplest possible statistical model: next value is a fraction of the current "
        "one, plus noise. The standard benchmark in macroeconomic forecasting."
    ),
    "HAR": (
        "Built specifically for market volatility. Combines yesterday's, last week's and "
        "last month's volatility, reflecting that traders act on all three horizons."
    ),
    "LogHAR": (
        "The same volatility model, fitted on the logarithm of volatility, which is much "
        "better behaved statistically. The strongest classical model in this study."
    ),
    "SARIMAX": (
        "ARIMA extended to read other series alongside the target, such as the VIX or "
        "short-term interest rates."
    ),
    "N-BEATS": (
        "A deep neural network built for forecasting, trained from scratch on this series "
        "alone. Learns everything it knows from the data it is given here."
    ),
    "DeepAR-LSTM": (
        "A recurrent neural network of the kind Amazon popularised for forecasting. Like "
        "N-BEATS, it starts from nothing and learns only from this series."
    ),
    "Chronos2-ZeroShot": (
        "Amazon's Chronos-2, used straight out of the box with no training on this data at "
        "all. It was pretrained on millions of unrelated time series, the same idea as a "
        "language model but for numbers."
    ),
    "Chronos2-FineTuned": (
        "The same pretrained model, then adapted to this specific series using LoRA — a "
        "technique that adjusts a small fraction of the model rather than retraining it."
    ),
    "ChronosBolt-ZeroShot": (
        "An earlier, smaller Chronos model, used untouched. Included to see whether the "
        "newer generation is actually better."
    ),
    "ChronosBolt-FineTuned": (
        "The earlier Chronos model, adapted to this data the same way. Note it cannot "
        "produce the most extreme predictions this study asks for — see Limitations."
    ),
}

#: Model families, for grouping in the interface.
MODEL_FAMILIES: dict[str, str] = {
    "RandomWalk": "Naive baseline",
    "SeasonalNaive": "Naive baseline",
    "ARIMA": "Classical statistics",
    "AR1": "Classical statistics",
    "HAR": "Classical statistics",
    "LogHAR": "Classical statistics",
    "SARIMAX": "Classical statistics",
    "N-BEATS": "Neural network, trained from scratch",
    "DeepAR-LSTM": "Neural network, trained from scratch",
    "Chronos2-ZeroShot": "Foundation model, untouched",
    "ChronosBolt-ZeroShot": "Foundation model, untouched",
    "Chronos2-FineTuned": "Foundation model, adapted to this data",
    "ChronosBolt-FineTuned": "Foundation model, adapted to this data",
}

#: What the two forecast targets are, in plain language.
SERIES_CARDS: dict[str, str] = {
    "spy_logrv": (
        "**Stock market volatility.** How much the S&P 500 moved around on a given day, "
        "estimated from its daily high, low, open and close. Volatility has real structure "
        "— calm periods follow calm periods — so it is genuinely forecastable."
    ),
    "dgs10": (
        "**The 10-year US Treasury yield.** The interest rate the US government pays to "
        "borrow for ten years. It behaves almost like a coin flip from day to day, and is "
        "included precisely because it is hard: it is where forecasting claims go to die."
    ),
}

#: The sixty-second explanation, used on the landing tab and in the README.
PROJECT_EXPLANATION = """
### What is this?

A fair fight between old-school statistical forecasting and new AI "foundation" models, on
real financial data, with the result reported honestly — including the parts where the AI
lost.

**The question.** Recently, large pretrained models — the same idea as ChatGPT, but trained
on millions of number sequences instead of text — have started claiming they can forecast
anything without being trained on it. Nobody agrees whether that holds up on financial data,
which is famously noisy.

**What I did.** I picked two financial series with opposite characteristics: stock-market
volatility, which has real structure, and the 10-year Treasury yield, which behaves almost
like a coin flip. Eleven models forecast the same 137 dates through identical code and were
scored on identical metrics.

**The part I care about most.** Before running anything, I wrote down what would count as
the AI model *losing* and committed it to git, so I could not move the goalposts afterwards.
"""

#: The headline finding. Kept here so the demo and the docs state it identically.
HEADLINE_FINDING = """
### What I found

**The fine-tuned foundation model lost, by the rule I set in advance.**

It could not beat a simple random walk on the Treasury yield, and it never achieved
statistical significance against the best classical model on either series. The strongest
model on stock-market volatility was LogHAR — a statistical model from 2009 that fits in
about forty lines of code.

**But the picture is not one-sided, and reporting only the headline would be dishonest.**

Given just *one year* of training data, the fine-tuned foundation model kept about **85% of
its full-strength accuracy**. The from-scratch neural networks, on the same year, performed
*worse than a coin flip*. Pretraining did not buy accuracy here — it bought the ability to
work with far less data, which in practice is often what you actually have.

There is also a statistical caveat that cuts against a clean story: across 137 forecast
dates, most of these models are not distinguishable from one another with confidence. The
foundation model lost by the pre-registered rule and is *not* demonstrably worse than what
beat it.
"""


def describe(model_id: str) -> str:
    """Return the plain-language gloss for a model.

    Args:
        model_id: Results-table identifier.

    Returns:
        A one-sentence description, or a fallback naming the missing entry.
    """
    return MODEL_CARDS.get(model_id, f"No description recorded for {model_id}.")


def family(model_id: str) -> str:
    """Return the model's family label.

    Args:
        model_id: Results-table identifier.

    Returns:
        The family name, or ``"Other"``.
    """
    return MODEL_FAMILIES.get(model_id, "Other")


def markdown_table() -> str:
    """Render every model card as a markdown table, grouped by family.

    Returns:
        A markdown string used by the demo's explanation tab and by the docs.
    """
    lines = ["| Model | Family | What it does |", "|---|---|---|"]
    for model_id in sorted(MODEL_CARDS, key=lambda m: (family(m), m)):
        lines.append(f"| **{model_id}** | {family(model_id)} | {describe(model_id)} |")
    return "\n".join(lines)
