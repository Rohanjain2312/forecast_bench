# Model Cards

Plain-language descriptions of every model in the panel.

**Generated from `space/model_cards.py`** — the same source the demo reads, so the interface
and the documentation cannot describe a model differently. Regenerate with
`poetry run python -m scripts.build_results` after editing that file.

## What is being forecast

**Stock market volatility.** How much the S&P 500 moved around on a given day, estimated from its daily high, low, open and close. Volatility has real structure — calm periods follow calm periods — so it is genuinely forecastable.
**The 10-year US Treasury yield.** The interest rate the US government pays to borrow for ten years. It behaves almost like a coin flip from day to day, and is included precisely because it is hard: it is where forecasting claims go to die.

## The panel

| Model | Family | What it does |
|---|---|---|
| **AR1** | Classical statistics | The simplest possible statistical model: next value is a fraction of the current one, plus noise. The standard benchmark in macroeconomic forecasting. |
| **ARIMA** | Classical statistics | The classic statistical forecaster, in use since the 1970s. Learns how strongly recent values predict the next one, and picks its own complexity on each refit. |
| **HAR** | Classical statistics | Built specifically for market volatility. Combines yesterday's, last week's and last month's volatility, reflecting that traders act on all three horizons. |
| **LogHAR** | Classical statistics | The same volatility model, fitted on the logarithm of volatility, which is much better behaved statistically. The strongest classical model in this study. |
| **SARIMAX** | Classical statistics | ARIMA extended to read other series alongside the target, such as the VIX or short-term interest rates. |
| **Chronos2-FineTuned** | Foundation model, adapted to this data | The same pretrained model, then adapted to this specific series using LoRA — a technique that adjusts a small fraction of the model rather than retraining it. |
| **ChronosBolt-FineTuned** | Foundation model, adapted to this data | The earlier Chronos model, adapted to this data the same way. Note it cannot produce the most extreme predictions this study asks for — see Limitations. |
| **Chronos2-ZeroShot** | Foundation model, untouched | Amazon's Chronos-2, used straight out of the box with no training on this data at all. It was pretrained on millions of unrelated time series, the same idea as a language model but for numbers. |
| **ChronosBolt-ZeroShot** | Foundation model, untouched | An earlier, smaller Chronos model, used untouched. Included to see whether the newer generation is actually better. |
| **RandomWalk** | Naive baseline | Predicts that tomorrow looks exactly like today. The baseline everything else is measured against — on financial data it is much harder to beat than it sounds. |
| **SeasonalNaive** | Naive baseline | Predicts that today looks like the same weekday last week. A sanity check: if a model cannot beat this, it has found nothing. |
| **DeepAR-LSTM** | Neural network, trained from scratch | A recurrent neural network of the kind Amazon popularised for forecasting. Like N-BEATS, it starts from nothing and learns only from this series. |
| **N-BEATS** | Neural network, trained from scratch | A deep neural network built for forecasting, trained from scratch on this series alone. Learns everything it knows from the data it is given here. |

## Why each family is here

**Naive baselines** exist to make the others prove something. The random walk is the
reference every skill score is measured against; on financial data it is far harder to beat
than it sounds, which is the point.

**Classical models** are the incumbent. HAR is not optional on a volatility study — it is
*the* benchmark in the realized-volatility literature and the model a referee asks about
first. A study whose classical arm is only ARIMA has a hole a quant interviewer finds in
thirty seconds.

**Neural networks trained from scratch** are the honest comparison point for a pretrained
model. They see only this series, so the gap between them and a foundation model is what
pretraining actually bought.

**Foundation models** appear in two states — untouched and adapted — because the difference
between those two is the one quantity pretraining contamination cannot confound. Both share
the same base weights and differ only in adaptation fitted on our data with our cutoffs.

## The caveat that applies to the foundation models only

Chronos-2 was released in October 2025. Most of the test span predates it, and its
pretraining corpus is not public, so an "untouched" forecast of 2019 volatility may not be
out-of-sample at all. This is unfixable and is reported as a first-class limitation rather
than a footnote. See [`limitations.md`](limitations.md).

The classical models have no equivalent problem: they are fitted from scratch inside every
fold and cannot have seen anything.
