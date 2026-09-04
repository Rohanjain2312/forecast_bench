# forecast_bench

[![tests](https://github.com/Rohanjain2312/forecast_bench/actions/workflows/tests.yml/badge.svg)](https://github.com/Rohanjain2312/forecast_bench/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/rohanjain2312/forecastbench-demo)

**A fair fight between classical statistical forecasting and AI foundation models on real
financial data — with the losing condition written down and committed to git before any
model ran.**

**[▶ Try the live demo](https://huggingface.co/spaces/rohanjain2312/forecastbench-demo)**

---

## The result

**The fine-tuned foundation model lost, by the rule set in advance.** It could not beat a
random walk on the 10-year Treasury yield, and never reached statistical significance
against the best classical model on either series.

**But it needs far less data.** Given one year of training data instead of fifteen, it kept
**85%** of its full accuracy. The from-scratch neural networks, on the same year, were *worse
than a coin flip*.

Both are reported. Reporting only the one that fits a narrative is the failure this project
exists to avoid.

## What this is

Forecasting is the problem of guessing what a number will be next week given what it has
been so far. For decades the standard tools were statistical models like ARIMA. Recently,
large pretrained models — the same idea as ChatGPT, but trained on millions of time series
instead of text — have started claiming they can forecast anything without being trained on
it. Nobody agrees whether that holds up on financial data, which is famously noisy.

I picked two financial series with **opposite** expected answers:

| | Primary | Contrast |
|---|---|---|
| **Target** | SPY log realized variance (Garman-Klass, daily OHLC) | 10-year Treasury yield, in levels |
| **Why** | Volatility has real structure — it should be forecastable | Near-unit-root; a random walk is brutal to beat |
| **Expected** | Classical HAR wins | Nobody beats the random walk |

Eleven models forecast the same 137 dates through **identical code**, scored on identical
metrics. A test suite fails the build if any model can see data from after its forecast date.

## Headline results

Arm A (univariate), matched refit cadence, 1 trading day ahead. **Skill score** is the
improvement in weighted quantile loss over a random walk: 0% means no better than assuming
tomorrow looks like today.

### SPY realized volatility — 137 forecast dates

| Model | Skill vs random walk | Family |
|---|---:|---|
| DeepAR-LSTM | **+16.9%** | Neural, from scratch |
| ARIMA | +16.8% | Classical |
| LogHAR | +16.6% | Classical |
| Chronos-2 fine-tuned | +15.3% | Foundation, adapted |
| Chronos-2 zero-shot | +15.1% | Foundation, untouched |
| Chronos-Bolt zero-shot | +12.7% | Foundation, untouched |
| Chronos-Bolt fine-tuned | +11.8% | Foundation, adapted |
| N-BEATS | +5.8% | Neural, from scratch |
| Random walk | 0.0% | Baseline |
| HAR | −4.1% | Classical |
| Seasonal naive | −21.9% | Baseline |

### 10-year Treasury yield — 136 forecast dates

Every model sits within ±1% of the random walk except the two that are much worse. This is
the registered prediction holding: on a near-unit-root series, there is nothing to find.

| Model | Skill vs random walk |
|---|---:|
| Chronos-2 zero-shot | +0.8% |
| AR(1) | +0.2% |
| Random walk | 0.0% |
| Chronos-2 fine-tuned | **−0.2%** |
| ARIMA | −0.5% |

**Full tables at every horizon:** [`docs/benchmark_results.md`](docs/benchmark_results.md),
generated from the data rather than typed by hand.

## The pre-registered verdict

[`PREREGISTRATION.md`](PREREGISTRATION.md) was committed **before any model code existed**.
The git timestamp is the evidence. It defined losing as:

> failing to beat the random walk on WQL skill at h=1 on either series, **or** failing to
> reach Diebold-Mariano significance against the best classical model at any horizon.

Both clauses triggered independently:

| Clause | Outcome |
|---|---|
| (a) Beat random walk at h=1 | `DGS10`: **−0.0016 — FAIL** |
| (b) DM significance vs best classical | 6 tests, p = 0.23–0.93, **all favouring classical — FAIL** |

Four of five registered predictions held. The one that failed: the gap to the best classical
model was predicted to *narrow* with horizon. It widened.

Reproduce it: `poetry run python -m scripts.evaluate_preregistration`

## The finding that points the other way

The most transferable result here is not the headline. Skill at h=1 by training-set size:

| Model | 1 year | 3 years | 10 years | Full (15y) |
|---|---:|---:|---:|---:|
| **Chronos-2 fine-tuned** | **+13.0%** | +15.1% | +15.2% | +15.3% |
| N-BEATS | −53.9% | −21.7% | +3.7% | +5.8% |
| DeepAR-LSTM | −516.0% | −18.4% | +12.7% | +16.9% |

The pretrained model is at 85% of its ceiling with one year of data. The from-scratch models
are catastrophic there. **Pretraining did not buy accuracy here — it bought not needing much
data**, which in practice is often the constraint that actually binds.

## How it works

- **Expanding-origin walk-forward**, training from 2000, testing 2015-01-01 → 2026-06-30.
- **Stride = 21 = max horizon**, so forecast windows never overlap — which is what makes the
  Diebold-Mariano test defensible. Most published backtests quietly overlap.
- **Horizons {1, 5, 21}** read off steps 1, 5 and 21 of a *single* 21-step path, so all three
  share identical folds and identical model fits.
- **Two refit cadences**, both reported. Matched (all models refit at annual block
  boundaries) is the headline; native (classical refit every fold) is secondary.
- **Everything probabilistic.** Every model emits 11 quantiles. Coverage and width are always
  reported as a pair, because a model can buy coverage with uselessly wide intervals.

### Leakage safety, made executable

Four guards, one of which fails the build:

1. **Only non-revised daily FRED series** may enter a model. Revised series are indexed by
   reference period rather than release date, so reading `CPIAUCSL` at a forecast origin
   reads the future. Restricting the inputs makes the point-in-time claim true *by
   construction*. See [`docs/data_protocol.md`](docs/data_protocol.md).
2. **Everything fitted is fitted inside the fold** — scalers, ARIMA orders, MASE
   denominators, residual quantiles.
3. **Regime thresholds frozen** on pre-2015 data and asserted at import.
4. **A canary test** injects a column containing the future target, confirms error collapses
   to zero, and confirms the guard fires. A guard nobody has watched fail is a guard nobody
   knows works.

## Installation

```bash
git clone git@github.com:Rohanjain2312/forecast_bench.git
cd forecast_bench
poetry install
cp .env.example .env      # then add your FRED key and Hugging Face token
```

## Quick start

```bash
# 1. Verify every credential and measure Chronos-2's CPU latency
poetry run python -m scripts.verify_setup

# 2. Build the two target series
poetry run python -m scripts.fetch_data --config spy_logrv
poetry run python -m scripts.fetch_data --config dgs10

# 3. Run the headline backtest
poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A \
    --with-foundation --with-finetuned

# 4. Build every results table
poetry run python -m scripts.build_results

# 5. Apply the pre-registered decision rules
poetry run python -m scripts.evaluate_preregistration
```

More detail in [`docs/quickstart.md`](docs/quickstart.md).

## Architecture

```
FRED + Yahoo ──► data/ ──► targets & covariates
                             │
                             ▼
                    backtest/splitter.py          137 non-overlapping folds
                             │
                             ▼
      ┌──────────── backtest/runner.py ────────────┐   one loop, no branching
      │                                            │   on model identity
      ▼            ▼            ▼            ▼     ▼
   naive      classical      neural      foundation (zero-shot & fine-tuned)
      └──────────────────┬─────────────────────────┘
                         ▼
              one tidy parquet  (backtest/writer.py)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     evaluation/metrics.py     viz/ ──► HF Space
     evaluation/stats.py            └─► docs/benchmark_results.md
```

**Every model implements one interface** — `backtest/protocol.py::Forecaster`. ARIMA and
Chronos-2 are indistinguishable to the runner. The harness never calls
`darts.historical_forecasts`: outsourcing the backtest would mean the foundation models and
the classical models no longer provably traverse identical code, which is the one claim the
harness exists to support.

## Project structure

| Path | What lives there |
|---|---|
| `forecast_bench/backtest/` | The harness: protocol, splitter, cadence, runner, writer |
| `forecast_bench/data/` | FRED and Yahoo clients, target construction, the allowlist |
| `forecast_bench/models/` | The panel — naive, classical, neural, foundation |
| `forecast_bench/evaluation/` | Metrics, DM test, Model Confidence Set, regimes |
| `forecast_bench/viz/` | Charts, shared by the docs and the Space |
| `scripts/` | Command-line entry points |
| `notebooks/` | Colab notebooks for the GPU work |
| `space/` | The demo, mirrored to Hugging Face |
| `tests/` | The test suite, including the leakage canary |

## Documentation

| Document | What it covers |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | The decision rules, committed before any result |
| [`docs/benchmark_results.md`](docs/benchmark_results.md) | All six results tables, full precision |
| [`docs/limitations.md`](docs/limitations.md) | What this does **not** establish |
| [`docs/data_protocol.md`](docs/data_protocol.md) | Point-in-time rules, and the CPI bug that motivated them |
| [`docs/methodology.md`](docs/methodology.md) | Fold scheme, cadence, metrics, DM assumptions |
| [`docs/architecture.md`](docs/architecture.md) | How the pieces fit |
| [`docs/model_cards.md`](docs/model_cards.md) | Plain-language description of every model |
| [`docs/planning/DECISIONS.md`](docs/planning/DECISIONS.md) | Why every design choice is what it is |

## Requirements at a glance

Python 3.11, Poetry. Core: `pandas`, `numpy`, `statsmodels`, `darts`, `chronos-forecasting`,
`peft`, `arch`. The only GPU work is LoRA fine-tuning and neural training, both done in Colab
notebooks that import from this package rather than reimplementing it.

## Honest caveats

Chronos-2 was released in October 2025 and most of the test span predates it, so zero-shot
results on the early period may not be genuinely out-of-sample — the pretraining corpus is
not inspectable and this cannot be fixed, only bounded. At 137 forecast dates most of these
models are not statistically distinguishable; the fine-tuned Chronos-2 survives in the Model
Confidence Set at every series and horizon, meaning it lost by the pre-registered rule while
not being *demonstrably worse* than what beat it. Full list:
[`docs/limitations.md`](docs/limitations.md).

## License

MIT — see [`LICENSE`](LICENSE).
