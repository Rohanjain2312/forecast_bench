# forecast_bench — Implementation Plan

Sections 4–7 of the brief, expanded into a build plan. No timeline, no week counts — the
ordering below is a dependency order, not a schedule.

Read `DECISIONS.md` first for the *why* behind any choice here.

- **Repo:** `https://github.com/Rohanjain2312/forecast_bench.git`
- **Local:** `/Users/rohanjain/Desktop/Projects/forecast_bench`
- **HF model:** `https://huggingface.co/rohanjain2312/forecastbench-chronos`
- **HF dataset:** `https://huggingface.co/datasets/rohanjain2312/forecastbench-data`
- **HF Space:** `https://huggingface.co/spaces/rohanjain2312/forecastbench-demo`

---

## 0. The recruiter-facing narrative

The brief says the demo is the primary artifact and that someone should understand the
project in about two minutes. That only works if the *explanation* is designed, not
improvised. Here it is, in the words that go on the Space landing tab, in the README's
first paragraph, and out of your mouth in an interview.

> **The one-liner.** I built a fair fight between old-school statistical forecasting
> models and new AI "foundation" models, on real financial data, and reported who actually
> won — including the parts where the AI lost.
>
> **The setup, in four sentences.** Forecasting is the problem of guessing what a number
> will be next week given what it has been so far. For decades the standard tools were
> statistical models like ARIMA, which are fitted to one series at a time. Recently, large
> pretrained models — the same idea as ChatGPT, but trained on millions of time series
> instead of text — have started claiming they can forecast anything without being trained
> on it. Nobody agrees on whether that actually holds up on financial data, which is
> famously noisy.
>
> **What I did.** I picked two financial series with opposite characteristics: stock-market
> volatility, which has real structure, and the 10-year Treasury yield, which behaves
> almost like a coin flip. I ran seven models on both — naive baselines, classical
> statistical models, two neural networks, and a foundation model both untouched and
> fine-tuned on my own data. Every model went through the exact same evaluation code,
> forecasting the same 137 dates, scored on the same metrics.
>
> **What I found.** *(filled in after results, whatever they are)*
>
> **The part I care about most.** Before I ran anything, I wrote down what would count as
> the AI model losing, and committed it to git so I couldn't move the goalposts afterwards.
> I also found and documented a way this kind of study can silently cheat — the foundation
> models may have already seen this data during their own pretraining — and built a
> separate evaluation that avoids it.

Every claim in there is checkable against the repo. That is the point.

---

## 1. Environment split — what runs where

| Workload | Where | Why |
|---|---|---|
| Data pull, cleaning, target construction | Local (Mac, Apple Silicon) | No GPU need; `pandas` only |
| Backtest harness, fold generation | Local | Pure orchestration |
| Naive, ARIMA, SARIMA, SARIMAX, HAR, AR(1) | Local | CPU-bound, minutes total |
| Chronos-2 / Bolt **zero-shot inference** | Local | 120M model, official CPU support; MPS if it helps |
| Chronos-2 / Bolt **LoRA fine-tuning** | Colab H100 | The only genuine GPU requirement |
| N-BEATS, DeepAR-class **training** | Colab H100 | Small models, but 4 sample-efficiency slices × 12 blocks × 2 series adds up |
| TimesFM 2.5 zero-shot (stretch) | Colab | Separate toolchain, keep it off the local env |
| Metrics, DM tests, MCS, plots | Local | Pure numpy/scipy |
| Writeup | Local | |
| Demo | HF Space, CPU Basic | See D12 |

**The rule that keeps this sane:** Colab notebooks never contain modelling logic. They
`pip install` the package from GitHub, import from `forecast_bench`, run a function, and
push the artifact to the Hub. If a Colab cell contains a `for` loop over folds, it belongs
in `forecast_bench/` instead. This is what stops the "the notebook and the repo disagree"
failure mode that kills most benchmark projects.

**Checkpoint-resumability** (Section 12 risk): every Colab training entry point takes a
`--resume-from` argument, writes a checkpoint to the Hub after each block, and on startup
checks the Hub for the latest checkpoint before starting from scratch. A dropped session
costs one block, not the run. This matters less with H100 access than with free T4, but it
is cheap insurance and it is the correct pattern to show in a portfolio repo.

---

## 2. Data pipeline (brief Section 5)

### 2.1 Sources and the point-in-time protocol

| Series | Source | Revised? | Role |
|---|---|---|---|
| `SPY` OHLC | yfinance | No | Input to RV target |
| `DGS10` | FRED | No | Target (rates track) + covariate |
| `DGS3MO`, `T10Y2Y`, `DFF` | FRED | No | Covariates (Arm B) |
| `VIXCLS` | FRED | No | Covariate + regime variable |
| `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `GS10`, `GS3M` | FRED | **Yes** | **Excluded from modelling** |

The exclusion list is the point-in-time protocol. Rather than building ALFRED vintage
machinery (which does not exist in your prior repos — see `DECISIONS.md` §0.3), the study
restricts itself to daily market-observed series that FRED does not revise. The claim
"every model saw only data available at the time" then holds by construction.

`docs/data_protocol.md` explains this, and explains the CPI release-lag bug found in
`market-regime-transformer-codex` as the motivating example. That section is a genuine
differentiator: showing you can find leakage is worth more than asserting you avoided it.

### 2.2 Target construction

**SPY log realized variance (primary target).** Garman-Klass estimator from daily OHLC:

```
sigma2_GK = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
target    = ln(sigma2_GK)
```

Guards: `sigma2_GK` can go non-positive on low-range days; floor it at the 0.1th percentile
of the training window before taking logs, and log the number of flooring events. Parkinson
(`(ln(H/L))^2 / (4*ln 2)`) is implemented as a fallback and a cross-check.

**`DGS10` (contrast target).** Used as-is, in levels, in percent. Deliberately *not*
differenced: the honest finding on a near-unit-root series is that the random walk is hard
to beat, and differencing away the persistence would hide the thing the study is meant to
measure. ARIMA is allowed to select its own `d` per fold; that is the model's decision, not
ours.

### 2.3 Storage

```
data/
├── raw/
│   ├── yfinance_SPY_ohlc.parquet          # + .meta.json with fetch timestamp + checksum
│   └── fred_{SERIES_ID}.parquet
└── processed/
    ├── spy_logrv.parquet                   # target + covariates, business-day indexed
    ├── dgs10.parquet
    └── regimes.parquet                     # frozen VIX tercile labels
```

Parquet over CSV: preserves dtypes and datetime index without parse gymnastics, and the
files are small enough that the free HF dataset tier is a non-issue. Each raw pull carries
a sidecar `.meta.json` with fetch timestamp, source, and content checksum — the same
caching pattern already in `market-regime-transformer-codex/src/data_loader.py`, which is
the one piece of that project worth carrying forward.

`data/` is gitignored. `forecastbench-data` on the Hub is the durable copy, pushed by
`scripts/push_artifacts.py` once the processed series stabilise.

---

## 3. Backtest harness (brief Section 7 + Section 8 items 5, 6, 10)

This is the core of the repo and the thing a reviewer reads first. It gets the most care.

### 3.1 The protocol

`forecast_bench/backtest/protocol.py` defines two things and nothing else:

```python
@dataclass(frozen=True)
class QuantileForecast:
    """A model's h-step-ahead quantile forecast from one origin.

    Attributes:
        origin: The last timestamp the model was allowed to see.
        index: Forecast timestamps, length h.
        quantiles: Mapping of quantile level -> array of length h.
        model_id: Stable identifier used as the results-table key.
    """

class Forecaster(Protocol):
    """Every model in the study implements exactly this.

    fit() may only read data at or before `origin`. Implementations must not
    close over any object fitted outside the current fold — this is what
    tests/test_no_leakage.py checks.
    """
    def fit(self, train: pd.DataFrame, origin: pd.Timestamp) -> None: ...
    def predict(self, horizon: int) -> QuantileForecast: ...
```

Seven model families, one interface. ARIMA and Chronos-2 are indistinguishable to the
runner. That is the whole design.

### 3.2 Fold generation — `splitter.py`

```python
def expanding_origin_folds(
    index: pd.DatetimeIndex,
    train_start: str,
    test_start: str,
    test_end: str,
    stride: int = 21,
    horizon: int = 21,
) -> Iterator[Fold]:
    """Yield non-overlapping expanding-window folds.

    Non-overlapping is the point: stride == horizon means no two forecast
    windows share an observation, which is what makes the Diebold-Mariano
    test in evaluation/stats.py defensible.
    """
```

Each `Fold` carries `train_slice`, `origin`, `forecast_index`, `block_id` (the calendar
year, used by the cadence policy), and `regime_label`.

### 3.3 Refit cadence — `cadence.py`

Implements D5. Two policies behind one interface:

- `EveryFoldCadence` — refit at every fold.
- `BlockCadence(freq="YS")` — fit at the first fold of each block, reuse within the block.

The runner takes a cadence per model class. The "matched" configuration puts every model
on `BlockCadence`; the "native" configuration puts classical models on `EveryFoldCadence`.
Both are run, both are reported.

### 3.4 The runner — `runner.py`

```
for fold in folds:
    for model_spec in panel:
        if cadence.should_refit(model_spec, fold):
            model = model_spec.build()
            model.fit(data.loc[fold.train_slice], origin=fold.origin)
            cache[model_spec.id] = model
        forecast = cache[model_spec.id].predict(horizon=21)
        assert forecast.index[0] > fold.origin        # cheap runtime guard
        writer.append(forecast, fold)
```

Writes one tidy parquet per (series, arm, cadence) to `experiments/results/forecasts/`.
Long format: `origin, target_date, step, model_id, quantile, value, actual, regime,
block_id`. Everything downstream — metrics, plots, DM tests, the Space — reads that one
schema. One format, computed once, and the demo shows the same numbers as the README
because they come from the same file.

### 3.5 Leakage tests — `tests/test_no_leakage.py`

The brief calls leakage a hard constraint, so it gets executable enforcement rather than a
convention:

1. For every fold, `max(train_index) <= origin` and `min(forecast_index) > origin`.
2. Fitted scalers/orders carry a `_fitted_on_origin` attribute; the test asserts it equals
   the fold origin for every model in the panel.
3. A synthetic canary: inject a column that is a perfect copy of the *future* target,
   confirm every model's error collapses, and confirm the leakage assertions fire. A guard
   nobody has ever seen fail is a guard nobody knows works.
4. MASE denominators and RW quantile calibrations are recomputed per fold; the test
   asserts they differ across folds (a constant value means someone cached globally).
5. Regime thresholds are asserted to match the frozen values in `regimes.yaml`.

This file runs in CI on every push.

---

## 4. Model implementations (brief Section 6)

### 4a. Naive and classical — `models/naive.py`, `models/classical/`

| File | Model | Notes |
|---|---|---|
| `naive.py` | `RandomWalk` | Median = last value. Quantiles from the empirical distribution of h-step changes in the training window, recomputed per fold. |
| `naive.py` | `SeasonalNaive` | Weekly seasonality (5 business days). Mostly a sanity baseline. |
| `classical/arima.py` | `ARIMA` | darts `ARIMA`; order selection per fold via AIC over a bounded grid. Grid bounds are a config value, chosen once on the *training* span, never touched after. |
| `classical/sarimax.py` | `SARIMAX` | statsmodels. Arm B only. Covariates are lagged so only values known at `t` enter. |
| `classical/har.py` | `HAR`, `LogHAR` | OLS on 1/5/22-day lagged mean log-RV. RV track only. Quantiles from OLS residual quantiles scaled by `sqrt(h)`. |
| `classical/ar1.py` | `AR1` | Rates track. The standard macro benchmark. |

HAR is ~40 lines and is non-negotiable — see `DECISIONS.md` D14.

### 4b. Neural — `models/neural/`

| File | Model | Config |
|---|---|---|
| `nbeats.py` | darts `NBEATSModel` | `likelihood=QuantileRegression(quantiles=GRID)`, input chunk 256, output chunk 21 |
| `deepar.py` | darts `RNNModel(model="LSTM")` | Quantile likelihood — darts documents this as equivalent to DeepAR in its probabilistic version |

Both train on Colab. Both accept the same `training_window_days` parameter that drives the
D9 sample-efficiency sweep. Early stopping on a validation slice carved from the *end* of
each training block, patience 5.

### 4c. Foundation — `models/foundation/`

| File | Model | Notes |
|---|---|---|
| `chronos2.py` | `amazon/chronos-2` zero-shot + fine-tuned | `Chronos2Pipeline.from_pretrained`. Native covariates for Arm B. Loads fine-tuned weights from `forecastbench-chronos` by revision tag. |
| `chronos_bolt.py` | `amazon/chronos-bolt-small` | Same two modes. The `transformers`+`peft` path that mirrors your Mistral/Qwen work. |
| `timesfm.py` | `google/timesfm-2.5-200m` | Zero-shot only. Stretch. Import guarded so a missing `timesfm` install cannot break the core suite. |
| `hub.py` | — | Push/pull checkpoints, one revision tag per (series, arm, block, training-window) combination |

Context length: 512 steps, matched across every foundation model and used as the input
chunk for the neural models too, so context is not a confound.

### 4d. Fine-tuning recipe (brief Section 7)

- LoRA via `peft`. Rank 8, alpha 16, dropout 0.05, targeting attention projections.
- Trainable parameter count logged to W&B and printed in the model card.
- Early stopping on a held-out validation fold from the end of the training block,
  patience 3, monitoring validation WQL.
- One fine-tune per (series, arm, block, training-window-size). With annual blocks over
  ~11 years, 2 series, 2 arms, and 4 window sizes, that is a lot of short runs rather than
  a few long ones — which is exactly the shape H100 access handles well and is also why
  the checkpoint-resume machinery matters.
- Push to `forecastbench-chronos` with a revision tag encoding the configuration. The Space
  and the local environment pull by tag, so the demo and the repo can never drift.

**Optional, flagged not required:** comparing LoRA against BitFit and LayerNorm-only tuning
(recent PEFT work on Chronos reports the lightweight methods can beat LoRA at a fraction of
the trainable parameters). This is a legitimate citable extension. It is out of the core
because it multiplies runs against a study that already has enough moving parts. If it
happens, it lands as `docs/peft_comparison.md`, not in the headline table.

---

## 5. Evaluation (brief Section 8 items 4, 7, 8)

### 5.1 `evaluation/metrics.py` — the single source of truth

Every metric is defined once here and imported everywhere. Nothing in the notebooks, the
scripts, or the Space recomputes a metric locally. (This is the same convention as
`graphbench/benchmark/metrics.py::normalize_answer`, and it exists for the same reason:
the moment two definitions of MASE exist, one of them is wrong and you will not know which.)

Point: `mae`, `rmse`, `mase`, `smape`. Directional: `directional_accuracy` on the *change*
from origin. Probabilistic: `pinball_loss`, `weighted_quantile_loss`, `interval_coverage`,
`interval_width`. Relative: `skill_score(metric, baseline="RandomWalk")`.

Coverage and width are always returned as a pair and always reported as a pair — a model
can buy coverage with uselessly wide intervals, and showing one without the other hides it.

### 5.2 `evaluation/stats.py`

- `diebold_mariano(losses_a, losses_b, horizon)` — Harvey-Leybourne-Newbold correction,
  Newey-West HAC variance. Docstring states the sample-size caveat explicitly.
- `model_confidence_set(loss_matrix, alpha=0.1)` — via `arch`. Answers "which models can we
  not distinguish," which is the honest framing when ~137 folds is a small sample.
- `bootstrap_skill_ci(...)` — block bootstrap CI on skill scores, for the results table.

### 5.3 `evaluation/regimes.py`

Frozen tercile thresholds loaded from `experiments/configs/regimes.yaml`, asserted against
the committed values at import time. Assignment by `VIXCLS` at the forecast origin.

### 5.4 `evaluation/aggregate.py`

Consumes the tidy forecast parquet, emits the results tables that go into
`docs/benchmark_results.md`, the README, and the Space:

1. **Headline** — model × series × horizon, matched cadence, Arm A.
2. **Cadence comparison** — matched vs. native.
3. **Covariates** — Arm A vs. Arm B.
4. **Regime-stratified** — model × regime × horizon.
5. **Sample efficiency** — skill vs. training-window size.
6. **Contamination-free** — post-release-date folds only, with the small-n warning
   attached to the table itself, not just the prose.

---

## 6. The demo (brief Section 9 — required deliverable)

`space/` in the repo mirrors to the Space. One source of truth, pushed by
`scripts/push_artifacts.py`.

**Hosting:** HF PRO + CPU Basic. Reasoning in `DECISIONS.md` D12 — the short version is
that ZeroGPU charges quota to the *visitor*, and a recruiter who hits a quota error is a
recruiter you have lost.

**Load path:** models pulled from `forecastbench-chronos` at a pinned revision; pre-computed
results pulled from `forecastbench-data`. Nothing is computed at build time except a single
warm-up forecast so the first visitor does not eat the cold start.

**Structure:** five tabs as laid out in `DECISIONS.md` D12. Landing tab is the live
forecast with a default already rendered.

**Design constraints that come from the two-minute rule:**

- Something real is on screen before the visitor clicks anything.
- Every axis is labelled in words, not symbols. "Predicted volatility (log variance)",
  not "log σ²".
- Every model name has a one-sentence plain-language gloss visible on hover, sourced from
  `space/model_cards.py` so the repo and the Space cannot disagree.
- The limitations — including pretraining contamination — are on a tab, not buried in a
  link. A demo that volunteers its own weaknesses reads as more credible, not less, to the
  kind of person worth impressing.
- No result appears in the Space that is not also in `docs/benchmark_results.md`.

---

## 7. Documentation standard (brief Section 11), applied

Applied to every file listed in `REPO_STRUCTURE.md`, matching the conventions already in
GraphBench (Google-style docstrings, type hints on all signatures, `logging` in library
code, `print`/`tqdm` in notebooks).

**Module docstrings** state what the file is for in one or two sentences.

**Function/class docstrings** state what it does, its inputs and outputs, and — the part
that matters here — any non-obvious assumption. Concretely, these are the assumptions that
must appear in docstrings somewhere, because each one silently breaks the study if violated:

- `data/targets.py`: "Assumes SPY OHLC is split- and dividend-adjusted consistently across
  the full span; a mid-series adjustment change would produce a spurious volatility jump."
- `data/covariates.py`: "Only non-revised daily FRED series are permitted here. Adding a
  revised series (CPIAUCSL, UNRATE, FEDFUNDS) will silently introduce look-ahead bias,
  because FRED indexes those by reference period, not release date."
- `backtest/protocol.py`: "fit() must not close over any object fitted outside the current
  fold. Violating this produces leakage that no metric will reveal."
- `evaluation/metrics.py::mase`: "The seasonal-naive denominator is computed on the
  training window only. Computing it on the full series is the most common way MASE is
  reported wrongly."
- `evaluation/stats.py::diebold_mariano`: "Assumes non-overlapping forecast windows, which
  holds only because stride == max horizon in splitter.py. Changing the stride invalidates
  this test."
- `models/foundation/chronos2.py`: "Zero-shot results on pre-October-2025 origins may be
  contaminated by pretraining exposure. See docs/limitations.md."

**Inline comments** explain why, not what. `# floor at the 0.1th training percentile —
low-range days can drive the GK estimator non-positive and ln() would produce NaN` rather
than `# take the log`.

**Notebooks** get a markdown cell before every major code block stating the goal in plain
language. Colab notebooks additionally open with a cell explaining what has to be true
before the notebook is run (which artifacts must already exist on the Hub).

**Per-module READMEs** at `forecast_bench/{data,backtest,models,evaluation}/README.md`,
each ~15 lines: what lives here, what depends on it, what would break if you changed it.

---

## 8. Build order

Dependency order, not a schedule. Commit after each.

1. `git clone`, `poetry init`, `pyproject.toml`, `.gitignore`, `.env.example`, `CLAUDE.md`,
   `LICENSE`, CI workflow skeleton.
2. **`PREREGISTRATION.md` — committed before any model code exists.** The git timestamp is
   the artifact. This has to be step 2, not step 20.
3. `config.py` (pydantic-settings), `data/` clients, target construction, `docs/data_protocol.md`.
4. `tests/test_data_pipeline.py`, `tests/test_no_leakage.py` scaffolding — **before** the
   harness, so the harness is written against its guards.
5. `backtest/protocol.py`, `splitter.py`, `cadence.py`, `runner.py` + their tests.
6. `models/naive.py`, `models/classical/*` — first end-to-end backtest run, local, RW +
   ARIMA + HAR only. This is the first moment the study exists.
7. `evaluation/metrics.py`, `stats.py`, `regimes.py`, `aggregate.py` + tests.
8. First real results table, classical arm only. Sanity-check that RW is hard to beat on
   `DGS10` and that HAR is strong on RV. If those two do not hold, something is wrong
   upstream and it is much cheaper to find out now.
9. `models/foundation/chronos2.py` zero-shot, run locally. Push processed series to
   `forecastbench-data`.
10. Colab notebook 04: LoRA fine-tune Chronos-2, push to `forecastbench-chronos`.
11. `models/foundation/chronos_bolt.py`, both modes.
12. Colab notebook 05: N-BEATS and DeepAR-class training.
13. Full matched-cadence run, both series, Arm A. Headline table.
14. Native cadence run. Arm B (covariates) run.
15. D9 sample-efficiency sweep.
16. D10-G4 contamination-free subset.
17. `space/app.py`, deploy, iterate on the two-minute test with someone who does not know
   the project.
18. `docs/benchmark_results.md`, `docs/limitations.md`, README results table.
19. Medium writeup, GraphBench voice: what it proves, what it does not, what surprised you.
20. Stretch, if and only if everything above has landed: TimesFM 2.5 zero-shot; PEFT-method
    comparison.

---

## 9. Risks, and what is actually done about each

| Risk (brief §12) | Mitigation in this plan |
|---|---|
| Chronos fine-tuning tooling is rough | Chronos-2 has an official fine-tuning notebook and an AutoGluon LoRA path; `chronos-bolt-small` is the fallback on the well-trodden `transformers`+`peft` route. Two paths, so neither is a single point of failure. |
| Small financial data → overfitting | LoRA rank 8, early stopping on a fold-local validation slice, trainable-param count published. The D9 curve independently exposes it. |
| Look-ahead bias invalidates everything | Structural (non-revised series only) + executable (`test_no_leakage.py` with a canary) + documented (`docs/data_protocol.md`). Three layers, one of which fails the build. |
| Over-claiming from few folds | Non-overlapping folds, DM with HLN correction, Model Confidence Set, pre-registered decision rule, bootstrap CIs on every headline number. |
| Free-tier compute limits | Largely dissolved by H100 access + PRO, but checkpoint-resume is implemented anyway. HF artifact sizes stay small: 120M and 48M models, parquet results. |
| **Pretraining contamination (not in the brief)** | See `DECISIONS.md` D10-G4. Stated as a first-class limitation, plus a post-release-only sub-evaluation. This is the risk most likely to be raised by a sharp reviewer and the one the brief did not anticipate. |
| Two targets doubles the work | Target is config, not code. The marginal cost is ~260 extra ARIMA fits and one extra column everywhere. |
| Demo breaks for a recruiter | CPU Basic (no per-visitor quota), pre-warmed, with a pre-computed fallback path if live inference latency disappoints. |
