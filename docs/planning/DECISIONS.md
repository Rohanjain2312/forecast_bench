# forecast_bench — Resolved Decisions

Every open item from Section 8 of the brief, resolved with reasoning. Where more than
one answer was reasonable, the trade-off is stated rather than hidden. Decisions that
constrain other decisions are marked with a **→ Cascades to** line.

This file is the "why" companion to `IMPLEMENTATION_PLAN.md`. If a design choice in the
code looks arbitrary six months from now, the reason is in here.

---

## 0. Three findings that changed the premises

Before the decisions, three things that are true as of September 2026 and were not true
when the brief was written. Each one invalidates part of the brief.

### 0.1 The free Hugging Face Space tier no longer supports the required demo

Static Spaces are free for everyone; Gradio and Docker Spaces run on compute and require
a paid plan to create (PRO for personal accounts). The one exception is that free personal
accounts in good standing can host up to 2 Gradio Spaces on ZeroGPU, where visitor quotas
are 2 GPU-minutes/day unauthenticated and 5 for logged-in free accounts.

The brief's constraint set — required interactive demo, live inference, free tier only —
no longer closes. Since you confirmed the $9/mo PRO subscription is acceptable, this is
resolved by buying PRO. See **D12**.

### 0.2 Chronos-2 exists and is a better fit than the model the brief plans around

Released October 2025: a 120M-parameter encoder-only model with native univariate,
multivariate, and covariate-informed support, quantile outputs, and working CPU inference.
It is fine-tunable with LoRA. This eliminates the need for ChronosX-style covariate
adapters entirely, and makes a CPU-hosted live demo viable. See **D3**, **D12**, **D13**.

The trade-off: Chronos-2 is not a vanilla `transformers` T5, so the brief's claim that
this is "literally the Mistral/Qwen workflow again" is weaker than stated. Handled in D13
by fine-tuning both Chronos-2 and the older `chronos-bolt-small`.

### 0.3 The reusable assets the brief assumes exist do not exist

I read your repos. Two corrections:

- **There is no ALFRED / point-in-time code in `market-regime-transformer-codex`.** I
  grepped the whole repo for `alfred`, `vintage`, `realtime_start`, and `point-in-time`
  and found nothing. What that project actually does is download FRED series, forward-fill
  them daily, and align to prices with `merge_asof`. The brief describes a protocol you
  have not built yet.
- **There are no HMM regime labels.** `market-regime-transformer-codex/src/features.py`
  builds regime labels as `(rolling_30d_return > threshold).astype(int)` — weak supervision
  from a rolling return, not a hidden Markov model.

There is a third, more useful finding buried in the first: **that forward-fill + merge_asof
pattern has look-ahead bias in it.** FRED indexes `CPIAUCSL` by its *reference month*, not
its *release date*. March CPI carries a `2024-03-01` index but is not published until
mid-April. Any model reading it on `2024-03-15` is reading the future. This is exactly the
class of bug the brief calls "the single easiest way to invalidate the whole study."

This is not a problem — it is an asset. "I found a leakage bug in my own earlier project
and this repo is the fix" is a stronger interview story than "I reused my existing
protocol." It also drives **D10**.

---

## D1 — Which series to forecast

**Decision: two targets, run through one shared harness.**

| | Primary | Contrast |
|---|---|---|
| **Target** | Log realized variance of SPY | 10-year Treasury yield (`DGS10`) |
| **Source** | Yahoo Finance OHLC → Garman-Klass estimator | FRED, daily, non-revised |
| **Why** | Options-pricing tie-in, best demo visuals, forecastable signal | Extends `Recession_BondsYield`, near-unit-root, RW is brutal |
| **Expected result** | Classical HAR wins | Nobody beats random walk by much |

**Reasoning.** The brief asks for a study that shows "where each model class wins, loses,
and why." A single series cannot show that — it produces one aggregate number and an
assertion. Two series with *genuinely opposite* expected answers produce the structure the
brief actually wants.

The expected answers are not guesses. A 2026 benchmark of foundation models against
HAR-family models on realized volatility found the classical HAR variants dominating, with
Chronos-Bolt and TimesFM-2.5 posting near-zero win rates at short horizons. Separately, a
2026 benchmark of foundation models on daily equity returns found skill scores against a
random walk clustered near zero in both directions. Rates and macro are the least-settled
of the three tracks.

**On the realized-volatility construction.** True realized variance requires intraday
data, which is not free. The standard free substitute is a range-based estimator from daily
OHLC. We use **Garman-Klass**, which uses open, high, low, and close and is roughly 7×
more efficient than close-to-close squared returns. Parkinson (high-low only) is the
fallback if OHLC quality is poor on any span. We model `log` of the estimator, which is
standard practice and makes the series far better-behaved (roughly Gaussian, homoskedastic).

**Explicitly rejected: equity returns as a target.** Expected null result, most crowded
literature, and the weakest portfolio story. It appears in the study only as the input to
the volatility estimator.

**Trade-off accepted.** Two targets roughly doubles the *evaluation* surface but barely
touches the *harness* surface, because the harness treats the target as configuration
(`experiments/configs/*.yaml`) rather than code. The extra cost is real but bounded: about
260 additional ARIMA fits, which is minutes on your machine.

**→ Cascades to:** D3 (which covariates exist), D9 (level vs. transform), D14 (HAR must
enter the classical panel), D8 (regime definition), and the entire demo series picker.

---

## D2 — Forecast horizons

**Decision: `{1, 5, 21}` trading days, all read off the same forecast path.**

Each backtest fold produces one 21-step-ahead quantile path. h=1, h=5, and h=21 are read
off steps 1, 5, and 21 of that same path. This is not just cheap — it means all three
horizons come from identical folds and identical model fits, so cross-horizon comparisons
are apples-to-apples, and no horizon gets a different effective sample size.

**Reasoning.** The brief correctly notes that classical and foundation models diverge
differently as horizon lengthens. That divergence *is* a headline result and costs nothing
extra to measure. 1 day is the noise floor, 5 days is a trading week, 21 days is a trading
month and lines up with the standard HAR monthly component.

---

## D3 — Univariate or with covariates

**Decision: two arms. Arm A (univariate) is the headline. Arm B (covariate-informed) is a
clearly-labeled extension.**

- **Arm A — univariate.** Every model sees only the target's own history. This is the
  clean, honest, comparable arm and it is what goes in the README results table.
- **Arm B — covariates.** Chronos-2 uses native covariate support (no adapter needed).
  Classical side uses SARIMAX. Neural side uses darts' past/future covariate support.

**Covariate set, deliberately restricted:**

| Target | Covariates |
|---|---|
| SPY log-RV | `VIXCLS`, `DGS10`, day-of-week |
| `DGS10` | `T10Y2Y`, `DGS3MO`, `VIXCLS`, `DFF` |

**Every one of these is a daily, market-observed, non-revised series.** `CPIAUCSL`,
`UNRATE`, `FEDFUNDS` (monthly), and monthly `GS10` are **excluded from all modeling** and
appear only in the writeup's discussion. This is D10's guardrail expressed as a data
constraint rather than a code constraint, and it makes the point-in-time claim true by
construction rather than by careful bookkeeping.

**Trade-off.** Excluding revised macro means we cannot claim "we forecast rates using
inflation and employment data." That is a real loss of scope. It buys a leakage claim that
cannot be argued with, which matters more for a study whose entire value proposition is
credibility. Stated in `docs/limitations.md`.

**→ Depends on:** D1. **→ Cascades to:** D10, D13 (Chronos-2 is chosen partly because
covariates are native).

---

## D4 — Point or probabilistic forecasts

**Decision: probabilistic, committed. Every model in the study emits quantiles.**

Quantile grid: `[0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]`.

**Consequences accepted now, because retrofitting this later is expensive:**

- The neural baseline must be a probabilistic one — darts `RNNModel` with a quantile
  likelihood, and N-BEATS with `QuantileRegression`, not the plain deterministic versions.
- Random walk needs a quantile forecast too. It gets one from the empirical distribution
  of h-step changes in the training window, refit per fold. A point-only naive baseline
  would be scored unfairly on quantile loss.
- ARIMA/SARIMAX prediction intervals get extracted properly via `get_forecast().conf_int()`
  across the grid, not approximated from a normal assumption after the fact.
- HAR needs quantiles: OLS residual quantiles, scaled by horizon, refit per fold.

**Reasoning.** Chronos is natively a quantile model — evaluating it as a point forecaster
would discard most of what it produces and would be the kind of methodological shortcut
that makes a benchmark unpersuasive. It also gives the demo a forecast fan instead of a
line, which is a materially better visual.

**→ Cascades to:** D7 (metric set), D12 (demo visuals), every model implementation.

---

## D5 — Refit cadence

**Decision: dual-cadence protocol, both reported. This is the fairness crux, so it gets
solved explicitly rather than assumed away.**

The problem: refitting ARIMA at every fold is cheap and correct. Retraining a fine-tuned
Chronos at every fold is not viable on any budget. So a naive "refit everything every
fold" rule is impossible, and a naive "refit classical every fold, learned models once" is
an unfair fight that the classical arm wins on freshness rather than on merit.

**Resolution — every model is run under two cadences:**

| Cadence | Classical | Neural / Foundation |
|---|---|---|
| **Matched (headline)** | Refit at each annual block boundary | Retrain at each annual block boundary |
| **Native (secondary)** | Refit every fold | Retrain at each annual block boundary |

The **matched** cadence is the headline comparison, because every model class is given the
same information-refresh rate and the comparison is therefore about model quality. The
**native** cadence is reported alongside, because refit-every-fold is what a practitioner
would actually do with ARIMA, and hiding it would understate the classical arm.

The gap between the two rows is itself a reportable finding: it measures how much of the
classical arm's performance comes from frequent refitting versus from the model.

**Trade-off.** Running two cadences roughly doubles classical-model compute. It costs
minutes. It buys the single most defensible claim in the study.

**→ Cascades to:** D6 (block boundaries must align with fold structure), D7 (results table
gains a cadence column).

---

## D6 — Backtesting scheme

**Decision: expanding-origin walk-forward, non-overlapping test windows, annual blocks.**

| Parameter | Value | Why |
|---|---|---|
| Scheme | Expanding window | More training data over time, matches real deployment |
| Initial train span | 2000-01-01 → 2014-12-31 | ~15y, enough for ARIMA order selection and neural training |
| Test span | 2015-01-01 → 2026-06-30 | ~11.5y, spans 2015 vol spike, 2018 Q4, 2020 COVID, 2022 rates, 2024–26 |
| Fold stride | 21 trading days | Equals the longest horizon → **non-overlapping** forecast windows |
| Folds per series | ~137 | Enough for Diebold-Mariano to mean something |
| Block boundary | 1 January each year | Defines the "matched cadence" retrain points from D5 |
| Embargo | 0 days, by design | Forecast origin is `t`; models see only data `≤ t`. Enforced by test, not by gap |

**Why non-overlapping matters.** With stride = 21 = max horizon, no two forecast windows
share an observation. Diebold-Mariano assumes the loss-differential series is not
pathologically autocorrelated; overlapping windows violate that and inflate significance.
Most published backtests quietly overlap. Ours does not, and `docs/methodology.md` says so.

**Why no embargo.** Embargoes exist to stop feature-engineering lookbacks from straddling
the train/test boundary. Our features are causal context windows ending at `t`, so there is
nothing to embargo. Rather than adding a cosmetic gap, we enforce the real invariant in
`tests/test_no_leakage.py`: for every fold, `max(train_index) <= forecast_origin` and every
scaler is fitted inside the fold.

---

## D7 — Evaluation metrics and statistical testing

**Decision: full metric set, plus formal significance testing, plus explicit multiple-
comparison handling.**

**Point accuracy:** MAE, RMSE, MASE (seasonal-naive denominator computed on the training
window only), sMAPE.

**Directional:** directional accuracy on the *change* from origin, not on the level. On a
persistent series like `DGS10`, directional accuracy on the level is trivially ~100% and
meaningless.

**Probabilistic:** weighted quantile loss (WQL) across the grid, per-quantile pinball loss,
80% and 95% interval coverage, and interval width. Coverage and width are reported together
— a model can win coverage by producing uselessly wide intervals, and reporting only one
of the pair hides that.

**Skill scores:** every metric also reported as skill relative to random walk,
`1 - metric_model / metric_RW`. Raw MAE on log-RV is uninterpretable to a reader; "3.4%
better than random walk" is not.

**Significance:** Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample
correction and Newey-West HAC variance, run pairwise against the best classical model at
each (series, horizon, regime). Reported with the honest caveat that ~137 folds is a small
sample for this test.

**Multiple comparisons:** with ~7 models × 2 series × 3 horizons × 4 regime cuts, running
DM everywhere and reporting whatever clears p<0.05 is p-hacking. Two guards: (a) the
headline claims are restricted to the pre-registered comparisons in `PREREGISTRATION.md`,
(b) a Model Confidence Set (Hansen, `arch` package) is reported per series-horizon, which
answers "which models can we not statistically distinguish" without requiring a
correction-per-test.

---

## D8 — Regime-stratified evaluation

**Decision: VIX terciles with thresholds frozen on pre-2015 data, plus two named crisis
case studies.**

- **Regime assignment:** by `VIXCLS` level at the forecast origin `t`. Known at time `t`,
  so no leakage.
- **Thresholds:** the 33rd and 67th percentiles of `VIXCLS` over 2000-01-01 → 2014-12-31
  **only**. Computed once, written into `experiments/configs/regimes.yaml`, and never
  recomputed. Computing terciles over the full sample would leak the test period's
  volatility distribution into the regime definition.
- **Labels:** `calm` / `normal` / `stressed`.
- **Case studies:** Feb–Apr 2020 (COVID) and Jan–Oct 2022 (rate shock) called out
  separately as narrative cuts in the writeup and the demo.

**Why not the HMM labels from the prior project:** they do not exist (see 0.3). The
existing labels are `rolling_30d_return > 0`, which is a bull/bear proxy, not a volatility
regime, and is the wrong conditioning variable for this study. Building an HMM here would
be a side quest that adds a tunable component to a study whose whole point is that nothing
is tuned on the test set.

**Trade-off.** VIX terciles are cruder than a fitted regime model. They are also
transparent, reproducible in one line, and impossible to accuse of overfitting — which is
worth more here.

**→ Depends on:** D1 (VIX is a covariate on the RV track, so it must be handled carefully
to avoid double-use — see `docs/methodology.md`; using it for *stratification* is
reporting, not modeling, and does not contaminate Arm A).

---

## D9 — Sample-efficiency framing

**Decision: in scope, minimal version.**

Fine-tune Chronos-2 on `{1y, 3y, 10y, full}` slices of the training window; train N-BEATS
and the DeepAR-class model on the same four slices. Plot skill-vs-RW against training-set
size, one line per model class.

**Reasoning.** The recent literature claim is that foundation models need far less
adaptation data than from-scratch models need training data. That is a testable,
falsifiable, citable claim and testing it costs four training runs per model instead of
one — trivial with H100 access. It also produces the single most legible chart in the
project: one axis is "how much data," one axis is "how good," and the shapes of the curves
tell the whole story without a caption.

**Also, honestly:** it is the experiment most likely to produce a result that favors the
foundation model. Including it is not stacking the deck as long as the headline comparison
(D1, matched cadence, full training window) is reported first and unchanged.

**→ Also serves as:** the overfitting safeguard the brief asks for in Section 12. The 1y
slice is where fine-tuning overfits, and the curve will show it.

---

## D10 — Data-leakage guardrails

**Decision: four guardrails, one of them new and more serious than the brief anticipates.**

**G1 — Non-revised series only.** Enforced by D3. No revised macro enters any model. This
makes the point-in-time claim structural rather than procedural.

**G2 — Fold-local fitting.** Every scaler, every ARIMA order selection, every quantile
calibration, every MASE denominator is computed inside the fold on training data only.
Enforced by `tests/test_no_leakage.py`, which fails the build if a fitted object's
provenance spans the forecast origin.

**G3 — Frozen regime thresholds.** Per D8.

**G4 — Pretraining contamination. This is the one the brief misses, and it is the biggest
threat to the study.**

Chronos-2 was released October 2025. TimesFM 2.5 in September 2025. Both were pretrained
on large corpora that plausibly include public financial series — SPY and Treasury yields
are among the most widely redistributed time series in existence. Our test span
(2015–2026) largely *predates* those releases. So a "zero-shot" forecast of SPY volatility
in 2019 may not be out-of-sample at all: the model may have seen that exact data.

The brief's Section 8 item 10 only asks about the fine-tuning cutoff. The pretraining
cutoff is the harder problem, and it is unfixable — we cannot inspect Amazon's or Google's
pretraining corpus.

**Mitigation, in three parts:**

1. **State it plainly** in `docs/limitations.md` and in the README, as a first-class
   limitation rather than a footnote. Any zero-shot number on the pre-release span is
   reported as *potentially contaminated*.
2. **Run a contamination-free sub-evaluation.** Restrict a secondary results table to
   forecast origins after each model's release date (Chronos-2: 2025-11-01 onward;
   TimesFM 2.5: 2025-10-01 onward). At 21-day stride that is only ~8 folds — too few for
   significance, and reported as such — but it is the only genuinely clean zero-shot read
   available, and reporting it *with* its inadequate sample size is more honest than not
   running it.
3. **Note the asymmetry.** Contamination inflates the *zero-shot* numbers, not the
   fine-tuned ones (which are trained on our own data with our own cutoffs). So if the
   fine-tuned model beats zero-shot, that gap is trustworthy; if zero-shot looks
   surprisingly strong on the pre-2025 span, treat it with suspicion.

This guardrail is, on its own, a defensible reason for the project to exist. Most public
foundation-model benchmarks on financial data do not address it at all.

---

## D11 — What "honest" means in practice

**Decision: written, git-timestamped pre-registration, committed before any results exist.**

`PREREGISTRATION.md` is committed to the repo as one of the first commits, before any
model runs. It states the decision rules in advance. The git timestamp is the proof that
they were not adjusted after the fact.

The operative rule:

> The fine-tuned foundation model is reported as **losing** if either: (a) it fails to
> beat the random-walk baseline on WQL skill score at h=1 on either series, or (b) it
> fails to achieve Diebold-Mariano significance at p<0.05 against the best classical model
> at any of the three horizons on either series, under the matched refit cadence.

Full text and the accompanying "changes we commit to *not* making after seeing results"
list is in `PREREGISTRATION.md`.

---

## D12 — Demo scope and hosting

**Decision: HF PRO subscription + a CPU Basic Gradio Space running Chronos-2 live on CPU.
Not ZeroGPU.**

You offered $9/mo and asked whether it has real benefit. It does, and the specific benefit
is not the one you would expect.

**Why not ZeroGPU, even though it is free and has a GPU:** ZeroGPU quota is charged to the
*visitor*, not the owner. An unauthenticated visitor — which is exactly what a recruiter
is — gets 2 GPU-minutes per day and queues behind PRO users. A recruiter who clicks three
times and hits a quota error or a 40-second queue has formed their opinion of your work,
and it is not the opinion you want. PRO also does not fix this; it fixes *your* quota, not
theirs.

**Why CPU Basic works:** Chronos-2 is 120M parameters and officially supports CPU
inference. A single-series, 21-step quantile forecast with a 512-step context should
complete in roughly 1–3 seconds on 2 vCPU. `chronos-bolt-small` (48M) is faster still.
There is no per-visitor quota on CPU Basic, no queue, and no failure mode that depends on
who the visitor is.

**What PRO actually buys, then:** the right to create a Gradio Space at all. Since July
2026 that requires a paid plan. PRO is the entry ticket, and CPU Basic is the correct
hardware once you are through the door.

**Fallback:** if measured CPU latency exceeds ~10s per forecast, downgrade the demo to
pre-computed forecasts over a fixed date grid (still interactive, still shows the fan
chart, just not arbitrary date ranges) rather than moving to ZeroGPU. Predictable and
always-fast beats fast-but-sometimes-broken for this audience.

**Demo scope — biased toward showing more, per the brief:**

| Tab | Contents |
|---|---|
| **1. Live forecast** | Series picker, date-range picker, horizon picker. Runs Chronos-2 zero-shot, fine-tuned Chronos-2, ARIMA, and random walk live. Fan chart with 80%/95% bands over actuals. |
| **2. Benchmark results** | Sortable table: model × series × horizon × cadence, all metrics from D7, skill scores, DM p-values. Loaded from `forecastbench-data`, identical numbers to the repo. |
| **3. Where each model wins** | Regime-stratified breakdown (D8) as a heatmap, plus the two crisis case studies. |
| **4. Sample efficiency** | The D9 curve. |
| **5. What am I looking at** | Plain-language model cards, the 60-second project explanation, the limitations list including D10-G4. |

The landing tab is Tab 1 with a pre-loaded default forecast already rendered, so the Space
shows something real before the visitor clicks anything.

---

## D13 — Which foundation model(s)

**Decision (not in the brief's Section 8, but forced by finding 0.2):**

| Model | Params | Role | Where it runs |
|---|---|---|---|
| `amazon/chronos-2` | 120M | **Core.** Zero-shot + LoRA fine-tuned | Fine-tune on Colab H100; inference local + Space CPU |
| `amazon/chronos-bolt-small` | 48M | **Secondary.** Zero-shot + LoRA fine-tuned | Same |
| `google/timesfm-2.5-200m` | 200M | **Stretch.** Zero-shot only | Colab |

**Reasoning.** Chronos-2 is the better model and the one that makes covariates and the CPU
demo work. But `chronos-bolt-small` costs almost nothing to add and buys two things: it
preserves the "same `transformers` + `peft` recipe as my Mistral and Qwen fine-tunes"
narrative that the brief wants, and it produces a *generational* comparison — old
foundation model vs. new foundation model vs. classical — which is a more interesting
results table than a two-way split.

The fine-tuned **Chronos-2** checkpoint is what gets published to
`huggingface.co/rohanjain2312/forecastbench-chronos`.

TimesFM 2.5 now has a documented HF Transformers + PEFT LoRA fine-tuning path, so it is
more tractable than the brief assumed — but it remains an entire second toolchain. Zero-shot
only, and only if everything else has landed.

**Overfitting safeguard** (Section 12 of the brief): LoRA rank 8, alpha 16, early stopping
on a held-out validation fold carved from the *end* of each training block, patience 3.
Trainable parameter count is logged to W&B and reported in the model card. The D9 sample-
efficiency curve independently exposes overfitting at small training sizes.

---

## D14 — Model panel (which baselines earn their place)

**Decision:**

**Naive:** random walk (quantiles from empirical h-step change distribution), seasonal
naive.

**Classical:** ARIMA (auto order selection per fold), SARIMA, SARIMAX (Arm B only),
**HAR and log-HAR** (RV track), AR(1) (rates track).

**HAR is not in your brief and it is not optional.** HAR-RV is *the* benchmark in the
realized-volatility literature — it is the model every referee would ask about, and it is
the model that beat foundation models in the 2026 benchmark cited in D1. A volatility study
whose classical arm is only ARIMA/SARIMA has a hole in it that a quant interviewer will
find in thirty seconds. Adding it is ~40 lines of OLS on 1-day, 5-day, and 22-day lagged
log-RV.

**AR(1)** plays the same role on the rates track: it is the standard macro-forecasting
benchmark, and the recent vintage-consistent macro literature benchmarks foundation models
against exactly it.

**Neural:** N-BEATS with `QuantileRegression`, and darts `RNNModel(model="LSTM")` with a
quantile likelihood — which the darts documentation describes as equivalent to DeepAR in
its probabilistic version. This earns the brief's "DeepAR-class" claim honestly rather than
substituting N-BEATS for both.

**Foundation:** per D13.

---

## D15 — Library strategy

**Decision: darts for model implementations, own harness on top.**

The brief asks for this trade-off explicitly. Both directions are reasonable:

- **All-darts** (including its `historical_forecasts` backtester) is less code and fewer
  bugs. But it outsources the backtest harness — which is the single most inspectable,
  most differentiating artifact in a benchmark repo. A reviewer wants to read your
  leakage guards, not confirm that a library has some.
- **All-hand-rolled** maximises the "I built this" signal but means reimplementing ARIMA
  wrappers and N-BEATS, which is undifferentiated work and a bug surface.

**Split the difference.** darts (0.44.0, actively maintained, April 2026) provides ARIMA,
N-BEATS, and the probabilistic RNN behind one API with covariate and quantile support.
`statsmodels` provides HAR (plain OLS) and SARIMAX. `chronos-forecasting` provides the
foundation models. All four are wrapped behind a single `Forecaster` protocol in
`forecast_bench/backtest/protocol.py`, and the fold generation, refit-cadence logic,
leakage enforcement, and scoring are ours.

**Consequence:** the harness never calls `darts.historical_forecasts`. Every model,
including the darts ones, is driven fold-by-fold by our runner, so the foundation models
and the classical models genuinely traverse identical code paths.

---

## Dependency map

```
D1 (target series)
 ├─→ D3 (which covariates exist)
 │    └─→ D10-G1 (non-revised constraint)
 ├─→ D9-transform (log-RV vs level)
 ├─→ D14 (HAR required on RV track; AR(1) on rates track)
 ├─→ D8 (VIX available as regime variable)
 └─→ D12 (series picker in demo)

D4 (probabilistic)
 ├─→ D7 (WQL, pinball, coverage in metric set)
 ├─→ D12 (fan chart is possible)
 └─→ every model must emit quantiles  → D14 (naive/HAR need quantile logic)

D5 (dual cadence)
 └─→ D6 (annual block boundaries must align with fold structure)
      └─→ D7 (results table gains a cadence column)

D2 (horizons off one path)
 └─→ D6 (stride = 21 = max horizon → non-overlapping)
      └─→ D7 (DM assumption satisfied)

D12-PRO (paid tier)
 └─→ unblocks the entire demo deliverable

D13 (Chronos-2)
 ├─→ D3 (native covariates, no ChronosX needed)
 └─→ D12 (CPU inference makes CPU Basic viable)

D10-G4 (pretraining contamination)
 └─→ D6 (adds a post-release-only fold subset)
      └─→ D7 (adds a secondary, small-n results table)
```
