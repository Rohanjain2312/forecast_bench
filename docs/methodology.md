# Methodology

The choices that make the numbers mean something, and what each one would cost if changed.

## Backtest scheme

Expanding-origin walk-forward, non-overlapping test windows, annual blocks.

| Parameter | Value | Why |
|---|---|---|
| Scheme | Expanding window | More training data over time, matching real deployment |
| Initial train | 2000-01-01 → 2014-12-31 | ~15 years, enough for ARIMA order selection and neural training |
| Test span | 2015-01-01 → 2026-06-30 | Spans 2018 Q4, COVID, the 2022 rate shock |
| Stride | 21 trading days | Equals the longest horizon → **non-overlapping** |
| Folds | 137 (SPY), 136 (`DGS10`) | Enough for Diebold-Mariano to mean something |
| Block boundary | 1 January | Defines the matched-cadence refit points |
| Embargo | 0 days, by design | See below |

### Why non-overlapping matters

With stride = 21 = max horizon, **no two forecast windows share an observation**.
Diebold-Mariano assumes the loss-differential series is not pathologically autocorrelated;
overlapping windows violate that and inflate significance. Most published backtests quietly
overlap. This one does not, and `splitter.py` raises if you try to set `stride != horizon`
rather than documenting the requirement and hoping.

### Why no embargo

Embargoes exist to stop feature-engineering lookbacks straddling the train/test boundary.
Every feature here is a causal context window ending at the origin, so there is nothing to
embargo. Rather than adding a cosmetic gap, the real invariant —
`max(train_index) <= origin < min(forecast_index)` — is enforced by construction and
asserted in `tests/test_no_leakage.py`.

## Horizons

`{1, 5, 21}` trading days, all read off **one** 21-step forecast path. Because all three
come from identical folds and identical model fits, cross-horizon comparisons are
apples-to-apples and no horizon gets a different effective sample size.

## Refit cadence

The fairness crux. Refitting ARIMA every fold is cheap; retraining a fine-tuned Chronos
every fold is not viable. "Refit everything every fold" is impossible, and "refit classical
every fold, learned models once" is an unfair fight the classical arm wins on freshness
rather than merit.

| Cadence | Classical | Neural / Foundation |
|---|---|---|
| **Matched** (headline) | Refit at annual block boundaries | Retrain at annual block boundaries |
| **Native** (secondary) | Refit every fold | Retrain at block boundaries |

**The cadence governs parameters, never conditioning data.** Every model is handed data
running to its own fold's origin on every fold. That distinction is not pedantic: a
zero-shot foundation model has *no parameters to refit at all*, so a cadence that gated
conditioning would hand it a months-old context window and call the result a benchmark.

**Measured result:** refitting ARIMA at all 137 folds instead of 13 buys **+0.004** of WQL
skill. LogHAR gains +0.001. The classical arm's advantage over the foundation models is the
model, not the information-refresh rate.

## Metrics

`evaluation/metrics.py` is the single source of truth. Nothing anywhere else recomputes a
metric — the moment two definitions of MASE exist, one is wrong and nobody knows which.

- **Point:** MAE, RMSE, MASE, sMAPE.
- **Directional:** accuracy on the *change from the origin*, never on the level. On a
  persistent series like `DGS10`, directional accuracy on the level is trivially ~100% and
  means nothing.
- **Probabilistic:** weighted quantile loss (the primary metric), pinball loss, and
  **interval coverage and width as a pair**.
- **Relative:** every metric also as skill against the random walk. Raw MAE on log realized
  variance is uninterpretable; "16.6% better than a random walk" is not.

### Two API choices that encode rules

`interval_coverage_and_width` returns a **pair**, and there is no function that returns
coverage alone. A model can buy coverage with uselessly wide intervals, and an API that
makes it easy to report only the flattering half invites exactly that.

The MASE denominator is a **separate function taking a training window**, so computing it on
the full series requires deliberately passing the wrong thing. That is the most common way
MASE is reported wrongly.

## Statistical testing

- **Diebold-Mariano** with the Harvey-Leybourne-Newbold small-sample correction and a
  Newey-West HAC variance. The HAC lag is `horizon - 1` by convention; because these windows
  do not overlap that is conservative rather than necessary, and at h=21 a lag-20 estimate on
  137 observations is itself noisy.
- **Model Confidence Set** (Hansen, via `arch`) reported alongside every pairwise test. At
  137 origins this is the more honest summary, and it materially changes the reading: the
  fine-tuned Chronos-2 survives at every series and horizon. It lost by the pre-registered
  rule and is *not* demonstrably worse than what beat it.
- **Multiple comparisons:** headline claims are restricted to the comparisons registered in
  `PREREGISTRATION.md` §3. Every other test in the tables is descriptive.

## Regime stratification

Volatility regimes are VIX terciles with thresholds computed **once**, on 2000–2014 only,
committed to `experiments/configs/regimes.yaml`, and asserted against the module constants
at import. Computing terciles over the full sample would define "stressed" using the
knowledge that 2020 and 2022 happened — leakage in the *reporting* layer, which is easier to
miss than leakage in the modelling layer and no less invalidating.

Assignment uses the VIX level at the forecast origin, which is known at time `t`.

## Uncertainty quantification

Every model emits the same 11 quantiles. How each gets them:

| Family | Method |
|---|---|
| Random walk | Empirical distribution of h-step changes, **centred** so the median is the last value |
| Seasonal naive | Empirical seasonal errors, centred the same way |
| ARIMA / AR(1) / SARIMAX | `get_forecast().conf_int(alpha)` per level — the fitted model's own distribution |
| HAR / LogHAR | The model's **measured** h-step-ahead residual quantiles |
| N-BEATS / DeepAR | Quantile-regression likelihood, sampled |
| Chronos | Native quantile output |

The naive baselines are centred deliberately. Without it, a training window with drift
produces a random walk *with* drift — a different and generally stronger model. That matters
more here than anywhere else, because this is the baseline every skill score is quoted
against.

HAR originally used a `sqrt(h)` widening of one-step residuals, the textbook approach for an
*integrated* process. Log realized variance is mean-reverting, so its h-step error spread
grows 1.23× by h=21 where `sqrt(h)` assumes 4.58×. The intervals came out roughly 3.7 times
too wide and covered 100% of actuals. Measuring the spread instead of assuming it is why
LogHAR's h=21 skill went from −0.101 to +0.101.
