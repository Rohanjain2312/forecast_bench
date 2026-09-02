# Point-in-Time Data Protocol

**The rule: a model may only read a value that had actually been published by the time of
the forecast origin.**

Every leakage guard in this repository serves that one sentence. This document explains how
it is enforced, and — because the enforcement is easier to trust once you see what it
prevents — the specific bug that motivated it.

---

## 1. The failure mode this repository is designed to prevent

### FRED indexes revised series by reference period, not release date

This is the whole problem in one line, and it is not obvious from the API.

When you ask FRED for `CPIAUCSL`, the observation for March 2024 comes back with the index
`2024-03-01`. That timestamp is the **month the inflation refers to**. It is not the date
the number existed. March 2024 CPI was published by the BLS on **10 April 2024**.

So the series looks like this:

| FRED index | Value | Actually knowable from |
|---|---|---|
| `2024-02-01` | Feb CPI | 2024-03-12 |
| `2024-03-01` | Mar CPI | 2024-04-10 |
| `2024-04-01` | Apr CPI | 2024-05-15 |

A model standing at a forecast origin of **2024-03-15** and reading "the latest CPI
observation at or before today" gets the `2024-03-01` row. That number did not exist on
2024-03-15. It was published 26 days later. The model is reading the future.

### Why the usual alignment code walks straight into it

The natural way to join a monthly macro series onto daily prices is to forward-fill it to a
daily index and then `merge_asof` onto the price dates. That is exactly what the earlier
project `market-regime-transformer-codex` does — per the audit recorded in
[`planning/DECISIONS.md`](planning/DECISIONS.md) §0.3, it downloads FRED series,
forward-fills them daily, and aligns to prices with `merge_asof`.

`merge_asof` is doing precisely what it was asked to: "give me the most recent value at or
before this date." The bug is not in `merge_asof`. The bug is that **the FRED index does not
mean what the join assumes it means.** The join treats the reference period as a release
date, and no amount of care with the join direction fixes that, because the release date is
simply not present in the data being joined.

This is the single most dangerous class of leakage in financial ML, for three reasons:

1. **It is invisible in every metric.** The model gets better. Nothing looks wrong. Backtest
   performance improves precisely because the leak is informative.
2. **It survives review.** The code reads as correct, and the join is a one-liner that a
   reviewer nods past.
3. **It is directionally flattering.** It makes results better, not worse, so nobody goes
   looking.

Finding this in your own prior work is more informative than asserting you avoided it. It is
also why this repository's point-in-time claim is made **structurally** rather than
procedurally.

---

## 2. How this repository makes the problem impossible instead of avoidable

There are two ways to be correct here.

**The heavy way** is to fetch *vintage* data — ALFRED, FRED's archive of what each series
looked like on each historical date — and align on release dates. That is the fully general
solution and it is genuinely correct. It is also a substantial subsystem, it introduces its
own alignment bugs, and it puts the study's central claim behind machinery that a reviewer
has to audit before believing anything else.

**The chosen way** is to restrict the input set to series that are never revised and are
published the same day they refer to. Then reference period and release date coincide, the
distinction that caused the bug collapses, and the claim "every model saw only data
available at the time" holds **by construction**.

The trade is scope for credibility. It is the right trade for a study whose entire value
proposition is that its numbers can be believed.

### The allowlist

```python
NON_REVISED_FRED_ALLOWLIST = {"DGS10", "DGS3MO", "T10Y2Y", "VIXCLS", "DFF"}
```

| Series | What it is | Why it is safe |
|---|---|---|
| `DGS10` | 10-year Treasury constant-maturity yield, daily | Market-observed close, published same day, never revised |
| `DGS3MO` | 3-month Treasury constant-maturity yield, daily | Same |
| `T10Y2Y` | 10-year minus 2-year spread, daily | Arithmetic on two same-day market observations |
| `VIXCLS` | CBOE VIX close, daily | Exchange-published close, same day, never revised |
| `DFF` | Effective federal funds rate, daily | Published next business day; used at `t` only via values dated before `t` |

`SPY` OHLC from Yahoo Finance is subject to the same logic: it is an exchange-observed daily
bar, published the day it refers to.

### What is excluded, and why

| Series | Frequency | Problem |
|---|---|---|
| `CPIAUCSL` | Monthly | Indexed by reference month, released ~2 weeks later, then revised |
| `UNRATE` | Monthly | Indexed by reference month, released the following month |
| `FEDFUNDS` | Monthly | Monthly average, indexed by reference month |
| `GS10`, `GS3M` | Monthly | Monthly averages of the daily series; use `DGS10` / `DGS3MO` |
| `GDP`, `PAYEMS` | Quarterly / monthly | Heavily revised for months after first release |

These appear in the writeup's discussion. They never enter a model input.

### Enforcement, in two layers

The allowlist is checked twice, deliberately.

1. **At fetch.** `forecast_bench/data/fred_client.py::assert_non_revised` raises
   `RevisedSeriesError` *before any network call is made*. The error names the series, the
   reason, and points here. There is no `force=True` escape hatch, because the only reason
   to want one is to do the thing this exists to prevent.
2. **At assembly.** `forecast_bench/data/covariates.py::validate_covariates` re-checks the
   assembled covariate frame's columns. A series could otherwise reach a model by being
   constructed, renamed, or loaded from a stale parquet rather than fetched.

Both layers are covered in `tests/test_data_pipeline.py` and `tests/test_targets.py`,
including a test asserting that a revised series is refused *before* the network is touched.

---

## 3. The other point-in-time rules

The release-lag problem is the headline, but it is not the only way a date can lie.

### Targets are never forward-filled

`DGS10` carries NaN on market holidays. Forward-filling those NaNs manufactures an
observation that never existed, and then scores a model against it. Both clients return NaNs
as NaNs, and `build_dgs10` **drops** holiday rows rather than filling them. The merge layer
treats a NaN in the target column as a fatal error, precisely so that nobody fixes it later
with a convenient `.ffill()`.

Covariate NaNs are *permitted* and left in place. A missing VIX print is a real absence, and
what a model does about it is a decision that belongs inside the fold, under the leakage
guards — not at build time, outside them.

### Constants derived from data come from the training window only

Anything computed from the data and then applied across the whole span is a channel for the
test period to influence the training target. Two exist in this study, and both are drawn
from pre-test data only:

- **The Garman-Klass variance floor** is the 0.1st percentile of strictly-positive estimates
  from *before* `TEST_START`. Computing it over the full sample would let the test period's
  volatility distribution set a constant that shapes the target.
- **The VIX regime thresholds** are the 33rd and 67th percentiles of `VIXCLS` over
  2000–2014 only, frozen in `experiments/configs/regimes.yaml` and asserted at import.
  Terciles over the full sample would define "stressed" using the knowledge that 2020 and
  2022 happened — leakage in the *reporting* layer, which is easier to miss and no less
  invalidating.

Both are constants rather than per-fold quantities. That is sound because both are derived
only from data available before the first forecast origin, so every fold is entitled to see
them.

### Everything fitted is fitted inside the fold

Scalers, ARIMA order selections, MASE denominators, residual quantiles: all fold-local, all
enforced by `tests/test_no_leakage.py`. That file is the executable form of this document
and must never be weakened to make a run pass.

---

## 4. What this protocol costs

Stated here as well as in `docs/limitations.md`, because a protocol that only advertises its
benefits is not being honest about itself.

- **No inflation or employment data in any model.** A practitioner forecasting the 10-year
  yield would want CPI and payrolls. This study cannot use them, so it cannot claim to
  forecast rates using macro fundamentals.
- **No vintage analysis.** Because ALFRED is not used, the study says nothing about how
  revisions themselves affect forecasting — a real and interesting question, deliberately
  out of scope.
- **A narrower covariate arm.** Arm B is four daily market series and a calendar feature. It
  is a smaller experiment than it could have been.

What it buys is a claim that does not require trust: not "we were careful about look-ahead
bias," but "look-ahead bias from data revisions is not reachable from the inputs this study
permits."
