# Progress Notes

Measurements and environment findings recorded as the build proceeds. Facts, not plans —
the plan lives in `BUILD_ORDER.md`.

---

## Step 1 — Repo scaffold (2026-09-02)

**The local toolchain assumed by `SETUP_CHECKLIST.md` §6a was not present.** The Mac had
no Homebrew, no Python 3.11 (system Python was 3.9.6), no Poetry, and no SSH key for
GitHub. Resolved without sudo:

- `uv` installed to `~/.local/bin`, and used to install CPython 3.11.16
- Poetry 2.4.2 installed with the official installer, using the uv interpreter
  (note: the installer must be pointed at the *real* interpreter path under
  `~/.local/share/uv/python/...`, not the `~/.local/bin/python3.11` symlink — a venv built
  from the symlink fails with `ModuleNotFoundError: No module named 'encodings'`)
- `export PATH="$HOME/.local/bin:$PATH"` appended to `~/.zshrc`
- The user generated an ed25519 key and added it to GitHub

**Resolved dependency versions** (from `poetry.lock`): darts 0.46.1,
chronos-forecasting 2.3.1, statsmodels 0.15.0, arch 8.0.0, torch 2.14.0,
transformers 5.16.1.

**A `.gitignore` bug caught before the first commit.** The pattern `data/` is unanchored
and therefore also matched `forecast_bench/data/`, which would have silently excluded the
entire data module from version control. Changed to `/data/`.

## Step 4 — Credential verification (2026-09-02)

Output of `poetry run python -m scripts.verify_setup`:

| Check | Result |
|---|---|
| FRED (`DGS10`) | PASS — last observation 2026-08-31 = 4.75 |
| Yahoo (SPY OHLC) | PASS — 10 bars, O/H/L/C all present |
| Chronos-2 on CPU | PASS — see latency below |
| Hugging Face access | PASS — authenticated as `rohanjain2312`, all 3 repos reachable |
| Space configuration | REVIEW — see hardware below |

### Chronos-2 CPU latency: **0.85 s** per forecast

Measured on Apple Silicon, `device_map="cpu"`, 512-step context to a 21-step quantile
forecast. Model load (cold, from local cache) takes a further 16.9 s and happens once at
Space startup, not per request.

**Consequence for the demo (DECISIONS.md D12):** comfortably under the 5 s threshold, so
the Space runs **live inference**. No caching layer and no pre-computed forecast grid are
needed. The Space's 2 vCPU will be slower than this machine, but there is roughly 6× of
headroom before the decision would change.

### Space hardware: **`zero-a10g`** (ZeroGPU) — needs changing to CPU Basic

The Space is on ZeroGPU, not CPU Basic. This is the one preflight item that was recorded
as unconfirmed, and it turns out to be wrong for this project.

Reading it required care. `SpaceRuntime.hardware` is `None` for a Space that has never
built (`stage=NO_APP_FILE`), because "current" hardware only exists once something has
run. The live setting is in `requested_hardware`. A check reading only `hardware` reports
`unknown` and passes the exact case it exists to catch, so `check_space_config` reads
`hardware or requested_hardware`.

Why this matters (DECISIONS.md D12): ZeroGPU charges GPU quota to the *visitor*, not the
owner. An unauthenticated visitor gets 2 GPU-minutes per day and queues behind PRO users.
Given the measured 0.85 s CPU latency, there is nothing to gain from a GPU and a real
failure mode to lose.

## Step 6 — Targets and covariates (2026-09-02)

### The Garman-Klass floor never fires on this data

`CLAUDE.md` lists "Garman-Klass can go non-positive on low-range days, and `ln()` then
yields NaN" as a known trap. Measured on SPY 2000-01-03 → 2026-09-02, 6,707 bars:

| | Count |
|---|---|
| Negative GK estimates | **0** |
| Exactly-zero GK estimates | **0** |
| Malformed bars (close outside the high-low range) | **0** |

The floor is a safety net that currently never fires. That is worth stating rather than
implying it is doing work.

**The estimator is also less fragile than the trap note suggests.** For a well-formed bar
(`H >= max(O, C)`, `L <= min(O, C)`) the Garman-Klass estimate cannot be negative: it would
need `|ln(C/O)| > 1.138 * ln(H/L)`, which is unreachable when the range contains both the
open and the close. The reachable degenerate case is *exactly zero*, on a fully flat bar.
Genuinely negative values require a malformed bar — a bad vendor tick. Both are floored,
but only the second indicates a data-quality problem. Pinned in `tests/test_targets.py`.

### A flooring bug caught by measuring instead of assuming

The first implementation used `variance.clip(lower=floor)`, which floors *everything*
below the training 0.1th percentile rather than only non-positive values as
`BUILD_ORDER.md` specifies. On real data that clipped **36 bars (0.54%)** — the quietest
valid days, such as 2013-12-24 with a range of 0.12% of price — and silently altered the
actuals every model is scored against. Corrected to `variance.where(variance > 0, floor)`.

Effect on the target: minimum log-RV moves from -12.88 (clipped) to -14.11 (true).

### `DGS10` is clipped to the study span

FRED serves `DGS10` from 1962. Unclipped, the processed artifact carried four decades that
no fold can reach and made covariate coverage look broken (`VIXCLS` "43% missing" is really
just VIX not existing before 1990). Both target builders now clip to `TRAIN_START`; the
full history remains in `data/raw/`.

### Built series

| Series | Rows | Span | Covariate gaps on the target's index |
|---|---|---|---|
| `spy_logrv` | 6,707 | 2000-01-03 → 2026-09-02 | `vixcls` 1 (0.01%), `dgs10` 50 (0.75%) |
| `dgs10` | 6,669 | 2000-01-03 → 2026-08-31 | `vixcls` 11 (0.16%), others 0 |

The `dgs10` gaps on the SPY index are bond-market holidays that are equity trading days.
Left as NaN — gap handling belongs inside the fold.

## Steps 10-11 — The harness (2026-09-02)

### Deviation: `predict()` takes the forecast index

`IMPLEMENTATION_PLAN.md` §3.1 specifies `predict(self, horizon: int) -> QuantileForecast`,
with the index carried inside the returned object. A model cannot construct that index. SPY
does not trade on every business day, so extending the training index by `BDay` puts a
forecast on a market holiday and silently misaligns every later date against the actuals.

Agreed with the user, 2026-09-02: the signature becomes

```python
def predict(self, horizon: int, index: pd.DatetimeIndex) -> QuantileForecast
```

`fit(train, origin)` is untouched, so the method carrying the study's central leakage
warning still reads exactly as specified. Supplying the calendar leaks nothing — an exchange
calendar is published years ahead, so the *dates* are known at the origin even though the
*values* are not, the same reasoning that makes day-of-week a legitimate covariate.

### Two xfails remain after Step 11, by dependency rather than by failure

Step 11's acceptance check says "no xfail remaining". Two of the five checks in §3.5 do not
depend on the harness at all:

| Check | Depends on | Becomes real at |
|---|---|---|
| 4 — MASE denominators and RW quantiles recomputed per fold | `models/naive.py`, `evaluation/metrics.py` | Step 12-13 |
| 5 — regime thresholds match the frozen config | `evaluation/regimes.py` | Step 13 |

They are `xfail(strict=True)`, so each will convert to a hard failure the moment its module
lands, forcing the marker to be removed deliberately. Checks 1-3 are live and enforced.

### The canary works, and it caught the runner first

Building the canary surfaced two real bugs before any model existed:

- `Fold` is frozen but **not hashable** — it carries a `slice` and a `DatetimeIndex` — so
  `run_backtest(return_fitted=True)` returns `list[tuple[Fold, dict]]`, not a dict.
- The first version of the leaked-frame test failed because `assert_fold_is_clean` fired
  correctly and aborted the run. The canary now measures the collapse with the guard
  explicitly disabled, and a separate test asserts the guard stops it. Measuring the damage
  and proving the guard are two different jobs.

**Detector calibration.** `assert_fold_is_clean` flags a training column whose absolute
correlation with the target at any lead 1-21 exceeds 0.999. Measured on the synthetic
fixtures: a benign lagged-rolling-mean covariate peaks at **0.905**, the injected
`target.shift(-21)` at **1.000**. The threshold is deliberately near 1.0 — this catches a
near-perfect copy, not a merely informative covariate. Preventing the latter is the
allowlist's job, not a correlation threshold's.

### Fold counts on real data

| Series | Folds | First origin | Last target date |
|---|---|---|---|
| `spy_logrv` | 137 | 2014-12-31 | 2026-06-11 |
| `dgs10` | 136 | 2014-12-31 | 2026-06-03 |

137 matches the figure in `DECISIONS.md` D6 exactly. The first fold's origin falls in the
previous calendar year, so block 2014 contains exactly one fold under the matched cadence.

## Step 14 — First end-to-end run (2026-09-02)

Arm A, matched cadence, both series. 137 folds on SPY (33 s), 136 on `DGS10` (35 s). Under
the matched cadence ARIMA refits 13 times rather than 137, so the AIC grid is cheap.

### Registered prediction 1 — HOLDS

*"On `DGS10`, no model will beat random walk by a meaningful margin at any horizon.
Expected skill scores in the range -0.05 to +0.05."*

Every WQL skill score on `DGS10` falls in **[-0.028, +0.031]**, comfortably inside the
registered band. The best any model manages is SeasonalNaive at h=1 with +0.031; ARIMA and
AR(1) are both marginally negative at every horizon.

### Registered prediction 2 — FAILS, and the cause is our implementation

*"On SPY log-RV, HAR or LogHAR will beat zero-shot Chronos-2 at h=1 and h=5"* — and Step 14's
weaker precondition, that HAR/LogHAR be clearly the strongest classical model.

Observed WQL skill versus random walk on SPY log-RV:

| Model | h=1 | h=5 | h=21 |
|---|---|---|---|
| ARIMA | **+0.074** | +0.129 | **+0.157** |
| LogHAR | +0.058 | **+0.136** | -0.101 |
| HAR | -0.070 | -0.143 | -0.290 |

LogHAR leads only at h=5. HAR is worse than a random walk everywhere.

**This is not HAR losing. It is our uncertainty quantification being wrong**, and the
coverage/width pair is what exposed it:

| Model | h | coverage 80% | width 80% | coverage 95% | width 95% |
|---|---|---|---|---|---|
| LogHAR | 1 | 0.438 | 1.98 | 0.679 | 3.03 |
| LogHAR | 21 | **1.000** | **9.07** | **1.000** | **13.86** |
| RandomWalk | 21 | 0.599 | 3.05 | 0.891 | 4.86 |

At h=21 LogHAR's intervals capture **100%** of actuals at a width three times the random
walk's. That is the exact failure mode `interval_coverage_and_width` exists to make visible:
coverage bought with useless width. Meanwhile LogHAR's *point* forecast beats the random
walk at every horizon (MAE skill +0.094 / +0.117 / +0.129). Good point forecast, unusable
intervals.

**Root cause.** `scaled_residual_quantiles` widens one-step residual quantiles by
`sqrt(h)`, as specified in `IMPLEMENTATION_PLAN.md` §4a. That scaling is correct for an
*integrated* process, where forecast error variance grows linearly in h. Log realized
variance is strongly **mean-reverting** — which is the entire reason HAR exists — so its
h-step error variance saturates. Measured on the 2000-2014 training window:

| h | std of h-step change | ratio vs h=1 | `sqrt(h)` assumes |
|---|---|---|---|
| 1 | 0.9446 | 1.00 | 1.00 |
| 5 | 1.0358 | 1.10 | 2.24 |
| 21 | 1.1578 | **1.23** | **4.58** |
| 63 | 1.2703 | 1.34 | 7.94 |

The intervals are inflated by a factor of roughly 3.7 at h=21. LogHAR's own MAE confirms it
independently: 1.180 at h=1 and 1.205 at h=21, a ratio of 1.02, not 4.58.

ARIMA is unaffected because it takes its intervals from `get_forecast().conf_int()`, so its
spread comes from the fitted dynamics rather than from an assumption imposed on top.

`HAR` (variance space) fails worse for a second, compounding reason: additive residual
quantiles in variance space are then log-transformed, and variance is strongly right-skewed,
so an additive symmetric offset is the wrong shape before the transform.

### Fix applied: measured h-step residuals replace the `sqrt(h)` assumption

Agreed with the user, 2026-09-02. `scaled_residual_quantiles` is replaced by
`stepwise_residual_quantiles`, and HAR/LogHAR now measure their own h-step-ahead errors by
iterating the fitted recursion forward from every training origin (vectorised across
origins). Effect on SPY log-RV:

| | before | after |
|---|---|---|
| LogHAR h=21 coverage 80% | 1.000 at width 9.07 | 0.555 at width 2.44 |
| LogHAR h=21 WQL skill | -0.101 | **+0.101** |
| HAR h=21 WQL skill | -0.290 | +0.026 |

Sanity check 2 still does not hold, but it now fails for a real reason rather than an
implementation one: ARIMA leads HAR/LogHAR on WQL at all three horizons
(+0.074 / +0.129 / +0.157 against +0.058 / +0.081 / +0.101). Whether that survives the
next finding is not yet known.

### A second, larger bug: the matched cadence was freezing conditioning data, not just parameters

Found by checking a property that should be true by definition — the random walk's median
forecast must equal the value at the forecast origin. It did not.

`IMPLEMENTATION_PLAN.md` §3.4's runner pseudocode builds and fits a model only on refit
folds and reuses the cached object otherwise. Because the cached object also holds its
*conditioning data*, every model under the matched cadence forecasts from whatever it last
saw at a block boundary. Measured on SPY log-RV, matched cadence, h=1:

| | value |
|---|---|
| Forecasts using a stale last value | **124 of 137** |
| Age of the carried-forward value | median **84 trading days**, max **231** |
| RandomWalk MAE, as run | 1.302 |
| RandomWalk MAE, conditioning on the origin | **0.848** |

The baseline is 54% worse than it should be, and every skill score in the headline table is
quoted against it. The same staleness applies to ARIMA, HAR and AR(1).

**Why this is a bug and not a design choice.** D5 defines the cadences in terms of "refit"
and "retrain" — parameter estimation. It says the matched cadence exists so that every model
class gets "the same information-refresh rate and the comparison is therefore about model
quality". Conditioning staleness is not model quality. Decisively, a zero-shot foundation
model has **no parameters to refit at all**: gating conditioning would hand Chronos-2 a
four-month-old context window and call the result a benchmark of Chronos. A random walk's
last value is state, not a fitted parameter.

The distinction the harness needs is between **parameters**, which the cadence governs, and
**conditioning data**, which must always run to the fold's origin.

### Fix applied: the cadence governs parameters, never conditioning data

Agreed with the user, 2026-09-02. `Forecaster.fit` gains a `refit_parameters` flag and is
now called on **every** fold; `BaseForecaster` splits into abstract `_estimate_parameters`
and `_update_state`, both required, so adding a model forces an explicit decision about
which attributes are learned and which are state. The statsmodels models use
`results.apply(..., refit=False)`, which keeps the coefficients and recomputes the filtered
state on the new sample.

`runner.assert_conditioned_on_origin` now fires on every fit, so this cannot recur
silently, and two regression tests pin it — one recording provenance at fit time, one
asserting the random walk's median equals the value at the origin under both cadences.
(Recording provenance *after* the run does not work: the block cadence reuses one object
across a block, so every reference to it shows the final fold's state. The first version of
that test got this wrong and caught itself.)

### Results after both fixes — the registered predictions now hold

`RandomWalk` median equals the value at the origin for all 137 SPY and 136 `DGS10` origins.
Interval coverage across the panel is 0.68-0.80 against a nominal 0.80, mild under-coverage
rather than the previous 1.00-at-triple-width.

**Registered prediction 1 — HOLDS.** On `DGS10` nothing beats the random walk: the best
skill score anywhere is AR(1) at h=1 with **+0.0018**, and ARIMA and AR(1) sit within
[-0.028, +0.002] at every horizon. `SeasonalNaive` is far *worse* (-1.065 at h=1), which is
outside the registered ±0.05 band but in the direction the prediction asserts — repeating
last week's yield is a poor one-day forecast of a near-unit-root series. The prediction's
claim was that nothing would *beat* the random walk, and nothing does.

**Sanity check 2 — now HOLDS.** LogHAR is the strongest or effectively tied-strongest
classical model at every horizon on SPY log-RV:

| Model | h=1 | h=5 | h=21 |
|---|---|---|---|
| LogHAR | +0.166 | **+0.219** | **+0.212** |
| ARIMA | **+0.168** | +0.210 | +0.205 |
| HAR | -0.041 | +0.039 | +0.087 |

At h=1 ARIMA leads by 0.0015 on WQL while LogHAR leads on MAE skill (+0.190 vs +0.183).
HAR, fitted in variance space, stays the weaker of the two HAR variants, which is what the
log-target specification predicts.

**Both bugs inflated results in opposite directions and neither was visible in a headline
number.** The stale conditioning made every model look better by crippling the baseline;
the `sqrt(h)` intervals made HAR look worse than it is. Each was found by checking a
property that had to be true by definition rather than by looking at a metric.

## Step 15 — Foundation models, zero-shot (2026-09-02)

Both series, Arm A, matched cadence, seven models. SPY 53 s, `DGS10` 57 s — the pipeline
cache means the ~17 s Chronos-2 weight load happens once per process rather than once per
parameter refit.

### Chronos-Bolt cannot produce the study's tail quantiles

Bolt was trained on levels 0.1-0.9 only. Requesting 0.025 and 0.975 returns its 0.1 and 0.9
predictions unchanged, so **Bolt's 95% interval is identical to its 80% interval by
construction**.

This is left in place rather than worked around. Extrapolating tails the checkpoint was
never trained to produce would be inventing a capability to make a number look better. It
does cost Bolt something real on the primary metric: weighted quantile loss averages over
all eleven levels and two of Bolt's eleven are duplicates, so it is penalised at the tails
relative to models with genuine tail predictions. Asserted in
`tests/test_foundation_zeroshot.py` so that a future checkpoint gaining real tails fails the
test rather than silently invalidating the limitation text.

Chronos-2 reports trained quantiles from 0.01 to 0.99, so the study grid is inside its range
and every level is a genuine prediction.

### Zero-shot results, SPY log-RV, WQL skill vs random walk

| Model | h=1 | h=5 | h=21 |
|---|---|---|---|
| LogHAR | 0.166 | **0.219** | **0.212** |
| ARIMA | **0.168** | 0.210 | 0.205 |
| Chronos2-ZeroShot | 0.151 | 0.187 | 0.184 |
| ChronosBolt-ZeroShot | 0.127 | 0.166 | 0.188 |

**Registered prediction 2 holds so far:** LogHAR beats zero-shot Chronos-2 at h=1 and h=5,
and at h=21 as well. The full check waits on the fine-tuned model at Step 18.

**Registered prediction 4 is in trouble.** It says the foundation model's relative position
improves as the horizon lengthens. The LogHAR-minus-Chronos-2 gap goes 0.015 at h=1 to 0.027
at h=21 — it widens. Recorded now, evaluated properly at Step 18 with the fine-tuned model,
which is what the prediction is actually about.

On `DGS10`, Chronos-2 zero-shot posts the only positive skill anywhere (+0.008 at h=1), well
inside the registered ±0.05 band. Prediction 1 continues to hold.

### Contamination-free cut: 7 origins

Restricting to origins after Chronos-2's 2025-11-01 release leaves **7 folds**, close to the
~8 anticipated in DECISIONS.md D10-G4. `n_origins` is a column in the table itself, not a
prose footnote, so no number can be quoted from it without its sample size attached. On that
cut Chronos-2 leads at h=21 (+0.367) while LogHAR leads at h=1 and h=5 — at n=7 this is
descriptive only and no claim rests on it.

### Published to the Hub

- `forecastbench-data`: both processed series plus a dataset card with `license: mit`
- `forecastbench-chronos`: model card with `license: apache-2.0`, written **before** any
  weights exist. Chronos is Apache 2.0 and a LoRA derivative inherits it, so a repo holding
  weights without that field would be an unlicensed redistribution.

## Step 16 — Fine-tuning recipe and Colab notebooks (2026-09-02)

### Two fine-tuning paths, because the two checkpoints are genuinely different

`Chronos2Pipeline` exposes an official `fit(finetune_mode="lora", lora_config=...)`, so
Chronos-2 uses it. `ChronosBoltPipeline` has **no** `fit` at all, so Bolt takes the standard
`transformers` + `peft` route with an explicit loop — which is exactly the split
`DECISIONS.md` D13 predicted. Bolt's `forward` returns its own quantile loss when handed a
target, so the loop optimises the model's native objective rather than a reconstruction of
it. Two independent paths means neither is a single point of failure.

Recipe as pre-registered: rank 8, alpha 16, dropout 0.05, targeting `q`/`k`/`v`/`o`
projections, early stopping with patience 3 on a validation slice taken from the **end** of
each block. Trainable-parameter counts are logged and written into the run metadata pushed
alongside each checkpoint.

### `pytorch-lightning` was missing

darts raised "The (Py)Torch module could not be imported" despite torch 2.14 being present:
darts' torch models need Lightning, which nothing had pulled in. Added to the main
dependencies rather than the `gpu` group, so the neural models are importable locally and
covered by the protocol tests.

### darts hands float64 to MPS, which cannot take it

`TimeSeries.from_values` defaulted to float64, and torch's MPS backend rejects that dtype
outright. Cast to float32 in `to_timeseries`. Not a platform workaround — float32 is the
standard dtype for neural training on every backend.

### Orchestration moved into the package

`scripts/run_backtest.py::run` became `backtest/runner.py::run_series_backtest`. Colab
installs the package with pip and never gets `scripts/`, so a notebook could not have called
the CLI's logic — it would have had to reimplement it, which is the drift this project's
notebook rule exists to prevent.

### Notebook conventions are enforced by tests, not by discipline

`tests/test_notebooks.py` asserts that each Colab notebook parses, contains no loop over
folds, imports `forecast_bench` for its heavy work, precedes every code cell with a plain-
language markdown cell, opens with its Hub prerequisites, stores no outputs, and contains no
literal secret. A convention nobody checks is a convention that decays.

### Colab hit two setup problems live, both fixed

**`pip install git+https://...` failed with exit 128 inside the Colab runtime**, even
though the repo is genuinely public (confirmed with an anonymous `git ls-remote` from
outside Colab). Root cause not fully diagnosed — likely a transient git-subprocess issue in
that container — but the fix is more robust regardless: install from the GitHub tarball URL
(`.../archive/refs/heads/main.tar.gz`), which uses pip's own HTTP fetcher instead of
shelling out to `git clone`.

**`get_config()` is a process-wide `lru_cache` singleton, and the first Colab run poisoned
it.** Something called `get_config()` before the credentials cell set `os.environ` — most
likely a re-run of the hub-check cell before the earlier `ModuleNotFoundError` was fixed, in
the same kernel. Once cached, the object holds empty secrets **for the rest of the
session**, and no subsequent `os.environ` assignment can reach it. `os.environ` is fine for
a script invoked fresh each time; it is the wrong default for a notebook whose credential
cell runs at a UI-controlled moment the module cannot see.

Fixed two ways:

1. Both Colab notebooks now call `get_config.cache_clear()` immediately after loading
   secrets from `userdata`, so the notebook is correct regardless of what ran before that
   cell — this is a defensive fix, not just an incident-specific patch.
2. `tests/test_config.py` reproduces the exact failure (`test_env_var_set_after_first_call_is_invisible_without_a_cache_clear`)
   and pins the fix (`test_cache_clear_picks_up_a_newly_set_env_var`), so a future change to
   `get_config()`'s caching strategy has to consciously address the notebook contract
   rather than silently reintroduce the trap.

### peft refuses to import unless torchao is at least 0.16.0

Colab preinstalls an old `torchao` (0.10.0). `peft` performs a hard version check at import
time in `peft.import_utils.is_torchao_available()` and raises `ImportError` if it fails —
even though this project's LoRA recipe (attention-projection targeting, no quantization)
never touches torchao's actual functionality. `pip install peft` does not force an upgrade
of an already-present-but-outdated dependency, so the stale version survives the install
cell silently until the first `import peft`, several cells later.

Fixed by adding `%pip install -q -U "torchao>=0.16.0"` to notebook 04's install cell.

### Both notebooks moved from `git+https://` to the source tarball

`pip install git+https://github.com/...` failed inside the Colab container with exit 128
even against a confirmed-public repo (verified with an anonymous `git ls-remote` run
outside Colab). `pip`'s git codepath shells out to the container's git binary; installing
from `.../archive/refs/heads/main.tar.gz` instead uses pip's own HTTP fetcher and sidesteps
whatever was wrong with git in that runtime. Both notebooks now use the tarball URL.

### `TRAINING_WINDOWS` conflated "raw days" with "usable training examples"

Hit live on the sample-efficiency sweep, first block, `"1y"`: `ValueError: Training window
'1y' at 2014-12-31 has 252 observations, fewer than the 533 needed for one example.`

`TRAINING_WINDOWS = {"1y": 252, ...}` treated "1 year" as 252 literal trading days. But
`CONTEXT_LENGTH = 512` is fixed across every model in the sweep — Chronos-2/Bolt
fine-tuning, N-BEATS, the DeepAR-class LSTM — specifically so that context length is not a
confound (IMPLEMENTATION_PLAN.md §4c). A 252-observation slice is *shorter than the context
window itself* and cannot supply a single `(context, target)` example, on any model.

**The same bug was latent in `models/neural/_darts.py`.** Notebook 05's sample-efficiency
cell hardcoded the identical `{"1y": 252, ...}` dict and passed it straight to
`training_window_days`, which `DartsQuantileForecaster` also slices against a fixed
512-step `input_chunk_length`. It had not failed yet only because the user had not reached
that cell — it was the same defect, waiting.

**Fix:** moved the concept to `models/base.py` as `SAMPLE_EFFICIENCY_DAYS` (unchanged
labels: 252/756/2520/None) plus `sample_efficiency_window_size()`, which resolves a label
to `context_length + horizon + days - 1` — enough raw observations for `days` *distinct
forecast origins* beyond the one context-plus-horizon window every slice needs at minimum.
`"1y"` now means 252 separate training examples, not 252 raw observations.

**A second bug fixed alongside it:** notebook 05 had re-hardcoded the training-window dict
instead of importing it from the package — exactly the notebook-drift the project's own
rule prohibits ("no modelling logic in notebooks; every heavy call is an import"). It now
imports `SAMPLE_EFFICIENCY_DAYS` and `sample_efficiency_window_size` directly, so there is
one definition instead of two that could silently diverge.

`tests/test_sample_efficiency.py` pins the minimum-example-size invariant for every finite
window and reproduces the exact failing call (first block, `"1y"`) as a regression test.

### Revision tags collided across models: Bolt fine-tuning silently pushed nothing

Hit live at the end of notebook 04: Step 8 (Chronos-Bolt) reported success, no error, and
the run finished — but `existing_hub_revisions()` afterward showed **zero** Bolt
checkpoints on `forecastbench-chronos`.

`revision_tag(series, arm, block, training_window)` had no model axis, so Chronos-2's and
Chronos-Bolt's full-window tags for the same `(series, arm, block)` were byte-identical:
both resolved to `"spy-logrv-armA-2014-full"`. Step 5 (Chronos-2) pushed those 13 tags
first. When Step 8 (Bolt) ran, every tag it computed already existed on the Hub — from a
different model entirely — so `run_campaign`'s resume check (working exactly as designed)
skipped all 13 blocks as "already done." No fit call, no push, no error: the run looked
identical to a fully successful one.

This is a gap in `IMPLEMENTATION_PLAN.md` §4c itself, which specifies "one revision tag per
(series, arm, block, training-window) combination" — a four-tuple that has no room for
which base checkpoint the adapter belongs to, even though D13 requires fine-tuning two
different base models. Fixed by adding `model: str = "chronos2"` to `revision_tag()`,
defaulting to the value that keeps every already-pushed Chronos-2 tag unchanged.

`tests/test_foundation_hub.py` reproduces the exact failure end-to-end: a `run_campaign`
call for Bolt, against a Hub state carrying only Chronos-2's tags for the same blocks, now
asserts the fit function is actually called for every block rather than silently skipped.

**Consequence:** Step 8 needs to be re-run. Nothing was lost — the Chronos-2 checkpoints
from Steps 5-7 are genuine and unaffected, since their tags were correct all along; only
the Bolt run produced no artifacts and needs to happen again, now that its tags are
distinct.

### Reopening from GitHub did not pick up the tag-collision fix, and this was the real bug

After the fix above, the user reopened notebook 04 fresh from GitHub and ran all cells.
Step 8 still logged the **old** tag format (`spy-logrv-armA-2014-full`, no model name) and
finished with zero real Bolt checkpoints — the exact same failure, after a fresh reopen.

The install cell was `%pip install -q "<tarball-url>"` with no reinstall flags. pip sees
`forecast-bench` already installed at version `0.1.0` — a version number that has never
changed — and silently skips reinstalling, regardless of whether the tarball's contents
changed underneath that version string. Colab reconnected the "freshly opened" notebook to
the same live backend runtime rather than allocating a new VM, so the stale, pre-fix
package kept running under a notebook that looked freshly reopened. Confirmed by re-checking
the Hub: still 0 Bolt checkpoints, and the 13 pre-fix tags were exactly what got matched.

**This means every fix pushed to the repo during this session was at risk of silently not
reaching a "freshly reopened" notebook**, which defeats the entire point of the reopen
workflow. Fixed by adding `--force-reinstall --no-deps` to both notebooks' install cells:
force-reinstall guarantees the package files on disk are always current regardless of what
pip thinks is "already satisfied"; no-deps keeps it fast by leaving torch, darts and
chronos-forecasting untouched.

`tests/test_notebooks.py::test_install_cell_forces_a_fresh_reinstall` pins this so it
cannot silently regress again.

**Consequence:** Step 8 must be re-run once more. Steps 5-7's checkpoints remain correct
and unaffected.

### `--force-reinstall` was not enough either — cache-busted the install URL

The user reopened notebook 04 fresh from GitHub a **second** time, after the
`--force-reinstall --no-deps` fix landed, and still got zero real Bolt checkpoints with the
old pre-fix tag format in the logs.

Verified directly that GitHub itself was not the problem: a fresh `curl` of the exact
tarball URL used by the notebook, run from outside Colab, returned the current
`revision_tag()` with the `model` parameter already in it. So the fix was genuinely
published and reachable — the staleness was happening somewhere between Colab and GitHub
(most likely pip's own HTTP response cache, keyed by URL; `--force-reinstall` forces pip to
*reinstall*, but does not by itself force pip to *re-fetch* if it believes it already has a
cached response for that exact URL).

Since the exact caching layer at fault could not be directly inspected, the fix defeats the
class of problem rather than one specific cache: the install line now appends a
`?_cb=<unix-timestamp>` query parameter computed fresh at cell-execution time, so every
run requests a **different URL** and no cache anywhere in the path — pip's, a proxy's, or
otherwise — has anything to serve.

**Also added a loud, immediate verification cell** right after install in notebook 04: it
asserts `revision_tag`'s signature carries the `model` parameter and raises with an
explicit "Restart session" instruction if not. This turns any future recurrence of this
class of problem — in this fix or a later one — into an obvious failure at cell 2, not a
silent no-op discovered only by checking the Hub after a full campaign finishes.

**Also bumped the package version, 0.1.0 → 0.2.0**, as defense in depth: a real version
change means `pip install <url>` correctly detects the update on its own even without
`--force-reinstall`, for any future run of this or another notebook.

### `--no-deps` broke the fresh-VM case, and only a fresh VM could reveal it

The user used "Disconnect and delete runtime" to guarantee a genuinely new VM, reopened
notebook 04, and cell 2b failed with `ModuleNotFoundError: No module named
'pydantic_settings'`.

Cause: the install cell used `--force-reinstall --no-deps` on its *only* install line.
`--no-deps` installs `forecast_bench` and none of its dependencies. Every previous run had
reused a runtime where pydantic-settings, darts and chronos-forecasting were already
present from the original `git+https://` install, so the flag looked harmless. On a truly
fresh VM the package installed with no dependencies at all and died on its first import.

The irony is that this bug was only reachable *because* the earlier advice to force a
brand-new VM finally worked — every prior "fresh reopen" had silently reconnected to the
old runtime and masked it.

**Fix:** two install lines, both load-bearing.

1. `pip install -q "<url>"` — with dependencies. This is what makes a fresh VM work.
2. `pip install -q --force-reinstall --no-deps "<url>"` — the package alone, forced. pip
   will not reinstall a package it considers version-satisfied, so this is what guarantees
   current *code* on a runtime that has seen an earlier version. `--no-deps` keeps it fast.

The `?_cb=<timestamp>` cache-bust applies to both.

A second defect was introduced and caught in the same edit: the automated rewrite of the
install cell silently dropped the `peft`, `accelerate` and `torchao` lines, because its
tail-detection matched on a literal `main.tar.gz` that no longer appears in the rewritten
`pip install` lines (they interpolate `{_tarball_url}` instead). Restored, and pinned by
`test_finetune_notebook_still_installs_peft_and_torchao`.

Both failures are now covered: `test_install_brings_in_dependencies` asserts at least one
install line runs without `--no-deps`, and `test_install_also_forces_the_package_itself_current`
asserts one runs with it.

### The tag fix orphaned 65 existing checkpoints, and the resume check dutifully retrained them

Caught only because the user noticed Step 5 *training* when it should have skipped.

Adding the `model` axis to `revision_tag()` changed the tag format from
`spy-logrv-armA-2014-full` to `spy-logrv-armA-chronos2-2014-full`. The 65 Chronos-2
checkpoints already on the Hub were all under the old format, so `existing_hub_revisions()`
found no match for any newly-computed tag and `run_campaign` began refitting every block
that had already been done — roughly 70 minutes of redundant H100 time across Steps 5-7,
about to be spent for nothing.

The resume logic was not wrong; it was working exactly as designed against a tag vocabulary
that had silently shifted underneath it. **Changing an identifier format is a migration, not
just a code change** — that consequence should have been stated when the fix was made,
rather than discovered by watching a training bar move.

**Resolved without retraining anything:** all 65 old-format branches were remapped by
creating the new model-qualified branch name from the same revision
(`HfApi.create_branch(repo, branch=new, revision=old)`). Additive and non-destructive — the
old names still exist, so nothing is broken for anything that referenced them. The repo now
carries 131 revisions: 65 old-format, 65 new-format pointing at identical commits, plus
`main`.

### Chronos-Bolt LoRA failed: peft assumes every model has token embeddings

First real Bolt fine-tuning run died immediately with
`NotImplementedError: get_input_embeddings not auto-handled for
ChronosBoltModelForForecasting`.

`peft.get_peft_model()` calls `get_input_embeddings()` while preparing any model — a
reasonable assumption for the text models peft was built for, and false for a time-series
model with no token vocabulary. `ChronosBoltModelForForecasting` inherits the
`PreTrainedModel` stub that raises rather than overriding it. It does still carry T5's
`shared` module, which is exactly what the base class would have returned.

Fixed with `_with_input_embedding_accessors()`, which binds `get_input_embeddings` and
`set_input_embeddings` to that module before the peft call. This satisfies peft's
preparation step without changing what is trained: LoRA targets the attention projections
only, so nothing in the recipe reads or writes input embeddings.

**This path was the least-covered code in the project and it showed.** `DECISIONS.md` D13
predicted Bolt would need the standard `transformers` + `peft` route while Chronos-2 had its
own `fit()`, and that asymmetry is precisely why Bolt broke and Chronos-2 did not. It was
also the only path that could not be exercised locally, since `peft` lives in the optional
`gpu` group.

Now covered: `test_bolt_gains_the_embedding_accessors_peft_requires` asserts the raw model
raises and the patched one returns `shared`, and
`test_bolt_finetune_produces_a_real_lora_adapter` runs a genuine six-step CPU fine-tune and
checks a loadable adapter is written (589,824 trainable parameters, 1.22% of the model).
Both skip cleanly when `peft` is absent; `poetry run pip install peft` enables them.

### `regimes.yaml` raised on absence, breaking every pip-installed consumer

Notebook 05, Step 4: `FrozenThresholdError: /usr/local/lib/python3.13/dist-packages/
experiments/configs/regimes.yaml is missing.`

Only `forecast_bench/` ships in the wheel — confirmed by building it and listing the
contents: **47 files, zero `.yaml`, no `experiments/` at all.** So the path
`PROJECT_ROOT / "experiments" / "configs" / "regimes.yaml"` cannot resolve in any installed
environment, including Colab and the Hugging Face Space.

This is the second instance of one bug. `config.py::_verify_base_yaml_agrees` already
handles exactly this for `base.yaml` — its docstring even spells out the reasoning
("A *missing* file is not an error... only `forecast_bench/` ships") — and I did not apply
the same treatment to `regimes.yaml` when writing it. The two now behave identically.

Absence is safe because `EXPECTED_CALM_UPPER`/`EXPECTED_NORMAL_UPPER` **are** the frozen
values, duplicated into the module deliberately. The YAML is a cross-check that makes any
change to them appear in a diff, not the source of truth. A missing file cannot silently
recompute anything; only a present-and-different file could, and that still raises —
pinned by `test_altered_thresholds_still_raise_even_though_absence_does_not`.

Verified by building the wheel, installing it into a clean venv with no repository
anywhere near it, and importing `forecast_bench.evaluation.regimes` from `/tmp`: constants
load as 15.9 / 22.5582 and regime assignment works.

### Notebook 05: Tensor Cores enabled, and two problems found in a dry run

Step 4 was taking over two hours on an A100, so `torch.set_float32_matmul_precision` was
enabled at the user's request. Rather than a bare line in a cell, this landed as
`config.enable_tensor_cores()`, off by default and switched on explicitly in notebook 05.

The reason it is opt-in rather than always-on: it changes numerics, so it must apply
**uniformly across everything being compared**. The sample-efficiency sweep retrains the
same models on nested windows and its `full` point is supposed to reproduce the headline
run; enabling TF32 partway through would quietly break that. Notebook 04's Chronos
fine-tuning ran without it, which is fine — the two notebooks produce separate model
families compared through forecasts, not weights.

Before handing the notebook back, its entire code path was executed locally on CPU at
reduced scale (4,000 synthetic observations, 1 epoch, context 64). That surfaced two things
neither the tests nor a reading of the notebook had caught.

**1. Step 6 re-ran Step 4 identically, for hours.** `sample_efficiency_window_size("full")`
returns `None`, meaning no truncation — so the sweep's `full` iteration was a byte-identical
repeat of the Step 4 call. The sweep now reuses Step 4's in-memory result (falling back to
its saved parquet in a fresh kernel), which removes roughly two hours of GPU time *and*
guarantees the curve's endpoint is exactly the headline number rather than a second run
that could drift from it.

**2. The sweep would have corrupted the headline table.** `load_forecasts()` globs every
parquet in `forecasts/`, and the sweep file contains a full duplicate of the headline run
in its `full` slice. Every headline metric would have been averaged over duplicated rows
and `n_origins` inflated — silently, with no error anywhere.

Fixed by separating the sweep into `forecasts/sample_efficiency/`, which the deliberately
non-recursive headline glob then excludes by construction; `push_forecasts` switched to
`rglob` so the sweep still publishes, and `build_results` scores it through a dedicated
`load_sample_efficiency()`. Verified end to end: the headline loader returns exactly the
Step 4 row count with the sweep on disk. Pinned by
`test_sweep_directory_is_excluded_from_the_headline_glob`.
