# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

# forecast_bench — CLAUDE.md

## How To Work In This Repo — read this before anything else

**`docs/planning/BUILD_ORDER.md` is your instruction set and your progress tracker.**

Every session, without being asked:

1. Read this file, then `docs/planning/DECISIONS.md`,
   `docs/planning/IMPLEMENTATION_PLAN.md`, `docs/planning/REPO_STRUCTURE.md`.
2. Open `BUILD_ORDER.md` and find the **first unticked checkbox**. That is the current step.
3. Do that step only. Run its acceptance check. Commit with the given message. Tick the box
   and commit that.
4. If the step says `GATE: AUTO`, continue to the next step immediately — do not ask
   permission. If it says `GATE: STOP`, print the step's "Tell the user" message verbatim
   and wait.

**Do not ask the user which step to do.** The unticked box is the answer.
**Do not ask permission to continue** at an AUTO gate.
**Do not batch steps** — one step, one acceptance check, one commit.

If reality contradicts the plan (a library API differs, a result violates a stated
assumption), stop, explain the discrepancy in one paragraph, propose a fix, and wait. Do
not silently improvise around the plan.

The user's manual tasks are listed in `docs/planning/MANUAL_TASKS.md`. The STOP gates in
`BUILD_ORDER.md` already contain the exact wording to give them — use it as written rather
than paraphrasing.

## What This Project Is

An honestly-reported forecasting benchmark: classical statistical models vs. a fine-tuned
time-series foundation model, on real financial data, with a leakage-safe backtest that
scores every model through identical code.

The deliverable is **not** "foundation models are better." It is a map of where each model
class wins and loses, with a pre-registered definition of what losing looks like.

**Two targets:**
- Primary: SPY log realized variance (Garman-Klass estimator from daily OHLC)
- Contrast: `DGS10` 10-year Treasury yield, in levels

**Model panel:** RandomWalk, SeasonalNaive, ARIMA, SARIMA(X), HAR/LogHAR, AR(1), N-BEATS,
DeepAR-class LSTM, Chronos-2 (zero-shot + LoRA fine-tuned), Chronos-Bolt-small (both),
TimesFM 2.5 (zero-shot, stretch).

## Repo & Paths

- Local: `/Users/rohanjain/Desktop/Projects/forecast_bench`
- GitHub: `https://github.com/Rohanjain2312/forecast_bench.git`
- HF model: `rohanjain2312/forecastbench-chronos`
- HF dataset: `rohanjain2312/forecastbench-data`
- HF Space: `rohanjain2312/forecastbench-demo`
- **Repo and local folder are already linked. NEVER run `git init` or `git remote add`.**
- **NEVER create files outside the local project path above.**

### Already provisioned — verified 2026-09-02, never ask the user to create these

HF PRO is active. The write token and FRED key exist. The GitHub repo exists but is
**completely empty** (no branches — the first push needs `git branch -M main &&
git push -u origin main`). All three HF repos exist and are empty; the Space is on the
Gradio SDK with no `app.py` yet, and its hardware is checked automatically in build Step 4.

The user's only remaining manual tasks are Steps 4–8 of `docs/planning/MANUAL_TASKS.md`,
and each one is triggered by a STOP gate in `BUILD_ORDER.md`. Do not invent others.

## Stack

- Python 3.11, Poetry
- `darts` — ARIMA, N-BEATS, probabilistic RNN (DeepAR-class)
- `statsmodels` — HAR (OLS), SARIMAX, AR(1)
- `chronos-forecasting` — Chronos-2 and Chronos-Bolt
- `peft` — LoRA fine-tuning (rank 8, alpha 16)
- `arch` — Model Confidence Set
- `pydantic-settings` — config
- `gradio` — HF Space demo (5 tabs, CPU Basic)
- Weights & Biases — experiment tracking
- Colab H100 — the only GPU-requiring work is fine-tuning and neural training

## Non-Negotiable Conventions

- Black formatting, line length 88; isort; ruff
- Google-style docstrings on ALL public classes and functions
- Type hints on ALL function signatures
- Library code uses Python `logging`, NEVER `print`. Notebooks use `print` + `tqdm`.
- NEVER hardcode secrets — everything via `.env` through `forecast_bench/config.py`
- NEVER commit `.env`, `data/`, `*.parquet`, `*.pt`, `checkpoints/`
- All heavy compute (fine-tuning, neural training) → Colab, not the local Mac

## The Five Rules That Are Specific To This Project

These are the ones that, if broken, silently invalidate the study. They matter more than
the style rules above.

1. **Every model implements `forecast_bench/backtest/protocol.py::Forecaster`.** ARIMA and
   Chronos-2 must be indistinguishable to `runner.py`. If a model needs special handling in
   the runner, the abstraction is wrong — fix the abstraction, not the runner.

2. **`fit()` may only read data at or before `origin`.** No fitted object may cross a fold
   boundary. This includes scalers, ARIMA order selections, MASE denominators, residual
   quantiles, and regime thresholds. `tests/test_no_leakage.py` enforces this and must
   never be weakened or skipped to make a run pass.

3. **`evaluation/metrics.py` is the single source of truth for every metric.** Import it.
   Never reimplement a metric in a notebook, a script, or `space/app.py`. The moment two
   definitions of MASE exist, one is wrong and nobody will know which.

4. **Only non-revised daily FRED series may appear in `data/covariates.py`.** The allowlist
   is `DGS10, DGS3MO, T10Y2Y, VIXCLS, DFF`. Adding `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, or
   monthly `GS10`/`GS3M` introduces look-ahead bias, because FRED indexes those by
   reference period, not release date. This is not a style preference — it is the
   study's central claim.

5. **`experiments/configs/regimes.yaml` contains frozen VIX tercile thresholds computed on
   pre-2015 data only. Never recompute them.** `evaluation/regimes.py` asserts against the
   committed values at import time.

## Key Files

| File | Role |
|---|---|
| `forecast_bench/config.py` | Single source of truth for all settings |
| `forecast_bench/backtest/protocol.py` | The `Forecaster` interface — read this first |
| `forecast_bench/backtest/runner.py` | The harness; the most important file in the repo |
| `forecast_bench/evaluation/metrics.py` | All metric definitions |
| `forecast_bench/models/registry.py` | Where the model panel is defined |
| `tests/test_no_leakage.py` | The hard constraint, made executable |
| `PREREGISTRATION.md` | Committed at build Step 2. Never edit after that commit — deviations go in its Amendments section. |
| `DECISIONS.md` | Why every design choice is what it is |

## Backtest Design (do not change without reading `docs/methodology.md`)

- Expanding-origin walk-forward, train from 2000-01-01, test 2015-01-01 → 2026-06-30
- **Stride = 21 = max horizon → forecast windows are non-overlapping.** This is what makes
  the Diebold-Mariano test valid. Changing the stride invalidates `evaluation/stats.py`.
- Horizons `{1, 5, 21}` are read off steps 1, 5, 21 of a single 21-step forecast path
- Two refit cadences, both reported: `matched` (all models, annual blocks — headline) and
  `native` (classical refit every fold — secondary)
- Quantile grid: `[0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]`

## Common Commands

```
poetry install                               # install
poetry run pytest tests/                     # test suite
poetry run pytest tests/test_no_leakage.py   # the one that matters
poetry run black . && poetry run isort . && poetry run ruff check .
poetry run python -m scripts.fetch_data --config spy_logrv
poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A
poetry run python -m scripts.build_results
poetry run python -m scripts.push_artifacts --target dataset
```

## Known Traps

- **Garman-Klass can go non-positive** on low-range days, and `ln()` then yields NaN. Floor
  at the 0.1th training-window percentile and log how often it fires.
- **yfinance returns a MultiIndex** for multi-ticker downloads and its adjusted-close column
  name varies. `data/yahoo_client.py` handles both; do not bypass it.
- **`DGS10` has NaNs on market holidays.** Do not forward-fill across them into a model
  input without recording it — forward-filling a target is a subtle leak.
- **Chronos-2 is not a vanilla `transformers` T5.** Use `Chronos2Pipeline`, not
  `AutoModelForSeq2SeqLM`. `chronos-bolt-small` is the one that takes the standard
  `transformers` + `peft` path.
- **Zero-shot foundation-model results before ~Oct 2025 may be contaminated** by
  pretraining exposure to public financial series. This is documented in
  `docs/limitations.md` and is why the post-release-only results table exists. Never
  present a pre-2025 zero-shot number as clean out-of-sample.
- **Do not call `darts.historical_forecasts`.** Every model, including the darts ones, must
  be driven fold-by-fold by our runner so all models traverse identical code.
