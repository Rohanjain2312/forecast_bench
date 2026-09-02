# Project Brief: Financial Time-Series Forecasting Benchmark
### Classical Models vs. Fine-Tuned Time-Series Foundation Models

---

## Instructions for the planning AI

Use this brief to produce an in-depth implementation plan before any code is written. Specifically:

1. Work through every open item in **Section 8 (Key Decisions to Resolve First)** and propose a concrete resolution for each, with reasoning — don't leave them as open questions.
2. Expand Sections 4–7 into an actual technical implementation plan (file-by-file where useful), but do not attach a timeline, week count, or day count to anything.
3. Flag any decision in Section 8 that has downstream consequences on other decisions (e.g., choice of target series affects which covariates are available, which affects whether ChronosX-style covariate injection is needed).
4. Where a decision has more than one reasonable answer, briefly note the trade-off rather than silently picking one.
5. Preserve the "honest comparison" framing throughout — the deliverable is a credible study, not a piece that's engineered to make the foundation model win.
6. Respect the environment and tooling constraints in **Section 3** exactly as stated — every step of the plan should be doable with a local CPU-only machine plus the free-tier cloud tools listed, nothing else. Use the exact project identifiers (repo URL, HF repo names, local folder path) given in Section 3 — they are already decided, do not propose alternatives.
7. Treat the demo (Section 9) as a **required** deliverable, not optional, and design it as the primary recruiter-facing artifact — see Section 9 for what it needs to show.
8. Produce a complete, ordered **manual setup checklist** per Section 10 — every account, API key, token, and secret the project owner needs to generate before any code runs, and where each one needs to be placed (local `.env`, Colab secrets, HF Space secrets, etc.).
9. Produce the **actual folder and file structure** for the repo — real file names and directory layout, not just the category list in Section 9.
10. Apply the documentation standard in **Section 11** to every file and notebook the plan calls for.
11. **When genuinely in doubt, ask — don't decide unilaterally.** Section 8 asks for reasoned proposals on each decision, which is different from silently picking and moving on: where a call is close, ambiguous, or has a materially different downstream path depending on the answer, surface it as a question back to the project owner rather than resolving it solo. Batch these into **one consolidated list of questions per planning pass**, not one question at a time — the goal is a single round of back-and-forth, not a drawn-out interrogation.

---

## 1. Core Idea

Build a rigorous, honestly-reported benchmark comparing classical statistical forecasting models against a fine-tuned time-series foundation model, on real financial data. The deliverable is not "foundation models are better" — it's a clear map of *where* each model class wins, loses, and why, backed by proper backtesting.

This extends existing work (`Recession_BondsYield`, the market-regime transformer projects) rather than starting from zero — same data sources, same point-in-time data discipline, same "publish the honest result, not the flattering one" writeup style already used for GraphBench.

## 2. Why This Project

- Closes a real, currently-empty skill area (classical + modern forecasting) with a single coherent project rather than several disconnected notebooks.
- The fine-tuning workflow (LoRA/PEFT on a pretrained model, eval harness, Colab-based training) is a direct rerun of the pattern already proven on the Mistral and Qwen projects — it's an existing skill pointed at new data, not a new skill from scratch.
- Time-series foundation models (Chronos, TimesFM, Moirai) are an active 2026 research area with finance-specific papers already published — this is a live topic, not a stale one.
- Volatility/rate forecasting is directly relevant to the kind of quantitative reasoning a finance-adjacent technical assessment (options pricing, statistics) would probe.
- **This project needs to double as a portfolio showpiece** — the working demo is what a recruiter will actually look at, so "is this legible and impressive to someone skimming for 60–120 seconds" is a real design constraint, not an afterthought (see Section 9).

## 3. Environment, Tooling & Compute Constraints

This project runs on a **local CPU-only MacBook** plus **free-tier cloud tools only**. No local LLM training or inference — the machine simply isn't built for it. The plan must draw a clear line between what runs locally and what has to run in the cloud, and every cloud step has to fit inside a free tier.

### Project identifiers — already decided, do not re-derive or propose alternatives

| Resource | Value |
|---|---|
| GitHub repo | `https://github.com/Rohanjain2312/forecast_bench.git` |
| Local project folder (git-linked to the above) | `/Users/rohanjain/Desktop/Projects/forecast_bench` |
| Hugging Face model repo | `https://huggingface.co/rohanjain2312/forecastbench-chronos` |
| Hugging Face dataset repo | `https://huggingface.co/datasets/rohanjain2312/forecastbench-data` |
| Hugging Face Space (demo) | `https://huggingface.co/spaces/rohanjain2312/forecastbench-demo` |

### Tooling

- **Coding workflow:** **Claude Code** is the primary way code gets written and iterated on, with **VS Code** as the editor alongside it. All classical modeling, the data pipeline, the evaluation/backtest harness, and plotting/analysis run locally through this workflow — no GPU needed for any of it.
- **Version control:** **GitHub**, at the repo above. The repo should be git-initialized from the start and committed incrementally as pieces land, following the same public-repo pattern as the existing portfolio projects.
- **GPU-bound work:** **Google Colab (free tier, T4)**. Anything that actually needs a GPU — Chronos/TimesFM zero-shot inference, LoRA/PEFT fine-tuning of the foundation model, and (if CPU training proves too slow) the N-BEATS / DeepAR-class baseline — happens here, not locally. Because free Colab sessions are time-limited and can disconnect, anything trained here needs to checkpoint somewhere durable (Hugging Face Hub, not just the ephemeral Colab disk).
- **Model & dataset hosting:** **Hugging Face (free tier)**, at the repos above — the fine-tuned Chronos checkpoint goes to `forecastbench-chronos`, processed/merged series go to `forecastbench-data`, and the required demo lives at the `forecastbench-demo` Space — all free tier, no paid inference endpoints.
- **Claude's file/storage and code-execution tools:** useful for prototyping snippets, quick data exploration, generating charts and tables during analysis, and drafting the writeup. Not a substitute for Colab when GPU compute is actually required.
- **What can stay fully local:** ARIMA/SARIMA (and SARIMAX if used), the data pipeline, the backtest/evaluation harness, and result visualization. These have no GPU dependency and should be developed and run entirely through Claude Code/VS Code on the local machine.
- **What must move to Colab:** any Chronos/TimesFM inference or fine-tuning. Whether N-BEATS/DeepAR also need to move to Colab is a judgment call for the planning AI based on how slow CPU training turns out to be — not a hard requirement either way.

## 4. Problem Formulation

At a high level, this is a **forecasting benchmark study**, structured as:

- One (or more) financial time series as the forecasting target
- A panel of classical models as one arm of the comparison
- A time-series foundation model, evaluated both zero-shot and fine-tuned, as the other arm
- A shared, leakage-safe backtesting harness that scores every model identically

The exact target series, horizon, and covariate structure are open decisions — see Section 8. This section exists so the planning AI knows the shape of the problem before resolving those specifics.

## 5. Data Pipeline

- **Sources:** FRED for rates/macro series — same source family as `Recession_BondsYield`. Yahoo Finance for equity/volatility series if that track is chosen — same source family as the market-regime-transformer projects.
- **Concrete starting points (free, no paid access needed):**
  - **FRED:** `fredapi` (or `pandas_datareader`), requiring a free API key from `https://fred.stlouisfed.org/docs/api/api_key.html`. Series IDs already validated in prior projects: `GS10`, `GS3M`, `DGS10`, `T10Y2Y`, `VIXCLS`, `CPIAUCSL`, `UNRATE`, `FEDFUNDS`.
  - **Yahoo Finance:** `yfinance`, no API key required, for `SPY`, `^GSPC`, `^IXIC`, `^VIX`.
  - **Chronos:** code and fine-tuning scripts at `github.com/amazon-science/chronos-forecasting` (`pip install chronos-forecasting`); pretrained checkpoints on Hugging Face under `amazon/chronos-*` (e.g. `amazon/chronos-t5-tiny`, `amazon/chronos-bolt-small`).
  - **TimesFM (if pursued as the stretch/secondary model):** `github.com/google-research/timesfm`, checkpoints under `google/timesfm-*` on Hugging Face.
  - **GluonTS (for the DeepAR-class baseline):** `pip install gluonts`.
  - **Worth flagging to the planning AI:** the `darts` library (`pip install darts`) wraps ARIMA, SARIMA, N-BEATS, and DeepAR-style models behind one consistent API. Using it could meaningfully simplify Section 6 instead of stitching together `statsmodels` + a standalone N-BEATS implementation + GluonTS separately — the planning AI should weigh this trade-off explicitly rather than defaulting to the fragmented approach.
- **Point-in-time discipline:** carry over the ALFRED-protocol habit from the market-regime-transformer-codex project (no look-ahead bias from revised macro data). This matters more here than in a classification project, because a forecasting backtest is *only* honest if every model saw only data available as of that point in time.
- **Storage pattern:** raw pulls cached to disk, a merged/processed series ready for modeling — same shape as the existing `yield_merged.csv` pattern, published to the `forecastbench-data` Hugging Face dataset repo once stable.
- **Splitting:** time-ordered only, never random. Needs an explicit walk-forward or rolling-origin scheme (see Section 7) rather than a single train/test cut.

## 6. Model Tracks

### 6a. Classical statistical baselines
- ARIMA
- SARIMA (and SARIMAX if exogenous macro covariates are in scope)
- These need to be refit correctly at each backtest fold, not fit once and reused — otherwise the comparison to the foundation model isn't fair (see Section 8, decision on refit cadence).
- Runs entirely locally — no GPU needed.

### 6b. Neural / "DeepAR-class" baseline
- N-BEATS as the primary neural baseline.
- One additional DeepAR-class model (e.g., GluonTS `DeepAR`, or a from-scratch probabilistic LSTM forecaster) to actually earn the "DeepAR-class" claim rather than substituting N-BEATS for both.
- Small enough to likely train on the local CPU; move to Colab only if training time becomes impractical.

### 6c. Foundation model track
- **Primary candidate: Chronos** (Amazon). It's T5-based and Hugging Face–native, which means it plugs into the exact `transformers` + `peft` LoRA workflow already used for the Mistral and Qwen fine-tunes — no new tooling to learn, just new data. Runs on Colab, not locally. Fine-tuned checkpoint published to `forecastbench-chronos`.
- **Secondary/optional: TimesFM** (Google) for a second foundation model data point, if bandwidth allows — harder to fine-tune, so treat as a stretch addition rather than a core requirement.
- Evaluate **both** zero-shot and fine-tuned versions of the foundation model. The zero-shot number is not a throwaway baseline — recent finance-specific literature shows zero-shot foundation models sometimes beat naive and even some classical baselines without any tuning, which is itself a finding worth reporting.

## 7. Fine-Tuning Approach

- Reuse the LoRA/PEFT recipe already validated in the Mistral and Qwen projects: parameter-efficient adaptation rather than a full fine-tune, trained on Colab's free-tier GPU (T4 has been the working constraint before — plan model size accordingly).
- Chronos ships in multiple sizes (tiny/mini/small/base/large) — the planning AI should size-select based on what's realistically trainable on free-tier Colab, not the largest available checkpoint.
- After training, push the resulting checkpoint straight to the `forecastbench-chronos` Hugging Face Model Hub repo (`huggingface_hub` push, same pattern as the existing HF-hosted models) rather than leaving it only on the ephemeral Colab disk — this also gives the local Claude Code environment, and the demo Space, a stable place to pull the fine-tuned model back from.
- Optional differentiator worth flagging to the planning AI: recent PEFT research on Chronos specifically found that lightweight methods (BitFit, LayerNorm tuning) can outperform LoRA at a fraction of the trainable parameters. Comparing 2–3 PEFT methods instead of just LoRA would be a legitimate, citable extension — flag as optional, not required.

## 8. Key Decisions to Resolve *Before* Writing Any Code

This is the list the planning AI should work through explicitly, with reasoning, before an implementation plan is produced.

1. **Which series to forecast.** Options: (a) extend the yield-curve work — forecast GS10, GS3M, or the T10Y2Y spread; (b) forecast realized/implied volatility (VIX or realized vol from SPY) — this has a direct tie-in to options pricing; (c) forecast equity returns directly, extending the market-regime work. Doing more than one series (e.g., a rates series *and* a volatility series) demonstrates breadth but roughly doubles the modeling and evaluation surface — worth an explicit trade-off call, and it also directly affects how much there is to show off in the demo.
2. **Forecast horizon(s).** Single horizon (e.g., 5-day-ahead) vs. a small horizon panel (1-day, 5-day, 21-day) to show how each model's relative performance changes as horizon lengthens — foundation models and classical models often diverge differently short- vs. long-horizon.
3. **Univariate or with covariates.** Plain univariate forecasting is simpler and is what Chronos supports natively. Adding macro covariates (rates, CPI, VIX) requires either a covariate-injection method (e.g., ChronosX-style adapters) for the foundation model, or SARIMAX for the classical side — decide whether covariates are in scope at all before committing to a foundation-model variant that supports them.
4. **Point forecasts or probabilistic forecasts.** Chronos is natively probabilistic (outputs a distribution, not a point estimate). Deciding to evaluate it probabilistically (pinball loss, coverage) rather than just as a point forecaster is more faithful to what the model actually does — but classical ARIMA/SARIMA also produce prediction intervals, so this is achievable on both sides if committed to early. This also affects what the demo can visually show (a forecast fan/band is more visually compelling than a single line).
5. **Refit cadence for classical models.** Refit ARIMA/SARIMA at every backtest fold (correct, more expensive) vs. fit once and roll forward (cheaper, but arguably not a fair fight against a model being evaluated fold-by-fold). This directly affects how defensible the final comparison is.
6. **Backtesting scheme.** Walk-forward / rolling-origin evaluation, with an explicit number of folds and window sizes — needs to be decided as a concrete scheme (e.g., expanding window vs. fixed-size rolling window), not left implicit.
7. **Evaluation metric set.** At minimum: point accuracy (MAE, RMSE, MASE, sMAPE), directional accuracy, and — if probabilistic forecasting is in scope — pinball/quantile loss and interval coverage. Also decide whether to include a formal statistical test (e.g., Diebold-Mariano) to say whether performance differences are significant, rather than reporting raw metric deltas as if they were conclusive.
8. **Regime-stratified evaluation.** Decide whether to break results out by market regime (calm vs. volatile periods, e.g., using VIX thresholds or the HMM regime labels already built in the market-regime-transformer project) so the "where it wins, where it loses" narrative has actual structure instead of being a single aggregate number.
9. **Sample-efficiency framing.** Recent literature reports foundation models needing far less fine-tuning data than classical/neural models need training data to hit comparable performance. Decide whether to explicitly test this (e.g., varying the training window size given to each model class) as part of the study — it's a strong, citable angle if included deliberately.
10. **Data-leakage guardrails.** Reconfirm the point-in-time protocol applies identically to every model in the study, including the foundation model's fine-tuning data cutoff, not just the classical models.
11. **What "honest" means in practice.** Decide up front what result would count as the foundation model *losing* convincingly, and commit to reporting it if it happens — this should be decided before results exist, not adjusted after seeing them.
12. **Demo scope — RESOLVED that it's required, open on how much it shows.** A public demo at the `forecastbench-demo` Space is mandatory, not optional (see Section 9). What remains open: exactly how much of the analysis surfaces interactively in the demo itself versus stays in the repo/writeup — resolve this explicitly, biased toward showing more rather than less, since the demo is the primary recruiter-facing artifact.

## 9. Deliverables (structure, not schedule)

- **GitHub repo** at `https://github.com/Rohanjain2312/forecast_bench.git`, developed locally at `/Users/rohanjain/Desktop/Projects/forecast_bench` via Claude Code + VS Code, following the existing project template: `data/`, `notebooks/`, `src/{data_pipeline, features, models/classical, models/foundation, evaluation, backtest}`, `README.md` with the same Problem Statement → Datasets → Architecture → Results table format used in prior repos.
- **Colab notebook** specifically for the GPU-bound steps (Chronos/TimesFM inference and fine-tuning), linked from the README rather than assumed to run inside the main repo's normal execution path.
- **Hugging Face:** the fine-tuned checkpoint at `forecastbench-chronos`, the processed/merged series at `forecastbench-data`.
- **Demo — required, not optional.** Lives at `https://huggingface.co/spaces/rohanjain2312/forecastbench-demo`. This is the artifact a recruiter actually opens, so it should show as much of the real work as reasonably fits into a clean interface, including:
  - An interactive picker for series / horizon (whatever the resolved scope from decision 1–2 turns out to be).
  - A side-by-side forecast plot showing classical, neural, and foundation-model (zero-shot *and* fine-tuned) predictions against actuals — visually, not just in a table.
  - A live, sortable results/metrics comparison table (the same metrics used in the actual backtest, not a simplified stand-in).
  - The regime-stratified breakdown, if that's in scope, so the "where each model wins/loses" story is visible, not just asserted in a writeup.
  - Plain-language model cards/descriptions for each model in the comparison, so a non-specialist skimming the Space still understands what they're looking at.
  - Ideally, live inference — letting a viewer select a date range and get a real forecast from the hosted models, rather than only static pre-computed charts — since that's a stronger "this actually works" signal than a fixed screenshot.
  - The goal: someone should be able to open the Space cold and understand the point of the project within about two minutes.
- **Results reported** as a comparison table across models, horizons, and (if in scope) regimes — not just a single headline number.
- **Writeup** in the same honest, limitations-forward voice as the GraphBench engineering post — explicitly stating what the benchmark does and does not prove.

## 10. Manual Setup Steps (to be resolved into an exact checklist by the planning AI)

Before any code runs, the following accounts, keys, and secrets are needed. The planning AI should turn this into a precise, ordered, copy-pasteable checklist — what to click, what to generate, and exactly where each value needs to be placed (local `.env`, Colab secrets, HF Space secrets) — rather than leaving it at this general level:

- **FRED API key** (free) — generated at `https://fred.stlouisfed.org/docs/api/api_key.html`.
- **Hugging Face access token with write scope** — needed to push the fine-tuned model to `forecastbench-chronos`, the dataset to `forecastbench-data`, and to deploy/update the `forecastbench-demo` Space. Generated from HF account settings; must never be hard-coded into notebooks or committed to GitHub.
- **GitHub authentication** — SSH key or personal access token, needed to push commits from the local machine to `Rohanjain2312/forecast_bench`.
- **Google account access for Colab**, with the GPU runtime (T4) explicitly enabled per session.
- **Local Python environment** — a virtual environment plus a pinned `requirements.txt` (or `pyproject.toml`), so the setup is reproducible rather than relying on whatever happens to be installed globally.
- **HF Space secrets configuration** — if the demo Space needs the write-scoped token or any other credential at runtime, it must be added through the Space's Secrets panel, not embedded in the Space's code.
- **Secrets-handling convention** — a single consistent approach across all three environments (local `.env` + `.gitignore`, Colab's secrets manager, HF Space Secrets), defined once and applied everywhere, so there's no risk of a key ending up committed to the public GitHub repo.

## 11. Documentation Standards

Every generated file needs to be understandable on a cold read, without reloading context from this brief or from memory of writing it. Concretely:

- Every Python module starts with a short docstring explaining what the file is for.
- Every function/class has a docstring covering what it does, its inputs/outputs, and any non-obvious assumption baked into it (e.g., "assumes point-in-time data; will silently produce look-ahead bias if fed revised FRED values").
- Inline comments explain **why** a step exists, not just restate what the code obviously does.
- Every notebook (Colab or local) has a markdown cell before each major code block explaining the goal of that block in plain language.
- A top-level `README.md`, plus a short README or docstring-level explanation per major module, describing how the repo is organized — matching the documentation style already used in the existing portfolio repos.
- The bar: the project owner should be able to open any file months later and understand what it does and why, without re-deriving the reasoning from scratch.

## 12. Risks to Flag for the Planning AI

- Chronos fine-tuning tooling is newer and less battle-tested than standard LLM fine-tuning — budget for rougher edges than the Mistral/Qwen workflows had.
- Small financial datasets risk overfitting during fine-tuning — the planning AI should propose a safeguard (e.g., early stopping on a held-out validation fold, or capping trainable parameters).
- Backtesting look-ahead bias is the single easiest way to invalidate the whole study — this should be treated as a hard constraint, not a nice-to-have.
- Temptation to over-claim a foundation-model win (or loss) from a small number of backtest folds — the statistical significance decision in Section 8, item 7, exists specifically to guard against this.
- **Free-tier compute limits.** Colab free-tier sessions have idle timeouts, a hard runtime cap, and no guaranteed GPU availability — training runs need to be checkpoint-resumable rather than assuming one uninterrupted session. Hugging Face's free tier also has storage/bandwidth limits — keep hosted artifacts reasonably sized by using the smallest Chronos variant that still gives meaningful results, since the demo Space also needs to load and run within free-tier resource limits.
