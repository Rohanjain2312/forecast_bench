# forecast_bench — Build Order

**This file is the execution script for Claude Code. It is also the progress tracker.**

## How to use this file (Claude Code reads this section first, every session)

1. Read `CLAUDE.md`, then `docs/planning/DECISIONS.md`, then
   `docs/planning/IMPLEMENTATION_PLAN.md`, then `docs/planning/REPO_STRUCTURE.md`.
2. Scroll this file to the **first step whose checkbox is unticked**. That is the current
   step. Do not skip ahead, do not batch steps.
3. Do that step, and only that step.
4. Run its **Acceptance check**. If it fails, fix it before moving on. Never weaken a test
   to make a check pass.
5. Commit with the given message.
6. Tick the checkbox in this file and commit that too (`chore: mark step N complete`).
7. Look at the step's **Gate**:
   - `GATE: AUTO` → continue straight to the next step without asking.
   - `GATE: STOP` → stop and print the exact "Tell the user" message. Do not continue
     until the user confirms the manual task is done.
8. Repeat.

**Never ask the user which step to do.** The unticked checkbox is the answer.

**Never ask the user for approval to continue** at an `AUTO` gate. Only `STOP` gates
require a human.

If a step is genuinely blocked (a library API differs from what this file assumes, a
result contradicts an assumption), stop, explain the discrepancy in one paragraph, propose
a fix, and wait. Do not silently improvise around the plan.

---

## Phase 0 — Foundation

### Preflight state — verified 2026-09-02, do not re-ask the user

All of this was checked live and is already true. **Claude Code must not ask the user to
create accounts, tokens, or repos.**

| | State |
|---|---|
| HF PRO subscription | Active |
| HF write token | Generated, held by the user |
| FRED API key | Generated, held by the user |
| `github.com/Rohanjain2312/forecast_bench` | Exists, **completely empty — no branches, no commits** |
| `rohanjain2312/forecastbench-chronos` | Exists, empty, no model card, no license set |
| `rohanjain2312/forecastbench-data` | Exists, empty |
| `rohanjain2312/forecastbench-demo` | Exists, Gradio SDK, no `app.py`. Hardware not yet confirmed — checked automatically in Step 4 |

**Start at Step 1.** The user's remaining manual tasks are Steps 4–8 of
`docs/planning/MANUAL_TASKS.md`, and each is triggered by a STOP gate below.

---

### [x] Step 1 — Repo scaffold

**Goal:** an installable, lintable, testable empty package.

**Create:**

- `pyproject.toml` — Poetry. Name `forecast-bench`, version `0.1.0`, Python `^3.11`,
  MIT, author `Rohan Jain <rohanjain2312@gmail.com>`, packages `[{include = "forecast_bench"}]`.
  Dependencies: `pandas`, `pyarrow`, `numpy`, `scipy`, `statsmodels`, `darts`,
  `chronos-forecasting`, `fredapi`, `yfinance`, `huggingface-hub`, `datasets`,
  `pydantic-settings`, `python-dotenv`, `pyyaml`, `matplotlib`, `plotly`, `arch`, `tqdm`,
  `wandb`. Dev group: `pytest`, `pytest-cov`, `pytest-mock`, `black`, `isort`, `ruff`,
  `pre-commit`. Optional group `gpu`: `torch`, `peft`, `transformers`, `accelerate`
  (Colab-only — do not install locally).
  Tool config mirrors GraphBench exactly: black line-length 88 target py311, isort profile
  black, ruff select `["E","F","I","N","W","UP"]` ignore `["E501"]`, pytest testpaths
  `["tests"]` with markers `slow` and `gpu`, default addopts excluding both.
- `.gitignore` — `.env`, `data/`, `checkpoints/`, `experiments/results/**/*.parquet`,
  `*.pt`, `*.safetensors`, `__pycache__/`, `.DS_Store`, `.venv/`, `wandb/`, `.ipynb_checkpoints/`
- `.pre-commit-config.yaml` — black, isort, ruff
- `.github/workflows/tests.yml` — Ubuntu, Python 3.11, poetry install (no `gpu` group),
  `black --check`, `isort --check`, `ruff check`, `pytest -m "not slow and not gpu"`
- `LICENSE` (MIT, 2026 Rohan Jain), `CHANGELOG.md`, `CONTRIBUTING.md`
- `forecast_bench/__init__.py`, `forecast_bench/version.py`
- Empty packages with `__init__.py` for `data/`, `backtest/`, `models/`,
  `models/classical/`, `models/neural/`, `models/foundation/`, `evaluation/`, `viz/`
- `tests/__init__.py`, `tests/conftest.py` (empty for now)
- Directory placeholders per `REPO_STRUCTURE.md`: `assets/`, `docs/`, `experiments/configs/`,
  `experiments/results/{forecasts,metrics,figures}/.gitkeep`, `notebooks/`, `scripts/`,
  `space/`
- `.env.example` — every key listed in `docs/planning/SETUP_CHECKLIST.md` Step 6c, with
  placeholder values and no real secrets

**Git state — the remote is empty (no branches, no commits).** Before creating files:

```bash
ssh -T git@github.com          # must greet Rohanjain2312; if not, stop and tell the user
git remote -v                  # must show Rohanjain2312/forecast_bench
```

If the local folder is not yet a git repo, clone it (it will warn that the repository is
empty — that is expected). If it already is one, leave it alone. **Never run `git init` or
`git remote add`.** The first push needs `git branch -M main && git push -u origin main`.

**Do not** run `git init` or `git remote add`. The folder is already linked.

**Acceptance check:**
```bash
poetry install && poetry run pytest && poetry run ruff check . && poetry run black --check .
git check-ignore -v .env       # must print a .gitignore line
git ls-remote --heads origin   # must now show refs/heads/main
```

**Commit:** `chore: scaffold repo, tooling, and CI`

**GATE: AUTO**

---

### [x] Step 2 — Commit the pre-registration

**Goal:** get `PREREGISTRATION.md` into git history before any model code exists. The
timestamp is the artifact; this is why it is Step 2 and not Step 20.

**Do:** verify `PREREGISTRATION.md` is at the repo root. Commit it alone, in its own
commit, with no other files.

**Never edit the text of this file again after this commit.** Any later deviation goes in
its Amendments section as a dated entry.

**Acceptance check:**
```bash
git log --oneline -- PREREGISTRATION.md   # exactly one commit, and no model code exists yet
```

**Commit:** `docs: pre-register study design and losing condition before any results exist`

**GATE: AUTO**

---

### [x] Step 3 — Config layer

**Goal:** one source of truth for every setting. Nothing downstream reads `os.environ`
directly.

**Create `forecast_bench/config.py`** using `pydantic-settings`:

- Secrets from `.env`: `FRED_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY` (optional)
- Repo IDs: `HF_MODEL_REPO`, `HF_DATASET_REPO`, `HF_SPACE_REPO`
- Paths: `DATA_DIR`, `RESULTS_DIR`, with `raw/` and `processed/` subpath helpers
- Study constants, hardcoded not env-driven (they are decisions, not configuration):
  `QUANTILE_GRID = [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]`,
  `HORIZONS = [1, 5, 21]`, `MAX_HORIZON = 21`, `STRIDE = 21`, `CONTEXT_LENGTH = 512`,
  `TRAIN_START = "2000-01-01"`, `TEST_START = "2015-01-01"`, `TEST_END = "2026-06-30"`,
  `RANDOM_SEED = 42`
- `NON_REVISED_FRED_ALLOWLIST = {"DGS10", "DGS3MO", "T10Y2Y", "VIXCLS", "DFF"}`
- A `get_config()` singleton and a `setup_logging()` helper

**Create `experiments/configs/base.yaml`** holding the same study constants, so scripts and
notebooks read one file. `config.py` loads it and asserts the two agree — a mismatch should
raise at import, not produce silently different runs.

**Acceptance check:** `poetry run python -c "from forecast_bench.config import get_config; print(get_config().model_dump())"` runs and prints without a token appearing in output (mask secrets in `__repr__`).

**Commit:** `feat: config layer with study constants and non-revised series allowlist`

**GATE: STOP**

Print exactly this:

> Scaffold and config are in. Now fill in your `.env` — Step 4 of
> `docs/planning/MANUAL_TASKS.md`. Copy `.env.example` to `.env` and paste your FRED key
> and HF token. Tell me when it's filled and I'll verify every key works.

---

### [x] Step 4 — Verify credentials end to end

**Goal:** catch a bad key now, not in the middle of a backtest.

**Create `scripts/verify_setup.py`** running five checks and printing a pass/fail table:

1. FRED: pull last 3 values of `DGS10`
2. Yahoo: pull 10 days of SPY OHLC, confirm all four columns present
3. Chronos-2: load `amazon/chronos-2` on CPU, forecast 21 steps from 512 random points,
   **print the wall-clock latency in seconds**
4. HF: `whoami()`, plus `repo_info` on all three repos (all three already exist)
5. Space config: `HfApi().space_info(HF_SPACE_REPO)` — print the SDK and the current
   hardware. Expected `gradio` and `cpu-basic`. If hardware is anything else (especially
   `zero-a10g` / ZeroGPU), flag it — this is the only thing left that may need a user click.

**Run it.** Record the Chronos CPU latency and the Space hardware in
`docs/planning/PROGRESS_NOTES.md` (create it).

**Acceptance check:** checks 1–4 pass. Check 5 reports, and does not block.

**Commit:** `feat: setup verification script`

**GATE: STOP**

Print exactly this, filling in the two findings:

> All credential checks pass. Two things to tell you:
>
> **Chronos-2 CPU latency: N seconds** per forecast.
> - Under 5s → live-inference demo works as planned. Nothing for you to do.
> - 5–10s → workable; I'll add a caching layer.
> - Over 10s → I'll switch the demo to pre-computed forecasts over a fixed date grid.
>
> I'm going with **[the applicable option]**.
>
> **Space hardware: [value].**
> - If `cpu-basic` → correct already, nothing for you to do.
> - Otherwise → this is Step 5 of `docs/planning/MANUAL_TASKS.md`. Go to
>   https://huggingface.co/spaces/rohanjain2312/forecastbench-demo/settings, set Space
>   hardware to **CPU Basic — FREE**, and save. Takes 30 seconds.
>
> Say "go" when done (or straight away if there was nothing to change).

---

## Phase 1 — Data

### [x] Step 5 — Data clients

**Goal:** cached, validated pulls from FRED and Yahoo.

**Create:**

- `forecast_bench/data/fred_client.py` — `fredapi` wrapper. **Must reject any series ID
  not in `NON_REVISED_FRED_ALLOWLIST`, raising `ValueError` with a message explaining
  that revised series are indexed by reference period, not release date, and would
  introduce look-ahead bias.** Caches to `data/raw/fred_{ID}.parquet` with a
  `.meta.json` sidecar holding fetch timestamp, source, and content checksum.
- `forecast_bench/data/yahoo_client.py` — `yfinance` OHLC. Handle MultiIndex columns from
  multi-ticker downloads and the varying adjusted-close column name. Same cache pattern.
- `forecast_bench/data/README.md` — 15 lines: what lives here, why the allowlist exists.

**Acceptance check:** `tests/test_data_pipeline.py` covering: cache hit avoids a second
network call; allowlist rejects `CPIAUCSL` with the expected error; MultiIndex repair works
on a synthetic frame.

**Commit:** `feat: FRED and Yahoo clients with caching and non-revised allowlist`

**GATE: AUTO**

---

### [ ] Step 6 — Targets and covariates

**Goal:** the two modelling series.

**Create `forecast_bench/data/targets.py`:**

- `garman_klass_variance(ohlc)` — `0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2`. Non-positive
  values are floored at the 0.1th percentile of the training window (never the full
  series), and the number of flooring events is logged at WARNING.
- `parkinson_variance(ohlc)` — `ln(H/L)^2 / (4*ln2)`, as a cross-check
- `build_spy_logrv()` → log of Garman-Klass variance, business-day indexed
- `build_dgs10()` → `DGS10` in levels, **not differenced**. Market holidays stay as NaN and
  are dropped, never forward-filled — forward-filling a target is a subtle leak.

**Create `forecast_bench/data/covariates.py`** and `merge.py` per `REPO_STRUCTURE.md`.
Covariate sets exactly as in `IMPLEMENTATION_PLAN.md` §2.1.

**Create `scripts/fetch_data.py`** — `--config spy_logrv|dgs10`, writes to `data/processed/`.

**Acceptance check:** Garman-Klass tested against a hand-computed value for one known bar.
Assert no forward-fill was applied to either target. Assert covariate frames contain only
allowlisted series.

**Commit:** `feat: Garman-Klass log-RV and DGS10 target construction`

**GATE: AUTO**

---

### [ ] Step 7 — Freeze the regime thresholds

**Goal:** compute the VIX terciles once, on pre-2015 data only, and commit them as data.

**Do:** compute the 33rd and 67th percentiles of `VIXCLS` over `2000-01-01 → 2014-12-31`.
Write them to `experiments/configs/regimes.yaml` with a header comment saying they must
never be recomputed and why.

**Acceptance check:** the YAML exists with two float values and the warning comment.

**Commit:** `feat: freeze VIX tercile regime thresholds from pre-2015 data only`

**GATE: AUTO**

---

### [ ] Step 8 — Data protocol documentation

**Goal:** write down the point-in-time rules while they are fresh.

**Create `docs/data_protocol.md`** covering: which series are used and why each is
non-revised; the allowlist mechanism; and — as the motivating example — the FRED
release-lag issue found in `market-regime-transformer-codex`, where `CPIAUCSL` is indexed
by reference month but published weeks later, so a `merge_asof` on the FRED index reads
the future. Frame it as the bug this repo is designed to prevent.

**Commit:** `docs: point-in-time data protocol and the CPI release-lag failure mode`

**GATE: AUTO**

---

## Phase 2 — The harness (the most important phase)

### [ ] Step 9 — Leakage tests, written BEFORE the harness

**Goal:** the guards exist first, so the harness is written against them.

**Create `tests/test_no_leakage.py`** with all five checks from `IMPLEMENTATION_PLAN.md`
§3.5. They will fail right now because the harness does not exist — that is expected and
correct. Mark them `xfail(strict=True)` with a reason string pointing at Step 11, and flip
them to real failures in Step 11.

The canary test matters most: inject a column that is a perfect copy of the future target,
confirm error collapses to ~0, and confirm the leakage assertions fire. A guard nobody has
seen fail is a guard nobody knows works.

**Also create `tests/conftest.py`** with synthetic fixtures: a deterministic 3000-day
series, a tiny 5-fold set, a stub `Forecaster` that returns constants.

**Acceptance check:** `poetry run pytest tests/test_no_leakage.py` — all xfail, none pass.

**Commit:** `test: leakage guards and canary, written before the harness`

**GATE: AUTO**

---

### [ ] Step 10 — Forecaster protocol and splitter

**Create `forecast_bench/backtest/protocol.py`** — `QuantileForecast` frozen dataclass and
`Forecaster` Protocol, exactly as specified in `IMPLEMENTATION_PLAN.md` §3.1, including the
docstring stating that `fit()` must not close over anything fitted outside the fold.

**Create `forecast_bench/backtest/splitter.py`** — `Fold` dataclass and
`expanding_origin_folds()` per §3.2. Docstring must state that `stride == horizon` is what
makes forecast windows non-overlapping and that changing it invalidates the DM test.

**Acceptance check:** `tests/test_splitter.py` asserting: windows never overlap; training
window expands monotonically; every `origin < forecast_index[0]`; block IDs are calendar
years; fold count is ~137 for the configured span.

**Commit:** `feat: Forecaster protocol and non-overlapping expanding-origin splitter`

**GATE: AUTO**

---

### [ ] Step 11 — Cadence policies and the runner

**Create `forecast_bench/backtest/cadence.py`** — `EveryFoldCadence` and
`BlockCadence(freq="YS")` behind one interface.

**Create `forecast_bench/backtest/writer.py`** — the tidy long-format schema, defined once:
`origin, target_date, step, model_id, quantile, value, actual, regime, block_id, series,
arm, cadence`. Everything downstream reads this and only this.

**Create `forecast_bench/backtest/runner.py`** per §3.4, including the runtime assertion
that `forecast.index[0] > fold.origin`.

**Then flip `tests/test_no_leakage.py` from xfail to real.** They must now pass.

**Acceptance check:**
```bash
poetry run pytest tests/test_no_leakage.py tests/test_splitter.py tests/test_cadence.py
```
All pass, no xfail remaining.

**Commit:** `feat: backtest runner with dual refit cadence; leakage guards now enforced`

**GATE: AUTO**

---

## Phase 3 — Models and evaluation

### [ ] Step 12 — Naive and classical models

**Create** `models/base.py`, `models/registry.py`, `models/naive.py`, and
`models/classical/{arima,sarimax,har,ar1}.py` per `REPO_STRUCTURE.md` and
`IMPLEMENTATION_PLAN.md` §4a.

Every one emits quantiles. Random walk quantiles come from the empirical distribution of
h-step changes in the training window, recomputed per fold. HAR quantiles come from OLS
residual quantiles scaled by `sqrt(h)`.

**Acceptance check:** `tests/test_models_protocol.py` — every entry in `registry.py`
satisfies `Forecaster`, returns the full quantile grid, and returns monotonically
non-decreasing quantiles at each step (a crossing quantile is a bug).

**Commit:** `feat: naive and classical model panel with quantile outputs`

**GATE: AUTO**

---

### [ ] Step 13 — Metrics, statistics, regimes, aggregation

**Create** `evaluation/{metrics,stats,regimes,aggregate}.py` per §5.

`metrics.py` is the single source of truth — nothing anywhere else recomputes a metric.
MASE denominators use the training window only. Coverage and width are always returned as
a pair.

`regimes.py` asserts the loaded thresholds match `experiments/configs/regimes.yaml` at
import time.

**Acceptance check:** `tests/test_metrics.py` checks every metric against hand-computed
values on a tiny fixture. `tests/test_stats.py` checks Diebold-Mariano against a
known-answer reference case.

**Commit:** `feat: metric suite, DM test, model confidence set, regime stratification`

**GATE: AUTO**

---

### [ ] Step 14 — First end-to-end run, classical only

**Goal:** the study exists. This is the milestone.

**Create `scripts/run_backtest.py`** (`--config`, `--cadence`, `--arm`) and
`scripts/build_results.py`.

**Run** both series, Arm A, matched cadence. The panel is per-series, resolved from
`registry.py`: both get RandomWalk, SeasonalNaive, ARIMA; SPY log-RV also gets HAR and
LogHAR; `DGS10` also gets AR(1).

**Sanity-check the two registered predictions from `PREREGISTRATION.md` §4:**
- On `DGS10`, no model should beat random walk by much (skill roughly -0.05 to +0.05)
- On SPY log-RV, HAR/LogHAR should be clearly the strongest classical model

**If either fails, stop and investigate before adding any more models.** A broken pipeline
is far cheaper to find here than after the foundation models are wired in.

**Acceptance check:** results parquet written; the two sanity checks hold, or the
discrepancy is explained.

**Commit:** `feat: first end-to-end backtest, classical arm, both series`

**GATE: STOP**

Print the two sanity-check results and:

> First real results are in. [State whether the two registered predictions held.] Say "go"
> to add the foundation models.

---

### [ ] Step 15 — Foundation models, zero-shot, local

**Create** `models/foundation/{chronos2,chronos_bolt,hub}.py`. Zero-shot paths only.
Chronos-2 via `Chronos2Pipeline` (not `AutoModelForSeq2SeqLM`). CPU device.

**Run** both series, Arm A, matched cadence, adding both zero-shot models to the panel.

**Create `forecast_bench/data/hub.py`** and push processed series to `forecastbench-data`
with a dataset card (`license: mit` in the YAML front-matter).

**Also push a model card to `forecastbench-chronos`** with `license: apache-2.0` in the YAML
front-matter. The repo currently has no card and no license; Chronos is Apache 2.0 and
fine-tuned derivatives inherit it, so this must be set before any checkpoint is pushed.

**Acceptance check:** zero-shot forecasts present in the results parquet for both series;
the dataset repo has files.

**Commit:** `feat: Chronos-2 and Chronos-Bolt zero-shot; publish processed series`

**GATE: AUTO**

---

## Phase 4 — GPU work

### [ ] Step 16 — Write the Colab notebooks

**Goal:** notebooks the user can open and run without editing anything.

**Create:**

- `forecast_bench/models/foundation/finetune.py` — LoRA recipe (rank 8, alpha 16, dropout
  0.05), early stopping on a fold-local validation slice (patience 3, monitoring validation
  WQL), checkpoint-to-Hub after each block, and `--resume-from` that checks the Hub for the
  latest checkpoint before starting fresh.
- `notebooks/04_colab_finetune_chronos.ipynb` — installs the package from GitHub, reads
  Colab Secrets, runs `finetune.py`, pushes to `forecastbench-chronos` with revision tags.
- `notebooks/05_colab_train_neural.ipynb` — same pattern for N-BEATS and the DeepAR-class
  LSTM, including the four sample-efficiency window sizes.
- `models/neural/{nbeats,deepar}.py`

**Every notebook** opens with a markdown cell listing what must already exist on the Hub
before it runs, and has a markdown cell before every code block explaining the goal in
plain language. Notebooks contain **no modelling logic** — they import and call.

**Acceptance check:** notebooks contain no `for` loop over folds; every heavy call is an
import from `forecast_bench`.

**Commit:** `feat: LoRA fine-tuning recipe and Colab notebooks`

**GATE: STOP**

Print exactly this:

> The Colab notebooks are ready. This is your turn — Step 6 of `docs/planning/MANUAL_TASKS.md`.
>
> 1. Open `notebooks/04_colab_finetune_chronos.ipynb` in Colab
> 2. Runtime → Change runtime type → **H100**
> 3. Run all cells. It checkpoints to the Hub after each block, so a disconnect costs one
>    block, not the run — just re-run and it resumes.
> 4. Then do the same for `notebooks/05_colab_train_neural.ipynb`
>
> Tell me when both have finished and pushed. Then I'll pull the checkpoints and run the
> full benchmark.

---

## Phase 5 — Full results

### [ ] Step 17 — Full benchmark runs

**Run, in this order, and write each to the results parquet:**

1. Arm A, matched cadence, full panel — **this is the headline**
2. Arm A, native cadence, full panel
3. Arm B (covariates), matched cadence
4. Sample-efficiency sweep (four training-window sizes)
5. Contamination-free subset (origins after 2025-11-01 only, ~8 folds)

**Acceptance check:** all six result tables from §5.4 generate without error.

**Commit:** `feat: full benchmark across both cadences, both arms, all models`

**GATE: AUTO**

---

### [ ] Step 18 — Evaluate against the pre-registration

**Goal:** apply the losing condition. Do not soften it.

**Do:** read `PREREGISTRATION.md` §3. Evaluate both conditions against the actual numbers.
State plainly whether the fine-tuned foundation model won or lost. Check the five registered
predictions in §4 and record which held.

**Create** `docs/benchmark_results.md` (all six tables, full precision) and
`docs/limitations.md` (what this does not prove; pretraining contamination as a first-class
section, not a footnote).

**Commit:** `docs: benchmark results and limitations, evaluated against pre-registration`

**GATE: STOP**

Print the verdict — the losing condition outcome and which of the five predictions held —
and:

> This is the honest result, whatever it is. Say "go" and I'll build the demo around it.

---

## Phase 6 — Demo and writeup

### [ ] Step 19 — Build the Space

**Create** `space/{app.py,model_cards.py,requirements.txt,README.md}` and
`forecast_bench/viz/{forecast_plots,results_plots}.py`.

Five tabs per `DECISIONS.md` D12. Landing tab is the live forecast with a default already
rendered, so something real is on screen before any click. Demo architecture follows the
decision made at Step 4's gate.

Constraints: every axis labelled in words; every model name has a plain-language hover
gloss from `model_cards.py`; limitations are a tab, not a link; no number appears in the
Space that is not also in `docs/benchmark_results.md`.

**Create `scripts/push_artifacts.py`** to mirror `space/` to the Space repo.

**Acceptance check:** Space builds and a forecast renders.

**Commit:** `feat: Gradio demo Space with live inference and full results`

**GATE: STOP**

Print exactly this:

> The Space is live at https://huggingface.co/spaces/rohanjain2312/forecastbench-demo
>
> Your turn — Step 7 of `docs/planning/MANUAL_TASKS.md`. Open it on your phone, then show it to one
> person who knows nothing about the project. Give them two minutes and no explanation,
> then ask what they think it does. Tell me what they said and I'll fix whatever confused
> them.

---

### [ ] Step 20 — README, architecture diagram, docs pass

**Create/finish:** `README.md` in the GraphBench format — badges, one-line description,
the plain-language explanation from `IMPLEMENTATION_PLAN.md` §0, how it works, results
table, installation, quick start, architecture, project structure, requirements at a
glance, license. Plus `docs/{index,quickstart,architecture,methodology,model_cards}.md`
and the four per-module READMEs.

Then sweep every module against `IMPLEMENTATION_PLAN.md` §7, confirming the six required
assumption-docstrings are present verbatim.

**Acceptance check:** every `.py` file has a module docstring; every public function has a
Google-style docstring with type hints; the six assumption-docstrings exist.

**Commit:** `docs: README, module docs, and documentation-standard sweep`

**GATE: STOP**

> Everything's built and documented. Last step is the Medium writeup — Step 8 of
> `docs/planning/MANUAL_TASKS.md`. I can draft it if you want; say the word.

---

### [ ] Step 21 — Optional stretch work

Only after every box above is ticked. In priority order:

1. TimesFM 2.5 zero-shot (`models/foundation/timesfm.py`, guarded import)
2. PEFT-method comparison — LoRA vs BitFit vs LayerNorm-only → `docs/peft_comparison.md`

Neither enters the headline table. Both are labelled exploratory per `PREREGISTRATION.md` §5.

**GATE: AUTO**
