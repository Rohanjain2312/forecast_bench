# forecast_bench — Repo Structure

Actual file names and layout, mirroring the GraphBench template (Poetry, `docs/`,
`experiments/`, `notebooks/`, `tests/`, `assets/`, `CLAUDE.md`, pre-commit, GitHub Actions)
so this repo reads as part of the same portfolio rather than a one-off.

Local path: `/Users/rohanjain/Desktop/Projects/forecast_bench`

```
forecast_bench/
├── .env.example
├── .github/
│   └── workflows/
│       ├── tests.yml                  # pytest + ruff + black --check on push/PR
│       └── publish.yml                # PyPI publish on tag (mirrors graphbench)
├── .gitignore                         # data/, checkpoints/, .env, *.parquet, *.pt
├── .pre-commit-config.yaml            # black, isort, ruff
├── CHANGELOG.md
├── CLAUDE.md                          # guidance for Claude Code — see separate draft
├── CONTRIBUTING.md
├── LICENSE                            # MIT
├── PREREGISTRATION.md                 # committed BEFORE any model code — see draft
├── README.md
├── pyproject.toml
│
├── assets/
│   ├── architecture_diagram.png       # data → harness → 4 model arms → metrics → demo
│   ├── banner.png
│   ├── demo_screenshot.png
│   └── results_headline.png           # the one chart that goes in the Medium post
│
├── docs/
│   ├── index.md                       # doc map
│   ├── quickstart.md                  # clone → poetry install → one backtest, in 5 commands
│   ├── architecture.md                # how the pieces fit; the Forecaster protocol
│   ├── data_protocol.md               # point-in-time rules; the CPI release-lag bug writeup
│   ├── methodology.md                 # fold scheme, refit cadence, metrics, DM assumptions
│   ├── benchmark_results.md           # all six result tables, full precision
│   ├── limitations.md                 # what this does NOT prove; pretraining contamination
│   └── model_cards.md                 # plain-language description of each model
│
├── forecast_bench/                    # the installable package
│   ├── __init__.py
│   ├── version.py
│   ├── config.py                      # pydantic-settings — single source of truth for
│   │                                  #   paths, tokens, series IDs, quantile grid, seeds
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── fred_client.py             # fredapi wrapper, cached pulls + .meta.json sidecars
│   │   ├── yahoo_client.py            # yfinance OHLC, MultiIndex repair, column fallbacks
│   │   ├── targets.py                 # Garman-Klass log-RV; DGS10 level; flooring guards
│   │   ├── covariates.py              # NON-REVISED SERIES ONLY — enforced by allowlist
│   │   ├── merge.py                   # business-day alignment, validation, gap handling
│   │   └── hub.py                     # push/pull forecastbench-data
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── protocol.py                # Forecaster Protocol + QuantileForecast dataclass
│   │   ├── splitter.py                # expanding_origin_folds(); Fold dataclass
│   │   ├── cadence.py                 # EveryFoldCadence, BlockCadence
│   │   ├── runner.py                  # the harness; writes tidy forecast parquet
│   │   └── writer.py                  # tidy long-format schema, one definition
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── base.py                    # shared quantile-from-residuals helpers
│   │   ├── registry.py                # model_id -> builder; the panel is defined here
│   │   ├── naive.py                   # RandomWalk, SeasonalNaive
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── arima.py               # darts ARIMA, per-fold AIC order selection
│   │   │   ├── sarimax.py             # statsmodels, Arm B, lagged exog
│   │   │   ├── har.py                 # HAR + LogHAR, RV track
│   │   │   └── ar1.py                 # AR(1), rates track
│   │   ├── neural/
│   │   │   ├── __init__.py
│   │   │   ├── nbeats.py              # darts NBEATSModel + QuantileRegression
│   │   │   └── deepar.py              # darts RNNModel(LSTM) + quantile likelihood
│   │   └── foundation/
│   │       ├── __init__.py
│   │       ├── chronos2.py            # amazon/chronos-2, zero-shot + fine-tuned
│   │       ├── chronos_bolt.py        # amazon/chronos-bolt-small, both modes
│   │       ├── timesfm.py             # google/timesfm-2.5-200m, zero-shot (guarded import)
│   │       ├── finetune.py            # LoRA recipe, early stopping, resume-from-Hub
│   │       └── hub.py                 # push/pull forecastbench-chronos by revision tag
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── metrics.py                 # SINGLE SOURCE OF TRUTH — never reimplement
│   │   ├── stats.py                   # Diebold-Mariano (HLN+HAC), MCS, block bootstrap
│   │   ├── regimes.py                 # frozen VIX terciles; asserts against config
│   │   └── aggregate.py               # the six results tables
│   │
│   └── viz/
│       ├── __init__.py
│       ├── forecast_plots.py          # fan charts (shared by repo figures and the Space)
│       └── results_plots.py           # regime heatmap, sample-efficiency curve
│
├── experiments/
│   ├── configs/
│   │   ├── base.yaml                  # quantile grid, seeds, context length, horizons
│   │   ├── spy_logrv.yaml             # primary target
│   │   ├── dgs10.yaml                 # contrast target
│   │   ├── arm_b_covariates.yaml      # covariate-informed arm
│   │   ├── sample_efficiency.yaml     # the {1y, 3y, 10y, full} sweep
│   │   └── regimes.yaml               # FROZEN VIX tercile thresholds — do not recompute
│   └── results/
│       ├── forecasts/.gitkeep         # tidy parquet, gitignored, pushed to HF
│       ├── metrics/.gitkeep
│       └── figures/.gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb              # local: series, GK estimator sanity, regimes
│   ├── 02_backtest_classical_local.ipynb      # local: first end-to-end run
│   ├── 03_colab_foundation_zeroshot.ipynb     # Colab: Chronos-2 / Bolt zero-shot
│   ├── 04_colab_finetune_chronos.ipynb        # Colab H100: LoRA, push to Hub
│   ├── 05_colab_train_neural.ipynb            # Colab H100: N-BEATS + DeepAR-class
│   └── 06_results_analysis.ipynb              # local: tables, DM tests, figures
│
├── scripts/
│   ├── fetch_data.py                  # python -m scripts.fetch_data --config spy_logrv
│   ├── run_backtest.py                # --config ... --cadence matched --arm A
│   ├── build_results.py               # forecasts parquet -> all six tables + figures
│   └── push_artifacts.py              # data -> HF dataset, space/ -> HF Space
│
├── space/                             # mirrored to the HF Space
│   ├── app.py                         # Gradio, 5 tabs
│   ├── model_cards.py                 # plain-language glosses; imported by docs too
│   ├── requirements.txt               # exported from poetry, Space-only subset
│   └── README.md                      # Space card w/ YAML front-matter (sdk, hardware)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    # synthetic series fixtures, tiny fold sets
    ├── test_data_pipeline.py          # GK estimator correctness, caching, allowlist
    ├── test_no_leakage.py             # THE hard constraint; includes the canary test
    ├── test_splitter.py               # non-overlap, expanding, block boundaries
    ├── test_cadence.py                # refit triggers fire exactly when expected
    ├── test_metrics.py                # each metric against hand-computed values
    ├── test_stats.py                  # DM against a known-answer reference case
    ├── test_models_protocol.py        # every registered model satisfies Forecaster
    └── test_regimes.py                # thresholds match frozen config
```

## Notes on a few choices

**`forecast_bench/` as an installable package, not a loose `src/`.** GraphBench does this
and it pays off here for a specific reason: the Colab notebooks and the HF Space both need
the modelling code, and `pip install git+https://github.com/Rohanjain2312/forecast_bench.git`
is the only way to guarantee all three environments run identical logic. A loose `src/`
would mean copy-pasting into Colab, which is how the notebook and the repo start disagreeing.

**`space/` lives in the repo, not only on the Hub.** The Space is a deliverable and should
be reviewable, diffable, and CI-checked like everything else. `scripts/push_artifacts.py`
mirrors it up.

**`registry.py` is where the model panel is defined.** Adding a model means one entry there
plus one file. Nothing else changes — not the runner, not the metrics, not the Space. This
is the payoff for the `Forecaster` protocol.

**`experiments/configs/regimes.yaml` is deliberately a data file, not code.** It contains
two numbers that must never be recomputed. Keeping them in a committed YAML makes any
change to them show up in a diff.

**`data/` is gitignored; the HF dataset repo is the durable copy.** Same pattern as
GraphBench's `checkpoints/`.
