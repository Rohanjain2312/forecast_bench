# Quickstart

From a clone to the pre-registered verdict, in five commands.

## Setup

```bash
git clone git@github.com:Rohanjain2312/forecast_bench.git
cd forecast_bench
poetry install
cp .env.example .env
```

Then edit `.env` and add a [FRED API key](https://fredaccount.stlouisfed.org/apikeys) and a
[Hugging Face token](https://huggingface.co/settings/tokens). Confirm `.env` is ignored
before your first commit — it holds real credentials:

```bash
git check-ignore -v .env    # must print a .gitignore line
```

## 1. Verify everything works

```bash
poetry run python -m scripts.verify_setup
```

Five checks: FRED, Yahoo, Chronos-2 on CPU, Hugging Face access, and the Space's
configuration. It prints Chronos-2's CPU forecast latency, which is the number that decides
whether the demo can run live inference. Measured at **0.85 s** here.

## 2. Build the two target series

```bash
poetry run python -m scripts.fetch_data --config spy_logrv
poetry run python -m scripts.fetch_data --config dgs10
```

Raw pulls are cached under `data/raw/` with a `.meta.json` sidecar recording when they were
fetched and a checksum. A parquet without its sidecar is treated as absent and refetched:
provenance is part of the artifact.

## 3. Run a backtest

```bash
poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A
```

That runs the naive and classical panel — a couple of minutes. Add the foundation models:

```bash
poetry run python -m scripts.run_backtest --config spy_logrv --cadence matched --arm A \
    --with-foundation --with-finetuned
```

`--with-finetuned` loads the LoRA adapters from the Hub, one per annual block. Add
`--merge-hub` to pull in the GPU-trained neural forecasts rather than retraining them
locally, which is not viable on CPU.

## 4. Build the results tables

```bash
poetry run python -m scripts.build_results
```

Writes all six tables to `experiments/results/metrics/` and regenerates
`docs/benchmark_results.md` from them. The document is generated rather than typed, so it
cannot contain a number the pipeline did not produce.

## 5. Apply the pre-registered decision rules

```bash
poetry run python -m scripts.evaluate_preregistration
```

Prints the losing condition clause by clause, the five registered predictions, and the Model
Confidence Set. The verdict follows from the committed rules rather than from reading a
table and deciding what it says.

## Running the tests

```bash
poetry run pytest                          # the default suite
poetry run pytest tests/test_no_leakage.py # the one that matters
poetry run pytest -m slow                  # downloads real checkpoints
```

`tests/test_no_leakage.py` is the hard constraint made executable. It must never be weakened
or skipped to make a run pass.

## The GPU work

Fine-tuning and neural training happen in Colab, in `notebooks/04_colab_finetune_chronos.ipynb`
and `notebooks/05_colab_train_neural.ipynb`. Both install this package from GitHub and call
into it — they contain no modelling logic, which is what stops the notebooks and the
repository from disagreeing. A test enforces that.
