# Contributing

This is a portfolio research repository, so the bar is less "does it work" and more "can a
reviewer read it and believe the result." A few conventions carry most of that weight.

## Development setup

```bash
git clone git@github.com:Rohanjain2312/forecast_bench.git
cd forecast_bench
poetry install
poetry run pre-commit install
cp .env.example .env    # then fill in the placeholders
```

Never install the `gpu` group locally — it exists for Colab:

```bash
poetry install --with gpu   # Colab only
```

## Before every commit

```bash
poetry run black . && poetry run isort . && poetry run ruff check .
poetry run pytest
```

`pre-commit` runs the first line for you if you installed the hooks.

## Code conventions

- Black, line length 88. isort with the `black` profile. Ruff with `E, F, I, N, W, UP`.
- Google-style docstrings on every public class and function.
- Type hints on every function signature.
- Library code uses the `logging` module. Notebooks use `print` and `tqdm`. Library code
  never calls `print`.
- Docstrings state non-obvious assumptions, not just behaviour. If a function would
  silently invalidate the study when misused, say so in the docstring.
- Inline comments explain *why*, not *what*.

## The rules that are specific to this study

These are not style preferences. Breaking one of them silently invalidates the results.

1. Every model implements `forecast_bench/backtest/protocol.py::Forecaster`. If a model
   needs special handling inside `runner.py`, the abstraction is wrong — fix the
   abstraction, not the runner.
2. `fit()` may only read data at or before `origin`. No fitted object may cross a fold
   boundary: not scalers, not ARIMA orders, not MASE denominators, not residual quantiles.
   `tests/test_no_leakage.py` enforces this and must never be weakened to make a run pass.
3. `forecast_bench/evaluation/metrics.py` is the single source of truth for every metric.
   Import it. Never reimplement a metric in a notebook, a script, or `space/app.py`.
4. Only non-revised daily FRED series may appear in `forecast_bench/data/covariates.py`.
   The allowlist is `DGS10, DGS3MO, T10Y2Y, VIXCLS, DFF`.
5. `experiments/configs/regimes.yaml` holds VIX tercile thresholds computed on pre-2015
   data only. Never recompute them.

`PREREGISTRATION.md` is not editable after its commit. Deviations go in its Amendments
section as dated entries.

## Secrets

A secret exists in exactly three places and nowhere else.

| Environment | Mechanism | Read via |
|---|---|---|
| Local Mac | `.env` + `.gitignore` | `pydantic-settings` in `forecast_bench/config.py` |
| Colab | Colab Secrets panel | `google.colab.userdata.get()` into `os.environ` |
| HF Space | Space Secrets / Variables panel | `os.getenv()` |

**Never** put a literal token in a `.py`, `.ipynb`, `.yaml`, or `.md` file, in a commit
message, or in a W&B run config. **Always** run `git check-ignore -v .env` before the first
commit on a new machine.

If a token is exposed: revoke it at https://huggingface.co/settings/tokens, regenerate, and
update all three environments. Rotating costs a minute; a leaked write-scoped token on a
public repo does not.
