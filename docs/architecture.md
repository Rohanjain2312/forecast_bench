# Architecture

How the pieces fit, and the one design decision everything else follows from.

## The central abstraction

`backtest/protocol.py` defines two things and nothing else:

```python
@dataclass(frozen=True)
class QuantileForecast:
    origin: pd.Timestamp
    index: pd.DatetimeIndex
    quantiles: Mapping[float, np.ndarray]
    model_id: str

class Forecaster(Protocol):
    def fit(self, train, origin, refit_parameters=True) -> None: ...
    def predict(self, horizon, index) -> QuantileForecast: ...
```

Thirteen model classes across four families reduce to this. **ARIMA and Chronos-2 are
indistinguishable to the runner.** That is the whole design, and it is what lets the study
claim every model traversed identical code.

The rule that keeps it honest: *if a model needs special handling inside `runner.py`, the
abstraction is wrong — fix the abstraction, not the runner.*

## Data flow

```
FRED ──┐
       ├──► data/{fred,yahoo}_client.py ──► data/targets.py ──► data/merge.py
Yahoo ─┘         (cached, allowlisted)      (Garman-Klass)       (aligned)
                                                                      │
                                                                      ▼
                                                        backtest/splitter.py
                                                        137 non-overlapping folds
                                                                      │
                                                                      ▼
                                    ┌───────── backtest/runner.py ─────────┐
                                    │   for fold: for model: fit, predict  │
                                    │   no branching on model identity     │
                                    └──────────────────┬───────────────────┘
                                                       ▼
                                        backtest/writer.py — one tidy schema
                                                       │
                                         ┌─────────────┴─────────────┐
                                         ▼                           ▼
                              evaluation/aggregate.py            viz/*.py
                              evaluation/stats.py                   │
                                         │                          ▼
                                         └──► docs/  ◄──────  space/app.py
```

## Why the harness is ours

`DECISIONS.md` D15 splits the difference deliberately. `darts` supplies ARIMA, N-BEATS and
the probabilistic RNN; `statsmodels` supplies HAR and SARIMAX; `chronos-forecasting`
supplies the foundation models. But the **fold generation, refit cadence, leakage
enforcement and scoring are ours**.

The harness never calls `darts.historical_forecasts`. Outsourcing the backtest would mean
the foundation models and the classical models no longer *provably* traverse identical code,
which is the single claim the harness exists to support. It is also the artifact a reviewer
reads first: they want to see your leakage guards, not confirm that a library has some.

## The parameter/state split

`fit()` takes `refit_parameters`, and `BaseForecaster` splits into two abstract halves:

- `_estimate_parameters` — everything *learned* from data: coefficients, ARIMA orders,
  residual quantiles, network weights. The refit cadence governs how often this runs.
- `_update_state` — the data a forecast is *conditioned* on: the last observation, the
  context window. This runs on **every** fold, whatever the cadence.

Both are abstract rather than one defaulting to the other, so adding a model forces an
explicit decision about which of its attributes are which.

This split exists because getting it wrong is invisible. An earlier version cached the whole
fitted object between refits, which froze each model's conditioning data along with its
parameters — 124 of 137 random-walk forecasts ran from a value a median of 84 trading days
old, inflating every skill score quoted against that baseline. Nothing raised.
`runner.assert_conditioned_on_origin` now fires on every fit.

## One schema, computed once

`backtest/writer.py` defines the results format exactly once:

```
origin, target_date, step, model_id, quantile, value, actual,
regime, block_id, series, arm, cadence
```

Everything downstream — metrics, plots, significance tests, the Space — reads that and only
that. It is why the demo cannot show a number the README disagrees with: they come from the
same file, and a test asserts the Space imports no metric code at all.

## Where each thing runs

| Workload | Where | Why |
|---|---|---|
| Data, harness, classical models | Local CPU | Minutes |
| Chronos zero-shot **and fine-tuned inference** | Local CPU | 120M params, 0.85 s/forecast |
| LoRA fine-tuning | Colab H100 | The only genuine GPU requirement |
| Neural training | Colab H100 | Retrains per block, per window |
| Metrics, tests, plots | Local | Pure numpy/scipy |
| Demo | HF Space, CPU Basic | No per-visitor quota, no queue |

The rule keeping this coherent: **Colab notebooks contain no modelling logic.** They install
the package, call a function, and push the artifact. A test enforces it.
