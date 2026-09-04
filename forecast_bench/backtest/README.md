# `forecast_bench/backtest`

The harness. The most inspectable artifact in the repository, and the thing a reviewer
should read first.

| File | Role |
|---|---|
| `protocol.py` | `Forecaster` and `QuantileForecast`. **Read this first.** |
| `splitter.py` | Expanding-origin folds. Refuses `stride != horizon`. |
| `cadence.py` | `EveryFoldCadence` and `BlockCadence` behind one interface. |
| `runner.py` | The loop. Drives every model fold-by-fold, with no branching on identity. |
| `writer.py` | The tidy results schema, defined exactly once. |

## What depends on this

Everything. `models/` implements `protocol.py`; `evaluation/` consumes `writer.py`'s schema;
the Space and the docs read tables built from it.

## What would break if you changed it

- **Changing `stride` away from `horizon`** makes forecast windows overlap, which
  invalidates every Diebold-Mariano p-value in the study. `splitter.py` raises rather than
  letting this happen quietly.
- **Adding a branch on `model_id` in `runner.py`** breaks the claim that every model
  traverses identical code — the one thing the harness exists to establish. If a model seems
  to need special handling, `protocol.py` is wrong, not the runner.
- **Letting the cadence gate conditioning data** rather than only parameters silently
  freezes each model's view of the world between refits. That bug inflated every skill score
  in an early run; `assert_conditioned_on_origin` now fires on every fit.
- **Adding a column to `writer.SCHEMA`** without updating `evaluation/aggregate.py` produces
  tables that silently drop it.

## The guards that run on every fold

`assert_fold_is_clean` checks two different things, because they catch different failures: no
training row postdates the origin, and no column is a near-perfect copy of a future target.
The second exists because the first cannot catch it — a `target.shift(-21)` column sits
entirely on rows dated before the origin, and every date comparison passes.
