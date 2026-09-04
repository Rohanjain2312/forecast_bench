# `forecast_bench/evaluation`

Metrics, significance testing, regime stratification, and the six results tables.

| File | Role |
|---|---|
| `metrics.py` | **The single source of truth for every metric.** Import it; never reimplement. |
| `stats.py` | Diebold-Mariano (HLN + HAC), Model Confidence Set, block bootstrap |
| `regimes.py` | Frozen VIX terciles, verified against the committed config at import |
| `aggregate.py` | The six results tables |

## What depends on this

`scripts/build_results.py`, `scripts/evaluate_preregistration.py`, `docs/benchmark_results.md`,
and the Space — though the Space reads only the *published tables*, never this code. A test
asserts `space/app.py` imports nothing from `evaluation.metrics`, which is what makes "no
number in the demo that is not in the docs" enforceable rather than aspirational.

## Two APIs that encode rules rather than documenting them

**`interval_coverage_and_width` returns a pair**, and no function returns coverage alone. A
model can buy coverage with uselessly wide intervals; an API offering only the flattering
half invites exactly that.

**The MASE denominator is a separate function taking a training window.** Computing it on
the full series — the most common way MASE is reported wrongly — requires deliberately
passing the wrong thing.

## What would break if you changed it

- **Recomputing the regime thresholds** defines "stressed" using knowledge that 2020 and
  2022 happened. `regimes.py` asserts against the committed values at import and raises.
- **Changing the fold stride** invalidates `diebold_mariano`, whose docstring says so.
- **Reimplementing a metric in a notebook or the Space** creates a second definition. One of
  them will be wrong and nobody will know which.
