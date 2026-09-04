# Documentation

A map of what is written down, and which question each document answers.

## Start here

| Document | Answers |
|---|---|
| [`../README.md`](../README.md) | What is this and what did it find? |
| [`quickstart.md`](quickstart.md) | How do I run it? |
| [`../PREREGISTRATION.md`](../PREREGISTRATION.md) | What counted as the AI model losing, decided in advance? |

## The results

| Document | Answers |
|---|---|
| [`benchmark_results.md`](benchmark_results.md) | All six results tables, full precision. Generated, never typed. |
| [`limitations.md`](limitations.md) | What does this **not** establish? |
| [`model_cards.md`](model_cards.md) | What does each model actually do? |

## How it works

| Document | Answers |
|---|---|
| [`architecture.md`](architecture.md) | How do the pieces fit together? |
| [`methodology.md`](methodology.md) | Why this fold scheme, these metrics, these tests? |
| [`data_protocol.md`](data_protocol.md) | How is look-ahead bias prevented, and what bug motivated it? |

## Why it is built this way

| Document | Answers |
|---|---|
| [`planning/DECISIONS.md`](planning/DECISIONS.md) | Why every design choice is what it is. |
| [`planning/IMPLEMENTATION_PLAN.md`](planning/IMPLEMENTATION_PLAN.md) | The build plan the code was written against. |
| [`planning/PROGRESS_NOTES.md`](planning/PROGRESS_NOTES.md) | What actually happened, including every bug found and why it mattered. |

`PROGRESS_NOTES.md` is the most useful document here for anyone judging the work rather than
using it. It records the bugs — stale conditioning that inflated every skill score, interval
scaling that made HAR look worse than it is, early stopping that never ran — each of which
produced *plausible* numbers and none of which any metric would have revealed.

## Per-module documentation

Each package directory carries its own README covering what lives there, what depends on it,
and what would break if you changed it:
[`data`](../forecast_bench/data/README.md) ·
[`backtest`](../forecast_bench/backtest/README.md) ·
[`models`](../forecast_bench/models/README.md) ·
[`evaluation`](../forecast_bench/evaluation/README.md)
