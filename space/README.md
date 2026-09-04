---
title: forecast_bench — AI vs Classical Forecasting
emoji: 📉
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: Classical statistics vs AI foundation models
---

# forecast_bench

Can a pretrained AI "foundation" model forecast financial markets better than statistical
methods from the 1970s?

**Short answer: not here — but it needs far less data to try.**

## What this is

A leakage-safe benchmark of eleven models on two financial series, all scored through
identical code on the same 137 forecast dates. The conclusion was written down and
committed to git *before any model ran*, so the goalposts could not move afterwards.

- **Code:** https://github.com/Rohanjain2312/forecast_bench
- **Pre-registration:** committed before the first result existed
- **Full results:** `docs/benchmark_results.md`, generated from the data rather than typed

## The finding

The fine-tuned Chronos-2 **lost** by the pre-registered rule: it could not beat a random
walk on the Treasury yield, and never reached statistical significance against the best
classical model. LogHAR — a volatility model from 2009 that fits in about forty lines —
was the strongest performer.

But given only *one year* of training data, the foundation model retained about **85%** of
its full accuracy, while from-scratch neural networks were worse than a coin flip. Pretraining
did not buy accuracy here. It bought not needing much data.

## Honest caveats

Chronos-2 was released in October 2025 and most of the test period predates it, so
untouched-model results on the early span may not be a fair test — the training data is not
public and this cannot be resolved. At 137 forecast dates, most of these models are not
statistically distinguishable. See the **What am I looking at?** tab.

## Hardware

CPU Basic, deliberately. ZeroGPU charges GPU quota to the *visitor*, and a recruiter who
hits a quota error has already formed an opinion. Chronos-2 forecasts in under a second on
CPU, with no queue and no per-visitor limit.
