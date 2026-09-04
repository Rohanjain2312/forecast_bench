# Limitations

What this benchmark does **not** establish, written at the same level of care as the
results. A study whose entire claim is that its numbers can be trusted has to be equally
precise about where they stop applying.

Every limitation here was either registered in advance (`PREREGISTRATION.md` §6) or found
during the build and recorded in `docs/planning/PROGRESS_NOTES.md` as it happened.

---

## 1. Pretraining contamination — the largest threat, and it is unfixable

Chronos-2 was released in **October 2025**. Chronos-Bolt earlier still. The test span runs
**2015-01-01 to 2026-06-30**, so the overwhelming majority of it *predates the model's own
release*.

SPY and Treasury yields are among the most widely redistributed time series in existence.
The pretraining corpora are not published and cannot be inspected, so there is no way to
establish whether a "zero-shot" forecast of 2019 volatility is genuinely out-of-sample or a
partial recall of data the model already saw.

**This is not fixable, only bounded.** Three things follow, and all three are done:

1. **Every zero-shot number on the pre-release span is potentially inflated.** It is never
   presented as a clean out-of-sample result.
2. **A contamination-free table exists** (`benchmark_results.md` §6), restricted to origins
   after 2025-11-01. It contains **7 forecast origins**. That is far too few for
   significance, and the count is printed inside the table rather than in a footnote, so
   no number can be lifted from it without its sample size attached.
3. **The asymmetry is the useful part.** Contamination inflates *zero-shot* results, not
   fine-tuned ones — the adapters were fitted on our own data with our own cutoffs. So the
   **fine-tuned-minus-zero-shot gap remains interpretable** even where the absolute
   zero-shot level does not. Where this study makes a claim about foundation models, it
   leans on that gap rather than on the raw zero-shot number.

The classical models have no equivalent problem. They are fitted from scratch inside every
fold and cannot have seen anything.

## 2. The sample is small for the statistics being run

**137 non-overlapping forecast origins** on SPY log-RV, 136 on `DGS10`. That is a small
sample for Diebold-Mariano, and the study does not pretend otherwise.

Three guards, all reported:

- The Harvey-Leybourne-Newbold small-sample correction is applied, with a Newey-West HAC
  variance.
- A **Model Confidence Set** is reported alongside every pairwise test. It is the more
  honest summary at this sample size, and it materially changes the reading: the fine-tuned
  Chronos-2 survives in the MCS at **every series and horizon**. It lost the pre-registered
  comparison, but it is *not* demonstrably worse than the models that beat it. Most of this
  panel is statistically indistinguishable on 137 origins.
- Headline claims are restricted to the comparisons registered in advance. Every other test
  in the results tables is descriptive.

The HAC truncation lag is `horizon - 1`, the Diebold-Mariano convention. Because this
study's forecast windows do **not** overlap, that lag is conservative rather than necessary,
and at `horizon = 21` a lag-20 HAC estimate on 137 observations is itself noisy.

## 3. The covariate set is deliberately narrow

Only daily, market-observed FRED series that FRED does not revise may enter a model:
`DGS10`, `DGS3MO`, `T10Y2Y`, `VIXCLS`, `DFF`.

This excludes inflation, employment, and every other revised macro series a practitioner
would reach for. **That is a real loss of scope**, taken deliberately: revised series are
indexed by reference period rather than release date, so using them would make the
point-in-time claim a matter of careful bookkeeping instead of something true by
construction. See `docs/data_protocol.md`.

Consequently this study says nothing about forecasting rates from macro fundamentals, and
nothing about how data revisions themselves affect forecast quality.

## 4. Chronos-Bolt cannot produce the study's tail quantiles

Bolt was trained on quantile levels 0.1–0.9 only. Requests for 0.025 and 0.975 return its
0.1 and 0.9 predictions unchanged, so **its 95% interval is identical to its 80% interval
by construction**. This is visible directly in the results: Bolt's `coverage_95` equals its
`coverage_80` exactly, in every row.

This costs Bolt something real on the primary metric, since weighted quantile loss averages
over all eleven levels and two of Bolt's eleven are duplicates. It was left in place rather
than worked around: extrapolating tails the checkpoint was never trained to produce would be
inventing a capability to improve a number. Every Bolt figure carries this caveat.

## 5. The two model families were trained in different numerical environments

The neural baselines (N-BEATS, DeepAR-class LSTM) were trained on a Colab A100 with
TF32 matmul precision enabled. The Chronos fine-tuning ran earlier, without it. The
classical models run on CPU in float64.

Within any single comparison the numerics are uniform — that constraint drove several
decisions during the build — but across model families they are not identical. The effect
is far below the noise floor of the task, and it is recorded here rather than left implicit.

## 6. Arm B is incomplete

The covariate-informed arm does not include covariate-informed foundation models.
`DECISIONS.md` D3 specifies Chronos-2 using its native covariate support; the implemented
wrapper conditions only on the target's own history, and the fine-tuned adapters were
trained on Arm A alone. Arm B therefore compares SARIMAX against the univariate panel, which
is a narrower question than the one D3 posed.

Reported as incomplete rather than quietly relabelled.

## 7. What a single benchmark can establish at all

Two series, one asset class, one eleven-year test span, one set of hyperparameters fixed in
advance. The result that a fine-tuned Chronos-2 does not beat LogHAR on SPY realized
variance is a fact about **this comparison**, not a general claim about foundation models,
volatility forecasting, or either model family.

The strongest and most transferable finding here is not the headline at all. It is the
**sample-efficiency curve**: the fine-tuned foundation model retains **85%** of its
full-window skill on one year of data, while the from-scratch neural models are *worse than
a random walk* at that size. That is a claim about where pretraining pays for itself, and it
points in the opposite direction from the headline verdict. Both are reported, because
reporting only the one that fits a narrative is the failure this project was built to avoid.
