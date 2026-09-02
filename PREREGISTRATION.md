# Pre-Registration

**Committed before any model was run. The git timestamp on this file is the evidence.**

This document exists because the easiest way to produce a dishonest benchmark is to run
everything, see what looks good, and then decide what the headline claim is. Writing the
decision rules down first and committing them removes that option.

If any statement in this file turns out to be wrong or unworkable once results exist, the
correct response is to add an amendment at the bottom explaining what changed and why —
**not** to edit the text above it. The diff history is the point.

---

## 1. What is being tested

Whether a fine-tuned time-series foundation model (Chronos-2) outperforms classical
statistical baselines on two financial forecasting tasks, under a leakage-safe walk-forward
backtest in which every model is scored by identical code.

## 2. Targets, horizons, arms — fixed in advance

- **Targets:** SPY log realized variance (Garman-Klass, daily OHLC); `DGS10` in levels
- **Horizons:** 1, 5, 21 trading days
- **Arms:** A (univariate, headline), B (covariate-informed, secondary)
- **Cadences:** matched (headline), native (secondary)
- **Test span:** 2015-01-01 → 2026-06-30, expanding origin, stride 21, non-overlapping
- **Primary metric:** weighted quantile loss (WQL), reported as skill score vs. random walk

The headline result is: **Arm A, matched cadence, WQL skill score vs. random walk.** Every
other cut is secondary and will be labelled as such.

## 3. The losing condition

> The fine-tuned foundation model is reported as **losing** if either:
>
> **(a)** it fails to beat the random-walk baseline on WQL skill score at h=1 on either
> series, or
>
> **(b)** it fails to achieve Diebold-Mariano significance at p<0.05 against the best
> classical model at any of the three horizons on either series, under the matched refit
> cadence, Arm A.

If either condition holds, the README results section and the Medium writeup will say so
in their first paragraph, using the word "lost" or an equally direct equivalent. Not
"showed mixed results." Not "was competitive."

## 4. Predictions registered in advance

Stated now so they can be checked against outcomes later. Being wrong about these is fine
and will be reported; quietly not having made them is not.

1. **On `DGS10`, no model will beat random walk by a meaningful margin at any horizon.**
   The series is near-unit-root. Expected skill scores in the range -0.05 to +0.05.
2. **On SPY log-RV, HAR or LogHAR will beat zero-shot Chronos-2 at h=1 and h=5.** Recent
   published benchmarks on realized volatility support this.
3. **Fine-tuning will improve on zero-shot on SPY log-RV, but by less than the gap between
   zero-shot and HAR.** That is, fine-tuning helps and does not close the gap.
4. **The foundation model's relative position will improve as horizon lengthens.** The
   h=21 gap will be smaller than the h=1 gap.
5. **The foundation model will show better sample efficiency** — reaching its plateau on a
   smaller training window than N-BEATS requires.

## 5. Changes committed to NOT making after seeing results

- Not adding, removing, or swapping models in the panel
- Not changing the primary metric from WQL skill score
- Not changing the test span, stride, horizon set, or fold structure
- Not re-tuning ARIMA grid bounds, LoRA hyperparameters, or context length
- Not recomputing the VIX tercile thresholds in `experiments/configs/regimes.yaml`
- Not promoting a secondary cut (Arm B, native cadence, a single regime, a single horizon)
  to headline status because it looks better than the registered headline
- Not dropping a target series because its result is uninteresting

Additional analyses may be added after seeing results. They will be labelled **exploratory**
in every table and in the writeup, and they will never appear above the registered headline.

## 6. Known threats to validity, registered in advance

**Pretraining contamination.** Chronos-2 (released October 2025) and TimesFM 2.5
(September 2025) were pretrained on large corpora that plausibly include public financial
series. SPY and Treasury yields are among the most widely redistributed series in
existence. Most of the test span predates those releases, so zero-shot results on that span
may not be genuinely out-of-sample.

This is not fixable — the pretraining corpora are not inspectable. It will be handled by:

- Stating it as a first-class limitation in the README, `docs/limitations.md`, and the
  demo, not as a footnote
- Reporting a separate contamination-free table restricted to forecast origins after each
  model's release date, with its inadequate sample size (~8 folds) stated *in the table*
- Noting that contamination inflates zero-shot numbers but not fine-tuned ones, so the
  fine-tuned-vs-zero-shot gap remains interpretable

**Small fold count.** ~137 non-overlapping folds is small for Diebold-Mariano. Mitigated by
reporting a Model Confidence Set alongside pairwise tests, and by bootstrap confidence
intervals on every headline number. No claim will rest on a single p-value.

**Multiple comparisons.** Approximately 7 models × 2 series × 3 horizons × 4 regime cuts.
Headline claims are restricted to the comparisons registered in §3 above. All other
significance tests are reported as descriptive.

**Restricted covariate set.** Only non-revised daily series are used, which excludes
inflation and employment data that a practitioner might want. This is a deliberate trade of
scope for an unarguable point-in-time claim, and is stated in `docs/limitations.md`.

---

## Amendments

*(Any deviation from the above gets an entry here, with a date and a reason. An empty
section at the end of the project is the ideal outcome; a section with honest entries is
the second-best. An edit to the text above with no entry here is the failure mode this
document exists to prevent.)*
