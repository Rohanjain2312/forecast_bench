---
title: Testing an AI forecasting model against classical statistics, with the rules set in advance
subtitle: A benchmark on real financial data, and what I took away from building it.
---

Most AI benchmarks have a weak spot: whoever runs them also decides, after seeing the results, what counts as a win. Nobody has to lie about it. They just run a few comparisons and write about the one that looks good.

To avoid that, I wrote down what "losing" would mean before writing any model code, committed that document to git, and left it unedited afterward. Here's what I tested, what happened, and what I took away from it.

## The setup

I compared two kinds of forecasting model on real financial data:

- **Classical statistics**: ARIMA, HAR, and a random walk baseline. Decades old.
- **Chronos-2**, a foundation model: same idea as a language model like ChatGPT, but trained on millions of time series instead of text. It can forecast a new series without being trained on it specifically. I tested it both out of the box and fine-tuned.

Two targets, chosen because I expected them to behave differently:

1. **Stock market volatility (SPY)**: calm and turbulent periods tend to cluster, so there should be real structure to find.
2. **The 10-year Treasury yield**: moves close to a random walk, so nothing should reliably beat "tomorrow looks like today."

Both models forecast the same 137 dates through identical code, scored the same way. I also wrote an automated test that fails the build if any model can see data from after the date it's forecasting. That's an easy mistake to make by accident, for example by fitting a scaler on the full dataset before splitting it.

## The result

The fine-tuned foundation model lost, by the rule set in advance: it needed to beat a random walk on the yield, and beat the best classical model by a meaningful margin on either target. Neither happened.

| Model | Skill vs. random walk (volatility) |
|---|---:|
| ARIMA | +16.8% |
| LogHAR | +16.6% |
| Chronos-2, fine-tuned | +15.3% |
| Chronos-2, zero-shot | +15.1% |

Classical models came out slightly ahead, and the gap held under a significance test. On the Treasury yield, nothing beat the random walk by more than noise. That matched the prediction I made beforehand, and it's a good sanity check that the setup can tell a real signal from none.

## The part I didn't expect

I reran everything with just one year of training data instead of fifteen.

| Model | 1 year of data | Full 15 years |
|---|---:|---:|
| Chronos-2, fine-tuned | +13.0% | +15.3% |
| Neural network, trained from scratch | -516% | +16.9% |

- The from-scratch model didn't just get worse. It performed far worse than guessing tomorrow looks like today.
- The pretrained model barely lost ground: 85% of its full-data performance using less than 7% of the data.

So the real advantage of a foundation model here wasn't accuracy. Classical statistics matched or beat it when data was plentiful. It was resilience when data was scarce, because it already had a general sense of what time series look like before seeing this one.

## Why pre-registering mattered

Without it, I could have written "foundation model rivals decades of statistics," which is defensible but beside the point, or led with the loss and left out the one-year result, which buries the most useful finding. Pre-registering doesn't prevent a bad result; it prevents hiding one. The commit defining the losing condition is dated before any model existed in the repo, so once results came in, I couldn't redefine success.

## Caveats

- Chronos-2 was released in October 2025, but the test data goes back to 2015. It may have seen similar data during training, which would flatter its "zero-shot" numbers. I also reported results restricted to dates after the model's release, and said plainly that the sample there is too small to lean on alone.
- 137 forecast dates is a small sample for a significance test. I applied the standard small-sample correction, and reported a second, more lenient test alongside it — most models are closer together than the headline numbers suggest.

## What I learned

- **A benchmark is only as trustworthy as the process behind it.** Writing down the losing condition before running any models cost almost nothing, and it's the only reason I can call this result honest instead of convenient.
- **"Newer AI model wins" isn't a safe assumption.** On a fair, data-rich comparison, plain statistics still beat the pretrained model here. Worth actually checking instead of assuming.
- **The most useful result wasn't the one I set out to measure.** I designed this to answer "which model wins," and the more useful answer turned out to be "which model still works with little data." That only showed up because I tested a scenario I hadn't planned to report on.

Live demo: https://huggingface.co/spaces/rohanjain2312/forecastbench-demo
Code: https://github.com/Rohanjain2312/forecast_bench
