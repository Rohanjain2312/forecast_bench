# `forecast_bench/models`

The panel. Thirteen model classes across four families, all behind one interface.

| Path | Contents |
|---|---|
| `base.py` | `BaseForecaster`, the parameter/state split, and quantile helpers |
| `registry.py` | **Where the panel is defined.** Adding a model means one entry here plus one file. |
| `naive.py` | RandomWalk, SeasonalNaive |
| `classical/` | ARIMA, SARIMAX, HAR, LogHAR, AR(1) |
| `neural/` | N-BEATS and a DeepAR-class LSTM, via darts |
| `foundation/` | Chronos-2 and Chronos-Bolt, zero-shot and LoRA-adapted, plus the fine-tuning recipe |

## The parameter/state split

`BaseForecaster` requires two methods rather than one:

- `_estimate_parameters` — anything **learned** from data. The refit cadence governs it.
- `_update_state` — the data the forecast is **conditioned** on. Runs every fold, always.

Both are abstract, so adding a model forces an explicit decision about which of its
attributes are which. Getting it wrong is invisible: an early version froze conditioning
along with parameters and left 124 of 137 random-walk forecasts running from a value a
median of 84 trading days old, inflating every skill score in the study.

## What would break if you changed it

- **Registering a fine-tuned model whose adapters do not exist** fails at whichever block
  first goes looking. `_FINETUNED` encodes what was actually trained: Chronos-2 on both
  series, Bolt on the volatility track only.
- **Applying a LoRA adapter to a cached base pipeline** would turn every zero-shot model in
  the process into a fine-tuned one, since `PeftModel.from_pretrained` wraps the object it is
  given. `load_finetuned_pipeline` always loads a fresh base, and a test asserts zero-shot
  forecasts are unchanged after a fine-tuned model has been used.
- **Un-centring the naive baselines' quantiles** turns the random walk into a random walk
  *with drift* — a different and stronger model, and the one every skill score is measured
  against.
- **Assuming a variance growth law** instead of measuring it. HAR's intervals once used
  `sqrt(h)` scaling, correct for an integrated process and wrong by a factor of 3.7 at h=21
  for a mean-reverting one.
