# `forecast_bench/data`

Acquisition and construction of the two modelling series, plus the point-in-time
guarantees that make the study's central claim true by construction.

| File | Role |
|---|---|
| `fred_client.py` | Cached FRED pulls. **Enforces the non-revised allowlist.** |
| `yahoo_client.py` | Cached SPY OHLC. Absorbs yfinance's MultiIndex and column-name quirks. |
| `_cache.py` | Parquet cache with `.meta.json` provenance sidecars, shared by both clients. |
| `targets.py` | Garman-Klass log realized variance; `DGS10` in levels. |
| `covariates.py` | Arm B covariate sets. Allowlisted series only. |
| `merge.py` | Business-day alignment and gap handling. |
| `hub.py` | Push/pull the processed series to `forecastbench-data`. |

## Why the allowlist exists

Only daily, market-observed FRED series that FRED does not revise may enter a model:
`DGS10`, `DGS3MO`, `T10Y2Y`, `VIXCLS`, `DFF`. `fred_client.assert_non_revised` raises on
anything else.

FRED indexes revised series by their **reference period**, not their **release date**.
March CPI carries a `2024-03-01` index but is not published until mid-April, so a model
reading `CPIAUCSL` at a `2024-03-15` forecast origin is reading the future. No metric in
this study would reveal that. Restricting the input set makes the point-in-time claim
structural rather than a matter of careful bookkeeping.

This is the bug this repository is designed to prevent — see `docs/data_protocol.md` for
the full account, and `DECISIONS.md` D10-G1.

## What would break if you changed it

- **Adding a series to the allowlist** without checking FRED's revision policy silently
  invalidates every result. This is the single most damaging one-line change possible here.
- **Forward-filling a target** across market holidays is a subtle leak. Both clients return
  NaNs as NaNs; the decision of what to do with a gap belongs to the caller and must be
  recorded.
- **Calling `yfinance` directly** rather than through `yahoo_client` reintroduces the
  MultiIndex and adjusted-close handling that this module centralises.
- **Deleting a `.meta.json` sidecar** makes its parquet untrusted: the cache treats a
  parquet without provenance as absent and refetches.
