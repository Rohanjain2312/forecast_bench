# forecast_bench

> **Status: under construction.** This README is a placeholder. The full version — with the
> results table, the architecture diagram, and the plain-language explanation — is written
> at the end of the build, once results exist. Writing it earlier would mean writing a
> conclusion before there is one.

An honestly-reported forecasting benchmark: classical statistical models against a
fine-tuned time-series foundation model, on real financial data, under a leakage-safe
walk-forward backtest in which every model is scored by identical code.

The deliverable is not "foundation models are better." It is a map of where each model
class wins and loses, with a pre-registered definition of what losing looks like.

- **Targets:** SPY log realized variance (Garman-Klass, daily OHLC) and the 10-year
  Treasury yield (`DGS10`) in levels — two series chosen for having opposite expected answers.
- **Pre-registration:** [`PREREGISTRATION.md`](PREREGISTRATION.md), committed to git before
  any model code existed. The timestamp is the evidence.
- **Design rationale:** [`docs/planning/DECISIONS.md`](docs/planning/DECISIONS.md).

## Installation

```bash
git clone git@github.com:Rohanjain2312/forecast_bench.git
cd forecast_bench
poetry install
cp .env.example .env    # then fill in the placeholders
```

## License

MIT — see [`LICENSE`](LICENSE).
