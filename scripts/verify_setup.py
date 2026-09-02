"""Verify every external dependency of the study before any modelling code runs.

Five checks, each isolating a different class of setup failure:

1. FRED API key works
2. Yahoo Finance returns usable SPY OHLC
3. Chronos-2 loads and forecasts on CPU, and *how fast* — this number decides the demo
   architecture (DECISIONS.md D12)
4. Hugging Face token has access to all three project repos
5. The Space is configured as expected (reported, never blocking)

Checks 1-4 are blocking: the script exits non-zero if any fails. Check 5 is informational,
because its remedy is a click in a browser rather than anything code can do.

Run with::

    poetry run python -m scripts.verify_setup

No secret is ever printed, including in error paths.
"""

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON, get_config

#: Model used for the latency measurement. Pinned to the study's core model (DECISIONS.md D13).
CHRONOS_MODEL_ID = "amazon/chronos-2"


@dataclass
class CheckResult:
    """Outcome of a single verification check.

    Attributes:
        name: Short label shown in the results table.
        passed: Whether the check succeeded.
        detail: One-line human-readable finding. Never contains a secret.
        blocking: Whether a failure should fail the whole script.
    """

    name: str
    passed: bool
    detail: str
    blocking: bool = True


def check_fred() -> CheckResult:
    """Pull the last three observations of DGS10 to prove the FRED key works."""
    from fredapi import Fred

    series = Fred(api_key=get_config().require_secret("fred_api_key")).get_series(
        "DGS10"
    )
    tail = series.dropna().tail(3)
    if tail.empty:
        return CheckResult("FRED (DGS10)", False, "returned no observations")
    values = ", ".join(f"{date.date()}={value:.2f}" for date, value in tail.items())
    return CheckResult("FRED (DGS10)", True, f"last 3: {values}")


def check_yahoo() -> CheckResult:
    """Pull ten days of SPY OHLC and confirm all four price columns are present.

    Note:
        yfinance returns a MultiIndex column frame for some request shapes. This check
        flattens defensively; ``data/yahoo_client.py`` handles it properly in Step 5.
    """
    import yfinance as yf

    frame = yf.download(
        "SPY",
        start="2024-01-02",
        end="2024-01-17",
        progress=False,
        auto_adjust=False,
    )
    if frame is None or frame.empty:
        return CheckResult("Yahoo (SPY OHLC)", False, "returned an empty frame")

    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        columns = columns.get_level_values(0)
    present = {str(name).lower() for name in columns}

    required = {"open", "high", "low", "close"}
    missing = required - present
    if missing:
        return CheckResult(
            "Yahoo (SPY OHLC)", False, f"missing columns: {sorted(missing)}"
        )
    return CheckResult(
        "Yahoo (SPY OHLC)",
        True,
        f"{len(frame)} bars, O/H/L/C all present",
    )


def check_chronos() -> CheckResult:
    """Load Chronos-2 on CPU and time one 21-step quantile forecast from 512 points.

    The measured latency is the input to the demo-architecture decision in
    DECISIONS.md D12: under ~5s means live inference on CPU Basic is viable, over ~10s
    means the Space serves pre-computed forecasts over a fixed date grid instead.
    """
    from chronos import Chronos2Pipeline

    load_start = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL_ID, device_map="cpu")
    load_seconds = time.perf_counter() - load_start

    # A random walk, not white noise: it is the shape of the series we actually forecast,
    # so the timing reflects realistic work rather than a degenerate input.
    rng = np.random.default_rng(0)
    context = rng.standard_normal(CONTEXT_LENGTH).cumsum().astype("float32")

    predict_start = time.perf_counter()
    _quantiles, _mean = pipeline.predict_quantiles(
        [context],
        prediction_length=MAX_HORIZON,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    predict_seconds = time.perf_counter() - predict_start

    return CheckResult(
        "Chronos-2 on CPU",
        True,
        f"load {load_seconds:.1f}s, forecast {predict_seconds:.2f}s "
        f"({CONTEXT_LENGTH} ctx -> {MAX_HORIZON} steps)",
    )


def check_huggingface() -> CheckResult:
    """Confirm the HF token authenticates and can see all three project repos."""
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    username = api.whoami()["name"]

    targets = [
        (config.hf_model_repo, "model"),
        (config.hf_dataset_repo, "dataset"),
        (config.hf_space_repo, "space"),
    ]
    unreachable: list[str] = []
    for repo_id, repo_type in targets:
        try:
            api.repo_info(repo_id, repo_type=repo_type)
        except (
            Exception
        ) as error:  # noqa: BLE001 - report which repo, not the traceback
            unreachable.append(f"{repo_id} ({type(error).__name__})")

    if unreachable:
        return CheckResult(
            "Hugging Face access", False, f"unreachable: {'; '.join(unreachable)}"
        )
    return CheckResult(
        "Hugging Face access",
        True,
        f"authenticated as {username}; all 3 repos reachable",
    )


def check_space_config() -> CheckResult:
    """Report the Space's SDK and current hardware. Never blocks.

    Expected ``gradio`` on ``cpu-basic``. Anything else — ZeroGPU especially — is flagged
    for the user to change in the browser, which is why this check reports rather than
    fails. See DECISIONS.md D12 for why CPU Basic is the deliberate choice.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    info = api.space_info(config.hf_space_repo)

    sdk = getattr(info, "sdk", None) or "unknown"
    runtime = getattr(info, "runtime", None)
    stage = getattr(runtime, "stage", None) or "unknown"

    # A Space that has never built reports hardware=None, because "current" hardware only
    # exists once something has run. The setting that matters in that state is the
    # *requested* hardware — reading only `hardware` would report "unknown" for a Space
    # sitting on ZeroGPU and quietly pass the very case this check exists to catch.
    current = getattr(runtime, "hardware", None)
    requested = getattr(runtime, "requested_hardware", None)
    effective = current or requested or "unknown"

    as_expected = sdk == "gradio" and effective == "cpu-basic"
    detail = f"sdk={sdk}, hardware={effective}"
    if current is None and requested is not None:
        detail += " (requested; Space has never built)"
    detail += f", stage={stage}"
    return CheckResult("Space configuration", as_expected, detail, blocking=False)


CHECKS: list[Callable[[], CheckResult]] = [
    check_fred,
    check_yahoo,
    check_chronos,
    check_huggingface,
    check_space_config,
]


def run_checks() -> list[CheckResult]:
    """Run every check, converting an unexpected exception into a failed result.

    Returns:
        One :class:`CheckResult` per check, in declaration order.
    """
    results: list[CheckResult] = []
    for check in CHECKS:
        label = check.__name__.removeprefix("check_")
        print(f"  running {label} ...", flush=True)
        try:
            results.append(check())
        except (
            Exception
        ) as error:  # noqa: BLE001 - a failed check is a result, not a crash
            traceback.print_exc()
            results.append(
                CheckResult(label, False, f"{type(error).__name__}: {error}")
            )
    return results


def print_table(results: list[CheckResult]) -> None:
    """Print the pass/fail table.

    Args:
        results: Outcomes in the order they should be displayed.
    """
    width = max(len(result.name) for result in results)
    print()
    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    print(f"{'-' * width}  ------  {'-' * 48}")
    for result in results:
        status = "PASS  " if result.passed else "FAIL  "
        if not result.passed and not result.blocking:
            status = "REVIEW"
        print(f"{result.name.ljust(width)}  {status}  {result.detail}")
    print()


def main() -> int:
    """Run every check and return a process exit code.

    Returns:
        ``0`` if every blocking check passed, ``1`` otherwise.
    """
    print("Verifying forecast_bench setup. Nothing here prints a secret.\n")
    results = run_checks()
    print_table(results)

    failed = [r for r in results if not r.passed and r.blocking]
    if failed:
        print(
            f"{len(failed)} blocking check(s) failed: {', '.join(r.name for r in failed)}"
        )
        return 1
    print("All blocking checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
