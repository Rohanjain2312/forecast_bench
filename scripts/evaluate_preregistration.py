"""Apply the pre-registered decision rules to the actual results.

Run with::

    poetry run python -m scripts.evaluate_preregistration

``PREREGISTRATION.md`` §3 states the losing condition and §4 lists five predictions, all
committed before any model ran. This script evaluates them mechanically against the
forecast parquets, so the verdict follows from the rules rather than from reading a table
and deciding what it says.

Nothing here may be softened. If the condition triggers, it triggers.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from forecast_bench.config import HORIZONS, QUANTILE_GRID, get_config, setup_logging
from forecast_bench.evaluation.stats import diebold_mariano, model_confidence_set

logger = logging.getLogger(__name__)

#: The model the losing condition is about.
FOUNDATION_MODEL = "Chronos2-FineTuned"

#: Its unadapted counterpart, for the fine-tuning-versus-zero-shot gap.
ZERO_SHOT_MODEL = "Chronos2-ZeroShot"

#: The baseline every skill score is measured against.
BASELINE = "RandomWalk"

#: Models that count as "classical" for the losing condition's second clause.
CLASSICAL = {"ARIMA", "SARIMAX", "HAR", "LogHAR", "AR1", "SeasonalNaive", "RandomWalk"}


def per_origin_losses(block: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-origin weighted quantile loss for every model at one horizon.

    Args:
        block: Tidy forecasts for one series.
        horizon: Step to score.

    Returns:
        A frame indexed by origin with one column per model, holding that origin's
        quantile loss summed across the grid.

    Note:
        Diebold-Mariano needs a loss *series*, one value per forecast origin, not an
        aggregate. Summing pinball loss across the quantile grid at a fixed step gives
        exactly that, and the origins are non-overlapping by construction.
    """
    at_horizon = block[block["step"] == horizon]
    frames = []
    for model_id, rows in at_horizon.groupby("model_id"):
        wide = rows.pivot_table(index="origin", columns="quantile", values="value")
        actual = rows.groupby("origin")["actual"].first().reindex(wide.index)
        loss = np.zeros(len(wide))
        for level in QUANTILE_GRID:
            if level not in wide.columns:
                continue
            difference = actual.to_numpy() - wide[level].to_numpy()
            loss += np.maximum(level * difference, (level - 1.0) * difference)
        frames.append(pd.Series(loss, index=wide.index, name=model_id))
    return pd.concat(frames, axis=1).dropna()


def evaluate_losing_condition(
    headline: pd.DataFrame, forecasts: dict[str, pd.DataFrame]
) -> dict:
    """Evaluate PREREGISTRATION.md §3 against the results.

    Args:
        headline: The headline metrics table.
        forecasts: Tidy forecasts keyed by series.

    Returns:
        A verdict dictionary with both clauses evaluated and the evidence for each.
    """
    verdict: dict = {"clause_a": {}, "clause_b": {}, "lost": False}

    # (a) fails to beat the random walk on WQL skill at h=1 on either series.
    for series in sorted(forecasts):
        row = headline[
            (headline.series == series)
            & (headline.model_id == FOUNDATION_MODEL)
            & (headline.horizon == 1)
        ]
        if row.empty:
            verdict["clause_a"][series] = {"skill": None, "beat_random_walk": None}
            continue
        skill = float(row["skill_wql"].iloc[0])
        verdict["clause_a"][series] = {
            "skill": skill,
            "beat_random_walk": bool(skill > 0),
        }
        if skill <= 0:
            verdict["lost"] = True

    # (b) fails DM significance at p<0.05 against the best classical model, at any of the
    #     three horizons, on either series.
    any_significant_win = False
    for series, block in forecasts.items():
        available_classical = sorted(CLASSICAL & set(block.model_id.unique()))
        for horizon in HORIZONS:
            losses = per_origin_losses(block, horizon)
            if FOUNDATION_MODEL not in losses.columns:
                continue
            classical_means = losses[available_classical].mean()
            best_classical = str(classical_means.idxmin())

            result = diebold_mariano(
                losses[FOUNDATION_MODEL].to_numpy(),
                losses[best_classical].to_numpy(),
                horizon=horizon,
            )
            beat = result.mean_loss_differential < 0 and result.p_value < 0.05
            any_significant_win = any_significant_win or beat
            verdict["clause_b"][f"{series}_h{horizon}"] = {
                "best_classical": best_classical,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "favours_foundation": bool(result.mean_loss_differential < 0),
                "significant_win": bool(beat),
            }

    verdict["any_significant_win_over_best_classical"] = any_significant_win
    if not any_significant_win:
        verdict["lost"] = True
    return verdict


def evaluate_predictions(
    headline: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    sweep: pd.DataFrame | None,
) -> dict:
    """Evaluate the five predictions registered in PREREGISTRATION.md §4.

    Args:
        headline: The headline metrics table.
        forecasts: Tidy forecasts keyed by series.
        sweep: The sample-efficiency table, or ``None`` if the sweep has not been run.

    Returns:
        A mapping of prediction number to its outcome and the evidence behind it.
    """
    out: dict = {}

    def skill(series: str, model: str, horizon: int) -> float | None:
        """Look up one model's WQL skill score in the headline table.

        Args:
            series: Series name.
            model: Model id.
            horizon: Forecast horizon.

        Returns:
            The skill score, or ``None`` if that row is absent.
        """
        row = headline[
            (headline.series == series)
            & (headline.model_id == model)
            & (headline.horizon == horizon)
        ]
        return float(row["skill_wql"].iloc[0]) if not row.empty else None

    # 1. On DGS10, no model beats the random walk by a meaningful margin (-0.05..+0.05).
    rates = headline[(headline.series == "dgs10") & (headline.model_id != BASELINE)]
    best = rates.loc[rates["skill_wql"].idxmax()] if not rates.empty else None
    out["1"] = {
        "statement": "On DGS10, no model beats random walk by a meaningful margin",
        "best_skill": float(best["skill_wql"]) if best is not None else None,
        "best_model": str(best["model_id"]) if best is not None else None,
        "held": bool(best is not None and best["skill_wql"] <= 0.05),
    }

    # 2. On SPY log-RV, HAR or LogHAR beats zero-shot Chronos-2 at h=1 and h=5.
    checks = {}
    for horizon in (1, 5):
        har = max(
            [
                s
                for s in (
                    skill("spy_logrv", "HAR", horizon),
                    skill("spy_logrv", "LogHAR", horizon),
                )
                if s is not None
            ],
            default=None,
        )
        zero = skill("spy_logrv", ZERO_SHOT_MODEL, horizon)
        checks[f"h{horizon}"] = {
            "best_har": har,
            "zero_shot": zero,
            "har_wins": bool(har is not None and zero is not None and har > zero),
        }
    out["2"] = {
        "statement": "HAR/LogHAR beats zero-shot Chronos-2 at h=1 and h=5 on SPY log-RV",
        "detail": checks,
        "held": all(c["har_wins"] for c in checks.values()),
    }

    # 3. Fine-tuning improves on zero-shot, by less than the zero-shot-to-HAR gap.
    detail = {}
    for horizon in HORIZONS:
        tuned = skill("spy_logrv", FOUNDATION_MODEL, horizon)
        zero = skill("spy_logrv", ZERO_SHOT_MODEL, horizon)
        har = max(
            [
                s
                for s in (
                    skill("spy_logrv", "HAR", horizon),
                    skill("spy_logrv", "LogHAR", horizon),
                )
                if s is not None
            ],
            default=None,
        )
        if None in (tuned, zero, har):
            continue
        detail[f"h{horizon}"] = {
            "finetune_gain": tuned - zero,
            "gap_to_har": har - zero,
            "helps": bool(tuned > zero),
            "does_not_close_gap": bool(tuned < har),
        }
    out["3"] = {
        "statement": "Fine-tuning helps but does not close the gap to HAR",
        "detail": detail,
        "held": bool(detail)
        and all(d["helps"] and d["does_not_close_gap"] for d in detail.values()),
    }

    # 4. The foundation model's relative position improves as the horizon lengthens.
    gaps = {}
    for horizon in HORIZONS:
        tuned = skill("spy_logrv", FOUNDATION_MODEL, horizon)
        classical_best = headline[
            (headline.series == "spy_logrv")
            & (headline.horizon == horizon)
            & (headline.model_id.isin(CLASSICAL - {BASELINE}))
        ]
        if tuned is None or classical_best.empty:
            continue
        gaps[f"h{horizon}"] = float(classical_best["skill_wql"].max()) - tuned
    out["4"] = {
        "statement": "The gap to the best classical model narrows as the horizon lengthens",
        "gaps": gaps,
        "held": bool(gaps) and gaps.get("h21", 1.0) < gaps.get("h1", 0.0),
    }

    # 5. The foundation model shows better sample efficiency than N-BEATS.
    if sweep is None or sweep.empty:
        out["5"] = {
            "statement": "Better sample efficiency than N-BEATS",
            "held": None,
            "detail": "sweep not available",
        }
    else:
        detail = {}
        for model in (FOUNDATION_MODEL, "N-BEATS"):
            rows = sweep[
                (sweep.series == "spy_logrv")
                & (sweep.model_id == model)
                & (sweep.horizon == 1)
            ]
            if rows.empty:
                continue
            by_window = rows.set_index("training_window")["skill_wql"].to_dict()
            full = by_window.get("full")
            small = by_window.get("1y")
            detail[model] = {
                "skill_1y": small,
                "skill_full": full,
                "retained_at_1y": (
                    (small / full)
                    if full not in (None, 0) and small is not None
                    else None
                ),
            }
        a = detail.get(FOUNDATION_MODEL, {}).get("retained_at_1y")
        b = detail.get("N-BEATS", {}).get("retained_at_1y")
        out["5"] = {
            "statement": "Better sample efficiency than N-BEATS",
            "detail": detail,
            "held": bool(a is not None and b is not None and a > b),
        }
    return out


def confidence_sets(forecasts: dict[str, pd.DataFrame]) -> dict:
    """Model Confidence Set per series and horizon.

    Args:
        forecasts: Tidy forecasts keyed by series.

    Returns:
        Mapping of ``"<series>_h<horizon>"`` to the surviving model ids.

    Note:
        The honest framing when 137 origins is a small sample: which models cannot be
        told apart, rather than which one has the smallest number.
    """
    out = {}
    for series, block in forecasts.items():
        for horizon in HORIZONS:
            losses = per_origin_losses(block, horizon)
            out[f"{series}_h{horizon}"] = model_confidence_set(
                losses.to_numpy(), list(losses.columns), alpha=0.1
            )
    return out


def main() -> int:
    """Evaluate the pre-registration and print the verdict.

    Returns:
        Process exit code.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    setup_logging("WARNING")
    config = get_config()

    headline = pd.read_parquet(config.metrics_dir / "headline.parquet")
    forecasts = {
        path.stem.split("_arm")[0]: pd.read_parquet(path)
        for path in sorted(config.forecasts_dir.glob("*_armA_block_ys.parquet"))
    }
    sweep_path = config.metrics_dir / "sample_efficiency.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else None

    verdict = evaluate_losing_condition(headline, forecasts)
    predictions = evaluate_predictions(headline, forecasts, sweep)
    mcs = confidence_sets(forecasts)

    print("=" * 78)
    print("PRE-REGISTERED LOSING CONDITION (PREREGISTRATION.md section 3)")
    print("=" * 78)
    print("\n(a) beats the random walk on WQL skill at h=1:")
    for series, data in verdict["clause_a"].items():
        mark = "PASS" if data["beat_random_walk"] else "FAIL"
        print(f"      {series:<12} skill = {data['skill']:+.4f}   {mark}")
    print("\n(b) Diebold-Mariano vs the best classical model:")
    for key, data in verdict["clause_b"].items():
        mark = "WIN" if data["significant_win"] else "no"
        print(
            f"      {key:<16} vs {data['best_classical']:<14} "
            f"p = {data['p_value']:.4f}  favours_foundation={data['favours_foundation']}  {mark}"
        )
    print(
        f"\n  any significant win anywhere: {verdict['any_significant_win_over_best_classical']}"
    )
    print()
    print("=" * 78)
    print(
        f"  VERDICT: the fine-tuned foundation model {'LOST' if verdict['lost'] else 'WON'}"
    )
    print("=" * 78)

    print("\nREGISTERED PREDICTIONS (section 4)\n")
    for number, data in sorted(predictions.items()):
        held = data["held"]
        mark = {True: "HELD", False: "FAILED", None: "N/A"}[held]
        print(f"  {number}. [{mark:>6}] {data['statement']}")

    print("\nMODEL CONFIDENCE SET (alpha=0.10)\n")
    for key, models in mcs.items():
        print(f"  {key:<18} {len(models)} models: {', '.join(models)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
