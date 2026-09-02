"""Significance testing: Diebold-Mariano, the Model Confidence Set, and bootstrap CIs.

Approximately 137 non-overlapping folds is a small sample for any of this, and the study
says so rather than hiding it. Three guards against over-claiming:

- Diebold-Mariano carries the Harvey-Leybourne-Newbold small-sample correction.
- A Model Confidence Set is reported alongside pairwise tests, which answers "which models
  can we not distinguish" without requiring a correction per test.
- Headline claims are restricted to the comparisons pre-registered in
  ``PREREGISTRATION.md`` §3; every other test is descriptive.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of a Diebold-Mariano test.

    Attributes:
        statistic: The HLN-corrected test statistic.
        p_value: Two-sided p-value from a t distribution with ``n - 1`` degrees of freedom.
        n_observations: Number of loss differentials used.
        horizon: Forecast horizon the losses came from.
        mean_loss_differential: Mean of ``loss_a - loss_b``. Negative favours model A.
    """

    statistic: float
    p_value: float
    n_observations: int
    horizon: int
    mean_loss_differential: float

    @property
    def favours(self) -> str:
        """Which model the mean loss differential favours."""
        return "a" if self.mean_loss_differential < 0 else "b"


def newey_west_variance(values: np.ndarray, lag: int) -> float:
    """Newey-West HAC estimate of the long-run variance of a series mean.

    Args:
        values: The series, typically a loss differential.
        lag: Truncation lag. ``horizon - 1`` for an h-step forecast.

    Returns:
        The estimated long-run variance, clipped at zero from below.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    centred = values - values.mean()

    variance = float(np.dot(centred, centred) / n)
    for order in range(1, lag + 1):
        if order >= n:
            break
        autocovariance = float(np.dot(centred[order:], centred[:-order]) / n)
        weight = 1.0 - order / (lag + 1.0)
        variance += 2.0 * weight * autocovariance
    return max(variance, 0.0)


def diebold_mariano(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    horizon: int = 1,
) -> DieboldMarianoResult:
    """Test whether two forecast loss series differ significantly.

    Args:
        losses_a: Per-origin losses for model A.
        losses_b: Per-origin losses for model B.
        horizon: Forecast horizon, which sets the HAC truncation lag to ``horizon - 1``.

    Returns:
        The test result.

    Raises:
        ValueError: If the inputs differ in length, are shorter than two usable
            observations, or the loss differential has zero variance.

    Note:
        **Assumes non-overlapping forecast windows**, which holds only because
        ``stride == max horizon`` in ``backtest/splitter.py``. Changing the stride
        invalidates this test: overlapping windows leave the loss differential
        autocorrelated by construction and inflate significance.

        The Harvey-Leybourne-Newbold correction and the t reference distribution are
        applied because ~137 observations is small. This is stated in the results tables
        as well as here; no claim in the study rests on a single p-value.

        The HAC truncation lag is ``horizon - 1``, the standard Diebold-Mariano choice.
        That convention exists because overlapping h-step forecasts leave the loss
        differential autocorrelated up to lag ``h - 1``. This study's windows do **not**
        overlap, so the differential should already be close to serially uncorrelated and
        the lag is conservative rather than necessary. It is kept because it is the
        convention a reader will expect, and because being conservative about
        significance is the right direction to err in a study whose point is not
        over-claiming. Note the cost: at ``horizon=21`` a lag-20 HAC estimate on ~137
        observations is itself noisy.
    """
    losses_a = np.asarray(losses_a, dtype=float)
    losses_b = np.asarray(losses_b, dtype=float)
    if losses_a.shape != losses_b.shape:
        raise ValueError(f"Shape mismatch: {losses_a.shape} vs {losses_b.shape}")

    differential = losses_a - losses_b
    differential = differential[np.isfinite(differential)]
    n = len(differential)
    if n < 2:
        raise ValueError(f"Need at least 2 loss differentials, got {n}")

    lag = max(horizon - 1, 0)
    variance = newey_west_variance(differential, lag)
    if variance <= 0:
        raise ValueError(
            "Loss differential has zero long-run variance; the two models produced "
            "identical losses."
        )

    statistic = float(np.mean(differential) / np.sqrt(variance / n))

    # Harvey, Leybourne & Newbold (1997): the raw statistic is oversized in small samples.
    correction = np.sqrt((n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n)
    statistic *= float(correction)

    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    return DieboldMarianoResult(
        statistic=statistic,
        p_value=p_value,
        n_observations=n,
        horizon=horizon,
        mean_loss_differential=float(np.mean(differential)),
    )


def model_confidence_set(
    loss_matrix: np.ndarray,
    model_ids: list[str],
    alpha: float = 0.1,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Models that cannot be statistically distinguished from the best.

    Args:
        loss_matrix: Shape ``(n_observations, n_models)`` of per-origin losses.
        model_ids: Model identifiers, one per column.
        alpha: Confidence level for the set.
        n_bootstrap: Bootstrap replications.
        seed: Random seed.

    Returns:
        The surviving model identifiers, in their input order.

    Raises:
        ValueError: If the matrix and identifiers disagree in width.

    Note:
        The honest framing when the sample is small. "These four models are
        indistinguishable" is a more defensible statement than "model X won by 0.3%", and
        it does not require a multiple-comparison correction per pairwise test.
    """
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    if loss_matrix.shape[1] != len(model_ids):
        raise ValueError(
            f"{loss_matrix.shape[1]} loss columns but {len(model_ids)} model ids"
        )

    try:
        from arch.bootstrap import MCS

        mcs = MCS(loss_matrix, size=alpha, reps=n_bootstrap, seed=seed)
        mcs.compute()
        included = np.asarray(mcs.included, dtype=int).ravel()
        return [model_ids[position] for position in included]
    except Exception as error:  # noqa: BLE001 - fall back rather than lose the table
        logger.warning(
            "Model Confidence Set via arch failed (%s: %s); falling back to a pairwise "
            "screen against the best model.",
            type(error).__name__,
            error,
        )
        return _mcs_fallback(loss_matrix, model_ids, alpha)


def _mcs_fallback(
    loss_matrix: np.ndarray, model_ids: list[str], alpha: float
) -> list[str]:
    """Pairwise screen used when ``arch``'s MCS is unavailable.

    Args:
        loss_matrix: Per-origin losses, one column per model.
        model_ids: Model identifiers.
        alpha: Significance level.

    Returns:
        Models not significantly worse than the best by a Diebold-Mariano test.

    Note:
        This is **not** a Model Confidence Set — it has no familywise error control. It
        exists so a results table can still be produced, and any table built from it is
        labelled as a fallback rather than presented as an MCS.
    """
    mean_losses = loss_matrix.mean(axis=0)
    best = int(np.argmin(mean_losses))

    survivors = [model_ids[best]]
    for position, model_id in enumerate(model_ids):
        if position == best:
            continue
        try:
            result = diebold_mariano(loss_matrix[:, position], loss_matrix[:, best])
        except ValueError:
            survivors.append(model_id)
            continue
        if result.p_value >= alpha:
            survivors.append(model_id)
    return [model_id for model_id in model_ids if model_id in set(survivors)]


def bootstrap_skill_ci(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for a skill score.

    Args:
        model_losses: Per-origin losses for the model.
        baseline_losses: Per-origin losses for the baseline.
        confidence: Interval confidence level.
        n_bootstrap: Bootstrap replications.
        block_size: Block length, which preserves any residual serial dependence.
        seed: Random seed.

    Returns:
        A ``(lower, upper)`` pair for ``1 - mean(model) / mean(baseline)``.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    model_losses = np.asarray(model_losses, dtype=float)
    baseline_losses = np.asarray(baseline_losses, dtype=float)
    if model_losses.shape != baseline_losses.shape:
        raise ValueError("Model and baseline loss arrays must be the same length")
    n = len(model_losses)
    if n == 0:
        raise ValueError("No losses to bootstrap")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    skills = np.empty(n_bootstrap)

    for replication in range(n_bootstrap):
        starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
        positions = np.concatenate(
            [np.arange(start, min(start + block_size, n)) for start in starts]
        )[:n]
        baseline_mean = baseline_losses[positions].mean()
        skills[replication] = (
            np.nan
            if baseline_mean == 0
            else 1.0 - model_losses[positions].mean() / baseline_mean
        )

    tail = (1.0 - confidence) / 2.0
    finite = skills[np.isfinite(skills)]
    return (
        float(np.quantile(finite, tail)),
        float(np.quantile(finite, 1.0 - tail)),
    )
