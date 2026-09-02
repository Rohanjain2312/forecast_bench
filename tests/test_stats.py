"""Tests for the significance machinery, against known-answer reference cases."""

import numpy as np
import pytest

from forecast_bench.evaluation.stats import (
    bootstrap_skill_ci,
    diebold_mariano,
    model_confidence_set,
    newey_west_variance,
)


def test_newey_west_with_zero_lag_is_the_sample_variance() -> None:
    """At lag 0 the HAC estimator reduces to the plain (biased) sample variance."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert newey_west_variance(values, lag=0) == pytest.approx(values.var())


def test_newey_west_widens_for_positively_autocorrelated_series() -> None:
    """Persistent series have a larger long-run variance than their sample variance."""
    rng = np.random.default_rng(0)
    values = np.empty(500)
    values[0] = 0.0
    for position in range(1, 500):
        values[position] = 0.8 * values[position - 1] + rng.standard_normal()

    assert newey_west_variance(values, lag=20) > newey_west_variance(values, lag=0)


def test_diebold_mariano_known_answer_case() -> None:
    """A hand-checkable case: constant loss differential of -1 with unit noise.

    Model A beats model B by 1.0 on average with small, serially uncorrelated noise, so
    the statistic is strongly negative and the p-value tiny.
    """
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(100) * 0.1
    losses_b = np.ones(100) * 2.0
    losses_a = losses_b - 1.0 + noise

    result = diebold_mariano(losses_a, losses_b, horizon=1)

    assert result.mean_loss_differential == pytest.approx(-1.0, abs=0.05)
    assert result.statistic < -50
    assert result.p_value < 1e-6
    assert result.favours == "a"
    assert result.n_observations == 100


def test_diebold_mariano_finds_no_difference_between_equivalent_models() -> None:
    """Two models drawing losses from the same distribution are not distinguishable."""
    rng = np.random.default_rng(7)
    losses_a = rng.standard_normal(200) + 5.0
    losses_b = rng.standard_normal(200) + 5.0

    result = diebold_mariano(losses_a, losses_b, horizon=1)

    assert result.p_value > 0.05


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_hln_correction_shrinks_the_statistic(horizon) -> None:
    """The HLN factor is below 1, so it always shrinks the raw statistic.

    Isolated from the HAC lag on purpose. Comparing the *returned* statistic across
    horizons would not test this, because the HAC truncation lag moves with the horizon
    too, and a long lag on a short sample can shrink the variance estimate enough to
    inflate the statistic more than the correction deflates it.
    """
    rng = np.random.default_rng(3)
    losses_a = rng.standard_normal(60) + 1.0
    losses_b = rng.standard_normal(60) + 1.4

    result = diebold_mariano(losses_a, losses_b, horizon=horizon)

    # Reconstruct the uncorrected statistic using the same HAC variance.
    differential = losses_a - losses_b
    n = len(differential)
    variance = newey_west_variance(differential, lag=max(horizon - 1, 0))
    raw = differential.mean() / np.sqrt(variance / n)

    assert abs(result.statistic) < abs(raw)


def test_hln_correction_is_stronger_at_longer_horizons() -> None:
    """The correction factor itself decreases with the horizon, at fixed sample size."""
    rng = np.random.default_rng(3)
    losses_a = rng.standard_normal(60) + 1.0
    losses_b = rng.standard_normal(60) + 1.4
    differential = losses_a - losses_b
    n = len(differential)

    def ratio(horizon: int) -> float:
        result = diebold_mariano(losses_a, losses_b, horizon=horizon)
        variance = newey_west_variance(differential, lag=max(horizon - 1, 0))
        raw = differential.mean() / np.sqrt(variance / n)
        return abs(result.statistic / raw)

    assert ratio(21) < ratio(5) < ratio(1) < 1.0


def test_diebold_mariano_rejects_identical_losses() -> None:
    """Identical loss series have no variance to test against."""
    losses = np.ones(50)
    with pytest.raises(ValueError, match="zero long-run variance"):
        diebold_mariano(losses, losses, horizon=1)


def test_diebold_mariano_rejects_a_shape_mismatch() -> None:
    """Misaligned loss series are an error, not a truncated comparison."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        diebold_mariano(np.ones(10), np.ones(9))


def test_diebold_mariano_rejects_too_few_observations() -> None:
    """One fold is not a sample."""
    with pytest.raises(ValueError, match="at least 2"):
        diebold_mariano(np.array([1.0]), np.array([2.0]))


def test_model_confidence_set_keeps_the_clearly_best_model() -> None:
    """A model that dominates decisively survives; a hopeless one does not."""
    rng = np.random.default_rng(11)
    n = 300
    losses = np.column_stack(
        [
            rng.standard_normal(n) * 0.1 + 1.0,  # good
            rng.standard_normal(n) * 0.1 + 1.05,  # nearly as good
            rng.standard_normal(n) * 0.1 + 5.0,  # hopeless
        ]
    )

    survivors = model_confidence_set(losses, ["good", "close", "hopeless"], alpha=0.1)

    assert "good" in survivors
    assert "hopeless" not in survivors


def test_model_confidence_set_keeps_indistinguishable_models_together() -> None:
    """Models drawn from the same distribution cannot be separated, and both survive."""
    rng = np.random.default_rng(5)
    n = 200
    losses = np.column_stack(
        [rng.standard_normal(n) + 3.0, rng.standard_normal(n) + 3.0]
    )

    survivors = model_confidence_set(losses, ["a", "b"], alpha=0.1)

    assert len(survivors) == 2


def test_model_confidence_set_rejects_mismatched_labels() -> None:
    """A label list that does not match the matrix width is a programming error."""
    with pytest.raises(ValueError, match="loss columns"):
        model_confidence_set(np.ones((10, 3)), ["a", "b"])


def test_bootstrap_skill_ci_brackets_the_point_estimate() -> None:
    """The interval contains the skill score computed on the full sample."""
    rng = np.random.default_rng(13)
    baseline = rng.standard_normal(200) * 0.1 + 2.0
    model = rng.standard_normal(200) * 0.1 + 1.6

    point = 1.0 - model.mean() / baseline.mean()
    lower, upper = bootstrap_skill_ci(model, baseline, n_bootstrap=500)

    assert lower < point < upper
    assert lower < upper


def test_bootstrap_skill_ci_of_an_identical_model_straddles_zero() -> None:
    """A model no better than the baseline has an interval containing zero skill."""
    rng = np.random.default_rng(17)
    losses = rng.standard_normal(200) + 4.0

    lower, upper = bootstrap_skill_ci(losses, losses.copy(), n_bootstrap=500)

    assert lower <= 0.0 <= upper


def test_bootstrap_skill_ci_rejects_mismatched_lengths() -> None:
    """Misaligned loss series cannot be paired per origin."""
    with pytest.raises(ValueError, match="same length"):
        bootstrap_skill_ci(np.ones(10), np.ones(9))
