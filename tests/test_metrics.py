"""Every metric checked against a value computed by hand on a tiny fixture.

Hand-computed rather than golden-file: a golden file records what the code did, which is
worthless if the code was wrong on the day it was written.
"""

import numpy as np
import pytest

from forecast_bench.evaluation import metrics as m

# actual    = [1, 2, 3, 4]
# predicted = [1, 3, 5, 3]
# errors    = [0, -1, -2, 1]  ->  |e| = [0, 1, 2, 1]
ACTUAL = np.array([1.0, 2.0, 3.0, 4.0])
PREDICTED = np.array([1.0, 3.0, 5.0, 3.0])


def test_mae_matches_hand_computation() -> None:
    """mean(|0, 1, 2, 1|) = 4 / 4 = 1.0"""
    assert m.mae(ACTUAL, PREDICTED) == pytest.approx(1.0)


def test_rmse_matches_hand_computation() -> None:
    """sqrt(mean(0, 1, 4, 1)) = sqrt(6 / 4) = sqrt(1.5)"""
    assert m.rmse(ACTUAL, PREDICTED) == pytest.approx(np.sqrt(1.5))


def test_smape_matches_hand_computation() -> None:
    """200 * mean(0/2, 1/5, 2/8, 1/7) = 200 * 0.5964285714.../4"""
    expected = 200.0 * np.mean([0.0 / 2.0, 1.0 / 5.0, 2.0 / 8.0, 1.0 / 7.0])
    assert m.smape(ACTUAL, PREDICTED) == pytest.approx(expected)


def test_seasonal_naive_denominator_matches_hand_computation() -> None:
    """With season=1 on [1, 2, 4, 7]: mean(|1|, |2|, |3|) = 2.0"""
    assert m.seasonal_naive_denominator(
        np.array([1.0, 2.0, 4.0, 7.0]), season=1
    ) == pytest.approx(2.0)


def test_mase_is_mae_over_the_denominator() -> None:
    """MASE = 1.0 / 2.0 = 0.5"""
    assert m.mase(ACTUAL, PREDICTED, denominator=2.0) == pytest.approx(0.5)


def test_mase_denominator_uses_the_training_window_only() -> None:
    """A denominator from the training window ignores the test period entirely.

    The series is flat in training and wildly volatile afterwards. A denominator computed
    over the whole thing would be far larger, which would flatter every MASE in the study.
    """
    train = np.array([1.0, 1.1, 1.0, 1.1, 1.0, 1.1])
    full = np.concatenate([train, np.array([50.0, -50.0, 50.0, -50.0])])

    from_train = m.seasonal_naive_denominator(train, season=1)
    from_full = m.seasonal_naive_denominator(full, season=1)

    assert from_train == pytest.approx(0.1)
    assert from_full > 10 * from_train


def test_mase_rejects_a_non_positive_denominator() -> None:
    """A zero denominator would make MASE infinite rather than undefined-looking."""
    with pytest.raises(ValueError, match="must be positive"):
        m.mase(ACTUAL, PREDICTED, denominator=0.0)


def test_directional_accuracy_is_measured_on_the_change_from_origin() -> None:
    """Origin 2.0: actual changes [-1, 0, +1, +2], predicted [-1, +1, +3, +1].

    The zero-change position is excluded, leaving three comparisons of which all three
    agree in sign: down/down, up/up, up/up.
    """
    assert m.directional_accuracy(ACTUAL, PREDICTED, origin_value=2.0) == pytest.approx(
        1.0
    )


def test_directional_accuracy_on_a_persistent_series_is_not_trivially_high() -> None:
    """Measuring on the change catches a model that only predicts the level well."""
    actual = np.array([4.0, 4.1, 4.2, 4.05])
    # Predicts the level almost perfectly but gets every direction backwards.
    predicted = np.array([3.99, 3.95, 4.02, 4.12])
    assert m.directional_accuracy(actual, predicted, origin_value=4.05) < 0.5


def test_pinball_loss_matches_hand_computation() -> None:
    """At level 0.9, differences [0, -1, -2, 1] give [0, 0.1, 0.2, 0.9], mean 0.3."""
    assert m.pinball_loss(ACTUAL, PREDICTED, level=0.9) == pytest.approx(0.3)


def test_pinball_loss_at_the_median_is_half_the_mae() -> None:
    """The 0.5 pinball loss is exactly MAE / 2, which is a useful cross-check."""
    assert m.pinball_loss(ACTUAL, PREDICTED, level=0.5) == pytest.approx(
        m.mae(ACTUAL, PREDICTED) / 2.0
    )


def test_pinball_loss_penalises_asymmetrically() -> None:
    """Over-forecasting costs more at a low quantile than under-forecasting does."""
    actual = np.array([0.0])
    too_high = m.pinball_loss(actual, np.array([1.0]), level=0.1)
    too_low = m.pinball_loss(actual, np.array([-1.0]), level=0.1)
    assert too_high > too_low


def test_pinball_loss_rejects_a_level_outside_the_unit_interval() -> None:
    """A level of 0 or 1 is not a quantile this loss is defined for."""
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        m.pinball_loss(ACTUAL, PREDICTED, level=1.0)


def test_weighted_quantile_loss_matches_hand_computation() -> None:
    """Two levels on a two-point series, worked through by hand.

    actual = [2, 4], sum|actual| = 6.
    level 0.1 forecast [1, 3]: differences [1, 1], pinball each 0.1 -> sum 0.2
        contribution 2 * 0.2 / 6 = 0.0666...
    level 0.9 forecast [3, 5]: differences [-1, -1], pinball each 0.1 -> sum 0.2
        contribution 2 * 0.2 / 6 = 0.0666...
    mean = 0.0666...
    """
    actual = np.array([2.0, 4.0])
    forecasts = {0.1: np.array([1.0, 3.0]), 0.9: np.array([3.0, 5.0])}
    result = m.weighted_quantile_loss(actual, forecasts, levels=[0.1, 0.9])
    assert result == pytest.approx(2.0 * 0.2 / 6.0)


def test_weighted_quantile_loss_is_zero_for_a_perfect_forecast() -> None:
    """Every quantile on the actual value costs nothing."""
    actual = np.array([2.0, 4.0])
    forecasts = {level: actual.copy() for level in [0.1, 0.5, 0.9]}
    assert m.weighted_quantile_loss(
        actual, forecasts, levels=[0.1, 0.5, 0.9]
    ) == pytest.approx(0.0)


def test_weighted_quantile_loss_requires_every_requested_level() -> None:
    """A missing level is an error, not a silently narrower average."""
    with pytest.raises(KeyError, match="Missing quantile level"):
        m.weighted_quantile_loss(
            np.array([1.0]), {0.5: np.array([1.0])}, levels=[0.5, 0.9]
        )


def test_interval_coverage_and_width_match_hand_computation() -> None:
    """Three of four actuals fall inside; widths are [2, 2, 2, 2]."""
    lower = np.array([0.0, 1.0, 2.0, 10.0])
    upper = np.array([2.0, 3.0, 4.0, 12.0])
    coverage, width = m.interval_coverage_and_width(ACTUAL, lower, upper)
    assert coverage == pytest.approx(0.75)
    assert width == pytest.approx(2.0)


def test_coverage_and_width_are_returned_together() -> None:
    """The API does not offer a way to report coverage without width.

    A model can buy coverage with uselessly wide intervals; reporting one without the
    other hides exactly that, so the function returns a pair by construction.
    """
    wide = m.interval_coverage_and_width(
        ACTUAL, np.full(4, -1000.0), np.full(4, 1000.0)
    )
    assert wide[0] == pytest.approx(1.0)
    assert wide[1] == pytest.approx(2000.0)
    assert not hasattr(m, "interval_coverage")
    assert not hasattr(m, "interval_width")


def test_skill_score_matches_hand_computation() -> None:
    """1 - 0.8 / 1.0 = 0.2, i.e. 20% better than the baseline."""
    assert m.skill_score(0.8, 1.0) == pytest.approx(0.2)
    assert m.skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert m.skill_score(1.5, 1.0) == pytest.approx(-0.5)


def test_skill_score_rejects_a_zero_baseline() -> None:
    """A perfect baseline makes skill undefined rather than infinite."""
    with pytest.raises(ValueError, match="undefined"):
        m.skill_score(0.5, 0.0)


def test_metrics_ignore_non_finite_pairs() -> None:
    """A NaN actual drops that position rather than poisoning the whole metric."""
    actual = np.array([1.0, np.nan, 3.0])
    predicted = np.array([1.0, 5.0, 5.0])
    assert m.mae(actual, predicted) == pytest.approx(1.0)


def test_metrics_reject_a_shape_mismatch() -> None:
    """Misaligned arrays are an error, not a silently truncated comparison."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        m.mae(np.array([1.0, 2.0]), np.array([1.0]))
