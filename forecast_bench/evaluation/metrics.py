"""The single source of truth for every metric in the study.

Import from here. Never reimplement a metric in a notebook, a script, or ``space/app.py``.
The moment two definitions of MASE exist, one of them is wrong and nobody will know which
— and the Space would then show numbers the README disagrees with.

Every function takes plain arrays and returns a float, so each one can be checked against a
value computed by hand. ``tests/test_metrics.py`` does exactly that.
"""

import logging

import numpy as np

from forecast_bench.config import QUANTILE_GRID

logger = logging.getLogger(__name__)

#: Seasonal period for the MASE denominator: one business week.
DEFAULT_SEASON = 5


def _clean_pair(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions where either input is non-finite.

    Args:
        actual: Realised values.
        predicted: Forecast values.

    Returns:
        The two arrays restricted to positions finite in both.

    Raises:
        ValueError: If the inputs differ in length or nothing survives.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")

    usable = np.isfinite(actual) & np.isfinite(predicted)
    if not usable.any():
        raise ValueError("No finite (actual, predicted) pairs to score.")
    return actual[usable], predicted[usable]


# --- Point accuracy --------------------------------------------------------------------


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error.

    Args:
        actual: Realised values.
        predicted: Point forecasts.

    Returns:
        ``mean(|actual - predicted|)``.
    """
    actual, predicted = _clean_pair(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error.

    Args:
        actual: Realised values.
        predicted: Point forecasts.

    Returns:
        ``sqrt(mean((actual - predicted)**2))``.
    """
    actual, predicted = _clean_pair(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def seasonal_naive_denominator(
    train: np.ndarray, season: int = DEFAULT_SEASON
) -> float:
    """Mean absolute seasonal-naive error over the training window.

    Args:
        train: Training-window values, in observation order.
        season: Seasonal lag. Five is a business week.

    Returns:
        ``mean(|y[t] - y[t - season]|)`` over the training window.

    Raises:
        ValueError: If the window is shorter than one season, or the denominator is zero.

    Note:
        **Computed on the training window only.** Computing it on the full series is the
        most common way MASE is reported wrongly: it makes the denominator depend on the
        test period, so a model is scored against a scale it partly determined. It is also
        recomputed per fold — a cached denominator is a fitted object that crossed a fold
        boundary, which ``tests/test_no_leakage.py`` check 4 looks for.
    """
    values = np.asarray(train, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= season:
        raise ValueError(
            f"Need more than {season} training observations for a seasonal-naive "
            f"denominator, got {len(values)}"
        )

    denominator = float(np.mean(np.abs(values[season:] - values[:-season])))
    if denominator == 0.0:
        raise ValueError(
            "Seasonal-naive denominator is zero; the training window is constant."
        )
    return denominator


def mase(actual: np.ndarray, predicted: np.ndarray, denominator: float) -> float:
    """Mean absolute scaled error.

    Args:
        actual: Realised values.
        predicted: Point forecasts.
        denominator: From :func:`seasonal_naive_denominator`, computed on the training
            window of the same fold.

    Returns:
        ``mae / denominator``. Below 1 means better than a seasonal naive on the training
        window's own scale.

    Raises:
        ValueError: If the denominator is not positive.
    """
    if not denominator > 0:
        raise ValueError(f"MASE denominator must be positive, got {denominator}")
    return mae(actual, predicted) / denominator


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Args:
        actual: Realised values.
        predicted: Point forecasts.

    Returns:
        ``mean(200 * |a - p| / (|a| + |p|))``.

    Note:
        Reported for completeness and comparability with other benchmarks, but it is not
        meaningful on SPY log realized variance: the target is negative throughout
        (roughly -14 to -5), so ``|a| + |p|`` is a scale with no interpretation as a
        percentage of anything. Read MASE and the skill scores instead. Positions where
        both values are zero contribute zero rather than a division by zero.
    """
    actual, predicted = _clean_pair(actual, predicted)
    scale = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(
        np.abs(actual - predicted), scale, out=np.zeros_like(scale), where=scale > 0
    )
    return float(200.0 * np.mean(ratio))


def directional_accuracy(
    actual: np.ndarray, predicted: np.ndarray, origin_value: float | np.ndarray
) -> float:
    """Fraction of forecasts that get the direction of the change from the origin right.

    Args:
        actual: Realised values.
        predicted: Point forecasts.
        origin_value: The target's value at the forecast origin, scalar or per-position.

    Returns:
        Fraction in ``[0, 1]``.

    Note:
        Measured on the **change from the origin**, not on the level. On a persistent
        series like ``DGS10``, directional accuracy on the level is trivially near 100%
        and says nothing: a model that predicts "still about 4%" is right about the level
        and has expressed no view at all.

        Positions where the actual change is exactly zero are excluded, since there is no
        direction to get right.
    """
    actual, predicted = _clean_pair(actual, predicted)
    origin = np.asarray(origin_value, dtype=float)

    actual_change = actual - origin
    predicted_change = predicted - origin

    moved = actual_change != 0
    if not moved.any():
        return float("nan")
    return float(
        np.mean(np.sign(actual_change[moved]) == np.sign(predicted_change[moved]))
    )


# --- Probabilistic ---------------------------------------------------------------------


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, level: float) -> float:
    """Pinball (quantile) loss at one quantile level.

    Args:
        actual: Realised values.
        predicted: Forecasts at quantile ``level``.
        level: Quantile level in ``(0, 1)``.

    Returns:
        ``mean(max(level * (a - p), (level - 1) * (a - p)))``. Under-forecasting is
        penalised more heavily at high levels and less at low ones, which is what makes
        the loss proper for that quantile.

    Raises:
        ValueError: If ``level`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"Quantile level must be in (0, 1), got {level}")

    actual, predicted = _clean_pair(actual, predicted)
    difference = actual - predicted
    return float(np.mean(np.maximum(level * difference, (level - 1.0) * difference)))


def weighted_quantile_loss(
    actual: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
    levels: list[float] = QUANTILE_GRID,
) -> float:
    """Weighted quantile loss across the grid — the study's primary metric.

    Args:
        actual: Realised values.
        quantile_forecasts: Mapping of level to a forecast array.
        levels: Levels to average over.

    Returns:
        ``mean over levels of (2 * sum(pinball) / sum(|actual|))``.

    Raises:
        KeyError: If a requested level is absent.
        ValueError: If the actuals sum to zero in absolute value.

    Note:
        Normalising by ``sum(|actual|)`` is the GluonTS convention and makes the number
        scale-free, so the two targets are on comparable footing. Because the study
        reports WQL as a **skill score against the random walk**, the normalisation
        cancels in the headline number — but it must still be applied identically to
        every model, which is why this lives here and nowhere else.
    """
    actual = np.asarray(actual, dtype=float)
    scale = float(np.sum(np.abs(actual[np.isfinite(actual)])))
    if scale == 0.0:
        raise ValueError("Cannot normalise WQL: the actuals sum to zero.")

    losses = []
    for level in levels:
        if level not in quantile_forecasts:
            raise KeyError(f"Missing quantile level {level}")
        predicted = np.asarray(quantile_forecasts[level], dtype=float)
        usable = np.isfinite(actual) & np.isfinite(predicted)
        difference = actual[usable] - predicted[usable]
        total = float(
            np.sum(np.maximum(level * difference, (level - 1.0) * difference))
        )
        losses.append(2.0 * total / scale)
    return float(np.mean(losses))


def interval_coverage_and_width(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    """Empirical coverage and mean width of a prediction interval, together.

    Args:
        actual: Realised values.
        lower: Lower interval bound.
        upper: Upper interval bound.

    Returns:
        A ``(coverage, width)`` pair. Coverage is the fraction of actuals inside the
        interval; width is the mean of ``upper - lower``.

    Note:
        Returned **as a pair, always**, and reported as a pair everywhere. A model can buy
        coverage with uselessly wide intervals, and showing one number without the other
        hides exactly that. Splitting these into two functions would make it easy to
        report only the flattering one, so the API does not offer the choice.
    """
    lower_clean, upper_clean = _clean_pair(lower, upper)
    actual_arr = np.asarray(actual, dtype=float)
    usable = (
        np.isfinite(actual_arr)
        & np.isfinite(np.asarray(lower, dtype=float))
        & np.isfinite(np.asarray(upper, dtype=float))
    )
    actual_used = actual_arr[usable]

    coverage = float(
        np.mean((actual_used >= lower_clean) & (actual_used <= upper_clean))
    )
    width = float(np.mean(upper_clean - lower_clean))
    return coverage, width


# --- Relative --------------------------------------------------------------------------


def skill_score(model_metric: float, baseline_metric: float) -> float:
    """Improvement over a baseline, as a fraction.

    Args:
        model_metric: The model's value of a lower-is-better metric.
        baseline_metric: The baseline's value of the same metric.

    Returns:
        ``1 - model / baseline``. Positive means better than the baseline; zero means
        indistinguishable; negative means worse.

    Raises:
        ValueError: If the baseline metric is zero.

    Note:
        Every headline number in this study is a skill score against the random walk.
        Raw MAE on log realized variance is uninterpretable to a reader; "3.4% better than
        a random walk" is not.
    """
    if baseline_metric == 0:
        raise ValueError("Baseline metric is zero; skill score is undefined.")
    return float(1.0 - model_metric / baseline_metric)
