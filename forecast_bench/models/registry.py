"""Where the model panel is defined.

Adding a model means one entry here plus one file. Nothing else changes — not the runner,
not the metrics, not the Space. That is the payoff for the ``Forecaster`` protocol, and it
is the thing to check if a change starts requiring edits elsewhere.

The panel is **per series**, because two of the models only make sense on one track:

- HAR and LogHAR are realized-volatility models and belong to the SPY log-RV panel.
- AR(1) is the standard macro benchmark and belongs to the rates panel.

Registering them everywhere would pad the comparison with models nobody would use on that
series, and would make the Model Confidence Set harder to read for no gain.
"""

import logging
from collections.abc import Callable

from forecast_bench.backtest.protocol import Forecaster
from forecast_bench.models.classical.ar1 import AR1
from forecast_bench.models.classical.arima import ARIMA
from forecast_bench.models.classical.har import HAR, LogHAR
from forecast_bench.models.classical.sarimax import SARIMAX
from forecast_bench.models.naive import RandomWalk, SeasonalNaive

logger = logging.getLogger(__name__)

#: The baseline every skill score is measured against.
BASELINE_MODEL_ID = "RandomWalk"

#: Models registered for every series.
_SHARED = {
    "RandomWalk": RandomWalk,
    "SeasonalNaive": SeasonalNaive,
    "ARIMA": ARIMA,
}

#: Models registered only for a particular target.
_SERIES_SPECIFIC: dict[str, dict[str, type]] = {
    "spy_logrv": {"HAR": HAR, "LogHAR": LogHAR},
    "dgs10": {"AR1": AR1},
}

#: Models registered only for the covariate-informed arm.
_ARM_B_ONLY = {"SARIMAX": SARIMAX}


def classical_panel(
    series: str,
    arm: str = "A",
    target_column: str | None = None,
) -> dict[str, Callable[[], Forecaster]]:
    """Build the naive and classical panel for one series and arm.

    Args:
        series: ``"spy_logrv"`` or ``"dgs10"``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        target_column: Column holding the target. Defaults to ``series``.

    Returns:
        Mapping of model id to a zero-argument builder. A builder is called afresh on every
        refit, so no fitted state can survive a fold boundary by accident.

    Raises:
        KeyError: If the series has no registered panel.
        ValueError: If the arm is not ``"A"`` or ``"B"``.
    """
    if series not in _SERIES_SPECIFIC:
        raise KeyError(
            f"No panel registered for series {series!r}; expected one of "
            f"{sorted(_SERIES_SPECIFIC)}"
        )
    if arm not in {"A", "B"}:
        raise ValueError(f"Unknown arm {arm!r}; expected 'A' or 'B'")

    column = target_column or series
    registered = {**_SHARED, **_SERIES_SPECIFIC[series]}
    if arm == "B":
        registered = {**registered, **_ARM_B_ONLY}

    return {
        model_id: _builder(model_class, column)
        for model_id, model_class in registered.items()
    }


def _builder(model_class: type, target_column: str) -> Callable[[], Forecaster]:
    """Return a zero-argument builder for a model class.

    Args:
        model_class: The class to construct.
        target_column: Column holding the target.

    Returns:
        A callable producing a fresh, unfitted instance.
    """
    return lambda: model_class(target_column=target_column)


def all_registered_model_classes() -> dict[str, type]:
    """Every model class in the registry, across all series and arms.

    Returns:
        Mapping of model id to class. Used by ``tests/test_models_protocol.py``, which
        checks the panel as a whole rather than a hand-maintained list that would drift.
    """
    combined: dict[str, type] = dict(_SHARED)
    for specific in _SERIES_SPECIFIC.values():
        combined.update(specific)
    combined.update(_ARM_B_ONLY)
    return combined
