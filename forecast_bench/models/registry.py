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
from forecast_bench.models.foundation.chronos2 import (
    Chronos2FineTuned,
    Chronos2ZeroShot,
)
from forecast_bench.models.foundation.chronos_bolt import (
    ChronosBoltFineTuned,
    ChronosBoltZeroShot,
)
from forecast_bench.models.naive import RandomWalk, SeasonalNaive
from forecast_bench.models.neural.deepar import DeepAR
from forecast_bench.models.neural.nbeats import NBEATS

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

#: Pretrained foundation models used without adaptation, registered for every series.
#:
#: Kept in their own group because their results carry a caveat the classical models do
#: not: zero-shot numbers on pre-October-2025 origins may be contaminated by pretraining
#: exposure. See DECISIONS.md D10-G4 and docs/limitations.md.
_FOUNDATION_ZERO_SHOT = {
    "Chronos2-ZeroShot": Chronos2ZeroShot,
    "ChronosBolt-ZeroShot": ChronosBoltZeroShot,
}


#: Fine-tuned foundation models, registered per series according to which adapters were
#: actually trained.
#:
#: Chronos-2 was fine-tuned on both targets; Chronos-Bolt only on the volatility track, per
#: DECISIONS.md D13, which makes Chronos-2 the core model and Bolt a secondary
#: generational comparison. Registering a model whose adapters do not exist would fail
#: mid-run at whichever block first went looking for one, so the panel states what is real.
_FINETUNED: dict[str, dict[str, type]] = {
    "spy_logrv": {
        "Chronos2-FineTuned": Chronos2FineTuned,
        "ChronosBolt-FineTuned": ChronosBoltFineTuned,
    },
    "dgs10": {
        "Chronos2-FineTuned": Chronos2FineTuned,
    },
}


#: From-scratch neural baselines, registered for every series.
#:
#: Kept in their own group because they are the only models that need GPU training, so a
#: local run can exclude them without touching the panel definition.
_NEURAL = {
    "N-BEATS": NBEATS,
    "DeepAR-LSTM": DeepAR,
}


def classical_panel(
    series: str,
    arm: str = "A",
    target_column: str | None = None,
    include_foundation: bool = False,
    include_neural: bool = False,
    include_finetuned: bool = False,
) -> dict[str, Callable[[], Forecaster]]:
    """Build the naive and classical panel for one series and arm.

    Args:
        series: ``"spy_logrv"`` or ``"dgs10"``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        target_column: Column holding the target. Defaults to ``series``.
        include_foundation: Add the zero-shot foundation models to the panel.
        include_neural: Add the from-scratch neural baselines, which need GPU training.
        include_finetuned: Add the LoRA-adapted foundation models. Requires that the
            adapters already exist on the Hub for every block of this series.

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
    if include_foundation:
        registered = {**registered, **_FOUNDATION_ZERO_SHOT}
    if include_neural:
        registered = {**registered, **_NEURAL}
    if include_finetuned:
        registered = {**registered, **_FINETUNED.get(series, {})}

    return {
        model_id: _builder(model_class, column, series, arm)
        for model_id, model_class in registered.items()
    }


def _builder(
    model_class: type, target_column: str, series: str, arm: str
) -> Callable[[], Forecaster]:
    """Return a zero-argument builder for a model class.

    Args:
        model_class: The class to construct.
        target_column: Column holding the target.
        series: Series name, passed to models that need it to resolve an adapter.
        arm: Experiment arm, passed on the same basis.

    Returns:
        A callable producing a fresh, unfitted instance.

    Note:
        Arguments are passed only where the constructor accepts them, so adding a model
        that needs more context does not force every other model to grow parameters it
        has no use for.
    """
    import inspect

    accepted = inspect.signature(model_class.__init__).parameters
    kwargs: dict[str, object] = {"target_column": target_column}
    if "series" in accepted:
        kwargs["series"] = series
    if "arm" in accepted:
        kwargs["arm"] = arm
    return lambda: model_class(**kwargs)


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
    combined.update(_FOUNDATION_ZERO_SHOT)
    combined.update(_NEURAL)
    for finetuned in _FINETUNED.values():
        combined.update(finetuned)
    return combined


def finetuned_model_ids(series: str | None = None) -> list[str]:
    """Fine-tuned model ids, for one series or across all of them.

    Args:
        series: Restrict to a series, or ``None`` for the union.

    Returns:
        Model ids, sorted.
    """
    if series is not None:
        return sorted(_FINETUNED.get(series, {}))
    return sorted({model_id for m in _FINETUNED.values() for model_id in m})


def neural_model_ids() -> list[str]:
    """Model ids that require GPU training.

    Returns:
        The neural baseline ids, sorted.
    """
    return sorted(_NEURAL)


def foundation_model_ids() -> list[str]:
    """Model ids whose results carry the pretraining-contamination caveat.

    Returns:
        The zero-shot foundation model ids, sorted.
    """
    return sorted(_FOUNDATION_ZERO_SHOT)
