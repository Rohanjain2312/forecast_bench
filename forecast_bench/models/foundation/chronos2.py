"""Chronos-2, the study's core foundation model.

Released October 2025: a 120M-parameter encoder-only model with native univariate,
multivariate and covariate-informed support, quantile outputs, and working CPU inference.
It is loaded through ``Chronos2Pipeline`` (dispatched by ``BaseChronosPipeline``), **not**
``AutoModelForSeq2SeqLM`` — Chronos-2 is not a vanilla ``transformers`` T5.

Zero-shot results on pre-October-2025 origins may be contaminated by pretraining exposure.
See ``docs/limitations.md``.
"""

from forecast_bench.models.foundation._pipeline import (
    ChronosFineTuned,
    ChronosZeroShot,
)

#: Hugging Face checkpoint for the base model.
CHRONOS2_MODEL_ID = "amazon/chronos-2"


class Chronos2ZeroShot(ChronosZeroShot):
    """Chronos-2 used out of the box, with no adaptation to this study's data.

    The checkpoint reports quantile levels from 0.01 to 0.99, so the study's grid — which
    reaches 0.025 and 0.975 — is inside its trained range and every level is a genuine
    prediction rather than a clamped one.

    Attributes:
        model_id: ``"Chronos2-ZeroShot"``.
        hf_model_id: ``"amazon/chronos-2"``.
    """

    model_id = "Chronos2-ZeroShot"
    hf_model_id = CHRONOS2_MODEL_ID


class Chronos2FineTuned(ChronosFineTuned):
    """Chronos-2 with a LoRA adapter fitted on this study's own data.

    The study's headline learned model. Its comparison against
    :class:`Chronos2ZeroShot` is the one quantity that pretraining contamination cannot
    confound, because both share the same base weights and differ only in the adaptation
    fitted on our data with our cutoffs.

    Attributes:
        model_id: ``"Chronos2-FineTuned"``.
        finetune_kind: ``"chronos2"``, the model axis of the revision tag.
    """

    model_id = "Chronos2-FineTuned"
    hf_model_id = CHRONOS2_MODEL_ID
    finetune_kind = "chronos2"
