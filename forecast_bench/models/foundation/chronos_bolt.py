"""Chronos-Bolt-small, the older-generation foundation model.

Included for two reasons beyond its own performance. It preserves the "same
``transformers`` + ``peft`` recipe as my Mistral and Qwen fine-tunes" narrative, since Bolt
takes the standard path where Chronos-2 does not. And it turns a two-way split into a
*generational* comparison — old foundation model, new foundation model, classical — which
is a more interesting result than either alone.

Zero-shot results on pre-October-2025 origins may be contaminated by pretraining exposure.
See ``docs/limitations.md``.
"""

import logging
import warnings

import numpy as np

from forecast_bench.models.foundation._pipeline import (
    ChronosFineTuned,
    ChronosZeroShot,
)

logger = logging.getLogger(__name__)

#: Whether the tail-clamping notice has been emitted in this process.
_WARNED_ABOUT_TAILS = False

#: Hugging Face checkpoint for the base model.
CHRONOS_BOLT_MODEL_ID = "amazon/chronos-bolt-small"

#: Levels Chronos-Bolt was trained to predict.
BOLT_TRAINED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ChronosBoltZeroShot(ChronosZeroShot):
    """Chronos-Bolt-small used out of the box.

    Attributes:
        model_id: ``"ChronosBolt-ZeroShot"``.
        hf_model_id: ``"amazon/chronos-bolt-small"``.

    Note:
        **This checkpoint cannot produce the study's tail quantiles.** Bolt was trained on
        levels 0.1 to 0.9 only. Asking for 0.025 and 0.975 returns its 0.1 and 0.9
        predictions unchanged, so Bolt's 95% interval is identical to its 80% interval by
        construction.

        This is a real limitation of the model, not of the harness, and it is left in place
        rather than papered over: extrapolating tails Bolt was never trained to produce
        would be inventing a capability in order to make a number look better.

        It does affect the primary metric. Weighted quantile loss averages over all eleven
        levels, and two of Bolt's eleven are duplicates of their neighbours, so Bolt is
        penalised at the tails relative to a model with genuine tail predictions. Every
        Bolt number in the study carries this caveat, and ``docs/limitations.md`` states it
        alongside the results rather than in a footnote.
    """

    model_id = "ChronosBolt-ZeroShot"
    hf_model_id = CHRONOS_BOLT_MODEL_ID

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Forecast the grid, reporting the tail clamping once rather than per fold.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path. The 0.025 and 0.975 entries are Bolt's 0.1 and 0.9
            predictions, because the checkpoint has no others to give.

        Note:
            The upstream library warns on every call. Across 137 folds that is 137
            identical lines, which is how a genuine warning gets missed, so it is emitted
            once per process with a message naming the consequence for the metric.
        """
        global _WARNED_ABOUT_TAILS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paths = super()._quantile_paths(horizon)

        if not _WARNED_ABOUT_TAILS:
            _WARNED_ABOUT_TAILS = True
            logger.warning(
                "%s was trained on quantiles %s, so the study grid's 0.025 and 0.975 "
                "levels are returned as its 0.1 and 0.9 predictions. Its 95%% interval "
                "therefore equals its 80%% interval, and weighted quantile loss penalises "
                "it at the tails relative to models with genuine tail predictions. "
                "Reported as a limitation rather than worked around.",
                self.model_id,
                BOLT_TRAINED_QUANTILES,
            )
        return paths


class ChronosBoltFineTuned(ChronosFineTuned):
    """Chronos-Bolt-small with a LoRA adapter fitted on this study's own data.

    Attributes:
        model_id: ``"ChronosBolt-FineTuned"``.
        finetune_kind: ``"bolt"``, the model axis of the revision tag.

    Note:
        The tail-quantile limitation of the base checkpoint applies here unchanged: LoRA
        adapts the attention projections, not the quantile head, so this model still
        cannot produce levels outside 0.1-0.9 and its 95% interval still equals its 80%.
    """

    model_id = "ChronosBolt-FineTuned"
    hf_model_id = CHRONOS_BOLT_MODEL_ID
    finetune_kind = "bolt"

    def _prepare_base_model(self, model):
        """Give the base model the embedding accessors peft requires before loading.

        Args:
            model: The freshly loaded ``ChronosBoltModelForForecasting``.

        Returns:
            The model with ``get_input_embeddings``/``set_input_embeddings`` bound.

        Note:
            Needed when *loading* an adapter as well as when creating one:
            ``PeftModel.from_pretrained`` prepares the model the same way
            ``get_peft_model`` does, and raises the same ``NotImplementedError`` without
            this.
        """
        from forecast_bench.models.foundation.finetune import (
            _with_input_embedding_accessors,
        )

        return _with_input_embedding_accessors(model)
