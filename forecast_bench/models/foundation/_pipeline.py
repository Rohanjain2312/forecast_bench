"""Shared machinery for the pretrained Chronos pipelines.

Private to :mod:`forecast_bench.models.foundation`. Two jobs:

1. **Cache loaded pipelines per process.** Loading Chronos-2 takes roughly 17 seconds. The
   runner constructs a fresh model object on every parameter refit, so without a cache the
   weights would be reloaded dozens of times per run for no benefit — the weights are
   fixed, and nothing about them is fold-specific.
2. **Normalise the pipelines' output shapes** so the rest of the study cannot tell a
   Chronos-2 apart from a Chronos-Bolt.
"""

import logging
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from forecast_bench.config import CONTEXT_LENGTH, QUANTILE_GRID
from forecast_bench.models.base import BaseForecaster

logger = logging.getLogger(__name__)

#: Loaded pipelines, keyed by (model id, device). Weights are fixed and fold-independent.
_PIPELINE_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LOCK = threading.Lock()


def load_pipeline(model_id: str, device: str = "cpu"):
    """Load a Chronos pipeline, reusing an already-loaded one when possible.

    Args:
        model_id: Hugging Face model id, or a local path.
        device: Torch device string.

    Returns:
        The loaded pipeline.

    Note:
        Chronos-2 is not a vanilla ``transformers`` T5, so it is loaded through
        ``BaseChronosPipeline``, which dispatches to ``Chronos2Pipeline`` or
        ``ChronosBoltPipeline`` as appropriate. Loading it with
        ``AutoModelForSeq2SeqLM`` does not work.
    """
    key = (model_id, device)
    with _CACHE_LOCK:
        if key not in _PIPELINE_CACHE:
            from chronos import BaseChronosPipeline

            logger.info("Loading %s onto %s (once per process)", model_id, device)
            _PIPELINE_CACHE[key] = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device
            )
        return _PIPELINE_CACHE[key]


def clear_pipeline_cache() -> None:
    """Drop every cached pipeline. Used by tests and by long-running notebooks."""
    with _CACHE_LOCK:
        _PIPELINE_CACHE.clear()


class ChronosZeroShot(BaseForecaster):
    """A pretrained Chronos pipeline used without any adaptation to our data.

    Zero-shot means there are **no parameters estimated from this study's data at all**.
    ``_estimate_parameters`` therefore loads weights and nothing else, and the refit
    cadence has no effect on this model beyond when that load happens. All of its
    fold-specific behaviour lives in the context window, which refreshes every fold.

    Zero-shot results on pre-October-2025 origins may be contaminated by pretraining
    exposure. SPY and Treasury yields are among the most widely redistributed time series
    in existence, and the pretraining corpora are not inspectable, so a "zero-shot"
    forecast of 2019 volatility may not be out-of-sample at all. See
    ``docs/limitations.md`` and ``DECISIONS.md`` D10-G4. Never present a pre-2025 zero-shot
    number as a clean out-of-sample result.

    Attributes:
        model_id: Results-table key.
        hf_model_id: Hugging Face model id the weights come from.
        device: Torch device.
        context_length: Observations of history fed to the model.
        trained_quantiles: Levels the checkpoint was trained on, or ``None`` if unknown.
    """

    model_id = "ChronosZeroShot"
    hf_model_id = "amazon/chronos-2"

    def __init__(
        self,
        target_column: str | None = None,
        hf_model_id: str | None = None,
        device: str = "cpu",
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            hf_model_id: Override the checkpoint. Defaults to the class's.
            device: Torch device string.
            context_length: Observations of history to condition on.
        """
        super().__init__(target_column=target_column)
        if hf_model_id is not None:
            self.hf_model_id = hf_model_id
        self.device = device
        self.context_length = context_length
        self._pipeline = None
        self._context: np.ndarray | None = None

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Load the pretrained weights. Nothing is learned from ``train``.

        Args:
            train: Training frame for this fold. Deliberately unused.
            series: The target column. Deliberately unused.
            origin: The fold's origin. Deliberately unused.
        """
        self._pipeline = load_pipeline(self.hf_model_id, self.device)

    def _update_state(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Refresh the context window to the most recent observations at the origin.

        Args:
            train: Training frame for this fold.
            series: The target column with NaNs dropped.
            origin: The fold's origin.
        """
        self._context = series.to_numpy(dtype=float)[-self.context_length :]

    @property
    def trained_quantiles(self) -> list[float] | None:
        """Quantile levels the loaded checkpoint was trained on, if it reports them."""
        levels = getattr(self._pipeline, "quantiles", None)
        return list(levels) if levels is not None else None

    def _quantile_paths(self, horizon: int) -> dict[float, np.ndarray]:
        """Forecast the study's quantile grid from the current context window.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Mapping of level to a path.

        Raises:
            RuntimeError: If the pipeline returns an unexpected shape.
        """
        context = torch.tensor(self._context, dtype=torch.float32)
        quantiles, _mean = self._pipeline.predict_quantiles(
            [context],
            prediction_length=horizon,
            quantile_levels=list(QUANTILE_GRID),
        )

        values = np.asarray(
            (
                quantiles[0].detach().cpu().numpy()
                if hasattr(quantiles[0], "detach")
                else quantiles[0]
            ),
            dtype=float,
        )
        # Chronos-2 returns (1, horizon, n_levels) for its multivariate path; Bolt returns
        # (horizon, n_levels). Collapse the leading axis so the runner sees one shape.
        while values.ndim > 2:
            values = values[0]

        if values.shape != (horizon, len(QUANTILE_GRID)):
            raise RuntimeError(
                f"{self.model_id}: expected ({horizon}, {len(QUANTILE_GRID)}) quantiles, "
                f"got {values.shape}"
            )

        return {
            level: values[:, position] for position, level in enumerate(QUANTILE_GRID)
        }


#: Fine-tuned pipelines, keyed by (base id, repo, revision, device).
#:
#: Bounded, unlike the zero-shot cache. Each entry holds a full model, and a run touches
#: one adapter per annual block; keeping them all would be gigabytes for no benefit, since
#: blocks are visited in order and never revisited.
_FINETUNED_CACHE: OrderedDict[tuple[str, str, str, str], object] = OrderedDict()
_FINETUNED_CACHE_SIZE = 2


def load_finetuned_pipeline(
    base_model_id: str,
    repo_id: str,
    revision: str,
    device: str = "cpu",
    prepare=None,
):
    """Load a base pipeline and apply a LoRA adapter from the Hub.

    Args:
        base_model_id: Hugging Face id of the base checkpoint.
        repo_id: Repo holding the adapters.
        revision: Revision tag identifying the adapter.
        device: Torch device string.
        prepare: Optional callable applied to the base model before the adapter, for
            checkpoints that need patching first.

    Returns:
        A pipeline whose model carries the adapter.

    Raises:
        FileNotFoundError: If no adapter exists at that revision, with the tag named.

    Note:
        A **fresh** base pipeline is loaded every time rather than reusing the one from
        :func:`load_pipeline`. ``PeftModel.from_pretrained`` wraps the model object it is
        given, so applying an adapter to the shared base would silently turn every
        zero-shot model in the process into a fine-tuned one — the same weights object,
        now wrapped. That would corrupt the study's central comparison without raising
        anything.
    """
    key = (base_model_id, repo_id, revision, device)
    if key in _FINETUNED_CACHE:
        _FINETUNED_CACHE.move_to_end(key)
        return _FINETUNED_CACHE[key]

    from chronos import BaseChronosPipeline
    from peft import PeftModel

    logger.info("Loading %s with adapter %s", base_model_id, revision)
    pipeline = BaseChronosPipeline.from_pretrained(base_model_id, device_map=device)

    model = pipeline.model
    if prepare is not None:
        model = prepare(model)

    try:
        pipeline.model = PeftModel.from_pretrained(model, repo_id, revision=revision)
    except Exception as error:  # noqa: BLE001 - re-raised with the tag that is missing
        raise FileNotFoundError(
            f"No LoRA adapter at {repo_id}@{revision} ({type(error).__name__}: {error}). "
            "Fine-tune that block first: notebooks/04_colab_finetune_chronos.ipynb."
        ) from error

    _FINETUNED_CACHE[key] = pipeline
    while len(_FINETUNED_CACHE) > _FINETUNED_CACHE_SIZE:
        evicted, _ = _FINETUNED_CACHE.popitem(last=False)
        logger.debug("Evicted fine-tuned pipeline %s", evicted[2])
    return pipeline


class ChronosFineTuned(ChronosZeroShot):
    """A Chronos checkpoint adapted to this study's data with LoRA.

    One adapter per annual block, loaded by revision tag. The block is taken from the
    fold's origin, so under the matched cadence — which refits at each block boundary —
    the model loads exactly the adapter that was fine-tuned on data ending at that
    boundary, and never one trained on data the fold cannot see.

    Unlike the zero-shot models, these carry **no pretraining-contamination caveat for the
    adaptation itself**: the adapter was fitted on our own data with our own cutoffs. The
    underlying base model's pretraining exposure is unchanged, so the *gap* between
    fine-tuned and zero-shot remains the interpretable quantity. See ``DECISIONS.md``
    D10-G4.

    Attributes:
        model_id: Results-table key.
        series: Target series the adapter was fine-tuned on.
        arm: Experiment arm the adapter belongs to.
        training_window: Sample-efficiency slice, ``"full"`` for the headline run.
        finetune_kind: ``"chronos2"`` or ``"bolt"``, the model axis of the revision tag.
    """

    model_id = "ChronosFineTuned"
    finetune_kind = "chronos2"

    def __init__(
        self,
        target_column: str | None = None,
        series: str | None = None,
        arm: str = "A",
        training_window: str = "full",
        repo_id: str | None = None,
        hf_model_id: str | None = None,
        device: str = "cpu",
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        """Initialise the model.

        Args:
            target_column: Column holding the target, or ``None`` for the first column.
            series: Series the adapter was fine-tuned on. Defaults to ``target_column``.
            arm: Experiment arm.
            training_window: Sample-efficiency slice to load.
            repo_id: Repo holding the adapters. Defaults to the configured model repo.
            hf_model_id: Base checkpoint. Defaults to the class's.
            device: Torch device.
            context_length: Observations of history to condition on.
        """
        super().__init__(
            target_column=target_column,
            hf_model_id=hf_model_id,
            device=device,
            context_length=context_length,
        )
        self.series = series or target_column
        self.arm = arm
        self.training_window = training_window
        self.repo_id = repo_id
        self.loaded_revision: str | None = None

    def _prepare_base_model(self, model):
        """Hook for checkpoints needing patching before an adapter is applied.

        Args:
            model: The freshly loaded base model.

        Returns:
            The model, unchanged by default.
        """
        return model

    def _estimate_parameters(
        self, train: pd.DataFrame, series: pd.Series, origin: pd.Timestamp
    ) -> None:
        """Load the adapter fine-tuned on the block this fold belongs to.

        Args:
            train: Training frame for this fold. Not read; the adapter is already fitted.
            series: The target column. Not read, for the same reason.
            origin: The fold's origin, whose calendar year selects the adapter.

        Raises:
            ValueError: If no series name is available to build a revision tag from.
        """
        from forecast_bench.config import get_config
        from forecast_bench.models.foundation.hub import revision_tag

        if not self.series:
            raise ValueError(
                f"{self.model_id}: needs a series name to resolve its adapter revision."
            )

        revision = revision_tag(
            self.series,
            self.arm,
            int(origin.year),
            self.training_window,
            model=self.finetune_kind,
        )
        repo = self.repo_id or get_config().hf_model_repo

        self._pipeline = load_finetuned_pipeline(
            self.hf_model_id,
            repo,
            revision,
            device=self.device,
            prepare=self._prepare_base_model,
        )
        self.loaded_revision = revision
