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
