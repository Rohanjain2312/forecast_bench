"""LoRA fine-tuning for Chronos-2 and Chronos-Bolt, with resume-from-Hub.

This module holds **all** the fine-tuning logic. The Colab notebooks import from here and
call one function; they contain no training loop and no loop over blocks. That is what
stops the notebook and the repository from disagreeing, which is the failure mode that
kills most benchmark projects.

Two paths, deliberately:

- **Chronos-2** uses the official ``Chronos2Pipeline.fit(finetune_mode="lora")``.
- **Chronos-Bolt** is a T5-style ``PreTrainedModel``, so it takes the standard
  ``transformers`` + ``peft`` route with an explicit training loop. Having two independent
  paths means neither is a single point of failure.

Every fit is **fold-local**: the training window ends at the block's origin, and the
validation slice is carved from the end of that window. Nothing after the origin is read.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bench.backtest.splitter import expanding_origin_folds
from forecast_bench.config import CONTEXT_LENGTH, MAX_HORIZON, RANDOM_SEED, get_config
from forecast_bench.models.base import (
    sample_efficiency_window_size,
)
from forecast_bench.models.foundation.chronos2 import CHRONOS2_MODEL_ID
from forecast_bench.models.foundation.chronos_bolt import CHRONOS_BOLT_MODEL_ID
from forecast_bench.models.foundation.hub import ensure_model_card, revision_tag

logger = logging.getLogger(__name__)

# --- The recipe, fixed in advance -------------------------------------------------------
#
# PREREGISTRATION.md section 5 commits to not re-tuning these after seeing results.

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

#: Attention projections targeted by LoRA. Both checkpoints are T5-style.
LORA_TARGET_MODULES = ["q", "k", "v", "o"]

#: Early stopping patience, in evaluations, on the fold-local validation slice.
EARLY_STOPPING_PATIENCE = 3

#: Number of validation windows carved from the end of each training block.
N_VALIDATION_WINDOWS = 32


@dataclass
class FinetuneResult:
    """Outcome of fine-tuning one configuration.

    Attributes:
        tag: Hub revision tag identifying this configuration.
        output_dir: Local directory holding the saved checkpoint.
        trainable_parameters: Number of parameters LoRA actually trains.
        total_parameters: Total parameters in the base model.
        steps: Optimiser steps actually run.
        best_validation_loss: Best validation quantile loss seen.
        skipped: Whether an existing Hub checkpoint made this a no-op.
    """

    tag: str
    output_dir: Path | None = None
    trainable_parameters: int = 0
    total_parameters: int = 0
    steps: int = 0
    best_validation_loss: float = float("nan")
    skipped: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def trainable_fraction(self) -> float:
        """Share of parameters LoRA trains. Reported in the model card."""
        if not self.total_parameters:
            return float("nan")
        return self.trainable_parameters / self.total_parameters


def slice_training_window(
    series: pd.Series, origin: pd.Timestamp, training_window: str = "full"
) -> np.ndarray:
    """Take the values a block may train on, honouring the sample-efficiency slice.

    Args:
        series: The full target series.
        origin: The block's origin. Nothing after it is read.
        training_window: Key into :data:`~forecast_bench.models.base.SAMPLE_EFFICIENCY_DAYS`.

    Returns:
        Training values in observation order.

    Raises:
        KeyError: If the window name is unknown.
        ValueError: If the slice is too short to build one training example.

    Note:
        ``training_window`` is resolved through
        :func:`~forecast_bench.models.base.sample_efficiency_window_size`, which converts
        a label like ``"1y"`` into a raw observation count large enough to contain at
        least one full context-plus-horizon window. A literal 252-day slice would be
        shorter than the 512-step context every model in the study uses and could not
        supply a single training example — see that function's docstring for what went
        wrong when this treated the label as a raw day count directly.
    """
    usable = series.loc[series.index <= origin].dropna()
    limit = sample_efficiency_window_size(training_window)
    values = usable.to_numpy(dtype=float)
    if limit is not None:
        values = values[-limit:]

    minimum = CONTEXT_LENGTH + MAX_HORIZON
    if len(values) < minimum:
        raise ValueError(
            f"Training window {training_window!r} at {origin.date()} has {len(values)} "
            f"observations, fewer than the {minimum} needed for one example."
        )
    return values


def split_validation(
    values: np.ndarray,
    prediction_length: int = MAX_HORIZON,
    context_length: int = CONTEXT_LENGTH,
    n_windows: int = N_VALIDATION_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation slice from the **end** of a training block.

    Args:
        values: Training values in observation order.
        prediction_length: Forecast horizon each validation window covers.
        context_length: Context each window needs.
        n_windows: How many validation windows to reserve.

    Returns:
        A ``(train_values, validation_values)`` pair. The validation slice keeps the
        context it needs, so the two overlap by ``context_length`` observations.

    Note:
        Validation comes from the end of the block rather than a random split, because a
        random split would let the model validate against points it effectively
        interpolates between. The end slice is the closest available proxy for the
        out-of-sample period the model will actually face.
    """
    reserve = n_windows + prediction_length
    if len(values) <= reserve + context_length + prediction_length:
        # Too short to reserve a validation slice; train on everything and skip early
        # stopping rather than producing a validation set of one point.
        return values, np.empty(0)

    split = len(values) - reserve
    return values[:split], values[split - context_length :]


def _log_trainable(model) -> tuple[int, int]:
    """Count trainable and total parameters, and log the share.

    Args:
        model: A torch module.

    Returns:
        A ``(trainable, total)`` pair.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA trains %s of %s parameters (%.3f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total else float("nan"),
    )
    return trainable, total


def finetune_chronos2_block(
    values: np.ndarray,
    output_dir: Path,
    prediction_length: int = MAX_HORIZON,
    context_length: int = CONTEXT_LENGTH,
    learning_rate: float = 1e-5,
    num_steps: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
) -> FinetuneResult:
    """LoRA fine-tune Chronos-2 on one block's training window.

    Args:
        values: Training values for this block, ending at its origin.
        output_dir: Where to save the fine-tuned checkpoint.
        prediction_length: Forecast horizon to fine-tune for.
        context_length: Context length used during fine-tuning.
        learning_rate: Optimiser learning rate. The upstream default of 1e-6 is for full
            fine-tuning; LoRA wants roughly 1e-5.
        num_steps: Maximum optimiser steps.
        batch_size: Series per batch.
        device: Torch device.

    Returns:
        The result, including the trainable-parameter count.

    Note:
        Uses the official ``Chronos2Pipeline.fit`` with ``finetune_mode="lora"``. The
        monitored validation quantity is the model's own quantile loss, which is weighted
        quantile loss up to a normalising constant — the same quantity
        ``evaluation/metrics.py`` reports, not a proxy for it.
    """
    import torch
    from chronos import Chronos2Pipeline

    train_values, validation_values = split_validation(
        values, prediction_length, context_length
    )

    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS2_MODEL_ID, device_map=device)
    total_before = sum(p.numel() for p in pipeline.model.parameters())

    inputs = [torch.tensor(train_values, dtype=torch.float32)]
    validation_inputs = (
        [torch.tensor(validation_values, dtype=torch.float32)]
        if validation_values.size
        else None
    )

    lora_config = {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
    }

    callbacks = []
    trainer_kwargs: dict = {}
    if validation_inputs is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        )
        trainer_kwargs.update(
            eval_strategy="steps",
            eval_steps=max(num_steps // 10, 1),
            save_strategy="steps",
            save_steps=max(num_steps // 10, 1),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    finetuned = pipeline.fit(
        inputs,
        prediction_length=prediction_length,
        validation_inputs=validation_inputs,
        finetune_mode="lora",
        lora_config=lora_config,
        context_length=context_length,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=str(output_dir),
        callbacks=callbacks or None,
        **trainer_kwargs,
    )

    trainable, total = _log_trainable(finetuned.model)
    finetuned.save_pretrained(str(output_dir))

    return FinetuneResult(
        tag="",
        output_dir=output_dir,
        trainable_parameters=trainable,
        total_parameters=total or total_before,
        steps=num_steps,
    )


def finetune_bolt_block(
    values: np.ndarray,
    output_dir: Path,
    prediction_length: int = MAX_HORIZON,
    context_length: int = CONTEXT_LENGTH,
    learning_rate: float = 1e-4,
    num_steps: int = 1000,
    batch_size: int = 32,
    eval_every: int = 100,
    device: str = "cuda",
    seed: int = RANDOM_SEED,
) -> FinetuneResult:
    """LoRA fine-tune Chronos-Bolt on one block, via ``transformers`` + ``peft``.

    Args:
        values: Training values for this block, ending at its origin.
        output_dir: Where to save the adapter.
        prediction_length: Forecast horizon to fine-tune for.
        context_length: Context length per training example.
        learning_rate: AdamW learning rate.
        num_steps: Maximum optimiser steps.
        batch_size: Windows per batch.
        eval_every: Steps between validation evaluations.
        device: Torch device.
        seed: Random seed for window sampling.

    Returns:
        The result, including the trainable-parameter count and best validation loss.

    Note:
        Bolt exposes no ``fit``, so the loop is explicit. Its ``forward`` returns its own
        quantile loss when given a target, so the objective here is the model's native
        training objective rather than something reconstructed.
    """
    import torch
    from chronos import ChronosBoltPipeline
    from peft import LoraConfig, get_peft_model

    train_values, validation_values = split_validation(
        values, prediction_length, context_length
    )

    pipeline = ChronosBoltPipeline.from_pretrained(
        CHRONOS_BOLT_MODEL_ID, device_map=device
    )
    model = get_peft_model(
        pipeline.model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        ),
    )
    trainable, total = _log_trainable(model)

    def sample_batch(source: np.ndarray, rng: np.random.Generator, size: int):
        """Draw a batch of (context, target) windows from a value array."""
        highest = len(source) - context_length - prediction_length
        starts = rng.integers(0, max(highest, 1), size=size)
        context = np.stack([source[s : s + context_length] for s in starts])
        target = np.stack(
            [
                source[s + context_length : s + context_length + prediction_length]
                for s in starts
            ]
        )
        return (
            torch.tensor(context, dtype=torch.float32, device=device),
            torch.tensor(target, dtype=torch.float32, device=device),
        )

    rng = np.random.default_rng(seed)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    best_loss = float("inf")
    since_improvement = 0
    completed = 0

    model.train()
    for step in range(1, num_steps + 1):
        context, target = sample_batch(train_values, rng, batch_size)
        loss = model(context=context, target=target).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        completed = step

        if validation_values.size and step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                validation_rng = np.random.default_rng(seed)
                v_context, v_target = sample_batch(
                    validation_values, validation_rng, batch_size
                )
                validation_loss = float(model(context=v_context, target=v_target).loss)
            model.train()

            logger.info("step %d | validation loss %.5f", step, validation_loss)
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                since_improvement = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(output_dir))
            else:
                since_improvement += 1
                if since_improvement >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping at step %d after %d evaluations without "
                        "improvement",
                        step,
                        EARLY_STOPPING_PATIENCE,
                    )
                    break

    output_dir.mkdir(parents=True, exist_ok=True)
    if not validation_values.size:
        model.save_pretrained(str(output_dir))

    return FinetuneResult(
        tag="",
        output_dir=output_dir,
        trainable_parameters=trainable,
        total_parameters=total,
        steps=completed,
        best_validation_loss=best_loss if np.isfinite(best_loss) else float("nan"),
    )


def existing_hub_revisions(repo_id: str | None = None) -> set[str]:
    """Revision tags already present on the model repo.

    Args:
        repo_id: Model repo. Defaults to the configured one.

    Returns:
        Tag names. Empty if the repo has none or cannot be read.

    Note:
        This is what makes a dropped Colab session cost one block rather than the run.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    try:
        refs = api.list_repo_refs(repo_id or config.hf_model_repo)
    except Exception as error:  # noqa: BLE001 - a fresh repo has no refs yet
        logger.info("Could not list revisions (%s); starting fresh", error)
        return set()
    return {ref.name for ref in list(refs.branches) + list(refs.tags)}


def push_block(
    output_dir: Path, tag: str, metadata: dict, repo_id: str | None = None
) -> str:
    """Upload one fine-tuned block to the Hub under its own revision.

    Args:
        output_dir: Directory holding the checkpoint.
        tag: Revision tag identifying the configuration.
        metadata: Run metadata written alongside the weights.
        repo_id: Model repo. Defaults to the configured one.

    Returns:
        The tag written to.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_model_repo

    (output_dir / "forecast_bench_run.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    try:
        api.create_branch(destination, branch=tag, exist_ok=True)
    except Exception as error:  # noqa: BLE001 - branch creation is best-effort
        logger.warning("Could not create branch %s (%s); pushing to main", tag, error)

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=destination,
        revision=tag,
        commit_message=f"forecast_bench: {tag}",
    )
    logger.info("Pushed %s to %s", tag, destination)
    return tag


def run_campaign(
    series: str,
    frame: pd.DataFrame | None = None,
    arm: str = "A",
    model: str = "chronos2",
    training_windows: tuple[str, ...] = ("full",),
    push: bool = True,
    resume: bool = True,
    device: str = "cuda",
    num_steps: int = 1000,
    output_root: Path | None = None,
) -> list[FinetuneResult]:
    """Fine-tune every block of one series, checkpointing to the Hub as it goes.

    This is the function the Colab notebooks call. The loop over blocks lives here, not in
    a notebook cell.

    Args:
        series: Target series name.
        frame: Processed frame. Pulled from the Hub when ``None``.
        arm: ``"A"`` or ``"B"``.
        model: ``"chronos2"`` or ``"bolt"``.
        training_windows: Sample-efficiency slices to run.
        push: Upload each block to the Hub when finished.
        resume: Skip configurations whose revision already exists on the Hub.
        device: Torch device.
        num_steps: Maximum optimiser steps per block.
        output_root: Local checkpoint directory. Defaults to ``checkpoints/``.

    Returns:
        One :class:`FinetuneResult` per configuration, in run order.

    Raises:
        ValueError: If ``model`` is not a supported checkpoint.
    """
    if model not in {"chronos2", "bolt"}:
        raise ValueError(f"Unknown model {model!r}; expected 'chronos2' or 'bolt'")

    if frame is None:
        from forecast_bench.data.hub import load_processed

        frame = load_processed(series)

    target = frame[series]
    folds = list(expanding_origin_folds(frame.index))

    # One fine-tune per block, at the block's first origin. Blocks are calendar years.
    block_origins: dict[int, pd.Timestamp] = {}
    for fold in folds:
        block_origins.setdefault(fold.block_id, fold.origin)

    if push:
        ensure_model_card()
    already = existing_hub_revisions() if resume else set()
    # checkpoints/ is gitignored; the Hub revision is the durable copy.
    root = output_root or Path("checkpoints")

    results: list[FinetuneResult] = []
    for window in training_windows:
        for block, origin in sorted(block_origins.items()):
            tag = revision_tag(series, arm, block, window)
            if tag in already:
                logger.info("Skipping %s; already on the Hub", tag)
                results.append(FinetuneResult(tag=tag, skipped=True))
                continue

            logger.info(
                "Fine-tuning %s | block %s | window %s | origin %s",
                model,
                block,
                window,
                origin.date(),
            )
            values = slice_training_window(target, origin, window)
            output_dir = root / tag

            fit = (
                finetune_chronos2_block if model == "chronos2" else finetune_bolt_block
            )
            result = fit(
                values,
                output_dir=output_dir,
                device=device,
                num_steps=num_steps,
            )
            result.tag = tag
            result.metadata = {
                "series": series,
                "arm": arm,
                "model": model,
                "block": block,
                "origin": str(origin.date()),
                "training_window": window,
                "n_observations": int(len(values)),
                "lora": {
                    "r": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGET_MODULES,
                },
                "trainable_parameters": result.trainable_parameters,
                "total_parameters": result.total_parameters,
                "trainable_fraction": result.trainable_fraction,
                "steps": result.steps,
                "best_validation_loss": result.best_validation_loss,
            }

            if push:
                push_block(output_dir, tag, result.metadata)
            results.append(result)

    return results
