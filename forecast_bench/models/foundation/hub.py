"""Push and pull fine-tuned checkpoints on ``forecastbench-chronos``.

One revision tag per (series, arm, block, training-window) combination. The Space and the
local environment both pull by tag, so the demo and the repository cannot drift apart.
"""

import logging

from forecast_bench.config import get_config

logger = logging.getLogger(__name__)

#: Model card written before any checkpoint is uploaded.
#:
#: The license is not optional bookkeeping. Chronos is Apache 2.0 and a LoRA-fine-tuned
#: derivative inherits that, so the field must be set before weights exist in the repo
#: rather than added afterwards.
MODEL_CARD = """---
license: apache-2.0
base_model: amazon/chronos-2
library_name: peft
tags:
  - time-series-forecasting
  - chronos
  - lora
  - finance
---

# forecastbench-chronos

LoRA adapters fine-tuning [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) on
two financial series, produced for
[forecast_bench](https://github.com/Rohanjain2312/forecast_bench).

## What this is

A leakage-safe benchmark comparing classical statistical forecasting against time-series
foundation models. The study's conclusion was written down and committed to git *before*
any model ran — see
[`PREREGISTRATION.md`](https://github.com/Rohanjain2312/forecast_bench/blob/main/PREREGISTRATION.md),
which states in advance what would count as the foundation model losing.

## Targets

- **SPY log realized variance**, from the Garman-Klass estimator on daily OHLC
- **`DGS10`**, the 10-year Treasury yield in levels

## Fine-tuning recipe

- LoRA via `peft`: rank 8, alpha 16, dropout 0.05, targeting attention projections
- Early stopping on a validation slice carved from the end of each training block,
  patience 3, monitoring validation weighted quantile loss
- One adapter per (series, arm, block, training-window-size), each tagged as its own
  revision

## Revisions

Pull a specific configuration by tag rather than by branch, so a result is always
reproducible from the exact weights that produced it:

```python
from forecast_bench.models.foundation.hub import load_adapter
adapter = load_adapter("spy-logrv-armA-chronos2-2020-full")
```

## Honest caveats

**Pretraining contamination.** The base model was released in October 2025 and pretrained
on a corpus that plausibly includes public financial series — SPY and Treasury yields are
among the most widely redistributed time series in existence. Most of the study's test span
(2015-2026) predates that release, so *zero-shot* numbers on the early span may not be
genuinely out-of-sample. This is unfixable, since the pretraining corpus is not
inspectable, and it is reported as a first-class limitation rather than a footnote.

The contamination inflates zero-shot results, not fine-tuned ones, so the
fine-tuned-versus-zero-shot gap remains interpretable.

**Small fold count.** Roughly 137 non-overlapping forecast origins is a small sample for
Diebold-Mariano. A Model Confidence Set is reported alongside pairwise tests, and no claim
rests on a single p-value.

## License

Apache 2.0, inherited from the Chronos base model.
"""


def ensure_model_card(repo_id: str | None = None) -> str:
    """Create the model repo if needed and write its card.

    Args:
        repo_id: Destination model repo. Defaults to the configured one.

    Returns:
        The repo id written to.

    Note:
        Called before any checkpoint upload. A model repo carrying weights but no license
        is an unlicensed redistribution of an Apache-2.0 derivative.
    """
    from huggingface_hub import HfApi

    config = get_config()
    api = HfApi(token=config.require_secret("hf_token"))
    destination = repo_id or config.hf_model_repo

    api.create_repo(destination, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=destination,
    )
    logger.info("Wrote model card to %s", destination)
    return destination


def revision_tag(
    series: str,
    arm: str,
    block: int | str,
    training_window: str = "full",
    model: str = "chronos2",
) -> str:
    """Build the revision tag for one fine-tuning configuration.

    Args:
        series: Target series name.
        arm: ``"A"`` or ``"B"``.
        block: Block identifier, normally the calendar year.
        training_window: Sample-efficiency slice, e.g. ``"1y"`` or ``"full"``.
        model: Which base checkpoint this adapter fine-tunes, ``"chronos2"`` or ``"bolt"``.

    Returns:
        A tag such as ``"spy-logrv-armA-chronos2-2020-full"``.

    Note:
        ``model`` is a required axis, not decoration. Without it, Chronos-2's and
        Chronos-Bolt's full-window tags for the same (series, arm, block) collide exactly
        — both would be ``"spy-logrv-armA-2020-full"``. Whichever model finishes first
        pushes that tag, and the other model's fine-tuning is then silently skipped forever
        as "already on the Hub," with no error at any point. This happened live: Step 8's
        Bolt run reported success and pushed nothing, because every tag it computed already
        existed from Step 5's Chronos-2 run. See docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    return f"{series.replace('_', '-')}-arm{arm}-{model}-{block}-{training_window}"
