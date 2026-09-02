"""Refit cadence policies.

The fairness crux of the study, from DECISIONS.md D5. Refitting ARIMA at every fold is
cheap and correct; retraining a fine-tuned Chronos at every fold is not viable on any
budget. "Refit everything every fold" is impossible, and "refit classical every fold,
learned models once" is an unfair fight the classical arm wins on freshness rather than on
merit.

The resolution is to run every model under both policies and report both:

- **matched** (headline): every model refits at annual block boundaries, so every model
  class gets the same information-refresh rate and the comparison is about model quality.
- **native** (secondary): classical models refit every fold, which is what a practitioner
  would actually do, and hiding it would understate the classical arm.

The gap between the two is itself a reportable finding: it measures how much of the
classical arm's performance comes from frequent refitting rather than from the model.
"""

import logging
from typing import Protocol

from forecast_bench.backtest.splitter import Fold

logger = logging.getLogger(__name__)

#: Mapping from a block frequency alias to the pandas period alias used to key it.
_PERIOD_ALIASES = {"YS": "Y", "QS": "Q", "MS": "M"}


class RefitCadence(Protocol):
    """Decides whether a model is refitted at a given fold."""

    name: str

    def should_refit(self, model_id: str, fold: Fold) -> bool:
        """Return whether ``model_id`` should be refitted at ``fold``.

        Args:
            model_id: Identifier of the model being considered.
            fold: The fold about to be evaluated.

        Returns:
            ``True`` to refit, ``False`` to reuse the cached fit.
        """
        ...

    def reset(self) -> None:
        """Clear any internal state, so one policy object can drive several runs."""
        ...


class EveryFoldCadence:
    """Refit at every fold.

    What a practitioner does with ARIMA, and the ``native`` half of the dual-cadence
    protocol.

    Attributes:
        name: Policy label written into the results table.
    """

    name = "every_fold"

    def should_refit(self, model_id: str, fold: Fold) -> bool:
        """Always refit.

        Args:
            model_id: Unused; every model refits.
            fold: Unused; every fold refits.

        Returns:
            Always ``True``.
        """
        return True

    def reset(self) -> None:
        """No state to clear."""


class BlockCadence:
    """Refit once per block, then reuse the fit within the block.

    The ``matched`` half of the dual-cadence protocol. Blocks are calendar years by
    default, aligned with :attr:`~forecast_bench.backtest.splitter.Fold.block_id`.

    Attributes:
        name: Policy label written into the results table.
        freq: Block frequency alias, ``"YS"`` for calendar years.
    """

    def __init__(self, freq: str = "YS") -> None:
        """Initialise the policy.

        Args:
            freq: Block frequency. ``"YS"`` (calendar year), ``"QS"``, or ``"MS"``.

        Raises:
            ValueError: If ``freq`` is not a recognised block frequency.
        """
        if freq not in _PERIOD_ALIASES:
            raise ValueError(
                f"Unsupported block frequency {freq!r}; expected one of "
                f"{sorted(_PERIOD_ALIASES)}"
            )
        self.freq = freq
        self.name = f"block_{freq.lower()}"
        self._last_block: dict[str, object] = {}

    def _block_key(self, fold: Fold) -> object:
        """Return the block a fold belongs to.

        Args:
            fold: The fold to classify.

        Returns:
            A hashable block key.
        """
        if self.freq == "YS":
            # block_id is already the calendar year, so use it rather than recomputing a
            # period and risking the two disagreeing.
            return fold.block_id
        return fold.origin.to_period(_PERIOD_ALIASES[self.freq])

    def should_refit(self, model_id: str, fold: Fold) -> bool:
        """Refit only at the first fold of each block.

        Args:
            model_id: Identifier of the model being considered.
            fold: The fold about to be evaluated.

        Returns:
            ``True`` at the first fold of a block for this model, ``False`` within it.
        """
        block = self._block_key(fold)
        if self._last_block.get(model_id) == block:
            return False
        self._last_block[model_id] = block
        return True

    def reset(self) -> None:
        """Forget which blocks have been fitted, so the policy can drive another run."""
        self._last_block.clear()


#: The two named configurations reported in the study.
CADENCES = {
    "matched": BlockCadence,
    "native": EveryFoldCadence,
}


def build_cadence(name: str) -> RefitCadence:
    """Construct a cadence policy by name.

    Args:
        name: ``"matched"`` or ``"native"``.

    Returns:
        A fresh policy object.

    Raises:
        KeyError: If the name is not one of the two reported configurations.
    """
    if name not in CADENCES:
        raise KeyError(f"Unknown cadence {name!r}; expected one of {sorted(CADENCES)}")
    return CADENCES[name]()
