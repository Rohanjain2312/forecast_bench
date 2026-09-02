"""Alignment of a target with its covariates onto one trading-day index.

The index is always the **target's own** observation dates. Covariates are aligned onto
it; they never extend it. A covariate that trades on a day the target does not is not a
reason to invent a target observation.

Nothing here forward-fills. Gap handling is recorded, reported, and left to the caller.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def align_to_target(
    target: pd.Series,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join a target to its covariates on the target's index.

    Args:
        target: The modelling target, named. Its index defines the output index.
        covariates: Optional covariate frame. Reindexed onto the target's index.

    Returns:
        A frame whose first column is the target, followed by the covariate columns,
        sorted by date with duplicates removed.

    Raises:
        ValueError: If the target has a duplicated index entry, which would silently
            double-count observations in the backtest.
    """
    if target.index.has_duplicates:
        duplicates = target.index[target.index.duplicated()].unique()
        raise ValueError(
            f"Target {target.name!r} has {len(duplicates)} duplicated index entries, "
            f"first at {duplicates[0]}. A duplicated date double-counts an observation "
            "in every fold that contains it."
        )

    frame = target.sort_index().to_frame()
    if covariates is not None and not covariates.empty:
        frame = frame.join(covariates.reindex(frame.index))

    validate_frame(frame, target_name=str(target.name))
    return frame


def validate_frame(frame: pd.DataFrame, target_name: str) -> None:
    """Check the merged frame and log what a modeller needs to know about its gaps.

    Args:
        frame: Merged target-and-covariates frame.
        target_name: Column holding the target.

    Raises:
        ValueError: If the target column contains NaNs, or the index is not monotonic.

    Note:
        Covariate NaNs are reported but permitted — a missing VIX print is a real absence.
        Target NaNs are fatal: a model cannot be scored against a value that does not
        exist, and the only ways to remove one are to drop it or to forward-fill it, the
        latter being a leak.
    """
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Merged frame index is not sorted ascending.")

    target_nans = int(frame[target_name].isna().sum())
    if target_nans:
        raise ValueError(
            f"Target column {target_name!r} contains {target_nans} NaNs after merging. "
            "Drop them at construction; never forward-fill a target."
        )

    for column in frame.columns:
        if column == target_name:
            continue
        missing = int(frame[column].isna().sum())
        if missing:
            logger.info(
                "Covariate %r has %d missing values (%.2f%%) on the target's index; "
                "left as NaN, since gap handling belongs inside the fold.",
                column,
                missing,
                100.0 * missing / len(frame),
            )

    logger.info(
        "Merged %s: %d rows, %s to %s, %d columns",
        target_name,
        len(frame),
        frame.index.min().date(),
        frame.index.max().date(),
        len(frame.columns),
    )
