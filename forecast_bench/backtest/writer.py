"""The tidy long-format results schema, defined exactly once.

Everything downstream — metrics, plots, Diebold-Mariano tests, the Hugging Face Space —
reads this schema and only this schema. One format, computed once, so the demo cannot show
a number the README disagrees with: they come from the same file.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bench.backtest.protocol import QuantileForecast
from forecast_bench.backtest.splitter import Fold

logger = logging.getLogger(__name__)

#: The one schema. Column order is fixed so the parquet is diff-friendly.
SCHEMA = [
    "origin",
    "target_date",
    "step",
    "model_id",
    "quantile",
    "value",
    "actual",
    "regime",
    "block_id",
    "series",
    "arm",
    "cadence",
]


def forecast_to_rows(
    forecast: QuantileForecast,
    fold: Fold,
    actuals: pd.Series,
    *,
    series: str,
    arm: str,
    cadence: str,
) -> pd.DataFrame:
    """Flatten one forecast into tidy long-format rows.

    Args:
        forecast: The model's quantile path for this fold.
        fold: The fold the forecast belongs to.
        actuals: The realised target series, indexed by date. Values are looked up by
            date, so a missing observation becomes NaN rather than a misalignment.
        series: Target series name, e.g. ``"spy_logrv"``.
        arm: ``"A"`` (univariate) or ``"B"`` (covariate-informed).
        cadence: Refit cadence label, e.g. ``"matched"``.

    Returns:
        One row per (step, quantile), with :data:`SCHEMA` columns in order.
    """
    levels = sorted(forecast.quantiles)
    horizon = forecast.horizon

    steps = np.tile(np.arange(1, horizon + 1), len(levels))
    dates = np.tile(np.asarray(forecast.index), len(levels))
    quantiles = np.repeat(levels, horizon)
    values = np.concatenate([np.asarray(forecast.quantiles[q]) for q in levels])

    frame = pd.DataFrame(
        {
            "origin": fold.origin,
            "target_date": dates,
            "step": steps,
            "model_id": forecast.model_id,
            "quantile": quantiles,
            "value": values,
            "actual": actuals.reindex(pd.DatetimeIndex(dates)).to_numpy(),
            "regime": fold.regime,
            "block_id": fold.block_id,
            "series": series,
            "arm": arm,
            "cadence": cadence,
        }
    )
    return frame[SCHEMA]


class ForecastWriter:
    """Accumulates tidy rows and writes one parquet per (series, arm, cadence).

    Attributes:
        series: Target series name.
        arm: Experiment arm.
        cadence: Refit cadence label.
    """

    def __init__(self, series: str, arm: str, cadence: str) -> None:
        """Initialise an empty writer.

        Args:
            series: Target series name.
            arm: ``"A"`` or ``"B"``.
            cadence: Refit cadence label.
        """
        self.series = series
        self.arm = arm
        self.cadence = cadence
        self._chunks: list[pd.DataFrame] = []

    def append(
        self, forecast: QuantileForecast, fold: Fold, actuals: pd.Series
    ) -> None:
        """Add one forecast to the buffer.

        Args:
            forecast: The model's quantile path for this fold.
            fold: The fold the forecast belongs to.
            actuals: The realised target series.
        """
        self._chunks.append(
            forecast_to_rows(
                forecast,
                fold,
                actuals,
                series=self.series,
                arm=self.arm,
                cadence=self.cadence,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Return everything appended so far as one frame.

        Returns:
            The tidy results frame, or an empty frame with :data:`SCHEMA` columns.
        """
        if not self._chunks:
            return pd.DataFrame(columns=SCHEMA)
        return pd.concat(self._chunks, ignore_index=True)

    def write(self, directory: Path) -> Path:
        """Write the accumulated rows to parquet.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            Path to the written file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.series}_arm{self.arm}_{self.cadence}.parquet"
        frame = self.to_frame()
        frame.to_parquet(path, index=False)
        logger.info("Wrote %d forecast rows to %s", len(frame), path.resolve())
        return path


def write_results(frame: pd.DataFrame, directory: Path) -> Path:
    """Write an already-tidy results frame, naming the file from its own columns.

    Args:
        frame: A frame in :data:`SCHEMA`, carrying ``series``, ``arm`` and ``cadence``.
        directory: Destination directory, created if absent.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If the frame is empty, or mixes more than one (series, arm, cadence)
            combination, which would produce a filename that lies about its contents.
    """
    if frame.empty:
        raise ValueError("Refusing to write an empty results frame")

    keys = {
        column: sorted(frame[column].unique())
        for column in ("series", "arm", "cadence")
    }
    mixed = {column: values for column, values in keys.items() if len(values) != 1}
    if mixed:
        raise ValueError(
            f"Results frame mixes multiple values for {mixed}; write one "
            "(series, arm, cadence) combination per file."
        )

    directory.mkdir(parents=True, exist_ok=True)
    series, arm, cadence = (keys[column][0] for column in ("series", "arm", "cadence"))
    path = directory / f"{series}_arm{arm}_{cadence}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("Wrote %d forecast rows to %s", len(frame), path.resolve())
    return path


def merge_forecasts(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine forecast frames covering different models of the same configuration.

    Args:
        *frames: Tidy frames to merge. Empty and ``None`` entries are ignored.

    Returns:
        One frame in :data:`SCHEMA` order, sorted by origin, model and step.

    Raises:
        ValueError: If nothing usable was passed, if the frames disagree on
            ``(series, arm, cadence)``, or if a model appears in more than one frame.

    Note:
        Needed because the panel is computed in two places. The neural models are trained
        on Colab and their forecasts come back through the Hub; everything else runs
        locally. Merging on ``model_id`` rather than concatenating blindly is what stops a
        model being counted twice — the same duplication that would silently average every
        metric over doubled rows.
    """
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        raise ValueError("No non-empty forecast frames to merge")

    for column in ("series", "arm", "cadence"):
        values = {v for frame in usable for v in frame[column].unique()}
        if len(values) != 1:
            raise ValueError(
                f"Cannot merge frames spanning multiple {column} values: {sorted(values)}"
            )

    seen: dict[str, int] = {}
    for position, frame in enumerate(usable):
        for model_id in frame["model_id"].unique():
            if model_id in seen:
                raise ValueError(
                    f"Model {model_id!r} appears in both frame {seen[model_id]} and "
                    f"frame {position}. Merging would double-count it."
                )
            seen[model_id] = position

    merged = pd.concat(usable, ignore_index=True)[SCHEMA]
    merged = merged.sort_values(["origin", "model_id", "step", "quantile"])
    logger.info(
        "Merged %d models into %d rows: %s",
        merged["model_id"].nunique(),
        len(merged),
        ", ".join(sorted(merged["model_id"].unique())),
    )
    return merged.reset_index(drop=True)
