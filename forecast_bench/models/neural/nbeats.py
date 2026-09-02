"""N-BEATS with a quantile-regression likelihood.

A from-scratch deep model trained on one series at a time — the counterpoint to a
foundation model that arrives already pretrained on millions of series. The
sample-efficiency sweep in DECISIONS.md D9 is largely a comparison of how fast this curve
rises against how fast Chronos-2's does.
"""

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.neural._darts import DartsQuantileForecaster


class NBEATS(DartsQuantileForecaster):
    """darts ``NBEATSModel`` with ``QuantileRegression``.

    Attributes:
        model_id: ``"N-BEATS"``.

    Note:
        The probabilistic variant is used deliberately. The deterministic one would emit a
        point forecast that then needed an interval bolted on, which is exactly the
        shortcut DECISIONS.md D4 rules out for every model in the panel.
    """

    model_id = "N-BEATS"

    def _build(self):
        """Construct the darts N-BEATS estimator.

        Returns:
            An unfitted ``NBEATSModel`` with a quantile likelihood.
        """
        from darts.models import NBEATSModel
        from darts.utils.likelihood_models.torch import QuantileRegression

        return NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            likelihood=QuantileRegression(quantiles=list(QUANTILE_GRID)),
            random_state=self.random_state,
            pl_trainer_kwargs=self._trainer_kwargs(),
        )
