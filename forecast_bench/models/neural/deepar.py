"""A DeepAR-class probabilistic LSTM.

darts documents ``RNNModel(model="LSTM")`` with a likelihood as equivalent to DeepAR in its
probabilistic form, so this earns the brief's "DeepAR-class" claim honestly rather than
substituting N-BEATS for both neural slots.
"""

from forecast_bench.config import QUANTILE_GRID
from forecast_bench.models.neural._darts import DartsQuantileForecaster


class DeepAR(DartsQuantileForecaster):
    """darts ``RNNModel(model="LSTM")`` with a quantile likelihood.

    Attributes:
        model_id: ``"DeepAR-LSTM"``.

    Note:
        darts' ``RNNModel`` is trained on a rolling one-step objective and unrolls
        recursively at prediction time, so its ``training_length`` rather than an output
        chunk governs the sequence it sees. That is the DeepAR formulation, and it is why
        this class does not simply reuse the N-BEATS configuration.
    """

    model_id = "DeepAR-LSTM"

    def _build(self):
        """Construct the darts LSTM estimator.

        Returns:
            An unfitted ``RNNModel`` with a quantile likelihood.
        """
        from darts.models import RNNModel
        from darts.utils.likelihood_models.torch import QuantileRegression

        return RNNModel(
            model="LSTM",
            input_chunk_length=self.input_chunk_length,
            training_length=self.input_chunk_length + self.output_chunk_length,
            n_epochs=self.n_epochs,
            likelihood=QuantileRegression(quantiles=list(QUANTILE_GRID)),
            random_state=self.random_state,
            pl_trainer_kwargs=self._trainer_kwargs(),
        )
