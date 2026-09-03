"""Tests for the zero-shot foundation models.

Marked ``slow`` because they download and run real checkpoints. Run them with::

    poetry run pytest tests/test_foundation_zeroshot.py -m slow
"""

import numpy as np
import pandas as pd
import pytest

from forecast_bench.config import MAX_HORIZON, QUANTILE_GRID
from forecast_bench.models.foundation.chronos2 import Chronos2ZeroShot
from forecast_bench.models.foundation.chronos_bolt import (
    BOLT_TRAINED_QUANTILES,
    ChronosBoltZeroShot,
)
from forecast_bench.models.foundation.finetune import (
    _with_input_embedding_accessors,
    finetune_bolt_block,
)
from forecast_bench.models.registry import foundation_model_ids

pytestmark = pytest.mark.slow


@pytest.fixture
def training_frame(synthetic_series) -> pd.DataFrame:
    """A single-column frame long enough to fill the context window."""
    return synthetic_series.to_frame(name="target")


@pytest.fixture
def forecast_index(training_frame) -> pd.DatetimeIndex:
    """Dates immediately after the training frame."""
    last = training_frame.index[-1]
    return pd.date_range(
        last + pd.tseries.offsets.BDay(1), periods=MAX_HORIZON, freq="B"
    )


def _forecast(model, frame, index):
    """Fit and forecast, returning the QuantileForecast."""
    model.fit(frame, origin=frame.index[-1])
    return model.predict(horizon=MAX_HORIZON, index=index)


def test_chronos2_emits_genuine_tail_quantiles(training_frame, forecast_index) -> None:
    """Chronos-2's trained range covers the study grid, so no level is clamped."""
    model = Chronos2ZeroShot(target_column="target")
    forecast = _forecast(model, training_frame, forecast_index)

    forecast.assert_monotonic()
    assert not np.allclose(forecast.quantiles[0.025], forecast.quantiles[0.1])
    assert not np.allclose(forecast.quantiles[0.975], forecast.quantiles[0.9])
    assert min(model.trained_quantiles) <= min(QUANTILE_GRID)
    assert max(model.trained_quantiles) >= max(QUANTILE_GRID)


def test_chronos_bolt_tails_are_clamped_to_its_trained_range(
    training_frame, forecast_index
) -> None:
    """Bolt cannot produce the study's tail levels, and the study records that as a fact.

    Asserted rather than described so that if a future checkpoint gains real tail
    predictions, this test fails and the limitation text gets revisited instead of
    silently becoming wrong.
    """
    model = ChronosBoltZeroShot(target_column="target")
    forecast = _forecast(model, training_frame, forecast_index)

    assert np.allclose(forecast.quantiles[0.025], forecast.quantiles[0.1])
    assert np.allclose(forecast.quantiles[0.975], forecast.quantiles[0.9])
    assert min(BOLT_TRAINED_QUANTILES) > min(QUANTILE_GRID)


def test_zero_shot_models_learn_nothing_from_our_data(
    training_frame, forecast_index
) -> None:
    """Two instances fitted on different windows forecast identically given equal context.

    Zero-shot means no parameters are estimated from this study's data. Only the context
    window differs between folds, so identical context must give identical output.
    """
    model_a = Chronos2ZeroShot(target_column="target")
    model_b = Chronos2ZeroShot(target_column="target")

    first = _forecast(model_a, training_frame, forecast_index)
    # A different training history, then the same final context restored.
    model_b.fit(training_frame.iloc[:1500], origin=training_frame.index[1499])
    second = _forecast(model_b, training_frame, forecast_index)

    assert np.allclose(first.median(), second.median())


def test_context_window_refreshes_between_folds(training_frame, forecast_index) -> None:
    """Conditioning on a later origin changes the forecast, as it must."""
    model = Chronos2ZeroShot(target_column="target")

    early = training_frame.iloc[:2000]
    model.fit(early, origin=early.index[-1])
    first = model.predict(horizon=MAX_HORIZON, index=forecast_index)

    model.fit(training_frame, origin=training_frame.index[-1], refit_parameters=False)
    second = model.predict(horizon=MAX_HORIZON, index=forecast_index)

    assert not np.allclose(first.median(), second.median())


def test_both_foundation_models_are_registered() -> None:
    """The registry exposes exactly the two zero-shot models the study runs."""
    assert foundation_model_ids() == ["Chronos2-ZeroShot", "ChronosBolt-ZeroShot"]


# --- Fine-tuning (Chronos-Bolt) ---------------------------------------------------------


def test_bolt_gains_the_embedding_accessors_peft_requires() -> None:
    """peft.get_peft_model calls get_input_embeddings; Chronos-Bolt raises on it.

    Regression test: this failed live on the first real Bolt fine-tuning run with
    ``NotImplementedError: get_input_embeddings not auto-handled for
    ChronosBoltModelForForecasting``. A time-series model reasonably has no token
    vocabulary, but it does keep T5's ``shared`` module, which is what the accessor
    should return. See docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    from chronos import ChronosBoltPipeline

    from forecast_bench.models.foundation.chronos_bolt import CHRONOS_BOLT_MODEL_ID

    model = ChronosBoltPipeline.from_pretrained(
        CHRONOS_BOLT_MODEL_ID, device_map="cpu"
    ).model

    with pytest.raises(NotImplementedError):
        model.get_input_embeddings()

    patched = _with_input_embedding_accessors(model)

    assert patched.get_input_embeddings() is patched.shared


def test_bolt_finetune_produces_a_real_lora_adapter(tmp_path) -> None:
    """A short Bolt fine-tune trains LoRA weights and writes a loadable adapter.

    Runs on CPU with a tiny step budget. The point is that the whole path executes --
    peft wrapping, the training loop, validation, early-stopping bookkeeping, and the
    save -- not that the result forecasts well. This is the least-covered path in the
    project because it is the one that only runs on Colab.

    ``peft`` lives in the optional ``gpu`` dependency group, so this skips unless it is
    present: ``poetry run pip install peft`` to exercise it locally.
    """
    pytest.importorskip("peft")

    import numpy as np

    values = np.random.default_rng(0).standard_normal(1200).cumsum()

    result = finetune_bolt_block(
        values,
        output_dir=tmp_path / "adapter",
        num_steps=6,
        eval_every=3,
        batch_size=4,
        device="cpu",
    )

    assert result.trainable_parameters > 0
    assert result.trainable_parameters < result.total_parameters
    assert 0 < result.trainable_fraction < 0.05, "LoRA should train a small fraction"
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
