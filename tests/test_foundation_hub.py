"""Tests for revision tagging and the resume logic built on top of it.

Regression coverage for a bug that produced no error at all and silently discarded a full
training run: ``revision_tag`` had no ``model`` axis, so Chronos-2's and Chronos-Bolt's
full-window tags for the same (series, arm, block) were byte-identical. Whichever model
finished first pushed that tag; the other model's fine-tuning was then skipped forever as
"already on the Hub." This happened live — see docs/planning/PROGRESS_NOTES.md, Step 16.
"""

from unittest.mock import patch

from forecast_bench.models.foundation.hub import revision_tag


def test_revision_tag_distinguishes_models_for_otherwise_identical_configurations() -> (
    None
):
    """The exact collision that was hit live: same series, arm, block, window."""
    chronos2 = revision_tag("spy_logrv", "A", 2014, "full", model="chronos2")
    bolt = revision_tag("spy_logrv", "A", 2014, "full", model="bolt")

    assert chronos2 != bolt


def test_revision_tag_defaults_to_chronos2_for_backward_compatibility() -> None:
    """Calls that predate the model parameter still resolve to a valid, specific tag."""
    assert revision_tag("spy_logrv", "A", 2014, "full") == revision_tag(
        "spy_logrv", "A", 2014, "full", model="chronos2"
    )


def test_revision_tag_still_distinguishes_every_other_axis() -> None:
    """Series, arm, block and window remain distinguishing, as before the fix."""
    base = revision_tag("spy_logrv", "A", 2014, "full", model="chronos2")
    assert base != revision_tag("dgs10", "A", 2014, "full", model="chronos2")
    assert base != revision_tag("spy_logrv", "B", 2014, "full", model="chronos2")
    assert base != revision_tag("spy_logrv", "A", 2015, "full", model="chronos2")
    assert base != revision_tag("spy_logrv", "A", 2014, "1y", model="chronos2")


def test_run_campaign_does_not_skip_bolt_because_chronos2_already_ran(tmp_path) -> None:
    """The end-to-end shape of the bug: run_campaign(model='bolt') must actually fit.

    Simulates the exact live scenario: the Hub already carries every full-window
    Chronos-2 tag (from an earlier campaign), and a Bolt campaign is run against the same
    series, arm, blocks and window. Before the fix, every Bolt tag matched an
    already-existing Chronos-2 tag and the fit function was never called.
    """
    import numpy as np
    import pandas as pd

    from forecast_bench.models.foundation import finetune as ft
    from forecast_bench.models.foundation.hub import revision_tag as real_revision_tag

    index = pd.date_range("2000-01-03", periods=4000, freq="B")
    frame = pd.DataFrame(
        {"spy_logrv": np.random.default_rng(0).standard_normal(4000).cumsum()},
        index=index,
    )

    chronos2_tags_already_on_hub = {
        real_revision_tag("spy_logrv", "A", block, "full", model="chronos2")
        for block in range(2014, 2027)
    }

    fit_calls: list[str] = []

    def fake_bolt_fit(values, output_dir, **kwargs):
        fit_calls.append(output_dir.name)
        return ft.FinetuneResult(tag="", output_dir=output_dir)

    with (
        patch.object(
            ft, "existing_hub_revisions", return_value=chronos2_tags_already_on_hub
        ),
        patch.object(ft, "ensure_model_card"),
        patch.object(ft, "push_block"),
        patch.object(ft, "finetune_bolt_block", side_effect=fake_bolt_fit),
    ):
        results = ft.run_campaign(
            series="spy_logrv",
            frame=frame,
            model="bolt",
            training_windows=("full",),
            push=True,
            output_root=tmp_path,
        )

    assert (
        fit_calls
    ), "Bolt fine-tuning was skipped entirely -- the collision reproduced"
    assert not any(result.skipped for result in results)
    assert all("bolt" in result.tag for result in results)
