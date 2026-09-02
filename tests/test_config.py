"""Tests for the config singleton, focused on the notebook-credential gotcha.

``get_config()`` is a process-wide ``lru_cache`` singleton so that every caller in one run
observes identical settings. That is the right default for a script, and the wrong default
for a notebook: Colab sets ``os.environ`` from a UI action (the Secrets panel) at a point
in time the module has no control over, and if anything calls ``get_config()`` even once
before that cell runs, the cached object is permanently secret-less for the rest of the
session. This bit a real Colab run — see docs/planning/PROGRESS_NOTES.md, Step 16.
"""

import pytest

from forecast_bench.config import Config, get_config


@pytest.fixture(autouse=True)
def _restore_config_cache(tmp_path, monkeypatch):
    """Isolate each test's cache state and working directory.

    Runs from an empty ``tmp_path`` rather than the repo root, so a developer's real
    ``.env`` (which has genuine secrets) cannot leak into these tests and cannot mask the
    exact absence these tests are checking for.
    """
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_get_config_is_cached_across_calls() -> None:
    """The singleton returns the identical object, not merely an equal one."""
    assert get_config() is get_config()


def test_env_var_set_after_first_call_is_invisible_without_a_cache_clear(
    monkeypatch,
) -> None:
    """Reproduces the exact Colab failure: a stale cache ignores a later os.environ set.

    This is not desired behaviour — it is the trap. The test exists so that if
    ``get_config`` is ever reimplemented without ``cache_clear`` (or without an lru_cache
    at all), this test forces a conscious decision about the notebook contract rather than
    a silent behaviour change.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    first = get_config()
    assert first.hf_token.get_secret_value() == ""

    monkeypatch.setenv("HF_TOKEN", "hf_afterwards")
    second = get_config()

    assert second is first
    assert second.hf_token.get_secret_value() == ""


def test_cache_clear_picks_up_a_newly_set_env_var(monkeypatch) -> None:
    """The documented fix actually works: clear the cache, then re-read the environment.

    This is the exact recovery step given to a user hitting the stale-cache trap, and the
    line every Colab notebook now runs immediately after loading its secrets.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_config()

    monkeypatch.setenv("HF_TOKEN", "hf_afterwards")
    get_config.cache_clear()
    refreshed = get_config()

    assert refreshed.hf_token.get_secret_value() == "hf_afterwards"
    assert refreshed.require_secret("hf_token") == "hf_afterwards"


def test_require_secret_fails_loudly_when_unset(monkeypatch) -> None:
    """An empty secret raises with guidance, rather than sending an empty string to an API."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    get_config.cache_clear()
    config = get_config()

    with pytest.raises(ValueError, match="FRED_API_KEY is not set"):
        config.require_secret("fred_api_key")


def test_secrets_are_masked_in_repr_and_dump(monkeypatch) -> None:
    """A real-looking token never appears in repr() or model_dump()."""
    monkeypatch.setenv("HF_TOKEN", "hf_ThisWouldBeARealToken")
    get_config.cache_clear()
    config = get_config()

    assert "hf_ThisWouldBeARealToken" not in repr(config)
    assert "hf_ThisWouldBeARealToken" not in str(config.model_dump())


def test_config_is_constructible_directly_without_the_cache(monkeypatch) -> None:
    """A fresh Config() always reflects the current environment, cache or no cache.

    The safer pattern for exactly this reason: code that cannot assume anything about
    prior get_config() calls can bypass the singleton entirely.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_config()

    monkeypatch.setenv("HF_TOKEN", "hf_direct")
    assert Config().hf_token.get_secret_value() == "hf_direct"
