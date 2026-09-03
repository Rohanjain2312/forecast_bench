"""The Colab notebooks contain no modelling logic and leak no secrets.

The rule these enforce: a notebook installs the package, reads its secrets, calls a
function, and pushes the artifact. If a cell grows a loop over folds, that logic belongs in
``forecast_bench/`` instead. This is what stops the notebook and the repository from
disagreeing, which is the failure mode that kills most benchmark projects.
"""

import json
import re
from pathlib import Path

import pytest

from forecast_bench.config import PROJECT_ROOT

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
COLAB_NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("0[45]_colab_*.ipynb"))


def _cells(path: Path) -> list[dict]:
    """Parse a notebook and return its cells."""
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code_source(path: Path) -> str:
    """Concatenate every code cell's source."""
    return "\n".join(
        "".join(cell["source"]) for cell in _cells(path) if cell["cell_type"] == "code"
    )


def test_colab_notebooks_exist() -> None:
    """Both GPU notebooks are present."""
    names = {path.name for path in COLAB_NOTEBOOKS}
    assert names == {
        "04_colab_finetune_chronos.ipynb",
        "05_colab_train_neural.ipynb",
    }


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json(path) -> None:
    """A notebook that will not parse is a notebook that will not open in Colab."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_loop_over_folds(path) -> None:
    """Fold iteration belongs in the runner, never in a cell."""
    matches = re.findall(r"for\s+\w+\s+in\s+.*fold", _code_source(path), re.IGNORECASE)
    assert not matches, f"{path.name} loops over folds: {matches}"


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_the_package_for_its_heavy_work(path) -> None:
    """Every notebook calls into forecast_bench rather than reimplementing it."""
    imports = set(re.findall(r"from (forecast_bench[\w.]*) import", _code_source(path)))
    assert imports, f"{path.name} never imports forecast_bench"


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_is_introduced_in_plain_language(path) -> None:
    """A markdown cell explains the goal before every code cell."""
    cells = _cells(path)
    orphans = [
        position
        for position, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and (position == 0 or cells[position - 1]["cell_type"] != "markdown")
    ]
    assert not orphans, f"{path.name} has code cells with no explanation: {orphans}"


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_opens_with_its_prerequisites(path) -> None:
    """The first cell says what must already exist on the Hub before the notebook runs."""
    first = "".join(_cells(path)[0]["source"])
    assert _cells(path)[0]["cell_type"] == "markdown"
    assert "Before you run this" in first
    assert "forecastbench-data" in first


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_contains_no_literal_secret(path) -> None:
    """No token is ever pasted into a cell; secrets come from the Colab panel."""
    source = path.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
    assert not re.search(r"\b[0-9a-f]{32}\b", source)
    assert "userdata.get" in _code_source(path)


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_install_cell_forces_a_fresh_reinstall(path) -> None:
    """The package install is idempotent-proof against Colab reusing a live runtime.

    Regression test: without --force-reinstall, pip sees forecast-bench already
    installed (its version number never changes) and silently skips reinstalling. A user
    who reopens the notebook from GitHub and clicks Run All can then keep running stale
    code from an earlier session with no indication anything is wrong -- this happened
    live and cost a full extra fine-tuning cycle before it was diagnosed. See
    docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    source = _code_source(path)
    assert "archive/refs/heads/main.tar.gz" in source
    assert "--force-reinstall" in source, (
        f"{path.name} installs the package without --force-reinstall; a reopened "
        "notebook can silently run stale code on a reused Colab runtime"
    )


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_install_url_is_cache_busted(path) -> None:
    """The install URL differs on every run, so no cache anywhere can serve stale code.

    Regression test: --force-reinstall alone was not enough. The notebook was reopened
    fresh from GitHub twice and both times ran stale, pre-fix code, because the tarball
    URL never changes even though its content does, and something between Colab and
    GitHub (most likely pip's own HTTP cache) kept serving what it had already fetched.
    See docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    source = _code_source(path)
    assert (
        "int(time.time())" in source or "time.time()" in source
    ), f"{path.name} does not appear to cache-bust its install URL"
    assert "_cb=" in source


def test_finetune_notebook_verifies_the_install_before_using_it() -> None:
    """A loud, immediate check that the fix this session needed is actually present.

    Better than trusting the install cell silently: if some cache anywhere still serves
    stale code despite the measures above, this fails right after install with an
    actionable message, instead of a campaign silently pushing nothing 20 minutes later.
    """
    path = NOTEBOOK_DIR / "04_colab_finetune_chronos.ipynb"
    source = _code_source(path)
    assert "inspect.signature(revision_tag)" in source
    assert "assert" in source
    assert "Restart session" in source


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_install_brings_in_dependencies(path) -> None:
    """At least one package install runs *without* --no-deps.

    Regression test for a bug shipped and hit on a genuinely fresh Colab VM: the install
    cell used --no-deps on every line, which had looked harmless only because every prior
    run reused a runtime where the dependencies were already present from an earlier
    install. On a new VM it installed forecast_bench and none of pydantic-settings, darts
    or chronos-forecasting, so the very first import died. See
    docs/planning/PROGRESS_NOTES.md, Step 16.
    """
    install_lines = [
        line
        for line in _code_source(path).splitlines()
        if "pip install" in line and "_tarball_url" in line
    ]
    assert install_lines, f"{path.name} has no package install line"
    assert any("--no-deps" not in line for line in install_lines), (
        f"{path.name} installs the package only with --no-deps, so a fresh VM would "
        "get no dependencies at all"
    )


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_install_also_forces_the_package_itself_current(path) -> None:
    """A --force-reinstall --no-deps line guarantees current code on a reused runtime."""
    install_lines = [
        line
        for line in _code_source(path).splitlines()
        if "pip install" in line and "_tarball_url" in line
    ]
    assert any(
        "--force-reinstall" in line and "--no-deps" in line for line in install_lines
    ), f"{path.name} never forces the package itself to be reinstalled"


def test_finetune_notebook_still_installs_peft_and_torchao() -> None:
    """The LoRA dependencies survive any rewrite of the install cell.

    These were briefly dropped by an automated edit to the install cell; peft is required
    for Chronos-Bolt fine-tuning and torchao must be upgraded or peft will not import.
    """
    source = _code_source(NOTEBOOK_DIR / "04_colab_finetune_chronos.ipynb")
    assert "peft" in source
    assert "torchao" in source


@pytest.mark.parametrize("path", COLAB_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_stored_outputs(path) -> None:
    """Outputs are stripped, so a rerun cannot show stale numbers as if they were fresh."""
    for cell in _cells(path):
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
