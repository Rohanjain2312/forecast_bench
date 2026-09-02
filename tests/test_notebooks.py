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
def test_notebook_has_no_stored_outputs(path) -> None:
    """Outputs are stripped, so a rerun cannot show stale numbers as if they were fresh."""
    for cell in _cells(path):
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
