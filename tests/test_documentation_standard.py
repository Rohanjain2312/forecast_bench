"""The documentation standard from IMPLEMENTATION_PLAN.md section 7, made executable.

Two things are checked, and both are conventions that decay silently without a test:

1. Every module has a docstring, every public function in library code has one with type
   hints, and every public function declares a return type.
2. The six specific assumption-docstrings exist. Each one records a fact that, if
   violated, invalidates the study without any metric revealing it — so their absence is
   a real regression, not a style lapse.
"""

import ast
import re
from pathlib import Path

import pytest

from forecast_bench.config import PROJECT_ROOT

#: Directories held to the full standard. Tests use pytest fixtures, which conventionally
#: go unannotated, so they are checked for module docstrings only.
LIBRARY_ROOTS = ["forecast_bench", "scripts", "space"]

ALL_ROOTS = LIBRARY_ROOTS + ["tests"]


def _python_files(root: str) -> list[Path]:
    """Every Python file under a project directory."""
    return sorted((PROJECT_ROOT / root).rglob("*.py"))


def _public_defs(tree: ast.AST):
    """Yield public function and class nodes, skipping private helpers."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        yield node


@pytest.mark.parametrize("root", ALL_ROOTS)
def test_every_module_has_a_docstring(root) -> None:
    """A file with no docstring gives a reader nothing to orient on."""
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _python_files(root)
        if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not missing, f"modules without a docstring: {missing}"


@pytest.mark.parametrize("root", LIBRARY_ROOTS)
def test_public_definitions_are_documented(root) -> None:
    """Every public class and function in library code carries a docstring."""
    missing = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _public_defs(tree):
            if not ast.get_docstring(node):
                missing.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {node.name}"
                )
    assert not missing, f"undocumented public definitions: {missing}"


@pytest.mark.parametrize("root", LIBRARY_ROOTS)
def test_public_functions_are_fully_annotated(root) -> None:
    """Arguments and return values in library code carry type hints."""
    missing = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _public_defs(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            where = f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {node.name}"
            unannotated = [
                a.arg
                for a in node.args.args
                if a.arg not in ("self", "cls") and a.annotation is None
            ]
            if unannotated:
                missing.append(f"{where} args {unannotated}")
            if node.returns is None and node.name != "__init__":
                missing.append(f"{where} has no return annotation")
    assert not missing, f"incomplete type hints: {missing}"


#: The six assumptions IMPLEMENTATION_PLAN.md section 7 requires to be stated in a
#: docstring, as (file, phrases that must appear). Each records something that silently
#: invalidates the study if violated.
REQUIRED_ASSUMPTIONS = [
    (
        "forecast_bench/data/targets.py",
        ["split- and dividend-adjusted", "spurious volatility jump"],
    ),
    (
        "forecast_bench/data/covariates.py",
        ["Only non-revised daily FRED series", "reference period", "release date"],
    ),
    (
        "forecast_bench/backtest/protocol.py",
        ["must not close over", "no metric will reveal"],
    ),
    (
        "forecast_bench/evaluation/metrics.py",
        ["training window only", "most common way MASE is reported wrongly"],
    ),
    (
        "forecast_bench/evaluation/stats.py",
        ["non-overlapping forecast windows", "stride == max"],
    ),
    (
        "forecast_bench/models/foundation/chronos2.py",
        ["pre-October-2025", "contaminated by pretraining", "docs/limitations.md"],
    ),
]


@pytest.mark.parametrize(
    ("path", "phrases"),
    REQUIRED_ASSUMPTIONS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_required_assumption_docstring_is_present(path, phrases) -> None:
    """The six assumption-docstrings exist and still say what they must.

    Whitespace is collapsed before matching, so a phrase wrapped across two lines by the
    formatter still counts — the requirement is that the statement is present, not that it
    fits on one line.
    """
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    collapsed = re.sub(r"\s+", " ", text)
    missing = [p for p in phrases if re.sub(r"\s+", " ", p) not in collapsed]
    assert not missing, f"{path} no longer states: {missing}"


# --- The documents themselves -----------------------------------------------------------

#: Documents IMPLEMENTATION_PLAN.md section 7 and REPO_STRUCTURE.md require to exist.
REQUIRED_DOCS = [
    "README.md",
    "PREREGISTRATION.md",
    "CONTRIBUTING.md",
    "docs/index.md",
    "docs/quickstart.md",
    "docs/architecture.md",
    "docs/methodology.md",
    "docs/data_protocol.md",
    "docs/benchmark_results.md",
    "docs/limitations.md",
    "docs/model_cards.md",
    "forecast_bench/data/README.md",
    "forecast_bench/backtest/README.md",
    "forecast_bench/models/README.md",
    "forecast_bench/evaluation/README.md",
]


@pytest.mark.parametrize("relative", REQUIRED_DOCS)
def test_required_document_exists_and_is_not_a_stub(relative) -> None:
    """Every planned document exists and has real content in it."""
    path = PROJECT_ROOT / relative
    assert path.is_file(), f"{relative} is missing"
    assert len(path.read_text(encoding="utf-8")) > 400, f"{relative} looks like a stub"


def test_no_broken_relative_links_in_the_documentation() -> None:
    """A link to a file that does not exist is worse than no link.

    Relative targets only — external URLs are not checked, since that would make the suite
    depend on the network and on other people's uptime.
    """
    broken = []
    documents = [
        PROJECT_ROOT / "README.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
        *sorted(PROJECT_ROOT.glob("forecast_bench/*/README.md")),
    ]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
            if target.startswith(("http", "#", "mailto")):
                continue
            resolved = (document.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                broken.append(f"{document.name}: [{label}]({target})")
    assert not broken, f"broken relative links: {broken}"


def test_readme_states_the_verdict_without_softening_it() -> None:
    """PREREGISTRATION.md section 3 commits to the word 'lost', not a euphemism.

    It also requires the counter-evidence to appear beside it: reporting only the headline
    would be the same failure in the other direction.
    """
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "lost" in readme.lower()
    assert "mixed results" not in readme.lower()
    assert "85%" in readme, "the sample-efficiency counter-evidence is missing"
