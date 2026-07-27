"""Contracts for the repository quality-tool consolidation policy."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "specs/quality/tool-consolidation.json"


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_bandit_is_consolidated_into_selected_ruff_security_rules() -> None:
    """Bandit must not return while Ruff owns the active source-security policy."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    selected = pyproject["tool"]["ruff"]["lint"]["select"]

    assert "S" in selected
    for path in (
        "pyproject.toml",
        "tox.ini",
        "noxfile.py",
        "pixi.toml",
        ".github/workflows/ci.yml",
    ):
        content = (ROOT / path).read_text(encoding="utf-8").lower()
        assert "bandit" not in content, f"duplicate Bandit control returned in {path}"


def test_tomli_backport_is_not_declared_or_imported() -> None:
    """Python 3.12+ provides the TOML reader used by repository tooling."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [
        *pyproject["project"]["dependencies"],
        *[
            dependency
            for dependencies in pyproject["project"]["optional-dependencies"].values()
            for dependency in dependencies
        ],
    ]

    assert not any(dependency.lower().startswith("tomli") for dependency in declared)
    for path in (
        "final_validation.py",
        "tests/test_extension_packaging_policy.py",
        "tests/test_python_rust_bridge.py",
        "tests/test_rust_workspace_contract.py",
        "tests/packaging/test_dependency_minimization.py",
    ):
        assert "tomli" not in (ROOT / path).read_text(encoding="utf-8")


def test_safety_and_pip_tools_do_not_return_to_the_tox_policy() -> None:
    """Dependency scanning must inspect the canonical locked resolution."""
    tox = (ROOT / "tox.ini").read_text(encoding="utf-8").lower()
    dispositions = {
        entry["tool"]: entry["disposition"] for entry in _registry()["tools"]
    }

    assert "[testenv:safety]" not in tox
    assert "pip-tools" not in tox
    assert dispositions["safety-and-pip-tools"] == "consolidated-into-pip-audit"


def test_quality_tool_dispositions_are_unique_and_evidence_backed() -> None:
    """Every retained or consolidated tool has a unique, checked disposition."""
    payload = _registry()
    tools = payload["tools"]
    assert isinstance(tools, list)
    names = [entry["tool"] for entry in tools]

    assert len(names) == len(set(names))
    dispositions = {entry["tool"]: entry["disposition"] for entry in tools}
    assert dispositions["bandit"] == "consolidated-into-ruff"
    for entry in tools:
        assert entry["contract"]
        evidence = entry["evidence"]
        assert evidence
        for path in evidence:
            if path.startswith("The ") or path.startswith("Ruff "):
                continue
            assert (ROOT / path).exists(), f"missing evidence path: {path}"
