"""Drift checks for the retained v1 binding matrix."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
MATRIX_PATH = REPO_ROOT / "specs" / "v1" / "binding-matrix.json"


def test_retained_binding_matrix_covers_supported_surfaces() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    bindings = matrix["bindings"]

    assert {entry["id"] for entry in bindings} == {
        "python",
        "r",
        "julia",
        "typescript",
        "go",
        "rust",
        "dotnet",
    }
    assert all(entry["status"] == "retained" for entry in bindings)
    assert matrix["execution_authority"] == "rust"
    assert all((REPO_ROOT / entry["path"]).exists() for entry in bindings)
    assert all(entry["registry"] and entry["tag_pattern"] for entry in bindings)


def test_binding_matrix_declares_ci_and_external_gate_for_each_surface() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    for entry in matrix["bindings"]:
        assert matrix["ci_workflow"] == ".github/workflows/bindings-ci.yml"
        assert matrix["contract_fixture_root"] == "specs/core-api/fixtures"
        assert entry["external_gate"]
