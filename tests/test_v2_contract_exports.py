"""Deterministic JSON Schema and example export tests for C13."""

from __future__ import annotations

from pathlib import Path

from scripts.export_v2_contracts import contract_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_v2_contract_artifacts_are_deterministic_and_current() -> None:
    first = contract_artifacts()
    second = contract_artifacts()
    assert first == second
    assert first
    for relative_path, expected in first.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.read_text(encoding="utf-8") == expected


def test_v2_exports_include_analysis_and_perspective_result_contracts() -> None:
    artifacts = contract_artifacts()
    assert "specs/core-api/schemas/v2/analysis-spec.schema.json" in artifacts
    assert "specs/core-api/schemas/v2/perspective-result.schema.json" in artifacts
    assert "specs/core-api/examples/v2/analysis-spec.example.json" in artifacts
    assert "specs/core-api/examples/v2/perspective-result.example.json" in artifacts
