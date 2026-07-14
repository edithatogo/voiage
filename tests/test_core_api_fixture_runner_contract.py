from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_core_api_contract as contract_validator
from scripts import validate_core_api_fixtures as fixture_smoke_validator


def test_runner_contract_doc_covers_language_and_ci_patterns() -> None:
    document = Path("specs/core-api/fixtures/v1/runner.md").read_text(encoding="utf-8")

    for needle in (
        "Python, R, Julia, TypeScript, Go, Rust, and .NET",
        "exact",
        "Future Binding CI Patterns",
        "npm pack --dry-run",
        "go test ./...",
        "cargo package --locked --allow-dirty",
        "uv run python scripts/validate_core_api_fixtures.py",
    ):
        assert needle in document


def test_fixture_smoke_validator_accepts_layout_only_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    input_artifact = fixture_root / "normative" / "inputs" / "input.json"
    input_artifact.parent.mkdir(parents=True)
    input_artifact.write_text("{}", encoding="utf-8")

    output_artifact = fixture_root / "normative" / "result.json"
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    output_artifact.write_text("{}", encoding="utf-8")

    manifest = {
        "version": "v1",
        "normative": [
            {
                "name": "basic normative case",
                "method_family": "evpi",
                "input_artifact": "normative/inputs/input.json",
                "expected_output_artifact": "normative/result.json",
                "tolerance_policy": "exact",
                "provenance": {"seed": 101, "execution_mode": "deterministic"},
            }
        ],
        "illustrative": [],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(contract_validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(contract_validator, "FIXTURE_MANIFEST", manifest_path)
    monkeypatch.setattr(fixture_smoke_validator.validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(
        fixture_smoke_validator.validator, "FIXTURE_MANIFEST", manifest_path
    )

    assert fixture_smoke_validator.main() == 0
    captured = capsys.readouterr()
    assert "validated" in captured.out
    assert "manifest.json" in captured.out


def test_fixture_smoke_validator_rejects_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    manifest = {
        "version": "v1",
        "normative": [
            {
                "name": "missing artifact",
                "method_family": "evpi",
                "input_artifact": "normative/inputs/input.json",
                "expected_output_artifact": "normative/result.json",
                "tolerance_policy": "exact",
                "provenance": {"seed": 101, "execution_mode": "deterministic"},
            }
        ],
        "illustrative": [],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(contract_validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(contract_validator, "FIXTURE_MANIFEST", manifest_path)
    monkeypatch.setattr(fixture_smoke_validator.validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(
        fixture_smoke_validator.validator, "FIXTURE_MANIFEST", manifest_path
    )

    with pytest.raises(contract_validator.ValidationError, match="missing artifact"):
        fixture_smoke_validator.validator.validate_fixture_catalog_layout(manifest_path)
