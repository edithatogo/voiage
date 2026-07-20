from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def test_compatibility_catalog_executes_normal_edge_and_invalid_cases() -> None:
    results = fixture_smoke_validator.execute_compatibility_catalog()

    assert len(results) == 25
    assert {result["status"] for result in results} == {"pass"}
    assert {result["case_id"].split("-", maxsplit=1)[0] for result in results} == {
        "evpi",
        "evppi",
        "evsi",
        "enbs",
        "ceaf",
        "dominance",
    }
    for result in results:
        assert set(result["provenance"]) == {
            "voiage_version",
            "core_version",
            "method",
            "settings",
        }
        assert result["provenance"]["settings"] == {"seed": 101}
        assert json.loads(result["serialized_provenance"]) == result["provenance"]
        assert result["serialized_provenance"] == json.dumps(
            result["provenance"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def test_compatibility_catalog_requires_seed_and_provenance_metadata() -> None:
    manifest = json.loads(
        Path("specs/core-api/fixtures/v1/compatibility-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["provenance"]["reference_implementation"] == "python"
    assert manifest["provenance"]["execution_mode"] == "deterministic"
    assert isinstance(manifest["seed"], int)
    assert manifest["seed"] >= 0


def test_compatibility_catalog_rejects_missing_execution_metadata() -> None:
    with pytest.raises(contract_validator.ValidationError, match="seed"):
        fixture_smoke_validator._validate_manifest_metadata(
            {"provenance": {"reference_implementation": "python"}}
        )


def test_compatibility_outcomes_must_be_deterministic_and_json_serializable() -> None:
    fixture_smoke_validator._assert_deterministic_result(
        {"value": [1.0, 2.0]}, {"value": [1.0, 2.0]}
    )
    fixture_smoke_validator._assert_json_serializable({"value": [1.0, 2.0]})

    with pytest.raises(contract_validator.ValidationError, match="deterministic"):
        fixture_smoke_validator._assert_deterministic_result(1.0, 2.0)
    with pytest.raises(contract_validator.ValidationError, match="JSON-serializable"):
        fixture_smoke_validator._assert_json_serializable({"value": float("nan")})


def test_compatibility_catalog_contains_non_finite_rejection_evidence() -> None:
    manifest = json.loads(
        Path("specs/core-api/fixtures/v1/compatibility-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert {
        case["case_id"]
        for case in manifest["cases"]
        if case.get("evidence") == "non-finite-rejection"
    } == {
        "evpi-nan-invalid-001",
        "evppi-nan-invalid-001",
        "evsi-nan-invalid-001",
        "enbs-nan-invalid-001",
        "ceaf-nan-invalid-001",
        "dominance-nan-invalid-001",
        "dominance-infinity-invalid-001",
    }


def test_runner_reapplies_declared_seed_for_each_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws: list[float] = []

    def execute(_method: str, _inputs: dict[str, object]) -> float:
        draw = float(np.random.random())
        draws.append(draw)
        return draw

    monkeypatch.setattr(fixture_smoke_validator, "_execute_method", execute)
    first = fixture_smoke_validator._execute_seeded("evpi", {}, seed=101)
    second = fixture_smoke_validator._execute_seeded("evpi", {}, seed=101)

    assert first == second
    assert draws == [first, second]


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ([], "non-empty"),
        (
            [{"case_id": "x", "method": "unknown", "classification": "normal"}],
            "unsupported",
        ),
        (
            [
                {"case_id": "x", "method": "evpi", "classification": "normal"},
                {"case_id": "x", "method": "evpi", "classification": "edge"},
            ],
            "unique",
        ),
    ],
)
def test_compatibility_catalog_fails_closed(cases: object, message: str) -> None:
    with pytest.raises(contract_validator.ValidationError, match=message):
        fixture_smoke_validator._validate_case_index(cases)


@pytest.mark.parametrize(
    "artifact", ["../secret.json", "/absolute/secret.json", "case.txt"]
)
def test_compatibility_catalog_rejects_unsafe_artifact_paths(artifact: str) -> None:
    with pytest.raises(contract_validator.ValidationError, match="artifact"):
        fixture_smoke_validator._resolve_artifact(artifact, directory="inputs")


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan"), True, "1"])
def test_compatibility_catalog_rejects_invalid_tolerances(tolerance: object) -> None:
    with pytest.raises(contract_validator.ValidationError, match="finite non-negative"):
        fixture_smoke_validator._read_tolerance(
            {"absolute_tolerance": tolerance}, "absolute_tolerance"
        )
