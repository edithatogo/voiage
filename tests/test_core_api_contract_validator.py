from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_core_api_contract as validator


CORE_API_CONTRACT_PAIRS: tuple[tuple[Path, Path], ...] = (
    (
        Path("specs/core-api/schemas/v1/decision-problem.schema.json"),
        Path("specs/core-api/examples/v1/decision-problem.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/intervention.schema.json"),
        Path("specs/core-api/examples/v1/intervention.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/trial-design.schema.json"),
        Path("specs/core-api/examples/v1/trial-design.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/parameter-set.schema.json"),
        Path("specs/core-api/examples/v1/parameter-set.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/value-array.schema.json"),
        Path("specs/core-api/examples/v1/value-array.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evpi.schema.json"),
        Path("specs/core-api/examples/v1/evpi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evppi.schema.json"),
        Path("specs/core-api/examples/v1/evppi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evsi.schema.json"),
        Path("specs/core-api/examples/v1/evsi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/enbs.schema.json"),
        Path("specs/core-api/examples/v1/enbs.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/ceac.schema.json"),
        Path("specs/core-api/examples/v1/ceac.example.json"),
    ),
)


def test_validate_fixture_manifest_accepts_minimal_v1_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    artifact = fixture_root / "normative" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    manifest = {
        "version": "v1",
        "normative": [
            {
                "name": "basic normative case",
                "method_family": "evpi",
                "expected_output_artifact": "normative/result.json",
                "tolerance_policy": "exact",
                "provenance": {"seed": 101, "execution_mode": "deterministic"},
            }
        ],
        "illustrative": [],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(validator, "FIXTURE_MANIFEST", manifest_path)

    validator._validate_fixture_manifest(manifest_path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            {"version": "v2", "normative": [], "illustrative": []},
            "manifest version",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "missing provenance",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                    }
                ],
                "illustrative": [],
            },
            "provenance",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "missing artifact",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/missing.json",
                        "tolerance_policy": "exact",
                        "provenance": {"seed": 101, "execution_mode": "deterministic"},
                    }
                ],
                "illustrative": [],
            },
            "missing artifact",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "bad provenance seed",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                        "provenance": {
                            "seed": "101",
                            "execution_mode": "deterministic",
                        },
                    }
                ],
                "illustrative": [],
            },
            "seed",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "bad provenance mode",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                        "provenance": {"seed": 101, "execution_mode": "random"},
                    }
                ],
                "illustrative": [],
            },
            "execution_mode",
        ),
    ],
)
def test_validate_fixture_manifest_rejects_invalid_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    message: str,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(validator, "FIXTURE_MANIFEST", manifest_path)

    with pytest.raises(validator.ValidationError, match=message):
        validator._validate_fixture_manifest(manifest_path)


def test_resolve_fixture_artifact_rejects_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)

    with pytest.raises(validator.ValidationError, match="escapes"):
        validator._resolve_fixture_artifact("../escape.json")


@pytest.mark.parametrize(("schema_relpath", "example_relpath"), CORE_API_CONTRACT_PAIRS)
def test_schema_example_pair_conforms_to_contract(
    schema_relpath: Path,
    example_relpath: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / schema_relpath
    example_path = root / example_relpath

    schema = validator._load_json(schema_path)
    example = validator._load_json(example_path)

    validator._validate(example, schema, "$", schema_path)
