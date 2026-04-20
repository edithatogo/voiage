from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_core_api_contract as validator


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
                        "name": "missing artifact",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/missing.json",
                        "tolerance_policy": "exact",
                    }
                ],
                "illustrative": [],
            },
            "missing artifact",
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
