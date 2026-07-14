"""Tests for the adjacent frontier shared maturity contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _shared_maturity_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "shared-maturity"
        / "v1"
    )


def test_shared_maturity_contract_schema_and_example_parse() -> None:
    contract_dir = _shared_maturity_contract_dir()

    with open(
        contract_dir / "schemas" / "adjacent-frontier-family-metadata.schema.json"
    ) as f:
        metadata_schema = json.load(f)
    with open(
        contract_dir / "examples" / "adjacent-frontier-family-metadata.example.json"
    ) as f:
        metadata_example = json.load(f)

    Draft202012Validator(metadata_schema).validate(metadata_example)

    assert metadata_schema["title"] == "AdjacentFrontierFamilyMetadataV1"
    assert metadata_example["method_maturity"] == "fixture-backed"
    assert metadata_example["reporting"]["reporting_standard"] == "CHEERS-VOI"
