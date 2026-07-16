"""Tests for the fixture-backed model-validation VOI contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _validation_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "specs" / "frontier" / "validation" / "v1"
    )


def test_validation_contract_schema_and_examples_parse() -> None:
    contract_dir = _validation_contract_dir()

    with open(contract_dir / "schemas" / "validation-set.schema.json") as f:
        validation_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-model-validation-result.schema.json"
    ) as f:
        validation_result_schema = json.load(f)
    with open(contract_dir / "examples" / "validation-set.example.json") as f:
        validation_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-model-validation.example.json"
    ) as f:
        validation_result_example = json.load(f)

    Draft202012Validator(validation_set_schema).validate(validation_set_example)
    Draft202012Validator(validation_result_schema).validate(validation_result_example)

    assert validation_set_schema["title"] == "ValidationSetV1FixtureBacked"
    assert (
        validation_result_schema["title"]
        == "ValueOfModelValidationResultV1FixtureBacked"
    )
    assert validation_result_example["analysis_type"] == "value_of_model_validation"


def test_validation_evidence_manifest_blocks_unverified_promotion() -> None:
    manifest = json.loads(
        (_validation_contract_dir() / "fixtures" / "evidence.json").read_text()
    )
    assert manifest["maturity"] == "fixture-backed"
    assert manifest["stable_claim_allowed"] is False
    assert manifest["parity"]["python"] == "verified"
    assert all(item["sha256"] for item in manifest["artifacts"])
