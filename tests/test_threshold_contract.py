"""Tests for the fixture-backed threshold, tipping-point, and robust VOI contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _threshold_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "specs" / "frontier" / "threshold" / "v1"
    )


def test_threshold_contract_schema_and_examples_parse() -> None:
    contract_dir = _threshold_contract_dir()

    with open(contract_dir / "schemas" / "threshold-set.schema.json") as f:
        threshold_set_schema = json.load(f)
    with open(contract_dir / "schemas" / "value-of-threshold-result.schema.json") as f:
        threshold_result_schema = json.load(f)
    with open(contract_dir / "examples" / "threshold-set.example.json") as f:
        threshold_set_example = json.load(f)
    with open(contract_dir / "examples" / "value-of-threshold.example.json") as f:
        threshold_result_example = json.load(f)

    Draft202012Validator(threshold_set_schema).validate(threshold_set_example)
    Draft202012Validator(threshold_result_schema).validate(threshold_result_example)

    assert threshold_set_schema["title"] == "ThresholdSetV1FixtureBacked"
    assert threshold_result_schema["title"] == "ValueOfThresholdResultV1FixtureBacked"
    assert threshold_result_example["analysis_type"] == "value_of_threshold_information"


def test_threshold_evidence_manifest_blocks_unverified_promotion() -> None:
    manifest = json.loads(
        (_threshold_contract_dir() / "fixtures" / "evidence.json").read_text()
    )
    assert manifest["maturity"] == "fixture-backed"
    assert manifest["stable_claim_allowed"] is False
    assert manifest["parity"]["python"] == "verified"
    assert all(item["sha256"] for item in manifest["artifacts"])
