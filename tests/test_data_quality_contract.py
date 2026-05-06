"""Tests for the planned data-quality, privacy, and linkage contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _data_quality_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "data-quality"
        / "v1"
    )


def test_data_quality_contract_schema_and_examples_parse() -> None:
    contract_dir = _data_quality_contract_dir()

    with open(contract_dir / "schemas" / "data-quality-set.schema.json") as f:
        data_quality_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-data-quality-result.schema.json"
    ) as f:
        data_quality_result_schema = json.load(f)
    with open(contract_dir / "examples" / "data-quality-set.example.json") as f:
        data_quality_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-data-quality.example.json"
    ) as f:
        data_quality_result_example = json.load(f)

    Draft202012Validator(data_quality_set_schema).validate(data_quality_set_example)
    Draft202012Validator(data_quality_result_schema).validate(data_quality_result_example)

    assert data_quality_set_schema["title"] == "DataQualitySetV1Planned"
    assert data_quality_result_schema["title"] == "ValueOfDataQualityResultV1Planned"
    assert (
        data_quality_result_example["analysis_type"]
        == "value_of_data_quality_privacy_linkage"
    )
