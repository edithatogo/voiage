"""Tests for the planned dynamic real-options VOI contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _dynamic_real_options_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "dynamic-real-options"
        / "v1"
    )


def test_dynamic_real_options_contract_schema_and_examples_parse() -> None:
    contract_dir = _dynamic_real_options_contract_dir()

    with open(contract_dir / "schemas" / "dynamic-real-options-set.schema.json") as f:
        options_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-dynamic-real-options-result.schema.json"
    ) as f:
        options_result_schema = json.load(f)
    with open(contract_dir / "examples" / "dynamic-real-options-set.example.json") as f:
        options_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-dynamic-real-options.example.json"
    ) as f:
        options_result_example = json.load(f)

    Draft202012Validator(options_set_schema).validate(options_set_example)
    Draft202012Validator(options_result_schema).validate(options_result_example)

    assert options_set_schema["title"] == "DynamicRealOptionsSetV1Planned"
    assert options_result_schema["title"] == "ValueOfDynamicRealOptionsResultV1Planned"
    assert options_result_example["analysis_type"] == "value_of_dynamic_real_options"
