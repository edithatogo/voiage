"""Tests for the planned computational and model-refinement contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _computational_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "computational"
        / "v1"
    )


def test_computational_contract_schema_and_examples_parse() -> None:
    contract_dir = _computational_contract_dir()

    with open(contract_dir / "schemas" / "computational-set.schema.json") as f:
        computational_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-computational-result.schema.json"
    ) as f:
        computational_result_schema = json.load(f)
    with open(contract_dir / "examples" / "computational-set.example.json") as f:
        computational_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-computational.example.json"
    ) as f:
        computational_result_example = json.load(f)

    Draft202012Validator(computational_set_schema).validate(computational_set_example)
    Draft202012Validator(computational_result_schema).validate(computational_result_example)

    assert computational_set_schema["title"] == "ComputationalSetV1Planned"
    assert computational_result_schema["title"] == "ValueOfComputationalResultV1Planned"
    assert (
        computational_result_example["analysis_type"]
        == "value_of_computational_refinement"
    )
