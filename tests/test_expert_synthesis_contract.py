"""Tests for the planned expert-elicitation and evidence-synthesis contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _expert_synthesis_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "expert-synthesis"
        / "v1"
    )


def test_expert_synthesis_contract_schema_and_examples_parse() -> None:
    contract_dir = _expert_synthesis_contract_dir()

    with open(contract_dir / "schemas" / "expert-synthesis-set.schema.json") as f:
        expert_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-expert-synthesis-result.schema.json"
    ) as f:
        expert_result_schema = json.load(f)
    with open(contract_dir / "examples" / "expert-synthesis-set.example.json") as f:
        expert_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-expert-synthesis.example.json"
    ) as f:
        expert_result_example = json.load(f)

    Draft202012Validator(expert_set_schema).validate(expert_set_example)
    Draft202012Validator(expert_result_schema).validate(expert_result_example)

    assert expert_set_schema["title"] == "ExpertSynthesisSetV1FixtureBacked"
    assert (
        expert_result_schema["title"] == "ValueOfExpertSynthesisResultV1FixtureBacked"
    )
    assert expert_result_example["analysis_type"] == "value_of_expert_synthesis"
