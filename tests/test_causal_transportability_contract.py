"""Tests for the causal-identification, transportability, and external-validity contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _causal_transportability_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "causal-transportability"
        / "v1"
    )


def test_causal_transportability_contract_schema_and_examples_parse() -> None:
    contract_dir = _causal_transportability_contract_dir()

    with open(
        contract_dir / "schemas" / "causal-transportability-set.schema.json"
    ) as f:
        causal_set_schema = json.load(f)
    with open(
        contract_dir / "schemas" / "value-of-causal-transportability-result.schema.json"
    ) as f:
        causal_result_schema = json.load(f)
    with open(
        contract_dir / "examples" / "causal-transportability-set.example.json"
    ) as f:
        causal_set_example = json.load(f)
    with open(
        contract_dir / "examples" / "value-of-causal-transportability.example.json"
    ) as f:
        causal_result_example = json.load(f)

    Draft202012Validator(causal_set_schema).validate(causal_set_example)
    Draft202012Validator(causal_result_schema).validate(causal_result_example)

    assert causal_set_schema["title"] == "CausalTransportabilitySetV1FixtureBacked"
    assert (
        causal_result_schema["title"] == "ValueOfCausalTransportabilityResultV1FixtureBacked"
    )
    assert causal_result_example["analysis_type"] == "value_of_causal_transportability"
