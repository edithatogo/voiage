"""Tests for the planned preference heterogeneity contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _preference_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "specs" / "frontier" / "preference" / "v1"
    )


def test_preference_contract_schema_and_examples_parse() -> None:
    contract_dir = _preference_contract_dir()

    with open(contract_dir / "schemas" / "preference-set.schema.json") as f:
        preference_set_schema = json.load(f)
    with open(contract_dir / "schemas" / "value-of-preference-result.schema.json") as f:
        preference_result_schema = json.load(f)
    with open(contract_dir / "examples" / "preference-set.example.json") as f:
        preference_set_example = json.load(f)
    with open(contract_dir / "examples" / "value-of-preference.example.json") as f:
        preference_result_example = json.load(f)

    Draft202012Validator(preference_set_schema).validate(preference_set_example)
    Draft202012Validator(preference_result_schema).validate(preference_result_example)

    assert preference_set_schema["title"] == "PreferenceSetV1Planned"
    assert preference_result_schema["title"] == "ValueOfPreferenceResultV1Planned"
    assert (
        preference_result_example["analysis_type"] == "value_of_preference_information"
    )
