"""Tests for the fixture-backed preference heterogeneity contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np

from voiage.methods.preference import (
    PreferenceProfile,
    PreferenceProfileSet,
    value_of_preference,
)


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

    assert preference_set_schema["title"] == "PreferenceSetV1FixtureBacked"
    assert preference_result_schema["title"] == "ValueOfPreferenceResultV1FixtureBacked"
    assert (
        preference_result_example["analysis_type"] == "value_of_preference_information"
    )


def test_preference_contract_result_matches_normative_fixture() -> None:
    """The normative preference payload should match the current contract shape."""
    contract_dir = _preference_contract_dir()

    with open(
        contract_dir / "fixtures" / "normative" / "value-of-preference.json"
    ) as f:
        normative_example = json.load(f)

    result = value_of_preference(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
                [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            ]
        ),
        preference_profiles=PreferenceProfileSet(
            [
                PreferenceProfile(
                    id="access_first",
                    label="Access first",
                    weight=0.25,
                ),
                PreferenceProfile(
                    id="outcomes_first",
                    label="Outcomes first",
                    weight=0.75,
                ),
            ]
        ),
        strategy_names=["A", "B", "C"],
        preference_profile_weights={"access_first": 0.25, "outcomes_first": 0.75},
        reference_preference_profile="access_first",
        analysis_id="preference-screening-001",
        decision_problem_id="screening-program-001",
    )

    assert result.to_dict() == normative_example
