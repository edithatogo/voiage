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


def test_dynamic_real_options_fixture_manifest_and_payload_are_deterministic() -> None:
    """The deterministic fixture set should anchor the planned contract."""
    fixture_root = _dynamic_real_options_contract_dir() / "fixtures"
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["status"] == "fixture-backed"

    normative = manifest["normative"]
    assert len(normative) == 1
    entry = normative[0]
    assert entry["name"] == "staged evidence dynamic real-options comparison"
    assert entry["method_family"] == "value_of_dynamic_real_options"
    assert entry["input_artifact"] == "normative/dynamic-real-options-set.json"
    assert (
        entry["expected_output_artifact"]
        == "normative/value-of-dynamic-real-options.json"
    )
    assert entry["tolerance_policy"] == "exact"
    assert entry["provenance"] == {
        "seed": 303,
        "execution_mode": "deterministic",
    }

    input_artifact = fixture_root / entry["input_artifact"]
    output_artifact = fixture_root / entry["expected_output_artifact"]
    assert input_artifact.is_file()
    assert output_artifact.is_file()

    expected = json.loads(output_artifact.read_text())
    assert expected["analysis_type"] == "value_of_dynamic_real_options"
