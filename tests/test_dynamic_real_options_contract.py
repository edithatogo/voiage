"""Tests for the planned dynamic real-options VOI contract scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from voiage.methods.dynamic_real_options import value_of_dynamic_real_options


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

    assert options_set_schema["title"] == "DynamicRealOptionsSetV1FixtureBacked"
    assert (
        options_result_schema["title"]
        == "ValueOfDynamicRealOptionsResultV1FixtureBacked"
    )
    assert options_result_example["analysis_type"] == "value_of_dynamic_real_options"
    assert options_result_example["reporting"]["method_maturity"] == "fixture-backed"


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
    assert entry["tolerance_policy"] == "absolute-1e-12"
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


def test_dynamic_real_options_runtime_matches_normative_fixture() -> None:
    fixture_root = _dynamic_real_options_contract_dir() / "fixtures/normative"
    payload = json.loads((fixture_root / "dynamic-real-options-set.json").read_text())
    expected = json.loads(
        (fixture_root / "value-of-dynamic-real-options.json").read_text()
    )
    result = value_of_dynamic_real_options(
        np.asarray(payload["net_benefit"], dtype=float),
        payload["decision_stage_names"],
        payload["strategy_names"],
        payload["stage_weights"],
        payload["discount_rate"],
        payload["irreversibility_penalty"],
        payload["lock_in_penalty"],
        payload["evidence_arrival_times"],
    )
    np.testing.assert_allclose(
        result.expected_net_benefits, expected["expected_net_benefits"], atol=1e-12
    )
    assert result.optimal_strategy_names == expected["optimal_strategy_names"]
    assert result.waiting_value == pytest.approx(expected["waiting_value"])
    assert result.option_value == pytest.approx(expected["option_value"])
    np.testing.assert_allclose(
        result.policy_path_regret, expected["policy_path_regret"], atol=1e-12
    )
    np.testing.assert_allclose(
        result.timing_sensitivity, expected["timing_sensitivity"], atol=1e-12
    )
    assert result.robust_strategy_name == expected["robust_strategy_name"]
    assert result.pareto_strategy_names == expected["pareto_strategy_names"]
    assert result.reporting == expected["reporting"]
