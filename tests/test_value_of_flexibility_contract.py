"""Schema, fixture and capability contracts for issue #559."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from voiage.contracts.value_flexibility import VALUE_OF_FLEXIBILITY_INPUT_SCHEMA_V1
from voiage.methods import dynamic_real_options

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/value-of-flexibility/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_value_of_flexibility_normative_fixture_validates() -> None:
    input_schema = _json(CONTRACT / "schemas/value-of-flexibility-input.schema.json")
    result_schema = _json(CONTRACT / "schemas/value-of-flexibility-result.schema.json")
    input_payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    Draft202012Validator(input_schema).validate(input_payload)
    Draft202012Validator(result_schema).validate(expected)
    assert input_schema == VALUE_OF_FLEXIBILITY_INPUT_SCHEMA_V1


def test_result_schema_rejects_materially_incomplete_output() -> None:
    result_schema = _json(CONTRACT / "schemas/value-of-flexibility-result.schema.json")
    incomplete = _json(CONTRACT / "fixtures/normative/expected.json")
    incomplete.pop("policy_path_regret")
    with pytest.raises(Exception, match="policy_path_regret"):
        Draft202012Validator(result_schema).validate(incomplete)


def test_input_schema_requires_explicit_stage_weights() -> None:
    input_schema = _json(CONTRACT / "schemas/value-of-flexibility-input.schema.json")
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload.pop("stage_weights")

    with pytest.raises(Exception, match="stage_weights"):
        Draft202012Validator(input_schema).validate(payload)


@pytest.mark.parametrize(
    "control",
    ["discount_rate", "irreversibility_penalty", "lock_in_penalty"],
)
def test_input_schema_rejects_ungoverned_nonzero_v1_controls(control: str) -> None:
    input_schema = _json(CONTRACT / "schemas/value-of-flexibility-input.schema.json")
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload[control] = 0.01

    with pytest.raises(Exception, match=control):
        Draft202012Validator(input_schema).validate(payload)


@pytest.mark.parametrize(
    "control",
    ["discount_rate", "irreversibility_penalty", "lock_in_penalty"],
)
def test_result_schema_rejects_ungoverned_nonzero_v1_controls(control: str) -> None:
    result_schema = _json(CONTRACT / "schemas/value-of-flexibility-result.schema.json")
    payload = _json(CONTRACT / "fixtures/normative/expected.json")
    payload["diagnostics"][control] = 0.01

    with pytest.raises(Exception, match=control):
        Draft202012Validator(result_schema).validate(payload)


def test_result_schema_requires_axes_provenance_and_unambiguous_change_diagnostic() -> (
    None
):
    result_schema = _json(CONTRACT / "schemas/value-of-flexibility-result.schema.json")
    required = set(result_schema["required"])

    assert {
        "decision_stage_names",
        "strategy_names",
        "provenance",
        "ordered_scenario_policy_changes",
    } <= required
    assert result_schema["properties"]["exercise_decisions"] == {"type": "null"}


def test_capabilities_fail_closed_for_unimplemented_bindings() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "executable"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_manifest_is_single_versioned_normative_reference() -> None:
    manifest = _json(CONTRACT / "fixtures/manifest.json")
    assert manifest["version"] == "v1"
    assert manifest["status"] == "fixture-backed"
    assert manifest["evidence_artifact"] == "evidence.json"
    assert len(manifest["normative"]) == 1
    record = manifest["normative"][0]
    assert record["method_family"] == "value_of_flexibility"
    assert record["tolerance_policy"] == "absolute-1e-12"


def test_fixture_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    expected_paths = {
        "specs/frontier/value-of-flexibility/v1/schemas/value-of-flexibility-input.schema.json",
        "specs/frontier/value-of-flexibility/v1/schemas/value-of-flexibility-result.schema.json",
        "specs/frontier/value-of-flexibility/v1/capabilities.json",
        "specs/frontier/value-of-flexibility/v1/fixtures/manifest.json",
        "specs/frontier/value-of-flexibility/v1/fixtures/normative/input.json",
        "specs/frontier/value-of-flexibility/v1/fixtures/normative/expected.json",
        "conductor/archive/supported_frontier_method_completion_20260723/value-of-flexibility-reference-review.md",
        "conductor/archive/supported_frontier_method_completion_20260723/value-of-flexibility-implementation-review.md",
        "voiage/contracts/value_flexibility.py",
        "voiage/methods/dynamic_real_options.py",
        "voiage/cli.py",
        "tests/test_value_of_flexibility.py",
        "tests/test_value_of_flexibility_contract.py",
        "docs/astro-site/src/content/docs/examples/value-of-flexibility.mdx",
        "specs/frontier/value-of-flexibility/v1/README.md",
    }
    assert {artifact["path"] for artifact in evidence["artifacts"]} == expected_paths
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_runtime_matches_normative_fixture() -> None:
    input_payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    result = dynamic_real_options.value_of_flexibility(
        np.asarray(input_payload["net_benefit"], dtype=float),
        input_payload["decision_stage_names"],
        input_payload["strategy_names"],
        input_payload["stage_weights"],
        input_payload["provenance"],
        discount_rate=input_payload["discount_rate"],
        irreversibility_penalty=input_payload["irreversibility_penalty"],
        lock_in_penalty=input_payload["lock_in_penalty"],
        evidence_arrival_times=input_payload["evidence_arrival_times"],
        flexible_policy_sets=input_payload["flexible_policy_sets"],
        constrained_strategy_names=input_payload["constrained_strategy_names"],
        value_unit=input_payload["value_unit"],
        stage_semantics=input_payload["stage_semantics"],
        information_value_included=input_payload["information_value_included"],
    )
    actual = asdict(result)
    actual["policy_path_regret"] = result.policy_path_regret.tolist()
    for key in ("value_of_flexibility", "option_value"):
        actual_value = actual.pop(key)
        expected_value = expected.pop(key)
        assert actual_value == pytest.approx(expected_value)
    assert actual == expected
