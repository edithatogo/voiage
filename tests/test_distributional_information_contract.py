"""Schema and fixture contracts for issue #557."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.distributional_information import (
    VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1,
    validate_distributional_information_semantics,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/value-of-distributional-information/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_normative_fixture_and_installed_schema_validate() -> None:
    input_schema = _json(
        CONTRACT
        / "schemas/value-of-distributional-information-input.schema.json"
    )
    result_schema = _json(
        CONTRACT
        / "schemas/value-of-distributional-information-result.schema.json"
    )
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    Draft202012Validator.check_schema(input_schema)
    Draft202012Validator.check_schema(result_schema)
    Draft202012Validator(input_schema).validate(payload)
    Draft202012Validator(result_schema).validate(expected)
    validate_distributional_information_semantics(payload)
    assert input_schema == VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_probabilities", [0.4, 0.4], "sum to 1"),
        ("model_probabilities", [float("nan"), 1.0], "finite"),
        ("conditional_values", [[10.0], [4.0, 12.0]], "align"),
        ("model_labels", {"family-a": "Only one"}, "exactly match"),
    ],
)
def test_semantic_validator_fails_closed(
    field: str, value: object, message: str
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_distributional_information_semantics(payload)


def test_result_schema_requires_complete_policy_and_assurance() -> None:
    schema = _json(
        CONTRACT
        / "schemas/value-of-distributional-information-result.schema.json"
    )
    payload = _json(CONTRACT / "fixtures/normative/expected.json")
    payload.pop("resolved_models")
    with pytest.raises(Exception, match="resolved_models"):
        Draft202012Validator(schema).validate(payload)


def test_capabilities_fail_closed_before_runtime_delivery() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "planned"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_contract_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    assert "runtime is not yet claimed" in evidence["evidence_scope"]
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_runtime_fixture_conformance_is_intentionally_red_until_f557_3() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("voiage.methods.distributional_information")
