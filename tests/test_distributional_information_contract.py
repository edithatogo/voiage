"""Schema and fixture contracts for issue #557."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
import pytest

from voiage.contracts.distributional_information import (
    VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1,
    validate_distributional_information_semantics,
)
from voiage.methods.distributional_information import (
    distributional_information_from_specification,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/value-of-distributional-information/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_normative_fixture_and_installed_schema_validate() -> None:
    input_schema = _json(
        CONTRACT / "schemas/value-of-distributional-information-input.schema.json"
    )
    result_schema = _json(
        CONTRACT / "schemas/value-of-distributional-information-result.schema.json"
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
        CONTRACT / "schemas/value-of-distributional-information-result.schema.json"
    )
    payload = _json(CONTRACT / "fixtures/normative/expected.json")
    payload.pop("resolved_models")
    with pytest.raises(Exception, match="resolved_models"):
        Draft202012Validator(schema).validate(payload)


def test_capabilities_fail_closed_for_unimplemented_bindings() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "executable"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_contract_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    assert "experimental Python evaluator" in evidence["evidence_scope"]
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_runtime_matches_normative_fixture() -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    result = distributional_information_from_specification(payload)
    assert result.to_contract_dict() == expected

    actual = result.to_contract_dict()
    assert [item["model_id"] for item in actual["resolved_models"]] == actual[
        "model_ids"
    ]
    assert [item["probability"] for item in actual["resolved_models"]] == actual[
        "model_probabilities"
    ]
    assert sum(
        item["weighted_contribution"] for item in actual["resolved_models"]
    ) == pytest.approx(actual["expected_resolved_value"])
    assert len(actual["current_expected_values"]) == len(actual["alternative_names"])
    assert (
        actual["estimator"]["input_value_status"]
        == actual["conditional_value_assurance"]["input_status"]
    )
    assert (
        actual["estimator"]["evidence_reference"]
        == actual["conditional_value_assurance"]["evidence_reference"]
    )


def _set_path(
    payload: dict[str, Any], path: tuple[str | int, ...], value: object
) -> None:
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_ids",), [], "model_ids"),
        (("model_ids",), "family-a", "model_ids"),
        (("model_ids",), ["family-a", "family-a"], "unique"),
        (("alternative_names",), [], "alternative_names"),
        (("alternative_names",), "A", "alternative_names"),
        (("alternative_names",), ["A", "A"], "unique"),
        (("model_labels",), [], "model_labels"),
        (("model_definitions",), [], "model_definitions"),
        (("model_definitions", 0), [], "entries must be objects"),
        (("model_definitions", 0), {"model_id": "family-a"}, "complete"),
        (
            ("model_definitions", 0, "parameterization"),
            " ",
            "complete",
        ),
        (("conditional_value_assurance",), {}, "assurance is incomplete"),
        (
            ("conditional_value_assurance", "source_uncertainty"),
            "estimated",
            "exact enumerated",
        ),
        (
            ("conditional_value_assurance", "enumeration_method"),
            " ",
            "exact enumerated",
        ),
        (("comparability",), {}, "complete verified contract"),
        (("comparability", "horizon_id"), " ", "explicitly verified"),
        (("model_probabilities",), [1.0], "align"),
        (("model_probabilities",), ["half", 0.5], "finite"),
        (("model_probabilities",), [-0.1, 1.1], "non-negative"),
        (("tolerances",), [], "tolerances must be an object"),
        (("tolerances", "probability_sum"), "small", "must be numeric"),
        (("tolerances", "probability_sum"), 0.0, "finite, positive"),
        (("tolerances", "probability_sum"), 1.0, "at most 1e-6"),
        (("model_probabilities",), [0.4, 0.4], "sum to 1"),
        (("conditional_values",), [[1.0, 2.0]], "rows must align"),
        (("conditional_values", 0), [1.0], "row must align"),
        (("conditional_values", 0), [1.0, "bad"], "finite numbers"),
        (("information_cost",), "free", "information_cost must be finite"),
    ],
)
def test_semantic_validator_rejects_every_structural_boundary(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    _set_path(payload, path, value)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_distributional_information_semantics(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "9.9.9"),
        ("analysis_type", "not_vdi"),
        ("method_maturity", "stable"),
        ("information_target", "all_parameters"),
        ("conditioning_order", "optimize_before_integrating"),
    ],
)
def test_public_specification_adapter_enforces_versioned_constants(
    field: str, value: object
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload[field] = value
    with pytest.raises(ValidationError):
        distributional_information_from_specification(payload)


def test_public_specification_adapter_rejects_unknown_fields() -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload["unreviewed_extension"] = True
    with pytest.raises(ValidationError, match="Additional properties"):
        distributional_information_from_specification(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("conditional_value_assurance", "source_values_exact"), False),
        (("conditional_value_assurance", "input_status"), "monte_carlo_estimate"),
        (("comparability", "verified"), False),
    ],
)
def test_schema_rejects_false_exactness_and_unverified_comparability(
    path: tuple[str, str], value: object
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload[path[0]][path[1]] = value
    with pytest.raises(ValidationError):
        distributional_information_from_specification(payload)
