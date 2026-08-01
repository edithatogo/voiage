"""Portable contract tests for issue #560 finite additive MCDA information."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

import voiage.contracts.mcda_information as mcda_contract

MCDA_INFORMATION_INPUT_SCHEMA_V1 = mcda_contract.MCDA_INFORMATION_INPUT_SCHEMA_V1
MCDA_INFORMATION_RESULT_SCHEMA_V1 = mcda_contract.MCDA_INFORMATION_RESULT_SCHEMA_V1
validate_mcda_information_result_semantics = (
    mcda_contract.validate_mcda_information_result_semantics
)
validate_mcda_information_semantics = mcda_contract.validate_mcda_information_semantics

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/mcda-information/v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input() -> dict[str, Any]:
    return _json(CONTRACT / "fixtures/normative/input.json")


def _expected() -> dict[str, Any]:
    return _json(CONTRACT / "fixtures/normative/expected.json")


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.strip("/").split("/")
    target: Any = payload
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value


def test_portable_schemas_equal_installed_contracts_and_validate_fixtures() -> None:
    input_schema = _json(CONTRACT / "schemas/mcda-information-input.schema.json")
    result_schema = _json(CONTRACT / "schemas/mcda-information-result.schema.json")
    Draft202012Validator.check_schema(input_schema)
    Draft202012Validator.check_schema(result_schema)
    assert input_schema == MCDA_INFORMATION_INPUT_SCHEMA_V1
    assert result_schema == MCDA_INFORMATION_RESULT_SCHEMA_V1
    Draft202012Validator(input_schema).validate(_input())
    Draft202012Validator(result_schema).validate(_expected())
    validate_mcda_information_semantics(_input())
    validate_mcda_information_result_semantics(_expected())


def test_normative_fixture_retains_joint_interaction_without_double_counting() -> None:
    result = _expected()
    actions = {item["action_type"]: item for item in result["conditional_actions"]}
    assert actions["criterion"]["gross_voi"] == 0.0
    assert actions["preference"]["gross_voi"] == 0.0
    assert actions["joint"]["gross_voi"] == pytest.approx(0.028)
    assert result["decomposition"]["interaction"] == pytest.approx(0.028)
    assert result["decomposition"]["no_double_counting_identity_residual"] == 0.0
    assert actions["joint"]["net_voi"] == pytest.approx(0.018)
    assert result["regret"]["baseline_expected"] == pytest.approx(0.028)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["joint_states"][0].update(probability=0.25),
            "probabilities must sum to 1",
        ),
        (
            lambda payload: payload["default_weights"].update(quality=0.8),
            "weights must sum to 1",
        ),
        (
            lambda payload: payload["alternatives"][1].update(
                alternative_id="service-a"
            ),
            "alternative IDs must be unique",
        ),
        (
            lambda payload: payload["default_weights"].update(
                unknown=payload["default_weights"].pop("burden")
            ),
            "weight keys must exactly match criterion IDs",
        ),
        (
            lambda payload: payload["joint_states"][0]["performances"].update(
                {"service-x": {"quality": 50.0, "burden": 50.0}}
            ),
            "exactly match alternatives",
        ),
        (
            lambda payload: payload["joint_states"][0]["partition_values"].pop(
                "preference_regime"
            ),
            "define every partition key",
        ),
        (
            lambda payload: payload["criteria"][0]["value_function"][
                "anchors"
            ].reverse(),
            "anchors and domain must increase",
        ),
        (
            lambda payload: payload["criteria"][0]["value_function"].update(
                valid_domain=[10.0, 100.0]
            ),
            "anchors must lie inside the valid domain",
        ),
        (
            lambda payload: payload["criteria"][0]["value_function"]["anchors"][
                1
            ].update(value=0.0),
            "follow criterion direction",
        ),
        (
            lambda payload: payload["joint_states"][0]["performances"][
                "service-a"
            ].update(quality=101.0),
            "outside a reject-extrapolation domain",
        ),
        (
            lambda payload: payload["latent_partitions"]["preference_keys"].append(
                "outcome_regime"
            ),
            "partition keys must be disjoint",
        ),
        (
            lambda payload: payload["joint_states"][0]["performances"][
                "service-a"
            ].update(
                unknown=payload["joint_states"][0]["performances"]["service-a"].pop(
                    "burden"
                )
            ),
            "performance rows must exactly match criteria",
        ),
        (
            lambda payload: payload["information_actions"][0].update(
                outcome_partition_keys=["unknown"]
            ),
            "unknown partition key",
        ),
        (
            lambda payload: payload["information_actions"][0].update(
                preference_partition_keys=["preference_regime"]
            ),
            "keys must match its declared type",
        ),
        (
            lambda payload: payload["information_actions"][0].update(
                action_type="preference"
            ),
            "exactly one criterion, preference and joint",
        ),
        (
            lambda payload: payload["joint_states"][0].pop("weights"),
            "state-specific weights",
        ),
    ],
)
def test_input_semantics_fail_closed(mutation, message: str) -> None:
    payload = _input()
    if "state-specific" in message:
        for state in payload["joint_states"]:
            state.pop("weights", None)
    else:
        mutation(payload)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_mcda_information_semantics(payload)


def test_v1_requires_exactly_one_action_of_each_information_type() -> None:
    payload = _input()
    payload["information_actions"].append(deepcopy(payload["information_actions"][0]))
    payload["information_actions"][-1]["action_id"] = "learn-outcome-again"
    with pytest.raises(ValueError, match="constraint: maxItems"):
        validate_mcda_information_semantics(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["default_weights"].update(quality="not-a-number"),
            "weights must be finite",
        ),
        (
            lambda payload: payload["default_weights"].update(quality=-0.1, burden=1.1),
            "weights must be non-negative",
        ),
        (
            lambda payload: payload["tolerances"].update(probability_sum=0.0),
            "probability tolerance must be",
        ),
        (
            lambda payload: payload["tolerances"].update(weight_sum=0.0),
            "weight tolerance must be",
        ),
        (
            lambda payload: payload["joint_states"][0].update(probability=-0.1),
            "state probabilities must be non-negative",
        ),
        (
            lambda payload: payload["information_actions"][0]["cost"].update(
                original_amount=-1.0
            ),
            "information costs must be non-negative",
        ),
    ],
)
def test_semantics_defend_numeric_boundaries_beyond_schema(
    monkeypatch, mutation, message: str
) -> None:
    monkeypatch.setattr(
        mcda_contract.Draft202012Validator,
        "validate",
        lambda self, instance: None,
    )
    payload = _input()
    mutation(payload)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_mcda_information_semantics(payload)


def test_joint_action_must_be_the_exact_declared_refinement() -> None:
    payload = _input()
    payload["latent_partitions"]["outcome_keys"].append("secondary_outcome")
    for state in payload["joint_states"]:
        state["partition_values"]["secondary_outcome"] = "constant"
    payload["information_actions"][2]["outcome_partition_keys"] = ["secondary_outcome"]
    with pytest.raises(ValueError, match="joint action must exactly refine"):
        validate_mcda_information_semantics(payload)


def test_zero_mass_joint_states_are_rejected_before_conditioning() -> None:
    payload = _input()
    payload["joint_states"][0]["probability"] += payload["joint_states"][1][
        "probability"
    ]
    payload["joint_states"][1]["probability"] = 0.0
    with pytest.raises(ValueError, match="exclusiveMinimum"):
        validate_mcda_information_semantics(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda result: result["conditional_actions"][2].update(gross_voi=0.03),
            "gross VOI",
        ),
        (
            lambda result: result["conditional_actions"][2].update(net_voi=0.02),
            "net VOI",
        ),
        (
            lambda result: result["decomposition"].update(interaction=0.0),
            "decomposition",
        ),
        (
            lambda result: result["rank_acceptability"]["by_alternative"].update(
                {"service-a": [0.2, 0.2]}
            ),
            "rank acceptability",
        ),
        (
            lambda result: result["conditional_actions"][0]["partitions"][0].update(
                probability=0.25
            ),
            "conditional partition probabilities",
        ),
        (
            lambda result: result["conditional_actions"][0].update(action_type="joint"),
            "exactly one criterion, preference and joint",
        ),
        (
            lambda result: result["baseline"]["expected_scores"].update(
                {"service-a": float("nan")}
            ),
            "baseline score must be finite",
        ),
        (
            lambda result: result["baseline"]["expected_scores"].update(
                {"service-x": result["baseline"]["expected_scores"].pop("service-a")}
            ),
            "baseline scores must exactly match alternatives",
        ),
        (
            lambda result: result["conditional_actions"][0]["partitions"][0][
                "conditional_scores"
            ].update(
                {
                    "service-x": result["conditional_actions"][0]["partitions"][0][
                        "conditional_scores"
                    ].pop("service-a")
                }
            ),
            "conditional scores must exactly match alternatives",
        ),
        (
            lambda result: result["decomposition"].update(
                criterion_action_id="unknown-action"
            ),
            "action IDs must identify result actions",
        ),
        (
            lambda result: result["decomposition"].update(
                criterion_action_id="learn-preference",
                preference_action_id="learn-outcome",
            ),
            "action IDs must match their action types",
        ),
        (
            lambda result: result["rank_acceptability"]["by_alternative"].update(
                {
                    "service-x": result["rank_acceptability"]["by_alternative"].pop(
                        "service-a"
                    )
                }
            ),
            "rank acceptability must exactly match alternatives",
        ),
    ],
)
def test_result_semantics_fail_closed(mutation, message: str) -> None:
    result = _expected()
    mutation(result)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_mcda_information_result_semantics(result)


def test_result_schema_errors_are_redacted_to_path_and_constraint() -> None:
    result = _expected()
    result.pop("unsupported_dispositions")
    with pytest.raises(ValueError, match="invalid MCDA information result.*required"):
        validate_mcda_information_result_semantics(result)


def test_result_rejects_action_id_reconciliation_drift(monkeypatch) -> None:
    original_ids = mcda_contract._ids

    def ids_with_ghost(
        records: list[dict[str, Any]], key: str, label: str
    ) -> list[str]:
        return [*original_ids(records, key, label), "ghost-action"]

    monkeypatch.setattr(mcda_contract, "_ids", ids_with_ghost)
    with pytest.raises(ValueError, match="result action IDs must be unique"):
        validate_mcda_information_result_semantics(_expected())


@pytest.mark.parametrize(
    "case_name",
    [
        "probability-sum",
        "negative-weight",
        "unknown-partition",
        "post-information-normalization",
    ],
)
def test_committed_pathology_fixtures_fail_closed(case_name: str) -> None:
    case = _json(CONTRACT / f"fixtures/cases/{case_name}.json")
    payload = _input()
    for operation in case["operations"]:
        assert operation["op"] == "replace"
        _set_path(payload, operation["path"], operation["value"])
    with pytest.raises(ValueError, match=case["expected_error"]):
        validate_mcda_information_semantics(payload)


def test_capabilities_and_result_fail_closed_for_undelivered_language_surfaces() -> (
    None
):
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    assert capabilities["surfaces"]["python"]["status"] == "executable"
    assert capabilities["surfaces"]["rust"]["status"] == "unsupported"
    assert capabilities["surfaces"]["r"]["status"] == "unsupported"
    assert capabilities["surfaces"]["julia"]["status"] == "unsupported"
    assert capabilities["surfaces"]["mojo"]["status"] == "external"
    assert _expected()["language_dispositions"] == {
        "python": "executable",
        "rust": "unsupported",
        "r": "unsupported",
        "julia": "unsupported",
        "mojo": "external",
    }
    assert "imperfect or sample information EVSI" in capabilities["unsupported"]


def test_contract_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    assert evidence["execution_status"] == "experimental_python"
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
