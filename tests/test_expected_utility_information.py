from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.basic import evpi
from voiage.methods.utility_information import (
    expected_utility_information_value,
    value_of_clairvoyance,
)

FIXTURE = (
    Path(__file__).parents[1]
    / "specs/frontier/expected-utility-information-pricing/v1/fixtures/normative"
)


def _request(name: str) -> dict[str, object]:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))["request"]


def test_python_facade_reproduces_nonlinear_reference() -> None:
    request = _request("log-buy-sell-asymmetry.json")
    result = expected_utility_information_value(request)
    assert result["schema_version"] == "expected-utility-information-result-v1"
    assert result["method_maturity"] == "experimental"
    assert result["bpi"]["value"] == pytest.approx(3.7521886610, abs=1e-7)
    assert result["spi"]["value"] == pytest.approx(3.4085030261, abs=1e-7)
    assert "voc" not in result
    repeated = expected_utility_information_value(request)
    assert json.dumps(result, sort_keys=True) == json.dumps(repeated, sort_keys=True)


def test_voc_presentation_delegates_to_canonical_callable(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def canonical(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return {
            "input_digest": {"algorithm": "rfc8785-sha256-v1", "value": "a" * 64},
            "presentation": {"presentation_label": "voc"},
            "eui": {"value": 2.0},
        }

    monkeypatch.setattr(
        "voiage.methods.utility_information.expected_utility_information_value",
        canonical,
    )
    result = value_of_clairvoyance(
        _request("affine-clairvoyant.json"), selected_measure="bpi"
    )
    assert len(calls) == 1
    assert calls[0]["presentation_label"] == "voc"
    assert result["presentation"]["selected_measure"] == "bpi"
    assert result["presentation"]["presentation_contract_version"] == "1.0.0"
    digest_input = {
        "canonical_input_digest": "a" * 64,
        "presentation_contract_version": "1.0.0",
        "presentation_label": "voc",
        "selected_measure": "bpi",
    }
    encoded = json.dumps(
        digest_input, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert (
        result["presentation"]["presentation_digest"]
        == hashlib.sha256(encoded).hexdigest()
    )
    assert "voc" not in result


def test_power_utility_matches_log_at_and_near_risk_aversion_one() -> None:
    logarithmic = _request("log-buy-sell-asymmetry.json")
    log_result = expected_utility_information_value(logarithmic)
    for risk_aversion in (1.0 - 1.0e-10, 1.0, 1.0 + 1.0e-10):
        power = json.loads(json.dumps(logarithmic))
        power["utility"] = {
            "family": "power",
            "risk_aversion": risk_aversion,
            "reference_wealth": 1.0,
        }
        result = expected_utility_information_value(power)
        assert result["eui"]["value"] == pytest.approx(
            log_result["eui"]["value"], abs=1.0e-9
        )
        assert result["cei"]["value"] == pytest.approx(
            log_result["cei"]["value"], abs=1.0e-8
        )
        assert (
            result["current_policy"]["tie_set"]
            == log_result["current_policy"]["tie_set"]
        )


def test_voc_rejects_finite_signal_presentation() -> None:
    request = _request("affine-clairvoyant.json")
    request["information"]["kind"] = "finite_signal"
    with pytest.raises(Exception, match="clairvoyant"):
        value_of_clairvoyance(request)


@pytest.mark.parametrize(
    ("payload", "selected_measure", "message"),
    [
        ({}, "eui", "information mapping"),
        (_request("affine-clairvoyant.json"), "unknown", "Unsupported VoC"),
    ],
)
def test_voc_rejects_invalid_public_presentation_inputs(
    payload: dict[str, object], selected_measure: str, message: str
) -> None:
    """The public adapter validates malformed selection and metadata inputs."""
    with pytest.raises(InputError, match=message):
        value_of_clairvoyance(payload, selected_measure=selected_measure)


@pytest.mark.parametrize("missing_key", ["affine_reduction", "presentation"])
def test_voc_rejects_incomplete_canonical_results(
    monkeypatch: pytest.MonkeyPatch, missing_key: str
) -> None:
    """A malformed Rust wire response cannot fabricate a presentation."""

    def incomplete(_: dict[str, object]) -> dict[str, object]:
        result: dict[str, object] = {
            "affine_reduction": {"status": "available", "monetary_measure": "evpi"},
            "input_digest": {"algorithm": "rfc8785-sha256-v1", "value": "a" * 64},
            "presentation": {"presentation_label": "voc"},
        }
        del result[missing_key]
        return result

    monkeypatch.setattr(
        "voiage.methods.utility_information.expected_utility_information_value",
        incomplete,
    )
    selected_measure = "evpi" if missing_key == "affine_reduction" else "eui"
    expected = (
        "affine reduction" if missing_key == "affine_reduction" else "presentation"
    )
    with pytest.raises(InputError, match=expected):
        value_of_clairvoyance(
            _request("affine-clairvoyant.json"), selected_measure=selected_measure
        )


def test_decision_analysis_exposes_explicit_state_contract() -> None:
    analysis = DecisionAnalysis(nb_array=np.array([[0.0, 1.0], [1.0, 0.0]]))
    result = analysis.expected_utility_information(_request("affine-clairvoyant.json"))
    assert result["affine_reduction"]["monetary_measure"] == "evpi"


def test_affine_clairvoyance_matches_stable_evpi() -> None:
    request = _request("affine-clairvoyant.json")
    result = expected_utility_information_value(request)
    monetary = evpi(np.asarray(request["payoffs"], dtype=float))
    assert result["affine_reduction"]["value"] == pytest.approx(monetary)


def test_native_result_validates_complete_wire_schema() -> None:
    root = FIXTURE.parents[1]
    schema = json.loads((root / "schemas/result.schema.json").read_text())
    request_schema = json.loads((root / "schemas/request.schema.json").read_text())
    schema["$defs"]["utility"] = request_schema["$defs"]["utility"]
    schema["$defs"]["solver"] = request_schema["$defs"]["solver"]
    schema = json.loads(
        json.dumps(schema)
        .replace("request.schema.json#/$defs/utility", "#/$defs/utility")
        .replace("request.schema.json#/$defs/solver", "#/$defs/solver")
    )
    request = _request("log-buy-sell-asymmetry.json")
    result = expected_utility_information_value(request)
    jsonschema.Draft202012Validator(schema).validate(result)
    assert result["input_digest"]["algorithm"] == "rfc8785-sha256-v1"
    assert len(result["input_digest"]["value"]) == 64
    assert result["backend"]["engine"] == "rust"
    assert result["backend"]["bridge"] == "pyo3"
    assert (
        result["decision_descriptor"]["decision_problem_id"]
        == request["decision_problem_id"]
    )
    assert len(result["informed_policies"]) == len(request["information"]["signal_ids"])
    assert {policy["signal_id"] for policy in result["informed_policies"]} == set(
        request["information"]["signal_ids"]
    )
    root = result["bpi_root"]
    assert root["evaluations"] > 1
    assert root["iterations"] > 0
    assert root["lower"] <= root["estimate"] <= root["upper"]
    assert root["final_bracket_width"] == pytest.approx(root["upper"] - root["lower"])
    assert root["residual"] != 0
    assert any(
        policy["domain_exclusions"]
        for evaluation in root["evaluated_policies"]
        for policy in evaluation["policies"]
    )
    assert all(
        len(evaluation["policies"]) == len(request["information"]["signal_ids"])
        for evaluation in root["evaluated_policies"]
    )
    rules = {
        (rule["scope"], rule["left_measure"], rule["right_measure"]): rule["status"]
        for rule in result["comparability"]["ranking_equivalence"]
    }
    assert rules[("within_problem", "eui", "bpi")] == "not_assured"
    assert rules[("cross_problem", "eui", "cei")] == "not_assured"


def test_native_boundary_rejects_unknown_fields() -> None:
    request = _request("affine-clairvoyant.json")
    request["unknown"] = "rejected"
    with pytest.raises(Exception, match="unknown field"):
        expected_utility_information_value(request)


@pytest.mark.parametrize(
    ("mutate", "diagnostic_code"),
    [
        (lambda request: request.update(schema_version="wrong"), "invalid_input"),
        (
            lambda request: request["information"].update(kind="oracle"),
            "invalid_input",
        ),
        (
            lambda request: request["information"].update(signal_ids=["low", "low"]),
            "invalid_input",
        ),
        (
            lambda request: request.update(action_ids=["same", "same"]),
            "invalid_input",
        ),
        (
            lambda request: request["information"].update(
                signal_state_probabilities=[
                    request["information"]["signal_state_probabilities"][1],
                    request["information"]["signal_state_probabilities"][0],
                ]
            ),
            "invalid_clairvoyance",
        ),
        (
            lambda request: request["information"]["signal_state_probabilities"][
                0
            ].__setitem__(0, -0.1),
            "invalid_probability",
        ),
        (
            lambda request: request["solver"].update(expansion_factor=1),
            "invalid_solver",
        ),
        (lambda request: request.update(price_date=None), "invalid_input"),
        (lambda request: request.update(price_date="31-07-2026"), "invalid_input"),
        (
            lambda request: request.update(presentation_label="clairvoyance"),
            "invalid_input",
        ),
        (
            lambda request: request.update(terminal_outcome_floor=1e6),
            "invalid_ppi_anchor",
        ),
    ],
)
def test_native_boundary_rejects_invalid_contract_states(
    mutate, diagnostic_code
) -> None:
    request = _request("affine-clairvoyant.json")
    mutate(request)
    with pytest.raises(InputError) as caught:
        expected_utility_information_value(request)
    assert caught.value.diagnostic_code == diagnostic_code


def test_voc_evpi_is_only_an_affine_presentation_alias() -> None:
    affine = value_of_clairvoyance(
        _request("affine-clairvoyant.json"), selected_measure="evpi"
    )
    assert affine["presentation"]["selected_measure"] == "evpi"
    assert affine["affine_reduction"]["monetary_measure"] == "evpi"

    with pytest.raises(InputError) as caught:
        value_of_clairvoyance(
            _request("log-buy-sell-asymmetry.json"), selected_measure="evpi"
        )
    assert caught.value.diagnostic_code == "affine_reduction_required"


def test_ppi_unavailability_has_discriminated_diagnostics() -> None:
    missing = _request("affine-clairvoyant.json")
    missing["terminal_outcome_floor"] = None
    result = expected_utility_information_value(missing)
    assert result["ppi"]["diagnostics_ref"] == "ppi_floor_missing"

    zero = _request("zero-value-result.json")
    zero["terminal_outcome_floor"] = 10
    result = expected_utility_information_value(zero)
    assert result["ppi"]["diagnostics_ref"] == "ppi_nonpositive_denominator"
