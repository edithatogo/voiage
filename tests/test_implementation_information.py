"""Contract and numerical tests for joint implementation-information value."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.exceptions import InputError
from voiage.methods.implementation_information import implementation_information_value

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/implementation-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _input() -> dict[str, object]:
    return json.loads(INPUT.read_text())


def _result(payload: dict[str, object] | None = None) -> dict[str, object]:
    return implementation_information_value(payload or _input()).to_contract_dict()


def test_portable_schemas_and_normative_fixture() -> None:
    input_schema = json.loads(
        (CONTRACT / "schemas/implementation-information-input.schema.json").read_text()
    )
    result_schema = json.loads(
        (CONTRACT / "schemas/implementation-information-result.schema.json").read_text()
    )
    Draft202012Validator(input_schema).validate(_input())
    result = _result()
    Draft202012Validator(result_schema).validate(result)
    assert result == json.loads(EXPECTED.read_text())


def test_four_cell_decomposition_and_interaction_identity() -> None:
    result = _result()
    matrix = result["matrix"]
    assert isinstance(matrix, dict)
    assert matrix["current_information_current_implementation"][
        "aggregate_value"
    ] == pytest.approx(12000)
    assert matrix["perfect_information_current_implementation"][
        "aggregate_value"
    ] == pytest.approx(12400)
    assert matrix["current_information_perfect_implementation"][
        "aggregate_value"
    ] == pytest.approx(12000)
    assert matrix["perfect_information_perfect_implementation"][
        "aggregate_value"
    ] == pytest.approx(14000)

    gross = result["gross_components"]
    assert isinstance(gross, dict)
    assert gross["realizable_evpi"] == pytest.approx(400)
    assert gross["evpim"] == pytest.approx(0)
    assert gross["evp"] == pytest.approx(2000)
    assert gross["interaction"] == pytest.approx(1600)
    assert gross["evp"] == pytest.approx(
        gross["realizable_evpi"] + gross["evpim"] + gross["interaction"]
    )
    assert result["identity_residuals"] == {
        "evp_equals_realizable_evpi_plus_evpim_plus_interaction": pytest.approx(0),
        "perfect_implementation_evpi_equals_evp_minus_evpim": pytest.approx(0),
    }


def test_specific_implementation_and_signal_dependent_ia_evsi() -> None:
    result = _result()
    gross = result["gross_components"]
    net = result["net_components"]
    assert isinstance(gross, dict)
    assert isinstance(net, dict)
    assert gross["evsim"] == pytest.approx(400)
    assert net["evsim"] == pytest.approx(350)
    assert gross["ia_evsi"] == pytest.approx(1000)
    assert net["ia_evsi"] == pytest.approx(925)
    assurance = result["assurance"]
    assert isinstance(assurance, dict)
    assert assurance["implementation_information_independence_assumed"] is False
    assert assurance["state_dependent_current_implementation_observed"] is True
    assert assurance["signal_dependent_post_sample_implementation_supported"] is True


def test_zero_uptake_edge_case_is_enumerated_without_independence() -> None:
    payload = _input()
    current = payload["current_implementation"]
    assert isinstance(current, dict)
    for state in current.values():
        assert isinstance(state, dict)
        state["new"] = {"standard": 1, "new": 0}
    payload.pop("sampling_model")
    result = _result(payload)
    matrix = result["matrix"]
    assert isinstance(matrix, dict)
    assert matrix["current_information_current_implementation"][
        "aggregate_value"
    ] == pytest.approx(10000)
    assert result["gross_components"]["evpim"] == pytest.approx(2000)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["states"][0].update(probability=0.7), "sum to one"),
        (lambda value: value.update(population=0), "population"),
        (
            lambda value: value["current_implementation"]["low"]["new"].update(new=0.3),
            "sum to one",
        ),
        (lambda value: value["states"][0]["net_benefit"].pop("new"), "all actions"),
        (
            lambda value: value["sampling_model"]["signals"][0][
                "likelihood_by_state"
            ].pop("high"),
            "every state",
        ),
    ],
)
def test_invalid_contracts_fail_closed(mutation: object, match: str) -> None:
    payload = deepcopy(_input())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        implementation_information_value(payload)


def test_public_api_and_cli(tmp_path: Path) -> None:
    assert voiage.implementation_information_value is implementation_information_value
    output = tmp_path / "result.json"
    invocation = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-implementation-information",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    cli_payload = json.loads(invocation.output)
    assert cli_payload["analysis_type"] == "implementation_information_decomposition"
    assert json.loads(output.read_text()) == json.loads(EXPECTED.read_text())
