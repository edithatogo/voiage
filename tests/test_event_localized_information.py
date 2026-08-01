"""Exact-contract tests for event-localized VOI and information density."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingModuleSource=false, reportUnknownLambdaType=false
# pyright: reportUnusedCallResult=false

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import matplotlib.pyplot as plt
import pytest
from typer.testing import CliRunner

import voiage
from voiage import methods
from voiage.cli import app
from voiage.contracts.event_localized_information import (
    EVENT_LOCALIZED_INFORMATION_INPUT_SCHEMA_V1,
    EVENT_LOCALIZED_INFORMATION_RESULT_SCHEMA_V1,
    validate_event_localized_information_result_semantics,
)
from voiage.exceptions import InputError, PlottingError
from voiage.methods.event_localized_information import (
    event_localized_information_value,
)
from voiage.plot.event_localized_information import (
    plot_event_accuracy_curve,
    plot_information_density,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/event-localized-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _input() -> dict[str, Any]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def _result(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return event_localized_information_value(payload or _input()).to_contract_dict()


def test_normative_exact_event_density_and_binary_channel() -> None:
    result = _result()
    assert result == json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert result["baseline"] == {
        "action_expected_values": {"safe": 4.0, "targeted": 4.9},
        "optimal_actions": ["targeted"],
        "reference_action": "targeted",
        "reference_value": 4.9,
    }
    event = result["event"]
    assert event["probability"] == pytest.approx(0.5)
    assert event["complement_probability"] == pytest.approx(0.5)
    assert event["perfect_gross_voi"] == pytest.approx(1.1)
    assert event["perfect_net_voi"] == pytest.approx(1.0)
    curve = {row["accuracy"]: row for row in event["imperfect_binary_channel"]}
    assert curve[0.5]["gross_voi"] == pytest.approx(0)
    assert curve[0.25]["gross_voi"] == pytest.approx(0.325)
    assert curve[0.75]["gross_voi"] == pytest.approx(0.325)
    assert curve[0.0]["gross_voi"] == pytest.approx(1.1)
    assert curve[1.0]["gross_voi"] == pytest.approx(1.1)


def test_hazen_policy_relative_and_centered_density_integrals() -> None:
    density = _result()["density"]
    assert density["information_value"] == pytest.approx(1.1)
    assert density["policy_relative_integral"] == pytest.approx(1.1)
    assert density["centered_integral"] == pytest.approx(1.1)
    assert density["integral_errors"] == {
        "policy_relative": pytest.approx(0),
        "centered": pytest.approx(0),
    }
    atoms = {tuple(row["coordinate"]): row for row in density["atoms"]}
    assert atoms[(-1.0, 0.0)]["policy_relative_density"] == pytest.approx(0.8)
    assert atoms[(0.0, 0.0)]["policy_relative_density"] == pytest.approx(0.3)
    assert atoms[(1.0, 1.0)]["policy_relative_density"] == pytest.approx(0)
    assert atoms[(-1.0, 0.0)]["centered_density"] == pytest.approx(-0.18)
    assert atoms[(0.0, 0.0)]["centered_density"] == pytest.approx(-0.27)
    assert atoms[(1.0, 1.0)]["centered_density"] == pytest.approx(1.55)
    assert density["modes"] == [[-1.0, 0.0]]
    assert density["directions_from_base"] == [[-2.0, -1.0]]


def test_complete_ties_reference_policy_and_grouped_coordinates() -> None:
    payload = _input()
    payload["states"][0]["coordinate"] = [0.0, 0.0]
    payload["states"][0]["action_values"] = {"safe": 4.0, "targeted": 4.0}
    payload["states"][1]["action_values"] = {"safe": 4.0, "targeted": 4.0}
    payload["states"][2]["action_values"] = {"safe": 4.0, "targeted": 4.0}
    payload["density"]["reference_action"] = "safe"
    result = _result(payload)
    assert result["baseline"]["optimal_actions"] == ["safe", "targeted"]
    assert len(result["density"]["atoms"]) == 2
    assert all(
        atom["optimal_actions"] == ["safe", "targeted"]
        for atom in result["density"]["atoms"]
    )
    assert result["density"]["information_value"] == pytest.approx(0)
    assert result["density"]["modes"] == []
    assert result["density"]["directions_from_base"] == []


def test_state_and_action_permutations_preserve_estimands() -> None:
    forward = _result()
    payload = _input()
    payload["states"].reverse()
    payload["actions"].reverse()
    reverse = _result(payload)
    assert reverse["event"]["perfect_gross_voi"] == pytest.approx(
        forward["event"]["perfect_gross_voi"]
    )
    assert reverse["density"]["information_value"] == pytest.approx(
        forward["density"]["information_value"]
    )


@pytest.mark.parametrize(
    ("operator", "threshold", "state_ids"),
    [
        ("less_than", 0.5, ["adverse", "borderline"]),
        ("less_than_or_equal", 0.0, ["adverse", "borderline"]),
        ("greater_than", 0.0, ["favourable"]),
        ("greater_than_or_equal", 0.5, ["favourable"]),
    ],
)
def test_every_threshold_operator_resolves_the_declared_partition(
    operator: str, threshold: float, state_ids: list[str]
) -> None:
    payload = _input()
    payload["event"]["definition"].update(operator=operator, threshold=threshold)
    assert _result(payload)["event"]["state_ids"] == state_ids


def test_explicit_state_set_matches_threshold_event() -> None:
    threshold = _result()
    payload = _input()
    payload["event"]["definition"] = {
        "kind": "state_set",
        "state_ids": ["borderline", "adverse"],
    }
    explicit = _result(payload)
    assert explicit["event"]["state_ids"] == ["adverse", "borderline"]
    assert explicit["event"]["perfect_gross_voi"] == pytest.approx(
        threshold["event"]["perfect_gross_voi"]
    )


def test_sparse_accuracy_grid_retains_explicit_unknown_assurance() -> None:
    payload = _input()
    payload["event"]["accuracy_grid"] = [0.2]
    result = _result(payload)
    assert result["assurance"]["accuracy_half_no_information_residual"] is None
    assert result["assurance"]["maximum_binary_channel_symmetry_error"] is None


def test_decimal_symmetric_accuracy_pair_is_matched_with_float_tolerance() -> None:
    payload = _input()
    payload["event"]["accuracy_grid"] = [0.07, 0.93]
    result = _result(payload)
    assert result["assurance"][
        "maximum_binary_channel_symmetry_error"
    ] == pytest.approx(0.0, abs=payload["integral_tolerance"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version="v2"), "schema_version"),
        (lambda value: value.update(direction="minimize"), "maximize"),
        (lambda value: value.update(unexpected=True), "keys must match"),
        (lambda value: value["states"][0].update(probability=0.4), "sum to one"),
        (lambda value: value["states"][0].update(probability=0), "positive"),
        (lambda value: value["states"][0].update(coordinate=[0.0]), "dimension"),
        (lambda value: value["states"][0]["action_values"].pop("safe"), "actions"),
        (
            lambda value: value["density"].update(reference_action="missing"),
            "reference",
        ),
        (lambda value: value["event"].update(accuracy_grid=[-0.1]), "accuracy"),
        (lambda value: value["event"].update(accuracy_grid=[]), "non-empty"),
        (lambda value: value["event"].update(accuracy_grid=[0.5, 0.5]), "unique"),
        (lambda value: value["event"].update(information_cost=-1), "cost"),
        (
            lambda value: value["event"]["definition"].update(coordinate_index=9),
            "coordinate_index",
        ),
    ],
)
def test_invalid_contracts_fail_closed(mutation: object, match: str) -> None:
    payload = deepcopy(_input())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        event_localized_information_value(payload)


def test_reference_action_must_be_baseline_optimal() -> None:
    payload = _input()
    payload["density"]["reference_action"] = "safe"
    with pytest.raises(InputError, match="baseline-optimal"):
        event_localized_information_value(payload)


def test_tolerance_tie_cannot_replace_the_true_baseline_optimizer() -> None:
    payload = _input()
    payload["tie_tolerance"] = 1e-6
    payload["states"][0]["action_values"]["safe"] += 0.8999995 / 0.2
    payload["density"]["reference_action"] = "safe"
    with pytest.raises(InputError, match="exactly baseline-optimal"):
        event_localized_information_value(payload)


def test_event_and_complement_must_both_have_positive_probability() -> None:
    payload = _input()
    payload["event"]["definition"]["threshold"] = 99
    with pytest.raises(InputError, match="event and complement"):
        event_localized_information_value(payload)


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        ([], "must be an object"),
        ({"kind": "state_set", "state_ids": []}, "non-empty"),
        ({"kind": "state_set", "state_ids": ["missing"]}, "unknown state"),
        ({"kind": "state_set", "state_ids": ["adverse"], "extra": 1}, "keys"),
        (
            {
                "kind": "threshold",
                "coordinate_index": True,
                "operator": "less_than",
                "threshold": 0,
            },
            "coordinate_index",
        ),
        (
            {
                "kind": "threshold",
                "coordinate_index": 0,
                "operator": "equal",
                "threshold": 0,
            },
            "operator",
        ),
        ({"kind": "unsupported"}, "strict threshold or state_set"),
    ],
)
def test_malformed_event_definitions_fail_closed(
    definition: object, match: str
) -> None:
    payload = _input()
    payload["event"]["definition"] = definition
    with pytest.raises(InputError, match=match):
        event_localized_information_value(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(analysis_id=""), "analysis_id"),
        (lambda value: value.update(value_unit=""), "value_unit"),
        (lambda value: value.update(analysis_type="evpi"), "analysis_type"),
        (lambda value: value.update(chronology=[]), "chronology"),
        (lambda value: value.update(actions="bad"), "actions must be an array"),
        (lambda value: value.update(actions=["one"]), "at least two"),
        (lambda value: value.update(actions=["a", "a"]), "unique"),
        (lambda value: value.update(actions=["a", ""]), "non-empty"),
        (lambda value: value.update(states=[]), "states must be a non-empty"),
        (lambda value: value["states"][0].update(state_id=""), "state_id"),
        (
            lambda value: value["states"][1].update(
                state_id=value["states"][0]["state_id"]
            ),
            "identifiers must be unique",
        ),
        (lambda value: value["states"][0].update(extra=1), "state keys"),
        (
            lambda value: value["states"][0].update(coordinate=[float("inf"), 0]),
            "finite",
        ),
        (
            lambda value: value["states"][0]["action_values"].update(safe=float("nan")),
            "finite",
        ),
        (lambda value: value["density"].update(measure="lebesgue"), "probability_mass"),
        (lambda value: value["density"].update(coordinate_names=[]), "non-empty"),
        (lambda value: value["density"].update(coordinate_names=["x", "x"]), "unique"),
        (lambda value: value["density"].update(coordinate_units=["u"]), "dimension"),
        (lambda value: value["density"].update(base_coordinate=[0]), "dimension"),
        (lambda value: value.update(tie_tolerance=-1), r"\[0, 1e-6\]"),
        (lambda value: value.update(tie_tolerance=1e-5), "1e-6"),
        (lambda value: value.update(integral_tolerance=0), r"\(0, 1e-6\]"),
        (lambda value: value.update(integral_tolerance=1e-5), "1e-6"),
        (lambda value: value["event"].update(information_cost=True), "schema"),
        (
            lambda value: value["event"].update(
                definition={
                    "kind": "state_set",
                    "state_ids": ["adverse", "adverse"],
                }
            ),
            "schema",
        ),
        (lambda value: value["provenance"].update(event_source=""), "non-empty"),
        (lambda value: value["provenance"].update(extra="bad"), "keys"),
    ],
)
def test_additional_strict_contract_failures(mutation: object, match: str) -> None:
    payload = deepcopy(_input())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        event_localized_information_value(payload)


def test_strict_schemas_and_capability_dispositions() -> None:
    input_schema = json.loads((CONTRACT / "schemas/input.schema.json").read_text())
    result_schema = json.loads((CONTRACT / "schemas/result.schema.json").read_text())
    assert input_schema == EVENT_LOCALIZED_INFORMATION_INPUT_SCHEMA_V1
    assert result_schema == EVENT_LOCALIZED_INFORMATION_RESULT_SCHEMA_V1
    Draft202012Validator(input_schema).validate(_input())
    Draft202012Validator(result_schema).validate(_result())
    invalid = _input()
    invalid["states"][0]["unexpected"] = True
    assert list(Draft202012Validator(input_schema).iter_errors(invalid))
    invalid_result = _result()
    invalid_result["event"]["unexpected"] = True
    assert list(Draft202012Validator(result_schema).iter_errors(invalid_result))
    capabilities = json.loads((CONTRACT / "capabilities.json").read_text())
    assert capabilities["languages"] == {
        "Python": "experimental_runtime",
        "Rust": "not_implemented",
        "R": "not_implemented",
        "Julia": "not_implemented",
        "Mojo": "external_upstream_boundary",
    }
    assert "BPI" in capabilities["delegated"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["baseline"].update(reference_value=0), "baseline"),
        (lambda value: value["event"].update(probability=0.6), "partition"),
        (lambda value: value["event"].update(perfect_gross_voi=99), "gross VOI"),
        (
            lambda value: value["event"]["imperfect_binary_channel"][0][
                "signal_probabilities"
            ].update(event_reported=0.9),
            "signal probabilities",
        ),
        (
            lambda value: value["density"]["atoms"][0].update(
                policy_relative_density=99
            ),
            "policy-relative density",
        ),
        (
            lambda value: value["density"]["atoms"][0].update(
                optimal_actions=["targeted"]
            ),
            "ties",
        ),
        (lambda value: value["density"].update(modes=[]), "modes"),
        (
            lambda value: value["assurance"].update(
                maximum_binary_channel_symmetry_error=None
            ),
            "symmetry result is missing",
        ),
        (lambda value: value.update(unexpected=True), "schema violation"),
    ],
)
def test_result_semantic_validator_rejects_mutations(
    mutation: object, match: str
) -> None:
    result = _result()
    mutation(result)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        validate_event_localized_information_result_semantics(result)


def test_cli_api_exports_and_deterministic_copy(tmp_path: Path) -> None:
    assert voiage.event_localized_information_value is event_localized_information_value
    assert (
        methods.event_localized_information_value is event_localized_information_value
    )
    result = event_localized_information_value(_input())
    first = result.to_contract_dict()
    first["analysis_id"] = "mutated"
    assert result.to_contract_dict()["analysis_id"] != "mutated"
    cli = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-event-localized-information", str(INPUT)],
    )
    assert cli.exit_code == 0, cli.output
    assert (
        json.loads(cli.stdout)["analysis_type"]
        == "event_localized_information_value_result"
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    failed = CliRunner().invoke(
        app, ["calculate-event-localized-information", str(invalid)]
    )
    assert failed.exit_code == 1
    assert "must be a JSON object" in failed.stderr


def test_plots_consume_result_only_and_label_units() -> None:
    result = event_localized_information_value(_input())
    _figure, supplied_ax = plt.subplots()
    accuracy_ax = plot_event_accuracy_curve(result, ax=supplied_ax, linewidth=2)
    assert accuracy_ax.get_xlabel() == "Symmetric binary-channel accuracy"
    assert "utility points" in accuracy_ax.get_ylabel()
    assert len(accuracy_ax.lines) == 3
    density_ax = plot_information_density(result)
    assert density_ax.get_xlabel() == "margin (score)"
    assert density_ax.get_ylabel() == "trajectory (score)"
    assert "utility points" in density_ax.figure.axes[-1].get_ylabel()
    assert len(density_ax.collections) >= 1


def test_univariate_density_plot_and_dimension_guard() -> None:
    payload = _input()
    payload["density"]["coordinate_names"] = ["margin"]
    payload["density"]["coordinate_units"] = ["score"]
    payload["density"]["base_coordinate"] = [1.0]
    for state in payload["states"]:
        state["coordinate"] = state["coordinate"][:1]
    ax = plot_information_density(event_localized_information_value(payload))
    assert len(ax.lines) >= 1
    three = _input()
    three["density"]["coordinate_names"].append("third")
    three["density"]["coordinate_units"].append("score")
    three["density"]["base_coordinate"].append(0.0)
    for state in three["states"]:
        state["coordinate"].append(0.0)
    with pytest.raises(ValueError, match="one or two dimensions"):
        plot_information_density(event_localized_information_value(three))


def test_plotting_dependency_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from voiage.plot import event_localized_information as plotting

    monkeypatch.setattr(plotting, "MATPLOTLIB_AVAILABLE", False)
    result = event_localized_information_value(_input())
    with pytest.raises(PlottingError, match="Matplotlib"):
        plotting.plot_event_accuracy_curve(result)
