"""Contract tests for outcome-conditional sample-information value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingModuleSource=false, reportUnknownLambdaType=false
# pyright: reportUnusedCallResult=false
# pyright: reportPrivateUsage=false

from copy import deepcopy
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

from voiage import methods
from voiage.cli import app
from voiage.contracts.outcome_conditional_sample_information import (
    OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1,
    OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1,
    validate_outcome_conditional_sample_information_result_semantics,
)
from voiage.exceptions import InputError
from voiage.methods import outcome_conditional_sample_information as implementation
from voiage.methods.outcome_conditional_sample_information import (
    outcome_conditional_sample_information_value,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/outcome-conditional-sample-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _input() -> dict[str, Any]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def _result(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return outcome_conditional_sample_information_value(
        payload or _input()
    ).to_contract_dict()


def test_normative_fixture_matches_exact_conditional_metrics() -> None:
    result = _result()
    assert result == json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert result["baseline"]["reference_expected_value"] == pytest.approx(5.0)
    rows = {row["outcome_id"]: row for row in result["outcomes"]}
    assert rows["adverse_signal"]["delta_ev"] == pytest.approx(-1.0)
    assert rows["adverse_signal"]["vsi"] == pytest.approx(2.0)
    assert rows["favourable_signal"]["delta_ev"] == pytest.approx(3.0)
    assert rows["favourable_signal"]["vsi"] == pytest.approx(0.0)
    assert result["aggregate"]["evsi"] == pytest.approx(1.0)
    assert result["aggregate"]["expected_delta_ev"] == pytest.approx(1.0)
    assert result["aggregate"]["sigma_vsi"] == pytest.approx(1.0)
    assert result["aggregate"]["net_evsi"] == pytest.approx(0.75)


def test_equation_10_is_weighted_population_sd_not_unweighted_sample_sd() -> None:
    payload = _input()
    payload["outcomes"][0]["likelihood_by_state"] = {
        "favourable": 0.1,
        "unfavourable": 0.7,
    }
    payload["outcomes"][1]["likelihood_by_state"] = {
        "favourable": 0.9,
        "unfavourable": 0.3,
    }
    result = _result(payload)
    rows = result["outcomes"]
    evsi = sum(row["predictive_probability"] * row["vsi"] for row in rows)
    expected_variance = sum(
        row["predictive_probability"] * (row["vsi"] - evsi) ** 2 for row in rows
    )
    unweighted_sample_sd = math.sqrt(
        sum((row["vsi"] - sum(item["vsi"] for item in rows) / 2) ** 2 for row in rows)
    )
    assert result["aggregate"]["sigma_vsi"] == pytest.approx(
        math.sqrt(expected_variance)
    )
    assert result["aggregate"]["sigma_vsi"] != pytest.approx(unweighted_sample_sd)
    assert result["assurance"]["ddof"] == 0


def test_splitting_equivalent_outcome_preserves_weighted_distribution_metrics() -> None:
    baseline = _result()
    payload = _input()
    favourable = payload["outcomes"].pop()
    payload["outcomes"].extend(
        [
            {
                "outcome_id": "favourable_signal_a",
                "likelihood_by_state": {
                    state: probability / 2
                    for state, probability in favourable["likelihood_by_state"].items()
                },
            },
            {
                "outcome_id": "favourable_signal_b",
                "likelihood_by_state": {
                    state: probability / 2
                    for state, probability in favourable["likelihood_by_state"].items()
                },
            },
        ]
    )
    split = _result(payload)
    for field in ("evsi", "variance_vsi", "sigma_vsi"):
        assert split["aggregate"][field] == pytest.approx(baseline["aggregate"][field])
    assert (
        split["aggregate"]["low_value_risks"]
        == baseline["aggregate"]["low_value_risks"]
    )


def test_tower_identity_is_expectation_only_not_dispersion_identity() -> None:
    result = _result()
    rows = result["outcomes"]
    delta_mean = sum(row["predictive_probability"] * row["delta_ev"] for row in rows)
    delta_sigma = math.sqrt(
        sum(
            row["predictive_probability"] * (row["delta_ev"] - delta_mean) ** 2
            for row in rows
        )
    )
    assert delta_mean == pytest.approx(result["aggregate"]["evsi"])
    assert delta_sigma == pytest.approx(2.0)
    assert delta_sigma != pytest.approx(result["aggregate"]["sigma_vsi"])
    assert result["assurance"]["tower_identity_scope"] == "expectations_only"


def test_rvsi_zero_is_not_policy_switch_or_tie_set_change_mass_under_ties() -> None:
    payload = _input()
    payload["states"] = [
        {
            "state_id": "left",
            "probability": 0.5,
            "action_values": {"adaptive": 0.0, "status_quo": 10.0},
        },
        {
            "state_id": "right",
            "probability": 0.5,
            "action_values": {"adaptive": 10.0, "status_quo": 0.0},
        },
    ]
    payload["outcomes"] = [
        {
            "outcome_id": "left",
            "likelihood_by_state": {"left": 1.0, "right": 0.0},
        },
        {
            "outcome_id": "right",
            "likelihood_by_state": {"left": 0.0, "right": 1.0},
        },
    ]
    result = _result(payload)
    assert result["baseline"]["optimal_actions"] == ["adaptive", "status_quo"]
    risks = {
        row["delta"]: row["probability"]
        for row in result["aggregate"]["low_value_risks"]
    }
    assert risks[0.0] == pytest.approx(0.5)
    assert result["aggregate"][
        "reference_action_excluded_probability"
    ] == pytest.approx(0.5)
    assert result["aggregate"]["mandatory_policy_switch_probability"] == pytest.approx(
        0.0
    )
    assert result["aggregate"]["complete_tie_set_changed_probability"] == pytest.approx(
        1.0
    )


def test_loss_minimization_is_direction_aware() -> None:
    payload = _input()
    payload["objective"] = {"measure": "loss", "direction": "minimize"}
    for state in payload["states"]:
        state["action_values"] = {
            action: -value for action, value in state["action_values"].items()
        }
    result = _result(payload)
    assert result["aggregate"]["evsi"] == pytest.approx(1.0)
    assert result["aggregate"]["sigma_vsi"] == pytest.approx(1.0)
    assert result["outcomes"][0]["delta_ev"] == pytest.approx(-1.0)


@pytest.mark.parametrize("scale", [1e-12, 1e-9, 1e-6, 1e6])
def test_value_unit_scaling_preserves_all_value_functionals(scale: float) -> None:
    baseline = _result()
    payload = _input()
    for state in payload["states"]:
        state["action_values"] = {
            action: value * scale for action, value in state["action_values"].items()
        }
    payload["information_cost"] *= scale
    payload["low_value_thresholds"] = [
        threshold * scale for threshold in payload["low_value_thresholds"]
    ]
    payload["tie_tolerance"] *= scale
    scaled = _result(payload)

    for field in (
        "evsi",
        "expected_delta_ev",
        "information_cost",
        "net_evsi",
        "sigma_vsi",
        "minimum_vsi",
        "maximum_vsi",
    ):
        assert math.isclose(
            scaled["aggregate"][field],
            baseline["aggregate"][field] * scale,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    assert math.isclose(
        scaled["aggregate"]["variance_vsi"],
        baseline["aggregate"]["variance_vsi"] * scale**2,
        rel_tol=1e-12,
        abs_tol=0.0,
    )
    assert scaled["aggregate"]["sigma_vsi"] > 0.0


def test_tie_tolerance_uses_declared_value_unit_without_artificial_cap() -> None:
    payload = _input()
    payload["tie_tolerance"] = 2.0
    result = _result(payload)
    assert result["baseline"]["optimal_actions"] == ["adaptive", "status_quo"]


@pytest.mark.parametrize("residual", [-1e-6, 1e-6])
def test_probability_vectors_are_normalized_within_tolerance(residual: float) -> None:
    payload = _input()
    payload["probability_tolerance"] = 1e-6
    payload["states"][1]["probability"] += residual
    payload["outcomes"][0]["likelihood_by_state"]["favourable"] += residual

    result = _result(payload)

    assert math.fsum(row["predictive_probability"] for row in result["outcomes"]) == (
        pytest.approx(1.0)
    )
    assert result["assurance"]["evsi_delta_ev_residual"] == 0.0
    assert result["assurance"]["predictive_probability_residual"] == 0.0
    assert result["assurance"]["prior_probability_residual"] == pytest.approx(
        abs(residual)
    )
    assert result["assurance"]["maximum_likelihood_row_residual"] == (
        pytest.approx(abs(residual))
    )
    assert result["assurance"]["probability_normalization_applied"] is True
    assert result["input_assurance"]["input_contract"] == payload


@pytest.mark.parametrize("target", ["prior", "likelihood"])
@pytest.mark.parametrize("residual", [-1.000001e-6, 1.000001e-6])
def test_probability_vectors_reject_values_just_outside_tolerance(
    target: str, residual: float
) -> None:
    payload = _input()
    payload["probability_tolerance"] = 1e-6
    if target == "prior":
        payload["states"][1]["probability"] += residual
    else:
        payload["outcomes"][0]["likelihood_by_state"]["favourable"] += residual

    with pytest.raises(InputError, match="sum to one"):
        outcome_conditional_sample_information_value(payload)


def test_roundoff_tolerance_is_exactly_zero_at_zero_scale() -> None:
    assert implementation._roundoff_tolerance(0.0, -0.0) == 0.0


def test_reference_action_must_be_exact_not_merely_numerically_close() -> None:
    payload = _input()
    payload["states"][0]["action_values"]["status_quo"] = 5.0 - 1e-14
    payload["states"][1]["action_values"]["status_quo"] = 5.0 - 1e-14
    payload["reference_action"] = "status_quo"
    with pytest.raises(InputError, match="exactly baseline optimal"):
        outcome_conditional_sample_information_value(payload)


def test_retrospective_scope_selects_but_does_not_reweight_distribution() -> None:
    prospective = _result()
    payload = _input()
    payload["scope"] = {
        "mode": "retrospective",
        "observed_outcome_id": "adverse_signal",
    }
    retrospective = _result(payload)
    assert retrospective["retrospective_outcome"] == retrospective["outcomes"][0]
    assert retrospective["aggregate"] == prospective["aggregate"]


def test_thresholds_and_levels_are_sorted_and_threshold_risk_is_monotone() -> None:
    payload = _input()
    payload["low_value_thresholds"] = [2.0, 0.0, 1.0]
    payload["quantile_levels"] = [1.0, 0.0, 0.5]
    result = _result(payload)
    assert [row["delta"] for row in result["aggregate"]["low_value_risks"]] == [
        0.0,
        1.0,
        2.0,
    ]
    assert [row["level"] for row in result["aggregate"]["weighted_quantiles"]] == [
        0.0,
        0.5,
        1.0,
    ]
    assert result["assurance"]["threshold_monotonic"] is True


def test_state_action_and_outcome_permutations_preserve_estimands() -> None:
    forward = _result()
    payload = _input()
    payload["states"].reverse()
    payload["actions"].reverse()
    payload["outcomes"].reverse()
    reverse = _result(payload)
    assert reverse["baseline"] == forward["baseline"]
    assert reverse["outcomes"] == forward["outcomes"]
    assert reverse["aggregate"] == forward["aggregate"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version="v2"), "schema_version"),
        (lambda value: value.update(extra=True), "Additional properties"),
        (
            lambda value: value["objective"].update(direction="minimize"),
            "utility must be maximized",
        ),
        (lambda value: value["states"][0].update(probability=0.6), "sum to one"),
        (
            lambda value: value["states"][0]["action_values"].pop("adaptive"),
            "enough properties",
        ),
        (
            lambda value: value["outcomes"][0]["likelihood_by_state"].pop("favourable"),
            "all states",
        ),
        (
            lambda value: value["outcomes"][0]["likelihood_by_state"].update(
                favourable=0.3
            ),
            "sum to one by state",
        ),
        (
            lambda value: value.update(reference_action="status_quo"),
            "exactly baseline optimal",
        ),
        (lambda value: value.update(information_cost=-1.0), "minimum"),
        (lambda value: value.update(low_value_thresholds=[-1.0]), "minimum"),
        (lambda value: value.update(quantile_levels=[1.1]), "maximum"),
        (
            lambda value: value["scope"].update(
                mode="prospective", observed_outcome_id="adverse_signal"
            ),
            "prospective scope",
        ),
        (
            lambda value: value["scope"].update(
                mode="retrospective", observed_outcome_id="missing"
            ),
            "declared observed outcome",
        ),
    ],
)
def test_invalid_inputs_fail_closed(mutation: object, match: str) -> None:
    payload = deepcopy(_input())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        outcome_conditional_sample_information_value(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(chronology="wrong"), "chronology"),
        (lambda value: value.update(actions="wrong"), "actions must be an array"),
        (
            lambda value: value.update(actions=["adaptive", "adaptive"]),
            "unique identifiers",
        ),
        (
            lambda value: value["states"][0].update(
                state_id=value["states"][1]["state_id"]
            ),
            "state identifiers must be unique",
        ),
        (
            lambda value: value["states"][0].update(probability=0.0),
            "state probabilities",
        ),
        (
            lambda value: value.update(probability_tolerance=0.0),
            "probability_tolerance",
        ),
        (
            lambda value: value["states"][0]["action_values"].pop("adaptive"),
            "declared actions",
        ),
        (
            lambda value: value["outcomes"][1].update(outcome_id="adverse_signal"),
            "outcome identifiers must be unique",
        ),
        (
            lambda value: value["outcomes"][0]["likelihood_by_state"].update(
                favourable=-0.1
            ),
            "likelihood probabilities",
        ),
        (
            lambda value: [
                outcome["likelihood_by_state"].update(
                    dict.fromkeys(("favourable", "unfavourable"), 0.0)
                )
                if index == 0
                else outcome["likelihood_by_state"].update(
                    dict.fromkeys(("favourable", "unfavourable"), 1.0)
                )
                for index, outcome in enumerate(value["outcomes"])
            ],
            "positive predictive mass",
        ),
        (lambda value: value.update(tie_tolerance=-1.0), "tie_tolerance"),
        (
            lambda value: value.update(reference_action="missing"),
            "declared action",
        ),
        (lambda value: value.update(information_cost=-1.0), "nonnegative"),
        (
            lambda value: value.update(cost_placement="inside_distribution"),
            "gross VSI distribution",
        ),
        (
            lambda value: value.update(low_value_thresholds="wrong"),
            "low_value_thresholds",
        ),
        (
            lambda value: value.update(quantile_levels="wrong"),
            "quantile_levels",
        ),
        (
            lambda value: value.update(low_value_thresholds=[0.0, 0.0]),
            "thresholds must be unique",
        ),
        (
            lambda value: value.update(quantile_levels=[0.5, 0.5]),
            "levels must be unique",
        ),
    ],
)
def test_semantic_builder_rejects_invalid_internal_values(
    mutation: object, match: str
) -> None:
    payload = deepcopy(_input())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=match):
        implementation._validate_and_build(payload)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: implementation._finite(True, "value"), "numeric"),
        (lambda: implementation._finite(math.nan, "value"), "finite"),
        (lambda: implementation._identifier("", "id"), "non-empty"),
        (lambda: implementation._mapping([], "mapping"), "object"),
        (lambda: implementation._records([], "records"), "non-empty array"),
        (lambda: implementation._require_nonnegative_vsi(-1.0), "nonnegative"),
        (
            lambda: implementation._assert_assurance(1.0, True, 1.0, 1.0),
            "tower identities",
        ),
        (
            lambda: implementation._assert_assurance(0.0, False, 1.0, 1.0),
            "monotone",
        ),
    ],
)
def test_defensive_assurance_helpers_fail_closed(call: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        call()  # type: ignore[operator]


def test_weighted_distribution_helper_defensive_fallbacks() -> None:
    rows = [
        {"vsi": 1.0, "predictive_probability": 0.2},
        {"vsi": 2.0, "predictive_probability": 0.2},
    ]
    assert implementation._weighted_quantile(rows, 0.9) == 2.0
    assert implementation._lower_tail_mean(rows, 0.9) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["aggregate"].update(evsi=999.0),
        lambda value: value["aggregate"].update(sigma_vsi=999.0),
        lambda value: value["aggregate"]["low_value_risks"][0].update(probability=0.25),
        lambda value: value["aggregate"]["weighted_quantiles"][0].update(vsi=1.0),
        lambda value: value["aggregate"]["lower_tail_means"][0].update(mean_vsi=1.0),
        lambda value: value["aggregate"].update(net_evsi=999.0),
        lambda value: value["outcomes"][0].update(vsi=999.0),
        lambda value: value["outcomes"][0].update(delta_ev=999.0),
        lambda value: value["outcomes"][0].update(optimal_actions=["adaptive"]),
        lambda value: value["aggregate"].update(
            complete_tie_set_changed_probability=0.0
        ),
        lambda value: value["assurance"].update(ddof=1),
        lambda value: value["scope"].update(mode="retrospective"),
        lambda value: value["baseline"].update(reference_action="status_quo"),
        lambda value: value["input_assurance"].update(input_sha256="0" * 64),
    ],
)
def test_standalone_result_assurance_rejects_mutations(mutation: object) -> None:
    result = _result()
    mutation(result)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_outcome_conditional_sample_information_result_semantics(result)


def test_schemas_are_strict_and_match_exported_files() -> None:
    Draft202012Validator.check_schema(
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1
    )
    Draft202012Validator.check_schema(
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1
    )
    assert (
        json.loads((CONTRACT / "schemas/input.schema.json").read_text(encoding="utf-8"))
        == OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1
    )
    assert (
        json.loads(
            (CONTRACT / "schemas/result.schema.json").read_text(encoding="utf-8")
        )
        == OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1
    )
    properties = cast(
        "dict[str, dict[str, Any]]",
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1["properties"],
    )
    reference = properties["reference_action"]
    tie_tolerance = properties["tie_tolerance"]
    assert "exact baseline extremum" in reference["description"]
    assert "not reference-action admissibility" in tie_tolerance["description"]


def test_public_api_and_cli_are_deterministic(tmp_path: Path) -> None:
    voiage_package = import_module("voiage")
    assert voiage_package.outcome_conditional_sample_information_value is (
        outcome_conditional_sample_information_value
    )
    assert methods.outcome_conditional_sample_information_value is (
        outcome_conditional_sample_information_value
    )
    output = tmp_path / "result.json"
    runner = CliRunner()
    invocation = runner.invoke(
        app,
        [
            "--format",
            "json",
            "calculate-outcome-conditional-sample-information",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert json.loads(output.read_text(encoding="utf-8")) == _result()
    stdout = runner.invoke(
        app,
        [
            "--format",
            "json",
            "calculate-outcome-conditional-sample-information",
            str(INPUT),
        ],
    )
    assert stdout.exit_code == 0
    assert json.loads(stdout.output) == _result()


def test_cli_text_output_and_invalid_top_level_fail_closed(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "result.txt"
    text_result = runner.invoke(
        app,
        [
            "--format",
            "text",
            "calculate-outcome-conditional-sample-information",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert text_result.exit_code == 0
    assert (
        "EVSI 1.000000; sigma-VSI 1.000000; net 0.750000 discounted utility points"
        in text_result.output
    )
    assert f"Result saved to {output}" in text_result.output
    assert "EVSI 1.000000" in output.read_text(encoding="utf-8")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    invalid_result = runner.invoke(
        app,
        ["calculate-outcome-conditional-sample-information", str(invalid)],
    )
    assert invalid_result.exit_code == 1
    assert "specification must be a JSON object" in invalid_result.output
