"""Tests for DecisionProblem and Intervention schemas and contract conformance."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from voiage.exceptions import InputError
from voiage.schema import DecisionProblem, Intervention

ROOT = Path(__file__).resolve().parents[1]


def test_intervention_valid_construction_and_properties() -> None:
    intervention = Intervention(
        intervention_id="treat_01",
        name="New Therapy",
        description="Standard dose therapy",
        is_reference=False,
        category="active",
    )
    assert intervention.intervention_id == "treat_01"
    assert intervention.name == "New Therapy"
    assert intervention.description == "Standard dose therapy"
    assert intervention.is_reference is False
    assert intervention.category == "active"

    data = intervention.to_dict()
    assert data["intervention_id"] == "treat_01"
    assert data["name"] == "New Therapy"
    assert data["is_reference"] is False
    assert data["description"] == "Standard dose therapy"
    assert data["category"] == "active"

    round_tripped = Intervention.from_dict(data)
    assert round_tripped == intervention


def test_intervention_validation_failures() -> None:
    with pytest.raises(InputError, match="intervention_id"):
        Intervention(intervention_id="", name="Valid")
    with pytest.raises(InputError, match="name"):
        Intervention(intervention_id="valid", name="")
    with pytest.raises(InputError, match="description"):
        Intervention(intervention_id="valid", name="Valid", description=123)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="is_reference"):
        Intervention(intervention_id="valid", name="Valid", is_reference="yes")  # type: ignore[arg-type]
    with pytest.raises(InputError, match="category"):
        Intervention(intervention_id="valid", name="Valid", category=456)  # type: ignore[arg-type]

    with pytest.raises(InputError, match="must be a dictionary"):
        Intervention.from_dict("not a dict")
    with pytest.raises(InputError, match="intervention_id"):
        Intervention.from_dict({"name": "No ID"})
    with pytest.raises(InputError, match="name"):
        Intervention.from_dict({"intervention_id": "id"})


def test_decision_problem_valid_construction_and_accessors() -> None:
    int1 = Intervention(
        intervention_id="soc", name="Standard of Care", is_reference=True
    )
    int2 = Intervention(intervention_id="treat_a", name="Treatment A")
    int3 = Intervention(intervention_id="treat_b", name="Treatment B")

    problem = DecisionProblem(
        decision_problem_id="prob_001",
        title="Colorectal Cancer Screening Strategy Selection",
        willingness_to_pay=50000.0,
        interventions=[int1, int2, int3],
        currency="USD",
        outcome_names=["QALY", "Life Years"],
    )

    assert problem.decision_problem_id == "prob_001"
    assert problem.title == "Colorectal Cancer Screening Strategy Selection"
    assert problem.willingness_to_pay == 50000.0
    assert problem.currency == "USD"
    assert problem.analysis_type == "net-benefit-first"
    assert problem.outcome_names == ["QALY", "Life Years"]
    assert problem.reference_intervention == int1
    assert problem.intervention_names == [
        "Standard of Care",
        "Treatment A",
        "Treatment B",
    ]
    assert problem.intervention_ids == ["soc", "treat_a", "treat_b"]

    data = problem.to_dict()
    assert data["decision_problem_id"] == "prob_001"
    assert data["analysis_type"] == "net-benefit-first"
    assert len(data["interventions"]) == 3

    round_tripped = DecisionProblem.from_dict(data)
    assert round_tripped == problem


def test_decision_problem_validation_failures() -> None:
    int1 = Intervention(
        intervention_id="soc", name="Standard of Care", is_reference=True
    )
    int2 = Intervention(intervention_id="treat_a", name="Treatment A")

    with pytest.raises(InputError, match="decision_problem_id"):
        DecisionProblem(
            decision_problem_id="",
            title="Valid",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
        )

    with pytest.raises(InputError, match="title"):
        DecisionProblem(
            decision_problem_id="prob",
            title="",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
        )

    with pytest.raises(InputError, match="willingness_to_pay"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=0.0,
            interventions=[int1, int2],
        )

    with pytest.raises(InputError, match="currency"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
            currency="US",
        )

    with pytest.raises(InputError, match="analysis_type"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
            analysis_type="cost-effectiveness-ratio",
        )

    with pytest.raises(InputError, match="interventions"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[],
        )

    with pytest.raises(InputError, match="Intervention objects"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=["not_an_intervention"],  # type: ignore[list-item]
        )

    # Duplicate intervention IDs
    int_dup = Intervention(intervention_id="soc", name="Duplicate SoC")
    with pytest.raises(InputError, match="unique"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[int1, int_dup],
        )

    with pytest.raises(InputError, match="outcome_names"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
            outcome_names=[],
        )

    with pytest.raises(InputError, match="outcome names"):
        DecisionProblem(
            decision_problem_id="prob",
            title="Title",
            willingness_to_pay=1000.0,
            interventions=[int1, int2],
            outcome_names=[""],
        )


def test_decision_problem_conforms_to_v1_json_schema() -> None:
    schema_path = ROOT / "specs/core-api/schemas/v1/decision-problem.schema.json"
    intervention_schema_path = (
        ROOT / "specs/core-api/schemas/v1/intervention.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    intervention_schema = json.loads(
        intervention_schema_path.read_text(encoding="utf-8")
    )

    schema_store = {
        schema_path.as_uri(): schema,
        intervention_schema_path.as_uri(): intervention_schema,
        "https://voiage.dev/specs/core-api/schemas/v1/decision-problem.schema.json": schema,
        "https://voiage.dev/specs/core-api/schemas/v1/intervention.schema.json": intervention_schema,
        "./intervention.schema.json": intervention_schema,
    }

    problem = DecisionProblem(
        decision_problem_id="prob_conformance_001",
        title="Screening Case Study",
        willingness_to_pay=30000.0,
        currency="EUR",
        interventions=[
            Intervention(
                intervention_id="int_01",
                name="Care as Usual",
                description="Usual primary care",
                is_reference=True,
            ),
            Intervention(
                intervention_id="int_02",
                name="Telehealth Monitoring",
                category="digital",
            ),
        ],
        outcome_names=["QALY"],
    )
    payload = problem.to_dict()

    resolver = jsonschema.RefResolver.from_schema(schema, store=schema_store)
    jsonschema.validate(instance=payload, schema=schema, resolver=resolver)


def test_decision_problem_example_file_round_trips_through_schema_class() -> None:
    example_path = ROOT / "specs/core-api/examples/v1/decision-problem.example.json"
    example_data = json.loads(example_path.read_text(encoding="utf-8"))

    problem = DecisionProblem.from_dict(example_data)
    assert problem.decision_problem_id == example_data["decision_problem_id"]
    assert problem.title == example_data["title"]
    assert problem.willingness_to_pay == example_data["willingness_to_pay"]
    assert len(problem.interventions) == len(example_data["interventions"])

    data = problem.to_dict()
    assert data["decision_problem_id"] == example_data["decision_problem_id"]
    assert data["analysis_type"] == example_data["analysis_type"]
