"""Changed-line assurance for governed COSS portfolio contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest

from voiage.contracts.study_design import (
    InformationValueInputV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.contracts.study_portfolio import (
    CossPortfolioCandidateV1,
    PortfolioCapacityConstraintV1,
    PortfolioIncrementalCostV1,
    PortfolioModelAssuranceV1,
    StudyPortfolioEvaluationV1,
    StudyPortfolioResultV1,
)
from voiage.exceptions import InputError
from voiage.experimental.study_design import calculate_coss, evsi_evpi_efficiency
import voiage.experimental.study_portfolio as portfolio_module

if TYPE_CHECKING:
    from collections.abc import Callable


def _context(*, decision_problem_id: str = "portfolio-policy") -> StudyDesignContextV1:
    return StudyDesignContextV1(
        decision_problem_id=decision_problem_id,
        value_unit="AUD-2026-present-value",
        population_scale=100_000.0,
        time_horizon="five-years",
        discounting_id="annual-3.5-percent",
        study_model_id="enumerated-design-v1",
        cost_model_id="full-economic-cost-v1",
        random_seed=42,
    )


def _candidate(
    study_id: str = "study-a",
    *,
    efficiency: bool = True,
    traffic: float = 1.0,
) -> CossPortfolioCandidateV1:
    context = _context()
    coss = calculate_coss(
        context=context,
        designs=(
            StudyDesignPointInputV1(
                design_id=f"{study_id}-design",
                sample_size=100,
                evsi=10.0,
                research_cost=2.0,
                estimator_provenance={"fixture": "coverage"},
            ),
        ),
    )
    efficiency_result = (
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=10.0, context=context),
            evpi=InformationValueInputV1(value=20.0, context=context),
        )
        if efficiency
        else None
    )
    model_ids = (
        "heterogeneity-v1",
        "delay-v1",
        "interference-v1",
        "monitoring-v1",
        "multiplicity-v1",
        "stop-v1",
    )
    return CossPortfolioCandidateV1(
        study_id=study_id,
        coss=coss,
        efficiency=efficiency_result,
        resource_use={"traffic": traffic},
        primary_metric_id="primary",
        secondary_metric_ids=("secondary",),
        guardrail_ids=("safety",),
        failed_guardrail_ids=(),
        heterogeneous_effect_model_id=model_ids[0],
        delayed_effect_model_id=model_ids[1],
        interference_model_id=model_ids[2],
        sequential_monitoring_plan_id=model_ids[3],
        multiplicity_adjustment_id=model_ids[4],
        stopping_rule_ids=(model_ids[5],),
        model_assurances=tuple(
            PortfolioModelAssuranceV1(
                model_id=model_id,
                handling="no_effect",
                provenance={"fixture": "coverage"},
            )
            for model_id in model_ids
        ),
        study_duration=12.0,
        duration_unit="months",
        incremental_cost=PortfolioIncrementalCostV1(
            opportunity_cost=1.0,
            implementation_delay_cost=1.0,
            excluded_from_coss_research_cost=True,
            basis_id="incremental-cost-v1",
            provenance={"fixture": "coverage"},
        ),
        expected_policy_change_id=f"{study_id}-policy",
    )


def _constraint(*, capacity: float = 1.0) -> PortfolioCapacityConstraintV1:
    return PortfolioCapacityConstraintV1(
        constraint_id="traffic", capacity=capacity, unit="million-users"
    )


def _result() -> StudyPortfolioResultV1:
    return portfolio_module.allocate_coss_portfolio(
        candidates=(_candidate(),), constraints=(_constraint(),)
    )


@pytest.mark.parametrize(
    ("contract", "payload", "message"),
    [
        (
            PortfolioModelAssuranceV1,
            {"model_id": "model", "handling": "no_effect", "provenance": {}},
            "provenance",
        ),
        (
            PortfolioIncrementalCostV1,
            {
                "opportunity_cost": 0.0,
                "implementation_delay_cost": 0.0,
                "excluded_from_coss_research_cost": True,
                "basis_id": "basis",
                "provenance": {},
            },
            "provenance",
        ),
    ],
)
def test_assurance_contracts_reject_empty_provenance(
    contract: type[PortfolioModelAssuranceV1 | PortfolioIncrementalCostV1],
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        contract.model_validate(payload)


def _candidate_payload() -> dict[str, object]:
    return deepcopy(_candidate().model_dump(mode="json"))


def _set_negative_resource(payload: dict[str, object]) -> None:
    payload["resource_use"] = {"traffic": -1.0}


def _set_self_dependency(payload: dict[str, object]) -> None:
    payload["required_study_ids"] = [payload["study_id"]]


def _set_duplicate_secondary(payload: dict[str, object]) -> None:
    payload["secondary_metric_ids"] = ["secondary", "secondary"]


def _set_primary_as_secondary(payload: dict[str, object]) -> None:
    payload["secondary_metric_ids"] = [payload["primary_metric_id"]]


def _set_duplicate_guardrails(payload: dict[str, object]) -> None:
    payload["guardrail_ids"] = ["safety", "safety"]


def _set_unknown_failed_guardrail(payload: dict[str, object]) -> None:
    payload["guardrails_passed"] = False
    payload["failed_guardrail_ids"] = ["undeclared"]


def _set_inconsistent_guardrail_state(payload: dict[str, object]) -> None:
    payload["guardrails_passed"] = False
    payload["failed_guardrail_ids"] = []


def _set_empty_stopping_rules(payload: dict[str, object]) -> None:
    payload["stopping_rule_ids"] = []
    payload["model_assurances"] = payload["model_assurances"][:-1]  # type: ignore[index]


def _set_duplicate_stopping_rules(payload: dict[str, object]) -> None:
    payload["stopping_rule_ids"] = ["stop-v1", "stop-v1"]


def _set_duplicate_model_assurances(payload: dict[str, object]) -> None:
    payload["model_assurances"].append(deepcopy(payload["model_assurances"][0]))  # type: ignore[union-attr,index]


def _set_missing_model_assurance(payload: dict[str, object]) -> None:
    payload["model_assurances"] = payload["model_assurances"][:-1]  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_set_negative_resource, "non-negative"),
        (_set_self_dependency, "cannot require itself"),
        (_set_duplicate_secondary, "secondary_metric_ids must be unique"),
        (_set_primary_as_secondary, "primary metric"),
        (_set_duplicate_guardrails, "guardrail_ids must be unique"),
        (_set_unknown_failed_guardrail, "declared guardrails"),
        (_set_inconsistent_guardrail_state, "guardrails_passed"),
        (_set_empty_stopping_rules, "non-empty and unique"),
        (_set_duplicate_stopping_rules, "non-empty and unique"),
        (_set_duplicate_model_assurances, "unique model_id"),
        (_set_missing_model_assurance, "cover every declared"),
    ],
)
def test_candidate_contract_rejects_incoherent_semantics(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _candidate_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(payload))


def test_candidate_contract_requires_a_feasible_coss_optimum() -> None:
    context = _context()
    infeasible_coss = calculate_coss(
        context=context,
        designs=(
            StudyDesignPointInputV1(
                design_id="infeasible",
                sample_size=100,
                evsi=10.0,
                research_cost=2.0,
                feasible=False,
                feasibility_codes=("capacity",),
            ),
        ),
    )
    payload = _candidate_payload()
    payload["coss"] = infeasible_coss.model_dump(mode="json")
    payload["efficiency"] = None

    with pytest.raises(ValidationError, match="feasible optimum"):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(payload))


def test_candidate_contract_rejects_efficiency_from_another_context() -> None:
    payload = _candidate_payload()
    other_context = _context(decision_problem_id="other-policy")
    payload["efficiency"] = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=10.0, context=other_context),
        evpi=InformationValueInputV1(value=20.0, context=other_context),
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="match the COSS context"):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(payload))


def _evaluation_payload() -> dict[str, object]:
    return deepcopy(_result().evaluations[0].model_dump(mode="json"))


def test_evaluation_contract_rejects_broken_value_identity() -> None:
    payload = _evaluation_payload()
    payload["net_enbs"] = 999.0

    with pytest.raises(ValidationError, match="identities disagree"):
        StudyPortfolioEvaluationV1.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("duplicate", [True, False])
def test_evaluation_contract_requires_exact_model_assurance_coverage(
    duplicate: bool,
) -> None:
    payload = _evaluation_payload()
    assurances = payload["model_assurances"]
    if duplicate:
        assurances.append(deepcopy(assurances[0]))  # type: ignore[union-attr,index]
        message = "unique model_id"
    else:
        payload["model_assurances"] = assurances[:-1]  # type: ignore[index]
        message = "cover every declared"

    with pytest.raises(ValidationError, match=message):
        StudyPortfolioEvaluationV1.model_validate_json(json.dumps(payload))


def _empty_result_payload() -> dict[str, object]:
    payload = deepcopy(_result().model_dump(mode="json"))
    payload.update(
        evaluations=[],
        selected_study_ids=[],
        total_gross_evsi=0.0,
        total_net_evsi=0.0,
        total_research_cost=0.0,
        total_opportunity_cost=0.0,
        total_implementation_delay_cost=0.0,
        total_gross_enbs=0.0,
        total_net_enbs=0.0,
        total_enbs=0.0,
        used_capacity={"traffic": 0.0},
        binding_constraint_ids=[],
        selected_policy_change_ids=[],
        selected_stopping_rule_ids=[],
    )
    return payload


def _duplicate_evaluation(payload: dict[str, object]) -> None:
    payload["evaluations"].append(deepcopy(payload["evaluations"][0]))  # type: ignore[union-attr,index]


def _disagree_selected_ids(payload: dict[str, object]) -> None:
    payload["selected_study_ids"] = []


def _duplicate_constraints(payload: dict[str, object]) -> None:
    payload["constraints"].append(deepcopy(payload["constraints"][0]))  # type: ignore[union-attr,index]


def _omit_used_capacity(payload: dict[str, object]) -> None:
    payload["used_capacity"] = {}


def _misstate_used_capacity(payload: dict[str, object]) -> None:
    payload["used_capacity"] = {"traffic": 0.5}


def _exceed_capacity(payload: dict[str, object]) -> None:
    payload["constraints"][0]["capacity"] = 0.5  # type: ignore[index]


def _disagree_policy_changes(payload: dict[str, object]) -> None:
    payload["selected_policy_change_ids"] = []


def _disagree_stopping_rules(payload: dict[str, object]) -> None:
    payload["selected_stopping_rule_ids"] = []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_duplicate_evaluation, "study_id values must be unique"),
        (_disagree_selected_ids, "selected_study_ids"),
        (_duplicate_constraints, "constraint_id values must be unique"),
        (_omit_used_capacity, "cover every declared constraint"),
        (_misstate_used_capacity, "disagrees with selected evaluations"),
        (_exceed_capacity, "exceeds a capacity constraint"),
        (_disagree_policy_changes, "selected_policy_change_ids"),
        (_disagree_stopping_rules, "selected_stopping_rule_ids"),
    ],
)
def test_result_contract_rejects_relational_corruption(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = deepcopy(_result().model_dump(mode="json"))
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        StudyPortfolioResultV1.model_validate_json(json.dumps(payload))


def test_result_contract_rejects_an_empty_evaluation_set() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        StudyPortfolioResultV1.model_validate_json(json.dumps(_empty_result_payload()))


def test_allocator_boundary_guards_and_optional_efficiency_diagnostic() -> None:
    with pytest.raises(InputError, match="at least one"):
        portfolio_module.allocate_coss_portfolio(candidates=())

    candidates = tuple(
        _candidate().model_copy(update={"study_id": f"study-{index}"})
        for index in range(25)
    )
    with pytest.raises(InputError, match="at most 24"):
        portfolio_module.allocate_coss_portfolio(candidates=candidates)

    duplicate = _candidate()
    with pytest.raises(InputError, match="study_id values must be unique"):
        portfolio_module.allocate_coss_portfolio(
            candidates=(duplicate, duplicate), constraints=(_constraint(),)
        )

    constraint = _constraint()
    with pytest.raises(InputError, match="constraint_id values must be unique"):
        portfolio_module.allocate_coss_portfolio(
            candidates=(_candidate(),), constraints=(constraint, constraint)
        )

    result = portfolio_module.allocate_coss_portfolio(
        candidates=(_candidate(efficiency=False),), constraints=(_constraint(),)
    )
    assert "efficiency_not_supplied_for_all_candidates" in result.diagnostics


def test_allocator_optimum_guard_is_fail_closed() -> None:
    candidate = _candidate()
    candidate_without_optimum = candidate.model_copy(
        update={"coss": candidate.coss.model_copy(update={"optimal_design_id": None})}
    )

    with pytest.raises(InputError, match="feasible optimum"):
        portfolio_module._optimum(candidate_without_optimum)


def test_allocator_translates_result_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_result(**_kwargs: object) -> StudyPortfolioResultV1:
        return StudyPortfolioResultV1.model_validate({})

    monkeypatch.setattr(portfolio_module, "StudyPortfolioResultV1", reject_result)

    with pytest.raises(InputError, match="scientific contract validation"):
        portfolio_module.allocate_coss_portfolio(
            candidates=(_candidate(),), constraints=(_constraint(),)
        )
