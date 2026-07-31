"""Assurance for exact allocation over governed COSS study optima."""

from copy import deepcopy
import json

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
    StudyPortfolioResultV1,
)
from voiage.exceptions import InputError
from voiage.experimental import allocate_coss_portfolio
from voiage.experimental.study_design import calculate_coss, evsi_evpi_efficiency


def _context(*, problem: str = "shared-policy") -> StudyDesignContextV1:
    return StudyDesignContextV1(
        decision_problem_id=problem,
        value_unit="AUD-2026-present-value",
        population_scale=100_000.0,
        time_horizon="five-years",
        discounting_id="annual-3.5-percent",
        study_model_id="enumerated-design-v1",
        cost_model_id="full-economic-cost-v1",
        random_seed=42,
    )


def _candidate(
    study_id: str,
    *,
    evsi: float,
    cost: float,
    traffic: float = 0.0,
    context: StudyDesignContextV1 | None = None,
    guardrails_passed: bool = True,
    required: tuple[str, ...] = (),
    exclusions: tuple[str, ...] = (),
    with_efficiency: bool = True,
    opportunity_cost: float = 0.0,
    implementation_delay_cost: float = 0.0,
) -> CossPortfolioCandidateV1:
    shared_context = context or _context()
    coss = calculate_coss(
        context=shared_context,
        designs=(
            StudyDesignPointInputV1(
                design_id=f"{study_id}-optimal",
                sample_size=100,
                evsi=evsi,
                research_cost=cost,
                estimator_provenance={"fixture": "enumerable"},
            ),
        ),
    )
    efficiency = None
    if with_efficiency:
        efficiency = evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=evsi, context=shared_context),
            evpi=InformationValueInputV1(value=20.0, context=shared_context),
        )
    return CossPortfolioCandidateV1(
        study_id=study_id,
        coss=coss,
        efficiency=efficiency,
        resource_use={"traffic": traffic},
        required_study_ids=required,
        exclusion_group_ids=exclusions,
        guardrails_passed=guardrails_passed,
        primary_metric_id="primary-net-benefit",
        secondary_metric_ids=("secondary-safety",),
        guardrail_ids=("safety",),
        failed_guardrail_ids=() if guardrails_passed else ("safety",),
        heterogeneous_effect_model_id="declared-no-heterogeneity-v1",
        delayed_effect_model_id="declared-no-delay-v1",
        interference_model_id="declared-no-interference-v1",
        sequential_monitoring_plan_id="fixed-horizon-v1",
        multiplicity_adjustment_id="single-primary-v1",
        stopping_rule_ids=("fixed-sample-completion",),
        model_assurances=tuple(
            PortfolioModelAssuranceV1(
                model_id=model_id,
                handling="no_effect",
                provenance={"fixture": "explicit-null-effect"},
            )
            for model_id in (
                "declared-no-heterogeneity-v1",
                "declared-no-delay-v1",
                "declared-no-interference-v1",
                "fixed-horizon-v1",
                "single-primary-v1",
                "fixed-sample-completion",
            )
        ),
        study_duration=12.0,
        duration_unit="months",
        incremental_cost=PortfolioIncrementalCostV1(
            opportunity_cost=opportunity_cost,
            implementation_delay_cost=implementation_delay_cost,
            excluded_from_coss_research_cost=True,
            basis_id="portfolio-incremental-cost-v1",
            provenance={"fixture": "disjoint-cost-ledger"},
        ),
        expected_policy_change_id=f"{study_id}-adopt-if-informative",
    )


def test_exact_portfolio_selects_highest_additive_enbs_bundle() -> None:
    candidates = (
        _candidate("large", evsi=14.0, cost=4.0, traffic=8.0),
        _candidate("medium", evsi=9.0, cost=2.0, traffic=4.0),
        _candidate("small", evsi=8.0, cost=2.0, traffic=4.0),
    )
    result = allocate_coss_portfolio(
        candidates=candidates,
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=8.0, unit="million-users"
            ),
        ),
    )

    assert result.selected_study_ids == ("medium", "small")
    assert result.total_gross_evsi == 17.0
    assert result.total_net_evsi == 17.0
    assert result.total_research_cost == 4.0
    assert result.total_enbs == 13.0
    assert result.total_gross_enbs == 13.0
    assert result.total_net_enbs == 13.0
    assert result.used_capacity == {"traffic": 8.0}
    assert result.binding_constraint_ids == ("traffic",)
    assert [item.efficiency_ratio for item in result.evaluations] == [0.7, 0.45, 0.4]


def test_empty_portfolio_beats_negative_enbs_and_guardrail_failures() -> None:
    result = allocate_coss_portfolio(
        candidates=(
            _candidate("negative", evsi=2.0, cost=3.0),
            _candidate("unsafe", evsi=10.0, cost=1.0, guardrails_passed=False),
        ),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=100.0, unit="million-users"
            ),
        ),
    )

    assert result.selected_study_ids == ()
    assert result.total_enbs == 0.0
    assert "empty_portfolio_selected" in result.diagnostics
    assert "guardrail_failed_candidates_excluded" in result.diagnostics


def test_dependencies_and_exclusions_are_hard_constraints() -> None:
    candidates = (
        _candidate("platform", evsi=4.0, cost=1.0),
        _candidate("dependent", evsi=9.0, cost=1.0, required=("platform",)),
        _candidate("variant-a", evsi=8.0, cost=1.0, exclusions=("shared-slot",)),
        _candidate("variant-b", evsi=7.0, cost=1.0, exclusions=("shared-slot",)),
    )
    result = allocate_coss_portfolio(
        candidates=candidates,
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=100.0, unit="million-users"
            ),
        ),
    )

    assert "platform" in result.selected_study_ids
    assert "dependent" in result.selected_study_ids
    assert "variant-a" in result.selected_study_ids
    assert "variant-b" not in result.selected_study_ids


def test_ties_prefer_lower_cost_then_lexicographic_ids() -> None:
    lower_cost = allocate_coss_portfolio(
        candidates=(
            _candidate("expensive", evsi=12.0, cost=2.0, traffic=1.0),
            _candidate("cheap", evsi=11.0, cost=1.0, traffic=1.0),
        ),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=1.0, unit="slot"
            ),
        ),
    )
    assert lower_cost.selected_study_ids == ("cheap",)

    lexical = allocate_coss_portfolio(
        candidates=(
            _candidate("zeta", evsi=11.0, cost=1.0, traffic=1.0),
            _candidate("alpha", evsi=11.0, cost=1.0, traffic=1.0),
        ),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=1.0, unit="slot"
            ),
        ),
    )
    assert lexical.selected_study_ids == ("alpha",)


def test_tolerance_ties_are_anchored_to_global_maximum_and_permutation_invariant() -> (
    None
):
    candidates = (
        _candidate("a", evsi=11.0, cost=10.0, traffic=1.0),
        _candidate("b", evsi=9.91, cost=9.0, traffic=1.0),
        _candidate("c", evsi=8.82, cost=8.0, traffic=1.0),
    )
    constraint = PortfolioCapacityConstraintV1(
        constraint_id="traffic", capacity=1.0, unit="slot"
    )
    forward = allocate_coss_portfolio(
        candidates=candidates,
        constraints=(constraint,),
        absolute_tolerance=0.1,
        relative_tolerance=0.0,
    )
    reverse = allocate_coss_portfolio(
        candidates=tuple(reversed(candidates)),
        constraints=(constraint,),
        absolute_tolerance=0.1,
        relative_tolerance=0.0,
    )

    assert forward.selected_study_ids == ("b",)
    assert reverse.selected_study_ids == ("b",)
    assert forward.total_enbs == pytest.approx(0.91)


def test_portfolio_rejects_incommensurate_or_unbound_inputs() -> None:
    with pytest.raises(InputError, match="commensurate"):
        allocate_coss_portfolio(
            candidates=(
                _candidate("a", evsi=5.0, cost=1.0),
                _candidate("b", evsi=5.0, cost=1.0, context=_context(problem="other")),
            ),
            constraints=(
                PortfolioCapacityConstraintV1(
                    constraint_id="traffic", capacity=2.0, unit="slot"
                ),
            ),
        )
    with pytest.raises(InputError, match="undeclared constraint"):
        allocate_coss_portfolio(candidates=(_candidate("a", evsi=5.0, cost=1.0),))
    with pytest.raises(InputError, match="unknown study_id"):
        allocate_coss_portfolio(
            candidates=(_candidate("a", evsi=5.0, cost=1.0, required=("missing",)),),
            constraints=(
                PortfolioCapacityConstraintV1(
                    constraint_id="traffic", capacity=2.0, unit="slot"
                ),
            ),
        )


def test_candidate_rejects_efficiency_for_a_different_evsi() -> None:
    context = _context()
    valid = _candidate("a", evsi=5.0, cost=1.0, context=context)
    wrong = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=4.0, context=context),
        evpi=InformationValueInputV1(value=20.0, context=context),
    )
    with pytest.raises(ValidationError, match="EVSI must match"):
        CossPortfolioCandidateV1.model_validate_json(
            json.dumps(
                {
                    **valid.model_dump(mode="json"),
                    "efficiency": wrong.model_dump(mode="json"),
                }
            )
        )


def test_candidate_rejects_unassured_models_and_overlapping_costs() -> None:
    valid = _candidate("a", evsi=5.0, cost=1.0)
    payload = valid.model_dump(mode="json")
    payload["interference_model_id"] = "spillover-model-v1"
    with pytest.raises(ValidationError, match="cover every declared"):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(payload))

    overlapping = deepcopy(valid.model_dump(mode="json"))
    overlapping["incremental_cost"]["excluded_from_coss_research_cost"] = False
    with pytest.raises(ValidationError, match="excluded_from_coss_research_cost"):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(overlapping))


def test_allocator_rejects_malformed_boundary_objects() -> None:
    with pytest.raises(InputError, match="CossPortfolioCandidateV1"):
        allocate_coss_portfolio(candidates=[object()])  # type: ignore[list-item]
    with pytest.raises(InputError, match="PortfolioCapacityConstraintV1"):
        allocate_coss_portfolio(
            candidates=(_candidate("a", evsi=5.0, cost=1.0),),
            constraints=[object()],  # type: ignore[list-item]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["absolute_tolerance", "relative_tolerance"])
def test_allocator_rejects_non_finite_tolerances(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(InputError, match="finite and non-negative"):
        allocate_coss_portfolio(
            candidates=(_candidate("a", evsi=5.0, cost=1.0),),
            constraints=(
                PortfolioCapacityConstraintV1(
                    constraint_id="traffic", capacity=1.0, unit="slot"
                ),
            ),
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("target", ["model", "cost"])
def test_candidate_rejects_blank_assurance_provenance_values(target: str) -> None:
    payload = _candidate("a", evsi=5.0, cost=1.0).model_dump(mode="json")
    if target == "model":
        payload["model_assurances"][0]["provenance"] = {"basis": ""}
    else:
        payload["incremental_cost"]["provenance"] = {"basis": ""}
    with pytest.raises(ValidationError, match="at least 1 character"):
        CossPortfolioCandidateV1.model_validate_json(json.dumps(payload))


def test_portfolio_carries_advanced_design_semantics_and_net_values() -> None:
    candidate = _candidate(
        "adjusted",
        evsi=10.0,
        cost=2.0,
        traffic=1.0,
        opportunity_cost=1.0,
        implementation_delay_cost=2.0,
    )
    result = allocate_coss_portfolio(
        candidates=(candidate,),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=1.0, unit="million-users"
            ),
        ),
    )

    evaluation = result.evaluations[0]
    assert evaluation.gross_evsi == 10.0
    assert evaluation.net_evsi == 7.0
    assert evaluation.gross_enbs == 8.0
    assert evaluation.net_enbs == 5.0
    assert evaluation.enbs == evaluation.net_enbs
    assert evaluation.interference_model_id == "declared-no-interference-v1"
    assert evaluation.multiplicity_adjustment_id == "single-primary-v1"
    assert evaluation.stopping_rule_ids == ("fixed-sample-completion",)
    assert result.total_opportunity_cost == 1.0
    assert result.total_implementation_delay_cost == 2.0
    assert result.selected_policy_change_ids == ("adjusted-adopt-if-informative",)
    assert result.selected_stopping_rule_ids == ("fixed-sample-completion",)


def test_result_contract_rejects_corrupted_totals() -> None:
    result = allocate_coss_portfolio(
        candidates=(_candidate("a", evsi=5.0, cost=1.0),),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=1.0, unit="slot"
            ),
        ),
    )
    payload = deepcopy(result.model_dump(mode="json"))
    payload["total_enbs"] = 99.0
    with pytest.raises(ValidationError, match="totals"):
        StudyPortfolioResultV1.model_validate_json(json.dumps(payload))


def test_result_contract_rejects_forged_bindings_and_resource_keys() -> None:
    result = allocate_coss_portfolio(
        candidates=(_candidate("a", evsi=5.0, cost=1.0, traffic=1.0),),
        constraints=(
            PortfolioCapacityConstraintV1(
                constraint_id="traffic", capacity=1.0, unit="slot"
            ),
        ),
    )
    binding_payload = deepcopy(result.model_dump(mode="json"))
    binding_payload["binding_constraint_ids"] = ["bogus"]
    with pytest.raises(ValidationError, match="binding_constraint_ids"):
        StudyPortfolioResultV1.model_validate_json(json.dumps(binding_payload))

    resource_payload = deepcopy(result.model_dump(mode="json"))
    resource_payload["evaluations"][0]["resource_use"]["undeclared"] = 1.0
    with pytest.raises(ValidationError, match="undeclared constraint"):
        StudyPortfolioResultV1.model_validate_json(json.dumps(resource_payload))


def test_experimental_export_is_lazy_and_available() -> None:
    from voiage import experimental

    assert experimental.allocate_coss_portfolio is allocate_coss_portfolio
