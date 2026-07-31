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
    assert result.total_research_cost == 4.0
    assert result.total_enbs == 13.0
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
            candidates=(
                _candidate("a", evsi=5.0, cost=1.0, required=("missing",)),
            ),
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
        CossPortfolioCandidateV1(
            study_id="a",
            coss=valid.coss,
            efficiency=wrong,
            resource_use={"traffic": 1.0},
        )


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


def test_experimental_export_is_lazy_and_available() -> None:
    from voiage import experimental

    assert experimental.allocate_coss_portfolio is allocate_coss_portfolio
