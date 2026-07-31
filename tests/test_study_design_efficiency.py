"""Reference tests for governed COSS and study-information efficiency."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from voiage.contracts.study_design import (
    FeasibleDesignRangeV1,
    InformationValueInputV1,
    SelectionUncertaintyV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.exceptions import InputError
from voiage.experimental.study_design import calculate_coss, evsi_evpi_efficiency


@pytest.fixture
def study_context() -> StudyDesignContextV1:
    """Return one fully declared and commensurate study-value context."""
    return StudyDesignContextV1(
        decision_problem_id="therapy-choice-v1",
        value_unit="AUD_2026",
        population_scale=10_000.0,
        time_horizon="2027-2036",
        discounting_id="aud-health-2026",
        study_model_id="normal-normal-v1",
        cost_model_id="trial-cost-v1",
        random_seed=571,
    )


def test_coss_matches_independent_enumerable_signed_enbs_reference(
    study_context: StudyDesignContextV1,
) -> None:
    """The native argmax agrees with an independently enumerated reference."""
    designs = (
        StudyDesignPointInputV1(
            design_id="n-50", sample_size=50, evsi=5.0, research_cost=8.0
        ),
        StudyDesignPointInputV1(
            design_id="n-100", sample_size=100, evsi=16.0, research_cost=10.0
        ),
        StudyDesignPointInputV1(
            design_id="n-150", sample_size=150, evsi=21.0, research_cost=18.0
        ),
    )
    independent_enbs = tuple(point.evsi - point.research_cost for point in designs)
    independent_optimum = max(
        range(len(designs)), key=lambda index: independent_enbs[index]
    )

    result = calculate_coss(context=study_context, designs=designs)

    assert tuple(point.enbs for point in result.evaluated_designs) == pytest.approx(
        independent_enbs
    )
    assert result.maximum_enbs == pytest.approx(independent_enbs[independent_optimum])
    assert result.optimal_design_id == designs[independent_optimum].design_id
    assert result.optimal_sample_size == designs[independent_optimum].sample_size
    assert result.feasible_sample_sizes == (50, 100, 150)
    assert result.plot_data.enbs == pytest.approx(independent_enbs)


def test_coss_preserves_negative_enbs_without_a_zero_floor(
    study_context: StudyDesignContextV1,
) -> None:
    """Economically unattractive evaluated designs remain signed evidence."""
    result = calculate_coss(
        context=study_context,
        designs=(
            StudyDesignPointInputV1(
                design_id="n-20", sample_size=20, evsi=1.0, research_cost=4.0
            ),
            StudyDesignPointInputV1(
                design_id="n-40", sample_size=40, evsi=2.0, research_cost=6.0
            ),
        ),
    )

    assert tuple(point.enbs for point in result.evaluated_designs) == (-3.0, -4.0)
    assert result.maximum_enbs == -3.0
    assert result.optimal_design_id == "n-20"
    assert result.boundary_state == "lower"


def _point(
    design_id: str,
    sample_size: int,
    evsi: float,
    cost: float,
    *,
    feasible: bool = True,
) -> StudyDesignPointInputV1:
    return StudyDesignPointInputV1(
        design_id=design_id,
        sample_size=sample_size,
        evsi=evsi,
        research_cost=cost,
        feasible=feasible,
        feasibility_codes=() if feasible else ("capacity_exceeded",),
    )


def test_coss_selects_an_interior_optimum_on_a_non_monotone_curve(
    study_context: StudyDesignContextV1,
) -> None:
    designs = (
        _point("n-20", 20, 8.0, 3.0),
        _point("n-40", 40, 17.0, 6.0),
        _point("n-60", 60, 14.0, 8.0),
        _point("n-80", 80, 20.0, 12.0),
    )

    result = calculate_coss(context=study_context, designs=designs)

    assert result.optimal_design_id == "n-40"
    assert result.boundary_state == "interior"
    assert "non_monotone_evsi" in result.diagnostics


def test_coss_applies_declared_tie_policy_deterministically(
    study_context: StudyDesignContextV1,
) -> None:
    designs = (
        _point("large", 100, 15.0, 5.0),
        _point("small-z", 50, 12.0, 2.0),
        _point("small-a", 50, 11.0, 1.0),
    )

    smallest = calculate_coss(context=study_context, designs=designs)
    largest = calculate_coss(
        context=study_context, designs=designs, tie_policy="largest_sample_size"
    )
    first = calculate_coss(
        context=study_context, designs=designs, tie_policy="first_declared"
    )

    assert smallest.optimal_design_id == "small-a"
    assert smallest.tied_optimal_design_ids == ("large", "small-z", "small-a")
    assert largest.optimal_design_id == "large"
    assert first.optimal_design_id == "large"


def test_coss_returns_no_optimum_when_every_design_is_infeasible(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(
            _point("n-20", 20, 9.0, 2.0, feasible=False),
            _point("n-40", 40, 14.0, 3.0, feasible=False),
        ),
    )

    assert result.optimal_design_id is None
    assert result.optimal_sample_size is None
    assert result.maximum_enbs is None
    assert result.boundary_state == "none"
    assert result.feasible_sample_sizes == ()
    assert "no_feasible_design" in result.diagnostics


def test_coss_reports_upper_boundary_and_retains_infeasible_records(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(
            _point("n-20", 20, 5.0, 2.0),
            _point("n-40", 40, 12.0, 4.0),
            _point("n-60", 60, 20.0, 5.0, feasible=False),
        ),
        declared_feasible_range=(20, 40),
    )

    assert result.optimal_design_id == "n-40"
    assert result.boundary_state == "upper"
    assert len(result.evaluated_designs) == 3
    assert result.plot_data.feasible == (True, True, False)


def test_coss_rejects_duplicate_design_identity_but_allows_shared_sample_size(
    study_context: StudyDesignContextV1,
) -> None:
    duplicate_id = (_point("same", 20, 5.0, 2.0), _point("same", 40, 7.0, 3.0))
    duplicate_size = (_point("z", 20, 5.0, 2.0), _point("a", 20, 7.0, 4.0))

    with pytest.raises(InputError, match="design_id"):
        calculate_coss(context=study_context, designs=duplicate_id)
    result = calculate_coss(context=study_context, designs=duplicate_size)
    assert result.optimal_design_id == "a"
    assert result.boundary_state == "both"


def test_coss_returns_uncertainty_and_complete_backend_independent_plot_data(
    study_context: StudyDesignContextV1,
) -> None:
    designs = (
        StudyDesignPointInputV1(
            design_id="n-20",
            sample_size=20,
            evsi=8.0,
            research_cost=3.0,
            enbs_standard_error=0.5,
            enbs_confidence_interval=(4.0, 6.0),
        ),
        StudyDesignPointInputV1(
            design_id="n-40",
            sample_size=40,
            evsi=17.0,
            research_cost=6.0,
            enbs_standard_error=0.75,
            enbs_confidence_interval=(9.5, 12.5),
        ),
    )
    uncertainty = SelectionUncertaintyV1(
        method="bootstrap",
        replicate_count=1_000,
        probability_by_design={"n-20": 0.2, "n-40": 0.8},
        confidence_set_design_ids=("n-40",),
    )

    result = calculate_coss(
        context=study_context,
        designs=designs,
        selection_uncertainty=uncertainty,
    )

    assert result.selection_uncertainty == uncertainty
    assert result.evaluated_designs[1].enbs_standard_error == 0.75
    assert result.plot_data.design_ids == ("n-20", "n-40")
    assert result.plot_data.sample_sizes == (20, 40)
    assert result.plot_data.evsi == (8.0, 17.0)
    assert result.plot_data.research_cost == (3.0, 6.0)
    assert result.plot_data.enbs == (5.0, 11.0)
    assert result.plot_data.enbs_lower == (4.0, 9.5)
    assert result.plot_data.enbs_upper == (6.0, 12.5)


def test_coss_reports_range_gaps_and_tolerance_aware_ties(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(
            _point("n-20", 20, 10.0, 2.0),
            _point("n-60", 60, 12.00000005, 4.0),
        ),
        declared_feasible_range=FeasibleDesignRangeV1(
            lower_sample_size=20, upper_sample_size=60, step=20
        ),
        absolute_tolerance=1e-6,
        relative_tolerance=0.0,
    )

    assert result.tied_optimal_design_ids == ("n-20", "n-60")
    assert result.optimal_design_id == "n-20"
    assert "feasible_set_has_gaps" in result.diagnostics
    assert result.declared_feasible_range is not None


def test_coss_records_unavailable_uncertainty_explicitly(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(_point("n-20", 20, 5.0, 2.0),),
    )

    assert result.selection_uncertainty.method == "unavailable"
    assert "selection_uncertainty_unavailable" in result.diagnostics


def test_evsi_evpi_efficiency_is_dimensionless_and_scale_invariant(
    study_context: StudyDesignContextV1,
) -> None:
    base = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=75.0, context=study_context),
        evpi=InformationValueInputV1(value=100.0, context=study_context),
    )
    scaled_context = study_context.model_copy(update={"population_scale": 100_000.0})
    scaled = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=750.0, context=scaled_context),
        evpi=InformationValueInputV1(value=1_000.0, context=scaled_context),
    )

    assert base.ratio == pytest.approx(0.75)
    assert scaled.ratio == pytest.approx(base.ratio)
    assert base.percentage == pytest.approx(75.0)
    assert base.status == "within_bounds"


def test_evsi_evpi_efficiency_has_explicit_zero_evpi_behavior(
    study_context: StudyDesignContextV1,
) -> None:
    result = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=0.0, context=study_context),
        evpi=InformationValueInputV1(value=0.0, context=study_context),
    )

    assert result.ratio is None
    assert result.percentage is None
    assert result.status == "undefined_zero_evpi"

    with pytest.raises(InputError, match="zero EVPI"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=1.0, context=study_context),
            evpi=InformationValueInputV1(value=0.0, context=study_context),
        )


def test_evsi_evpi_efficiency_preserves_small_monte_carlo_bound_excursions(
    study_context: StudyDesignContextV1,
) -> None:
    result = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=100.00000005, context=study_context),
        evpi=InformationValueInputV1(value=100.0, context=study_context),
        absolute_tolerance=1e-6,
        relative_tolerance=0.0,
    )

    assert result.ratio is not None
    assert result.ratio > 1.0
    assert result.status == "above_one_within_tolerance"
    assert "ratio_not_clamped" in result.diagnostics


def test_evsi_evpi_efficiency_rejects_material_bounds_and_context_mismatch(
    study_context: StudyDesignContextV1,
) -> None:
    with pytest.raises(InputError, match="theoretical"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=101.0, context=study_context),
            evpi=InformationValueInputV1(value=100.0, context=study_context),
            absolute_tolerance=1e-6,
            relative_tolerance=0.0,
        )

    incompatible = study_context.model_copy(update={"value_unit": "NZD_2026"})
    with pytest.raises(InputError, match="commensurate"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=75.0, context=study_context),
            evpi=InformationValueInputV1(value=100.0, context=incompatible),
        )


def test_evsi_evpi_efficiency_checks_lower_bound_relative_tolerance_and_inputs(
    study_context: StudyDesignContextV1,
) -> None:
    near_lower = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=-0.00005, context=study_context),
        evpi=InformationValueInputV1(value=100.0, context=study_context),
        absolute_tolerance=0.0,
        relative_tolerance=1e-6,
    )
    assert near_lower.ratio == pytest.approx(-0.0000005)
    assert near_lower.status == "below_zero_within_tolerance"

    with pytest.raises(InputError, match="EVPI"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=0.0, context=study_context),
            evpi=InformationValueInputV1(value=-1.0, context=study_context),
        )
    with pytest.raises(InputError, match="tolerance"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=1.0, context=study_context),
            evpi=InformationValueInputV1(value=2.0, context=study_context),
            absolute_tolerance=-1.0,
        )
    with pytest.raises(ValidationError, match="finite number"):
        InformationValueInputV1(value=float("nan"), context=study_context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_problem_id", "other-decision"),
        ("value_unit", "NZD_2026"),
        ("population_scale", 20_000.0),
        ("time_horizon", "2037-2046"),
        ("discounting_id", "none"),
    ],
)
def test_evsi_evpi_efficiency_rejects_every_context_mismatch(
    study_context: StudyDesignContextV1,
    field: str,
    value: str | float,
) -> None:
    incompatible = study_context.model_copy(update={field: value})
    with pytest.raises(InputError, match="commensurate"):
        evsi_evpi_efficiency(
            evsi=InformationValueInputV1(value=1.0, context=study_context),
            evpi=InformationValueInputV1(value=2.0, context=incompatible),
        )
