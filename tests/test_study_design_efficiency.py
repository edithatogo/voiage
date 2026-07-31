"""Reference tests for governed COSS and study-information efficiency."""

from __future__ import annotations

from copy import deepcopy
import json

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
import pytest

from voiage.contracts.study_design import (
    CossCurvePointV1,
    CossPlotDataV1,
    FeasibleDesignRangeV1,
    InformationValueInputV1,
    SelectionUncertaintyV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.exceptions import InputError
from voiage.experimental.study_design import calculate_coss, evsi_evpi_efficiency


def _context() -> StudyDesignContextV1:
    return StudyDesignContextV1(
        decision_problem_id="property-decision",
        value_unit="unit",
        population_scale=1.0,
        time_horizon="horizon",
        discounting_id="none",
        study_model_id="enumerated",
        cost_model_id="enumerated",
        random_seed=571,
    )


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
            estimator_provenance={"method": "bootstrap", "replicates": 1_000},
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
    assert result.evaluated_designs[0].estimator_provenance["method"] == "bootstrap"
    assert result.estimator_provenance["runtime"] == "rust"


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


def test_complete_selection_probability_map_must_be_normalized(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(
            _point("n-20", 20, 5.0, 2.0),
            _point("n-40", 40, 8.0, 3.0),
        ),
        selection_uncertainty=SelectionUncertaintyV1(
            method="bootstrap",
            replicate_count=100,
            probability_by_design={"n-20": 0.4, "n-40": 0.6},
        ),
    )
    payload = result.model_dump(mode="json")
    payload["selection_uncertainty"]["probability_by_design"] = {
        "n-20": 0.25,
        "n-40": 0.25,
    }

    with pytest.raises(ValidationError, match="complete selection probability"):
        type(result).model_validate_json(json.dumps(payload))


def test_coss_reports_declared_range_disagreement_and_infeasible_members(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(
            _point("n-20", 20, 5.0, 2.0),
            _point("n-40", 40, 8.0, 3.0),
            _point("n-60", 60, 10.0, 4.0, feasible=False),
        ),
        declared_feasible_range=FeasibleDesignRangeV1(
            lower_sample_size=20, upper_sample_size=100, step=20
        ),
    )

    assert "feasible_range_set_disagreement" in result.diagnostics
    assert "infeasible_design_within_declared_range" in result.diagnostics
    assert "feasible_set_has_gaps" in result.diagnostics


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


def test_efficiency_deserialization_rejects_impossible_zero_and_bound_states(
    study_context: StudyDesignContextV1,
) -> None:
    zero = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=0.0, context=study_context),
        evpi=InformationValueInputV1(value=0.0, context=study_context),
    )
    zero_payload = zero.model_dump(mode="json")
    zero_payload["evsi"] = 999.0
    with pytest.raises(ValidationError, match="zero EVPI requires EVSI"):
        type(zero).model_validate_json(json.dumps(zero_payload))

    bounded = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=75.0, context=study_context),
        evpi=InformationValueInputV1(value=100.0, context=study_context),
    )
    upper_payload = bounded.model_dump(mode="json")
    upper_payload.update(
        evsi=101.0,
        ratio=1.01,
        percentage=101.0,
        status="above_one_within_tolerance",
    )
    with pytest.raises(ValidationError, match="materially exceeds EVPI"):
        type(bounded).model_validate_json(json.dumps(upper_payload))

    lower_payload = bounded.model_dump(mode="json")
    lower_payload.update(
        evsi=-1.0,
        ratio=-0.01,
        percentage=-1.0,
        status="below_zero_within_tolerance",
    )
    with pytest.raises(ValidationError, match="materially below zero"):
        type(bounded).model_validate_json(json.dumps(lower_payload))


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


@given(
    evsi_values=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=1,
        max_size=30,
    ),
    costs=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=1,
        max_size=30,
    ),
    shift=st.integers(min_value=0, max_value=100_000),
)
@settings(max_examples=40, deadline=None)
def test_coss_matches_independent_argmax_and_common_value_shift(
    evsi_values: list[int], costs: list[int], shift: int
) -> None:
    size = min(len(evsi_values), len(costs))
    designs = tuple(
        _point(f"d-{index}", index + 1, float(evsi_values[index]), float(costs[index]))
        for index in range(size)
    )
    shifted = tuple(
        _point(
            point.design_id,
            point.sample_size,
            point.evsi + shift,
            point.research_cost + shift,
        )
        for point in designs
    )

    result = calculate_coss(context=_context(), designs=designs)
    shifted_result = calculate_coss(context=_context(), designs=shifted)
    expected = tuple(point.evsi - point.research_cost for point in designs)

    assert tuple(point.enbs for point in result.evaluated_designs) == pytest.approx(
        expected
    )
    assert tuple(
        point.enbs for point in shifted_result.evaluated_designs
    ) == pytest.approx(expected)
    assert shifted_result.optimal_design_id == result.optimal_design_id


@given(
    evpi=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    fraction=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=40, deadline=None)
def test_information_efficiency_property_is_scale_invariant(
    evpi: float, fraction: float, scale: float
) -> None:
    base = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=evpi * fraction, context=_context()),
        evpi=InformationValueInputV1(value=evpi, context=_context()),
        absolute_tolerance=0.0,
        relative_tolerance=1e-12,
    )
    scaled = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=evpi * fraction * scale, context=_context()),
        evpi=InformationValueInputV1(value=evpi * scale, context=_context()),
        absolute_tolerance=0.0,
        relative_tolerance=1e-12,
    )
    assert scaled.ratio == pytest.approx(base.ratio)


def test_coss_contract_round_trips_canonical_json_and_is_frozen(
    study_context: StudyDesignContextV1,
) -> None:
    result = calculate_coss(
        context=study_context,
        designs=(_point("n-20", 20, 5.0, 2.0),),
    )
    payload = result.model_dump_json()

    assert type(result).model_validate_json(payload) == result
    with pytest.raises(ValidationError, match="frozen"):
        result.maximum_enbs = 99.0  # type: ignore[misc]


def test_coss_rejects_invalid_selection_uncertainty(
    study_context: StudyDesignContextV1,
) -> None:
    uncertainty = SelectionUncertaintyV1(
        method="bootstrap",
        replicate_count=20,
        probability_by_design={"n-20": 0.4, "n-40": 0.5},
    )
    with pytest.raises(InputError, match="sum to one"):
        calculate_coss(
            context=study_context,
            designs=(_point("n-20", 20, 5.0, 2.0), _point("n-40", 40, 8.0, 3.0)),
            selection_uncertainty=uncertainty,
        )


def test_coss_fails_closed_on_a_malformed_native_result(
    monkeypatch: pytest.MonkeyPatch,
    study_context: StudyDesignContextV1,
) -> None:
    from voiage.experimental import study_design as module

    monkeypatch.setattr(
        module._runtime, "compute_coss", lambda **_: {"contract_version": "9"}
    )
    with pytest.raises(InputError, match="native COSS result"):
        module.calculate_coss(
            context=study_context,
            designs=(_point("n-20", 20, 5.0, 2.0),),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "contract_version": "1.0.0",
            "estimator": "native",
            "enbs": [],
            "feasible_indices": [0],
            "tied_indices": [0],
            "optimal_index": 0,
            "maximum_enbs": 3.0,
            "boundary_state": "both",
        },
        {
            "contract_version": "1.0.0",
            "estimator": "native",
            "enbs": [3.0],
            "feasible_indices": [0],
            "tied_indices": [-1],
            "optimal_index": -1,
            "maximum_enbs": 3.0,
            "boundary_state": "both",
        },
        {
            "contract_version": "1.0.0",
            "estimator": "native",
            "enbs": [3.0],
            "feasible_indices": [0],
            "tied_indices": [0],
            "optimal_index": True,
            "maximum_enbs": 3.0,
            "boundary_state": "both",
        },
        {
            "contract_version": "1.0.0",
            "estimator": "native",
            "enbs": [3.0],
            "feasible_indices": [0],
            "tied_indices": [0],
            "optimal_index": 0,
            "maximum_enbs": 999.0,
            "boundary_state": "upper",
        },
    ],
)
def test_coss_fails_closed_on_malformed_native_vectors_and_indices(
    monkeypatch: pytest.MonkeyPatch,
    study_context: StudyDesignContextV1,
    payload: dict[str, object],
) -> None:
    from voiage.experimental import study_design as module

    monkeypatch.setattr(module._runtime, "compute_coss", lambda **_: payload)
    with pytest.raises(InputError, match="native COSS result"):
        module.calculate_coss(
            context=study_context,
            designs=(_point("n-20", 20, 5.0, 2.0),),
        )


def test_result_contracts_reject_corrupted_serialized_relations(
    study_context: StudyDesignContextV1,
) -> None:
    coss = calculate_coss(
        context=study_context,
        designs=(_point("n-20", 20, 5.0, 2.0),),
    )
    coss_payload = coss.model_dump(mode="json")
    corrupted_coss = deepcopy(coss_payload)
    corrupted_coss["optimal_design_id"] = "ghost"
    corrupted_coss["maximum_enbs"] = 999.0
    corrupted_coss["plot_data"]["optimal_design_id"] = "different"
    with pytest.raises(ValidationError):
        type(coss).model_validate_json(json.dumps(corrupted_coss))

    efficiency = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=5.0, context=study_context),
        evpi=InformationValueInputV1(value=10.0, context=study_context),
    )
    efficiency_payload = efficiency.model_dump(mode="json")
    efficiency_payload["ratio"] = 0.9
    efficiency_payload["percentage"] = 1.0
    efficiency_payload["status"] = "above_one_within_tolerance"
    with pytest.raises(ValidationError):
        type(efficiency).model_validate_json(json.dumps(efficiency_payload))


def test_selection_uncertainty_rejects_excess_mass_and_infeasible_probability(
    study_context: StudyDesignContextV1,
) -> None:
    with pytest.raises(ValidationError, match="mass"):
        SelectionUncertaintyV1(
            method="bootstrap",
            replicate_count=10,
            probability_by_design={"a": 0.8, "b": 0.8},
        )
    uncertainty = SelectionUncertaintyV1(
        method="bootstrap",
        replicate_count=10,
        probability_by_design={"a": 0.5, "b": 0.5},
    )
    with pytest.raises(InputError, match="infeasible"):
        calculate_coss(
            context=study_context,
            designs=(
                _point("a", 20, 5.0, 2.0),
                _point("b", 40, 8.0, 3.0, feasible=False),
            ),
            selection_uncertainty=uncertainty,
        )


def test_contract_leaf_validators_reject_corruption() -> None:
    with pytest.raises(ValidationError, match="lower_sample_size"):
        FeasibleDesignRangeV1(lower_sample_size=20, upper_sample_size=10)
    with pytest.raises(ValidationError, match="confidence interval"):
        StudyDesignPointInputV1(
            design_id="a",
            sample_size=20,
            evsi=5.0,
            research_cost=2.0,
            enbs_confidence_interval=(4.0, 3.0),
        )
    with pytest.raises(ValidationError, match="enbs must equal"):
        CossCurvePointV1(
            design_id="a",
            sample_size=20,
            evsi=5.0,
            research_cost=2.0,
            enbs=99.0,
            feasible=True,
        )
    with pytest.raises(ValidationError, match="confidence interval"):
        CossCurvePointV1(
            design_id="a",
            sample_size=20,
            evsi=5.0,
            research_cost=2.0,
            enbs=3.0,
            feasible=True,
            enbs_confidence_interval=(4.0, 3.0),
        )
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        SelectionUncertaintyV1(method="analytic", probability_by_design={"a": -0.1})
    with pytest.raises(ValidationError, match="replicate_count"):
        SelectionUncertaintyV1(method="monte_carlo")
    with pytest.raises(ValidationError, match="unavailable"):
        SelectionUncertaintyV1(method="unavailable", replicate_count=2)
    with pytest.raises(ValidationError, match="equal lengths"):
        CossPlotDataV1(
            design_ids=("a",),
            sample_sizes=(),
            evsi=(1.0,),
            research_cost=(1.0,),
            enbs=(0.0,),
            feasible=(True,),
            enbs_lower=(None,),
            enbs_upper=(None,),
            boundary_state="both",
        )


def test_coss_result_validator_rejects_relational_corruption(
    study_context: StudyDesignContextV1,
) -> None:
    valid = calculate_coss(
        context=study_context,
        designs=(_point("a", 20, 5.0, 2.0), _point("b", 40, 8.0, 3.0)),
    ).model_dump(mode="json")
    corruptions: list[dict[str, object]] = []

    payload = deepcopy(valid)
    payload["evaluated_designs"] = []
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["evaluated_designs"][1]["design_id"] = "a"
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["feasible_sample_sizes"] = [999]
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["tied_optimal_design_ids"] = ["ghost"]
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["maximum_enbs"] = 999.0
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["tied_optimal_design_ids"] = ["a", "b"]
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["optimal_design_id"] = "a"
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["optimal_sample_size"] = 999
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["boundary_state"] = "lower"
    payload["plot_data"]["boundary_state"] = "lower"
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["plot_data"]["enbs"] = [99.0, 5.0]
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["estimator_provenance"] = {}
    corruptions.append(payload)

    for corrupted in corruptions:
        with pytest.raises(ValidationError):
            type(
                calculate_coss(
                    context=study_context,
                    designs=(_point("a", 20, 5.0, 2.0), _point("b", 40, 8.0, 3.0)),
                )
            ).model_validate_json(json.dumps(corrupted))

    no_feasible = calculate_coss(
        context=study_context,
        designs=(_point("a", 20, 5.0, 2.0, feasible=False),),
    ).model_dump(mode="json")
    no_feasible["optimal_design_id"] = "a"
    with pytest.raises(ValidationError, match="cannot have an optimum"):
        type(
            calculate_coss(
                context=study_context,
                designs=(_point("a", 20, 5.0, 2.0, feasible=False),),
            )
        ).model_validate_json(json.dumps(no_feasible))


def test_native_coss_parser_rejects_every_structural_corruption(
    monkeypatch: pytest.MonkeyPatch,
    study_context: StudyDesignContextV1,
) -> None:
    from voiage.experimental import study_design as module

    valid: dict[str, object] = {
        "contract_version": "1.0.0",
        "estimator": "native",
        "enbs": [3.0, 5.0],
        "feasible_indices": [0, 1],
        "tied_indices": [1],
        "optimal_index": 1,
        "maximum_enbs": 5.0,
        "boundary_state": "upper",
    }
    corruptions: list[dict[str, object]] = []
    for key, value in (
        ("enbs", "not-a-list"),
        ("enbs", [3.0, True]),
        ("enbs", [3.0, float("nan")]),
        ("enbs", [3.0, 4.0]),
        ("feasible_indices", [0, 0]),
        ("feasible_indices", [0]),
        ("tied_indices", [0]),
        ("boundary_state", 1),
        ("maximum_enbs", True),
        ("maximum_enbs", 4.0),
        ("optimal_index", 0),
        ("boundary_state", "lower"),
        ("estimator", ""),
    ):
        payload = deepcopy(valid)
        payload[key] = value
        corruptions.append(payload)
    for payload in corruptions:
        monkeypatch.setattr(
            module._runtime,
            "compute_coss",
            lambda _payload=payload, **_: _payload,
        )
        with pytest.raises(InputError, match="native COSS result"):
            module.calculate_coss(
                context=study_context,
                designs=(_point("a", 20, 5.0, 2.0), _point("b", 40, 8.0, 3.0)),
            )


def test_public_facade_rejects_bad_container_range_and_uncertainty_references(
    study_context: StudyDesignContextV1,
) -> None:
    with pytest.raises(InputError, match="at least one"):
        calculate_coss(context=study_context, designs=())
    with pytest.raises(InputError, match="records"):
        calculate_coss(context=study_context, designs=(object(),))  # type: ignore[arg-type]
    with pytest.raises(InputError, match="declared_feasible_range"):
        calculate_coss(
            context=study_context,
            designs=(_point("a", 20, 5.0, 2.0),),
            declared_feasible_range=(20,),  # type: ignore[arg-type]
        )
    unknown_confidence = SelectionUncertaintyV1(
        method="analytic", confidence_set_design_ids=("ghost",)
    )
    with pytest.raises(InputError, match="unknown"):
        calculate_coss(
            context=study_context,
            designs=(_point("a", 20, 5.0, 2.0),),
            selection_uncertainty=unknown_confidence,
        )
    unknown_probability = SelectionUncertaintyV1(
        method="analytic", probability_by_design={"ghost": 0.5}
    )
    with pytest.raises(InputError, match="unknown"):
        calculate_coss(
            context=study_context,
            designs=(_point("a", 20, 5.0, 2.0),),
            selection_uncertainty=unknown_probability,
        )
