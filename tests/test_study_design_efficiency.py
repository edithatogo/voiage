"""Reference tests for governed COSS and study-information efficiency."""

from __future__ import annotations

import pytest

from voiage.contracts.study_design import (
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.experimental.study_design import calculate_coss


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
