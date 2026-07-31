"""Microbenchmark coverage for the governed enumerated COSS path."""

from voiage.contracts.study_design import StudyDesignContextV1, StudyDesignPointInputV1
from voiage.experimental.study_design import calculate_coss


def test_coss_ten_thousand_design_benchmark(benchmark) -> None:
    """Exercise a production-sized enumerated curve without interpolation."""
    context = StudyDesignContextV1(
        decision_problem_id="benchmark",
        value_unit="unit",
        population_scale=1.0,
        time_horizon="benchmark",
        discounting_id="none",
        study_model_id="enumerated",
        cost_model_id="linear",
        random_seed=571,
    )
    designs = tuple(
        StudyDesignPointInputV1(
            design_id=f"n-{sample_size}",
            sample_size=sample_size,
            evsi=float(sample_size) ** 0.5,
            research_cost=float(sample_size) / 200.0,
        )
        for sample_size in range(1, 10_001)
    )

    result = benchmark(calculate_coss, context=context, designs=designs)

    assert len(result.evaluated_designs) == 10_000
    assert result.optimal_design_id is not None
