"""Regression tests for the synthetic health example in the preprint."""

from scripts.generate_paper_health_example import calculate_example


def test_health_example_is_bounded_and_decision_relevant() -> None:
    """The illustrative outputs retain their intended economic interpretation."""
    example = calculate_example()

    assert 0 <= example.evppi_cost <= example.evpi
    assert 0 <= example.evppi_effect <= example.evpi
    assert example.evppi_effect > example.evppi_cost
    assert all(
        0 <= probability <= 1 for probability in example.probability_cost_effective
    )
    assert all(
        later >= earlier
        for earlier, later in zip(
            example.evsi_per_person[:-1],
            example.evsi_per_person[1:],
            strict=True,
        )
    )
    assert all(
        immediate > delayed
        for immediate, delayed in zip(
            example.enbs_immediate, example.enbs_delayed, strict=True
        )
    )
