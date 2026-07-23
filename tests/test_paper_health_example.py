"""Regression tests for the synthetic health example in the preprint."""

import pytest

from scripts.generate_paper_health_example import (
    calculate_example,
    normal_normal_evsi,
)


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
    assert example.enbs_immediate[1] < 0 < example.enbs_immediate[2]
    assert example.enbs_delayed[4] < 0 < example.enbs_delayed[5]


def test_normal_normal_evsi_uses_declared_equal_allocation_likelihood() -> None:
    """The study model is bounded by perfect information and rejects odd sizes."""
    example = calculate_example()

    assert normal_normal_evsi(50) == pytest.approx(63.12, abs=0.01)
    assert normal_normal_evsi(1_200) < example.evppi_effect
    with pytest.raises(ValueError, match="positive even integer"):
        normal_normal_evsi(51)


def test_reported_manuscript_results_are_reproducible() -> None:
    """The prose-level results remain tied to the fixed-seed calculation."""
    example = calculate_example()
    reference_index = list(example.thresholds).index(50_000.0)

    assert example.probability_cost_effective[reference_index] == pytest.approx(0.4924)
    assert example.evpi == pytest.approx(644.15, abs=0.01)
    assert example.evppi_effect == pytest.approx(589.67, abs=0.01)
    assert example.evppi_cost == pytest.approx(249.59, abs=0.01)
