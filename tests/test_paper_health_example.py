"""Regression tests for the synthetic health example in the preprint."""

import pytest

from scripts.generate_paper_health_example import (
    _bootstrap_intervals,
    calculate_example,
    calculate_sensitivity,
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
    with pytest.raises(ValueError, match="outcome_sd must be positive"):
        normal_normal_evsi(50, outcome_sd=0)


def test_reported_manuscript_results_are_reproducible() -> None:
    """The prose-level results remain tied to the fixed-seed calculation."""
    example = calculate_example()
    reference_index = list(example.thresholds).index(50_000.0)

    assert example.probability_cost_effective[reference_index] == pytest.approx(0.4924)
    assert example.evpi == pytest.approx(644.15, abs=0.01)
    assert example.evppi_effect == pytest.approx(589.67, abs=0.01)
    assert example.evppi_cost == pytest.approx(249.59, abs=0.01)
    assert example.preference_mcse == pytest.approx(0.005, abs=0.0001)


def test_bootstrap_and_sensitivity_are_declared_and_decision_relevant() -> None:
    """Uncertainty and sensitivity outputs remain bounded and interpretable."""
    example = calculate_example()
    scenarios = {scenario.name: values for scenario, values in calculate_sensitivity()}

    for metric, point in (
        ("probability_preferred", example.preference_reference),
        ("evpi", example.evpi),
        ("evppi_effect", example.evppi_effect),
        ("evppi_cost", example.evppi_cost),
    ):
        lower, upper = example.bootstrap_intervals[metric]
        assert lower < point < upper
    assert scenarios["One-year delay; 80% uptake"][2] < 0
    assert scenarios["One-year delay; 80% uptake"][3] > 0
    assert all(scenarios["Three-year delay; 40% uptake"] < 0)


def test_bootstrap_requires_enough_replicates() -> None:
    """Small bootstrap runs are rejected as uninformative."""
    example = calculate_example()
    with pytest.raises(ValueError, match="at least 20"):
        _bootstrap_intervals(
            example.thresholds[:10],
            example.thresholds[:10],
            replicates=19,
        )
