"""Regression tests for the synthetic health example in the preprint."""

from math import pi, sqrt

import pytest

from scripts.generate_paper_health_example import (
    COST_PRIOR_SD,
    EFFECT_PRIOR_SD,
    OUTCOME_SD,
    REFERENCE_WTP,
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


def test_reported_values_agree_with_independent_analytical_references() -> None:
    """Known Normal-model expectations independently bound the simulation."""
    example = calculate_example()
    expected_evpi = sqrt(
        (REFERENCE_WTP * EFFECT_PRIOR_SD) ** 2 + COST_PRIOR_SD**2
    ) / sqrt(2 * pi)
    expected_effect_evppi = REFERENCE_WTP * EFFECT_PRIOR_SD / sqrt(2 * pi)
    expected_cost_evppi = COST_PRIOR_SD / sqrt(2 * pi)

    assert expected_evpi == pytest.approx(652.1821719528144)
    assert expected_effect_evppi == pytest.approx(598.4134206021490)
    assert expected_cost_evppi == pytest.approx(259.3124822609312)
    assert example.bootstrap_intervals["evpi"][0] < expected_evpi
    assert example.bootstrap_intervals["evpi"][1] > expected_evpi
    assert example.bootstrap_intervals["evppi_effect"][0] < expected_effect_evppi
    assert example.bootstrap_intervals["evppi_effect"][1] > expected_effect_evppi
    assert example.bootstrap_intervals["evppi_cost"][0] < expected_cost_evppi
    assert example.bootstrap_intervals["evppi_cost"][1] > expected_cost_evppi

    expected_evsi = []
    for sample_size in example.sample_sizes:
        sampling_variance = 4 * OUTCOME_SD**2 / sample_size
        preposterior_variance = (
            EFFECT_PRIOR_SD**4 / (EFFECT_PRIOR_SD**2 + sampling_variance)
        )
        expected_evsi.append(
            REFERENCE_WTP * sqrt(preposterior_variance) / sqrt(2 * pi)
        )
    assert example.evsi_per_person == pytest.approx(expected_evsi, abs=1e-10)
    assert all(
        0 <= value <= expected_effect_evppi <= expected_evpi
        for value in example.evsi_per_person
    )


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
    assert scenarios["One-year delay; 80% value realisation"][2] < 0
    assert scenarios["One-year delay; 80% value realisation"][3] > 0
    assert all(scenarios["Three-year delay; 40% value realisation"] < 0)


def test_bootstrap_requires_enough_replicates() -> None:
    """Small bootstrap runs are rejected as uninformative."""
    example = calculate_example()
    with pytest.raises(ValueError, match="at least 20"):
        _bootstrap_intervals(
            example.thresholds[:10],
            example.thresholds[:10],
            replicates=19,
        )
