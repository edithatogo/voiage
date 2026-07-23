"""Generate the deterministic synthetic health example used in the preprint."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import enbs
from voiage.schema import ParameterSet

SEED = 20260723
N_PSA = 10_000
REFERENCE_WTP = 50_000.0
EFFECT_MEAN = 0.060
EFFECT_PRIOR_SD = 0.030
COST_MEAN = 3_000.0
COST_PRIOR_SD = 650.0
OUTCOME_SD = 1.0
ANNUAL_POPULATION = 1_300.0
TIME_HORIZON = 10
DISCOUNT_RATE = 0.03
STUDY_FIXED_COST = 1_200_000.0
STUDY_COST_PER_PARTICIPANT = 100.0


@dataclass(frozen=True)
class HealthExample:
    """Calculated values for the synthetic treatment-adoption example."""

    thresholds: np.ndarray
    probability_cost_effective: np.ndarray
    evpi: float
    evppi_effect: float
    evppi_cost: float
    sample_sizes: np.ndarray
    evsi_per_person: np.ndarray
    research_cost: np.ndarray
    enbs_immediate: np.ndarray
    enbs_delayed: np.ndarray


def _expected_positive_normal(mean: float, standard_deviation: float) -> float:
    """Return E[max(0, X)] for a normally distributed random variable X."""
    if standard_deviation <= 0:
        return max(0.0, mean)
    z_score = mean / standard_deviation
    density = exp(-(z_score**2) / 2.0) / sqrt(2.0 * pi)
    distribution = 0.5 * (1.0 + erf(z_score / sqrt(2.0)))
    return standard_deviation * density + mean * distribution


def normal_normal_evsi(total_sample_size: int) -> float:
    """Calculate EVSI for the declared equal-allocation normal study model.

    The uncertain incremental health effect has a Normal prior. A two-arm
    study allocates half of the total sample to each strategy, with a common
    individual outcome standard deviation. The difference in sample means is
    therefore Normal around the true incremental effect with variance
    ``4 * outcome_variance / total_sample_size``. The preposterior distribution
    of the posterior mean is available analytically.
    """
    if total_sample_size <= 0 or total_sample_size % 2:
        raise ValueError("total_sample_size must be a positive even integer")

    prior_variance = EFFECT_PRIOR_SD**2
    sampling_variance = 4.0 * OUTCOME_SD**2 / total_sample_size
    posterior_mean_variance = prior_variance**2 / (prior_variance + sampling_variance)
    incremental_nb_mean = REFERENCE_WTP * EFFECT_MEAN - COST_MEAN
    incremental_nb_sd = REFERENCE_WTP * sqrt(posterior_mean_variance)
    expected_after_study = _expected_positive_normal(
        incremental_nb_mean,
        incremental_nb_sd,
    )
    current_value = max(0.0, incremental_nb_mean)
    return expected_after_study - current_value


def calculate_example() -> HealthExample:
    """Calculate one internally consistent, explicitly synthetic example."""
    rng = np.random.default_rng(SEED)
    effect = rng.normal(EFFECT_MEAN, EFFECT_PRIOR_SD, N_PSA)
    cost = rng.normal(COST_MEAN, COST_PRIOR_SD, N_PSA)
    parameters = ParameterSet.from_numpy_or_dict(
        {"incremental_qaly": effect, "incremental_cost": cost}
    )

    thresholds = np.arange(0.0, 100_001.0, 2_500.0)
    probability_cost_effective = np.array(
        [np.mean(threshold * effect - cost > 0.0) for threshold in thresholds]
    )
    reference_nb = np.column_stack([np.zeros(N_PSA), REFERENCE_WTP * effect - cost])
    perfect_information = evpi(reference_nb)
    effect_evppi = evppi(
        reference_nb,
        parameters,
        ["incremental_qaly"],
    )
    cost_evppi = evppi(
        reference_nb,
        parameters,
        ["incremental_cost"],
    )

    sample_sizes = np.array([50, 100, 200, 400, 800, 1_200])
    evsi_per_person = np.array([normal_normal_evsi(int(size)) for size in sample_sizes])

    discounted_opportunities = ANNUAL_POPULATION * sum(
        (1.0 + DISCOUNT_RATE) ** -year for year in range(1, TIME_HORIZON + 1)
    )
    delayed_opportunities = (
        0.60
        * ANNUAL_POPULATION
        * sum((1.0 + DISCOUNT_RATE) ** -year for year in range(3, TIME_HORIZON + 1))
    )
    research_cost = STUDY_FIXED_COST + STUDY_COST_PER_PARTICIPANT * sample_sizes
    enbs_immediate = np.array(
        [
            enbs(value * discounted_opportunities, cost_value)
            for value, cost_value in zip(evsi_per_person, research_cost, strict=True)
        ]
    )
    enbs_delayed = np.array(
        [
            enbs(value * delayed_opportunities, cost_value)
            for value, cost_value in zip(evsi_per_person, research_cost, strict=True)
        ]
    )
    return HealthExample(
        thresholds=thresholds,
        probability_cost_effective=probability_cost_effective,
        evpi=perfect_information,
        evppi_effect=effect_evppi,
        evppi_cost=cost_evppi,
        sample_sizes=sample_sizes,
        evsi_per_person=evsi_per_person,
        research_cost=research_cost,
        enbs_immediate=enbs_immediate,
        enbs_delayed=enbs_delayed,
    )


def render(example: HealthExample, output_stem: Path) -> None:
    """Render accessible vector and raster figures for the manuscript."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(7.2, 6.8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])

    blue = "#0072B2"
    orange = "#E69F00"
    green = "#009E73"
    purple = "#CC79A7"

    axis_a.plot(
        example.thresholds / 1_000,
        example.probability_cost_effective,
        color=blue,
        linewidth=2.2,
    )
    axis_a.axvline(REFERENCE_WTP / 1_000, color="#555555", linestyle="--")
    axis_a.set(
        xlabel="Value placed on one QALY (thousand value units)",
        ylabel="Probability new programme is preferred",
        ylim=(0, 1),
        title="A  Decision uncertainty",
    )

    partial_values = [example.evppi_effect, example.evppi_cost]
    axis_b.bar(
        ["Health effect", "Programme cost"],
        partial_values,
        color=[green, orange],
    )
    axis_b.axhline(
        example.evpi,
        color=purple,
        linestyle="--",
        label="EVPI (all uncertainty)",
    )
    axis_b.set(
        ylabel="Value per person",
        title="B  Priorities for further evidence",
    )
    axis_b.legend(frameon=False, fontsize=8)

    axis_c.axhline(0, color="#555555", linewidth=1)
    axis_c.plot(
        example.sample_sizes,
        example.enbs_immediate / 1_000_000,
        color=blue,
        marker="o",
        label="Immediate results; full uptake",
    )
    axis_c.plot(
        example.sample_sizes,
        example.enbs_delayed / 1_000_000,
        color=orange,
        marker="s",
        linestyle="--",
        label="Two-year delay; 60% uptake",
    )
    axis_c.set(
        xlabel="Total study sample size",
        ylabel="Population ENBS (million value units)",
        title="C  Study value depends on delivery",
    )
    axis_c.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Synthetic health example: a new programme compared with standard care",
        fontsize=11,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        metadata={"Software": "voiage"},
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        metadata={
            "Creator": "voiage",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main() -> None:
    """Generate the tracked manuscript figure and print review values."""
    example = calculate_example()
    output_stem = Path("paper/figures/synthetic_health_example")
    render(example, output_stem)
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"EVPI={example.evpi:.2f}")
    print(f"EVPPI(effect)={example.evppi_effect:.2f}")
    print(f"EVPPI(cost)={example.evppi_cost:.2f}")
    print("EVSI=" + ", ".join(f"{value:.2f}" for value in example.evsi_per_person))
    print(
        "ENBS immediate="
        + ", ".join(f"{value:.0f}" for value in example.enbs_immediate)
    )
    print("ENBS delayed=" + ", ".join(f"{value:.0f}" for value in example.enbs_delayed))


if __name__ == "__main__":
    main()
