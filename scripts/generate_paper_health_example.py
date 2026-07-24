"""Generate the deterministic synthetic health example used in the preprint."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import enbs, normal_normal_two_arm_evsi
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
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20260724


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
    preference_reference: float
    preference_mcse: float
    bootstrap_intervals: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SensitivityScenario:
    """One declared scenario for study-value sensitivity analysis."""

    name: str
    outcome_sd: float = OUTCOME_SD
    annual_population: float = ANNUAL_POPULATION
    fixed_cost: float = STUDY_FIXED_COST
    delay_years: int = 0
    value_realisation: float = 1.0


def normal_normal_evsi(
    total_sample_size: int,
    *,
    outcome_sd: float = OUTCOME_SD,
) -> float:
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
    if outcome_sd <= 0:
        raise ValueError("outcome_sd must be positive")
    return normal_normal_two_arm_evsi(
        prior_mean=EFFECT_MEAN,
        prior_standard_deviation=EFFECT_PRIOR_SD,
        outcome_standard_deviation=outcome_sd,
        total_sample_size=total_sample_size,
        net_benefit_slope=REFERENCE_WTP,
        net_benefit_intercept=-COST_MEAN,
    )


def _bootstrap_intervals(
    effect: np.ndarray,
    cost: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, tuple[float, float]]:
    """Estimate simulation uncertainty for reported PSA quantities.

    The bootstrap resamples paired probabilistic-analysis draws. EVPPI is
    recalculated with the same package estimator used for the point estimates.
    """
    if replicates < 20:
        raise ValueError("replicates must be at least 20")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = {
        "probability_preferred": np.empty(replicates),
        "evpi": np.empty(replicates),
        "evppi_effect": np.empty(replicates),
        "evppi_cost": np.empty(replicates),
    }
    sample_count = len(effect)
    for replicate in range(replicates):
        index = rng.integers(0, sample_count, sample_count)
        sampled_effect = effect[index]
        sampled_cost = cost[index]
        sampled_parameters = ParameterSet.from_numpy_or_dict(
            {
                "incremental_qaly": sampled_effect,
                "incremental_cost": sampled_cost,
            }
        )
        sampled_nb = np.column_stack(
            [
                np.zeros(sample_count),
                REFERENCE_WTP * sampled_effect - sampled_cost,
            ]
        )
        estimates["probability_preferred"][replicate] = np.mean(sampled_nb[:, 1] > 0.0)
        estimates["evpi"][replicate] = evpi(sampled_nb)
        estimates["evppi_effect"][replicate] = evppi(
            sampled_nb,
            sampled_parameters,
            ["incremental_qaly"],
        )
        estimates["evppi_cost"][replicate] = evppi(
            sampled_nb,
            sampled_parameters,
            ["incremental_cost"],
        )
    return {
        name: tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
        for name, values in estimates.items()
    }


@lru_cache(maxsize=1)
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
    reference_index = int(np.flatnonzero(thresholds == REFERENCE_WTP)[0])
    preference_reference = float(probability_cost_effective[reference_index])
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
        preference_reference=preference_reference,
        preference_mcse=sqrt(
            preference_reference * (1.0 - preference_reference) / N_PSA
        ),
        bootstrap_intervals=_bootstrap_intervals(effect, cost),
    )


def calculate_sensitivity() -> list[tuple[SensitivityScenario, np.ndarray]]:
    """Calculate ENBS under prespecified sensitivity scenarios."""
    scenarios = [
        SensitivityScenario("Base case"),
        SensitivityScenario("Outcome SD 0.75", outcome_sd=0.75),
        SensitivityScenario("Outcome SD 1.25", outcome_sd=1.25),
        SensitivityScenario("Annual population 1,000", annual_population=1_000),
        SensitivityScenario("Annual population 1,600", annual_population=1_600),
        SensitivityScenario("Fixed study cost 0.9m", fixed_cost=900_000),
        SensitivityScenario("Fixed study cost 1.5m", fixed_cost=1_500_000),
        SensitivityScenario(
            "One-year delay; 80% value realisation",
            delay_years=1,
            value_realisation=0.8,
        ),
        SensitivityScenario(
            "Three-year delay; 40% value realisation",
            delay_years=3,
            value_realisation=0.4,
        ),
    ]
    sample_sizes = np.array([50, 100, 200, 400, 800, 1_200])
    output: list[tuple[SensitivityScenario, np.ndarray]] = []
    for scenario in scenarios:
        opportunities = (
            scenario.value_realisation
            * scenario.annual_population
            * sum(
                (1.0 + DISCOUNT_RATE) ** -year
                for year in range(
                    1 + scenario.delay_years,
                    TIME_HORIZON + 1,
                )
            )
        )
        values = np.array(
            [
                normal_normal_evsi(int(size), outcome_sd=scenario.outcome_sd)
                * opportunities
                - (scenario.fixed_cost + STUDY_COST_PER_PARTICIPANT * size)
                for size in sample_sizes
            ]
        )
        output.append((scenario, values))
    return output


def write_results(
    example: HealthExample,
    sensitivity: list[tuple[SensitivityScenario, np.ndarray]],
    output_directory: Path,
) -> None:
    """Write machine-readable values behind the manuscript figure and prose."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "synthetic_health_example_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "metric",
                "estimate",
                "monte_carlo_standard_error",
                "bootstrap_95pct_lower",
                "bootstrap_95pct_upper",
                "unit",
            ],
        )
        writer.writeheader()
        summary_values = [
            (
                "probability_preferred",
                example.preference_reference,
                example.preference_mcse,
                "proportion",
            ),
            ("evpi", example.evpi, "", "value_units_per_person"),
            ("evppi_effect", example.evppi_effect, "", "value_units_per_person"),
            ("evppi_cost", example.evppi_cost, "", "value_units_per_person"),
        ]
        for metric, estimate, standard_error, unit in summary_values:
            lower, upper = example.bootstrap_intervals[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "estimate": f"{estimate:.8f}",
                    "monte_carlo_standard_error": (
                        f"{standard_error:.8f}" if standard_error != "" else ""
                    ),
                    "bootstrap_95pct_lower": f"{lower:.8f}",
                    "bootstrap_95pct_upper": f"{upper:.8f}",
                    "unit": unit,
                }
            )
    with (output_directory / "synthetic_health_example_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "sample_size",
                "evsi_per_person",
                "research_cost",
                "enbs_immediate_full_realisation",
                "enbs_two_year_delay_60pct_realisation",
            ],
        )
        writer.writeheader()
        for index, sample_size in enumerate(example.sample_sizes):
            writer.writerow(
                {
                    "sample_size": int(sample_size),
                    "evsi_per_person": f"{example.evsi_per_person[index]:.6f}",
                    "research_cost": f"{example.research_cost[index]:.2f}",
                    "enbs_immediate_full_realisation": (
                        f"{example.enbs_immediate[index]:.2f}"
                    ),
                    "enbs_two_year_delay_60pct_realisation": (
                        f"{example.enbs_delayed[index]:.2f}"
                    ),
                }
            )
    with (output_directory / "synthetic_health_example_sensitivity.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "scenario",
                "outcome_sd",
                "annual_population",
                "fixed_cost",
                "delay_years",
                "value_realisation",
                "sample_size",
                "enbs",
            ],
        )
        writer.writeheader()
        for scenario, values in sensitivity:
            for sample_size, value in zip(example.sample_sizes, values, strict=True):
                writer.writerow(
                    {
                        "scenario": scenario.name,
                        "outcome_sd": scenario.outcome_sd,
                        "annual_population": scenario.annual_population,
                        "fixed_cost": scenario.fixed_cost,
                        "delay_years": scenario.delay_years,
                        "value_realisation": scenario.value_realisation,
                        "sample_size": int(sample_size),
                        "enbs": f"{value:.2f}",
                    }
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
    purple = "#9B3F75"

    axis_a.plot(
        example.thresholds / 1_000,
        example.probability_cost_effective,
        color=blue,
        linewidth=2.2,
    )
    axis_a.axvline(REFERENCE_WTP / 1_000, color="#555555", linestyle="--")
    axis_a.set(
        xlabel="Value per QALY (thousand value units)",
        ylabel="Probability programme is preferred",
        ylim=(0, 1),
        title="A  Decision uncertainty",
    )

    partial_values = [example.evppi_effect, example.evppi_cost]
    axis_b.bar(
        ["Health gain", "Programme cost"],
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
        ylabel="Value units per person",
        title="B  Priorities for further evidence",
    )
    axis_b.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=9,
    )

    axis_c.axhline(0, color="#555555", linewidth=1)
    axis_c.plot(
        example.sample_sizes,
        example.enbs_immediate / 1_000_000,
        color=blue,
        marker="o",
        label="Immediate evidence; full value realisation",
    )
    axis_c.plot(
        example.sample_sizes,
        example.enbs_delayed / 1_000_000,
        color=orange,
        marker="s",
        linestyle="--",
        label="Two-year delay; 60% value realisation",
    )
    axis_c.set(
        xlabel="Total study sample size",
        ylabel="Population ENBS (million value units)",
        title="C  Expected net benefit of sampling",
    )
    axis_c.legend(frameon=False, fontsize=9)

    figure.suptitle(
        "Synthetic health example: a programme compared with current practice",
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
    sensitivity = calculate_sensitivity()
    output_stem = Path("paper/figures/synthetic_health_example")
    render(example, output_stem)
    write_results(example, sensitivity, Path("paper/data"))
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"EVPI={example.evpi:.2f}")
    print(f"EVPPI(effect)={example.evppi_effect:.2f}")
    print(f"EVPPI(cost)={example.evppi_cost:.2f}")
    print(
        "95% bootstrap intervals="
        + ", ".join(
            f"{name}[{interval[0]:.2f}, {interval[1]:.2f}]"
            for name, interval in example.bootstrap_intervals.items()
        )
    )
    print("EVSI=" + ", ".join(f"{value:.2f}" for value in example.evsi_per_person))
    print(
        "ENBS immediate="
        + ", ".join(f"{value:.0f}" for value in example.enbs_immediate)
    )
    print("ENBS delayed=" + ", ".join(f"{value:.0f}" for value in example.enbs_delayed))


if __name__ == "__main__":
    main()
