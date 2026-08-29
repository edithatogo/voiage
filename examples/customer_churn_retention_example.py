"""Customer Churn and Retention Strategy Worked Example.

This worked example demonstrates Value of Information (VOI) analysis applied to
a customer retention decision in subscription and telecommunications business
strategy (Issue #574).

Scenario:
A digital subscription service faces customer churn and evaluates three
candidate intervention strategies:
1. Standard Care: Status quo (no proactive retention offer).
2. Targeted Discount: Proactive 20% discount offered to high-churn-risk cohort.
3. Proactive Concierge: High-touch customer success onboarding outreach.

Uncertain parameters include:
- Base churn rate across the customer cohort
- Relative churn reduction under targeted discount
- Relative churn reduction under concierge outreach
- Customer lifetime value (CLV)
- Operational cost per contacted customer

Analysis:
- Net Benefit calculations per strategy across Monte Carlo samples
- EVPI (Expected Value of Perfect Information)
- EVPPI (Expected Value of Partial Perfect Information) on intervention effectiveness
- ENBS (Expected Net Benefit of Sampling) for pilot randomized retention trials
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voiage.domain_templates import get_domain_template
from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import enbs
from voiage.schema import DecisionProblem, Intervention, ParameterSet, ValueArray


def generate_churn_dataset(
    n_samples: int = 2000, seed: int = 42
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic Monte Carlo parameter samples for churn decision model.

    Parameters
    ----------
    n_samples : int, default=2000
        Number of Monte Carlo simulations.
    seed : int, default=42
        Random seed for deterministic reproducibility.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        (net_benefits, parameter_dict)
    """
    rng = np.random.default_rng(seed)

    # 1. Base annual churn rate (Beta distribution: mean ~ 0.18, range ~ [0.10, 0.28])
    base_churn = rng.beta(a=9.0, b=41.0, size=n_samples)

    # 2. Targeted discount relative risk reduction (Beta: mean ~ 0.25 reduction)
    discount_reduction = rng.beta(a=5.0, b=15.0, size=n_samples)

    # 3. Concierge outreach relative risk reduction (Beta: mean ~ 0.40 reduction)
    concierge_reduction = rng.beta(a=8.0, b=12.0, size=n_samples)

    # 4. Customer lifetime value in USD (LogNormal: median ~$600, range ~$350 - $1100)
    clv = rng.lognormal(mean=6.4, sigma=0.25, size=n_samples)

    # 5. Program costs per retained customer attempt
    cost_discount = rng.normal(loc=40.0, scale=5.0, size=n_samples)
    cost_discount = np.clip(cost_discount, 20.0, 70.0)

    cost_concierge = rng.normal(loc=95.0, scale=12.0, size=n_samples)
    cost_concierge = np.clip(cost_concierge, 50.0, 160.0)

    # Cohort scale: annual net benefit per 1000 evaluated subscribers
    # Strategy 0: Status Quo (no intervention cost, full base churn loss)
    # Expected value per user = CLV * (1 - churn)
    nb_status_quo = clv * (1.0 - base_churn)

    # Strategy 1: Targeted Discount
    churn_discount = base_churn * (1.0 - discount_reduction)
    nb_discount = clv * (1.0 - churn_discount) - cost_discount

    # Strategy 2: Proactive Concierge
    churn_concierge = base_churn * (1.0 - concierge_reduction)
    nb_concierge = clv * (1.0 - churn_concierge) - cost_concierge

    net_benefits = np.column_stack([nb_status_quo, nb_discount, nb_concierge])

    parameters = {
        "base_churn": base_churn,
        "discount_reduction": discount_reduction,
        "concierge_reduction": concierge_reduction,
        "clv": clv,
        "cost_discount": cost_discount,
        "cost_concierge": cost_concierge,
    }

    return net_benefits, parameters


def build_decision_problem() -> DecisionProblem:
    """Instantiate the canonical DecisionProblem schema representation."""
    template = get_domain_template("churn_retention")
    interventions = [
        Intervention(
            intervention_id="status_quo",
            name="Status Quo (No Proactive Campaign)",
            description="Allow natural churn dynamics without proactive marketing spend",
            is_reference=True,
            category="baseline",
        ),
        Intervention(
            intervention_id="targeted_discount",
            name="Targeted Proactive Discount",
            description="Offer automated 20% discount coupon to predicted churners",
            category="pricing",
        ),
        Intervention(
            intervention_id="concierge_onboarding",
            name="Dedicated Concierge Success Outreach",
            description="Human specialist outreach to re-engage accounts",
            category="service",
        ),
    ]

    return DecisionProblem(
        decision_problem_id="churn_retention_campaign_2026",
        title=template.title,
        willingness_to_pay=1.0,
        interventions=interventions,
        currency="USD",
        outcome_names=["Net Revenue per Account ($)", "Annual Churn Rate"],
    )


def run_churn_retention_analysis(
    n_samples: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """Execute complete VOI workflow for customer churn and return summary metrics."""
    strategy_names = ["Status Quo", "Targeted Discount", "Concierge Outreach"]
    net_benefits, param_dict = generate_churn_dataset(n_samples=n_samples, seed=seed)

    value_array = ValueArray.from_numpy(net_benefits, strategy_names=strategy_names)
    param_set = ParameterSet.from_numpy_or_dict(param_dict)
    decision_problem = build_decision_problem()

    # 1. Expected Net Benefit per strategy
    expected_nb = {
        name: float(np.mean(net_benefits[:, i]))
        for i, name in enumerate(strategy_names)
    }
    optimal_prior_strategy = max(expected_nb, key=expected_nb.get)  # type: ignore[arg-type]

    # 2. EVPI
    evpi_result = float(evpi(value_array))

    # 3. EVPPI on treatment effectiveness parameters
    evppi_reduction_params = float(
        evppi(
            value_array,
            param_set,
            parameters_of_interest=["discount_reduction", "concierge_reduction"],
        )
    )

    evppi_clv = float(
        evppi(
            value_array,
            param_set,
            parameters_of_interest=["clv"],
        )
    )

    # 4. ENBS for a pilot retention randomized trial
    # Pilot EVSI per account of $3.50 across 50,000 account target population over 3 years discounted
    pilot_evsi_per_account = 3.50
    pilot_study_cost = 15000.0
    population = 50000
    time_horizon = 3
    discount_rate = 0.05
    annuity_factor = sum(
        (1.0 / (1.0 + discount_rate) ** t) for t in range(time_horizon)
    )
    population_evsi = pilot_evsi_per_account * population * annuity_factor

    enbs_val = float(enbs(evsi_result=population_evsi, research_cost=pilot_study_cost))

    return {
        "decision_problem": decision_problem.to_dict(),
        "expected_net_benefits": expected_nb,
        "optimal_prior_strategy": optimal_prior_strategy,
        "evpi_per_account": evpi_result,
        "evppi_effectiveness_per_account": evppi_reduction_params,
        "evppi_clv_per_account": evppi_clv,
        "enbs_pilot_trial": enbs_val,
        "population_evsi": population_evsi,
    }


def main() -> None:
    """Print formatted summary of the customer churn retention VOI analysis."""
    results = run_churn_retention_analysis()
    print("=" * 60)
    print("VOI Analysis: Customer Churn Retention Decision (#574)")
    print("=" * 60)
    print(f"Decision Problem: {results['decision_problem']['title']}")
    print("\nExpected Net Benefit per Subscriber:")
    for strat, val in results["expected_net_benefits"].items():
        print(f"  - {strat:20s}: ${val:8.2f}")
    print(f"\nPrior Optimal Strategy: {results['optimal_prior_strategy']}")
    print(f"EVPI per Account:        ${results['evpi_per_account']:8.2f}")
    print(
        f"EVPPI (Effectiveness):   ${results['evppi_effectiveness_per_account']:8.2f}"
    )
    print(f"EVPPI (CLV):             ${results['evppi_clv_per_account']:8.2f}")
    print(f"Population Pilot EVSI:   ${results['population_evsi']:10.2f}")
    print(f"ENBS of Pilot Trial:     ${results['enbs_pilot_trial']:10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
