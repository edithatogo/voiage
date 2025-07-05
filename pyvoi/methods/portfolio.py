# pyvoi/methods/portfolio.py

"""Implementation of VOI methods for research portfolio optimization.

- Portfolio VOI (Value of Information for a portfolio of research studies)

Portfolio VOI aims to select an optimal subset of candidate research studies
that maximizes the total value (e.g., population EVSI or ENBS) subject to
constraints like a fixed budget.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pyvoi.core.data_structures import PortfolioSpec, PortfolioStudy
from pyvoi.exceptions import InputError, PyVoiNotImplementedError

# Type alias for a function that calculates the value of a single study
# This could be an EVSI or ENBS calculator for that study.
StudyValueCalculator = Callable[[PortfolioStudy], float]


def portfolio_voi(
    portfolio_specification: PortfolioSpec,
    study_value_calculator: StudyValueCalculator,
    optimization_method: str = "greedy",  # e.g., "greedy", "integer_programming", "dynamic_programming"
    # wtp: float, # Usually implicit in how study_value_calculator works
    # population: Optional[float] = None, # Usually implicit
    # discount_rate: Optional[float] = None, # Usually implicit
    # time_horizon: Optional[float] = None, # Usually implicit
    **kwargs: Any,
) -> Dict[str, Any]:  # noqa: C901
    """
    Optimizes a portfolio of research studies to maximize total value.

    This function selects a subset of studies from the `portfolio_specification`
    that maximizes their combined value (calculated by `study_value_calculator`),
    subject to constraints like a total budget (if specified in `PortfolioSpec`).

    Args:
        portfolio_specification (PortfolioSpec):
            An object defining the set of candidate studies, their costs,
            and any overall constraints (e.g., budget).
        study_value_calculator (StudyValueCalculator):
            A function that takes a `PortfolioStudy` object and returns its
            estimated value (e.g., its individual EVSI or ENBS). This value
            should be on a scale that allows for meaningful addition/comparison
            (e.g., population-level monetary value).
        optimization_method (str):
            The algorithm to use for selecting the optimal portfolio. Examples:
            - "greedy": A heuristic that might, e.g., pick studies with the best
                        value-to-cost ratio until budget is exhausted.
            - "integer_programming": Formulates as a 0-1 knapsack-type problem
                                     (requires an IP solver like PuLP, Pyomo, SciPy).
            - "dynamic_programming": Exact method for 0-1 knapsack if applicable.
            (Note: Only a placeholder structure for v0.1)
        **kwargs: Additional arguments for the chosen optimization method.

    Returns
    -------
        Dict[str, Any]: A dictionary containing:
            - 'selected_studies': List[PortfolioStudy] of the chosen studies.
            - 'total_value': float, the sum of values of selected studies.
            - 'total_cost': float, the sum of costs of selected studies.
            - 'method_details': Optional details from the optimization algorithm.

    Raises
    ------
        InputError: If inputs are invalid.
        NotImplementedError: If the chosen optimization method is not implemented.
    """
    if not isinstance(portfolio_specification, PortfolioSpec):
        raise InputError("`portfolio_specification` must be a PortfolioSpec object.")
    if not callable(study_value_calculator):
        raise InputError("`study_value_calculator` must be a callable function.")

    if not portfolio_specification.studies:
        return {
            "selected_studies": [],
            "total_value": 0.0,
            "total_cost": 0.0,
            "method_details": "No studies in portfolio.",
        }

    if optimization_method == "greedy":
        # Example Greedy Algorithm: Prioritize by value/cost ratio (if ENBS is value)
        # or just by value if costs are similar or budget is very large.
        # This is a simplified greedy approach.
        # A common greedy for knapsack is value/cost ratio.
        # If study_value_calculator returns EVSI, we need costs from PortfolioStudy.
        # If it returns ENBS (EVSI - cost), then we can rank by ENBS directly if budget is the only concern.

        # Let's assume study_value_calculator returns raw value (e.g. EVSI), and we use cost from PortfolioStudy.
        # We need to handle the budget constraint.

        studies_with_values = []
        for study in portfolio_specification.studies:
            value = study_value_calculator(study)
            cost = study.cost
            if (
                cost <= 0
            ):  # Avoid division by zero, treat as infinitely good or handle as error
                # If cost is 0 and value is positive, it should always be picked if value > 0.
                # For simplicity, let's assume costs are positive for ratio calculation.
                # If cost is 0, its ratio is infinite if value > 0.
                ratio = float("inf") if value > 0 else 0
            else:
                ratio = value / cost  # Value per unit cost
            studies_with_values.append(
                {"study": study, "value": value, "cost": cost, "ratio": ratio}
            )

        # Sort by ratio in descending order
        studies_with_values.sort(key=lambda x: x["ratio"], reverse=True)

        selected_studies_list: List[PortfolioStudy] = []
        current_total_cost = 0.0
        current_total_value = 0.0
        budget = portfolio_specification.budget_constraint

        for item in studies_with_values:
            study_obj = item["study"]
            study_cost = item["cost"]
            study_value = item["value"]  # This is the pre-calculated value

            if budget is None or (current_total_cost + study_cost <= budget):
                selected_studies_list.append(study_obj)
                current_total_cost += study_cost
                current_total_value += (
                    study_value  # Summing individual EVsIs is usually okay
                )
                # but for ENBS, it's more direct.
                # If values are interdependent, this is too simple.

        return {
            "selected_studies": selected_studies_list,
            "total_value": current_total_value,
            "total_cost": current_total_cost,
            "method_details": "Greedy selection by value-to-cost ratio.",
        }

    elif optimization_method == "integer_programming":
        raise PyVoiNotImplementedError(
            "Integer programming for portfolio VOI requires an IP solver. "
            "Not implemented in v0.1.",
        )
    elif optimization_method == "dynamic_programming":
        raise PyVoiNotImplementedError(
            "Dynamic programming for portfolio VOI (knapsack) is not implemented in v0.1.",
        )
    else:
        raise PyVoiNotImplementedError(
            f"Optimization method '{optimization_method}' is not recognized or implemented.",
        )


if __name__ == "__main__":
    # Add local imports for classes used in this test block
    import numpy as np  # np is used for np.isclose

    from pyvoi.core.data_structures import (
        PortfolioSpec,
        PortfolioStudy,
        TrialArm,
        TrialDesign,
    )

    print("--- Testing portfolio.py ---")

    # Dummy study value calculator (e.g., returns a fixed EVSI or ENBS based on study name)
    def dummy_calculator(study: PortfolioStudy) -> float:
        if "Alpha" in study.name:
            return 100.0  # EVSI or ENBS
        if "Beta" in study.name:
            return 150.0
        if "Gamma" in study.name:
            return 80.0
        if "Delta" in study.name:
            return 120.0  # High value, high cost
        return 0.0

    # Create dummy portfolio studies
    study_A_design = TrialDesign([TrialArm("T1", 10)])
    study_B_design = TrialDesign([TrialArm("T2", 20)])
    study_C_design = TrialDesign([TrialArm("T3", 15)])
    study_D_design = TrialDesign([TrialArm("T4", 30)])

    studies = [
        PortfolioStudy(
            name="Study Alpha", design=study_A_design, cost=50
        ),  # Ratio 100/50 = 2
        PortfolioStudy(
            name="Study Beta", design=study_B_design, cost=60
        ),  # Ratio 150/60 = 2.5
        PortfolioStudy(
            name="Study Gamma", design=study_C_design, cost=40
        ),  # Ratio 80/40 = 2
        PortfolioStudy(
            name="Study Delta", design=study_D_design, cost=100
        ),  # Ratio 120/100 = 1.2
    ]

    print("\n--- Greedy Portfolio VOI (No Budget) ---")
    portfolio_spec_no_budget = PortfolioSpec(studies=studies, budget_constraint=None)
    result_no_budget = portfolio_voi(
        portfolio_spec_no_budget, dummy_calculator, "greedy"
    )
    print(
        f"Selected (no budget): {[s.name for s in result_no_budget['selected_studies']]}"
    )
    print(
        f"Total Value: {result_no_budget['total_value']}, Total Cost: {result_no_budget['total_cost']}"
    )
    # Expected: All studies selected, order might vary but content should be all.
    # Value = 100+150+80+120 = 450. Cost = 50+60+40+100 = 250
    assert len(result_no_budget["selected_studies"]) == 4
    assert np.isclose(result_no_budget["total_value"], 450.0)
    assert np.isclose(result_no_budget["total_cost"], 250.0)
    print("Greedy (no budget) PASSED.")

    print("\n--- Greedy Portfolio VOI (With Budget) ---")
    # Budget = 100. Expected selection by ratio:
    # Beta (cost 60, val 150, ratio 2.5) -> remaining budget 40
    # Alpha (cost 50, val 100, ratio 2) -> cannot pick
    # Gamma (cost 40, val 80, ratio 2) -> can pick. remaining budget 0.
    # Delta (cost 100, val 120, ratio 1.2) -> cannot pick
    # Selected: Beta, Gamma. Total Cost = 60+40=100. Total Value = 150+80=230.
    portfolio_spec_budget_100 = PortfolioSpec(studies=studies, budget_constraint=100)
    result_budget_100 = portfolio_voi(
        portfolio_spec_budget_100, dummy_calculator, "greedy"
    )
    selected_names_b100 = sorted(
        [s.name for s in result_budget_100["selected_studies"]]
    )
    print(f"Selected (budget 100): {selected_names_b100}")
    print(
        f"Total Value: {result_budget_100['total_value']}, Total Cost: {result_budget_100['total_cost']}"
    )
    assert selected_names_b100 == sorted(["Study Beta", "Study Gamma"])
    assert np.isclose(result_budget_100["total_value"], 230.0)
    assert np.isclose(result_budget_100["total_cost"], 100.0)
    print("Greedy (budget 100) PASSED.")

    print("\n--- Greedy Portfolio VOI (Budget allows only highest ratio) ---")
    # Budget = 70.
    # Beta (cost 60, val 150, ratio 2.5) -> remaining budget 10.
    # No other study can be picked.
    # Selected: Beta. Total Cost = 60. Total Value = 150.
    portfolio_spec_budget_70 = PortfolioSpec(studies=studies, budget_constraint=70)
    result_budget_70 = portfolio_voi(
        portfolio_spec_budget_70, dummy_calculator, "greedy"
    )
    selected_names_b70 = sorted([s.name for s in result_budget_70["selected_studies"]])
    print(f"Selected (budget 70): {selected_names_b70}")
    print(
        f"Total Value: {result_budget_70['total_value']}, Total Cost: {result_budget_70['total_cost']}"
    )
    assert selected_names_b70 == ["Study Beta"]
    assert np.isclose(result_budget_70["total_value"], 150.0)
    assert np.isclose(result_budget_70["total_cost"], 60.0)
    print("Greedy (budget 70) PASSED.")

    # Test other methods (expect PyVoiNotImplementedError)
    try:
        portfolio_voi(portfolio_spec_no_budget, dummy_calculator, "integer_programming")
    except PyVoiNotImplementedError as e:
        print(f"Caught expected error for IP method: {e}")
    else:
        raise AssertionError("IP method did not raise PyVoiNotImplementedError.")

    print("--- portfolio.py tests completed ---")
