# voiage/methods/portfolio.py

"""Implementation of VOI methods for research portfolio optimization.

- Portfolio VOI (Value of Information for a portfolio of research studies)

Portfolio VOI aims to select an optimal subset of candidate research studies
that maximizes the total value (e.g., population EVSI or ENBS) subject to
constraints like a fixed budget.
"""

from typing import Any, Callable, Dict, List

from voiage.schema import PortfolioSpec, PortfolioStudy
from voiage.exceptions import InputError, VoiageNotImplementedError

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
) -> Dict[str, Any]:
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
        raise VoiageNotImplementedError(
            "Integer programming for portfolio VOI requires an IP solver. "
            "Not implemented in v0.1.",
        )
    elif optimization_method == "dynamic_programming":
        raise VoiageNotImplementedError(
            "Dynamic programming for portfolio VOI (knapsack) is not implemented in v0.1.",
        )
    else:
        raise VoiageNotImplementedError(
            f"Optimization method '{optimization_method}' is not recognized or implemented.",
        )

