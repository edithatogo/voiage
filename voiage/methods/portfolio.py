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


try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

def portfolio_voi(
    portfolio_specification: PortfolioSpec,
    study_value_calculator: StudyValueCalculator,
    optimization_method: str = "greedy",
    **kwargs: Any,
) -> Dict[str, Any]:
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
        studies_with_values = []
        for study in portfolio_specification.studies:
            value = study_value_calculator(study)
            cost = study.cost
            if cost <= 0:
                ratio = float("inf") if value > 0 else 0
            else:
                ratio = value / cost
            studies_with_values.append(
                {"study": study, "value": value, "cost": cost, "ratio": ratio}
            )

        studies_with_values.sort(key=lambda x: x["ratio"], reverse=True)

        selected_studies_list: List[PortfolioStudy] = []
        current_total_cost = 0.0
        current_total_value = 0.0
        budget = portfolio_specification.budget_constraint

        for item in studies_with_values:
            study_obj = item["study"]
            study_cost = item["cost"]
            study_value = item["value"]

            if budget is None or (current_total_cost + study_cost <= budget):
                selected_studies_list.append(study_obj)
                current_total_cost += study_cost
                current_total_value += study_value

        return {
            "selected_studies": selected_studies_list,
            "total_value": current_total_value,
            "total_cost": current_total_cost,
            "method_details": "Greedy selection by value-to-cost ratio.",
        }

    elif optimization_method == "integer_programming":
        if not PULP_AVAILABLE:
            raise VoiageNotImplementedError(
                "Integer programming for portfolio VOI requires `pulp` to be installed."
            )
        prob = LpProblem("PortfolioVOI", LpMaximize)
        study_vars = LpVariable.dicts("Study", [s.name for s in portfolio_specification.studies], 0, 1, LpBinary)

        prob += lpSum([study_value_calculator(s) * study_vars[s.name] for s in portfolio_specification.studies])

        if portfolio_specification.budget_constraint is not None:
            prob += lpSum([s.cost * study_vars[s.name] for s in portfolio_specification.studies]) <= portfolio_specification.budget_constraint

        prob.solve()

        selected_studies_list = [s for s in portfolio_specification.studies if study_vars[s.name].varValue == 1]
        total_value = sum([study_value_calculator(s) for s in selected_studies_list])
        total_cost = sum([s.cost for s in selected_studies_list])

        return {
            "selected_studies": selected_studies_list,
            "total_value": total_value,
            "total_cost": total_cost,
            "method_details": "Integer programming selection.",
        }
    elif optimization_method == "dynamic_programming":
        raise VoiageNotImplementedError(
            "Dynamic programming for portfolio VOI (knapsack) is not implemented in v0.1.",
        )
    else:
        raise VoiageNotImplementedError(
            f"Optimization method '{optimization_method}' is not recognized or implemented.",
        )

