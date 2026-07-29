# voiage/methods/portfolio.py

"""Value of information helpers for research portfolio optimization.

The main entry point is ``portfolio_voi``, which selects studies subject to
budget or algorithm-specific constraints.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import cast

import numpy as np

from voiage.exceptions import InputError, VoiageNotImplementedError
from voiage.schema import PortfolioSpec, PortfolioStudy

# Type alias for a function that calculates the value of a single study
# This could be an EVSI or ENBS calculator for that study.
StudyValueCalculator = Callable[[PortfolioStudy], float]


_PORTFOLIO_SPEC_MESSAGE = "`portfolio_specification` must be a PortfolioSpec object."
_STUDY_VALUE_CALCULATOR_MESSAGE = (
    "`study_value_calculator` must be a callable function."
)
_INTEGER_PROGRAMMING_MESSAGE = "Integer programming optimization failed."
_UNKNOWN_METHOD_MESSAGE = (
    "Optimization method '{optimization_method}' is not recognized or implemented."
)


@dataclass(frozen=True)
class _StudyValueRow:
    """Cached value/cost summary for one portfolio study."""

    study: PortfolioStudy
    value: float
    cost: float
    ratio: float


def _portfolio_spec_error() -> InputError:
    return InputError(_PORTFOLIO_SPEC_MESSAGE)


def _study_value_calculator_error() -> InputError:
    return InputError(_STUDY_VALUE_CALCULATOR_MESSAGE)


def _integer_programming_error() -> RuntimeError:
    return RuntimeError(_INTEGER_PROGRAMMING_MESSAGE)


def _unknown_method_error(optimization_method: str) -> VoiageNotImplementedError:
    return VoiageNotImplementedError(
        _UNKNOWN_METHOD_MESSAGE.format(optimization_method=optimization_method)
    )


def _study_value_rows(
    studies: list[PortfolioStudy],
    study_value_calculator: StudyValueCalculator,
) -> list[_StudyValueRow]:
    rows: list[_StudyValueRow] = []
    for study in studies:
        value = float(study_value_calculator(study))
        cost = float(study.cost)
        ratio = (float("inf") if value > 0 else 0.0) if cost <= 0 else value / cost
        rows.append(_StudyValueRow(study=study, value=value, cost=cost, ratio=ratio))
    return rows


def _dependency_groups_for_study(
    study_name: str,
    dependency_groups: Mapping[str, str | Iterable[str]] | None,
) -> tuple[str, ...]:
    if dependency_groups is None or study_name not in dependency_groups:
        return ()
    groups = dependency_groups[study_name]
    if isinstance(groups, str):
        return (groups,)
    return tuple(str(group) for group in groups)


def _dynamic_programming_portfolio(
    portfolio_specification: PortfolioSpec,
    study_value_calculator: StudyValueCalculator,
    *,
    dependency_groups: Mapping[str, str | Iterable[str]] | None = None,
    dependency_discount: float = 0.0,
) -> dict[str, object]:
    """Select the value-maximizing study subset by memoized 0/1 knapsack.

    ``dependency_groups`` maps study names to shared uncertainty groups. When a
    later selected study targets a group already targeted by an earlier selected
    study, its marginal value is multiplied by ``dependency_discount``. This
    preserves the existing callback-based API while allowing simple dependency
    modelling without changing ``PortfolioStudy``.
    """
    if not 0.0 <= dependency_discount <= 1.0:
        raise InputError("`dependency_discount` must be between 0 and 1.")

    studies = portfolio_specification.studies
    rows = _study_value_rows(studies, study_value_calculator)
    budget = portfolio_specification.budget_constraint
    if budget is None:
        selected_indices = [index for index, row in enumerate(rows) if row.value > 0.0]
        selected_studies = [studies[index] for index in selected_indices]
        return {
            "selected_studies": selected_studies,
            "total_value": float(sum(rows[index].value for index in selected_indices)),
            "total_cost": float(sum(rows[index].cost for index in selected_indices)),
            "method_details": {
                "algorithm": "dynamic_programming",
                "budget_constraint": None,
                "dependency_discount": dependency_discount,
                "dependency_groups_used": dependency_groups is not None,
            },
        }

    @cache
    def best(
        index: int,
        remaining_budget: float,
        selected_groups: tuple[str, ...],
    ) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
        if index >= len(rows):
            return 0.0, (), selected_groups

        skip_value, skip_indices, skip_groups = best(
            index + 1, remaining_budget, selected_groups
        )

        row = rows[index]
        cost = row.cost
        if cost > remaining_budget:
            return skip_value, skip_indices, skip_groups

        study = row.study
        study_groups = _dependency_groups_for_study(study.name, dependency_groups)
        selected_group_set = set(selected_groups)
        overlap = bool(selected_group_set.intersection(study_groups))
        marginal_value = row.value * (dependency_discount if overlap else 1.0)

        next_groups = tuple(sorted(selected_group_set.union(study_groups)))
        take_tail_value, take_tail_indices, take_groups = best(
            index + 1,
            remaining_budget - cost,
            next_groups,
        )
        take_value = marginal_value + take_tail_value
        take_indices = (index, *take_tail_indices)

        if take_value > skip_value:
            return take_value, take_indices, take_groups
        return skip_value, skip_indices, skip_groups

    total_value, selected_indices, selected_groups = best(0, float(budget), ())
    selected_studies = [studies[index] for index in selected_indices]
    total_cost = float(sum(rows[index].cost for index in selected_indices))

    return {
        "selected_studies": selected_studies,
        "total_value": float(total_value),
        "total_cost": total_cost,
        "method_details": {
            "algorithm": "dynamic_programming",
            "budget_constraint": float(budget),
            "states_evaluated": best.cache_info().currsize,
            "dependency_discount": dependency_discount,
            "dependency_groups_used": dependency_groups is not None,
            "selected_dependency_groups": list(selected_groups),
        },
    }


def portfolio_voi(
    portfolio_specification: PortfolioSpec,
    study_value_calculator: StudyValueCalculator,
    optimization_method: str = "greedy",  # e.g., "greedy", "integer_programming", "dynamic_programming"
    # wtp: float, # Usually implicit in how study_value_calculator works
    # population: Optional[float] = None, # Usually implicit
    # discount_rate: Optional[float] = None, # Usually implicit
    # time_horizon: Optional[float] = None, # Usually implicit
    **kwargs: object,
) -> dict[str, object]:
    """Optimize a portfolio of research studies.

    Parameters
    ----------
    portfolio_specification : PortfolioSpec
        Candidate studies, costs, and optional budget constraint.
    study_value_calculator : callable
        Function that returns the value of a single study.
    optimization_method : str, default="greedy"
        Selection algorithm. Supported values include ``greedy``,
        ``integer_programming``, and ``dynamic_programming``.
    **kwargs : object
        Additional algorithm-specific options.

    Returns
    -------
    dict[str, object]
        Dictionary containing the selected studies, total value, total cost,
        and method details.

    Notes
    -----
    The dynamic-programming path is the exact budget-constrained optimizer.
    Greedy selection remains available as a fast heuristic.
    """
    if not isinstance(portfolio_specification, PortfolioSpec):
        raise _portfolio_spec_error()
    if not callable(study_value_calculator):
        raise _study_value_calculator_error()

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

        studies_with_values = _study_value_rows(
            portfolio_specification.studies, study_value_calculator
        )

        # Sort by ratio in descending order
        studies_with_values.sort(key=lambda x: x.ratio, reverse=True)

        selected_studies_list: list[PortfolioStudy] = []
        current_total_cost = 0.0
        current_total_value = 0.0
        budget = portfolio_specification.budget_constraint

        for item in studies_with_values:
            study_obj = item.study
            study_cost = item.cost
            study_value = item.value  # This is the pre-calculated value

            if budget is None or (current_total_cost + study_cost <= budget):
                selected_studies_list.append(study_obj)
                current_total_cost += study_cost
                current_total_value += study_value
                # but for ENBS, it's more direct.
                # If values are interdependent, this is too simple.

        return {
            "selected_studies": selected_studies_list,
            "total_value": current_total_value,
            "total_cost": current_total_cost,
            "method_details": "Greedy selection by value-to-cost ratio.",
        }

    if optimization_method == "integer_programming":
        from scipy.optimize import LinearConstraint, milp

        values = np.array(
            [study_value_calculator(s) for s in portfolio_specification.studies]
        )
        costs = np.array([s.cost for s in portfolio_specification.studies])

        # The objective is to maximize the total value, which is equivalent to minimizing the negative total value.
        c = -values

        # The constraints are that the total cost must be less than or equal to the budget.
        constraints = LinearConstraint(
            costs, lb=0, ub=portfolio_specification.budget_constraint
        )

        # The decision variables are binary (0 or 1), indicating whether to select each study.
        integrality = np.ones_like(c)

        res = milp(c=c, constraints=constraints, integrality=integrality)

        if not res.success:
            raise _integer_programming_error()

        selected_indices = np.where(res.x > 0.5)[0]
        selected_studies_list = [
            portfolio_specification.studies[i] for i in selected_indices
        ]
        total_value = np.sum(values[selected_indices])
        total_cost = np.sum(costs[selected_indices])

        return {
            "selected_studies": selected_studies_list,
            "total_value": total_value,
            "total_cost": total_cost,
            "method_details": "Integer programming selection.",
        }
    if optimization_method == "dynamic_programming":
        dependency_groups = kwargs.get("dependency_groups")
        if dependency_groups is not None and not isinstance(dependency_groups, Mapping):
            raise InputError("`dependency_groups` must be a mapping of study names.")
        raw_dependency_discount = kwargs.get("dependency_discount", 0.0)
        if not isinstance(raw_dependency_discount, (int, float)):
            raise InputError("`dependency_discount` must be a number.")
        dependency_discount = float(raw_dependency_discount)
        return _dynamic_programming_portfolio(
            portfolio_specification,
            study_value_calculator,
            dependency_groups=cast(
                "Mapping[str, str | Iterable[str]] | None", dependency_groups
            ),
            dependency_discount=dependency_discount,
        )
    raise _unknown_method_error(optimization_method)


if __name__ == "__main__":  # pragma: no cover
    # Add local imports for classes used in this test block

    from voiage.schema import (
        DecisionOption,
        PortfolioSpec,
        PortfolioStudy,
        TrialDesign,
    )

    print("--- Testing portfolio.py ---")

    # Dummy study value calculator (e.g., returns a fixed EVSI or ENBS based on study name)
    def dummy_calculator(study: PortfolioStudy) -> float:
        """Return a deterministic smoke-test value for each study."""
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
    study_a_design = TrialDesign([DecisionOption("T1", 10)])
    study_b_design = TrialDesign([DecisionOption("T2", 20)])
    study_c_design = TrialDesign([DecisionOption("T3", 15)])
    study_d_design = TrialDesign([DecisionOption("T4", 30)])

    studies = [
        PortfolioStudy(
            name="Study Alpha", design=study_a_design, cost=50
        ),  # Ratio 100/50 = 2
        PortfolioStudy(
            name="Study Beta", design=study_b_design, cost=60
        ),  # Ratio 150/60 = 2.5
        PortfolioStudy(
            name="Study Gamma", design=study_c_design, cost=40
        ),  # Ratio 80/40 = 2
        PortfolioStudy(
            name="Study Delta", design=study_d_design, cost=100
        ),  # Ratio 120/100 = 1.2
    ]

    print("\n--- Greedy Portfolio VOI (No Budget) ---")
    portfolio_spec_no_budget = PortfolioSpec(studies=studies, budget_constraint=None)
    result_no_budget = portfolio_voi(
        portfolio_specification=portfolio_spec_no_budget,
        study_value_calculator=dummy_calculator,
        optimization_method="greedy",
    )
    print(
        f"Selected (no budget): {[s.name for s in result_no_budget['selected_studies']]}"
    )
    print(
        f"Total Value: {result_no_budget['total_value']}, Total Cost: {result_no_budget['total_cost']}"
    )
    # Expected: All studies selected, order might vary but content should be all.
    # Value = 100+150+80+120 = 450. Cost = 50+60+40+100 = 250
    np.testing.assert_allclose(len(result_no_budget["selected_studies"]), 4)
    np.testing.assert_allclose(result_no_budget["total_value"], 450.0)
    np.testing.assert_allclose(result_no_budget["total_cost"], 250.0)
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
        portfolio_specification=portfolio_spec_budget_100,
        study_value_calculator=dummy_calculator,
        optimization_method="greedy",
    )
    selected_names_b100 = sorted(
        [s.name for s in result_budget_100["selected_studies"]]
    )
    print(f"Selected (budget 100): {selected_names_b100}")
    print(
        f"Total Value: {result_budget_100['total_value']}, Total Cost: {result_budget_100['total_cost']}"
    )
    np.testing.assert_array_equal(
        selected_names_b100, sorted(["Study Beta", "Study Gamma"])
    )
    np.testing.assert_allclose(result_budget_100["total_value"], 230.0)
    np.testing.assert_allclose(result_budget_100["total_cost"], 100.0)
    print("Greedy (budget 100) PASSED.")

    print("\n--- Greedy Portfolio VOI (Budget allows only highest ratio) ---")
    # Budget = 70.
    # Beta (cost 60, val 150, ratio 2.5) -> remaining budget 10.
    # No other study can be picked.
    # Selected: Beta. Total Cost = 60. Total Value = 150.
    portfolio_spec_budget_70 = PortfolioSpec(studies=studies, budget_constraint=70)
    result_budget_70 = portfolio_voi(
        portfolio_specification=portfolio_spec_budget_70,
        study_value_calculator=dummy_calculator,
        optimization_method="greedy",
    )
    selected_names_b70 = sorted([s.name for s in result_budget_70["selected_studies"]])
    print(f"Selected (budget 70): {selected_names_b70}")
    print(
        f"Total Value: {result_budget_70['total_value']}, Total Cost: {result_budget_70['total_cost']}"
    )
    np.testing.assert_array_equal(selected_names_b70, ["Study Beta"])
    np.testing.assert_allclose(result_budget_70["total_value"], 150.0)
    np.testing.assert_allclose(result_budget_70["total_cost"], 60.0)
    print("Greedy (budget 70) PASSED.")

    print("\n--- Dynamic Programming Portfolio VOI (With Budget) ---")
    result_dp_budget_100 = portfolio_voi(
        portfolio_specification=portfolio_spec_budget_100,
        study_value_calculator=dummy_calculator,
        optimization_method="dynamic_programming",
    )
    selected_names_dp_b100 = sorted(
        [s.name for s in result_dp_budget_100["selected_studies"]]
    )
    print(f"Selected (budget 100): {selected_names_dp_b100}")
    print(
        f"Total Value: {result_dp_budget_100['total_value']}, Total Cost: {result_dp_budget_100['total_cost']}"
    )
    np.testing.assert_array_equal(
        selected_names_dp_b100, sorted(["Study Beta", "Study Gamma"])
    )
    np.testing.assert_allclose(result_dp_budget_100["total_value"], 230.0)
    np.testing.assert_allclose(result_dp_budget_100["total_cost"], 100.0)
    print("Dynamic Programming (budget 100) PASSED.")

    print("\n--- Integer Programming Portfolio VOI (With Budget) ---")
    result_ip_budget_100 = portfolio_voi(
        portfolio_specification=portfolio_spec_budget_100,
        study_value_calculator=dummy_calculator,
        optimization_method="integer_programming",
    )
    selected_names_ip_b100 = sorted(
        [s.name for s in result_ip_budget_100["selected_studies"]]
    )
    print(f"Selected (budget 100): {selected_names_ip_b100}")
    print(
        f"Total Value: {result_ip_budget_100['total_value']}, Total Cost: {result_ip_budget_100['total_cost']}"
    )
    np.testing.assert_array_equal(
        selected_names_ip_b100, sorted(["Study Beta", "Study Gamma"])
    )
    np.testing.assert_allclose(result_ip_budget_100["total_value"], 230.0)
    np.testing.assert_allclose(result_ip_budget_100["total_cost"], 100.0)
    print("Integer Programming (budget 100) PASSED.")

    print("--- portfolio.py tests completed ---")
