"""Fixture-backed VOI for regulatory and market-access decisions."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class RegulatoryMarketAccessResult:
    """Result envelope for approval, reimbursement, and access evidence."""

    value: float
    selected_scenario_indices: np.ndarray
    scenario_names: list[str]
    approval_probability: float
    reimbursement_probability: float
    joint_access_probability: float
    regulatory_uncertainty: float
    payer_uncertainty: float
    access_delay_months: float
    access_delay_cost: float
    evidence_package_cost: float
    label_expansion_value: float
    price_threshold_value: float
    expected_access_value: float
    baseline_value: float
    expected_value_market_access_information: float
    approval_threshold: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_regulatory_market_access(
    scenario_values: np.ndarray | list[float],
    approval_probabilities: np.ndarray | list[float],
    reimbursement_probabilities: np.ndarray | list[float],
    access_delays_months: np.ndarray | list[float],
    evidence_package_costs: np.ndarray | list[float],
    label_expansion_values: np.ndarray | list[float],
    price_threshold_values: np.ndarray | list[float],
    *,
    scenario_names: list[str] | None = None,
    approval_threshold: float = 0.5,
    monthly_access_delay_cost: float = 0.0,
) -> RegulatoryMarketAccessResult:
    """Estimate VOI from information that changes approval or market access.

    A scenario is selected when its joint approval and reimbursement
    probability meets ``approval_threshold``. Selected scenarios contribute
    expected access, label, and price value after evidence-package and delay
    costs. The result is fixture-backed pending calibrated regulatory and payer
    data, cross-language parity, and mature/stable governance review.
    """
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            scenario_values,
            approval_probabilities,
            reimbursement_probabilities,
            access_delays_months,
            evidence_package_costs,
            label_expansion_values,
            price_threshold_values,
        )
    ]
    values, approvals, reimbursements, delays, package_costs, labels, prices = arrays
    if values.ndim != 1 or len(values) < 2:
        raise_input_error(
            "scenario_values must be a vector with at least two scenarios."
        )
    if any(array.ndim != 1 or len(array) != len(values) for array in arrays[1:]):
        raise_input_error(
            "market-access inputs must have the same one-dimensional length."
        )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("market-access inputs must be finite.")
    if np.any(values < 0) or np.any(delays < 0) or np.any(package_costs < 0):
        raise_input_error(
            "scenario values, delays, and package costs must be non-negative."
        )
    if any(np.any((array < 0) | (array > 1)) for array in (approvals, reimbursements)):
        raise_input_error(
            "approval and reimbursement probabilities must be between zero and one."
        )
    if np.any(labels < 0) or np.any(prices < 0):
        raise_input_error("label and price values must be non-negative.")
    if not np.isfinite(approval_threshold) or not 0 <= approval_threshold <= 1:
        raise_input_error("approval_threshold must be finite and between zero and one.")
    if not np.isfinite(monthly_access_delay_cost) or monthly_access_delay_cost < 0:
        raise_input_error("monthly_access_delay_cost must be finite and non-negative.")

    names = scenario_names or [f"scenario_{index + 1}" for index in range(len(values))]
    if len(names) != len(values) or len(set(names)) != len(names):
        raise_input_error("scenario_names must be unique and match the scenario count.")
    joint = approvals * reimbursements
    selected = joint >= approval_threshold
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices):
        selected_approvals = approvals[selected]
        selected_reimbursements = reimbursements[selected]
        approval_probability = float(np.mean(selected_approvals))
        reimbursement_probability = float(np.mean(selected_reimbursements))
        joint_probability = float(np.mean(joint[selected]))
        access_delay_months = float(np.mean(delays[selected]))
        evidence_package_cost = float(np.sum(package_costs[selected]))
        label_value = float(np.sum(labels[selected] * joint[selected]))
        price_value = float(np.sum(prices[selected] * joint[selected]))
        access_delay_cost = access_delay_months * monthly_access_delay_cost
        expected_access_value = float(
            np.sum(values[selected] * joint[selected])
            + label_value
            + price_value
            - evidence_package_cost
            - access_delay_cost
        )
    else:
        approval_probability = reimbursement_probability = joint_probability = 0.0
        access_delay_months = evidence_package_cost = label_value = price_value = 0.0
        access_delay_cost = 0.0
        expected_access_value = 0.0
    regulatory_uncertainty = float(1.0 - np.mean(approvals))
    payer_uncertainty = float(1.0 - np.mean(reimbursements))
    baseline_value = float(np.max(values) * approval_threshold**2)
    value = max(0.0, expected_access_value - baseline_value)
    diagnostics: dict[str, object] = {
        "scenario_count": len(values),
        "selected_scenario_count": int(np.sum(selected)),
        "market_access_decision": True,
        "joint_probability_definition": "approval probability multiplied by reimbursement probability",
        "regulatory_uncertainty_definition": "one minus mean approval probability",
        "payer_uncertainty_definition": "one minus mean reimbursement probability",
        "access_delay_cost_definition": "mean selected delay months multiplied by monthly delay cost",
        "parity_status": "deferred",
        "open_data_status": "blocked: no calibrated regulatory or payer evidence snapshot committed",
        "stable_promotion": "blocked pending external regulatory validation, payer calibration, parity, and governance review",
    }
    return RegulatoryMarketAccessResult(
        value=value,
        selected_scenario_indices=selected_indices,
        scenario_names=list(names),
        approval_probability=approval_probability,
        reimbursement_probability=reimbursement_probability,
        joint_access_probability=joint_probability,
        regulatory_uncertainty=regulatory_uncertainty,
        payer_uncertainty=payer_uncertainty,
        access_delay_months=access_delay_months,
        access_delay_cost=access_delay_cost,
        evidence_package_cost=evidence_package_cost,
        label_expansion_value=label_value,
        price_threshold_value=price_value,
        expected_access_value=expected_access_value,
        baseline_value=baseline_value,
        expected_value_market_access_information=expected_access_value,
        approval_threshold=approval_threshold,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_regulatory_market_access",
            method_family="regulatory_market_access",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
