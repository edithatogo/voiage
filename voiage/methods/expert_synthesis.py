"""Expert elicitation and evidence-synthesis design VOI."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class ExpertSynthesisResult:
    """Fixture-backed result envelope for expert synthesis VOI."""

    value: float
    elicitation_value: float
    synthesis_design_value: float
    expert_profile_ids: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_expert_profile: dict[str, str]
    elicitation_cost_matrix: np.ndarray
    synthesis_penalty_matrix: np.ndarray
    consensus_strategy: str
    robust_strategy: str
    pareto_strategies: list[str]
    reference_expert_profile: str
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_expert_synthesis(
    net_benefits: np.ndarray,
    expert_profile_ids: Sequence[str],
    strategy_names: Sequence[str],
    elicitation_costs: np.ndarray,
    synthesis_penalties: np.ndarray,
    *,
    analysis_id: str = "expert-synthesis-analysis",
    decision_problem_id: str = "unspecified",
    reference_expert_profile: str | None = None,
) -> ExpertSynthesisResult:
    """Value elicitation and synthesis design under profile disagreement.

    ``net_benefits`` has shape ``(samples, strategies, expert_profiles)``.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    profiles = [str(item) for item in expert_profile_ids]
    strategies = [str(item) for item in strategy_names]
    costs = np.asarray(elicitation_costs, dtype=DEFAULT_DTYPE)
    penalties = np.asarray(synthesis_penalties, dtype=DEFAULT_DTYPE)
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error(
            "net_benefits must be a non-empty 3D array (samples, strategies, profiles)."
        )
    if values.shape[1:] != (len(strategies), len(profiles)):
        raise_input_error(
            "net_benefits dimensions must match strategy and profile names."
        )
    expected_shape = (len(profiles), len(strategies))
    if costs.shape != expected_shape or penalties.shape != expected_shape:
        raise_input_error("Synthesis matrices must have profile x strategy shape.")
    if len(set(profiles)) != len(profiles) or len(set(strategies)) != len(strategies):
        raise_input_error("Profile and strategy names must be unique.")
    if not all(np.all(np.isfinite(matrix)) for matrix in (values, costs, penalties)):
        raise_input_error("Inputs must contain only finite values.")
    if np.any(costs < 0) or np.any(penalties < 0):
        raise_input_error(
            "Elicitation costs and synthesis penalties must be non-negative."
        )
    reference_expert_profile = reference_expert_profile or profiles[0]
    if reference_expert_profile not in profiles:
        raise_input_error("reference_expert_profile must identify a profile.")

    raw = np.mean(values, axis=0).T
    adjusted = raw - costs - penalties
    optimal = np.argmax(adjusted, axis=1)
    reference_index = profiles.index(reference_expert_profile)
    reference_strategy = int(optimal[reference_index])
    reference_value = float(adjusted[reference_index, reference_strategy])
    flexible_value = float(np.mean(np.max(adjusted, axis=1)))
    value = max(0.0, flexible_value - reference_value)
    elicitation_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw - costs)))
    synthesis_value = max(0.0, float(np.mean(raw - costs)) - float(np.mean(adjusted)))
    pareto = [
        strategies[i]
        for i in range(len(strategies))
        if not any(
            i != j
            and np.all(adjusted[:, j] >= adjusted[:, i])
            and np.any(adjusted[:, j] > adjusted[:, i])
            for j in range(len(strategies))
        )
    ]
    consensus = strategies[int(np.argmax(np.mean(adjusted, axis=0)))]
    robust = strategies[int(np.argmax(np.min(adjusted, axis=0)))]
    return ExpertSynthesisResult(
        value=value,
        elicitation_value=elicitation_value,
        synthesis_design_value=synthesis_value,
        expert_profile_ids=profiles,
        strategy_names=strategies,
        expected_net_benefits=adjusted,
        optimal_strategy_by_expert_profile={
            profile: strategies[int(index)]
            for profile, index in zip(profiles, optimal, strict=True)
        },
        elicitation_cost_matrix=costs,
        synthesis_penalty_matrix=penalties,
        consensus_strategy=consensus,
        robust_strategy=robust,
        pareto_strategies=pareto,
        reference_expert_profile=reference_expert_profile,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_profiles": len(profiles),
            "n_strategies": len(strategies),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_expert_synthesis",
            "method_maturity": "fixture-backed",
        },
    )
