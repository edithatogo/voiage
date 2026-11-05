# Adapted from edithatogo/vop_poc_nz at ad84d309 (Apache-2.0).
"""Monte Carlo diagnostics for directional Value of Perspective estimates.

These helpers quantify simulation error in the estimator; they do not represent
structural, evidential, or normative uncertainty. The non-parametric bootstrap
re-runs the decision rule after resampling PSA draws, so strategy-selection
instability is reflected in the interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
lazy from collections.abc import Iterable

import numpy as np

from .perspective_tensor import (
    METHOD_CONTRACT_VERSION,
    DecisionRule,
    NetBenefitTensor,
    TiePolicy,
)


@dataclass(frozen=True)
class EVoPBootstrapInterval:
    """Directional regret and its Monte Carlo bootstrap interval."""

    choose_under: str
    evaluate_under: str
    decision_rule: DecisionRule
    estimate: float
    standard_error: float
    lower: float
    upper: float
    confidence: float
    bootstrap_replicates: int
    seed: int | None
    selection_tie_policy: TiePolicy
    method_contract_version: str = METHOD_CONTRACT_VERSION
    bootstrap_estimates: tuple[float, ...] = field(default=(), repr=False)

    def as_dict(self, *, include_estimates: bool = False) -> dict[str, Any]:
        """Serialize diagnostics with their method contract and units."""
        output: dict[str, Any] = {
            "choose_under": self.choose_under,
            "evaluate_under": self.evaluate_under,
            "decision_rule": self.decision_rule.value,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "bootstrap_replicates": self.bootstrap_replicates,
            "seed": self.seed,
            "selection_tie_policy": self.selection_tie_policy.value,
            "method_contract_version": self.method_contract_version,
        }
        if include_estimates:
            output["bootstrap_estimates"] = list(self.bootstrap_estimates)
        return output


@dataclass(frozen=True)
class EVoPConvergenceRow:
    """Directional regret estimated from a prefix of the paired draws."""

    draws: int
    repeats: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    full_sample_estimate: float
    method_contract_version: str = METHOD_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize diagnostics with their method contract and units."""
        return {
            "draws": self.draws,
            "repeats": self.repeats,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "full_sample_estimate": self.full_sample_estimate,
            "method_contract_version": self.method_contract_version,
        }


def _resampled_tensor(
    tensor: NetBenefitTensor, indices: np.ndarray
) -> NetBenefitTensor:
    draw_ids = None
    if tensor.draw_ids is not None:
        draw_ids = tuple(
            f"{tensor.draw_ids[int(index)]}#resample{position}"
            for position, index in enumerate(indices)
        )
    return NetBenefitTensor(
        values=tensor.values[indices, :, :],
        strategies=tensor.strategies,
        perspectives=tensor.perspectives,
        case_id=tensor.case_id,
        draw_ids=draw_ids,
        attrs={**tensor.attrs, "resampled": True},
    )


def bootstrap_evop(
    tensor: NetBenefitTensor,
    *,
    choose_under: str,
    evaluate_under: str,
    decision_rule: DecisionRule | str = DecisionRule.EXPECTED_VALUE,
    selection_tie_policy: TiePolicy | str = TiePolicy.SPLIT,
    bootstrap_replicates: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
    atol: float = 1e-12,
) -> EVoPBootstrapInterval:
    """Estimate a percentile bootstrap interval for per-person EVoP.

    The interval reflects finite-PSA Monte Carlo uncertainty. It must not be
    described as an interval over the normative choice of perspective.
    """
    if bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rule = DecisionRule(decision_rule)
    tie_policy = TiePolicy(selection_tie_policy)
    point = tensor.evop(
        choose_under=choose_under,
        evaluate_under=evaluate_under,
        decision_rule=rule,
        selection_tie_policy=tie_policy,
        atol=atol,
    ).per_person
    rng = np.random.default_rng(seed)
    estimates = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, tensor.n_draws, size=tensor.n_draws)
        estimates[replicate] = (
            _resampled_tensor(tensor, indices)
            .evop(
                choose_under=choose_under,
                evaluate_under=evaluate_under,
                decision_rule=rule,
                selection_tie_policy=tie_policy,
                atol=atol,
            )
            .per_person
        )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return EVoPBootstrapInterval(
        choose_under=choose_under,
        evaluate_under=evaluate_under,
        decision_rule=rule,
        estimate=float(point),
        standard_error=float(np.std(estimates, ddof=1)),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        bootstrap_replicates=int(bootstrap_replicates),
        seed=seed,
        selection_tie_policy=tie_policy,
        bootstrap_estimates=tuple(float(value) for value in estimates),
    )


def evop_convergence_profile(
    tensor: NetBenefitTensor,
    *,
    choose_under: str,
    evaluate_under: str,
    draw_counts: Iterable[int],
    repeats: int = 20,
    decision_rule: DecisionRule | str = DecisionRule.EXPECTED_VALUE,
    selection_tie_policy: TiePolicy | str = TiePolicy.SPLIT,
    seed: int | None = 0,
    atol: float = 1e-12,
) -> list[EVoPConvergenceRow]:
    """Summarise EVoP stability across random draw subsets of increasing size."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    counts = sorted({int(value) for value in draw_counts})
    if not counts or counts[0] < 2 or counts[-1] > tensor.n_draws:
        raise ValueError(
            "draw_counts must contain values from 2 through tensor.n_draws"
        )
    rule = DecisionRule(decision_rule)
    tie_policy = TiePolicy(selection_tie_policy)
    full = tensor.evop(
        choose_under=choose_under,
        evaluate_under=evaluate_under,
        decision_rule=rule,
        selection_tie_policy=tie_policy,
        atol=atol,
    ).per_person
    rng = np.random.default_rng(seed)
    rows: list[EVoPConvergenceRow] = []
    for count in counts:
        run_count = 1 if count == tensor.n_draws else repeats
        estimates = np.empty(run_count, dtype=np.float64)
        for replicate in range(run_count):
            indices = (
                np.arange(tensor.n_draws)
                if count == tensor.n_draws
                else rng.choice(tensor.n_draws, size=count, replace=False)
            )
            estimates[replicate] = (
                _resampled_tensor(tensor, np.asarray(indices, dtype=np.int64))
                .evop(
                    choose_under=choose_under,
                    evaluate_under=evaluate_under,
                    decision_rule=rule,
                    selection_tie_policy=tie_policy,
                    atol=atol,
                )
                .per_person
            )
        rows.append(
            EVoPConvergenceRow(
                draws=count,
                repeats=run_count,
                mean=float(np.mean(estimates)),
                standard_deviation=float(np.std(estimates, ddof=1))
                if run_count > 1
                else 0.0,
                minimum=float(np.min(estimates)),
                maximum=float(np.max(estimates)),
                full_sample_estimate=float(full),
            )
        )
    return rows
