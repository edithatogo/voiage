"""Lightweight interfaces for experimental VOI workflows.

Bayesian model fitting and neural amortization remain optional backend
responsibilities, so importing this module never requires JAX or NumPyro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray: Final = NDArray[np.float64]


@dataclass(frozen=True)
class InformationGainEstimate:
    """Monte Carlo estimate and uncertainty for expected information gain."""

    estimate: float
    standard_error: float
    samples: int


@dataclass(frozen=True)
class ExperimentalDesign:
    """A named candidate design with cost and estimated information gain."""

    name: str
    cost: float
    expected_information_gain: float


def expected_information_gain(
    log_likelihood_ratios: Sequence[float],
) -> InformationGainEstimate:
    """Estimate expected information gain from simulated log ratios."""
    values = np.asarray(log_likelihood_ratios, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_likelihood_ratios must be a non-empty 1-D sequence")
    if not np.isfinite(values).all():
        raise ValueError("log_likelihood_ratios must contain only finite values")
    return InformationGainEstimate(
        estimate=float(values.mean()),
        standard_error=float(values.std(ddof=1) / np.sqrt(values.size))
        if values.size > 1
        else 0.0,
        samples=int(values.size),
    )


def select_bayesian_design(designs: Sequence[ExperimentalDesign]) -> ExperimentalDesign:
    """Select the candidate with greatest information gain per unit cost."""
    if not designs:
        raise ValueError("designs must contain at least one candidate")
    if any(design.cost <= 0 for design in designs):
        raise ValueError("design costs must be positive")
    return max(
        designs, key=lambda design: design.expected_information_gain / design.cost
    )


def select_active_learning_batch(
    scores: Sequence[float], batch_size: int
) -> NDArray[np.int_]:
    """Return indices of the highest-scoring candidates for active learning."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1-D sequence")
    if not 0 < batch_size <= values.size:
        raise ValueError("batch_size must be between one and the number of scores")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    return np.argsort(values)[-batch_size:][::-1]


def amortized_evsi(
    predicted_net_benefits: Sequence[float], baseline_net_benefit: float
) -> float:
    """Summarize simulated amortized EVSI predictions against current care."""
    predictions = np.asarray(predicted_net_benefits, dtype=float)
    if predictions.ndim != 1 or predictions.size == 0:
        raise ValueError("predicted_net_benefits must be a non-empty 1-D sequence")
    if not np.isfinite(predictions).all() or not np.isfinite(baseline_net_benefit):
        raise ValueError("net benefits must contain only finite values")
    return float(predictions.mean() - baseline_net_benefit)
