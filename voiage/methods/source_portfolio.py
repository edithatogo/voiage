"""Dependent information-source portfolio value."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class InformationSourcePortfolioResult:
    """Joint and incremental value for dependent information sources."""

    joint_value: float
    individual_values: dict[str, float]
    incremental_values: dict[str, float]


def value_of_information_source_portfolio(
    source_net_benefits: Mapping[str, np.ndarray], baseline: np.ndarray
) -> InformationSourcePortfolioResult:
    """Calculate joint source value relative to a baseline strategy vector."""
    if not source_net_benefits:
        raise_input_error("At least one information source is required.")
    base = np.asarray(baseline, dtype=DEFAULT_DTYPE)
    if base.ndim != 1 or not np.all(np.isfinite(base)):
        raise_input_error("baseline must be a finite strategy vector.")
    values = {name: np.asarray(value, dtype=DEFAULT_DTYPE) for name, value in source_net_benefits.items()}
    if any(value.shape != base.shape or not np.all(np.isfinite(value)) for value in values.values()):
        raise_input_error("Each source must match the finite baseline shape.")
    base_value = float(np.max(base))
    individual = {name: max(0.0, float(np.max(value) - base_value)) for name, value in values.items()}
    joint = max(0.0, float(np.max(np.mean(np.stack(list(values.values())), axis=0)) - base_value))
    incremental = {name: max(0.0, joint - individual[name]) for name in values}
    return InformationSourcePortfolioResult(joint, individual, incremental)
