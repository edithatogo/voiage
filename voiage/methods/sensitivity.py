"""Deterministic parameter/scenario sensitivity analysis (DSA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class DeterministicSensitivityResult:
    """One-way deterministic scenario deltas, ordered by absolute impact."""

    baseline: float
    scenario_values: dict[str, float]
    deltas: dict[str, float]


def deterministic_sensitivity_analysis(
    baseline: float, scenarios: Mapping[str, float]
) -> DeterministicSensitivityResult:
    """Calculate deterministic scenario deltas without probabilistic claims."""
    baseline_value = float(baseline)
    values = {str(name): float(value) for name, value in scenarios.items()}
    ordered = dict(sorted(values.items(), key=lambda item: (-abs(item[1] - baseline_value), item[0])))
    deltas = {name: value - baseline_value for name, value in ordered.items()}
    return DeterministicSensitivityResult(baseline_value, ordered, deltas)
