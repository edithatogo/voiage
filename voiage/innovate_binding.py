"""Innovate-to-Voiage DecisionBinding adapter (E02)."""

from __future__ import annotations

from dataclasses import dataclass
from operator import eq
from typing import Any

import numpy as np

from voiage.exceptions import raise_value_error
from voiage.schema import ValueArray


@dataclass(frozen=True)
class InnovateDecisionBinding:
    """Validated Innovate-to-Voiage decision binding (E02)."""

    schema_version: str
    perspective: str
    population: float
    horizon: float
    discount_rate: float
    units: str
    strategy_ids: list[str]
    draw_ids: list[Any]
    value_array: ValueArray


_INNOVATE_REQUIRED_FIELDS = (
    "schema_version",
    "perspective",
    "population",
    "horizon",
    "discount_rate",
    "units",
    "fit_status",
    "strategies",
)
_INNOVATE_SCHEMA_VERSION = "1.0"
_INNOVATE_FORBIDDEN_UNITS = {
    "prescriptions",
    "raw_counts_without_utility_mapping",
}


def load_innovate_decision_binding(binding: dict[str, Any]) -> InnovateDecisionBinding:
    """Validate an Innovate DecisionBinding payload and convert correlated draws to a ValueArray."""
    for field in _INNOVATE_REQUIRED_FIELDS:
        if field not in binding:
            raise_value_error(f"Missing required DecisionBinding field: {field}")

    if binding["schema_version"] != _INNOVATE_SCHEMA_VERSION:
        raise_value_error(
            f"Unsupported DecisionBinding schema version: {binding['schema_version']!r}"
        )

    if binding["fit_status"] != "valid":
        raise_value_error(
            "Cannot bind an Innovate fit whose fit_status is not 'valid'."
        )

    perspective = binding["perspective"]
    if not isinstance(perspective, str) or not perspective.strip():
        raise_value_error("DecisionBinding perspective must be a non-empty string.")

    units = binding["units"]
    if not isinstance(units, str):
        raise_value_error("DecisionBinding units must be a string.")
    if units in _INNOVATE_FORBIDDEN_UNITS or "net_benefit" not in units:
        raise_value_error(
            "Prescription counts or unmapped raw units cannot become Voiage utilities."
        )

    population = float(binding["population"])
    horizon = float(binding["horizon"])
    discount_rate = float(binding["discount_rate"])
    if not np.isfinite(population) or population <= 0:
        raise_value_error("DecisionBinding population must be positive and finite.")
    if not np.isfinite(horizon) or horizon <= 0:
        raise_value_error("DecisionBinding horizon must be positive and finite.")
    if not np.isfinite(discount_rate) or not 0 <= discount_rate <= 1:
        raise_value_error(
            "DecisionBinding discount_rate must be finite and between 0 and 1."
        )

    strategies = binding["strategies"]
    if not isinstance(strategies, list) or len(strategies) < 2:
        raise_value_error("At least two strategies are required for decision binding.")

    strategy_ids: list[str] = []
    columns: list[np.ndarray] = []
    reference_draw_ids: list[Any] | None = None

    for strat in strategies:
        strat_id = strat["id"]
        if not isinstance(strat_id, str) or not strat_id.strip():
            raise_value_error(
                "Each DecisionBinding strategy id must be a non-empty string."
            )
        if strat_id in strategy_ids:
            raise_value_error(f"Duplicate strategy id in DecisionBinding: {strat_id}")
        draws = strat["draws"]
        if not draws:
            raise_value_error(f"Strategy '{strat_id}' has no draws.")
        draw_ids = [d["draw_id"] for d in draws]
        try:
            unique_draw_ids = set(draw_ids)
        except TypeError as exc:
            raise_value_error("Each draw ID must be a hashable value.", exc)
        # NaN-like identifiers cannot establish a reliable joint draw.
        if any(not eq(draw_id, draw_id) for draw_id in draw_ids):
            raise_value_error("Each draw ID must be a reflexive value.")
        if len(unique_draw_ids) != len(draw_ids):
            raise_value_error(f"Duplicate draw ID in strategy '{strat_id}'.")
        if reference_draw_ids is None:
            reference_draw_ids = draw_ids
        elif draw_ids != reference_draw_ids:
            raise_value_error("Correlated draw_ids must align across strategies.")
        strategy_ids.append(strat_id)
        values = np.asarray([float(d["net_benefit"]) for d in draws], dtype=float)
        if not np.isfinite(values).all():
            raise_value_error(f"Strategy '{strat_id}' net benefits must be finite.")
        columns.append(values)

    assert reference_draw_ids is not None
    matrix = np.column_stack(columns)
    value_array = ValueArray.from_numpy(matrix, strategy_ids)
    return InnovateDecisionBinding(
        schema_version=_INNOVATE_SCHEMA_VERSION,
        perspective=perspective,
        population=population,
        horizon=horizon,
        discount_rate=discount_rate,
        units=units,
        strategy_ids=strategy_ids,
        draw_ids=reference_draw_ids,
        value_array=value_array,
    )


def evaluate_innovate_decision_binding(
    binding: dict[str, Any],
    *,
    population_scaled: bool = False,
) -> dict[str, Any]:
    """Load an Innovate DecisionBinding and compute EVPI via Voiage's DecisionAnalysis."""
    from voiage.analysis import DecisionAnalysis

    loaded = load_innovate_decision_binding(binding)
    analysis = DecisionAnalysis(nb_array=loaded.value_array)
    if population_scaled:
        evpi = float(
            analysis.evpi(
                population=loaded.population,
                time_horizon=loaded.horizon,
                discount_rate=loaded.discount_rate,
            )
        )
    else:
        evpi = float(analysis.evpi())
    return {
        "schema_version": loaded.schema_version,
        "perspective": loaded.perspective,
        "units": loaded.units,
        "strategy_ids": loaded.strategy_ids,
        "n_draws": len(loaded.draw_ids),
        "evpi": evpi,
    }
