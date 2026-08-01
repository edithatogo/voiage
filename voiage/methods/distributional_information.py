"""Experimental value of perfect distribution-family-index information."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING, Any, cast

from jsonschema import Draft202012Validator
import numpy as np

from voiage.contracts.distributional_information import (
    VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1,
    validate_distributional_information_semantics,
)
from voiage.exceptions import InputError, raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ANALYSIS_TYPE = "distribution_family_information_value"
INFORMATION_TARGET = "model_family_index"
CONDITIONING_ORDER = "integrate_within_family_then_resolve_family_index"
TIE_POLICY = "complete-set-canonical-lexicographic-representative"


@dataclass(frozen=True)
class ResolvedDistributionModel:
    """Decision and contribution after resolving one model-family state."""

    model_id: str
    model_label: str
    probability: float
    conditional_values: list[float]
    optimal_alternatives: list[str]
    representative: str
    resolved_value: float
    weighted_contribution: float


@dataclass(frozen=True)
class DistributionalInformationResult:
    """Exact experimental distribution-family information result contract."""

    schema_version: str
    analysis_id: str
    analysis_type: str
    method_maturity: str
    information_target: str
    conditioning_order: str
    direction: str
    value_unit: str
    model_ids: list[str]
    model_labels: dict[str, str]
    model_definitions: list[dict[str, str]]
    model_probabilities: list[float]
    alternative_names: list[str]
    conditional_values: list[list[float]]
    conditional_value_assurance: dict[str, object]
    current_expected_values: list[float]
    current_value: float
    current_optimal_alternatives: list[str]
    current_representative: str
    resolved_models: list[ResolvedDistributionModel]
    expected_resolved_value: float
    gross_vdi: float
    information_cost: float
    net_vdi: float
    estimator: dict[str, object]
    comparability: dict[str, object]
    provenance: dict[str, str]
    diagnostics: dict[str, object]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output matching the checked-in v1 schema."""
        return asdict(self)


def _named_strings(values: Sequence[str], *, label: str) -> list[str]:
    if len(values) == 0 or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise_input_error(f"{label} must contain non-empty strings.")
    names = [item.strip() for item in values]
    if len(set(names)) != len(names):
        raise_input_error(f"{label} must be unique.")
    return names


def _exact_string_mapping(
    value: Mapping[str, str], keys: Sequence[str], *, label: str
) -> dict[str, str]:
    if set(value) != set(keys):
        raise_input_error(f"{label} keys must exactly match model_ids.")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in keys):
        raise_input_error(f"{label} values must be non-empty strings.")
    return {key: value[key].strip() for key in keys}


def _metadata_mapping(
    value: Mapping[str, str], required: set[str], *, label: str
) -> dict[str, str]:
    if set(value) != required:
        raise_input_error(f"{label} must contain exactly {sorted(required)}.")
    if any(
        not isinstance(value[key], str) or not value[key].strip() for key in required
    ):
        raise_input_error(f"{label} values must be non-empty strings.")
    return {key: value[key].strip() for key in sorted(required)}


def _validate_information_values(
    *,
    current_value: float,
    expected_resolved: float,
    gross: float,
    cost: float,
    net: float,
) -> None:
    """Reject violations of the exact value-of-information arithmetic contract."""
    if gross < 0:
        raise ArithmeticError(
            "Computed VDI is negative; exact conditioning or arithmetic is inconsistent."
        )
    if not all(
        math.isfinite(item)
        for item in (current_value, expected_resolved, gross, cost, net)
    ):
        raise ArithmeticError("Computed distribution-family information is non-finite.")


def _optimal_set(
    values: np.ndarray,
    alternatives: Sequence[str],
    *,
    direction: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, list[str], str]:
    selected = float(np.max(values) if direction == "maximize" else np.min(values))
    tied = sorted(
        name
        for name, value in zip(alternatives, values, strict=True)
        if math.isclose(
            float(value),
            selected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    )
    return selected, tied, tied[0]


def value_of_distributional_information(
    conditional_values: Sequence[Sequence[float]] | np.ndarray,
    model_ids: Sequence[str],
    alternative_names: Sequence[str],
    model_probabilities: Sequence[float],
    value_unit: str,
    provenance: Mapping[str, str],
    *,
    model_labels: Mapping[str, str],
    model_definitions: Sequence[Mapping[str, str]],
    conditional_value_assurance: Mapping[str, object],
    comparability: Mapping[str, object],
    analysis_id: str = "distribution-family-information-analysis",
    direction: str = "maximize",
    information_cost: float = 0.0,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    probability_sum_tolerance: float = 1e-12,
) -> DistributionalInformationResult:
    """Value perfect knowledge of a finite model-family index.

    Every row must already contain within-family conditional expected values.
    This function never optimizes within family draws and never infers or
    renormalizes model probabilities.
    """
    try:
        models = _named_strings(model_ids, label="model_ids")
        alternatives = _named_strings(alternative_names, label="alternative_names")
        if not isinstance(value_unit, str) or not value_unit.strip():
            raise_input_error("value_unit must be a non-empty string.")
        unit = value_unit.strip()
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            raise_input_error("analysis_id must be a non-empty string.")
        identifier = analysis_id.strip()
        if direction not in {"maximize", "minimize"}:
            raise_input_error("direction must be 'maximize' or 'minimize'.")
        tolerances = np.asarray(
            [absolute_tolerance, relative_tolerance, probability_sum_tolerance],
            dtype=float,
        )
        if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0):
            raise_input_error("tolerances must be finite and non-negative.")
        if probability_sum_tolerance <= 0:
            raise_input_error("probability_sum_tolerance must be positive.")

        labels = _exact_string_mapping(
            model_labels,
            models,
            label="model_labels",
        )
        definition_records = [dict(item) for item in model_definitions]
        assurance_record = dict(conditional_value_assurance)
        comparable = dict(comparability)
        provenance_record = _metadata_mapping(
            provenance,
            {
                "fixture_id",
                "probability_source",
                "value_source",
                "family_definition_source",
            },
            label="provenance",
        )

        values = np.asarray(conditional_values, dtype=float)
        probabilities = np.asarray(model_probabilities, dtype=float)
        payload = {
            "model_ids": models,
            "alternative_names": alternatives,
            "model_labels": labels,
            "model_definitions": definition_records,
            "model_probabilities": probabilities.tolist(),
            "conditional_values": values.tolist(),
            "conditional_value_assurance": assurance_record,
            "information_cost": information_cost,
            "tolerances": {
                "absolute": absolute_tolerance,
                "relative": relative_tolerance,
                "probability_sum": probability_sum_tolerance,
            },
            "comparability": comparable,
        }
        validate_distributional_information_semantics(payload)
        cost = float(information_cost)
    except InputError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise_input_error(str(error))
    if cost < 0:
        raise_input_error("information_cost must be non-negative.")

    current_expected = np.asarray(
        [
            math.fsum(
                float(probabilities[model_index])
                * float(values[model_index, alternative_index])
                for model_index in range(len(models))
            )
            for alternative_index in range(len(alternatives))
        ],
        dtype=float,
    )
    current_value, current_ties, current_representative = _optimal_set(
        current_expected,
        alternatives,
        direction=direction,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    resolved_models: list[ResolvedDistributionModel] = []
    for model_id, probability, row in zip(models, probabilities, values, strict=True):
        resolved_value, ties, representative = _optimal_set(
            row,
            alternatives,
            direction=direction,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        contribution = float(probability) * resolved_value
        resolved_models.append(
            ResolvedDistributionModel(
                model_id=model_id,
                model_label=labels[model_id],
                probability=float(probability),
                conditional_values=[float(item) for item in row],
                optimal_alternatives=ties,
                representative=representative,
                resolved_value=resolved_value,
                weighted_contribution=contribution,
            )
        )
    expected_resolved = math.fsum(
        item.weighted_contribution for item in resolved_models
    )
    raw_gross = (
        expected_resolved - current_value
        if direction == "maximize"
        else current_value - expected_resolved
    )
    gross = float(raw_gross)
    net = gross - cost
    _validate_information_values(
        current_value=current_value,
        expected_resolved=expected_resolved,
        gross=gross,
        cost=cost,
        net=net,
    )

    return DistributionalInformationResult(
        schema_version="1.0.0",
        analysis_id=identifier,
        analysis_type=ANALYSIS_TYPE,
        method_maturity="experimental",
        information_target=INFORMATION_TARGET,
        conditioning_order=CONDITIONING_ORDER,
        direction=direction,
        value_unit=unit,
        model_ids=models,
        model_labels=labels,
        model_definitions=[
            {key: str(value) for key, value in record.items()}
            for record in definition_records
        ],
        model_probabilities=[float(item) for item in probabilities],
        alternative_names=alternatives,
        conditional_values=[[float(item) for item in row] for row in values],
        conditional_value_assurance=assurance_record,
        current_expected_values=[float(item) for item in current_expected],
        current_value=current_value,
        current_optimal_alternatives=current_ties,
        current_representative=current_representative,
        resolved_models=resolved_models,
        expected_resolved_value=expected_resolved,
        gross_vdi=gross,
        information_cost=cost,
        net_vdi=net,
        estimator={
            "status": "exact_enumeration",
            "uncertainty_status": "exact",
            "input_value_status": assurance_record["input_status"],
            "evidence_reference": assurance_record["evidence_reference"],
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
        },
        comparability=comparable,
        provenance=provenance_record,
        diagnostics={
            "tie_policy": TIE_POLICY,
            "probability_sum": float(math.fsum(probabilities)),
            "nonnegativity_residual": 0.0,
            "conditioning_verified": bool(
                comparable["verified"] and assurance_record["source_values_exact"]
            ),
            "structural_evpi_relation": (
                "not_computed_model_family_only_is_bounded_by_matched_full_information"
            ),
        },
    )


def distributional_information_from_specification(
    payload: Mapping[str, object],
) -> DistributionalInformationResult:
    """Evaluate a semantically validated v1 JSON-compatible request."""
    Draft202012Validator(VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1).validate(
        payload
    )
    validate_distributional_information_semantics(payload)
    tolerances = cast("Mapping[str, float]", payload["tolerances"])
    return value_of_distributional_information(
        conditional_values=cast(
            "Sequence[Sequence[float]]", payload["conditional_values"]
        ),
        model_ids=cast("Sequence[str]", payload["model_ids"]),
        alternative_names=cast("Sequence[str]", payload["alternative_names"]),
        model_probabilities=cast("Sequence[float]", payload["model_probabilities"]),
        value_unit=str(payload["value_unit"]),
        provenance=cast("Mapping[str, str]", payload["provenance"]),
        model_labels=cast("Mapping[str, str]", payload["model_labels"]),
        model_definitions=cast(
            "Sequence[Mapping[str, str]]", payload["model_definitions"]
        ),
        conditional_value_assurance=cast(
            "Mapping[str, object]", payload["conditional_value_assurance"]
        ),
        comparability=cast("Mapping[str, object]", payload["comparability"]),
        analysis_id=str(payload["analysis_id"]),
        direction=str(payload["direction"]),
        information_cost=float(cast("float", payload["information_cost"])),
        absolute_tolerance=float(tolerances["absolute"]),
        relative_tolerance=float(tolerances["relative"]),
        probability_sum_tolerance=float(tolerances["probability_sum"]),
    )
