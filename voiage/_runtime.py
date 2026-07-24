"""Private lazy-loading adapter for the native Rust extension."""

from __future__ import annotations

from importlib import import_module
from typing import Any, NoReturn

from voiage.exceptions import (
    DimensionMismatchError,
    InputError,
    SerializationError,
)

_NATIVE_MODULE = "voiage._core"


def _native() -> Any:
    """Load the private extension only when a Rust-owned operation is used."""
    return import_module(_NATIVE_MODULE)


def _raise_native_error(error: Exception) -> NoReturn:
    """Translate a stable native category without hiding unexpected failures."""
    category = getattr(error, "category", None)
    diagnostic_code = getattr(error, "diagnostic_code", None)
    message = str(error)
    native = _native()
    native_categories = (
        (getattr(native, "InputError", ()), "input"),
        (getattr(native, "DimensionMismatchError", ()), "dimension_mismatch"),
        (getattr(native, "SerializationError", ()), "serialization"),
    )
    for native_error, native_category in native_categories:
        if native_error and isinstance(error, native_error):
            category = native_category
            if len(error.args) == 2:
                diagnostic_code, message = error.args
            break
    if category == "input":
        raise InputError(message, diagnostic_code=diagnostic_code) from error
    if category == "dimension_mismatch":
        raise DimensionMismatchError(
            message,
            diagnostic_code=diagnostic_code,
        ) from error
    if category == "serialization":
        translated = SerializationError(message)
        vars(translated)["diagnostic_code"] = diagnostic_code
        raise translated from error
    raise error


def serialize_ceaf_result(**payload: object) -> dict[str, object]:
    """Return the Rust-owned canonical CEAF result payload."""
    try:
        result = _native().serialize_ceaf_result(**payload)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def serialize_dominance_result(**payload: object) -> dict[str, object]:
    """Return the Rust-owned canonical dominance result payload."""
    try:
        result = _native().serialize_dominance_result(**payload)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def serialize_expected_loss_result(**payload: object) -> dict[str, object]:
    """Return the Rust-owned canonical expected-loss result payload."""
    try:
        result = _native().serialize_expected_loss_result(**payload)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_evpi(net_benefit: list[list[float]]) -> float:
    """Compute the stable EVPI kernel in Rust."""
    try:
        result = _native().compute_evpi(net_benefit)
    except Exception as error:
        _raise_native_error(error)
    return float(result)


def compute_expected_loss(net_benefit: list[list[float]]) -> dict[str, object]:
    """Compute the stable expected opportunity-loss kernel in Rust."""
    try:
        result = _native().compute_expected_loss(net_benefit)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_enbs(evsi_result: float, research_cost: float) -> float:
    """Compute the stable ENBS kernel in Rust."""
    try:
        result = _native().compute_enbs(evsi_result, research_cost)
    except Exception as error:
        _raise_native_error(error)
    return float(result)


def compute_heterogeneity(
    net_benefit: list[list[float]], subgroups: list[str]
) -> dict[str, object]:
    """Compute the stable value-of-heterogeneity kernel in Rust."""
    try:
        result = _native().compute_heterogeneity(net_benefit, subgroups)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_structural_evpi(
    net_benefit_by_structure: list[list[list[float]]],
    structure_probabilities: list[float],
) -> float:
    """Aggregate structural EVPI in the Rust core."""
    try:
        result = _native().compute_structural_evpi(
            net_benefit_by_structure,
            structure_probabilities,
        )
    except Exception as error:
        _raise_native_error(error)
    return float(result)


def compute_structural_evppi(
    net_benefit_by_structure: list[list[list[float]]],
    structure_probabilities: list[float],
    structures_of_interest: list[int],
) -> float:
    """Aggregate structural EVPPI in the Rust core."""
    try:
        result = _native().compute_structural_evppi(
            net_benefit_by_structure,
            structure_probabilities,
            structures_of_interest,
        )
    except Exception as error:
        _raise_native_error(error)
    return float(result)


def compute_dominance(costs: list[float], effects: list[float]) -> dict[str, object]:
    """Compute the stable dominance kernel in Rust."""
    try:
        result = _native().compute_dominance(costs, effects)
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_ceaf(
    net_benefit: list[list[list[float]]],
    wtp_thresholds: list[float],
    confidence_level: float,
) -> dict[str, object]:
    """Compute the stable CEAF kernel in Rust."""
    try:
        result = _native().compute_ceaf(
            net_benefit,
            wtp_thresholds,
            confidence_level,
        )
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_evppi(
    net_benefit: list[list[float]], parameter_samples: list[list[float]]
) -> float:
    """Compute the stable full-sample linear EVPPI kernel in Rust."""
    try:
        result = _native().compute_evppi(net_benefit, parameter_samples)
    except Exception as error:
        _raise_native_error(error)
    return float(result)


def compute_evsi(
    net_benefit: list[list[float]],
    trial_sample_size: int,
    resample_count: int,
    seed: int,
) -> dict[str, object]:
    """Compute the explicit seeded-bootstrap EVSI kernel in Rust."""
    try:
        result = _native().compute_evsi(
            net_benefit,
            trial_sample_size,
            resample_count,
            seed,
        )
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_evsi_efficient_linear(
    net_benefit: list[list[float]],
    parameter_samples: list[list[float]],
    trial_sample_size: int,
) -> dict[str, object]:
    """Compute the explicit deterministic efficient-linear EVSI kernel."""
    try:
        result = _native().compute_evsi_efficient_linear(
            net_benefit,
            parameter_samples,
            trial_sample_size,
        )
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_evsi_moment_based(
    net_benefit: list[list[float]],
    parameter_samples: list[list[float]],
    trial_sample_size: int,
) -> dict[str, object]:
    """Compute the explicit deterministic moment-based EVSI kernel."""
    try:
        result = _native().compute_evsi_moment_based(
            net_benefit,
            parameter_samples,
            trial_sample_size,
        )
    except Exception as error:
        _raise_native_error(error)
    return dict(result)


def compute_evsi_regression(
    targets: list[list[float]],
    parameter_samples: list[list[float]],
    prediction_samples: list[list[float]],
) -> dict[str, object]:
    """Compute the deterministic Rust regression aggregation kernel."""
    try:
        result = _native().compute_evsi_regression(
            targets,
            parameter_samples,
            prediction_samples,
        )
    except Exception as error:
        _raise_native_error(error)
    return dict(result)
