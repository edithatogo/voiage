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
