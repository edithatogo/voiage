"""Compatibility adapters for established VOIAGE runtime backends."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from voiage.contracts.analysis import (
    AnalysisSpec,
    NumericalPolicy,
    ParameterDType,
    ParameterSpec,
)
from voiage.schema import ParameterSet, ValueArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from voiage.contracts.capabilities import BackendCapabilities
    from voiage.main_backends import Backend


@dataclass(frozen=True)
class LegacyBackendAdapter:
    """Expose an existing backend through the capability-aware pilot contract."""

    backend: Backend
    capabilities: BackendCapabilities

    def calculate_evpi(self, net_benefits: np.ndarray) -> float:
        """Delegate without changing established backend return behavior."""
        return float(self.backend.evpi(net_benefits))


def adapt_backend(backend: Backend) -> LegacyBackendAdapter:
    """Describe a built-in backend without adding abstract legacy methods."""
    return LegacyBackendAdapter(
        backend=backend,
        capabilities=backend.capability_descriptor,
    )


def adapt_value_array(
    values: ValueArray | ArrayLike,
    *,
    strategy_names: Sequence[str] | None = None,
    perspective_names: Sequence[str] | None = None,
) -> ValueArray:
    """Normalize supported arrays while preserving established instances."""
    if isinstance(values, ValueArray):
        return values
    array = np.asarray(values)
    strategies = list(strategy_names) if strategy_names is not None else None
    if array.ndim == 2:
        if perspective_names is not None:
            raise ValueError("perspective_names require a 3D value array")
        return ValueArray.from_numpy(array, strategies)
    if array.ndim == 3:
        return ValueArray.from_numpy_perspectives(
            array,
            strategies,
            list(perspective_names) if perspective_names is not None else None,
        )
    raise ValueError("values must be a 2D or 3D array")


def adapt_parameter_set(
    parameters: ParameterSet | Mapping[str, np.ndarray] | np.ndarray | None,
) -> ParameterSet | None:
    """Normalize parameter samples while preserving established instances."""
    if parameters is None or isinstance(parameters, ParameterSet):
        return parameters
    normalized = dict(parameters) if isinstance(parameters, Mapping) else parameters
    return ParameterSet.from_numpy_or_dict(normalized)


def _parameter_dtype(values: np.ndarray) -> ParameterDType:
    if np.issubdtype(values.dtype, np.bool_):
        return "bool"
    if np.issubdtype(values.dtype, np.integer):
        return "int64"
    if np.issubdtype(values.dtype, np.str_):
        return "string"
    return "float64" if values.dtype.itemsize > 4 else "float32"


def analysis_spec_from_inputs(
    *,
    analysis_id: str,
    decision_problem_id: str,
    method_family: str,
    method_contract_version: str,
    values: ValueArray,
    parameters: ParameterSet | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> AnalysisSpec:
    """Build a declarative specification from existing numerical containers."""
    parameter_specs = tuple(
        ParameterSpec(
            parameter_id=name,
            role="uncertain",
            dtype=_parameter_dtype(np.asarray(samples)),
            dimensions=("n_samples",),
        )
        for name, samples in (parameters.parameters.items() if parameters else ())
    )
    return AnalysisSpec(
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        method_family=method_family,
        method_contract_version=method_contract_version,
        strategy_names=tuple(values.strategy_names),
        parameters=parameter_specs,
        numerical_policy=numerical_policy or NumericalPolicy(),
    )
