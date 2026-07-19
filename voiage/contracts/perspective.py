"""Additive perspective-result envelope and compatibility adapters."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import numpy as np
from pydantic import JsonValue  # noqa: TC002 - Pydantic resolves runtime annotations
from pydantic_core import to_jsonable_python

from voiage.contracts.adapters import (
    adapt_backend,
    adapt_parameter_set,
    adapt_value_array,
    analysis_spec_from_inputs,
)
from voiage.contracts.analysis import (
    AnalysisResult,
    ContractModel,
    DiagnosticEnvelope,
    NumericalPolicy,
    Provenance,
    RunContext,
)
from voiage.contracts.capabilities import (
    Capability,
    KernelRequirements,
    select_backend,
)
from voiage.main_backends import get_backend
from voiage.methods.perspective import (
    METHOD_CONTRACT_VERSION,
    Perspective,
    PerspectiveSet,
    ValueOfPerspectiveResult,
    value_of_perspective,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from voiage.schema import ParameterSet, ValueArray


class PerspectivePayload(ContractModel):
    """JSON-safe representation of the established perspective result."""

    value: float
    perspective_ids: tuple[str, ...]
    perspective_labels: tuple[str, ...]
    strategy_names: tuple[str, ...]
    expected_net_benefits: tuple[tuple[float, ...], ...]
    optimal_strategy_indices: tuple[int, ...]
    optimal_strategy_names: tuple[str, ...]
    optimal_expected_net_benefits: tuple[float, ...]
    regret_matrix: tuple[tuple[float, ...], ...]
    switching_values: tuple[float, ...]
    consensus_strategy_index: int
    consensus_strategy_name: str
    consensus_weighted_expected_net_benefit: float
    robust_strategy_index: int
    robust_strategy_name: str
    pareto_strategy_indices: tuple[int, ...]
    pareto_strategy_names: tuple[str, ...]
    perspective_weights: tuple[float, ...]
    reference_perspective_id: str
    method_maturity: str
    diagnostics: dict[str, JsonValue]
    reporting: dict[str, JsonValue]


def _rows(values: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    rows = cast("list[list[float]]", values.tolist())
    return tuple(tuple(row) for row in rows)


def _float_values(values: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(cast("list[float]", values.tolist()))


def _int_values(values: NDArray[np.int64]) -> tuple[int, ...]:
    return tuple(cast("list[int]", values.tolist()))


def _json_mapping(values: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized: object = to_jsonable_python(  # pyright: ignore[reportAny]
        values, serialize_unknown=True, fallback=str
    )
    return cast("dict[str, JsonValue]", normalized)


def adapt_perspective_result(result: ValueOfPerspectiveResult) -> PerspectivePayload:
    """Convert a perspective dataclass without mutating or replacing it."""
    return PerspectivePayload(
        value=float(result.value),
        perspective_ids=tuple(result.perspective_ids),
        perspective_labels=tuple(result.perspective_labels),
        strategy_names=tuple(result.strategy_names),
        expected_net_benefits=_rows(result.expected_net_benefits),
        optimal_strategy_indices=_int_values(result.optimal_strategy_indices),
        optimal_strategy_names=tuple(result.optimal_strategy_names),
        optimal_expected_net_benefits=_float_values(
            result.optimal_expected_net_benefits
        ),
        regret_matrix=_rows(result.regret_matrix),
        switching_values=_float_values(result.switching_values),
        consensus_strategy_index=int(result.consensus_strategy_index),
        consensus_strategy_name=result.consensus_strategy_name,
        consensus_weighted_expected_net_benefit=float(
            result.consensus_weighted_expected_net_benefit
        ),
        robust_strategy_index=int(result.robust_strategy_index),
        robust_strategy_name=result.robust_strategy_name,
        pareto_strategy_indices=tuple(result.pareto_strategy_indices),
        pareto_strategy_names=tuple(result.pareto_strategy_names),
        perspective_weights=_float_values(result.perspective_weights),
        reference_perspective_id=result.reference_perspective_id,
        method_maturity=result.method_maturity,
        diagnostics=_json_mapping(result.diagnostics),
        reporting=_json_mapping(result.reporting),
    )


def _package_version() -> str:
    try:
        return version("voiage")
    except PackageNotFoundError:  # pragma: no cover - editable installs provide it
        return "0.0.0"


def run_perspective(
    net_benefits: ValueArray | np.ndarray,
    *,
    analysis_id: str,
    decision_problem_id: str,
    parameters: ParameterSet | Mapping[str, np.ndarray] | np.ndarray | None = None,
    perspectives: PerspectiveSet | Sequence[Perspective | str] | None = None,
    strategy_names: Sequence[str] | None = None,
    perspective_names: Sequence[str] | None = None,
    perspective_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_perspective: str | int | None = None,
    tie_policy: str = "first",
    tie_tolerance: float = 1e-12,
    policy: NumericalPolicy | None = None,
) -> AnalysisResult[PerspectivePayload]:
    """Run perspective VOI and return an opt-in canonical envelope."""
    resolved_policy = policy or NumericalPolicy()
    values = adapt_value_array(
        net_benefits,
        strategy_names=strategy_names,
        perspective_names=perspective_names,
    )
    parameter_set = adapt_parameter_set(parameters)
    spec = analysis_spec_from_inputs(
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        method_family="value_of_perspective",
        method_contract_version=METHOD_CONTRACT_VERSION,
        values=values,
        parameters=parameter_set,
        numerical_policy=resolved_policy,
    )
    candidates = [
        adapt_backend(get_backend(name)) for name in resolved_policy.backend_preference
    ]
    requirements = KernelRequirements(
        method_family="value_of_perspective",
        dtype=resolved_policy.dtype,
        device=resolved_policy.device,
        required_features=frozenset({Capability.DENSE_ARRAY, Capability.DETERMINISTIC}),
    )
    selected = select_backend(
        candidates if resolved_policy.allow_fallback else candidates[:1], requirements
    )
    descriptor = selected.capabilities
    legacy = value_of_perspective(
        values,
        perspectives=perspectives,
        strategy_names=strategy_names,
        perspective_names=perspective_names,
        perspective_weights=perspective_weights,
        reference_perspective=reference_perspective,
        tie_policy=tie_policy,
        tie_tolerance=tie_tolerance,
    )
    package_version = _package_version()
    value_bytes = np.asarray(values.numpy_values, dtype="<f8").tobytes(order="C")
    context = RunContext(
        run_id=uuid4().hex,
        spec_digest=spec.contract_digest(),
        input_digest=hashlib.sha256(value_bytes).hexdigest(),
        requested_backend=resolved_policy.backend_preference[0],
        selected_backend=descriptor.backend_name,
        backend_version=descriptor.backend_version,
        device=resolved_policy.device or sorted(descriptor.devices)[0],
        capabilities=frozenset(item.value for item in descriptor.features),
        package_version=package_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    return AnalysisResult[PerspectivePayload](
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        method_family="value_of_perspective",
        method_contract_version=METHOD_CONTRACT_VERSION,
        method_maturity="fixture-backed",
        numerical_policy=resolved_policy,
        payload=adapt_perspective_result(legacy),
        run_context=context,
        diagnostics=DiagnosticEnvelope(
            analysis_id=analysis_id,
            backend=descriptor.backend_name,
        ),
        provenance=Provenance(
            backend=descriptor.backend_name,
            method_family="value_of_perspective",
            package_version=package_version,
            seed=resolved_policy.seed,
            input_artifact_ids=spec.input_artifact_ids,
            details={"adapter": "legacy-perspective-result"},
        ),
    )
