"""Estimation-focused variance-reduction value of information.

This module is intentionally separate from decision-focused EVPPI and EVSI.
Its public names describe reductions in uncertainty about a declared target,
not expected changes in a decision's net benefit.
"""

from __future__ import annotations

from hashlib import sha256
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from voiage._runtime import compute_evppi_variance, compute_evsi_variance
from voiage.contracts.estimation import (
    EstimationVarianceDiagnostics,
    EstimationVarianceProvenance,
    EstimationVarianceResult,
    EstimationVarianceSpec,
)
from voiage.exceptions import InputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_EVPPI_VAR: Final[Mapping[str, object]] = MappingProxyType(
    {
        "method_id": "evppi_var",
        "aliases": ("evppi_variance",),
        "family": "estimation-focused-variance-voi",
        "estimand_kind": "variance_reduction",
        "information_kind": "partial_perfect",
        "decision_focused": False,
        "sensitivity_index": False,
        "estimator_uncertainty": False,
        "maturity": "experimental",
    }
)
_EVSI_VAR: Final[Mapping[str, object]] = MappingProxyType(
    {
        "method_id": "evsi_var",
        "aliases": ("evsi_variance",),
        "family": "estimation-focused-variance-voi",
        "estimand_kind": "variance_reduction",
        "information_kind": "sample",
        "decision_focused": False,
        "sensitivity_index": False,
        "estimator_uncertainty": False,
        "maturity": "experimental",
    }
)

ESTIMATION_VARIANCE_METHODS: Final[Mapping[str, Mapping[str, object]]] = (
    MappingProxyType(
        {
            "evppi_var": _EVPPI_VAR,
            "evsi_var": _EVSI_VAR,
        }
    )
)

_ESTIMATION_VARIANCE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "evppi_var": "evppi_var",
        "evppi_variance": "evppi_var",
        "evsi_var": "evsi_var",
        "evsi_variance": "evsi_var",
    }
)


def estimation_variance_method(name: str) -> Mapping[str, object]:
    """Return the governed descriptor for an estimation-variance method.

    Decision-focused EVPPI/EVSI, sensitivity indices, and posterior-estimator
    uncertainty are deliberately not accepted as aliases.

    Parameters
    ----------
    name
        Canonical estimation method ID or an explicitly governed alias.

    Returns
    -------
    collections.abc.Mapping
        Immutable method metadata.

    Raises
    ------
    ValueError
        If ``name`` is not an estimation-focused variance method.
    """
    try:
        method_id = _ESTIMATION_VARIANCE_ALIASES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{name!r} is not an estimation-focused variance method"
        ) from error
    return ESTIMATION_VARIANCE_METHODS[method_id]


def _validate_runtime_spec(
    specification: EstimationVarianceSpec,
    method_id: str,
) -> None:
    if specification.method_id != method_id:
        raise InputError(
            f"{method_id} requires a matching EstimationVarianceSpec",
            diagnostic_code="estimation_method_mismatch",
        )
    if (
        specification.target.shape != "scalar"
        or specification.target.covariance_functional != "variance"
    ):
        message = " ".join(
            (
                "the experimental runtime supports scalar variance targets only;",
                "vector covariance scalarization remains pending scientific review",
            )
        )
        raise InputError(
            message,
            diagnostic_code="unsupported_estimation_target",
        )


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"native estimation result field {key!r} is not numeric")
    return float(value)


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"native estimation result field {key!r} is not an integer")
    return value


def _optional_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"native estimation result field {key!r} is not numeric or null"
        )
    return float(value)


def _confidence_interval(
    payload: Mapping[str, object],
) -> tuple[float, float] | None:
    value = payload["confidence_interval"]
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("native estimation confidence interval is not a pair")
    components = cast("Sequence[object]", value)
    if len(components) != 2:
        raise TypeError("native estimation confidence interval is not a pair")
    lower, upper = components
    if (
        isinstance(lower, bool)
        or not isinstance(lower, (int, float))
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
    ):
        raise TypeError("native estimation confidence interval is not numeric")
    return float(lower), float(upper)


def _specification_digest(specification: EstimationVarianceSpec) -> str:
    canonical = json.dumps(
        specification.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return sha256(canonical).hexdigest()


def _result_from_native(
    specification: EstimationVarianceSpec,
    payload: Mapping[str, object],
) -> EstimationVarianceResult:
    relative_value = payload["relative_reduction"]
    if relative_value is not None and not isinstance(relative_value, (int, float)):
        raise TypeError(
            "native estimation result field 'relative_reduction' is not numeric or null"
        )
    kernel_version = payload["kernel_version"]
    if not isinstance(kernel_version, str):
        raise TypeError("native estimation kernel version is not text")
    prior = _number(payload, "prior_variance")
    posterior = _number(payload, "expected_posterior_variance")
    converged = payload["converged"]
    if not isinstance(converged, bool):
        raise TypeError("native estimation convergence field is not boolean")
    standard_error = _optional_number(payload, "monte_carlo_standard_error")
    diagnostic_codes = (
        ("monte_carlo_uncertainty_not_estimated",)
        if standard_error is None
        else (() if converged else ("convergence_threshold_not_met",))
    )
    return EstimationVarianceResult(
        method_id=specification.method_id,
        target=specification.target,
        prior_covariance=((prior,),),
        expected_posterior_covariance=((posterior,),),
        prior_functional=prior,
        expected_posterior_functional=posterior,
        raw_reduction=_number(payload, "raw_reduction"),
        absolute_reduction=_number(payload, "absolute_reduction"),
        relative_reduction=(
            None if relative_value is None else float(relative_value)
        ),
        functional_units=f"{specification.target.component_units[0]}^2",
        diagnostics=EstimationVarianceDiagnostics(
            prior_sample_count=_integer(payload, "prior_sample_count"),
            posterior_evaluation_count=_integer(
                payload, "posterior_evaluation_count"
            ),
            bootstrap_replicates=_integer(payload, "bootstrap_replicates"),
            monte_carlo_standard_error=standard_error,
            confidence_interval=_confidence_interval(payload),
            converged=converged,
            diagnostic_codes=diagnostic_codes,
        ),
        provenance=EstimationVarianceProvenance(
            backend="rust",
            kernel_version=kernel_version,
            estimator_id=specification.estimator.estimator_id,
            seed=specification.estimator.seed,
            specification_digest=_specification_digest(specification),
        ),
    )


def evppi_var(
    target_samples: Sequence[float],
    conditioning_groups: Sequence[str],
    *,
    specification: EstimationVarianceSpec,
) -> EstimationVarianceResult:
    """Estimate scalar variance reduction from perfect discrete conditioning."""
    _validate_runtime_spec(specification, "evppi_var")
    payload = compute_evppi_variance(
        [float(value) for value in target_samples],
        [str(group) for group in conditioning_groups],
        specification.estimator.bootstrap_replicates,
        specification.estimator.seed,
        specification.estimator.convergence_threshold,
    )
    return _result_from_native(specification, payload)


def evsi_var(
    prior_target_samples: Sequence[float],
    posterior_variances: Sequence[float],
    *,
    specification: EstimationVarianceSpec,
) -> EstimationVarianceResult:
    """Aggregate scalar variance reduction across declared study outcomes."""
    _validate_runtime_spec(specification, "evsi_var")
    payload = compute_evsi_variance(
        [float(value) for value in prior_target_samples],
        [float(value) for value in posterior_variances],
        specification.estimator.bootstrap_replicates,
        specification.estimator.seed,
        specification.estimator.convergence_threshold,
    )
    return _result_from_native(specification, payload)


__all__ = [
    "ESTIMATION_VARIANCE_METHODS",
    "estimation_variance_method",
    "evppi_var",
    "evsi_var",
]
