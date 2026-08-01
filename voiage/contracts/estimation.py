"""Typed contracts for estimation-focused variance-reduction VOI."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .analysis import ContractModel, Identifier, Sha256Digest


class EstimationTargetSpec(ContractModel):
    """Declared scalar or vector model output whose uncertainty is valued."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    target_id: Identifier
    shape: Literal["scalar", "vector"]
    component_units: tuple[Identifier, ...] = Field(min_length=1)
    covariance_functional: Literal[
        "variance",
        "trace",
        "determinant",
        "weighted_quadratic",
    ]
    functional_weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_shape_and_functional(self) -> Self:
        """Require an explicit, dimension-compatible covariance functional."""
        component_count = len(self.component_units)
        if self.shape == "scalar":
            if component_count != 1:
                raise ValueError("scalar targets require exactly one component unit")
            if self.covariance_functional != "variance":
                raise ValueError(
                    "scalar targets require covariance_functional='variance'"
                )
            if self.functional_weights is not None:
                raise ValueError(
                    "scalar variance targets do not accept functional weights"
                )
            return self

        if component_count < 2:
            raise ValueError("vector targets require at least two component units")
        if self.covariance_functional == "variance":
            raise ValueError("vector targets require a covariance scalarization")
        if self.covariance_functional == "weighted_quadratic":
            if self.functional_weights is None:
                raise ValueError("weighted_quadratic requires functional_weights")
            if len(self.functional_weights) != component_count:
                raise ValueError(
                    "functional_weights must match the vector component count"
                )
        elif self.functional_weights is not None:
            raise ValueError("functional_weights are only valid for weighted_quadratic")
        return self


class ConditioningSpec(ContractModel):
    """Sigma-field and averaging convention for partial perfect information."""

    parameter_subset: tuple[Identifier, ...] = Field(min_length=1)
    sigma_field: Identifier
    averaging_convention: Literal[
        "prior_predictive",
        "posterior_predictive",
        "empirical_reference",
    ]

    @model_validator(mode="after")
    def validate_unique_parameters(self) -> Self:
        """Reject ambiguous duplicate parameter identities."""
        if len(set(self.parameter_subset)) != len(self.parameter_subset):
            raise ValueError("parameter_subset identifiers must be unique")
        return self


class SamplingModelSpec(ContractModel):
    """Declared design and likelihood for sample-information variance VOI."""

    design_id: Identifier
    likelihood_id: Identifier
    conditioning_sigma_field: Identifier
    averaging_convention: Literal[
        "prior_predictive",
        "posterior_predictive",
        "empirical_reference",
    ]


class EstimatorAssuranceSpec(ContractModel):
    """Estimator identity, reproducibility seed, and numerical tolerances."""

    estimator_id: Identifier
    seed: int = Field(ge=0)
    absolute_tolerance: float = Field(default=1e-12, ge=0.0)
    relative_tolerance: float = Field(default=1e-10, ge=0.0)
    bootstrap_replicates: int = Field(default=0, ge=0)
    convergence_threshold: float = Field(default=0.01, gt=0.0)

    @model_validator(mode="after")
    def validate_bootstrap_replicates(self) -> Self:
        """Require enough replicates to estimate a standard error."""
        if self.bootstrap_replicates == 1:
            raise ValueError("bootstrap_replicates must be zero or at least two")
        return self


class EstimationVarianceSpec(ContractModel):
    """Complete scientific contract for one variance-reduction estimand."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    method_id: Literal["evppi_var", "evsi_var"]
    target: EstimationTargetSpec
    prior_model_id: Identifier
    conditioning: ConditioningSpec | None = None
    sampling_model: SamplingModelSpec | None = None
    estimator: EstimatorAssuranceSpec
    zero_variance_policy: Literal["absolute_zero_relative_null"] = (
        "absolute_zero_relative_null"
    )

    @model_validator(mode="after")
    def validate_information_contract(self) -> Self:
        """Require exactly the conditioning contract used by the estimand."""
        if self.method_id == "evppi_var":
            if self.conditioning is None:
                raise ValueError("evppi_var requires conditioning")
            if self.sampling_model is not None:
                raise ValueError("evppi_var does not accept sampling_model")
        else:
            if self.sampling_model is None:
                raise ValueError("evsi_var requires sampling_model")
            if self.conditioning is not None:
                raise ValueError("evsi_var does not accept conditioning")
            if self.sampling_model.averaging_convention != "prior_predictive":
                raise ValueError(
                    "evsi_var requires prior_predictive sampling-model averaging"
                )
        return self


class EstimationVarianceDiagnostics(ContractModel):
    """Numerical assurance kept distinct from the variance estimand."""

    prior_sample_count: int = Field(ge=2)
    posterior_evaluation_count: int = Field(ge=1)
    bootstrap_replicates: int = Field(default=0, ge=0)
    monte_carlo_standard_error: float | None = Field(default=None, ge=0.0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    confidence_interval: tuple[float, float] | None = None
    converged: bool
    diagnostic_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_confidence_interval(self) -> Self:
        """Reject inverted uncertainty intervals."""
        if (
            self.confidence_interval is not None
            and self.confidence_interval[0] > self.confidence_interval[1]
        ):
            raise ValueError("confidence_interval lower bound must not exceed upper")
        return self


class EstimationVarianceProvenance(ContractModel):
    """Replay identity for an estimation-focused variance result."""

    backend: Identifier
    kernel_version: Identifier
    estimator_id: Identifier
    seed: int = Field(ge=0)
    specification_digest: Sha256Digest
    input_digest: Sha256Digest


class EstimationVarianceResult(ContractModel):
    """Versioned scalarized variance-reduction result envelope."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    method_id: Literal["evppi_var", "evsi_var"]
    target: EstimationTargetSpec
    prior_covariance: tuple[tuple[float, ...], ...]
    expected_posterior_covariance: tuple[tuple[float, ...], ...]
    prior_functional: float = Field(ge=0.0)
    expected_posterior_functional: float = Field(ge=0.0)
    raw_reduction: float
    absolute_reduction: float = Field(ge=0.0)
    relative_reduction: float | None = Field(default=None, ge=0.0)
    functional_units: Identifier
    negative_estimate_policy: Literal["retain_raw_clip_reported"] = (
        "retain_raw_clip_reported"
    )
    zero_variance_policy: Literal["absolute_zero_relative_null"] = (
        "absolute_zero_relative_null"
    )
    diagnostics: EstimationVarianceDiagnostics
    provenance: EstimationVarianceProvenance

    @model_validator(mode="after")
    def validate_covariance_and_reduction(self) -> Self:
        """Require dimension-compatible covariance and reduction fields."""
        dimension = len(self.target.component_units)
        for field_name, covariance in (
            ("prior_covariance", self.prior_covariance),
            ("expected_posterior_covariance", self.expected_posterior_covariance),
        ):
            if len(covariance) != dimension or any(
                len(row) != dimension for row in covariance
            ):
                raise ValueError(
                    f"{field_name} must be square with the target component count"
                )
            for left in range(dimension):
                for right in range(dimension):
                    if abs(covariance[left][right] - covariance[right][left]) > 1e-10:
                        raise ValueError(f"{field_name} must be symmetric")

        if self.target.shape == "scalar":
            expected_units = f"{self.target.component_units[0]}^2"
            if self.functional_units != expected_units:
                raise ValueError(
                    "functional_units must equal the squared scalar target units"
                )
            for field_name, covariance, functional in (
                ("prior_covariance", self.prior_covariance, self.prior_functional),
                (
                    "expected_posterior_covariance",
                    self.expected_posterior_covariance,
                    self.expected_posterior_functional,
                ),
            ):
                variance = covariance[0][0]
                if variance < 0.0:
                    raise ValueError(
                        f"{field_name} scalar variance must be nonnegative"
                    )
                if variance != functional:
                    raise ValueError(
                        f"{field_name} scalar variance must equal its functional"
                    )

        expected_raw = self.prior_functional - self.expected_posterior_functional
        if abs(self.raw_reduction - expected_raw) > 1e-10:
            raise ValueError(
                "raw_reduction must equal prior minus expected posterior functional"
            )
        if abs(self.absolute_reduction - max(0.0, expected_raw)) > 1e-10:
            raise ValueError(
                "absolute_reduction must retain zero after clipping a negative estimate"
            )
        if self.prior_functional == 0.0:
            if self.relative_reduction is not None:
                raise ValueError(
                    "relative_reduction must be null when prior functional is zero"
                )
        else:
            expected_relative = self.absolute_reduction / self.prior_functional
            if (
                self.relative_reduction is None
                or abs(self.relative_reduction - expected_relative) > 1e-10
            ):
                raise ValueError(
                    "relative_reduction must equal absolute/prior functional"
                )
        return self


__all__ = [
    "ConditioningSpec",
    "EstimationTargetSpec",
    "EstimationVarianceDiagnostics",
    "EstimationVarianceProvenance",
    "EstimationVarianceResult",
    "EstimationVarianceSpec",
    "EstimatorAssuranceSpec",
    "SamplingModelSpec",
]
