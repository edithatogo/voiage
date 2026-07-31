"""Typed contracts for estimation-focused variance-reduction VOI."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .analysis import ContractModel, Identifier


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
        return self


__all__ = [
    "ConditioningSpec",
    "EstimationTargetSpec",
    "EstimationVarianceSpec",
    "EstimatorAssuranceSpec",
    "SamplingModelSpec",
]
