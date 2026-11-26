"""Versioned Python contracts for Q05 coherent EVSI study models and diagnostics."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .analysis import ContractModel, Identifier


class SampleSizeSpecV1(ContractModel):
    """Explicit per-arm vs. total sample-size specification for EVSI studies."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    convention: Literal["per_arm", "total_equal_two_arm", "multi_arm"]
    per_arm: int | None = Field(default=None, gt=0)
    total: int | None = Field(default=None, gt=0)
    per_arm_sizes: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_convention(self) -> Self:
        """Enforce unambiguous sample-size semantics per convention."""
        if self.convention == "per_arm":
            if (
                self.per_arm is None
                or self.total is not None
                or self.per_arm_sizes is not None
            ):
                raise ValueError("per_arm convention requires only `per_arm > 0`")
        elif self.convention == "total_equal_two_arm":
            if (
                self.total is None
                or self.per_arm is not None
                or self.per_arm_sizes is not None
            ):
                raise ValueError(
                    "total_equal_two_arm convention requires only `total > 0`"
                )
            if self.total % 2 != 0:
                raise ValueError(
                    "total_equal_two_arm requires an even total sample size"
                )
        elif self.convention == "multi_arm" and (
            self.per_arm_sizes is None
            or len(self.per_arm_sizes) < 2
            or any(n <= 0 for n in self.per_arm_sizes)
            or self.per_arm is not None
            or self.total is not None
        ):
            raise ValueError(
                "multi_arm convention requires `per_arm_sizes` with >= 2 positive arm sizes"
            )
        return self

    def resolved_total(self) -> int:
        """Return resolved total sample size across all study arms."""
        if self.convention == "per_arm":
            assert self.per_arm is not None
            return 2 * self.per_arm
        if self.convention == "total_equal_two_arm":
            assert self.total is not None
            return self.total
        assert self.per_arm_sizes is not None
        return sum(self.per_arm_sizes)


class EvsiDiagnosticsV1(ContractModel):
    """Coherent diagnostic fields shared across EVSI estimators."""

    ess: float | None = Field(default=None, ge=0.0)
    khat: float | None = None
    decision_error: float | None = Field(default=None, ge=0.0, le=1.0)
    monte_carlo_se: float | None = Field(default=None, ge=0.0)
    bias: float | None = None
    rmse: float | None = Field(default=None, ge=0.0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)


class EvsiStudyResultV1(ContractModel):
    """Versioned EVSI study result with estimator provenance and diagnostics."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    estimator: Literal[
        "conjugate_analytic",
        "nested_monte_carlo",
        "regression",
        "moment_based",
        "importance_sampling",
        "qmc_halton",
        "multilevel",
    ]
    study_model: Identifier
    sample_size: SampleSizeSpecV1
    evsi: float
    expected_current_value: float
    expected_sample_value: float
    diagnostics: EvsiDiagnosticsV1


class PosteriorSampleBoundaryV1(ContractModel):
    """F02 framework-agnostic posterior-sample and MCMC diagnostic boundary."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    framework: Identifier
    parameter_names: tuple[Identifier, ...] = Field(min_length=1)
    draw_count: int = Field(gt=0)
    chain_count: int = Field(gt=0)
    min_bulk_ess: float = Field(gt=0.0)
    max_rhat: float = Field(gt=0.0)
    divergent_transitions: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_convergence(self) -> Self:
        """Fail closed on duplicate parameter names or severe convergence failures."""
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")
        if self.max_rhat > 1.1:
            raise ValueError(f"max_rhat={self.max_rhat} exceeds 1.1 convergence gate")
        if self.divergent_transitions > 0:
            raise ValueError(
                f"divergent_transitions={self.divergent_transitions} must be zero"
            )
        return self
