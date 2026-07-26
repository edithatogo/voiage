"""Typed runtime evidence for estimator statistical assurance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

ReportingClass = Literal[
    "deterministic",
    "sample-average",
    "regression-or-metamodel",
    "nested-monte-carlo",
    "moment-matching",
]
StoppingReason = Literal[
    "deterministic-complete",
    "fixed-budget",
    "tolerance-met",
    "budget-exhausted",
    "failed",
]


@dataclass(frozen=True)
class AssuranceConfidenceInterval:
    """Confidence interval attached to an estimator assurance result."""

    level: float
    lower: float
    upper: float
    method: str


@dataclass(frozen=True)
class AssuranceConvergence:
    """Convergence criterion and observed value."""

    converged: bool
    criterion: str
    observed: float


@dataclass(frozen=True)
class AssuranceRng:
    """Versioned random-number stream identity."""

    algorithm: str
    version: str
    seed: int
    stream: str


@dataclass(frozen=True)
class AssuranceBudget:
    """Computational budget observed at the estimator boundary."""

    draws: int
    evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class AssuranceNumericalError:
    """Declared floating-point error bounds and their source."""

    absolute_bound: float | None
    relative_bound: float | None
    source: str


@dataclass(frozen=True)
class StatisticalAssurance:
    """Portable estimator assurance matching the normative v1 schema."""

    reporting_class: ReportingClass
    bias_assessment: str | None
    variance_estimate: float | None
    monte_carlo_standard_error: float | None
    confidence_interval: AssuranceConfidenceInterval | None
    convergence: AssuranceConvergence | None
    effective_sample_size: float | None
    rng: AssuranceRng | None
    replications: int
    budget: AssuranceBudget
    stopping_reason: StoppingReason
    numerical_error: AssuranceNumericalError

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> StatisticalAssurance:
        """Construct typed assurance from a native runtime mapping."""
        interval_payload = cast(
            "dict[str, object] | None", payload["confidence_interval"]
        )
        convergence_payload = cast("dict[str, object] | None", payload["convergence"])
        rng_payload = cast("dict[str, object] | None", payload["rng"])
        budget_payload = cast("dict[str, object]", payload["budget"])
        numerical_error_payload = cast("dict[str, object]", payload["numerical_error"])

        return cls(
            reporting_class=cast("ReportingClass", payload["reporting_class"]),
            bias_assessment=cast("str | None", payload["bias_assessment"]),
            variance_estimate=cast("float | None", payload["variance_estimate"]),
            monte_carlo_standard_error=cast(
                "float | None", payload["monte_carlo_standard_error"]
            ),
            confidence_interval=(
                AssuranceConfidenceInterval(
                    level=cast("float", interval_payload["level"]),
                    lower=cast("float", interval_payload["lower"]),
                    upper=cast("float", interval_payload["upper"]),
                    method=cast("str", interval_payload["method"]),
                )
                if interval_payload is not None
                else None
            ),
            convergence=(
                AssuranceConvergence(
                    converged=cast("bool", convergence_payload["converged"]),
                    criterion=cast("str", convergence_payload["criterion"]),
                    observed=cast("float", convergence_payload["observed"]),
                )
                if convergence_payload is not None
                else None
            ),
            effective_sample_size=cast(
                "float | None", payload["effective_sample_size"]
            ),
            rng=(
                AssuranceRng(
                    algorithm=cast("str", rng_payload["algorithm"]),
                    version=cast("str", rng_payload["version"]),
                    seed=cast("int", rng_payload["seed"]),
                    stream=cast("str", rng_payload["stream"]),
                )
                if rng_payload is not None
                else None
            ),
            replications=cast("int", payload["replications"]),
            budget=AssuranceBudget(
                draws=cast("int", budget_payload["draws"]),
                evaluations=cast("int", budget_payload["evaluations"]),
                elapsed_seconds=cast("float", budget_payload["elapsed_seconds"]),
            ),
            stopping_reason=cast("StoppingReason", payload["stopping_reason"]),
            numerical_error=AssuranceNumericalError(
                absolute_bound=cast(
                    "float | None", numerical_error_payload["absolute_bound"]
                ),
                relative_bound=cast(
                    "float | None", numerical_error_payload["relative_bound"]
                ),
                source=cast("str", numerical_error_payload["source"]),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the schema-compatible plain mapping."""
        return cast("dict[str, object]", asdict(self))
