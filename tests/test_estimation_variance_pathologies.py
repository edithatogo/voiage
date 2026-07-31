"""Pathological contract cases for estimation-focused variance VOI."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from voiage.contracts.estimation import (
    EstimationTargetSpec,
    EstimationVarianceDiagnostics,
    EstimationVarianceProvenance,
    EstimationVarianceResult,
)


def _result_fields() -> dict[str, object]:
    return {
        "method_id": "evppi_var",
        "target": EstimationTargetSpec(
            target_id="net_cases",
            shape="scalar",
            component_units=("count",),
            covariance_functional="variance",
        ),
        "prior_covariance": ((1.0,),),
        "expected_posterior_covariance": ((0.5,),),
        "prior_functional": 1.0,
        "expected_posterior_functional": 0.5,
        "raw_reduction": 0.5,
        "absolute_reduction": 0.5,
        "relative_reduction": 0.5,
        "functional_units": "count^2",
        "diagnostics": EstimationVarianceDiagnostics(
            prior_sample_count=20,
            posterior_evaluation_count=4,
            monte_carlo_standard_error=0.01,
            converged=True,
        ),
        "provenance": EstimationVarianceProvenance(
            backend="rust",
            kernel_version="1.0.0",
            estimator_id="discrete_conditioning",
            seed=0,
            specification_digest="a" * 64,
        ),
    }


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_result_rejects_non_finite_numerical_fields(non_finite: float) -> None:
    fields = _result_fields()
    fields["prior_covariance"] = ((non_finite,),)
    with pytest.raises(ValidationError, match="finite"):
        _ = EstimationVarianceResult.model_validate(fields)


def test_diagnostics_reject_insufficient_prior_samples() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        _ = EstimationVarianceDiagnostics(
            prior_sample_count=1,
            posterior_evaluation_count=1,
            monte_carlo_standard_error=0.0,
            converged=True,
        )


def test_result_rejects_dimension_mismatch_and_asymmetric_covariance() -> None:
    fields = _result_fields()
    target = EstimationTargetSpec(
        target_id="two_outputs",
        shape="vector",
        component_units=("count", "usd"),
        covariance_functional="trace",
    )

    for covariance, message in (
        (((1.0,),), "square with the target component count"),
        (((1.0, 0.2), (0.1, 2.0)), "must be symmetric"),
    ):
        case: dict[str, object] = {
            **fields,
            "target": target,
            "prior_covariance": covariance,
            "expected_posterior_covariance": ((0.5, 0.0), (0.0, 1.0)),
            "prior_functional": 3.0,
            "expected_posterior_functional": 1.5,
            "raw_reduction": 1.5,
            "absolute_reduction": 1.5,
            "relative_reduction": 0.5,
        }
        with pytest.raises(ValidationError, match=message):
            _ = EstimationVarianceResult.model_validate(case)


def test_non_convergence_is_explicit_diagnostic_not_a_silent_success() -> None:
    fields = _result_fields()
    fields["diagnostics"] = EstimationVarianceDiagnostics(
        prior_sample_count=20,
        posterior_evaluation_count=4,
        monte_carlo_standard_error=0.2,
        converged=False,
        diagnostic_codes=("convergence_threshold_not_met",),
    )
    result = EstimationVarianceResult.model_validate(fields)
    assert result.diagnostics.converged is False
    assert result.diagnostics.diagnostic_codes == ("convergence_threshold_not_met",)
