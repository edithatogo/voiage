"""Pathological contract cases for estimation-focused variance VOI."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from voiage.contracts.estimation import (
    EstimationRuntimeBinding,
    EstimationTargetSpec,
    EstimationVarianceDiagnostics,
    EstimationVarianceProvenance,
    EstimationVarianceResult,
)


def _result_fields() -> dict[str, object]:
    runtime_binding = EstimationRuntimeBinding(
        method_id="evppi_var",
        target_id="net_cases",
        target_shape="scalar",
        component_units=("count",),
        covariance_functional="variance",
        prior_model_id="prior-v1",
        parameter_subset=("theta",),
        conditioning_sigma_field="sigma(theta)",
        averaging_convention="prior_predictive",
        estimator_design="exact",
        solver_id="rust-estimation-variance-v1",
    )
    return {
        "method_id": "evppi_var",
        "target": EstimationTargetSpec(
            target_id="net_cases",
            shape="scalar",
            component_units=("count",),
            covariance_functional="variance",
        ),
        "runtime_binding": runtime_binding,
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
            input_digest="b" * 64,
            runtime_binding_digest=runtime_binding.content_digest(),
            runtime_request_digest="c" * 64,
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


def test_vector_matrix_cases_cannot_bypass_reserved_boundary() -> None:
    fields = _result_fields()
    target = EstimationTargetSpec(
        target_id="two_outputs",
        shape="vector",
        component_units=("count", "usd"),
        covariance_functional="trace",
    )

    for covariance in (
        ((1.0,),),
        ((1.0, 0.2), (0.1, 2.0)),
        ((1.0, 2.0), (2.0, 1.0)),
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
        with pytest.raises(ValidationError, match="reserved vocabulary only"):
            _ = EstimationVarianceResult.model_validate(case)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"raw_reduction": 0.4}, "raw_reduction must equal"),
        ({"absolute_reduction": 0.4}, "absolute_reduction must retain"),
        (
            {
                "prior_covariance": ((0.0,),),
                "expected_posterior_covariance": ((0.0,),),
                "prior_functional": 0.0,
                "expected_posterior_functional": 0.0,
                "raw_reduction": 0.0,
                "absolute_reduction": 0.0,
                "relative_reduction": 0.0,
            },
            "relative_reduction must be null",
        ),
        ({"relative_reduction": None}, "relative_reduction must equal"),
        ({"relative_reduction": 0.4}, "relative_reduction must equal"),
    ],
)
def test_result_reduction_identities_fail_closed(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = EstimationVarianceResult.model_validate({**_result_fields(), **updates})


def test_expected_posterior_covariance_is_also_dimension_checked() -> None:
    fields = _result_fields()
    fields["expected_posterior_covariance"] = ((0.5, 0.0),)
    with pytest.raises(ValidationError, match="expected_posterior_covariance"):
        _ = EstimationVarianceResult.model_validate(fields)


def test_vector_result_envelopes_fail_closed_before_matrix_claims() -> None:
    fields = _result_fields()
    fields.update(
        {
            "target": EstimationTargetSpec(
                target_id="two_outputs",
                shape="vector",
                component_units=("count", "usd"),
                covariance_functional="trace",
            ),
            "prior_covariance": ((1.0, 0.0), (0.0, 2.0)),
            "expected_posterior_covariance": ((0.5, 0.0), (0.0, 1.0)),
            "prior_functional": 3.0,
            "expected_posterior_functional": 1.5,
            "raw_reduction": 1.4,
            "absolute_reduction": 1.5,
            "relative_reduction": 0.5,
            "functional_units": "declared_pending_vector_review",
        }
    )

    with pytest.raises(ValidationError, match="reserved vocabulary only"):
        _ = EstimationVarianceResult.model_validate(fields)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"prior_covariance": ((1.0 + 1e-12,),)},
            "prior_covariance scalar variance must equal",
        ),
        (
            {"expected_posterior_covariance": ((0.4,),)},
            "expected_posterior_covariance scalar variance must equal",
        ),
        ({"prior_covariance": ((-1.0,),)}, "scalar variance must be nonnegative"),
        ({"functional_units": "count"}, "squared scalar target units"),
    ],
)
def test_scalar_covariance_functional_and_units_fail_closed(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = EstimationVarianceResult.model_validate({**_result_fields(), **updates})


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


def test_runtime_binding_and_digest_forgery_fail_closed() -> None:
    fields = _result_fields()
    binding = fields["runtime_binding"]
    assert isinstance(binding, EstimationRuntimeBinding)
    fields["runtime_binding"] = binding.model_copy(update={"target_id": "forged"})
    with pytest.raises(ValidationError, match="runtime_binding disagrees"):
        EstimationVarianceResult.model_validate(fields)

    fields = _result_fields()
    provenance = fields["provenance"]
    assert isinstance(provenance, EstimationVarianceProvenance)
    fields["provenance"] = provenance.model_copy(
        update={"runtime_binding_digest": "d" * 64}
    )
    with pytest.raises(ValidationError, match="runtime_binding_digest disagrees"):
        EstimationVarianceResult.model_validate(fields)
