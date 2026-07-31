"""Native-backed Python façade tests for estimation variance VOI."""

# pyright: reportAny=false

from __future__ import annotations

import json

import pytest

from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationTargetSpec,
    EstimationVarianceResult,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
    SamplingModelSpec,
)
from voiage.exceptions import DimensionMismatchError, InputError
import voiage.methods.estimation as estimation_module


def _target() -> EstimationTargetSpec:
    return EstimationTargetSpec(
        target_id="net_cases",
        shape="scalar",
        component_units=("count",),
        covariance_functional="variance",
    )


def _assurance(
    estimator_id: str,
    *,
    bootstrap_replicates: int = 0,
    convergence_threshold: float = 0.01,
) -> EstimatorAssuranceSpec:
    return EstimatorAssuranceSpec(
        estimator_id=estimator_id,
        seed=17,
        bootstrap_replicates=bootstrap_replicates,
        convergence_threshold=convergence_threshold,
    )


def _evppi_spec() -> EstimationVarianceSpec:
    return EstimationVarianceSpec(
        method_id="evppi_var",
        target=_target(),
        prior_model_id="enumerable_prior",
        conditioning=ConditioningSpec(
            parameter_subset=("risk_state",),
            sigma_field="sigma_risk_state",
            averaging_convention="empirical_reference",
        ),
        estimator=_assurance("discrete_conditioning"),
    )


def _evsi_spec() -> EstimationVarianceSpec:
    return EstimationVarianceSpec(
        method_id="evsi_var",
        target=_target(),
        prior_model_id="study_prior",
        sampling_model=SamplingModelSpec(
            design_id="binary_study",
            likelihood_id="binary_accuracy",
            conditioning_sigma_field="sigma_observation",
            averaging_convention="prior_predictive",
        ),
        estimator=_assurance("posterior_variance_aggregation"),
    )


def test_evppi_var_python_facade_matches_discrete_reference() -> None:
    result = estimation_module.evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=_evppi_spec(),
    )
    assert result.prior_functional == pytest.approx(1.25)
    assert result.expected_posterior_functional == pytest.approx(1.0)
    assert result.absolute_reduction == pytest.approx(0.25)
    assert result.relative_reduction == pytest.approx(0.2)
    assert result.provenance.backend == "rust"
    assert result.provenance.estimator_id == "discrete_conditioning"
    assert result.diagnostics.monte_carlo_standard_error is None
    assert (
        result.model_dump_json()
        == EstimationVarianceResult.model_validate_json(
            result.model_dump_json()
        ).model_dump_json()
    )


def test_evsi_var_python_facade_preserves_negative_raw_estimate() -> None:
    result = estimation_module.evsi_var(
        [0.0, 1.0, 2.0, 3.0],
        [1.5, 1.5],
        specification=_evsi_spec(),
    )
    assert result.raw_reduction == pytest.approx(-0.25)
    assert result.absolute_reduction == 0.0
    assert result.relative_reduction == 0.0
    payload = json.loads(result.model_dump_json())
    assert payload["negative_estimate_policy"] == "retain_raw_clip_reported"


def test_estimation_facade_translates_native_dimension_and_input_errors() -> None:
    with pytest.raises(DimensionMismatchError, match="conditioning-group count"):
        _ = estimation_module.evppi_var(
            [0.0, 1.0], ["only-one"], specification=_evppi_spec()
        )
    with pytest.raises(InputError, match="posterior variances must be nonnegative"):
        _ = estimation_module.evsi_var([0.0, 1.0], [-0.1], specification=_evsi_spec())


def test_estimation_facade_rejects_method_mismatch_before_native_dispatch() -> None:
    with pytest.raises(InputError, match="matching EstimationVarianceSpec"):
        _ = estimation_module.evppi_var(
            [0.0, 1.0], ["a", "b"], specification=_evsi_spec()
        )


def test_seeded_bootstrap_assurance_is_deterministic_and_typed() -> None:
    specification = _evppi_spec().model_copy(
        update={
            "estimator": _assurance(
                "discrete_conditioning",
                bootstrap_replicates=128,
                convergence_threshold=1.0,
            )
        }
    )
    first = estimation_module.evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=specification,
    )
    second = estimation_module.evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=specification,
    )
    assert first.model_dump_json() == second.model_dump_json()
    assert first.diagnostics.bootstrap_replicates == 128
    assert first.diagnostics.monte_carlo_standard_error is not None
    assert first.diagnostics.confidence_interval is not None
    assert first.diagnostics.converged is True
    assert first.diagnostics.diagnostic_codes == ()


def test_assurance_contract_rejects_one_bootstrap_replicate() -> None:
    with pytest.raises(
        ValueError, match="bootstrap_replicates must be zero or at least two"
    ):
        _ = _assurance("discrete_conditioning", bootstrap_replicates=1)


def test_runtime_rejects_vector_targets_pending_scientific_review() -> None:
    specification = _evppi_spec().model_copy(
        update={
            "target": EstimationTargetSpec(
                target_id="joint",
                shape="vector",
                component_units=("count", "count"),
                covariance_functional="trace",
            )
        }
    )
    with pytest.raises(InputError, match="scalar variance targets only"):
        _ = estimation_module.evppi_var(
            [0.0, 1.0], ["a", "b"], specification=specification
        )


def _native_payload() -> dict[str, object]:
    return {
        "prior_variance": 1.0,
        "expected_posterior_variance": 0.5,
        "raw_reduction": 0.5,
        "absolute_reduction": 0.5,
        "relative_reduction": 0.5,
        "prior_sample_count": 4,
        "posterior_evaluation_count": 2,
        "bootstrap_replicates": 2,
        "monte_carlo_standard_error": 0.2,
        "confidence_interval": [0.1, 0.9],
        "converged": False,
        "kernel_version": "1.0.0",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"prior_variance": True}, "prior_variance.*not numeric"),
        ({"prior_sample_count": True}, "prior_sample_count.*not an integer"),
        ({"monte_carlo_standard_error": "bad"}, "not numeric or null"),
        ({"confidence_interval": "bad"}, "not a pair"),
        ({"confidence_interval": [0.1]}, "not a pair"),
        ({"confidence_interval": [False, 0.2]}, "not numeric"),
        ({"relative_reduction": "bad"}, "relative_reduction.*not numeric"),
        ({"relative_reduction": True}, "relative_reduction.*not numeric"),
        ({"kernel_version": 1}, "kernel version is not text"),
        ({"converged": 1}, "convergence field is not boolean"),
    ],
)
def test_native_result_boundary_rejects_malformed_fields(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _ = estimation_module._result_from_native(
            _evppi_spec(),
            {**_native_payload(), **updates},
        )


def test_native_result_boundary_handles_null_interval_and_nonconvergence() -> None:
    result = estimation_module._result_from_native(
        _evppi_spec(),
        {
            **_native_payload(),
            "confidence_interval": None,
            "relative_reduction": None,
            "prior_variance": 0.0,
            "expected_posterior_variance": 0.0,
            "raw_reduction": 0.0,
            "absolute_reduction": 0.0,
        },
    )
    assert result.relative_reduction is None
    assert result.diagnostics.confidence_interval is None
    assert result.diagnostics.diagnostic_codes == ("convergence_threshold_not_met",)
