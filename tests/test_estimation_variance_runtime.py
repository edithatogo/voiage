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
from voiage.methods.estimation import evppi_var, evsi_var


def _target() -> EstimationTargetSpec:
    return EstimationTargetSpec(
        target_id="net_cases",
        shape="scalar",
        component_units=("count",),
        covariance_functional="variance",
    )


def _assurance(estimator_id: str) -> EstimatorAssuranceSpec:
    return EstimatorAssuranceSpec(estimator_id=estimator_id, seed=17)


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
    result = evppi_var(
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
    assert result.model_dump_json() == EstimationVarianceResult.model_validate_json(
        result.model_dump_json()
    ).model_dump_json()


def test_evsi_var_python_facade_preserves_negative_raw_estimate() -> None:
    result = evsi_var(
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
        _ = evppi_var([0.0, 1.0], ["only-one"], specification=_evppi_spec())
    with pytest.raises(InputError, match="posterior variances must be nonnegative"):
        _ = evsi_var([0.0, 1.0], [-0.1], specification=_evsi_spec())


def test_estimation_facade_rejects_method_mismatch_before_native_dispatch() -> None:
    with pytest.raises(InputError, match="matching EstimationVarianceSpec"):
        _ = evppi_var([0.0, 1.0], ["a", "b"], specification=_evsi_spec())
