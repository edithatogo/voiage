"""Contracts and numerical evidence for estimation-focused variance VOI."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationTargetSpec,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
    SamplingModelSpec,
)
from voiage.methods.estimation import (
    ESTIMATION_VARIANCE_METHODS,
    estimation_variance_method,
)


def test_estimation_variance_registry_separates_decision_and_sensitivity_methods() -> (
    None
):
    assert set(ESTIMATION_VARIANCE_METHODS) == {"evppi_var", "evsi_var"}
    for method_id, descriptor in ESTIMATION_VARIANCE_METHODS.items():
        assert descriptor["method_id"] == method_id
        assert descriptor["family"] == "estimation-focused-variance-voi"
        assert descriptor["estimand_kind"] == "variance_reduction"
        assert descriptor["decision_focused"] is False
        assert descriptor["sensitivity_index"] is False
        assert descriptor["estimator_uncertainty"] is False
        assert method_id not in {"evppi", "evsi"}


@pytest.mark.parametrize(
    ("name", "method_id"),
    [
        ("evppi_var", "evppi_var"),
        ("evppi_variance", "evppi_var"),
        ("evsi_var", "evsi_var"),
        ("evsi_variance", "evsi_var"),
    ],
)
def test_estimation_variance_aliases_are_explicit(name: str, method_id: str) -> None:
    assert estimation_variance_method(name)["method_id"] == method_id


@pytest.mark.parametrize("name", ["evppi", "evsi", "sobol", "posterior_variance"])
def test_decision_and_adjacent_names_are_not_estimation_variance_aliases(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="estimation-focused variance"):
        _ = estimation_variance_method(name)


def test_scalar_evppi_variance_contract_is_explicit() -> None:
    specification = EstimationVarianceSpec(
        method_id="evppi_var",
        target=EstimationTargetSpec(
            target_id="total_cost",
            shape="scalar",
            component_units=("NZD_2026",),
            covariance_functional="variance",
        ),
        prior_model_id="prior-v1",
        conditioning=ConditioningSpec(
            parameter_subset=("unit_cost",),
            sigma_field="sigma(theta_unit_cost)",
            averaging_convention="prior_predictive",
        ),
        estimator=EstimatorAssuranceSpec(
            estimator_id="exact_discrete_conditioning",
            seed=20260731,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-10,
        ),
    )
    assert specification.sampling_model is None
    assert specification.zero_variance_policy == "absolute_zero_relative_null"


def test_vector_target_requires_declared_covariance_scalarization() -> None:
    target = EstimationTargetSpec(
        target_id="cost_and_health",
        shape="vector",
        component_units=("NZD_2026", "QALY"),
        covariance_functional="weighted_quadratic",
        functional_weights=(0.001, 1.0),
    )
    assert target.functional_weights == (0.001, 1.0)


@pytest.mark.parametrize(
    "target",
    [
        {
            "target_id": "bad",
            "shape": "scalar",
            "component_units": ("USD", "QALY"),
            "covariance_functional": "variance",
        },
        {
            "target_id": "bad",
            "shape": "vector",
            "component_units": ("USD", "QALY"),
            "covariance_functional": "variance",
        },
        {
            "target_id": "bad",
            "shape": "vector",
            "component_units": ("USD", "QALY"),
            "covariance_functional": "weighted_quadratic",
            "functional_weights": (1.0,),
        },
    ],
)
def test_invalid_target_functional_contracts_fail_closed(
    target: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _ = EstimationTargetSpec.model_validate(target)


def test_evsi_variance_requires_sampling_model_and_forbids_conditioning_subset() -> (
    None
):
    with pytest.raises(ValidationError, match="sampling_model"):
        _ = EstimationVarianceSpec(
            method_id="evsi_var",
            target=EstimationTargetSpec(
                target_id="prevalence",
                shape="scalar",
                component_units=("proportion",),
                covariance_functional="variance",
            ),
            prior_model_id="beta-prior-v1",
            estimator=EstimatorAssuranceSpec(
                estimator_id="enumerated_prior_predictive",
                seed=1,
            ),
        )

    specification = EstimationVarianceSpec(
        method_id="evsi_var",
        target=EstimationTargetSpec(
            target_id="prevalence",
            shape="scalar",
            component_units=("proportion",),
            covariance_functional="variance",
        ),
        prior_model_id="beta-prior-v1",
        sampling_model=SamplingModelSpec(
            design_id="binomial-n20",
            likelihood_id="binomial",
            conditioning_sigma_field="sigma(Y,design)",
            averaging_convention="prior_predictive",
        ),
        estimator=EstimatorAssuranceSpec(
            estimator_id="enumerated_prior_predictive",
            seed=1,
        ),
    )
    assert specification.conditioning is None
