"""Contracts and numerical evidence for estimation-focused variance VOI."""

# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationRuntimeBinding,
    EstimationTargetSpec,
    EstimationVarianceDiagnostics,
    EstimationVarianceProvenance,
    EstimationVarianceResult,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
    SamplingModelSpec,
)
from voiage.methods.estimation import (
    ESTIMATION_VARIANCE_METHODS,
    estimation_variance_method,
)

ROOT = Path(__file__).resolve().parents[1]
ESTIMATION_SPEC_ROOT = ROOT / "specs" / "estimation-variance" / "v1"


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


@pytest.mark.parametrize(
    "target",
    [
        {
            "target_id": "bad",
            "shape": "scalar",
            "component_units": ("USD",),
            "covariance_functional": "trace",
        },
        {
            "target_id": "bad",
            "shape": "scalar",
            "component_units": ("USD",),
            "covariance_functional": "variance",
            "functional_weights": (1.0,),
        },
        {
            "target_id": "bad",
            "shape": "vector",
            "component_units": ("USD",),
            "covariance_functional": "trace",
        },
        {
            "target_id": "bad",
            "shape": "vector",
            "component_units": ("USD", "QALY"),
            "covariance_functional": "weighted_quadratic",
        },
        {
            "target_id": "bad",
            "shape": "vector",
            "component_units": ("USD", "QALY"),
            "covariance_functional": "trace",
            "functional_weights": (1.0, 1.0),
        },
    ],
)
def test_remaining_target_shape_and_functional_combinations_fail_closed(
    target: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _ = EstimationTargetSpec.model_validate(target)


def test_conditioning_parameter_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        _ = ConditioningSpec(
            parameter_subset=("risk", "risk"),
            sigma_field="sigma_risk",
            averaging_convention="prior_predictive",
        )


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


@pytest.mark.parametrize(
    "averaging_convention", ["posterior_predictive", "empirical_reference"]
)
def test_evsi_variance_rejects_non_prior_predictive_averaging(
    averaging_convention: str,
) -> None:
    with pytest.raises(ValidationError, match="requires prior_predictive"):
        _ = EstimationVarianceSpec(
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
                averaging_convention=averaging_convention,  # type: ignore[arg-type]
            ),
            estimator=EstimatorAssuranceSpec(
                estimator_id="enumerated_prior_predictive",
                seed=1,
            ),
        )


def test_method_specific_information_contracts_reject_opposite_inputs() -> None:
    target = EstimationTargetSpec(
        target_id="prevalence",
        shape="scalar",
        component_units=("proportion",),
        covariance_functional="variance",
    )
    assurance = EstimatorAssuranceSpec(estimator_id="enumerated", seed=1)
    sampling = SamplingModelSpec(
        design_id="binomial-n20",
        likelihood_id="binomial",
        conditioning_sigma_field="sigma_y",
        averaging_convention="prior_predictive",
    )
    conditioning = ConditioningSpec(
        parameter_subset=("prevalence",),
        sigma_field="sigma_prevalence",
        averaging_convention="prior_predictive",
    )
    with pytest.raises(ValidationError, match="evppi_var requires conditioning"):
        _ = EstimationVarianceSpec(
            method_id="evppi_var",
            target=target,
            prior_model_id="prior",
            estimator=assurance,
        )
    with pytest.raises(ValidationError, match="does not accept sampling_model"):
        _ = EstimationVarianceSpec(
            method_id="evppi_var",
            target=target,
            prior_model_id="prior",
            conditioning=conditioning,
            sampling_model=sampling,
            estimator=assurance,
        )
    with pytest.raises(ValidationError, match="does not accept conditioning"):
        _ = EstimationVarianceSpec(
            method_id="evsi_var",
            target=target,
            prior_model_id="prior",
            conditioning=conditioning,
            sampling_model=sampling,
            estimator=assurance,
        )


def test_diagnostics_reject_inverted_confidence_interval() -> None:
    with pytest.raises(ValidationError, match="lower bound"):
        _ = EstimationVarianceDiagnostics(
            prior_sample_count=2,
            posterior_evaluation_count=1,
            confidence_interval=(0.2, 0.1),
            converged=False,
        )


def test_result_contract_preserves_raw_negative_estimate_and_zero_variance_policy() -> (
    None
):
    runtime_binding = EstimationRuntimeBinding(
        method_id="evsi_var",
        target_id="constant",
        target_shape="scalar",
        component_units=("count",),
        covariance_functional="variance",
        prior_model_id="prior-v1",
        design_id="study-v1",
        likelihood_id="likelihood-v1",
        sampling_conditioning_sigma_field="sigma(Y)",
        averaging_convention="prior_predictive",
        estimator_design="exact",
        solver_id="rust-estimation-variance-v1",
    )
    result = EstimationVarianceResult(
        method_id="evsi_var",
        target=EstimationTargetSpec(
            target_id="constant",
            shape="scalar",
            component_units=("count",),
            covariance_functional="variance",
        ),
        runtime_binding=runtime_binding,
        prior_covariance=((0.0,),),
        expected_posterior_covariance=((0.1,),),
        prior_functional=0.0,
        expected_posterior_functional=0.1,
        raw_reduction=-0.1,
        absolute_reduction=0.0,
        relative_reduction=None,
        functional_units="count^2",
        diagnostics=EstimationVarianceDiagnostics(
            prior_sample_count=20,
            posterior_evaluation_count=4,
            monte_carlo_standard_error=0.03,
            converged=False,
            diagnostic_codes=("negative_finite_sample_estimate",),
        ),
        provenance=EstimationVarianceProvenance(
            backend="rust",
            kernel_version="1.0.0",
            estimator_id="posterior_variance_aggregation",
            seed=7,
            specification_digest="a" * 64,
            input_digest="b" * 64,
            runtime_binding_digest=runtime_binding.content_digest(),
            runtime_request_digest="c" * 64,
        ),
    )
    assert result.absolute_reduction == 0.0
    assert result.relative_reduction is None
    assert result.raw_reduction == -0.1


def test_versioned_estimation_variance_schemas_and_fixtures_validate() -> None:
    input_schema = json.loads(
        (ESTIMATION_SPEC_ROOT / "input.schema.json").read_text(encoding="utf-8")
    )
    result_schema = json.loads(
        (ESTIMATION_SPEC_ROOT / "result.schema.json").read_text(encoding="utf-8")
    )
    runtime_data_schema = json.loads(
        (ESTIMATION_SPEC_ROOT / "runtime-data.schema.json").read_text(encoding="utf-8")
    )
    assert result_schema["x-semantic-validation-required"] is True
    assert (
        result_schema["x-semantic-validator"]
        == "voiage.contracts.estimation.EstimationVarianceResult"
    )
    manifest = json.loads(
        (ESTIMATION_SPEC_ROOT / "fixtures" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["method_family"] == "estimation-focused-variance-voi"
    for fixture in manifest["fixtures"]:
        input_payload = json.loads(
            (ESTIMATION_SPEC_ROOT / "fixtures" / fixture["input"]).read_text(
                encoding="utf-8"
            )
        )
        result_payload = json.loads(
            (ESTIMATION_SPEC_ROOT / "fixtures" / fixture["result"]).read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(input_schema).validate(input_payload)
        Draft202012Validator(result_schema).validate(result_payload)
        assert (
            EstimationVarianceSpec.model_validate_json(
                json.dumps(input_payload)
            ).model_dump(mode="json")
            == input_payload
        )
        assert (
            EstimationVarianceResult.model_validate_json(
                json.dumps(result_payload)
            ).model_dump(mode="json")
            == result_payload
        )

        non_prior_spec = {
            **input_payload,
            "method_id": "evsi_var",
            "conditioning": None,
            "sampling_model": {
                "design_id": "study",
                "likelihood_id": "likelihood",
                "conditioning_sigma_field": "sigma_y",
                "averaging_convention": "posterior_predictive",
            },
        }
        assert list(Draft202012Validator(input_schema).iter_errors(non_prior_spec))

        forged_result = {
            **result_payload,
            "prior_covariance": [[999.0]],
            "expected_posterior_covariance": [[888.0]],
        }
        assert not list(Draft202012Validator(result_schema).iter_errors(forged_result))
        with pytest.raises(ValidationError, match="must equal its functional"):
            _ = EstimationVarianceResult.model_validate_json(json.dumps(forged_result))
        for structural_forgery in (
            {**result_payload, "prior_covariance": [[-1.0]]},
            {**result_payload, "prior_covariance": [[1.25, 0.0], [0.0, 1.0]]},
            {**result_payload, "functional_units": "bananas"},
        ):
            assert list(
                Draft202012Validator(result_schema).iter_errors(structural_forgery)
            )

    Draft202012Validator(runtime_data_schema).validate(
        {
            "prior_target_samples": [-2.0, 0.0, 2.0],
            "posterior_variances": [0.0, 3.0],
            "predictive_probabilities": [0.9, 0.1],
        }
    )
