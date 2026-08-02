"""Native-backed Python façade tests for estimation variance VOI."""

# pyright: reportAny=false

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationTargetSpec,
    EstimationVarianceResult,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
    SamplingModelSpec,
    TruthKnownAssuranceSpec,
)
from voiage.exceptions import DimensionMismatchError, InputError
import voiage.methods.estimation as estimation_module

ROOT = Path(__file__).resolve().parents[1]
ESTIMATION_FIXTURES = ROOT / "specs" / "estimation-variance" / "v1" / "fixtures"


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
        [0.5, 0.5],
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
        _ = estimation_module.evsi_var(
            [0.0, 1.0], [-0.1], [1.0], specification=_evsi_spec()
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([1.0], "predictive-probability count must match"),
        ([0.8, 0.8], "must sum to one"),
        ([1.1, -0.1], "must be nonnegative"),
        ([float("nan"), 0.0], "invalid domain collection"),
    ],
)
def test_evsi_var_rejects_pathological_predictive_probabilities(
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises((DimensionMismatchError, InputError), match=message):
        _ = estimation_module.evsi_var(
            [0.0, 1.0],
            [0.1, 0.2],
            probabilities,
            specification=_evsi_spec(),
        )


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


def test_evsi_bootstrap_accepts_zero_tolerance_for_six_outcomes() -> None:
    specification = _evsi_spec().model_copy(
        update={
            "estimator": EstimatorAssuranceSpec(
                estimator_id="posterior_variance_aggregation",
                seed=17,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                bootstrap_replicates=2,
                convergence_threshold=1.0,
            )
        }
    )

    result = estimation_module.evsi_var(
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        specification=specification,
    )

    assert result.diagnostics.bootstrap_replicates == 2
    assert result.diagnostics.monte_carlo_standard_error is not None


def test_assurance_contract_rejects_one_bootstrap_replicate() -> None:
    with pytest.raises(
        ValueError, match="bootstrap_replicates must be zero or at least two"
    ):
        _ = _assurance("discrete_conditioning", bootstrap_replicates=1)


@pytest.mark.parametrize(
    ("method_id", "functional"),
    [
        ("evppi_var", "trace"),
        ("evppi_var", "determinant"),
        ("evppi_var", "weighted_quadratic"),
        ("evsi_var", "trace"),
        ("evsi_var", "determinant"),
        ("evsi_var", "weighted_quadratic"),
    ],
)
def test_runtime_rejects_all_vector_targets_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    method_id: str,
    functional: str,
) -> None:
    def unexpected_native_dispatch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("vector request reached the native scalar kernel")

    monkeypatch.setattr(
        estimation_module, "compute_evppi_variance", unexpected_native_dispatch
    )
    monkeypatch.setattr(
        estimation_module, "compute_evsi_variance", unexpected_native_dispatch
    )
    target = EstimationTargetSpec(
        target_id="joint",
        shape="vector",
        component_units=("count", "count"),
        covariance_functional=functional,
        functional_weights=(1.0, 1.0) if functional == "weighted_quadratic" else None,
    )
    specification = (
        _evppi_spec() if method_id == "evppi_var" else _evsi_spec()
    ).model_copy(update={"target": target})

    def request_vector_result() -> object:
        if method_id == "evppi_var":
            return estimation_module.evppi_var(
                [0.0, 1.0], ["a", "b"], specification=specification
            )
        return estimation_module.evsi_var(
            [0.0, 1.0],
            [0.1, 0.2],
            [0.5, 0.5],
            specification=specification,
        )

    with pytest.raises(InputError, match="scalar variance targets only") as captured:
        _ = request_vector_result()
    assert captured.value.diagnostic_code == "unsupported_estimation_target"


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
            input_digest="b" * 64,
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
        input_digest="b" * 64,
    )
    assert result.relative_reduction is None
    assert result.diagnostics.confidence_interval is None
    assert result.diagnostics.diagnostic_codes == ("convergence_threshold_not_met",)


def test_evsi_var_uses_unequal_prior_predictive_probabilities() -> None:
    result = estimation_module.evsi_var(
        [-2.0, 0.0, 2.0],
        [0.0, 3.0],
        [0.9, 0.1],
        specification=_evsi_spec(),
    )
    assert result.prior_functional == pytest.approx(8.0 / 3.0)
    assert result.expected_posterior_functional == pytest.approx(0.3)
    assert result.raw_reduction == pytest.approx((8.0 / 3.0) - 0.3)


def test_replay_digest_binds_actual_estimation_inputs() -> None:
    first = estimation_module.evsi_var(
        [0.0, 1.0], [0.1, 0.2], [0.25, 0.75], specification=_evsi_spec()
    )
    changed_value = estimation_module.evsi_var(
        [0.0, 2.0], [0.1, 0.2], [0.25, 0.75], specification=_evsi_spec()
    )
    changed_weight = estimation_module.evsi_var(
        [0.0, 1.0], [0.1, 0.2], [0.75, 0.25], specification=_evsi_spec()
    )

    assert (
        first.provenance.specification_digest
        == changed_value.provenance.specification_digest
    )
    assert first.provenance.input_digest != changed_value.provenance.input_digest
    assert first.provenance.input_digest != changed_weight.provenance.input_digest


def test_runtime_binding_covers_scientific_and_solver_request() -> None:
    specification = _evsi_spec().model_copy(
        update={
            "estimator": EstimatorAssuranceSpec(
                estimator_id="nested-posterior-variance",
                seed=17,
                estimator_design="coupled_nested_monte_carlo",
                outer_replicates=128,
                inner_replicates=64,
                coupling_id="common-random-numbers-v1",
                solver_id="rust-nested-variance-v1",
            )
        }
    )
    result = estimation_module.evsi_var(
        [0.0, 1.0],
        [0.1, 0.2],
        [0.25, 0.75],
        specification=specification,
    )

    binding = result.runtime_binding
    assert binding.method_id == "evsi_var"
    assert binding.target_id == specification.target.target_id
    assert binding.design_id == specification.sampling_model.design_id
    assert binding.likelihood_id == specification.sampling_model.likelihood_id
    assert binding.estimator_design == "coupled_nested_monte_carlo"
    assert binding.outer_replicates == 128
    assert binding.inner_replicates == 64
    assert binding.coupling_id == "common-random-numbers-v1"
    assert binding.solver_id == "rust-nested-variance-v1"
    assert len(result.provenance.runtime_binding_digest) == 64
    assert len(result.provenance.runtime_request_digest) == 64


@pytest.mark.parametrize(
    "updates",
    [
        {"estimator_design": "outer_monte_carlo"},
        {
            "estimator_design": "nested_monte_carlo",
            "outer_replicates": 20,
        },
        {
            "estimator_design": "coupled_nested_monte_carlo",
            "outer_replicates": 20,
            "inner_replicates": 10,
        },
        {"estimator_design": "exact", "outer_replicates": 20},
    ],
)
def test_estimator_design_contract_rejects_incomplete_or_contradictory_requests(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        EstimatorAssuranceSpec(estimator_id="invalid", seed=0, **updates)


def test_truth_known_outer_replicates_report_bias_rmse_coverage_and_calibration() -> (
    None
):
    result = estimation_module.evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=_evppi_spec().model_copy(
            update={
                "estimator": _assurance(
                    "discrete_conditioning", convergence_threshold=1.0
                )
            }
        ),
        truth_assurance=TruthKnownAssuranceSpec(
            true_reduction=0.25,
            replicate_reductions=(0.2, 0.25, 0.3, 0.35),
            confidence_intervals=(
                (0.1, 0.3),
                (0.2, 0.3),
                (0.2, 0.4),
                (0.3, 0.4),
            ),
            dependence_structure="independent_outer",
            replay_artifact="fixtures/evppi-truth-known-outer-v1.json",
        ),
    )

    assurance = result.truth_known_assurance
    assert assurance is not None
    assert assurance.bias == pytest.approx(0.025)
    assert assurance.rmse == pytest.approx(0.00375**0.5)
    assert assurance.standard_error == pytest.approx(0.03227486121839514)
    assert assurance.empirical_coverage == pytest.approx(0.75)
    assert assurance.calibration_error == pytest.approx(0.2)
    assert assurance.converged is True
    assert assurance.replicate_unit == "complete_outer_dataset"
    assert len(assurance.replay_digest) == 64


def test_truth_known_portable_fixture_replays_through_python_and_rust() -> None:
    assurance_payload = json.loads(
        (ESTIMATION_FIXTURES / "evppi-var-truth-known.assurance.json").read_text()
    )
    expected = json.loads(
        (ESTIMATION_FIXTURES / "evppi-var-truth-known.result.json").read_text()
    )
    result = estimation_module.evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=_evppi_spec(),
        truth_assurance=TruthKnownAssuranceSpec.model_validate_json(
            json.dumps(assurance_payload)
        ),
    )
    assert result.truth_known_assurance is not None
    assert result.truth_known_assurance.model_dump(mode="json") == expected


def test_nested_and_coupled_assurance_require_dependence_preserving_outer_units() -> (
    None
):
    nested_spec = _evsi_spec().model_copy(
        update={
            "estimator": EstimatorAssuranceSpec(
                estimator_id="nested",
                seed=1,
                estimator_design="nested_monte_carlo",
                outer_replicates=20,
                inner_replicates=10,
            )
        }
    )
    independent = TruthKnownAssuranceSpec(
        true_reduction=0.1,
        replicate_reductions=(0.08, 0.12),
        dependence_structure="independent_outer",
        replay_artifact="fixture.json",
    )
    with pytest.raises(InputError, match="dependence disagrees"):
        estimation_module.evsi_var(
            [0.0, 1.0],
            [0.1, 0.2],
            [0.5, 0.5],
            specification=nested_spec,
            truth_assurance=independent,
        )

    accepted = estimation_module.evsi_var(
        [0.0, 1.0],
        [0.1, 0.2],
        [0.5, 0.5],
        specification=nested_spec,
        truth_assurance=independent.model_copy(
            update={"dependence_structure": "nested_shared_outer"}
        ),
    )
    assert accepted.truth_known_assurance is not None
    assert accepted.truth_known_assurance.dependence_structure == "nested_shared_outer"
