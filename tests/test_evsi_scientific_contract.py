"""Scientific reference tests for study-specific EVSI calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods import sample_information as sample_information_module
from voiage.methods.sample_information import (
    evsi,
    normal_normal_two_arm_evsi,
)
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray

PRIOR_MEAN = 0.060
PRIOR_SD = 0.030
OUTCOME_SD = 1.0
WILLINGNESS_TO_PAY = 50_000.0
INCREMENTAL_COST = 3_000.0
CORRELATED_ARM_SD = 0.030
CORRELATED_ARM_RHO = 0.90
CORRELATED_PRIOR_MEAN = np.array([0.0, PRIOR_MEAN])
CORRELATED_PRIOR_COVARIANCE = CORRELATED_ARM_SD**2 * np.array(
    [[1.0, CORRELATED_ARM_RHO], [CORRELATED_ARM_RHO, 1.0]]
)


def _prior(sample_count: int) -> ParameterSet:
    quantiles = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    rng = np.random.default_rng(20260724)
    # A fixed generated sample is sufficient for the nested-Monte-Carlo
    # recovery test; the analytical reference below remains independent.
    arm_prior_sd = PRIOR_SD / np.sqrt(2.0)
    control = rng.normal(0.0, arm_prior_sd, sample_count)
    programme = rng.normal(PRIOR_MEAN, arm_prior_sd, sample_count)
    return ParameterSet.from_numpy_or_dict(
        {
            "mean_control": control,
            "mean_programme": programme,
            "sd_outcome": np.full(sample_count, OUTCOME_SD),
            "quantile_marker": quantiles,
        }
    )


def _model(parameters: ParameterSet) -> ValueArray:
    effect = (
        parameters.parameters["mean_programme"] - parameters.parameters["mean_control"]
    )
    net_benefit = np.column_stack(
        [
            np.zeros(parameters.n_samples),
            WILLINGNESS_TO_PAY * effect - INCREMENTAL_COST,
        ]
    )
    return ValueArray.from_numpy(net_benefit, ["Control", "Programme"])


def _design(total_sample_size: int = 200) -> TrialDesign:
    per_arm = total_sample_size // 2
    return TrialDesign(
        [
            DecisionOption("Control", per_arm),
            DecisionOption("Programme", per_arm),
        ]
    )


def test_public_normal_normal_evsi_matches_the_analytical_reference() -> None:
    """The stable study-specific path must recover a prespecified benchmark."""
    result = normal_normal_two_arm_evsi(
        prior_mean=PRIOR_MEAN,
        prior_standard_deviation=PRIOR_SD,
        outcome_standard_deviation=OUTCOME_SD,
        total_sample_size=200,
        net_benefit_slope=WILLINGNESS_TO_PAY,
        net_benefit_intercept=-INCREMENTAL_COST,
    )

    assert result == pytest.approx(124.1793655206238, abs=1e-5)


@pytest.mark.parametrize(
    "sample_size",
    [True, 200.0, -2, 3, 2**32],
)
def test_public_normal_normal_evsi_rejects_invalid_sample_size(
    sample_size: object,
) -> None:
    """The analytical facade must not leak Python-to-Rust conversion errors."""
    with pytest.raises(InputError) as captured:
        normal_normal_two_arm_evsi(
            prior_mean=PRIOR_MEAN,
            prior_standard_deviation=PRIOR_SD,
            outcome_standard_deviation=OUTCOME_SD,
            total_sample_size=sample_size,  # type: ignore[arg-type]
            net_benefit_slope=WILLINGNESS_TO_PAY,
            net_benefit_intercept=-INCREMENTAL_COST,
        )

    assert captured.value.diagnostic_code == "invalid_total_sample_size"


def test_builtin_two_loop_rejects_unrepresentable_arm_sample_size() -> None:
    """Observation-scale conversion must remain inside the package error boundary."""
    design = TrialDesign(
        [
            DecisionOption("Control", 10**400),
            DecisionOption("Programme", 100),
        ]
    )

    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            _prior(200),
            design,
            method="two_loop",
            n_outer_loops=2,
            n_inner_loops=2,
            seed=7,
        )

    assert captured.value.diagnostic_code == "invalid_sample_size"


def test_custom_two_loop_returns_negative_estimate_without_truncation() -> None:
    """A known negative custom estimate is returned with coherence guidance."""
    prior = ParameterSet.from_numpy_or_dict({"theta": np.array([0.0, 1.0, 0.0, 1.0])})

    def model(parameters: ParameterSet) -> ValueArray:
        theta = np.asarray(parameters.parameters["theta"], dtype=float)
        return ValueArray.from_numpy(
            np.column_stack([np.zeros(theta.size), theta]),
            ["Current", "Alternative"],
        )

    def simulator(
        _truth: dict[str, float],
        _design: TrialDesign,
        _rng: np.random.Generator,
    ) -> object:
        return None

    def incoherent_posterior(
        _prior: ParameterSet,
        _data: object,
        _design: TrialDesign,
        n_draws: int,
        _rng: np.random.Generator,
    ) -> ParameterSet:
        return ParameterSet.from_numpy_or_dict({"theta": np.zeros(n_draws)})

    with pytest.warns(
        RuntimeWarning,
        match=r"evsi_negative_monte_carlo_estimate.*coherent model",
    ):
        result = evsi(
            model,
            prior,
            TrialDesign([DecisionOption("Observation", 1)]),
            method="two_loop",
            n_outer_loops=2,
            n_inner_loops=2,
            seed=13,
            trial_simulator=simulator,
            posterior_sampler=incoherent_posterior,
        )

    assert result == -0.5


def _correlated_prior(sample_count: int = 20_000) -> ParameterSet:
    """Return a fixed joint prior with material positive arm correlation."""
    draws = np.random.default_rng(1729).multivariate_normal(
        CORRELATED_PRIOR_MEAN,
        CORRELATED_PRIOR_COVARIANCE,
        size=sample_count,
    )
    return ParameterSet.from_numpy_or_dict(
        {
            "mean_control": draws[:, 0],
            "mean_programme": draws[:, 1],
            "sd_outcome": np.full(sample_count, OUTCOME_SD),
        }
    )


def _joint_trial_simulator(
    true_parameters: dict[str, float],
    trial_design: TrialDesign,
    rng: np.random.Generator,
) -> object:
    """Simulate sufficient statistics from the declared two-arm likelihood."""
    return {
        arm.name: rng.normal(
            true_parameters[f"mean_{arm.name.lower()}"],
            OUTCOME_SD / np.sqrt(arm.sample_size),
        )
        for arm in trial_design.arms
    }


def _joint_posterior_sampler(
    _prior_samples: ParameterSet,
    trial_data: object,
    trial_design: TrialDesign,
    n_draws: int,
    rng: np.random.Generator,
) -> ParameterSet:
    """Draw from the declared correlated multivariate-normal posterior."""
    assert isinstance(trial_data, dict)
    sample_means = np.array([float(trial_data[arm.name]) for arm in trial_design.arms])
    likelihood_precision = np.diag(
        [arm.sample_size / OUTCOME_SD**2 for arm in trial_design.arms]
    )
    prior_precision = np.linalg.inv(CORRELATED_PRIOR_COVARIANCE)
    posterior_covariance = np.linalg.inv(prior_precision + likelihood_precision)
    posterior_mean = posterior_covariance @ (
        prior_precision @ CORRELATED_PRIOR_MEAN + likelihood_precision @ sample_means
    )
    draws = rng.multivariate_normal(
        posterior_mean,
        posterior_covariance,
        size=n_draws,
    )
    return ParameterSet.from_numpy_or_dict(
        {
            "mean_control": draws[:, 0],
            "mean_programme": draws[:, 1],
            "sd_outcome": np.full(n_draws, OUTCOME_SD),
        }
    )


def _correlated_analytical_reference() -> float:
    """Return the independent conjugate reference for the callback test."""
    likelihood_precision = np.diag([100.0, 100.0])
    posterior_covariance = np.linalg.inv(
        np.linalg.inv(CORRELATED_PRIOR_COVARIANCE) + likelihood_precision
    )
    difference = np.array([-1.0, 1.0])
    preposterior_variance = float(
        difference @ (CORRELATED_PRIOR_COVARIANCE - posterior_covariance) @ difference
    )
    net_benefit_sd = WILLINGNESS_TO_PAY * np.sqrt(preposterior_variance)
    return float(net_benefit_sd / np.sqrt(2.0 * np.pi))


def test_custom_two_loop_requires_both_declared_callbacks() -> None:
    """A custom study model must never run with only half of its contract."""
    with pytest.raises(
        InputError,
        match="require both `trial_simulator` and `posterior_sampler`",
    ):
        evsi(
            _model,
            _prior(200),
            _design(),
            method="two_loop",
            n_outer_loops=2,
            n_inner_loops=2,
            trial_simulator=_joint_trial_simulator,
        )


def test_builtin_two_loop_preserves_a_correlated_joint_prior() -> None:
    """The v2 normal model must update all correlated parameters jointly."""
    estimates = np.array(
        [
            evsi(
                _model,
                _correlated_prior(),
                _design(),
                method="two_loop",
                n_outer_loops=2_000,
                n_inner_loops=200,
                seed=seed,
            )
            for seed in (17, 29, 43)
        ]
    )
    reference = _correlated_analytical_reference()
    standard_error = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    difference = np.array([-1.0, 1.0])
    difference_variance = float(difference @ CORRELATED_PRIOR_COVARIANCE @ difference)
    evpi_reference = (
        WILLINGNESS_TO_PAY * math.sqrt(difference_variance) / math.sqrt(2.0 * math.pi)
    )

    assert float(np.mean(estimates)) == pytest.approx(
        reference,
        abs=max(2.5, 4.0 * standard_error),
    )
    assert np.all(estimates >= 0.0)
    assert np.all(estimates <= evpi_reference + 4.0 * max(standard_error, 1.0))


def test_two_loop_callbacks_preserve_a_correlated_joint_prior() -> None:
    """Declared joint callbacks recover a correlated conjugate reference."""
    estimates = np.array(
        [
            evsi(
                _model,
                _correlated_prior(),
                _design(),
                method="two_loop",
                n_outer_loops=500,
                n_inner_loops=300,
                seed=seed,
                trial_simulator=_joint_trial_simulator,
                posterior_sampler=_joint_posterior_sampler,
            )
            for seed in (17, 29, 43, 71)
        ]
    )
    reference = _correlated_analytical_reference()
    standard_error = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))

    assert reference == pytest.approx(25.27504830697345, abs=1e-9)
    assert float(np.mean(estimates)) == pytest.approx(
        reference,
        abs=max(8.0, 4.0 * standard_error),
    )


def test_joint_posterior_matches_complete_three_parameter_reference() -> None:
    """Posterior moments include a correlated parameter not observed by the study."""
    mean = np.array([0.1, 0.4, -0.2])
    covariance = np.array(
        [
            [0.09, 0.045, 0.024],
            [0.045, 0.16, -0.032],
            [0.024, -0.032, 0.25],
        ]
    )
    prior_draws = np.random.default_rng(20260724).multivariate_normal(
        mean,
        covariance,
        size=30_000,
    )
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": prior_draws[:, 0],
            "mean_programme": prior_draws[:, 1],
            "correlated_context": prior_draws[:, 2],
            "sd_outcome": np.full(prior_draws.shape[0], 1.2),
        }
    )
    design = TrialDesign(
        [DecisionOption("Control", 60), DecisionOption("Programme", 120)]
    )
    observed = {"Control": 0.17, "Programme": 0.49}

    fitted_mean = np.mean(prior_draws, axis=0)
    fitted_covariance = np.cov(prior_draws, rowvar=False, ddof=1)
    observation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    observation_covariance = np.diag([1.2**2 / 60.0, 1.2**2 / 120.0])
    gain = (
        np.linalg.solve(
            observation_matrix @ fitted_covariance @ observation_matrix.T
            + observation_covariance,
            observation_matrix @ fitted_covariance,
        )
    ).T
    expected_mean = fitted_mean + gain @ (
        np.array([0.17, 0.49]) - observation_matrix @ fitted_mean
    )
    identity_minus_gain = np.eye(3) - gain @ observation_matrix
    expected_covariance = (
        identity_minus_gain @ fitted_covariance @ identity_minus_gain.T
        + gain @ observation_covariance @ gain.T
    )

    posterior = sample_information_module._joint_normal_posterior(
        prior,
        observed,
        design,
        200_000,
        np.random.default_rng(1905),
    )
    posterior_matrix = np.column_stack(
        [
            posterior.parameters["mean_control"],
            posterior.parameters["mean_programme"],
            posterior.parameters["correlated_context"],
        ]
    )

    np.testing.assert_allclose(
        np.mean(posterior_matrix, axis=0), expected_mean, atol=0.0015
    )
    np.testing.assert_allclose(
        np.cov(posterior_matrix, rowvar=False, ddof=1),
        expected_covariance,
        rtol=0.012,
        atol=3e-4,
    )
    assert expected_mean[2] != pytest.approx(fitted_mean[2])
    assert expected_covariance[2, 0] != pytest.approx(fitted_covariance[2, 0])
    assert expected_covariance[2, 1] != pytest.approx(fitted_covariance[2, 1])


@pytest.mark.parametrize("correlation", [-0.999, 0.999])
def test_joint_posterior_handles_near_unit_correlations(correlation: float) -> None:
    """Well-defined near-singular priors remain finite at either correlation sign."""
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    draws = np.random.default_rng(771).multivariate_normal(
        [0.0, 0.1],
        covariance,
        size=12_000,
    )
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": draws[:, 0],
            "mean_programme": draws[:, 1],
            "sd_outcome": np.ones(draws.shape[0]),
        }
    )
    posterior = sample_information_module._joint_normal_posterior(
        prior,
        {"Control": -0.1, "Programme": 0.2},
        _design(),
        5_000,
        np.random.default_rng(772),
    )
    assert all(np.all(np.isfinite(values)) for values in posterior.parameters.values())


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_joint_posterior_handles_representable_extreme_scales(scale: float) -> None:
    """Numerical guards are relative to the problem scale rather than one."""
    base = np.random.default_rng(991).multivariate_normal(
        [0.0, 0.2],
        [[1.0, -0.25], [-0.25, 2.0]],
        size=2_000,
    )
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": base[:, 0] * scale,
            "mean_programme": base[:, 1] * scale,
            "sd_outcome": np.full(base.shape[0], scale),
        }
    )
    posterior = sample_information_module._joint_normal_posterior(
        prior,
        {"Control": -0.1 * scale, "Programme": 0.25 * scale},
        _design(),
        400,
        np.random.default_rng(992),
    )
    assert all(np.all(np.isfinite(values)) for values in posterior.parameters.values())


def test_builtin_two_loop_rejects_unidentifiable_fitted_covariance() -> None:
    """A PSA matrix with no covariance degrees of freedom fails closed."""
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.array([0.0, 0.1, 0.2]),
            "mean_programme": np.array([0.1, 0.2, 0.3]),
            "context": np.array([1.0, 1.5, 2.0]),
            "sd_outcome": np.ones(3),
        }
    )
    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            prior,
            _design(),
            n_outer_loops=2,
            n_inner_loops=4,
            seed=90,
        )
    assert captured.value.diagnostic_code == "rank_deficient_prior"


def test_gaussian_sampler_preserves_tail_and_fourth_moment() -> None:
    """Posterior integration uses genuine Gaussian draws for nonlinear models."""
    draws = sample_information_module._positive_semidefinite_draws(
        np.array([0.0]),
        np.array([[1.0]]),
        250_000,
        np.random.default_rng(271828),
    )[:, 0]
    assert float(np.mean(draws**4)) == pytest.approx(3.0, abs=0.06)
    assert float(np.mean(draws > 1.96)) == pytest.approx(0.025, abs=0.0015)
    single = sample_information_module._positive_semidefinite_draws(
        np.array([0.0]),
        np.array([[1.0]]),
        1,
        np.random.default_rng(271828),
    )
    assert single[0, 0] != 0.0


def test_builtin_two_loop_rejects_multidimensional_parameter_variables() -> None:
    """The built-in model accepts one scalar value per parameter and PSA row."""
    import xarray as xr

    prior = ParameterSet(
        xr.Dataset(
            {
                "mean_a": (
                    ("n_samples", "component"),
                    np.arange(20, dtype=float).reshape(10, 2),
                ),
                "sd_outcome": (("n_samples",), np.ones(10)),
            },
            coords={"n_samples": np.arange(10), "component": [0, 1]},
        )
    )
    with pytest.raises(InputError) as captured:
        evsi(
            lambda parameters: ValueArray.from_numpy(
                np.zeros((parameters.n_samples, 2)),
                ["A", "B"],
            ),
            prior,
            TrialDesign([DecisionOption("A", 10)]),
            n_outer_loops=2,
            n_inner_loops=4,
            seed=3,
        )
    assert captured.value.diagnostic_code == "non_scalar_parameter"


def test_builtin_two_loop_rejects_non_finite_aggregate_results() -> None:
    """Finite model entries cannot overflow into a non-finite public result."""
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_a": np.linspace(-1.0, 1.0, 100),
            "sd_outcome": np.ones(100),
        }
    )

    def extreme_model(parameters: ParameterSet) -> ValueArray:
        return ValueArray.from_numpy(
            np.full((parameters.n_samples, 2), 1e308),
            ["A", "B"],
        )

    with pytest.raises(InputError) as captured:
        evsi(
            extreme_model,
            prior,
            TrialDesign([DecisionOption("A", 10)]),
            n_outer_loops=2,
            n_inner_loops=4,
            seed=4,
        )
    assert captured.value.diagnostic_code == "non_finite_reduction"


def test_prior_covariance_linear_algebra_failures_are_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public built-in execution never exposes a raw NumPy linear-algebra error."""
    monkeypatch.setattr(
        np.linalg,
        "eigvalsh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            np.linalg.LinAlgError("forced failure")
        ),
    )
    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            _prior(50),
            _design(),
            n_outer_loops=2,
            n_inner_loops=4,
            seed=5,
        )
    assert captured.value.diagnostic_code == "covariance_decomposition_failed"


@pytest.mark.parametrize(
    "values",
    [
        np.array([1e-15, -1e-13]),
        np.array([1e-15, 5e-13]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1e-300, 1e-300]),
        np.array([1e300, 1e300]),
    ],
)
def test_builtin_two_loop_rejects_invalid_or_nonconstant_outcome_sd(
    values: np.ndarray,
) -> None:
    """Every fixed-SD draw must be equal, finite, positive, and representable."""
    count = len(values)
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.linspace(-0.1, 0.1, count),
            "mean_programme": np.linspace(0.0, 0.2, count),
            "sd_outcome": values,
        }
    )
    with pytest.raises(InputError):
        evsi(
            _model,
            prior,
            _design(),
            method="two_loop",
            n_outer_loops=2,
            n_inner_loops=2,
            seed=7,
        )


def test_builtin_two_loop_rejects_arm_parameter_name_collisions() -> None:
    """Distinct arm labels cannot silently observe the same latent parameter."""
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_a_b": np.linspace(-0.2, 0.2, 20),
            "sd_outcome": np.ones(20),
        }
    )
    design = TrialDesign([DecisionOption("A B", 10), DecisionOption("A_B", 10)])
    with pytest.raises(InputError) as captured:
        evsi(
            lambda parameters: ValueArray.from_numpy(
                np.column_stack(
                    [np.zeros(parameters.n_samples), parameters.parameters["mean_a_b"]]
                ),
                ["A", "B"],
            ),
            prior,
            design,
            n_outer_loops=2,
            n_inner_loops=2,
            seed=8,
        )
    assert captured.value.diagnostic_code == "arm_parameter_name_collision"


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n_outer_loops", True),
        ("n_outer_loops", 1.5),
        ("n_inner_loops", True),
        ("n_inner_loops", 1.5),
    ],
)
def test_two_loop_rejects_non_integer_loop_counts(argument: str, value: object) -> None:
    """Loop counts fail through InputError instead of raw range errors."""
    kwargs: dict[str, object] = {
        "n_outer_loops": 2,
        "n_inner_loops": 2,
    }
    kwargs[argument] = value
    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            _prior(20),
            _design(),
            method="two_loop",
            seed=11,
            **kwargs,
        )
    assert captured.value.diagnostic_code == "invalid_loop_count"


def test_builtin_two_loop_uses_one_coherent_fitted_gaussian_prior() -> None:
    """A non-Gaussian PSA sample is not mixed with a Gaussian posterior target."""
    sample_count = 20_000
    theta = np.tile(np.array([-1.0, 1.0]), sample_count // 2)
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_a": theta,
            "sd_outcome": np.full(sample_count, math.sqrt(0.4)),
        }
    )
    design = TrialDesign([DecisionOption("A", 1)])

    def nonlinear_model(parameters: ParameterSet) -> ValueArray:
        uncertain = parameters.parameters["mean_a"]
        return ValueArray.from_numpy(
            np.column_stack([np.zeros(parameters.n_samples), uncertain**2 - 0.8]),
            ["Current", "Alternative"],
        )

    fitted_variance = float(np.var(theta, ddof=1))
    posterior_variance = fitted_variance * 0.4 / (fitted_variance + 0.4)
    preposterior_variance = fitted_variance - posterior_variance
    threshold = 0.8 - posterior_variance
    standard_threshold = math.sqrt(threshold / preposterior_variance)
    upper_tail = 0.5 * math.erfc(standard_threshold / math.sqrt(2.0))
    density = math.exp(-(standard_threshold**2) / 2.0) / math.sqrt(2.0 * math.pi)
    expected_sample_value = 2.0 * (
        preposterior_variance * (standard_threshold * density + upper_tail)
        - threshold * upper_tail
    )
    current_value = max(0.0, fitted_variance - 0.8)
    coherent_reference = expected_sample_value - current_value

    estimate = evsi(
        nonlinear_model,
        prior,
        design,
        method="two_loop",
        n_outer_loops=4_000,
        n_inner_loops=1_000,
        seed=314159,
    )

    assert coherent_reference == pytest.approx(0.216, abs=0.002)
    assert estimate == pytest.approx(coherent_reference, abs=0.025)


@pytest.mark.parametrize("method", ["regression", "efficient", "moment_based"])
def test_estimators_without_a_likelihood_are_explicitly_nonstable(method: str) -> None:
    """Compatibility estimators must not silently imply scientific validation."""
    with pytest.warns(
        FutureWarning,
        match="does not expose a complete, validated study-model contract",
    ):
        evsi(
            _model,
            _prior(200),
            _design(),
            method=method,
            n_outer_loops=20,
            n_inner_loops=20,
        )
