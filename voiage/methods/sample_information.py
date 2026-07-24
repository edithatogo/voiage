"""
Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from collections.abc import Callable
import importlib.util
from typing import Any
import warnings

import numpy as np

from voiage import _runtime
from voiage.exceptions import (
    raise_backend_not_available_error,
    raise_input_error,
)
from voiage.schema import ParameterSet, TrialDesign, ValueArray

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

EconomicModelFunctionType = Callable[[ParameterSet], ValueArray]
MetamodelName = str
TrialSimulatorType = Callable[
    [dict[str, float], TrialDesign, np.random.Generator],
    object,
]
PosteriorSamplerType = Callable[
    [ParameterSet, object, TrialDesign, int, np.random.Generator],
    ParameterSet,
]

_MAX_ANALYTICAL_SAMPLE_SIZE = 2**32 - 1


def _finite_float_argument(value: object, name: str) -> float:
    """Convert one public analytical scalar through the package error boundary."""
    if isinstance(value, bool):
        raise_input_error(
            f"`{name}` must be a finite real number, not a Boolean.",
            diagnostic_code="invalid_scalar_input",
        )
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise_input_error(
            f"`{name}` must be a finite real number.",
            diagnostic_code="invalid_scalar_input",
        )
        raise AssertionError("unreachable") from error
    if not np.isfinite(converted):
        raise_input_error(
            f"`{name}` must be finite.",
            diagnostic_code="non_finite_value",
        )
    return converted


def normal_normal_two_arm_evsi(
    *,
    prior_mean: float,
    prior_standard_deviation: float,
    outcome_standard_deviation: float,
    total_sample_size: int,
    net_benefit_slope: float,
    net_benefit_intercept: float,
) -> float:
    """Calculate EVSI for a declared equal-allocation normal study.

    The uncertain incremental effect has a normal prior. The proposed two-arm
    study has a common known outcome standard deviation and equal allocation.
    Incremental net benefit is linear in the effect.
    """
    if (
        isinstance(total_sample_size, bool)
        or not isinstance(total_sample_size, int)
        or total_sample_size <= 0
        or total_sample_size % 2 != 0
        or total_sample_size > _MAX_ANALYTICAL_SAMPLE_SIZE
    ):
        raise_input_error(
            "`total_sample_size` must be a positive even integer within the "
            "supported unsigned 32-bit range.",
            diagnostic_code="invalid_total_sample_size",
        )
    return _runtime.compute_normal_normal_two_arm_evsi(
        _finite_float_argument(prior_mean, "prior_mean"),
        _finite_float_argument(
            prior_standard_deviation,
            "prior_standard_deviation",
        ),
        _finite_float_argument(
            outcome_standard_deviation,
            "outcome_standard_deviation",
        ),
        total_sample_size,
        _finite_float_argument(net_benefit_slope, "net_benefit_slope"),
        _finite_float_argument(net_benefit_intercept, "net_benefit_intercept"),
    )


def _parameter_matrix(psa_prior: ParameterSet) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return PSA parameters as a finite 2D NumPy array."""
    if not psa_prior.parameters:
        raise_input_error("`psa_prior` must contain at least one parameter.")

    columns = [
        np.asarray(values, dtype=float) for values in psa_prior.parameters.values()
    ]
    if any(column.ndim != 1 for column in columns):
        raise_input_error(
            "EVSI requires one scalar value per parameter and PSA row.",
            diagnostic_code="non_scalar_parameter",
        )
    matrix = np.column_stack(columns)
    if matrix.ndim != 2 or matrix.shape[0] != psa_prior.n_samples:
        raise_input_error("PSA parameters must form a 2D sample-by-parameter matrix.")
    if not np.all(np.isfinite(matrix)):
        raise_input_error(
            "PSA parameters must contain only finite values.",
            diagnostic_code="non_finite_value",
        )
    return matrix


def _validate_net_benefits(
    net_benefits: np.ndarray[Any, Any],
    expected_samples: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Validate model output as finite sample-by-strategy net benefits."""
    values = np.asarray(net_benefits, dtype=float)
    if values.ndim != 2:
        raise_input_error("Model function must return 2D net-benefit values.")
    if values.shape[0] != expected_samples:
        raise_input_error("Model output sample count must match `psa_prior`.")
    if values.shape[1] < 1:
        raise_input_error("Model output must contain at least one strategy.")
    if not np.all(np.isfinite(values)):
        raise_input_error(
            "Model output must contain only finite net-benefit values.",
            diagnostic_code="non_finite_value",
        )
    return values


def _finite_strategy_means(
    net_benefits: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Reduce net benefits without allowing finite-input overflow to escape."""
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.asarray(np.mean(net_benefits, axis=0), dtype=float)
    if not np.all(np.isfinite(means)):
        raise_input_error(
            "Net-benefit aggregation produced a non-finite strategy mean.",
            diagnostic_code="non_finite_reduction",
        )
    return means


def _trial_information_fraction(
    trial_design: TrialDesign,
    prior_sample_size: int,
) -> float:
    """Map a trial design to a conservative preposterior information fraction."""
    total_trial_size = float(sum(max(0, arm.sample_size) for arm in trial_design.arms))
    if total_trial_size <= 0:
        return 0.0
    denominator = total_trial_size + max(1.0, float(prior_sample_size))
    return float(np.clip(total_trial_size / denominator, 0.0, 1.0))


def _clip_expected_max_to_bounds(
    expected_max: float,
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """Keep approximate EVSI estimators within current-information and EVPI bounds."""
    prior_means = np.mean(nb_prior_values, axis=0)
    current = float(np.max(prior_means))
    perfect = float(np.mean(np.max(nb_prior_values, axis=1)))
    return float(np.clip(expected_max, current, max(current, perfect)))


def _preposterior_expected_max(
    predicted_nb: np.ndarray[Any, np.dtype[np.float64]],
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]],
    information_fraction: float,
) -> float:
    """Estimate expected maximum net benefit after sampling from a surrogate."""
    prior_means = np.mean(nb_prior_values, axis=0)
    conditional_means = prior_means + information_fraction * (
        predicted_nb - prior_means
    )
    expected_max = float(np.mean(np.max(conditional_means, axis=1)))
    return _clip_expected_max_to_bounds(expected_max, nb_prior_values)


def _fit_strategy_metamodel(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    metamodel: MetamodelName,
) -> Any:
    """Fit one strategy-level metamodel for efficient EVSI."""
    if metamodel == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
    elif metamodel in {"random_forest", "rf"}:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=max(1, x.shape[0] // 100),
            random_state=0,
        )
    else:
        raise_backend_not_available_error(
            f"Efficient EVSI metamodel '{metamodel}' is not recognized."
        )
    model.fit(x, y)
    return model


def _evsi_efficient_regression(
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    metamodel: MetamodelName = "linear",
) -> float:
    """Efficient PSA-sample regression EVSI approximation.

    This follows the same preposterior target as PSA-sample regression methods:
    estimate conditional expected net benefit by strategy, then average the
    strategy maximum over the PSA sample. The implementation uses trial sample
    size as a conservative information-fraction proxy, avoiding inner-loop
    economic model runs.
    """
    x = _parameter_matrix(psa_prior)
    if metamodel == "linear":
        # The deterministic linear contract is now Rust-owned.  Keep the
        # public function's historical return shape by returning the native
        # expected-sample value; the dispatcher subtracts current information
        # below and applies population scaling uniformly.
        trial_sample_size = sum(max(0, arm.sample_size) for arm in trial_design.arms)
        if trial_sample_size <= 0:
            return float(np.max(np.mean(nb_prior_values, axis=0)))
        result = _runtime.compute_evsi_efficient_linear(
            nb_prior_values.tolist(),
            x.tolist(),
            trial_sample_size,
        )

        current_value = float(np.max(np.mean(nb_prior_values, axis=0)))
        try:
            if (
                result.get("estimator") != "efficient_linear"
                or result.get("contract_version") != 1
                or result.get("sample_count") != psa_prior.n_samples
                or result.get("strategy_count") != nb_prior_values.shape[1]
                or result.get("parameter_count") != x.shape[1]
            ):
                raise ValueError("native efficient-linear metadata mismatch")  # noqa: TRY301
            native_current = float(result["expected_current_value"])
            expected_sample = float(result["expected_sample_value"])
            expected_perfect = float(result["expected_perfect_information"])
            information_fraction = float(result["information_fraction"])
            native_evsi = float(result["evsi"])
            values = (
                native_current,
                expected_sample,
                expected_perfect,
                information_fraction,
                native_evsi,
            )
            if not all(np.isfinite(value) for value in values):
                raise ValueError("native efficient-linear result is non-finite")  # noqa: TRY301
            if not 0.0 <= information_fraction <= 1.0:
                raise ValueError(  # noqa: TRY301
                    "native efficient-linear information fraction is invalid"
                )
            if not np.isclose(native_current, current_value, rtol=1e-12, atol=1e-12):
                raise ValueError("native efficient-linear current value mismatch")  # noqa: TRY301
            if not np.isclose(
                native_evsi, expected_sample - native_current, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("native efficient-linear EVSI mismatch")  # noqa: TRY301
            if expected_sample < native_current or expected_sample > max(
                native_current, expected_perfect
            ):
                raise ValueError(  # noqa: TRY301
                    "native efficient-linear sample value is out of bounds"
                )
        except (KeyError, TypeError, ValueError) as error:
            raise_input_error(
                "Native efficient-linear EVSI returned an invalid result envelope."
            )
            raise AssertionError("unreachable") from error
        else:
            return expected_sample

    if not SKLEARN_AVAILABLE:
        raise_backend_not_available_error("Efficient EVSI requires scikit-learn.")

    predictions = np.empty_like(nb_prior_values, dtype=float)
    for strategy_idx in range(nb_prior_values.shape[1]):
        model = _fit_strategy_metamodel(
            x,
            nb_prior_values[:, strategy_idx],
            metamodel,
        )
        predictions[:, strategy_idx] = np.asarray(model.predict(x), dtype=float)

    information_fraction = _trial_information_fraction(
        trial_design,
        psa_prior.n_samples,
    )
    return _preposterior_expected_max(
        predictions,
        nb_prior_values,
        information_fraction,
    )


def _evsi_moment_based(
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
) -> float:
    """Moment-based EVSI approximation using the native kernel when available."""
    x = _parameter_matrix(psa_prior)
    trial_sample_size = sum(max(0, arm.sample_size) for arm in trial_design.arms)
    if trial_sample_size <= 0:
        return float(np.max(np.mean(nb_prior_values, axis=0)))
    result = _runtime.compute_evsi_moment_based(
        nb_prior_values.tolist(),
        x.tolist(),
        trial_sample_size,
    )

    current_value = float(np.max(np.mean(nb_prior_values, axis=0)))
    try:
        if (
            result.get("estimator") != "moment_based"
            or result.get("contract_version") != 1
            or result.get("sample_count") != psa_prior.n_samples
            or result.get("strategy_count") != nb_prior_values.shape[1]
            or result.get("parameter_count") != x.shape[1]
        ):
            raise ValueError("native moment-based metadata mismatch")  # noqa: TRY301
        native_current = float(result["expected_current_value"])
        expected_sample = float(result["expected_sample_value"])
        expected_perfect = float(result["expected_perfect_information"])
        information_fraction = float(result["information_fraction"])
        native_evsi = float(result["evsi"])
        values = (
            native_current,
            expected_sample,
            expected_perfect,
            information_fraction,
            native_evsi,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("native moment-based result is non-finite")  # noqa: TRY301
        if not 0.0 <= information_fraction <= 1.0:
            raise ValueError(  # noqa: TRY301
                "native moment-based information fraction is invalid"
            )
        if not np.isclose(native_current, current_value, rtol=1e-12, atol=1e-12):
            raise ValueError("native moment-based current value mismatch")  # noqa: TRY301
        if not np.isclose(
            native_evsi, expected_sample - native_current, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("native moment-based EVSI mismatch")  # noqa: TRY301
        if expected_sample < native_current or expected_sample > max(
            native_current, expected_perfect
        ):
            raise ValueError(  # noqa: TRY301
                "native moment-based sample value is out of bounds"
            )
    except (KeyError, TypeError, ValueError) as error:
        raise_input_error(
            "Native moment-based EVSI returned an invalid result envelope."
        )
        raise AssertionError("unreachable") from error
    else:
        return expected_sample


def _simulate_trial_data(
    true_parameters: dict[str, Any],
    trial_design: TrialDesign,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Simulate trial data based on true parameters."""
    data = {}
    for arm in trial_design.arms:
        mean = true_parameters[f"mean_{arm.name.lower().replace(' ', '_')}"]
        std_dev = true_parameters["sd_outcome"]
        if rng is None:
            data[arm.name] = np.random.normal(mean, std_dev, arm.sample_size)
        else:
            data[arm.name] = rng.normal(mean, std_dev, arm.sample_size)
    return data


def _arm_parameter_name(arm_name: str) -> str:
    """Map one trial-arm label to its declared mean-parameter name."""
    return f"mean_{arm_name.lower().replace(' ', '_')}"


def _known_outcome_standard_deviation(prior_samples: ParameterSet) -> float:
    """Return the one finite positive outcome SD required by the built-in model."""
    if "sd_outcome" not in prior_samples.parameters:
        raise_input_error(
            "Built-in normal two-loop EVSI requires a fixed `sd_outcome` parameter."
        )
    values = np.asarray(prior_samples.parameters["sd_outcome"], dtype=float)
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or not np.all(values > 0.0)
        or not np.all(values == values[0])
    ):
        raise_input_error(
            "Built-in normal two-loop EVSI requires one finite, positive known "
            "`sd_outcome`."
        )
    outcome_sd = float(values[0])
    if outcome_sd < np.sqrt(np.finfo(float).tiny) or outcome_sd > np.sqrt(
        np.finfo(float).max
    ):
        raise_input_error(
            "Built-in normal two-loop EVSI requires `sd_outcome` whose squared "
            "observation variance is representable as a finite positive float.",
            diagnostic_code="invalid_observation_scale",
        )
    observation_variance = outcome_sd * outcome_sd
    if not np.isfinite(observation_variance) or observation_variance <= 0.0:
        raise_input_error(
            "Built-in normal two-loop EVSI requires `sd_outcome` whose squared "
            "observation variance is representable as a finite positive float.",
            diagnostic_code="invalid_observation_scale",
        )
    return outcome_sd


def _validate_builtin_normal_contract(
    prior_samples: ParameterSet,
    trial_design: TrialDesign,
) -> None:
    """Validate the complete built-in two-loop study-model contract."""
    _known_outcome_standard_deviation(prior_samples)
    arm_parameters = [_arm_parameter_name(arm.name) for arm in trial_design.arms]
    if len(set(arm_parameters)) != len(arm_parameters):
        raise_input_error(
            "Trial-arm names must map to unique `mean_<arm>` parameter names.",
            diagnostic_code="arm_parameter_name_collision",
        )
    for arm in trial_design.arms:
        try:
            sample_size = float(arm.sample_size)
        except (TypeError, ValueError, OverflowError) as error:
            raise_input_error(
                "Built-in normal two-loop EVSI requires each arm sample size "
                "to be representable as a finite positive float.",
                diagnostic_code="invalid_sample_size",
            )
            raise AssertionError("unreachable") from error
        if not np.isfinite(sample_size) or sample_size <= 0.0:
            raise_input_error(
                "Built-in normal two-loop EVSI requires each arm sample size "
                "to be representable as a finite positive float.",
                diagnostic_code="invalid_sample_size",
            )
    missing = [name for name in arm_parameters if name not in prior_samples.parameters]
    if missing:
        raise_input_error(
            "Built-in normal two-loop EVSI requires one parameter named "
            f"`mean_<arm>` for every trial arm; missing {', '.join(missing)}."
        )


def _joint_normal_prior_parameters(
    prior_samples: ParameterSet,
) -> tuple[
    list[str],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Fit the one joint Gaussian prior used by the built-in EVSI contract."""
    uncertain_names = [
        name for name in prior_samples.parameters if name != "sd_outcome"
    ]
    if not uncertain_names:
        raise_input_error(
            "Built-in normal two-loop EVSI requires uncertain model parameters."
        )
    if prior_samples.n_samples <= len(uncertain_names):
        raise_input_error(
            "Built-in normal two-loop EVSI requires more prior draws than "
            "uncertain parameters so the fitted covariance is identifiable.",
            diagnostic_code="rank_deficient_prior",
        )
    prior_columns = [
        np.asarray(prior_samples.parameters[name], dtype=float)
        for name in uncertain_names
    ]
    if any(column.ndim != 1 for column in prior_columns):
        raise_input_error(
            "Built-in normal two-loop EVSI requires one scalar value per "
            "uncertain parameter and PSA row.",
            diagnostic_code="non_scalar_parameter",
        )
    prior_matrix = np.column_stack(prior_columns)
    if not np.all(np.isfinite(prior_matrix)):
        raise_input_error(
            "Built-in normal two-loop EVSI requires finite prior draws.",
            diagnostic_code="non_finite_value",
        )
    prior_mean = np.asarray(np.mean(prior_matrix, axis=0), dtype=float)
    prior_covariance = np.atleast_2d(
        np.asarray(np.cov(prior_matrix, rowvar=False, ddof=1), dtype=float)
    )
    if not np.all(np.isfinite(prior_covariance)):
        raise_input_error(
            "Built-in normal two-loop EVSI requires a finite joint prior covariance."
        )
    try:
        eigenvalues = np.linalg.eigvalsh((prior_covariance + prior_covariance.T) / 2.0)
    except np.linalg.LinAlgError as error:
        raise_input_error(
            "Joint prior covariance decomposition failed.",
            diagnostic_code="covariance_decomposition_failed",
        )
        raise AssertionError("unreachable") from error
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = (
        100.0 * np.finfo(float).eps * max(1, len(uncertain_names)) * spectral_scale
    )
    if float(np.min(eigenvalues)) < -tolerance:
        raise_input_error(
            "Built-in normal two-loop EVSI requires a positive-semidefinite "
            "joint prior covariance.",
            diagnostic_code="invalid_prior_covariance",
        )
    return uncertain_names, prior_mean, prior_covariance


def _positive_semidefinite_draws(
    mean: np.ndarray[Any, np.dtype[np.float64]],
    covariance: np.ndarray[Any, np.dtype[np.float64]],
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Draw from a possibly singular Gaussian using an audited eigensystem."""
    symmetric = (covariance + covariance.T) / 2.0
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as error:
        raise_input_error(
            "Gaussian covariance decomposition failed.",
            diagnostic_code="covariance_decomposition_failed",
        )
        raise AssertionError("unreachable") from error
    scale = max(
        float(np.max(np.abs(eigenvalues))),
        float(np.max(np.abs(covariance))),
        np.finfo(float).tiny,
    )
    tolerance = 100.0 * np.finfo(float).eps * max(1, covariance.shape[0]) * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise_input_error(
            "Gaussian covariance is not positive semidefinite at its numerical scale.",
            diagnostic_code="invalid_posterior_covariance",
        )
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    standard = rng.standard_normal((n_draws, mean.size))
    return np.asarray(mean + standard @ root.T, dtype=float)


def _normal_observation_contract(
    prior_samples: ParameterSet,
    trial_design: TrialDesign,
) -> tuple[
    list[str],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return the fitted Gaussian prior and its known-variance observation model."""
    _validate_builtin_normal_contract(prior_samples, trial_design)
    uncertain_names, prior_mean, prior_covariance = _joint_normal_prior_parameters(
        prior_samples
    )
    parameter_index = {name: index for index, name in enumerate(uncertain_names)}
    design_matrix = np.zeros((len(trial_design.arms), len(uncertain_names)))
    for row, arm in enumerate(trial_design.arms):
        design_matrix[row, parameter_index[_arm_parameter_name(arm.name)]] = 1.0
    outcome_sd = _known_outcome_standard_deviation(prior_samples)
    observation_covariance = np.diag(
        [outcome_sd**2 / float(arm.sample_size) for arm in trial_design.arms]
    )
    if not np.all(np.isfinite(observation_covariance)) or not np.all(
        np.diag(observation_covariance) > 0.0
    ):
        raise_input_error(
            "Trial observation variances must be finite and positive.",
            diagnostic_code="invalid_observation_scale",
        )
    return (
        uncertain_names,
        prior_mean,
        prior_covariance,
        design_matrix,
        observation_covariance,
    )


def _joint_normal_posterior_from_contract(
    uncertain_names: list[str],
    prior_mean: np.ndarray[Any, np.dtype[np.float64]],
    prior_covariance: np.ndarray[Any, np.dtype[np.float64]],
    design_matrix: np.ndarray[Any, np.dtype[np.float64]],
    observation_covariance: np.ndarray[Any, np.dtype[np.float64]],
    observation: np.ndarray[Any, np.dtype[np.float64]],
    outcome_sd: float,
    n_draws: int,
    rng: np.random.Generator,
) -> ParameterSet:
    """Sample a posterior from one prevalidated joint Gaussian contract."""
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws <= 0:
        raise_input_error("Posterior draw count must be positive.")
    if not np.all(np.isfinite(observation)):
        raise_input_error(
            "Built-in normal trial data must contain one finite mean per arm."
        )

    predictive_covariance = (
        design_matrix @ prior_covariance @ design_matrix.T + observation_covariance
    )
    try:
        predictive_factor = np.linalg.cholesky(predictive_covariance)
        solved = np.linalg.solve(
            predictive_factor,
            design_matrix @ prior_covariance,
        )
        gain = np.linalg.solve(predictive_factor.T, solved).T
    except np.linalg.LinAlgError as error:
        raise_input_error(
            "Built-in normal EVSI predictive covariance is singular or not "
            "positive definite.",
            diagnostic_code="singular_predictive_covariance",
        )
        raise AssertionError("unreachable") from error
    posterior_mean = prior_mean + gain @ (observation - design_matrix @ prior_mean)
    identity_minus_gain = np.eye(prior_covariance.shape[0]) - gain @ design_matrix
    posterior_covariance = (
        identity_minus_gain @ prior_covariance @ identity_minus_gain.T
        + gain @ observation_covariance @ gain.T
    )
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2.0
    draws = np.atleast_2d(
        _positive_semidefinite_draws(
            posterior_mean,
            posterior_covariance,
            n_draws,
            rng,
        )
    )
    if draws.shape != (n_draws, len(uncertain_names)):
        draws = draws.reshape(n_draws, len(uncertain_names))
    posterior_parameters = {
        name: draws[:, index] for index, name in enumerate(uncertain_names)
    }
    posterior_parameters["sd_outcome"] = np.full(n_draws, outcome_sd)
    return ParameterSet.from_numpy_or_dict(posterior_parameters)


def _joint_normal_posterior(
    prior_samples: ParameterSet,
    sample_means: dict[str, float],
    trial_design: TrialDesign,
    n_draws: int,
    rng: np.random.Generator,
) -> ParameterSet:
    """Sample the joint Gaussian posterior for the built-in normal study model."""
    outcome_sd = _known_outcome_standard_deviation(prior_samples)
    (
        uncertain_names,
        prior_mean,
        prior_covariance,
        design_matrix,
        observation_covariance,
    ) = _normal_observation_contract(prior_samples, trial_design)
    observation = np.asarray(
        [sample_means[arm.name] for arm in trial_design.arms],
        dtype=float,
    )
    return _joint_normal_posterior_from_contract(
        uncertain_names,
        prior_mean,
        prior_covariance,
        design_matrix,
        observation_covariance,
        observation,
        outcome_sd,
        n_draws,
        rng,
    )


def _bayesian_update(
    prior_samples: ParameterSet,
    trial_data: dict[str, np.ndarray[Any, Any]],
    trial_design: TrialDesign,
    rng: np.random.Generator | None = None,
    n_draws: int | None = None,
) -> ParameterSet:
    """Update a joint Gaussian prior under a known-variance arm-mean likelihood."""
    draw_count = n_draws if n_draws is not None else prior_samples.n_samples
    generator = rng if rng is not None else np.random.default_rng()
    try:
        sample_means = {
            arm.name: float(np.mean(np.asarray(trial_data[arm.name], dtype=float)))
            for arm in trial_design.arms
        }
    except (KeyError, TypeError, ValueError) as error:
        raise_input_error(
            "Normal posterior updating requires finite trial data for every arm."
        )
        raise AssertionError("unreachable") from error
    if not all(np.isfinite(value) for value in sample_means.values()):
        raise_input_error(
            "Normal posterior updating requires finite trial data for every arm."
        )
    return _joint_normal_posterior(
        prior_samples,
        sample_means,
        trial_design,
        draw_count,
        generator,
    )


def _builtin_normal_trial_simulator(
    true_parameters: dict[str, float],
    trial_design: TrialDesign,
    rng: np.random.Generator,
) -> object:
    """Simulate arm sample means for the built-in known-variance normal model."""
    outcome_sd = float(true_parameters["sd_outcome"])
    return {
        arm.name: float(
            rng.normal(
                true_parameters[_arm_parameter_name(arm.name)],
                outcome_sd / np.sqrt(float(arm.sample_size)),
            )
        )
        for arm in trial_design.arms
    }


def _builtin_joint_normal_posterior_sampler(
    prior_samples: ParameterSet,
    trial_data: object,
    trial_design: TrialDesign,
    n_draws: int,
    rng: np.random.Generator,
) -> ParameterSet:
    """Adapt built-in trial means to the joint Gaussian posterior sampler."""
    if not isinstance(trial_data, dict):
        raise_input_error("Built-in normal trial data must be a mapping of arm means.")
    try:
        sample_means = {
            arm.name: float(trial_data[arm.name]) for arm in trial_design.arms
        }
    except (KeyError, TypeError, ValueError) as error:
        raise_input_error(
            "Built-in normal trial data must contain one finite mean per arm."
        )
        raise AssertionError("unreachable") from error
    if not all(np.isfinite(value) for value in sample_means.values()):
        raise_input_error(
            "Built-in normal trial data must contain one finite mean per arm."
        )
    return _joint_normal_posterior(
        prior_samples,
        sample_means,
        trial_design,
        n_draws,
        rng,
    )


def _parameter_set_from_joint_normal_draws(
    uncertain_names: list[str],
    draws: np.ndarray[Any, np.dtype[np.float64]],
    outcome_sd: float,
) -> ParameterSet:
    """Build a parameter set from joint Gaussian draws and the fixed study SD."""
    parameters = {name: draws[:, index] for index, name in enumerate(uncertain_names)}
    parameters["sd_outcome"] = np.full(draws.shape[0], outcome_sd)
    return ParameterSet.from_numpy_or_dict(parameters)


def _evsi_builtin_joint_normal(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_outer_loops: int,
    n_inner_loops: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Estimate EVSI coherently under one fitted joint Gaussian prior.

    The empirical PSA draws are used only to fit the prior moments. Current
    value, predictive trial outcomes, and posterior values are then all
    integrated under that same fitted Gaussian model.
    """
    (
        uncertain_names,
        prior_mean,
        prior_covariance,
        design_matrix,
        observation_covariance,
    ) = _normal_observation_contract(psa_prior, trial_design)
    outcome_sd = _known_outcome_standard_deviation(psa_prior)

    current_draw_count = max(n_inner_loops, psa_prior.n_samples)
    prior_draws = _positive_semidefinite_draws(
        prior_mean,
        prior_covariance,
        current_draw_count,
        rng,
    )
    gaussian_prior = _parameter_set_from_joint_normal_draws(
        uncertain_names,
        prior_draws,
        outcome_sd,
    )
    prior_net_benefits = _validate_net_benefits(
        model_func(gaussian_prior).numpy_values,
        current_draw_count,
    )
    current_value = float(np.max(_finite_strategy_means(prior_net_benefits)))

    predictive_mean = design_matrix @ prior_mean
    predictive_covariance = (
        design_matrix @ prior_covariance @ design_matrix.T + observation_covariance
    )
    try:
        predictive_factor = np.linalg.cholesky(predictive_covariance)
    except np.linalg.LinAlgError as error:
        raise_input_error(
            "Built-in normal EVSI predictive covariance is singular or not "
            "positive definite.",
            diagnostic_code="singular_predictive_covariance",
        )
        raise AssertionError("unreachable") from error

    posterior_values: list[float] = []
    for _ in range(n_outer_loops):
        observation = predictive_mean + predictive_factor @ rng.standard_normal(
            predictive_mean.size
        )
        posterior_psa = _joint_normal_posterior_from_contract(
            uncertain_names,
            prior_mean,
            prior_covariance,
            design_matrix,
            observation_covariance,
            observation,
            outcome_sd,
            n_inner_loops,
            rng,
        )
        net_benefits = _validate_net_benefits(
            model_func(posterior_psa).numpy_values,
            n_inner_loops,
        )
        posterior_values.append(float(np.max(_finite_strategy_means(net_benefits))))
    with np.errstate(over="ignore", invalid="ignore"):
        expected_sample_value = float(np.mean(posterior_values))
    if not np.isfinite(expected_sample_value):
        raise_input_error(
            "Posterior net-benefit aggregation produced a non-finite result.",
            diagnostic_code="non_finite_reduction",
        )
    return expected_sample_value, current_value


def _evsi_two_loop(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_outer_loops: int,
    n_inner_loops: int,
    trial_simulator: TrialSimulatorType,
    posterior_sampler: PosteriorSamplerType,
    rng: np.random.Generator,
) -> float:
    """Calculate EVSI using caller-declared sampling and posterior models.

    Each outer loop samples one possible trial data set. The inner loop then
    receives exactly ``n_inner_loops`` joint posterior draws from the caller's
    posterior sampler. Keeping both callbacks explicit prevents this generic
    interface from silently replacing a joint prior with marginal updates.
    """
    max_nb_post_study = []
    for _ in range(n_outer_loops):
        true_params_idx = rng.integers(0, psa_prior.n_samples)
        true_params = {
            name: float(values[true_params_idx])
            for name, values in psa_prior.parameters.items()
        }

        trial_data = trial_simulator(true_params, trial_design, rng)
        posterior_psa = posterior_sampler(
            psa_prior,
            trial_data,
            trial_design,
            n_inner_loops,
            rng,
        )
        if not isinstance(posterior_psa, ParameterSet):
            raise_input_error("`posterior_sampler` must return a ParameterSet.")
        if posterior_psa.n_samples != n_inner_loops:
            raise_input_error(
                "`posterior_sampler` must return exactly `n_inner_loops` joint draws."
            )
        nb_posterior = _validate_net_benefits(
            model_func(posterior_psa).numpy_values,
            n_inner_loops,
        )
        max_nb_post_study.append(float(np.max(_finite_strategy_means(nb_posterior))))

    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(max_nb_post_study))
    if not np.isfinite(result):
        raise_input_error(
            "Posterior net-benefit aggregation produced a non-finite result.",
            diagnostic_code="non_finite_reduction",
        )
    return result


def _evsi_regression(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_regression_samples: int,
) -> float:
    """EVSI calculation using a regression-based approach."""
    # Subsample parameter sets for regression
    n_samples = psa_prior.n_samples
    if n_regression_samples >= n_samples:
        # Use all samples if requested number is greater than or equal to available
        indices = np.arange(n_samples)
        sampled_psa = psa_prior
    else:
        # Randomly sample parameter sets
        indices = np.random.choice(n_samples, n_regression_samples, replace=False)
        # Create a new ParameterSet with sampled parameters
        sampled_parameters = {}
        for name, values in psa_prior.parameters.items():
            sampled_parameters[name] = values[indices]
        import xarray as xr

        dataset = xr.Dataset(
            {k: (("n_samples",), v) for k, v in sampled_parameters.items()},
            coords={"n_samples": np.arange(len(indices))},
        )
        sampled_psa = ParameterSet(dataset=dataset)

    # Run model on prior samples to get prior net benefits
    _ = model_func(sampled_psa).numpy_values

    # For each sampled parameter set, simulate trial data and get posterior net benefits
    nb_posterior_list = []
    for i in range(len(indices)):
        # Extract true parameters for this sample
        true_params = {
            name: values[i] for name, values in sampled_psa.parameters.items()
        }

        # Simulate trial data
        trial_data = _simulate_trial_data(true_params, trial_design)

        # Bayesian update
        posterior_psa = _bayesian_update(
            psa_prior,
            trial_data,
            trial_design,
            n_draws=psa_prior.n_samples,
        )

        # Run model on posterior
        nb_posterior = model_func(posterior_psa).numpy_values
        nb_posterior_list.append(nb_posterior)

    # Stack posterior net benefits
    nb_posterior_array = np.stack(
        nb_posterior_list, axis=0
    )  # (n_samples, n_strategies)

    # Calculate max net benefit for each posterior sample
    mean_nb_per_strategy = np.mean(
        nb_posterior_array, axis=1
    )  # (n_samples, n_strategies)
    max_nb_posterior = np.max(mean_nb_per_strategy, axis=1)  # (n_samples,)

    # Prepare data for regression
    # x: prior parameter values (flatten to 2D array)
    x = np.stack(
        list(sampled_psa.parameters.values()), axis=1
    )  # (n_samples, n_parameters)

    # y: max net benefit from posterior
    y = max_nb_posterior

    # Predict max net benefit for all prior samples
    x_all = np.stack(
        list(psa_prior.parameters.values()), axis=1
    )  # (n_samples, n_parameters)

    # Callback execution remains Python-owned; deterministic OLS aggregation
    # is Rust-owned under the versioned regression envelope.
    result = _runtime.compute_evsi_regression(
        y.reshape(-1, 1).tolist(),
        x.tolist(),
        x_all.tolist(),
    )
    try:
        if (
            result.get("estimator") != "regression"
            or result.get("contract_version") != 1
            or result.get("sample_count") != len(y)
            or result.get("prediction_count") != len(x_all)
            or result.get("parameter_count") != x.shape[1]
        ):
            raise ValueError("native regression metadata mismatch")  # noqa: TRY301
        expected_sample_value = float(result["expected_sample_value"])
        if not np.isfinite(expected_sample_value):
            raise ValueError("native regression result is non-finite")  # noqa: TRY301
    except (KeyError, TypeError, ValueError) as error:
        raise_input_error("Native regression EVSI returned an invalid result envelope.")
        raise AssertionError("unreachable") from error
    else:
        return expected_sample_value


def evsi(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
    method: str = "two_loop",
    n_outer_loops: int = 100,
    n_inner_loops: int = 1000,
    metamodel: MetamodelName = "linear",
    *,
    seed: int | None = None,
    trial_simulator: TrialSimulatorType | None = None,
    posterior_sampler: PosteriorSamplerType | None = None,
) -> float:
    r"""Calculate expected value of sample information.

    Parameters
    ----------
    model_func : callable
        Economic model that maps a :class:`~voiage.schema.ParameterSet` to a
        :class:`~voiage.schema.ValueArray`.
    psa_prior : ParameterSet
        Prior PSA sample used as the analysis base.
    trial_design : TrialDesign
        Proposed study design to evaluate.
    population : float, optional
        Population size for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    method : {"two_loop", "regression", "efficient", "moment_based"}
        EVSI approximation method.
    n_outer_loops : int, default=100
        Number of outer Monte Carlo loops for ``two_loop``.
    n_inner_loops : int, default=1000
        Number of posterior Monte Carlo draws per simulated trial data set for
        ``two_loop``. Larger values reduce inner-loop Monte Carlo noise at the
        cost of additional model evaluations.
    metamodel : str, default="linear"
        Strategy-level surrogate model used by the efficient approximation.
    seed : int, optional
        A uint64-range seed for reproducible Python two-loop simulation. When
        omitted, a fresh local random generator is used.
    trial_simulator : callable, optional
        Given one joint prior draw, the trial design, and a random generator,
        returns data from a declared sampling model. Supply this together with
        ``posterior_sampler`` for a custom two-loop model. When both callbacks
        are omitted, the built-in joint multivariate-normal prior and
        known-variance arm-mean likelihood are used.
    posterior_sampler : callable, optional
        Given the joint prior, simulated data, trial design, requested draw
        count, and random generator, returns exactly that many joint posterior
        draws as a ``ParameterSet``. It must be supplied together with
        ``trial_simulator``.

    Returns
    -------
    float
        EVSI on a per-decision basis unless population scaling is requested.

    Notes
    -----
    EVSI is the increase in expected net benefit from observing a proposed
    study before making the decision:

    .. math::

       \mathrm{EVSI} = E_y\left[\max_d E[NB_d \mid y]\right] - \max_d E[NB_d].

    Without custom callbacks, the built-in v2 contract estimates a joint
    multivariate-normal prior from the PSA draws and uses one ``mean_<arm>``
    parameter per trial arm plus a fixed ``sd_outcome``. Current value,
    predictive trial outcomes, and posterior value are all integrated under
    that fitted Gaussian prior, and correlated parameters are updated jointly.
    Custom callbacks may instead declare another coherent sampling and
    posterior model. Compatibility estimators remain available for older
    workflows but are not stable scientific contracts.

    References
    ----------
    Brennan, A., Kharroubi, S., O'Hagan, A., Chilcott, J., & Claxton, K.
    (2007). Calculating expected value of sample information via Bayesian
    numerical analysis.
    Heath, A., Manolopoulou, I., & Baio, G. (2018). Efficient computation of
    expected value of sample information using regression-based methods.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.sample_information import evsi
    >>> from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray
    >>> def model(psa: ParameterSet) -> ValueArray:
    ...     values = np.column_stack([
    ...         10.0 + psa.parameters["shift"],
    ...         11.0 - psa.parameters["shift"],
    ...     ])
    ...     return ValueArray.from_numpy(values, ["A", "B"])
    >>> psa = ParameterSet.from_numpy_or_dict({"shift": np.array([0.1, 0.2, 0.3])})
    >>> design = TrialDesign(arms=[DecisionOption(name="A", sample_size=10)])
    >>> result = evsi(model, psa, design, method="efficient")
    >>> result >= 0.0
    True
    """
    if not isinstance(psa_prior, ParameterSet):
        raise_input_error("`psa_prior` must be a ParameterSet object.")
    if not isinstance(trial_design, TrialDesign):
        raise_input_error("`trial_design` must be a TrialDesign object.")
    if not callable(model_func):
        raise_input_error("`model_func` must be a callable function.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
    ):
        raise_input_error("seed must be an integer between 0 and 2**64 - 1 inclusive.")
    if seed is not None and method != "two_loop":
        raise_input_error("seed is currently supported only for method='two_loop'.")
    if (
        isinstance(n_outer_loops, bool)
        or not isinstance(n_outer_loops, int)
        or n_outer_loops <= 0
        or isinstance(n_inner_loops, bool)
        or not isinstance(n_inner_loops, int)
        or n_inner_loops <= 0
    ):
        raise_input_error(
            "n_outer_loops and n_inner_loops must be non-Boolean positive integers.",
            diagnostic_code="invalid_loop_count",
        )
    _parameter_matrix(psa_prior)

    expected_max_nb_post_study: float
    max_expected_nb_current_info: float
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]]

    if method == "two_loop":
        if (trial_simulator is None) != (posterior_sampler is None):
            raise_input_error(
                "Custom method='two_loop' models require both `trial_simulator` "
                "and `posterior_sampler` callbacks."
            )
        if trial_simulator is None:
            (
                expected_max_nb_post_study,
                max_expected_nb_current_info,
            ) = _evsi_builtin_joint_normal(
                model_func,
                psa_prior,
                trial_design,
                n_outer_loops,
                n_inner_loops,
                np.random.default_rng(seed),
            )
        else:
            nb_prior_values = _validate_net_benefits(
                model_func(psa_prior).numpy_values,
                psa_prior.n_samples,
            )
            max_expected_nb_current_info = float(
                np.max(_finite_strategy_means(nb_prior_values))
            )
            assert posterior_sampler is not None
            expected_max_nb_post_study = _evsi_two_loop(
                model_func,
                psa_prior,
                trial_design,
                n_outer_loops,
                n_inner_loops,
                trial_simulator,
                posterior_sampler,
                np.random.default_rng(seed),
            )
    elif method == "regression":
        nb_prior_values = _validate_net_benefits(
            model_func(psa_prior).numpy_values,
            psa_prior.n_samples,
        )
        max_expected_nb_current_info = float(
            np.max(_finite_strategy_means(nb_prior_values))
        )
        warnings.warn(
            "[evsi_regression_nonstable] The regression EVSI compatibility "
            "estimator does not expose a "
            "complete, validated study-model contract and is not a stable "
            "scientific estimator. Use method='two_loop' with a declared "
            "study model or normal_normal_two_arm_evsi(); see the v1-to-v2 "
            "migration guide.",
            FutureWarning,
            stacklevel=2,
        )
        # Implement regression-based EVSI method
        expected_max_nb_post_study = _evsi_regression(
            model_func, psa_prior, trial_design, n_outer_loops
        )

    elif method in {"efficient", "efficient_regression"}:
        nb_prior_values = _validate_net_benefits(
            model_func(psa_prior).numpy_values,
            psa_prior.n_samples,
        )
        max_expected_nb_current_info = float(
            np.max(_finite_strategy_means(nb_prior_values))
        )
        warnings.warn(
            "[evsi_efficient_nonstable] The efficient EVSI compatibility "
            "estimator does not expose a "
            "complete, validated study-model contract and is not a stable "
            "scientific estimator. Use method='two_loop' with a declared "
            "study model or normal_normal_two_arm_evsi(); see the v1-to-v2 "
            "migration guide.",
            FutureWarning,
            stacklevel=2,
        )
        expected_max_nb_post_study = _evsi_efficient_regression(
            nb_prior_values,
            psa_prior,
            trial_design,
            metamodel=metamodel,
        )

    elif method == "moment_based":
        nb_prior_values = _validate_net_benefits(
            model_func(psa_prior).numpy_values,
            psa_prior.n_samples,
        )
        max_expected_nb_current_info = float(
            np.max(_finite_strategy_means(nb_prior_values))
        )
        warnings.warn(
            "[evsi_moment_based_nonstable] The moment-based EVSI compatibility "
            "estimator does not expose a "
            "complete, validated study-model contract and is not a stable "
            "scientific estimator. Use method='two_loop' with a declared "
            "study model or normal_normal_two_arm_evsi(); see the v1-to-v2 "
            "migration guide.",
            FutureWarning,
            stacklevel=2,
        )
        expected_max_nb_post_study = _evsi_moment_based(
            nb_prior_values,
            psa_prior,
            trial_design,
        )

    else:
        raise_backend_not_available_error(
            f"EVSI method '{method}' is not recognized or implemented."
        )

    raw_per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    if not np.isfinite(raw_per_decision_evsi):
        raise_input_error(
            "EVSI subtraction produced a non-finite result.",
            diagnostic_code="non_finite_reduction",
        )
    if method == "two_loop" and raw_per_decision_evsi < 0.0:
        model_guidance = (
            " Also verify that the custom prior, trial simulator, and posterior "
            "sampler define one coherent model."
            if trial_simulator is not None
            else ""
        )
        warnings.warn(
            "[evsi_negative_monte_carlo_estimate] The untruncated Monte Carlo "
            f"EVSI estimate was {raw_per_decision_evsi:.12g}. Repeat across "
            "seeds, increase outer and inner loops, and assess convergence "
            f"before interpreting it.{model_guidance}",
            RuntimeWarning,
            stacklevel=2,
        )
    per_decision_evsi = (
        raw_per_decision_evsi
        if method == "two_loop"
        else max(0.0, raw_per_decision_evsi)
    )

    if population is not None and time_horizon is not None:
        if population <= 0:
            raise_input_error("Population must be positive.")
        if time_horizon <= 0:
            raise_input_error("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise_input_error("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_evsi = per_decision_evsi * population * annuity
        if not np.isfinite(scaled_evsi):
            raise_input_error(
                "Population scaling produced a non-finite EVSI result.",
                diagnostic_code="non_finite_reduction",
            )
        return scaled_evsi

    return per_decision_evsi


def enbs(evsi_result: float, research_cost: float) -> float:
    r"""Calculate expected net benefit of sampling.

    Parameters
    ----------
    evsi_result : float
        Result from an EVSI calculation.
    research_cost : float
        Total cost of the proposed research.

    Returns
    -------
    float
        ``EVSI - research_cost``.

    Notes
    -----
    ENBS is the net gain from commissioning a study:

    .. math::

       \mathrm{ENBS} = \mathrm{EVSI} - C_\mathrm{research}.

    References
    ----------
    Claxton, K. (1999). The irrelevance of inference: A decision-making
    approach to the stochastic evaluation of health care technologies.
    Jalal, H., & Alarid-Escudero, F. (2023). Cost-effectiveness and the value
    of information in health economic evaluation.

    Examples
    --------
    >>> from voiage.methods.sample_information import enbs
    >>> enbs(12.5, 5.0)
    7.5
    """
    if not isinstance(evsi_result, (int, float)):
        raise_input_error("EVSI result must be a number.")
    if not isinstance(research_cost, (int, float)):
        raise_input_error("Research cost must be a number.")
    if not np.isfinite(evsi_result) or not np.isfinite(research_cost):
        raise_input_error(
            "EVSI result and research cost must be finite.",
            diagnostic_code="non_finite_value",
        )
    if research_cost < 0:
        raise_input_error("Research cost cannot be negative.")
    try:
        return _runtime.compute_enbs(float(evsi_result), float(research_cost))
    except (ModuleNotFoundError, ImportError):
        raise_backend_not_available_error("ENBS requires the Rust runtime extension.")
