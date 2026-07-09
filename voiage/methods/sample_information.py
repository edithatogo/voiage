"""
Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from collections.abc import Callable, Sequence
import importlib.util
from typing import Any

import numpy as np

from voiage.exceptions import (
    raise_input_error,
    raise_not_implemented_error,
)
from voiage.schema import ParameterSet, TrialDesign, ValueArray
from voiage.stats import normal_normal_update

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

EconomicModelFunctionType = Callable[[ParameterSet], ValueArray]
MetamodelName = str


def _parameter_matrix(psa_prior: ParameterSet) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return PSA parameters as a finite 2D NumPy array."""
    if not psa_prior.parameters:
        raise_input_error("`psa_prior` must contain at least one parameter.")

    matrix = np.column_stack(
        [np.asarray(values, dtype=float) for values in psa_prior.parameters.values()]
    )
    if matrix.ndim != 2 or matrix.shape[0] != psa_prior.n_samples:
        raise_input_error("PSA parameters must form a 2D sample-by-parameter matrix.")
    if not np.all(np.isfinite(matrix)):
        raise_input_error("PSA parameters must contain only finite values.")
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
        raise_input_error("Model output must contain only finite net-benefit values.")
    return values


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
        raise_not_implemented_error(
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
    if not SKLEARN_AVAILABLE:
        raise_not_implemented_error("Efficient EVSI requires scikit-learn.")

    x = _parameter_matrix(psa_prior)
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


def _quadratic_design_matrix(
    x: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Build a centered linear-plus-quadratic design matrix."""
    centered = x - np.mean(x, axis=0)
    quadratic_terms = [centered[:, i] ** 2 for i in range(centered.shape[1])]
    interaction_terms = [
        centered[:, i] * centered[:, j]
        for i in range(centered.shape[1])
        for j in range(i + 1, centered.shape[1])
    ]
    linear_terms = [centered[:, i] for i in range(centered.shape[1])]
    columns: Sequence[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.ones(centered.shape[0]),
        *linear_terms,
        *quadratic_terms,
        *interaction_terms,
    ]
    return np.column_stack(columns)


def _evsi_moment_based(
    nb_prior_values: np.ndarray[Any, np.dtype[np.float64]],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
) -> float:
    """Moment-based EVSI approximation using local surrogate moments."""
    x = _parameter_matrix(psa_prior)
    design = _quadratic_design_matrix(x)
    coefficients, *_ = np.linalg.lstsq(design, nb_prior_values, rcond=None)
    predicted_nb = np.asarray(design @ coefficients, dtype=float)
    information_fraction = _trial_information_fraction(
        trial_design,
        psa_prior.n_samples,
    )
    return _preposterior_expected_max(
        predicted_nb,
        nb_prior_values,
        information_fraction,
    )


def _simulate_trial_data(
    true_parameters: dict[str, Any],
    trial_design: TrialDesign,
) -> dict[str, np.ndarray[Any, Any]]:
    """Simulate trial data based on true parameters."""
    data = {}
    for arm in trial_design.arms:
        mean = true_parameters[f"mean_{arm.name.lower().replace(' ', '_')}"]
        std_dev = true_parameters["sd_outcome"]
        data[arm.name] = np.random.normal(mean, std_dev, arm.sample_size)
    return data


def _bayesian_update(
    prior_samples: ParameterSet,
    trial_data: dict[str, np.ndarray[Any, Any]],
    trial_design: TrialDesign,
) -> ParameterSet:
    """Update prior beliefs with simulated trial data."""
    posterior_samples = {}
    for param_name, prior_values in prior_samples.parameters.items():
        if "mean" in param_name:
            arm_name = param_name.replace("mean_", "").replace("_", " ").title()
            if arm_name in trial_data:
                data = trial_data[arm_name]
                posterior_mean, posterior_std = normal_normal_update(
                    prior_values,
                    prior_samples.parameters["sd_outcome"],
                    np.mean(data),
                    np.std(data),
                    len(data),
                )
                posterior_samples[param_name] = np.random.normal(
                    posterior_mean, posterior_std, len(prior_values)
                )
            else:
                posterior_samples[param_name] = prior_values
        else:
            posterior_samples[param_name] = prior_values
    import xarray as xr

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in posterior_samples.items()},
        coords={"n_samples": np.arange(len(next(iter(posterior_samples.values()))))},
    )
    return ParameterSet(dataset=dataset)


def _evsi_two_loop(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_outer_loops: int,
    n_inner_loops: int,
) -> float:
    """EVSI calculation using a two-loop Monte Carlo simulation."""
    max_nb_post_study = []
    for _ in range(n_outer_loops):
        true_params_idx = np.random.randint(0, psa_prior.n_samples)
        true_params = {
            name: values[true_params_idx]
            for name, values in psa_prior.parameters.items()
        }

        trial_data = _simulate_trial_data(true_params, trial_design)
        posterior_psa = _bayesian_update(psa_prior, trial_data, trial_design)
        nb_posterior = model_func(posterior_psa).numpy_values
        max_nb_post_study.append(np.max(np.mean(nb_posterior, axis=0)))

    return float(np.mean(max_nb_post_study))


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
        posterior_psa = _bayesian_update(sampled_psa, trial_data, trial_design)

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

    # Fit regression model
    from sklearn.linear_model import LinearRegression

    regression_model = LinearRegression()
    regression_model.fit(x, y)

    # Predict max net benefit for all prior samples
    x_all = np.stack(
        list(psa_prior.parameters.values()), axis=1
    )  # (n_samples, n_parameters)
    predicted_max_nb = regression_model.predict(x_all)

    # Return expected max net benefit
    return float(np.mean(predicted_max_nb))


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
        Number of inner Monte Carlo loops for ``two_loop``.
    metamodel : str, default="linear"
        Strategy-level surrogate model used by the efficient approximation.

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

    The exact two-loop estimator uses nested Monte Carlo. The efficient and
    moment-based estimators approximate the same preposterior target while
    avoiding the inner simulation loop.

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
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise_input_error("n_outer_loops and n_inner_loops must be positive.")

    nb_prior_values = _validate_net_benefits(
        model_func(psa_prior).numpy_values,
        psa_prior.n_samples,
    )
    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    expected_max_nb_post_study: float

    if method == "two_loop":
        expected_max_nb_post_study = _evsi_two_loop(
            model_func, psa_prior, trial_design, n_outer_loops, n_inner_loops
        )
    elif method == "regression":
        if not SKLEARN_AVAILABLE:
            raise_not_implemented_error(
                "Regression method for EVSI requires scikit-learn."
            )

        # Implement regression-based EVSI method
        expected_max_nb_post_study = _evsi_regression(
            model_func, psa_prior, trial_design, n_outer_loops
        )

    elif method in {"efficient", "efficient_regression"}:
        expected_max_nb_post_study = _evsi_efficient_regression(
            nb_prior_values,
            psa_prior,
            trial_design,
            metamodel=metamodel,
        )

    elif method == "moment_based":
        expected_max_nb_post_study = _evsi_moment_based(
            nb_prior_values,
            psa_prior,
            trial_design,
        )

    else:
        raise_not_implemented_error(
            f"EVSI method '{method}' is not recognized or implemented."
        )

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)

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
        return per_decision_evsi * population * annuity

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
    if research_cost < 0:
        raise_input_error("Research cost cannot be negative.")
    return evsi_result - research_cost
