"""
Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from typing import Callable, Optional

import numpy as np

from voiage.exceptions import (
    InputError,
    VoiageNotImplementedError,
)
from voiage.schema import ParameterSet, TrialDesign, ValueArray
from voiage.stats import normal_normal_update

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None

EconomicModelFunctionType = Callable[[ParameterSet], ValueArray]


def _simulate_trial_data(true_parameters, trial_design):
    """Simulate trial data based on true parameters."""
    data = {}
    for arm in trial_design.arms:
        mean = true_parameters[f"mean_{arm.name.lower().replace(' ', '_')}"]
        std_dev = true_parameters["sd_outcome"]
        data[arm.name] = np.random.normal(mean, std_dev, arm.sample_size)
    return data


def _bayesian_update(prior_samples, trial_data, trial_design):
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


def _evsi_two_loop(model_func, psa_prior, trial_design, n_outer_loops, n_inner_loops):
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
        nb_posterior = model_func(posterior_psa).values
        max_nb_post_study.append(np.max(np.mean(nb_posterior, axis=0)))

    return np.mean(max_nb_post_study)


def _evsi_regression(model_func, psa_prior, trial_design, n_regression_samples):
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
            coords={"n_samples": np.arange(len(indices))}
        )
        sampled_psa = ParameterSet(dataset=dataset)

    # Run model on prior samples to get prior net benefits
    _ = model_func(sampled_psa).values

    # For each sampled parameter set, simulate trial data and get posterior net benefits
    nb_posterior_list = []
    for i in range(len(indices)):
        # Extract true parameters for this sample
        true_params = {
            name: values[i]
            for name, values in sampled_psa.parameters.items()
        }

        # Simulate trial data
        trial_data = _simulate_trial_data(true_params, trial_design)

        # Bayesian update
        posterior_psa = _bayesian_update(sampled_psa, trial_data, trial_design)

        # Run model on posterior
        nb_posterior = model_func(posterior_psa).values
        nb_posterior_list.append(nb_posterior)

    # Stack posterior net benefits
    nb_posterior_array = np.stack(nb_posterior_list, axis=0)  # (n_samples, n_strategies)

    # Calculate max net benefit for each posterior sample
    mean_nb_per_strategy = np.mean(nb_posterior_array, axis=1)  # (n_samples, n_strategies)
    max_nb_posterior = np.max(mean_nb_per_strategy, axis=1)  # (n_samples,)

    # Prepare data for regression
    # x: prior parameter values (flatten to 2D array)
    x = np.stack(list(sampled_psa.parameters.values()), axis=1)  # (n_samples, n_parameters)

    # y: max net benefit from posterior
    y = max_nb_posterior

    # Fit regression model
    regression_model = LinearRegression()
    regression_model.fit(x, y)

    # Predict max net benefit for all prior samples
    x_all = np.stack(list(psa_prior.parameters.values()), axis=1)  # (n_samples, n_parameters)
    predicted_max_nb = regression_model.predict(x_all)

    # Return expected max net benefit
    return np.mean(predicted_max_nb)


def evsi(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    method: str = "two_loop",
    n_outer_loops: int = 100,
    n_inner_loops: int = 1000,
) -> float:
    """Calculate the Expected Value of Sample Information (EVSI)."""
    if not isinstance(psa_prior, ParameterSet):
        raise InputError("`psa_prior` must be a ParameterSet object.")
    if not isinstance(trial_design, TrialDesign):
        raise InputError("`trial_design` must be a TrialDesign object.")
    if not callable(model_func):
        raise InputError("`model_func` must be a callable function.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    nb_prior_values = model_func(psa_prior).values
    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    expected_max_nb_post_study: float

    if method == "two_loop":
        expected_max_nb_post_study = _evsi_two_loop(
            model_func, psa_prior, trial_design, n_outer_loops, n_inner_loops
        )
    elif method == "regression":
        if not SKLEARN_AVAILABLE:
            raise VoiageNotImplementedError(
                "Regression method for EVSI requires scikit-learn."
            )

        # Implement regression-based EVSI method
        expected_max_nb_post_study = _evsi_regression(
            model_func, psa_prior, trial_design, n_outer_loops
        )

    else:
        raise VoiageNotImplementedError(
            f"EVSI method '{method}' is not recognized or implemented."
        )

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)

    if population is not None and time_horizon is not None:
        if population <= 0:
            raise InputError("Population must be positive.")
        if time_horizon <= 0:
            raise InputError("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise InputError("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_evsi * population * annuity

    return per_decision_evsi


def enbs(evsi_result: float, research_cost: float) -> float:
    """
    Calculate the Expected Net Benefit of Sampling (ENBS).

    Args:
        evsi_result (float): The result from an EVSI calculation.
        research_cost (float): The cost of the proposed research.

    Returns
    -------
        float: The calculated ENBS.
    """
    if not isinstance(evsi_result, (int, float)):
        raise InputError("EVSI result must be a number.")
    if not isinstance(research_cost, (int, float)):
        raise InputError("Research cost must be a number.")
    if research_cost < 0:
        raise InputError("Research cost cannot be negative.")
    return evsi_result - research_cost
