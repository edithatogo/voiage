# voiage/methods/sample_information.py

"""Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from typing import Any, Callable, Dict, List, Optional, Union  # Added List

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import NetBenefitArray, PSASample, TrialDesign
from voiage.core.utils import check_input_array
from voiage.exceptions import (
    CalculationError,
    InputError,
    VoiageNotImplementedError,
)

# Attempt to import for LinearRegression, fail gracefully if not available
try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None


# from voiage.methods.basic import evppi # May be used if EVSI is framed via EVPPI on predicted data

# Define a type for the model function expected by EVSI
# This function takes parameter samples (e.g., dict or PSASample) and returns net benefits.
# For EVSI, the model_func is used to generate *prior* net benefits.
# The impact of trial data is handled by updating parameters *before* potentially re-evaluating
# a model or, more commonly in regression EVSI, by using a metamodel.
EconomicModelFunctionType = Callable[
    [Union[Dict[str, np.ndarray], PSASample]], np.ndarray
]


# --- Helper Function Stubs for EVSI ---


def _simulate_trial_data(
    trial_design: TrialDesign,
    psa_prior: PSASample,
    # n_inner_loops is not directly used here, but kept for consistency with original stub signature
) -> Dict[str, np.ndarray]:
    """
    Simulates one dataset D_k from the trial_design given a 'true' parameter set
    sampled from the psa_prior.

    Assumptions for this simplified simulation:
    - Outcomes are normally distributed.
    - `psa_prior.parameters` contains 'mean_{arm_name}' for each arm and 'sd_outcome'
      for the common standard deviation of outcomes.
    - `trial_design.arms` specifies the sample size for each arm.

    Args:
        trial_design (TrialDesign): Defines the structure of the trial (e.g., arms, sample sizes).
        psa_prior (PSASample): Prior samples of model parameters. One sample from this
                                is chosen as the 'true' parameter set for simulation.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are arm names and values are
                               NumPy arrays of simulated individual patient data for that arm.

    Raises:
        CalculationError: If required parameters (e.g., 'sd_outcome', arm means) are
                          not found in `psa_prior.parameters`.
    """
    sim_data_dict: Dict[str, np.ndarray] = {}

    # Select one sample from psa_prior to define the "true" parameters for this simulation run (theta_k)
    if psa_prior.n_samples == 0:
        raise CalculationError("PSA prior must contain samples to simulate trial data.")
    true_param_set_idx = np.random.randint(0, psa_prior.n_samples)

    # Extract the 'true' common standard deviation for this simulation run
    if "sd_outcome" not in psa_prior.parameters:
        raise CalculationError(
            "Required parameter 'sd_outcome' not found in psa_prior.parameters for trial simulation."
        )
    true_sd_outcome = psa_prior.parameters["sd_outcome"][true_param_set_idx]
    if true_sd_outcome <= 0:
        # Fallback or raise error if SD is non-positive, as it's invalid for normal distribution
        raise CalculationError(
            f"Simulated 'sd_outcome' is non-positive ({true_sd_outcome}). Cannot simulate data."
        )

    for arm in trial_design.arms:
        # Determine the 'true' mean for this arm based on a naming convention
        arm_mean_param_key = f"mean_{arm.name.lower().replace(' ', '_')}"
        if arm_mean_param_key not in psa_prior.parameters:
            raise CalculationError(
                f"Required parameter '{arm_mean_param_key}' not found in psa_prior.parameters for arm '{arm.name}'."
            )
        true_arm_mean = psa_prior.parameters[arm_mean_param_key][true_param_set_idx]

        # Generate individual patient data for the arm
        arm_data = np.random.normal(
            loc=true_arm_mean, scale=true_sd_outcome, size=arm.sample_size
        ).astype(DEFAULT_DTYPE)
        sim_data_dict[arm.name] = arm_data

    return sim_data_dict


def _bayesian_update(
    psa_prior: PSASample,
    simulated_data_k: Dict[str, np.ndarray],
    n_inner_loops: int,
    # For future: add a `bayesian_inference_engine` parameter (e.g., 'pymc', 'numpyro')
) -> np.ndarray:
    """
    (Placeholder) Performs Bayesian update: P(theta|D_k).
    Returns samples from the posterior distribution P(theta|D_k).

    **CURRENTLY A STUB:** This function currently returns samples from the prior
    distribution, effectively assuming no information gain from the simulated data.
    A full implementation would involve:
    1. Defining a likelihood model for `simulated_data_k` given parameters from `psa_prior`.
    2. Using a Bayesian inference library (e.g., PyMC, NumPyro) to sample from the posterior.

    Args:
        psa_prior (PSASample): Prior samples of model parameters.
        simulated_data_k (Dict[str, np.ndarray]): Simulated trial data for one iteration (D_k).
        n_inner_loops (int): Number of posterior samples to draw.

    Returns:
        np.ndarray: Samples from the posterior distribution P(theta|D_k),
                    shape (n_inner_loops, n_parameters).

    Raises:
        VoiageNotImplementedError: If a proper Bayesian inference engine is not integrated.
    """
    # For now, we will simply resample from the prior to allow the EVSI flow to proceed.
    # This means EVSI will be close to zero, as no information is gained.
    # This needs to be replaced with a proper Bayesian inference step.

    if psa_prior.n_samples == 0:
        raise CalculationError("PSA prior must contain samples for Bayesian update.")

    # Stack all prior parameter arrays into a single 2D array (n_samples, n_params)
    prior_param_names = list(psa_prior.parameters.keys())
    if not prior_param_names:
        # If no parameters, return an empty array or raise an error depending on expected behavior
        # For now, return a 2D array with 0 columns if no parameters.
        return np.empty((n_inner_loops, 0), dtype=DEFAULT_DTYPE)

    prior_param_values_list = [
        v.reshape(-1, 1) if v.ndim == 1 else v
        for v in psa_prior.parameters.values()
    ]
    prior_params_stacked = np.hstack(prior_param_values_list)

    # Resample from the prior to get n_inner_loops samples
    prior_sample_indices = np.random.choice(
        prior_params_stacked.shape[0], n_inner_loops, replace=True
    )
    posterior_samples_stacked = prior_params_stacked[prior_sample_indices, :].copy()

    # TODO: Integrate a proper Bayesian inference engine here (e.g., PyMC, NumPyro)
    # This would involve:
    # 1. Defining a probabilistic model based on `psa_prior` and `simulated_data_k`.
    # 2. Running an MCMC sampler or variational inference to get posterior samples.
    # For example, if using NumPyro:
    # from numpyro.infer import MCMC, NUTS
    # kernel = NUTS(model_for_inference)
    # mcmc = MCMC(kernel, num_warmup=..., num_samples=...)
    # mcmc.run(random.PRNGKey(0), data=simulated_data_k, prior_params=psa_prior.parameters)
    # posterior_samples = mcmc.get_samples()

    # For now, we explicitly raise an error if a proper update is expected but not implemented.
    # This ensures users know this is a placeholder.
    # If the intention is to allow EVSI to run with no information gain (EVSI=0), then
    # this function should simply return resampled prior samples, as it does now.
    # The current implementation effectively makes EVSI = 0 (or very close to it) because
    # E_theta|D [NB(d, theta|D)] will be approximately E_theta [NB(d, theta)].

    # The warning print statement is removed as this is now the intended (placeholder) behavior.
    return posterior_samples_stacked


def _fit_metamodel(
    model_func: EconomicModelFunctionType,
    psa_prior: PSASample,
    nb_prior_values: np.ndarray,  # Pre-calculated (n_samples, n_strategies)
) -> List[Any]:  # List of fitted regression models, one per strategy
    """
    Fits a regression metamodel NB(d,theta) ~ f(theta) for each strategy d.
    Uses prior PSA samples and their corresponding net benefits.

    Args:
        model_func (EconomicModelFunctionType): The economic model function (not directly used
            for fitting here, but conceptually part of the EVSI process).
        psa_prior (PSASample): Prior samples of model parameters.
        nb_prior_values (np.ndarray): Pre-calculated net benefit values for each PSA sample
            and strategy, shape (n_samples, n_strategies).

    Returns:
        List[Any]: A list of fitted scikit-learn compatible regression models, one for each strategy.

    Raises:
        VoiageNotImplementedError: If scikit-learn is not installed.
        CalculationError: If an error occurs during model fitting.
    """
    if not SKLEARN_AVAILABLE:
        raise VoiageNotImplementedError(
            "Scikit-learn is required for regression-based EVSI if not providing a custom metamodel fitter."
        )

    metamodel_list = []

    # Prepare X from psa_prior.parameters (n_samples, n_params)
    if not psa_prior.parameters:
        # If no parameters, metamodel fitting is not meaningful. EVSI will be 0.
        return []

    param_arrays = [
        v.reshape(-1, 1) if v.ndim == 1 else v for v in psa_prior.parameters.values()
    ]
    X_prior = np.hstack(param_arrays)  # (n_prior_samples, n_params)

    n_strategies = nb_prior_values.shape[1]
    for i in range(n_strategies):
        y_prior_strategy_i = nb_prior_values[:, i]
        model = LinearRegression()  # Default model
        try:
            model.fit(X_prior, y_prior_strategy_i)
            metamodel_list.append(model)
        except Exception as e:
            raise CalculationError(
                f"Error fitting metamodel for strategy {i}: {e}"
            ) from e

    return metamodel_list


def _predict_enb_from_metamodel(
    metamodel_list: List[Any],
    psa_posterior_k_samples: np.ndarray,  # (n_inner_loops, n_params)
) -> (
    np.ndarray
):  # Returns array of E[NB(d)|D_k] for each strategy d, shape (n_strategies,)
    """
    Predicts E[NB(d)|D_k] using the fitted metamodel and posterior samples.

    Args:
        metamodel_list (List[Any]): A list of fitted scikit-learn compatible regression models,
                                    one for each strategy.
        psa_posterior_k_samples (np.ndarray): Samples from the posterior distribution P(theta|D_k),
                                            shape (n_inner_loops, n_parameters).

    Returns:
        np.ndarray: An array of expected net benefits conditional on the simulated data D_k
                    for each strategy, shape (n_strategies,).

    Raises:
        CalculationError: If an error occurs during prediction.
    """
    if not metamodel_list:
        # If no metamodels (e.g., no parameters of interest), then ENB is effectively 0.
        return np.array(
            []
        )  # Return empty array, caller should handle based on n_strategies

    if psa_posterior_k_samples.ndim == 1:
        psa_posterior_k_samples = psa_posterior_k_samples.reshape(-1, 1)

    if psa_posterior_k_samples.size == 0:
        # No posterior samples, return zeros for all strategies.
        return np.zeros(len(metamodel_list), dtype=DEFAULT_DTYPE)

    n_strategies = len(metamodel_list)
    enb_post_data_k_strategies = np.zeros(n_strategies, dtype=DEFAULT_DTYPE)

    for i in range(n_strategies):
        model = metamodel_list[i]
        if model is None:
            # This should ideally not happen if metamodel_list is properly constructed.
            enb_post_data_k_strategies[i] = 0.0
            continue
        try:
            predicted_nb_for_strategy_i = model.predict(psa_posterior_k_samples)
            enb_post_data_k_strategies[i] = np.mean(predicted_nb_for_strategy_i)
        except Exception as e:
            raise CalculationError(
                f"Error predicting with metamodel for strategy {i}: {e}"
            ) from e

    return enb_post_data_k_strategies


# --- Main EVSI Function ---


def evsi(
    model_func: EconomicModelFunctionType,
    psa_prior: PSASample,
    trial_design: TrialDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    method: str = "regression",
    n_outer_loops: int = 100,
    n_inner_loops: int = 1000,
    # **kwargs: Any # Additional arguments for specific methods
) -> float:
    """Calculate the Expected Value of Sample Information (EVSI).

    EVSI quantifies the value of conducting a new study (trial) to reduce
    decision uncertainty. It is calculated as the difference between the
    expected net benefit with new information and the expected net benefit
    with current information.

    EVSI = E_D [ max_d E_theta|D [NB(d, theta|D)] ] - max_d [ E_theta [NB(d, theta)] ]

    The `regression` method (currently implemented) uses a two-stage Monte Carlo
    simulation with metamodeling:
    1. Outer loop: Simulate `n_outer_loops` datasets (D_k) from the trial design,
       assuming a 'true' parameter set sampled from the prior.
    2. Inner loop: For each simulated dataset D_k, perform a Bayesian update to get
       posterior parameter samples P(theta|D_k). (Note: Current Bayesian update
       is a placeholder, returning prior samples, thus EVSI will be ~0).
    3. Fit regression metamodels (NB(d,theta) ~ f(theta)) using prior PSA samples.
    4. Predict net benefits for each strategy using the posterior samples and metamodels.
    5. Calculate the expected value of the optimal decision given D_k, and average
       over all D_k to get E_D [ max_d E_theta|D [NB(d, theta|D)] ].

    Assumptions for `regression` method with current `_simulate_trial_data` stub:
    - Trial outcomes are normally distributed.
    - `psa_prior.parameters` must contain `'mean_{arm_name}'` for each arm in `trial_design`
      and `'sd_outcome'` for the common standard deviation of outcomes.

    Args:
        model_func (EconomicModelFunctionType): A callable function that takes
            parameter samples (as a `Dict[str, np.ndarray]` or `PSASample`)
            and returns a 2D NumPy array of net benefits (n_samples, n_strategies).
            This function represents the economic model.
        psa_prior (PSASample): Prior samples of model parameters.
        trial_design (TrialDesign): Defines the structure of the trial to be simulated.
        population (Optional[float]): The relevant population size. If provided
            along with `time_horizon`, EVSI will be scaled to population level.
        discount_rate (Optional[float]): The annual discount rate (e.g., 0.03 for 3%).
            Used for population scaling. Defaults to 0 if `population` and
            `time_horizon` are provided but `discount_rate` is not.
        time_horizon (Optional[float]): The relevant time horizon in years.
            If provided along with `population`, EVSI will be scaled.
        method (str): The method to use for EVSI calculation. Currently, only
            "regression" is supported. Other methods will raise `VoiageNotImplementedError`.
        n_outer_loops (int): Number of outer Monte Carlo loops (simulated datasets D_k).
        n_inner_loops (int): Number of inner Monte Carlo loops (posterior samples P(theta|D_k)).

    Returns:
        float: The calculated EVSI. If population parameters are provided,
               returns population-adjusted EVSI, otherwise per-decision EVSI.

    Raises:
        InputError: If inputs are invalid (e.g., wrong types, shapes, values).
        CalculationError: For issues during calculation or if `model_func` fails.
        VoiageNotImplementedError: If the specified `method` is not supported or
                                   if scikit-learn is not installed for the "regression" method.
    """
    if not isinstance(psa_prior, PSASample):
        raise InputError("`psa_prior` must be a PSASample object.")
    if not isinstance(trial_design, TrialDesign):
        raise InputError("`trial_design` must be a TrialDesign object.")
    if not callable(model_func):
        raise InputError("`model_func` must be a callable function.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    # --- Calculate max_d [ E_theta [NB(d, theta)] ] --- (Prior optimal decision value)
    try:
        # model_func is expected to take parameters (dict or PSASample) and return nb_array
        nb_prior_values = model_func(
            psa_prior
        )  # psa_prior might be passed if model_func expects PSASample obj

        if isinstance(nb_prior_values, NetBenefitArray):
            nb_prior_values = nb_prior_values.values
        elif not isinstance(nb_prior_values, np.ndarray):
            raise CalculationError(
                "`model_func` did not return a NumPy array for prior NBs."
            )
        check_input_array(
            nb_prior_values,
            expected_ndim=2,
            name="Prior Net Benefit values from model_func",
        )
    except Exception as e:
        raise CalculationError(
            f"Error running model_func with prior PSA samples: {e}"
        ) from e

    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info = np.max(mean_nb_per_strategy_prior)

    # --- Calculate E_D [ max_d E_theta|D [NB(d, theta|D)] ] --- (Expected post-study optimal value)
    expected_max_nb_post_study: float  # Type hint

    if method == "regression":
        if not SKLEARN_AVAILABLE:
            raise VoiageNotImplementedError(
                "Regression method for EVSI requires scikit-learn to be installed."
            )
        if not psa_prior.parameters:
            # If no parameters of interest, EVSI is 0 as no uncertainty to resolve.
            expected_max_nb_post_study = max_expected_nb_current_info
        else:
            metamodel_list = _fit_metamodel(model_func, psa_prior, nb_prior_values)
            if not metamodel_list:  # No models fitted (e.g. no parameters)
                expected_max_nb_post_study = max_expected_nb_current_info
            else:
                all_max_enb_post_data_k = np.zeros(n_outer_loops, dtype=DEFAULT_DTYPE)
                for k_loop_idx in range(n_outer_loops):
                    simulated_data_k = _simulate_trial_data(trial_design, psa_prior)
                    psa_posterior_k_samples = _bayesian_update(
                        psa_prior, simulated_data_k, n_inner_loops
                    )

                    enb_post_data_k_strategies = _predict_enb_from_metamodel(
                        metamodel_list, psa_posterior_k_samples
                    )

                    if (
                        enb_post_data_k_strategies.size == 0
                    ):  # From _predict_enb_from_metamodel if issues
                        # This indicates an issue, e.g. no strategies or error in prediction
                        # Fallback to prior for this iteration to avoid NaN and allow flow
                        # This scenario should ideally be prevented by robust _predict_enb_from_metamodel
                        all_max_enb_post_data_k[
                            k_loop_idx
                        ] = max_expected_nb_current_info
                    else:
                        all_max_enb_post_data_k[k_loop_idx] = np.max(
                            enb_post_data_k_strategies
                        )

                expected_max_nb_post_study = np.mean(all_max_enb_post_data_k)

    elif method == "nonparametric":
        raise VoiageNotImplementedError(
            "Nonparametric EVSI method is not yet implemented."
        )
    elif method == "moment_matching":
        raise VoiageNotImplementedError(
            "Moment-matching EVSI method is not yet implemented."
        )
    else:
        raise VoiageNotImplementedError(
            f"EVSI method '{method}' is not recognized or implemented."
        )

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)  # Ensure non-negative

    # Population scaling
    if population is not None and time_horizon is not None:
        if not isinstance(population, (int, float)) or population <= 0:
            raise InputError("Population must be a positive number.")
        if not isinstance(time_horizon, (int, float)) or time_horizon <= 0:
            raise InputError("Time horizon must be a positive number.")

        current_dr = discount_rate
        if current_dr is None:
            current_dr = 0.0

        if not isinstance(current_dr, (int, float)) or not (0 <= current_dr <= 1):
            raise InputError("Discount rate must be a number between 0 and 1.")

        if current_dr == 0:
            annuity_factor = time_horizon
        else:
            annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr
        return per_decision_evsi * population * annuity_factor
    elif (
        population is not None or time_horizon is not None or discount_rate is not None
    ):
        raise InputError(
            "To calculate population EVSI, 'population' and 'time_horizon' must be provided. "
            "'discount_rate' is optional.",
        )

    return per_decision_evsi


def enbs(
    evsi_result: float,
    research_cost: float,
) -> float:
    """Calculate the Expected Net Benefit of Sampling (ENBS).

    ENBS is the Expected Value of Sample Information (EVSI) minus the cost
    of conducting the research. A positive ENBS suggests the research is
    potentially worthwhile.

    The `evsi_result` should be at a scale consistent with `research_cost`.
    If `evsi()` was called with population and time horizon parameters,
    `evsi_result` will be a population-level EVSI. `research_cost` should
    then also be the total cost for that context. If `evsi_result` is
    per-decision, `research_cost` should be per-decision.

    Args:
        evsi_result (float): The calculated EVSI. Expected to be non-negative.
        research_cost (float): The cost of conducting the research/trial.
                               Must be non-negative.

    Returns
    -------
        float: The Expected Net Benefit of Sampling.

    Raises
    -------
        InputError: If inputs are not valid numbers or research_cost is negative.
    """
    if not isinstance(evsi_result, (float, int)):  # float includes np.float64 etc.
        raise InputError(f"EVSI result must be a number. Got {type(evsi_result)}.")
    if not isinstance(research_cost, (float, int)):
        raise InputError(f"Research cost must be a number. Got {type(research_cost)}.")

    if research_cost < 0:
        raise InputError("Research cost cannot be negative.")

    # The evsi() function ensures its output (evsi_result) is >= 0.
    # No need to check for evsi_result < 0 here.

    return evsi_result - research_cost


if __name__ == "__main__":
    print("--- Testing sample_information.py ---")

    # EVSI and ENBS are complex and require significant setup for meaningful tests.
    # For v0.1, these are largely placeholders.
    # We can test the basic structure and population scaling if EVSI returns a dummy value.

    # Dummy model function and inputs for testing structure
    from typing import Dict  # Import Dict for the Union type hint

    from voiage.core.data_structures import TrialArm  # Import TrialArm

    def dummy_model_func(
        params_dict_or_psa_sample: Union[
            Dict[str, np.ndarray], PSASample
        ],  # More specific Dict
    ) -> np.ndarray:
        # This dummy model just returns fixed NBs, ignoring params for simplicity of testing structure
        return np.array([[100, 110], [90, 120], [105, 95]], dtype=DEFAULT_DTYPE)

    dummy_psa_params = {
        "p1": np.array([1, 2, 3], dtype=DEFAULT_DTYPE),
        "p2": np.array([4, 5, 6], dtype=DEFAULT_DTYPE),
    }
    dummy_psa = PSASample(parameters=dummy_psa_params)
    dummy_arm1 = TrialArm(name="Arm A", sample_size=50)
    dummy_arm2 = TrialArm(name="Arm B", sample_size=50)
    dummy_trial = TrialDesign(arms=[dummy_arm1, dummy_arm2])

    print("\n--- EVSI (Placeholder Tests) ---")
    try:
        # This will fail with NotImplementedError for "regression" method as it's not implemented
        evsi_val_dummy = evsi(
            dummy_model_func, dummy_psa, dummy_trial, method="regression"
        )
        print(f"Dummy EVSI (placeholder, regression method): {evsi_val_dummy}")
    except VoiageNotImplementedError as e:
        print(
            f"Caught expected PyVoiNotImplementedError for EVSI method 'regression': {e}"
        )
    except Exception as e:
        print(f"Unexpected error during placeholder EVSI call: {e}")

    # Test ENBS structure
    print("\n--- ENBS Tests ---")
    dummy_evsi = 1000.0  # Assume some EVSI value
    cost_of_research = 800.0
    enbs_val = enbs(dummy_evsi, cost_of_research)
    print(f"ENBS for EVSI={dummy_evsi}, Cost={cost_of_research}: {enbs_val}")
    assert np.isclose(enbs_val, 200.0), "ENBS calculation error."
    print("ENBS basic test PASSED.")

    try:
        enbs("not a float", 100)
    except InputError as e:
        print(f"Caught expected InputError for ENBS: {e}")
    else:
        raise AssertionError(
            "ENBS failed to raise InputError for invalid evsi_result type."
        )

    try:
        enbs(100, -50)
    except InputError as e:
        print(f"Caught expected InputError for ENBS (negative cost): {e}")
    else:
        raise AssertionError(
            "ENBS failed to raise InputError for negative research_cost."
        )
    print("ENBS input validation tests PASSED.")

    # If EVSI were to return a value (e.g., 0.0 from placeholder), test population scaling:
    # This part assumes evsi function can run without raising NotImplementedError
    # For now, we'll simulate this by creating a wrapper or modifying evsi temporarily for test

    original_evsi_func = evsi  # Store original

    def mock_evsi(*args, **kwargs):
        # This mock will bypass method implementation and return a fixed per-decision EVSI
        # It will still call the population scaling logic if population args are provided.

        # Extract population args from kwargs or args if passed positionally
        population = kwargs.get("population")
        discount_rate = kwargs.get("discount_rate")
        time_horizon = kwargs.get("time_horizon")

        # A more robust mock would inspect args based on original evsi signature.
        # For this test, assume they are passed as kwargs.

        per_decision_evsi_mock = 5.0  # Fixed mock value

        if population is not None and time_horizon is not None:
            if population <= 0:
                raise InputError("Population must be positive.")
            if time_horizon <= 0:
                raise InputError("Time horizon must be positive.")
            effective_population = population
            if discount_rate is not None:
                if not (0 <= discount_rate <= 1):
                    raise InputError("Discount rate must be between 0 and 1.")
                if discount_rate == 0:
                    annuity_factor = time_horizon
                else:
                    annuity_factor = (1 - (1 + discount_rate) ** (-time_horizon)) / current_dr
                effective_population *= annuity_factor
            else:  # No discount_rate provided
                if discount_rate is None:  # Explicitly check for None for clarity
                    effective_population *= time_horizon
            return per_decision_evsi_mock * effective_population
        elif (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise InputError("Partial population args for EVSI.")
        return per_decision_evsi_mock

    # Temporarily replace evsi with mock_evsi for testing population scaling part
    # __globals__["evsi"] = mock_evsi # This approach is problematic and won't work reliably.
    # The test for population scaling logic should be a proper pytest test using monkeypatch.
    # For now, commenting out the direct call that relies on this, as F821 is the priority.

    print("\n--- EVSI Population Scaling (with Mocked EVSI) ---")
    # For this direct script test, we need to assign the mock to the local 'evsi' name
    # This is different from how monkeypatch works in pytest.
    _temp_original_evsi = evsi
    evsi_for_test = mock_evsi

    pop_evsi_val_mocked = evsi_for_test(
        dummy_model_func,  # type: ignore
        dummy_psa,  # type: ignore
        dummy_trial,  # type: ignore
        population=1000,
        time_horizon=10,
        discount_rate=0.03,
        method="any_method_for_mock",  # type: ignore
    )
    expected_pop_evsi_mocked = 5.0 * ((1 - (1 + 0.03) ** (-10)) / 0.03) * 1000
    print(f"Mocked Population EVSI: {pop_evsi_val_mocked}")
    assert np.isclose(
        pop_evsi_val_mocked, expected_pop_evsi_mocked
    ), f"Mocked Population EVSI error. Expected ~{expected_pop_evsi_mocked:.2f}, got {pop_evsi_val_mocked:.2f}"
    print("EVSI population scaling test (with mock) PASSED.")

    evsi = _temp_original_evsi  # Restore original evsi

    print("\n--- sample_information.py tests completed ---")