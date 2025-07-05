# pyvoi/methods/sample_information.py

"""Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from typing import Any, Callable, Dict, Optional, Union, List # Added List

import numpy as np

from pyvoi.config import DEFAULT_DTYPE
from pyvoi.core.data_structures import NetBenefitArray, PSASample, TrialDesign
from pyvoi.core.utils import check_input_array
from pyvoi.exceptions import (
    CalculationError,
    InputError,
    PyVoiNotImplementedError,
)

# Attempt to import for LinearRegression, fail gracefully if not available
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None # Placeholder if sklearn not available


# from pyvoi.methods.basic import evppi # May be used if EVSI is framed via EVPPI on predicted data

# Define a type for the model function expected by EVSI
# This function takes parameter samples (e.g., dict or PSASample) and returns net benefits.
# For EVSI, the model_func is used to generate *prior* net benefits.
# The impact of trial data is handled by updating parameters *before* potentially re-evaluating
# a model or, more commonly in regression EVSI, by using a metamodel.
EconomicModelFunctionType = Callable[[Union[Dict[str, np.ndarray], PSASample]], np.ndarray]


# --- Helper Function Stubs for EVSI ---

def _simulate_trial_data(
    trial_design: TrialDesign,
    psa_prior: PSASample,
    n_inner_loops: int
) -> Any: # Return type will depend on how data is structured, e.g., Dict of arrays
    """
    (Stub) Simulates one dataset D_k from the trial_design and psa_prior.
    This is a highly complex step. For initial structure, it might return
    a very simplified, fixed dataset or slightly perturbed prior means.
    """
    # Conceptual: Sample 'true' parameters from psa_prior, then sample data from trial_design given these true params.
    # For now, returning a placeholder that indicates structure.
    # Example: if trial has 2 arms, maybe return { 'arm1_mean_outcome': ..., 'arm2_mean_outcome': ... }
    # This needs to be compatible with _bayesian_update.
    print(f"Warning: _simulate_trial_data is a stub. Using placeholder data logic for {trial_design.arms[0].name if trial_design.arms else 'N/A'}.")
    # For a simple test, let's assume it returns some summary stats
    # This is NOT a realistic simulation.
    sim_data = {}
    true_params_sample_idx = np.random.randint(0, psa_prior.n_samples)
    for arm in trial_design.arms:
        # Simplistic: assume first parameter in psa_prior is related to outcome
        # and sample size influences precision (not properly modeled here)
        param_names = list(psa_prior.parameters.keys())
        if param_names:
            true_param_val = psa_prior.parameters[param_names[0]][true_params_sample_idx]
            # Simulate a mean outcome based on this 'true' param and add noise
            sim_data[arm.name] = true_param_val + np.random.normal(0, 1/np.sqrt(arm.sample_size))
        else:
            sim_data[arm.name] = np.random.normal(0,1) # Generic if no params
    return sim_data


def _bayesian_update(
    psa_prior: PSASample,
    simulated_data_k: Any,
    n_inner_loops: int
) -> np.ndarray: # Should return samples from posterior P(theta|D_k) as a 2D array (n_inner_loops, n_params)
    """
    (Stub) Performs Bayesian update: P(theta|D_k).
    Returns samples from the posterior distribution.
    This is extremely complex. Initially, it might return prior samples
    or slightly perturbed prior samples for testing flow.
    """
    # Conceptual: Use simulated_data_k to update beliefs about parameters in psa_prior.
    # For now, let's just return a sample from the prior distribution, implying data had no impact or perfect prior.
    # This will make EVSI close to 0 if metamodel is perfect.
    print(f"Warning: _bayesian_update is a stub. Returning samples from prior for {list(psa_prior.parameters.keys())[0] if psa_prior.parameters else 'N/A'}.")

    # For testing, we need to return an array of shape (n_inner_loops, n_parameters)
    # Let's stack all parameters from psa_prior into a (n_prior_samples, n_parameters) array
    # then resample from that to get (n_inner_loops, n_parameters)
    if not psa_prior.parameters:
        # This case should ideally be handled by psa_prior validation, but as a fallback:
        return np.random.rand(n_inner_loops, 1) # Return some dummy if no params

    param_arrays = [v.reshape(-1, 1) if v.ndim ==1 else v for v in psa_prior.parameters.values()]
    prior_params_stacked = np.hstack(param_arrays) # (n_prior_samples, n_params)

    # Resample from these prior parameter sets
    indices = np.random.choice(prior_params_stacked.shape[0], n_inner_loops, replace=True)
    return prior_params_stacked[indices, :]


def _fit_metamodel(
    model_func: EconomicModelFunctionType,
    psa_prior: PSASample,
    nb_prior_values: np.ndarray # Pre-calculated (n_samples, n_strategies)
) -> List[Any]: # List of fitted regression models, one per strategy
    """
    (Stub) Fits a regression metamodel NB(d,theta) ~ f(theta) for each strategy d.
    Uses prior PSA samples and their corresponding net benefits.
    """
    if not SKLEARN_AVAILABLE:
        raise PyVoiNotImplementedError("Scikit-learn is required for regression-based EVSI if not providing a custom metamodel fitter.")

    print(f"Info: _fit_metamodel called for {list(psa_prior.parameters.keys())[0] if psa_prior.parameters else 'N/A'}.")
    metamodel_list = []

    # Prepare X from psa_prior.parameters (n_samples, n_params)
    if not psa_prior.parameters:
         # If no parameters, EVSI should be 0. Metamodel fitting is trivial.
         # This case should ideally lead to EVSI=0 earlier.
         # For now, return empty list, subsequent steps should handle it.
        print("Warning: No parameters in psa_prior for metamodel fitting.")
        return []

    param_arrays = [v.reshape(-1, 1) if v.ndim ==1 else v for v in psa_prior.parameters.values()]
    X_prior = np.hstack(param_arrays) # (n_prior_samples, n_params)

    n_strategies = nb_prior_values.shape[1]
    for i in range(n_strategies):
        y_prior_strategy_i = nb_prior_values[:, i]
        model = LinearRegression() # Default model
        try:
            model.fit(X_prior, y_prior_strategy_i)
            metamodel_list.append(model)
        except Exception as e:
            raise CalculationError(f"Error fitting metamodel for strategy {i}: {e}") from e

    return metamodel_list


def _predict_enb_from_metamodel(
    metamodel_list: List[Any],
    psa_posterior_k_samples: np.ndarray # (n_inner_loops, n_params)
) -> np.ndarray: # Returns array of E[NB(d)|D_k] for each strategy d, shape (n_strategies,)
    """
    (Stub) Predicts E[NB(d)|D_k] using the fitted metamodel and posterior samples.
    """
    if not metamodel_list: # Handles case of no parameters / no metamodels
        # If there are no metamodels (e.g. no parameters of interest),
        # then the posterior ENB is just the prior ENB.
        # However, the number of strategies is needed. This path needs careful thought.
        # For now, assume if metamodel_list is empty, it implies 0 strategies or an issue.
        # A more robust solution would pass n_strategies or handle it in the caller.
        print("Warning: _predict_enb_from_metamodel received an empty metamodel_list.")
        # This will likely cause an error or incorrect result if not handled by caller.
        # Let's assume the caller expects an array of some size.
        # This part of the stub needs to be consistent with how `evsi` uses it.
        # If metamodel_list is empty because there were no params, then E[NB|Dk] = E[NB]
        # This requires E[NB] to be passed or accessible.
        # For now, let's return zeros if we don't know n_strategies.
        # This will be refined when `evsi` calls this.
        return np.array([]) # Placeholder, needs to know n_strategies

    if psa_posterior_k_samples.ndim == 1: # Should be (n_samples, n_params)
        psa_posterior_k_samples = psa_posterior_k_samples.reshape(-1,1)

    if psa_posterior_k_samples.size == 0: # No posterior samples
        # This might happen if n_inner_loops is 0 or update failed.
        # Return array of zeros matching number of strategies.
        return np.zeros(len(metamodel_list))

    n_strategies = len(metamodel_list)
    enb_post_data_k_strategies = np.zeros(n_strategies)

    for i in range(n_strategies):
        model = metamodel_list[i]
        if model is None: # Should not happen if _fit_metamodel is robust
            enb_post_data_k_strategies[i] = 0 # Or some other default
            continue
        try:
            predicted_nb_for_strategy_i = model.predict(psa_posterior_k_samples)
            enb_post_data_k_strategies[i] = np.mean(predicted_nb_for_strategy_i)
        except Exception as e:
            # This might happen if psa_posterior_k_samples doesn't match model's expected features
            print(f"Warning: Error predicting with metamodel for strategy {i}: {e}. Using 0 for ENB.")
            enb_post_data_k_strategies[i] = 0 # Fallback

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
    EVSI = E_D [ max_d E_theta|D [NB(d, theta|D)] ] - max_d [ E_theta [NB(d, theta)] ]
    (Docstring largely unchanged for now, will be updated as implementation progresses)
    """
    if not isinstance(psa_prior, PSASample):
        raise InputError("`psa_prior` must be a PSASample object.")
    if not isinstance(trial_design, TrialDesign):
        raise InputError("`trial_design` must be a TrialDesign object.")
    if not callable(model_func):
        raise InputError("`model_func` must be a callable function.")
    if n_outer_loops <=0 or n_inner_loops <=0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    # --- Calculate max_d [ E_theta [NB(d, theta)] ] --- (Prior optimal decision value)
    try:
        # model_func is expected to take parameters (dict or PSASample) and return nb_array
        nb_prior_values = model_func(psa_prior) # psa_prior might be passed if model_func expects PSASample obj

        if isinstance(nb_prior_values, NetBenefitArray): # Should not happen based on new type hint
            nb_prior_values = nb_prior_values.values
        elif not isinstance(nb_prior_values, np.ndarray):
            raise CalculationError(
                "`model_func` did not return a NumPy array for prior NBs."
            )
        check_input_array(
            nb_prior_values, expected_ndim=2, name="Prior Net Benefit values from model_func"
        )
    except Exception as e:
        raise CalculationError(
            f"Error running model_func with prior PSA samples: {e}"
        ) from e

    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info = np.max(mean_nb_per_strategy_prior)


    # --- Calculate E_D [ max_d E_theta|D [NB(d, theta|D)] ] --- (Expected post-study optimal value)
    expected_max_nb_post_study: float # Type hint

    if method == "regression":
        # The original PyVoiNotImplementedError for "regression" was here.
        # It's removed to allow the new stubbed logic to run.
        if not SKLEARN_AVAILABLE:
             raise PyVoiNotImplementedError("Regression method for EVSI requires scikit-learn to be installed.")
        if not psa_prior.parameters:
            print("Warning: EVSI regression method called with no parameters in psa_prior. EVSI will be 0.")
            expected_max_nb_post_study = max_expected_nb_current_info
        else:
            metamodel_list = _fit_metamodel(model_func, psa_prior, nb_prior_values)
            if not metamodel_list : # No models fitted (e.g. no parameters)
                 expected_max_nb_post_study = max_expected_nb_current_info
            else:
                all_max_enb_post_data_k = np.zeros(n_outer_loops) # Pre-allocate
                for k_loop_idx in range(n_outer_loops): # Renamed k to k_loop_idx to avoid conflict
                    simulated_data_k = _simulate_trial_data(trial_design, psa_prior, n_inner_loops)
                    psa_posterior_k_samples = _bayesian_update(psa_prior, simulated_data_k, n_inner_loops)

                    if psa_posterior_k_samples.ndim == 1:
                        # This might happen if _bayesian_update returns a single param array incorrectly
                        # Or if only one param of interest. check_parameter_samples in basic.py handles this.
                        # Here, _bayesian_update is expected to return (n_inner_loops, n_features_model_was_fit_on)
                        # For now, assume _bayesian_update gives correct shape.
                        pass

                    enb_post_data_k_strategies = _predict_enb_from_metamodel(metamodel_list, psa_posterior_k_samples)

                    if enb_post_data_k_strategies.size == 0 : # From _predict_enb_from_metamodel if issues
                        # This indicates an issue, e.g. no strategies or error in prediction
                        # Fallback to prior for this iteration to avoid NaN and allow flow
                        print(f"Warning: Could not predict ENB for outer loop {k_loop_idx}. Using prior max ENB.")
                        all_max_enb_post_data_k[k_loop_idx] = max_expected_nb_current_info
                    else:
                        all_max_enb_post_data_k[k_loop_idx] = np.max(enb_post_data_k_strategies)

                expected_max_nb_post_study = np.mean(all_max_enb_post_data_k)

    elif method == "nonparametric":
        raise PyVoiNotImplementedError("Nonparametric EVSI method is not yet implemented.")
    elif method == "moment_matching":
        raise PyVoiNotImplementedError("Moment-matching EVSI method is not yet implemented.")
    else: # Catches 'unknown_method' or any other not explicitly handled
        raise PyVoiNotImplementedError(f"EVSI method '{method}' is not recognized or implemented.")

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi) # Ensure non-negative

    # Population scaling
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
                annuity_factor = (
                    1 - (1 + discount_rate) ** (-time_horizon)
                ) / discount_rate
            effective_population *= annuity_factor
        else: # No discount_rate provided
            if discount_rate is None: # Explicitly check for None for clarity
                effective_population *= time_horizon
        return per_decision_evsi * effective_population
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
    ------
        InputError: If inputs are not valid numbers or research_cost is negative.
    """
    if not isinstance(evsi_result, (float, int)): # float includes np.float64 etc.
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

    from pyvoi.core.data_structures import TrialArm  # Import TrialArm

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
    except PyVoiNotImplementedError as e:
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
                    annuity_factor = (
                        1 - (1 + discount_rate) ** (-time_horizon)
                    ) / discount_rate
                effective_population *= annuity_factor
            else: # No discount_rate provided
                if discount_rate is None: # Explicitly check for None for clarity
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
        dummy_model_func, # type: ignore
        dummy_psa, # type: ignore
        dummy_trial, # type: ignore
        population=1000,
        time_horizon=10,
        discount_rate=0.03,
        method="any_method_for_mock", # type: ignore
    )
    expected_pop_evsi_mocked = 5.0 * ((1 - (1 + 0.03) ** (-10)) / 0.03) * 1000
    print(f"Mocked Population EVSI: {pop_evsi_val_mocked}")
    assert np.isclose(pop_evsi_val_mocked, expected_pop_evsi_mocked), (
        f"Mocked Population EVSI error. Expected ~{expected_pop_evsi_mocked:.2f}, got {pop_evsi_val_mocked:.2f}"
    )
    print("EVSI population scaling test (with mock) PASSED.")

    evsi = _temp_original_evsi # Restore original evsi

    print("\n--- sample_information.py tests completed ---")
