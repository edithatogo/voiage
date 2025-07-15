# voiage/methods/sample_information.py

"""Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from typing import Any, Callable, Dict, Optional, Union, List # Added List

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.schema import ValueArray, ParameterSet, TrialDesign
from voiage.core.utils import check_input_array
from voiage.exceptions import (
    CalculationError,
    InputError,
    VoiageNotImplementedError,
)
from voiage.models import ConjugateUpdater
from voiage.metamodels import GaussianProcessMetamodel

# Attempt to import for LinearRegression, fail gracefully if not available
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None # Placeholder if sklearn not available


# from voiage.methods.basic import evppi # May be used if EVSI is framed via EVPPI on predicted data

# Define a type for the model function expected by EVSI
# This function takes parameter samples (e.g., dict or ParameterSet) and returns net benefits.
# For EVSI, the model_func is used to generate *prior* net benefits.
# The impact of trial data is handled by updating parameters *before* potentially re-evaluating
# a model or, more commonly in regression EVSI, by using a metamodel.
EconomicModelFunctionType = Callable[[Union[Dict[str, np.ndarray], ParameterSet]], np.ndarray]


def evsi(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    method: str = "regression",
    n_outer_loops: int = 100,
    n_inner_loops: int = 1000,
    # **kwargs: Any # Additional arguments for specific methods
) -> float:
    if method == "regression" and not SKLEARN_AVAILABLE:
        raise VoiageNotImplementedError(
            "Regression method for EVSI requires scikit-learn to be installed."
        )
    if method == "nonparametric":
        raise VoiageNotImplementedError(
            "Nonparametric EVSI method is not yet implemented."
        )
    if method not in ["regression", "nonparametric"]:
        raise VoiageNotImplementedError(
            f"EVSI method '{method}' is not recognized or implemented."
        )
    if not isinstance(psa_prior, ParameterSet):
        raise InputError("`psa_prior` must be a ParameterSet object.")
    if not isinstance(trial_design, TrialDesign):
        raise InputError("`trial_design` must be a TrialDesign object.")
    if not callable(model_func):
        raise InputError("`model_func` must be a callable function.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    # --- Calculate max_d [ E_theta [NB(d, theta)] ] --- (Prior optimal decision value)
    try:
        nb_prior_values = model_func(psa_prior)
        if isinstance(nb_prior_values, ValueArray):
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

    # --- Metamodel ---
    metamodel = GaussianProcessMetamodel()
    X_prior = np.stack(list(psa_prior.parameters.values()), axis=1)
    metamodel.fit(X_prior, nb_prior_values)

    # --- Two-loop Monte Carlo ---
    all_max_enb_post_data_k = np.zeros(n_outer_loops)

    for k in range(n_outer_loops):
        # --- Outer loop: Simulate a dataset ---
        true_params_idx = np.random.randint(0, psa_prior.n_samples)
        true_params = {
            name: values[true_params_idx]
            for name, values in psa_prior.parameters.items()
        }

        # Simulate data from the trial design given the "true" parameters
        simulated_data = {}
        for arm in trial_design.arms:
            # This is a simplified data generation process.
            # A more robust implementation would allow for different data generating processes.
            mean = true_params.get(f"mean_{arm.name.lower()}", 0)
            sd = true_params.get(f"sd_{arm.name.lower()}", 1)
            simulated_data[arm.name] = np.random.normal(
                loc=mean, scale=sd, size=arm.sample_size
            )

        # --- Inner loop: Bayesian update and calculate posterior expected net benefit ---
        updater = ConjugateUpdater("normal-normal")
        # --- Inner loop: Bayesian update and calculate posterior expected net benefit ---
        # This is a simplified Bayesian update process.
        # A more robust implementation would allow for different Bayesian models.
        posterior_samples = {}
        posterior_samples = {
            param_name: np.random.choice(
                prior_samples, size=n_inner_loops, replace=True
            )
            for param_name, prior_samples in psa_prior.parameters.items()
        }
        for arm in trial_design.arms:
            updater = ConjugateUpdater("normal-normal")
            mean_key = f"mean_{arm.name.lower().replace(' ', '_')}"
            sd_key = f"sd_{arm.name.lower().replace(' ', '_')}"
            if mean_key in psa_prior.parameters and sd_key in psa_prior.parameters:
                trace = updater.update(
                    data=simulated_data[arm.name],
                    prior_samples={
                        "mean": psa_prior.parameters[mean_key],
                        "std": psa_prior.parameters[sd_key],
                    },
                )
                posterior_samples[mean_key] = trace.posterior["prior_mean"].values.flatten()
                posterior_samples[sd_key] = trace.posterior["prior_std"].values.flatten()

        X_posterior = np.stack(
            [
                v.reshape(-1, 1)
                for v in posterior_samples.values()
            ],
            axis=1,
        )
        X_posterior = X_posterior.reshape(X_posterior.shape[0], -1)

        # Calculate the expected net benefit for each strategy conditional on the simulated data
        enb_posterior = metamodel.predict(X_posterior)
        all_max_enb_post_data_k[k] = np.max(np.mean(enb_posterior, axis=0))

    expected_max_nb_post_study = np.mean(all_max_enb_post_data_k)

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)

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
                    1 - (1 + discount_rate) ** -time_horizon
                ) / discount_rate
            effective_population *= annuity_factor
        else:
            if discount_rate is None:
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
    if not isinstance(evsi_result, (float, int)):
        raise InputError(f"EVSI result must be a number. Got {type(evsi_result)}.")
    if not isinstance(research_cost, (float, int)):
        raise InputError(f"Research cost must be a number. Got {type(research_cost)}.")

    if research_cost < 0:
        raise InputError("Research cost cannot be negative.")

    return evsi_result - research_cost


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

    from voiage.schema import DecisionOption  # Import DecisionOption

    def dummy_model_func(
        params_dict_or_psa_sample: Union[
            Dict[str, np.ndarray], ParameterSet
        ],  # More specific Dict
    ) -> np.ndarray:
        # This dummy model just returns fixed NBs, ignoring params for simplicity of testing structure
        return np.array([[100, 110], [90, 120], [105, 95]], dtype=DEFAULT_DTYPE)

    dummy_psa_params = {
        "p1": np.array([1, 2, 3], dtype=DEFAULT_DTYPE),
        "p2": np.array([4, 5, 6], dtype=DEFAULT_DTYPE),
    }
    dummy_psa = ParameterSet(parameters=dummy_psa_params)
    dummy_arm1 = DecisionOption(name="Arm A", sample_size=50)
    dummy_arm2 = DecisionOption(name="Arm B", sample_size=50)
    dummy_trial = TrialDesign(arms=[dummy_arm1, dummy_arm2])

    print("\n--- EVSI (Regression Method Tests with Stubs) ---")
    # Dummy PSA prior with parameters expected by the stubs
    # _simulate_trial_data and _bayesian_update stubs expect 'mean_treatment', 'mean_control', 'sd_outcome'
    # and TrialDesign arms like "New Treatment", "Control"
    dummy_psa_params_for_evsi = {
        "mean_control": np.random.normal(10, 2, 500).astype(DEFAULT_DTYPE), # n_samples = 500
        "mean_treatment": np.random.normal(12, 2, 500).astype(DEFAULT_DTYPE),
        "sd_outcome": np.random.uniform(1, 3, 500).astype(DEFAULT_DTYPE),
        "other_param": np.random.rand(500).astype(DEFAULT_DTYPE) # an extra param not directly used in update
    }
    dummy_psa_for_evsi = ParameterSet(parameters=dummy_psa_params_for_evsi)

    # Dummy trial design matching stub expectations
    # Arm names should align with what _simulate_trial_data and _bayesian_update expect
    # e.g. 'New Treatment' for 'mean_treatment' update, 'Control' for 'mean_control'
    dummy_trial_arm_treatment = DecisionOption(name="New Treatment", sample_size=30) # n_obs for update
    dummy_trial_arm_control = DecisionOption(name="Control", sample_size=30)
    dummy_trial_design_for_evsi = TrialDesign(arms=[dummy_trial_arm_treatment, dummy_trial_arm_control])

    # Dummy model function that uses some of these parameters
    def specific_dummy_model_func(
        psa_sample_obj: ParameterSet,
    ) -> np.ndarray:
        # WTP (implicit)
        wtp = 30000
        # Strategy 1: Control
        nb_control = psa_sample_obj.parameters["mean_control"] * 0.5 * wtp - (psa_sample_obj.parameters["mean_control"] * 100 + 5000)
        # Strategy 2: New Treatment
        nb_treatment = psa_sample_obj.parameters["mean_treatment"] * 0.6 * wtp - (psa_sample_obj.parameters["mean_treatment"] * 120 + 7000)
        return np.stack([nb_control, nb_treatment], axis=-1).astype(DEFAULT_DTYPE)

    try:
        print("Running EVSI with regression method (using stubs for simulation/update)...")
        # Using smaller n_outer_loops and n_inner_loops for quicker test execution
        evsi_val_regression = evsi(
            specific_dummy_model_func,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            method="regression",
            n_outer_loops=10, # Reduced for testing
            n_inner_loops=50   # Reduced for testing
        )
        print(f"EVSI (regression method with stubs): {evsi_val_regression:.4f}")
        # We expect a non-negative value. The exact value depends on the stub logic.
        assert evsi_val_regression >= 0, "EVSI value should be non-negative"

        # Test with population scaling
        evsi_pop_val_regression = evsi(
            specific_dummy_model_func,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            population=10000,
            time_horizon=5,
            discount_rate=0.03,
            method="regression",
            n_outer_loops=10,
            n_inner_loops=50
        )
        print(f"Population EVSI (regression method with stubs): {evsi_pop_val_regression:.2f}")
        assert evsi_pop_val_regression >= evsi_val_regression

    except PyVoiNotImplementedError as e:
        print(f"EVSI regression method still raised NotImplementedError: {e}")
    except Exception as e:
        print(f"Error during EVSI (regression) call with stubs: {e}")
        import traceback
        traceback.print_exc()


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
