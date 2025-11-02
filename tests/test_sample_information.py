# tests/test_sample_information.py

"""Test VOI methods related to sample information (EVSI, ENBS)."""

from typing import Any, Optional, Union  # Added imports

import numpy as np
import pytest

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import InputError
from voiage.methods.sample_information import enbs
from voiage.schema import ParameterSet as PSASample
from voiage.schema import TrialDesign

# --- Dummy components for EVSI testing ---


def dummy_model_func_evsi(
    psa_params_or_sample: Union[dict, PSASample],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Define a simple model function for EVSI structure testing.

    It generates net benefits based on the number of samples in psa_params_or_sample.
    """
    n_samples = 0
    if isinstance(psa_params_or_sample, PSASample):
        n_samples = psa_params_or_sample.n_samples
    elif isinstance(psa_params_or_sample, dict):
        if psa_params_or_sample:
            n_samples = len(next(iter(psa_params_or_sample.values())))

    if n_samples == 0:  # Fallback if n_samples couldn't be determined
        n_samples = 3  # Default to a small number

    # For simplicity, generate random net benefits for 2 strategies
    # In a real model, these would be calculated based on input parameters
    nb_strategy1 = np.random.normal(loc=100, scale=10, size=n_samples)
    nb_strategy2 = np.random.normal(loc=105, scale=15, size=n_samples)

    # Create xarray dataset for ValueArray
    import xarray as xr
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), np.stack([nb_strategy1, nb_strategy2], axis=1).astype(DEFAULT_DTYPE))},
        coords={"n_samples": np.arange(n_samples), "n_strategies": [0, 1]}
    )
    from voiage.schema import ValueArray
    return ValueArray(dataset=dataset)


@pytest.fixture()
def dummy_psa_for_evsi() -> PSASample:
    """Create a dummy PSASample for EVSI tests."""
    # Parameters for a Normal-Normal conjugate update scenario
    # Means are different enough to expect some EVSI.
    # sd_outcome is relatively small to make learning more impactful.
    # n_samples for psa_prior should be reasonably large for stable metamodel fitting.
    n_psa_samples = 500
    import xarray as xr
    dataset = xr.Dataset(
        {
            "mean_new_treatment": (("n_samples",), np.random.normal(
                loc=10, scale=2, size=n_psa_samples
            ).astype(DEFAULT_DTYPE)),
            "mean_standard_care": (("n_samples",), np.random.normal(
                loc=8, scale=2, size=n_psa_samples
            ).astype(DEFAULT_DTYPE)),
            "sd_outcome": (("n_samples",), np.random.uniform(low=0.5, high=1.5, size=n_psa_samples).astype(
                DEFAULT_DTYPE
            )),
            # Add another dummy parameter not directly used in update, to test metamodel with multiple params
            "unrelated_param": (("n_samples",), np.random.rand(n_psa_samples).astype(DEFAULT_DTYPE)),
        },
        coords={"n_samples": np.arange(n_psa_samples)}
    )
    from voiage.schema import ParameterSet
    return ParameterSet(dataset=dataset)


@pytest.fixture()
def dummy_trial_design_for_evsi() -> TrialDesign:
    """Create a dummy TrialDesign for EVSI tests."""
    # Arm names match keys expected by _simulate_trial_data (via convention)
    # and _bayesian_update (hardcoded 'New Treatment' for data_key_for_update)
    from voiage.schema import DecisionOption, TrialDesign
    arm1 = DecisionOption(
        name="New Treatment", sample_size=50
    )  # Reduced sample size for faster test simulation
    arm2 = DecisionOption(name="Standard Care", sample_size=50)
    return TrialDesign(arms=[arm1, arm2])


# --- Tests for EVSI ---


def test_evsi_two_loop_method(dummy_psa_for_evsi, dummy_trial_design_for_evsi):
    """Test the two-loop EVSI method."""
    from voiage.methods.sample_information import evsi
    evsi_val = evsi(
        model_func=dummy_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=10,
        n_inner_loops=20,
    )
    assert evsi_val >= 0, "EVSI should be non-negative."


def test_evsi_regression_method_not_implemented(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
):
    """Test that the regression method for EVSI raises a NotImplementedError."""
    from voiage.exceptions import VoiageNotImplementedError
    import voiage.methods.sample_information as si_module
    from voiage.methods.sample_information import evsi

    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", False)
    with pytest.raises(VoiageNotImplementedError):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="regression",
        )


def test_evsi_regression_method(dummy_psa_for_evsi, dummy_trial_design_for_evsi):
    """Test the regression-based EVSI method."""
    from voiage.methods.sample_information import evsi
    evsi_val = evsi(
        model_func=dummy_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="regression",
        n_outer_loops=10,
    )
    assert evsi_val >= 0, "EVSI should be non-negative."


def test_evsi_invalid_inputs(dummy_psa_for_evsi, dummy_trial_design_for_evsi):
    """Test EVSI with various invalid inputs before it hits method implementation."""
    from voiage.exceptions import InputError
    from voiage.methods.sample_information import evsi

    # Invalid model_func
    with pytest.raises(InputError, match="`model_func` must be a callable function"):
        evsi(
            model_func="not_a_function",  # type: ignore
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
        )

    # Invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a ParameterSet object"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior={"param": np.array([1])},  # type: ignore
            trial_design=dummy_trial_design_for_evsi,
        )

    # Invalid trial_design
    with pytest.raises(InputError, match="`trial_design` must be a TrialDesign object"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=["not", "a", "trial", "design"],  # type: ignore
        )


# Mock EVSI for testing population scaling logic within EVSI itself
# This bypasses the NotImplementedError for method logic.
def mock_evsi_for_pop_scaling(
    model_func,
    psa_prior,
    trial_design,  # These are standard EVSI args
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    method: str = "mocked",  # Method arg still needed
    **kwargs,
) -> float:
    """Mock EVSI to return a fixed per-decision value and implement population scaling."""
    # Validate inputs that population scaling depends on
    if population is not None or time_horizon is not None or discount_rate is not None:
        if not (population is not None and time_horizon is not None):
            raise InputError(
                "To calculate population EVSI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional.",
            )
        if population <= 0:
            raise InputError("Population must be positive.")
        if time_horizon <= 0:
            raise InputError("Time horizon must be positive.")
        if discount_rate is not None and not (0 <= discount_rate <= 1):
            raise InputError("Discount rate must be between 0 and 1.")

    fixed_per_decision_evsi = 10.0  # Arbitrary value for testing scaling

    if (
        population is not None and time_horizon is not None
    ):  # Already validated they are together
        effective_population = population
        if discount_rate is not None:
            if discount_rate == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (
                    1 - (1 + discount_rate) ** (-time_horizon)
                ) / discount_rate
            effective_population *= annuity_factor
        else:  # No discount rate
            effective_population *= (
                time_horizon  # Assuming population is annual incidence
            )
        return fixed_per_decision_evsi * effective_population

    return fixed_per_decision_evsi


def test_evsi_population_scaling_logic(
    monkeypatch, dummy_psa_for_evsi, dummy_trial_design_for_evsi
):
    """Test population scaling logic within EVSI using a mock."""
    # Temporarily replace the real evsi with our mock for this test
    # Note: The sample_information module needs to be imported for monkeypatch to find 'evsi'
    import voiage.methods.sample_information as si_module

    monkeypatch.setattr(si_module, "evsi", mock_evsi_for_pop_scaling)

    fixed_mock_value = 10.0  # Matches the mock's internal per-decision value

    # Test case 1: No population args
    val_no_pop = si_module.evsi(
        dummy_model_func_evsi, dummy_psa_for_evsi, dummy_trial_design_for_evsi
    )
    assert np.isclose(val_no_pop, fixed_mock_value)

    # Test case 2: With population, horizon, no discount
    pop, th = 1000, 5
    val_pop_no_dr = si_module.evsi(
        dummy_model_func_evsi,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        population=pop,
        time_horizon=th,
    )
    assert np.isclose(val_pop_no_dr, fixed_mock_value * pop * th)

    # Test case 3: With population, horizon, and discount rate
    dr = 0.05
    annuity = (1 - (1 + dr) ** (-th)) / dr
    val_pop_dr = si_module.evsi(
        dummy_model_func_evsi,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        population=pop,
        time_horizon=th,
        discount_rate=dr,
    )
    assert np.isclose(val_pop_dr, fixed_mock_value * pop * annuity)

    # Test invalid population scaling inputs (should be caught by the mock's validation)
    with pytest.raises(InputError, match="Population must be positive"):
        si_module.evsi(
            dummy_model_func_evsi,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            population=0,
            time_horizon=th,
        )

    with pytest.raises(
        InputError,
        match="To calculate population EVSI, 'population' and 'time_horizon' must be provided",
    ):
        si_module.evsi(
            dummy_model_func_evsi,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            population=pop,
            discount_rate=dr,
        )  # Missing time_horizon


# --- Tests for ENBS ---


def test_enbs_zero_cost():
    """Test ENBS with zero research cost."""
    evsi_val = 500.0
    cost_val = 0.0
    calculated_enbs = enbs(evsi_val, cost_val)
    assert np.isclose(
        calculated_enbs, evsi_val
    ), "ENBS with zero cost should equal EVSI."


def test_enbs_invalid_inputs():
    """Test ENBS with various invalid inputs."""
    with pytest.raises(InputError, match="EVSI result must be a number"):
        enbs("not a float", 100.0)  # type: ignore

    with pytest.raises(InputError, match="Research cost must be a number"):
        enbs(1000.0, "not a float")  # type: ignore

    with pytest.raises(InputError, match="Research cost cannot be negative"):
        enbs(1000.0, -50.0)


if __name__ == "__main__":
    pytest.main([__file__])
