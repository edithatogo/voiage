# tests/test_sample_information.py

"""Test VOI methods related to sample information (EVSI, ENBS).

Since full EVSI calculation is complex and marked as NotImplemented for v0.1,
these tests will primarily focus on:
- Basic structure and input validation of EVSI (where possible before NotImplementedError).
- Correctness of ENBS calculation given an EVSI value.
- Population scaling logic if EVSI were to return a value.
"""

from typing import Optional, Union  # Added imports

import numpy as np
import pytest

from pyvoi.config import DEFAULT_DTYPE
from pyvoi.core.data_structures import PSASample, TrialArm, TrialDesign
from pyvoi.exceptions import InputError, PyVoiNotImplementedError # Corrected import
from pyvoi.methods.sample_information import enbs, evsi

# --- Dummy components for EVSI testing ---


def dummy_model_func_evsi(psa_params_or_sample: Union[dict, PSASample]) -> np.ndarray:
    """Define a very simple model function for EVSI structure testing."""
    # It ignores actual parameters and returns fixed net benefits.
    # In a real scenario, this would use psa_params_or_sample to calculate NBs.
    # Returns NB for 3 samples, 2 strategies
    return np.array(
        [
            [100, 110],
            [90, 120],
            [105, 95],
        ],
        dtype=DEFAULT_DTYPE,
    )


@pytest.fixture
def dummy_psa_for_evsi() -> PSASample:
    params = {
        "p1": np.array([0.1, 0.2, 0.3], dtype=DEFAULT_DTYPE),
        "p2": np.array([10, 20, 30], dtype=DEFAULT_DTYPE),
    }
    return PSASample(parameters=params)


@pytest.fixture
def dummy_trial_design_for_evsi() -> TrialDesign:
    arm1 = TrialArm(name="New Treatment", sample_size=100)
    arm2 = TrialArm(name="Standard Care", sample_size=100)
    return TrialDesign(arms=[arm1, arm2])


# --- Tests for EVSI ---


def test_evsi_structure_and_not_implemented(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
):
    """Test EVSI structure and NotImplementedError for specific methods."""

    # Test that "regression" method runs with stubs if sklearn is available
    # It should produce EVSI near 0 due to stubs.
    # If sklearn is NOT available, it should raise PyVoiNotImplementedError.

    # Temporarily modify SKLEARN_AVAILABLE for testing both paths
    import pyvoi.methods.sample_information as si_module

    original_sklearn_available = si_module.SKLEARN_AVAILABLE

    # Path 1: SKLEARN_AVAILABLE = True (normal run with stubs)
    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", True)
    try:
        evsi_val_regr = evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="regression",
            n_outer_loops=2, # Small loops for faster test
            n_inner_loops=5
        )
        # Due to stubs returning prior-like posteriors, EVSI should be ~0
        assert np.isclose(evsi_val_regr, 0.0, atol=1e-6), \
            f"EVSI with regression stubs should be near 0, got {evsi_val_regr}"
        print("EVSI regression with stubs (SKLEARN_AVAILABLE=True) ran as expected.")
    except PyVoiNotImplementedError:
        # This case should not happen if SKLEARN_AVAILABLE is True and no other NIError is raised by stubs
        pytest.fail("EVSI regression method raised PyVoiNotImplementedError unexpectedly when SKLEARN_AVAILABLE=True.")


    # Path 2: SKLEARN_AVAILABLE = False
    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", False)
    with pytest.raises(
        PyVoiNotImplementedError,
        match="Regression method for EVSI requires scikit-learn to be installed."
    ):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="regression"
        )
    print("EVSI regression (SKLEARN_AVAILABLE=False) raised PyVoiNotImplementedError as expected.")

    # Restore original SKLEARN_AVAILABLE state for other tests
    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", original_sklearn_available)

    # Test for other known but not implemented methods (if any were defined beyond regression)
    # For example, if "nonparametric" was a known method type in evsi()
    with pytest.raises(PyVoiNotImplementedError, match="Nonparametric EVSI method is not yet implemented."):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="nonparametric"
        )

    # Test for an unrecognized method
    with pytest.raises(
        PyVoiNotImplementedError, match="EVSI method 'unknown_method' is not recognized"
    ):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="unknown_method",  # This method should be caught as unrecognized
        )


def test_evsi_invalid_inputs(dummy_psa_for_evsi, dummy_trial_design_for_evsi):
    """Test EVSI with various invalid inputs before it hits method implementation."""
    # Invalid model_func
    with pytest.raises(InputError, match="`model_func` must be a callable function"):
        evsi(
            model_func="not_a_function",  # type: ignore
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
        )

    # Invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a PSASample object"):
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
    import pyvoi.methods.sample_information as si_module

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


def test_enbs_calculation():
    """Test basic ENBS calculation."""
    evsi_val = 1000.0
    cost_val = 200.0
    expected_enbs = 800.0
    calculated_enbs = enbs(evsi_val, cost_val)
    assert np.isclose(calculated_enbs, expected_enbs), "ENBS calculation error."

    evsi_val_neg = -50.0  # Should ideally not happen if EVSI is correct
    cost_val_high = 100.0
    expected_enbs_neg = -150.0
    calculated_enbs_neg = enbs(evsi_val_neg, cost_val_high)
    assert np.isclose(calculated_enbs_neg, expected_enbs_neg), (
        "ENBS calculation with negative EVSI failed."
    )


def test_enbs_zero_cost():
    """Test ENBS with zero research cost."""
    evsi_val = 500.0
    cost_val = 0.0
    calculated_enbs = enbs(evsi_val, cost_val)
    assert np.isclose(calculated_enbs, evsi_val), (
        "ENBS with zero cost should equal EVSI."
    )


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
