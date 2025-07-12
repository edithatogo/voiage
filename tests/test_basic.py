# tests/test_basic.py

"""Unit tests for the basic VOI methods (EVPI, EVPPI) in voiage.methods.basic."""

import numpy as np
import pytest

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import NetBenefitArray, PSASample
from voiage.exceptions import (
    DimensionMismatchError,
    InputError,
    OptionalDependencyError,
)
from voiage.methods.basic import evpi, evppi

# Import fixtures from conftest.py if needed, e.g.:
# from .conftest import sample_nb_data_array_2strat, sample_net_benefit_array_2strat


# --- Tests for EVPI ---


@pytest.mark.parametrize(
    ("nb_array", "expected_evpi"),
    [
        (np.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]]), 6.0),
        (np.array([[100, 100], [100, 100]]), 0.0),
    ],
)
def test_evpi_calculation_simple(nb_array, expected_evpi):
    """Test EVPI calculation with a simple, known scenario."""
    calculated_evpi = evpi(nb_array)
    np.testing.assert_allclose(calculated_evpi, expected_evpi)


def test_evpi_with_netbenefitarray_input(
    sample_net_benefit_array_2strat: NetBenefitArray,
):
    """Test EVPI with NetBenefitArray object as input."""
    expected_evpi = 6.0  # Same data as above
    calculated_evpi = evpi(sample_net_benefit_array_2strat)
    np.testing.assert_allclose(calculated_evpi, expected_evpi)


def test_evpi_single_strategy():
    """Test EVPI when there's only one strategy (should be 0)."""
    nb_single_strat = np.array([[100], [110], [90]], dtype=DEFAULT_DTYPE)
    calculated_evpi = evpi(nb_single_strat)
    np.testing.assert_allclose(calculated_evpi, 0.0)


def test_evpi_no_uncertainty():
    """Test EVPI when all samples for each strategy are identical (no uncertainty)."""
    # If NB for strategy A is always 100, and for B is always 90.
    # E[max(NB)] = max(100,90) = 100. max(E[NB_A], E[NB_B]) = max(100,90) = 100. EVPI = 0.
    nb_no_uncertainty = np.array([[100, 90], [100, 90], [100, 90]], dtype=DEFAULT_DTYPE)
    calculated_evpi = evpi(nb_no_uncertainty)
    np.testing.assert_allclose(calculated_evpi, 0.0)


def test_evpi_population_scaling(sample_nb_data_array_2strat: np.ndarray):
    """Test EVPI population scaling logic."""
    per_decision_evpi = 6.0  # From previous test
    population = 1000
    time_horizon = 10
    discount_rate = 0.03

    # Annuity factor = (1 - (1 + dr)^-th) / dr
    annuity = (1 - (1 + discount_rate) ** (-time_horizon)) / discount_rate
    expected_pop_evpi = per_decision_evpi * population * annuity

    calculated_pop_evpi = evpi(
        sample_nb_data_array_2strat,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )
    np.testing.assert_allclose(calculated_pop_evpi, expected_pop_evpi)

    # Test with no discount rate
    expected_pop_evpi_no_dr = per_decision_evpi * population * time_horizon
    calculated_pop_evpi_no_dr = evpi(
        sample_nb_data_array_2strat,
        population=population,
        time_horizon=time_horizon,
        # discount_rate=None # Default
    )
    np.testing.assert_allclose(calculated_pop_evpi_no_dr, expected_pop_evpi_no_dr)

    # Test with zero discount rate
    calculated_pop_evpi_zero_dr = evpi(
        sample_nb_data_array_2strat,
        population=population,
        time_horizon=time_horizon,
        discount_rate=0.0,
    )
    np.testing.assert_allclose(calculated_pop_evpi_zero_dr, expected_pop_evpi_no_dr)


def test_evpi_invalid_inputs():
    """Test EVPI with various invalid inputs."""
    with pytest.raises(InputError, match="must be a NumPy array or NetBenefitArray"):
        evpi("not an array")  # type: ignore

    with pytest.raises(
        DimensionMismatchError, match=r"must have \(2,\) dimension\(s\). Got 1."
    ):
        evpi(np.array([1, 2, 3], dtype=DEFAULT_DTYPE))  # 1D array

    np.testing.assert_allclose(
        evpi(np.array([[], []], dtype=DEFAULT_DTYPE).reshape(0, 2)), 0.0
    )  # Empty array

    np.testing.assert_allclose(
        evpi(np.array([[]], dtype=DEFAULT_DTYPE).reshape(0, 1)), 0.0
    )

    # Population scaling input errors
    nb_data = np.array([[10, 20], [11, 19]], dtype=DEFAULT_DTYPE)
    with pytest.raises(InputError, match="Population must be a positive number."):
        evpi(nb_data, population=0, time_horizon=5)
    with pytest.raises(InputError, match="Time horizon must be a positive number."):
        evpi(nb_data, population=100, time_horizon=-1)
    with pytest.raises(
        InputError, match="Discount rate must be a number between 0 and 1."
    ):
        evpi(nb_data, population=100, time_horizon=5, discount_rate=1.1)
    with pytest.raises(
        InputError,
        match="To calculate population EVPI, 'population' and 'time_horizon' must be provided",
    ):
        evpi(nb_data, population=100)  # Missing time_horizon
    with pytest.raises(
        InputError,
        match="To calculate population EVPI, 'population' and 'time_horizon' must be provided",
    ):
        evpi(nb_data, time_horizon=5)  # Missing population
    with pytest.raises(
        InputError,
        match="To calculate population EVPI, 'population' and 'time_horizon' must be provided",
    ):
        evpi(nb_data, discount_rate=0.05)  # Missing pop and horizon

    # Test with zero strategies
    np.testing.assert_allclose(
        evpi(np.array([[]], dtype=DEFAULT_DTYPE).reshape(1, 0)), 0.0
    )


# --- Tests for EVPPI ---
# EVPPI tests are more complex due to the regression step.
# We need scikit-learn, which might be an optional dependency.
# For now, focus on input validation and basic properties.

# Skip EVPPI tests if scikit-learn is not available
# The evppi function itself will try to import sklearn components.
# The SKLEARN_AVAILABLE flag can be set based on whether that import succeeds within the evppi function,
# or we can attempt a lightweight check here. For now, let's assume evppi handles its own import errors
# and tests will fail gracefully or be skipped if evppi cannot run.
# The primary F401 was about 'import sklearn' not being used in *this* file.
SKLEARN_AVAILABLE = True  # Assume available, let tests fail if not. More robust skipping might be needed later.
try:
    from sklearn.linear_model import (
        LinearRegression,
    )
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark_evppi = pytest.mark.skipif(
    not SKLEARN_AVAILABLE, reason="scikit-learn not available, skipping EVPPI tests"
)


@pytestmark_evppi
def test_evppi_basic_properties(evppi_test_data_simple):
    """Test basic properties of EVPPI."""
    # - EVPPI >= 0
    # - EVPPI <= EVPI
    # Uses the evppi_test_data_simple fixture.
    data = evppi_test_data_simple
    nb_values = data["nb_values"]
    p_samples = data["p_samples"]
    expected_evpi = data["expected_evpi_approx"]

    calculated_evppi = evppi(nb_values, p_samples)

    assert (
        calculated_evppi >= -1e-9
    ), f"EVPPI should be non-negative, got {calculated_evppi}."
    # Small tolerance for numerical precision
    assert (
        calculated_evppi <= expected_evpi + 1e-9
    ), f"EVPPI ({calculated_evppi}) should be less than or equal to EVPI ({expected_evpi})."


@pytestmark_evppi
def test_evppi_input_types(evppi_test_data_simple):
    """Test EVPPI with different input types for parameter_samples."""
    data = evppi_test_data_simple
    nb_values = data["nb_values"]
    p_samples_np = data["p_samples"]  # This is a 1D NumPy array

    # 1. NumPy array (1D, should be reshaped internally)
    evppi_np_1d = evppi(nb_values, p_samples_np)

    # 2. NumPy array (2D)
    p_samples_np_2d = p_samples_np.reshape(-1, 1)
    evppi_np_2d = evppi(nb_values, p_samples_np_2d)
    np.testing.assert_allclose(evppi_np_1d, evppi_np_2d)

    # 3. PSASample
    psa_obj = PSASample(parameters={"param_of_interest": p_samples_np})
    evppi_psa = evppi(nb_values, psa_obj)
    np.testing.assert_allclose(evppi_np_1d, evppi_psa, atol=1e-3)
    # Note: if n_regression_samples is used and is < total_samples, some minor variation is expected.

    # 4. Dictionary
    param_dict = {"param_of_interest": p_samples_np}
    evppi_dict = evppi(nb_values, param_dict)
    np.testing.assert_allclose(evppi_np_1d, evppi_dict, atol=1e-3)


@pytestmark_evppi
def test_evppi_population_scaling(evppi_test_data_simple):
    """Test EVPPI population scaling."""
    data = evppi_test_data_simple
    nb_values = data["nb_values"]
    p_samples = data["p_samples"]

    per_decision_evppi = evppi(nb_values, p_samples)

    population = 10000
    time_horizon = 20
    discount_rate = 0.035

    annuity = (1 - (1 + discount_rate) ** (-time_horizon)) / discount_rate
    expected_pop_evppi = per_decision_evppi * population * annuity

    calculated_pop_evppi = evppi(
        nb_values,
        p_samples,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )
    # Allow some tolerance due to regression step if subsampling is used
    np.testing.assert_allclose(
        calculated_pop_evppi, expected_pop_evppi, rtol=1e-2, atol=1e-3
    )


@pytestmark_evppi
def test_evppi_invalid_inputs(evppi_test_data_simple):
    """Test EVPPI with various invalid inputs."""
    data = evppi_test_data_simple
    nb_values = data["nb_values"]
    p_samples_valid = data["p_samples"]

    with pytest.raises(
        InputError, match="`nb_array` must be a NumPy array or NetBenefitArray object."
    ):
        evppi("not an array", p_samples_valid)  # type: ignore

    with pytest.raises(
        InputError,
        match=r"`parameter_samples` must be a NumPy array, PSASample, or Dict\. Got <class 'str'>\.",
    ):
        evppi(nb_values, "not valid params")  # type: ignore

    with pytest.raises(
        DimensionMismatchError,
        match=r"Number of samples in `parameter_samples` \(\d+\) does not match `nb_array` \(\d+\)\.",
    ):
        evppi(nb_values, p_samples_valid[:-1])  # Mismatched sample size

    with pytest.raises(
        InputError,
        match="n_regression_samples, if provided, must be a positive integer.",
    ):
        evppi(nb_values, p_samples_valid, n_regression_samples=0)

    with pytest.raises(
        InputError,
        match=r"n_regression_samples \(\d+\) cannot exceed total samples \(\d+\)\.",
    ):
        evppi(
            nb_values, p_samples_valid, n_regression_samples=len(p_samples_valid) + 10
        )

    # Population scaling errors (similar to EVPI)
    with pytest.raises(
        InputError,
        match="To calculate population EVPPI, 'population' and 'time_horizon' must be provided",
    ):
        evppi(nb_values, p_samples_valid, population=100)


def test_evppi_sklearn_unavailable(evppi_test_data_simple, monkeypatch):
    """Test that EVPPI raises OptionalDependencyError if scikit-learn is not available."""
    monkeypatch.setattr("voiage.methods.basic.SKLEARN_AVAILABLE", False)
    data = evppi_test_data_simple
    nb_values = data["nb_values"]
    p_samples_valid = data["p_samples"]

    with pytest.raises(OptionalDependencyError):
        evppi(nb_values, p_samples_valid)


@pytestmark_evppi
def test_evppi_perfect_parameter(evppi_test_data_simple):
    """Test EVPPI when the parameter of interest perfectly explains decision uncertainty."""
    # In this case, EVPPI should be very close to EVPI.
    # This requires constructing a specific scenario.
    np.random.seed(123)
    n_s = 1000
    # Parameter p perfectly determines which strategy is better
    # If p > 0, strat1 is better. If p < 0, strat2 is better.
    p_perfect = np.random.normal(0, 1, n_s)

    # NB_strat1 = p_perfect * 10  (e.g. 10 if p=1, -10 if p=-1)
    # NB_strat2 = 0 (fixed)
    # Decision rule: pick strat1 if p > 0, else strat2.
    # NB_optimal_if_p_known = max(p_perfect*10, 0)

    nb_s1 = p_perfect * 10
    nb_s2 = np.zeros(n_s)
    nb_vals_perfect = np.stack([nb_s1, nb_s2], axis=1).astype(DEFAULT_DTYPE)

    evpi_val = evpi(nb_vals_perfect)
    evppi_val = evppi(nb_vals_perfect, p_perfect)

    # print(f"Perfect Param Test: EVPI={evpi_val:.4f}, EVPPI={evppi_val:.4f}")
    np.testing.assert_allclose(evppi_val, evpi_val, rtol=0.1, atol=0.01)
    # Regression won't be perfect, so rtol needs to be somewhat tolerant.


@pytestmark_evppi
def test_evppi_irrelevant_parameter(evppi_test_data_simple):
    """Test EVPPI when the parameter of interest is irrelevant to decision uncertainty."""
    # EVPPI should be close to 0.
    np.random.seed(321)
    n_s = 1000
    # True decision based on some hidden parameter `h`
    h_true = np.random.normal(0, 1, n_s)
    nb_s1 = h_true * 10
    nb_s2 = np.zeros(n_s)
    nb_vals_irrelevant = np.stack([nb_s1, nb_s2], axis=1).astype(DEFAULT_DTYPE)

    # Parameter of interest `p_irr` is pure noise, unrelated to h_true or NBs
    p_irrelevant = np.random.normal(100, 50, n_s)

    evppi_val = evppi(nb_vals_irrelevant, p_irrelevant)
    # print(f"Irrelevant Param Test: EVPPI={evppi_val:.4f}")
    np.testing.assert_allclose(evppi_val, 0.0, atol=0.1)
    # atol might need adjustment based on regression stability and n_samples.
    # If regression overfits noise, it might find spurious correlation.


# TODO: Add tests for EVPPI with multiple parameters of interest if the fixture supports it
# or create a new fixture. The evppi_test_data_simple is for single param.

# Example of a test that might use Hypothesis for property-based testing
# from hypothesis import given, strategies as st
# @given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=2, max_size=10),
#        st.lists(st.floats(min_value=0, max_value=1e6), min_size=2, max_size=10))
# def test_evpi_properties_hypothesis(nb_list1, nb_list2):
#     # This is a very basic example, would need more robust strategy generation
#     # for realistic net benefit arrays.
#     if len(nb_list1) != len(nb_list2) or len(nb_list1) < 2:
#         pytest.skip("Requires at least 2 samples and equal length lists for this simple setup.")
#
#     nb_data = np.array([nb_list1, nb_list2], dtype=DEFAULT_DTYPE).T # (samples, 2 strategies)
#     if nb_data.shape[0] < 2 : pytest.skip("Need at least 2 samples after transpose.")
#
#     val = evpi(nb_data)
#     assert val >= 0.0
#     # Could add more properties, e.g., if one strategy always dominates, EVPI is 0.
#     # max_val = np.max(nb_data) # Max possible value across all samples/strategies
#     # assert val <= max_val # EVPI cannot exceed the max possible gain

if __name__ == "__main__":
    # This allows running tests with `python tests/test_basic.py`
    # Useful for debugging individual test files.
    pytest.main([__file__])
