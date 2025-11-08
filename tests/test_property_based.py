"""Property-based tests for voiage functions."""

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.schema import ParameterSet, ValueArray

# Strategy for generating valid net benefit arrays
net_benefit_arrays = arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=2, max_value=100),  # n_samples
        st.integers(min_value=2, max_value=5)     # n_strategies
    ),
    elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
)


@given(nb_array=net_benefit_arrays)
def test_evpi_non_negative(nb_array):
    """Test that EVPI is always non-negative."""
    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    assert evpi_result >= 0


@given(nb_array=net_benefit_arrays)
def test_evpi_bounded_by_max_strategy_evpi(nb_array):
    """Test that EVPI is bounded by the maximum possible EVPI for any single strategy."""
    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()

    # Maximum possible EVPI is the difference between max and min net benefits
    max_nb = np.max(nb_array)
    min_nb = np.min(nb_array)
    max_possible_evpi = max_nb - min_nb

    # Allow for small floating point errors
    assert evpi_result <= max_possible_evpi + 1e-10


@given(net_benefits=arrays(dtype=np.float64, shape=st.integers(min_value=2, max_value=100),
              elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)))
def test_evpi_single_strategy_zero(net_benefits):
    """Test that EVPI is zero for single strategy problems."""
    nb_array = net_benefits.reshape(-1, 1)
    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    assert evpi_result == 0


@given(nb_array=arrays(dtype=np.float64, shape=st.tuples(st.integers(min_value=2, max_value=50), st.integers(min_value=2, max_value=5)),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
def test_evpi_identical_strategies_zero(nb_array):
    """Test that EVPI is zero when all strategies have identical net benefits."""
    # Make all strategies identical
    n_strategies = nb_array.shape[1]
    identical_columns = [nb_array[:, 0] for _ in range(n_strategies)]
    identical_array = np.column_stack(identical_columns)
    value_array = ValueArray.from_numpy(identical_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    # Allow for small floating point errors
    assert abs(evpi_result) < 1e-10


@given(n_samples=st.integers(min_value=1, max_value=10000))
def test_evpi_population_scaling_properties(n_samples):
    """Test properties of EVPI population scaling."""
    # Create simple net benefit data
    strategy1 = np.full(n_samples, 100.0)
    strategy2 = np.full(n_samples, 110.0)
    nb_array = np.column_stack([strategy1, strategy2])

    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)

    # Base EVPI
    base_evpi = analysis.evpi()

    # Population-scaled EVPI
    pop_evpi = analysis.evpi(population=1000, time_horizon=10, discount_rate=0.03)

    # Population-scaled EVPI should be larger than base EVPI
    assert pop_evpi >= base_evpi


# Strategy for generating valid parameter arrays for EVPPI testing
parameter_arrays = arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=10, max_value=100),  # n_samples
        st.integers(min_value=1, max_value=5)      # n_parameters
    ),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)


@given(nb_array=arrays(dtype=np.float64, shape=st.tuples(st.integers(min_value=10, max_value=50), st.integers(min_value=2, max_value=5)),
       elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
def test_evppi_non_negative(nb_array):
    """Test that EVPPI is always non-negative."""
    # Generate parameter array inside the function
    n_samples = nb_array.shape[0]
    n_params = 3  # Fixed number of parameters for testing
    param_array = np.random.randn(n_samples, n_params)

    # Create parameter dictionary with matching number of samples
    trimmed_nb_array = nb_array[:n_samples, :]
    trimmed_param_array = param_array[:n_samples, :n_params]

    # Create parameter dictionary
    param_dict = {f"param_{i}": trimmed_param_array[:, i] for i in range(n_params)}

    # Create ParameterSet
    dataset = xr.Dataset(
        {k: ("n_samples", v) for k, v in param_dict.items()},
        coords={"n_samples": np.arange(n_samples)}
    )
    parameter_set = ParameterSet(dataset=dataset)

    value_array = ValueArray.from_numpy(trimmed_nb_array)

    analysis = DecisionAnalysis(trimmed_nb_array, parameter_set)
    evppi_result = analysis.evppi()
    assert evppi_result >= 0


@given(nb_array=arrays(dtype=np.float64, shape=st.tuples(st.integers(min_value=10, max_value=50), st.integers(min_value=1, max_value=1)),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
def test_evppi_single_strategy_zero(nb_array):
    """Test that EVPPI is zero for single strategy problems."""
    single_strategy_array = nb_array  # Only one strategy
    value_array = ValueArray.from_numpy(single_strategy_array)

    # Create parameter dictionary
    n_samples = single_strategy_array.shape[0]
    param_dict = {"param1": np.random.randn(n_samples)}

    # Create ParameterSet
    dataset = xr.Dataset(
        {k: ("n_samples", v) for k, v in param_dict.items()},
        coords={"n_samples": np.arange(n_samples)}
    )
    parameter_set = ParameterSet(dataset=dataset)

    analysis = DecisionAnalysis(value_array, parameter_set)
    evppi_result = analysis.evppi()
    # Allow for small floating point errors
    assert abs(evppi_result) < 1e-10
