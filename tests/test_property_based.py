"""Property-based tests for voiage functions."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.schema import ParameterSet, ValueArray


@st.composite
def _float_array(
    draw,
    *,
    min_rows: int,
    max_rows: int,
    min_cols: int,
    max_cols: int,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """Build a NumPy float array from plain Hypothesis strategies."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    values = draw(
        st.lists(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_rows * n_cols,
            max_size=n_rows * n_cols,
        )
    )
    return np.array(values, dtype=np.float64).reshape(n_rows, n_cols)


# Strategy for generating valid net benefit arrays
net_benefit_arrays = _float_array(
    min_rows=2,
    max_rows=100,
    min_cols=2,
    max_cols=5,
    min_value=-1000,
    max_value=1000,
)


@given(nb_array=net_benefit_arrays)
@settings(deadline=None)
def test_evpi_non_negative(nb_array) -> None:
    """Test that EVPI is always non-negative."""
    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    assert evpi_result >= 0


@given(nb_array=net_benefit_arrays)
@settings(deadline=None)
def test_evpi_bounded_by_max_strategy_evpi(nb_array) -> None:
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


@given(
    net_benefits=st.lists(
        st.floats(
            min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=100,
    )
)
@settings(deadline=None)
def test_evpi_single_strategy_zero(net_benefits) -> None:
    """Test that EVPI is zero for single strategy problems."""
    nb_array = np.array(net_benefits, dtype=np.float64).reshape(-1, 1)
    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    assert evpi_result == 0


@given(
    nb_array=_float_array(
        min_rows=2,
        max_rows=50,
        min_cols=2,
        max_cols=5,
        min_value=-100,
        max_value=100,
    )
)
@settings(deadline=None)
def test_evpi_identical_strategies_zero(nb_array) -> None:
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
@settings(deadline=None)
def test_evpi_population_scaling_properties(n_samples) -> None:
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
parameter_arrays = _float_array(
    min_rows=10,
    max_rows=100,
    min_cols=1,
    max_cols=5,
    min_value=-100,
    max_value=100,
)


@given(
    nb_array=_float_array(
        min_rows=10,
        max_rows=50,
        min_cols=2,
        max_cols=5,
        min_value=-100,
        max_value=100,
    )
)
@settings(deadline=None)
def test_evppi_non_negative(nb_array) -> None:
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
        coords={"n_samples": np.arange(n_samples)},
    )
    parameter_set = ParameterSet(dataset=dataset)

    _ = ValueArray.from_numpy(trimmed_nb_array)

    analysis = DecisionAnalysis(trimmed_nb_array, parameter_set)
    evppi_result = analysis.evppi()
    assert evppi_result >= 0


@given(
    nb_array=_float_array(
        min_rows=10,
        max_rows=50,
        min_cols=1,
        max_cols=1,
        min_value=-100,
        max_value=100,
    )
)
@settings(deadline=None)
def test_evppi_single_strategy_zero(nb_array) -> None:
    """Test that EVPPI is zero for single strategy problems."""
    single_strategy_array = nb_array  # Only one strategy
    value_array = ValueArray.from_numpy(single_strategy_array)

    # Create parameter dictionary
    n_samples = single_strategy_array.shape[0]
    param_dict = {"param1": np.random.randn(n_samples)}

    # Create ParameterSet
    dataset = xr.Dataset(
        {k: ("n_samples", v) for k, v in param_dict.items()},
        coords={"n_samples": np.arange(n_samples)},
    )
    parameter_set = ParameterSet(dataset=dataset)

    analysis = DecisionAnalysis(value_array, parameter_set)
    evppi_result = analysis.evppi()
    # Allow for small floating point errors
    assert abs(evppi_result) < 1e-10
