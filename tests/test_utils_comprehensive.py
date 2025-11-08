"""Comprehensive tests for voiage.core.utils module to improve coverage."""

import numpy as np
import pytest

from voiage.core.utils import (
    calculate_net_benefit,
    check_input_array,
    get_optimal_strategy_index,
)
from voiage.exceptions import DimensionMismatchError, InputError
from voiage.schema import ValueArray


class TestUtilsComplete:
    """Comprehensive tests for utils module functions."""

    def test_check_input_array_valid(self):
        """Test check_input_array with valid inputs."""
        # Test 2D array with expected dimensions
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        check_input_array(arr_2d, expected_ndim=2, name="test_array")

        # Test with expected shape
        check_input_array(arr_2d, expected_ndim=2, expected_shape=(2, 2))

        # Test with expected dtype
        check_input_array(arr_2d, expected_ndim=2, expected_dtype=arr_2d.dtype)

        # Test with sequence of accepted dimensions
        check_input_array(arr_2d, expected_ndim=[1, 2, 3])

    def test_check_input_array_invalid_types(self):
        """Test check_input_array with invalid types."""
        # Test with non-array input
        with pytest.raises(InputError, match="must be a NumPy array"):
            check_input_array("not_an_array", expected_ndim=2)

        with pytest.raises(InputError, match="must be a NumPy array"):
            check_input_array([1, 2, 3], expected_ndim=1)

        with pytest.raises(InputError, match="must be a NumPy array"):
            check_input_array(123, expected_ndim=1)

    def test_check_input_array_invalid_dimensions(self):
        """Test check_input_array with invalid dimensions."""
        # Test with wrong number of dimensions
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(DimensionMismatchError, match="must have.*dimension"):
            check_input_array(arr_2d, expected_ndim=1)

        # Test with sequence of dimensions where none match
        with pytest.raises(DimensionMismatchError, match="must have.*dimension"):
            check_input_array(arr_2d, expected_ndim=[1, 3])

    def test_check_input_array_empty_array(self):
        """Test check_input_array with empty array."""
        # Test with empty array when not allowed
        empty_arr = np.array([])

        with pytest.raises(InputError, match="cannot be empty"):
            check_input_array(empty_arr, expected_ndim=1, allow_empty=False)

        # Test with empty array when allowed
        check_input_array(empty_arr, expected_ndim=1, allow_empty=True)

    def test_check_input_array_invalid_dtype(self):
        """Test check_input_array with invalid dtype."""
        # Create an array with specific dtype
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Integer array

        with pytest.raises(InputError, match="must have dtype"):
            # Try to validate with float64 dtype expectation
            check_input_array(arr, expected_ndim=2, expected_dtype=np.float64)

    def test_check_input_array_valid_dtype(self):
        """Test check_input_array with valid dtype."""
        # Create array with expected dtype
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        check_input_array(arr, expected_ndim=2, expected_dtype=np.float64)

    def test_check_input_array_with_any_dtype(self):
        """Test check_input_array with 'any' dtype."""
        # Test with 'any' dtype (this should work regardless of actual dtype)
        arr_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        arr_float = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        check_input_array(arr_int, expected_ndim=2, expected_dtype="any")
        check_input_array(arr_float, expected_ndim=2, expected_dtype="any")

    def test_check_input_array_invalid_shape(self):
        """Test check_input_array with invalid shape."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)

        # Test with mismatched expected shape
        with pytest.raises(DimensionMismatchError, match="has size"):
            check_input_array(arr, expected_ndim=2, expected_shape=(2, 2))

        with pytest.raises(DimensionMismatchError, match="has size"):
            check_input_array(arr, expected_ndim=2, expected_shape=(3, 3))

        # Test with wrong number of shape dimensions
        with pytest.raises(DimensionMismatchError, match="shape tuple length"):
            check_input_array(arr, expected_ndim=2, expected_shape=(2,))  # Only 1 element for 2D array

    def test_check_input_array_valid_shape(self):
        """Test check_input_array with valid shape."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2)

        # Test with exact matching shape
        check_input_array(arr, expected_ndim=2, expected_shape=(2, 2))

        # Test with partial matching shape (None for flexible dimensions)
        check_input_array(arr, expected_ndim=2, expected_shape=(None, 2))  # Flexible first dim
        check_input_array(arr, expected_ndim=2, expected_shape=(2, None))  # Flexible second dim
        check_input_array(arr, expected_ndim=2, expected_shape=(None, None))  # Both flexible

    def test_calculate_net_benefit_basic(self):
        """Test calculate_net_benefit with basic inputs."""
        # Create test costs, effects, and WTP
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])  # (2, 2)
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])   # (2, 2)
        wtp = 20000.0  # Scalar WTP

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        assert result.shape == costs.shape
        # NMB = (effects * wtp) - costs
        expected = (effects * wtp) - costs
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_net_benefit_with_array_wtp(self):
        """Test calculate_net_benefit with array WTP."""
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])  # (2, 2)
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])   # (2, 2)
        wtp = np.array([15000.0, 20000.0, 25000.0])  # (3,) - multiple thresholds

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        # Result should be (2, 2, 3) - (n_samples, n_strategies, n_wtp_thresholds)
        assert result.shape == (2, 2, 3)

        # Verify calculation: result[n, s, k] = effects[n, s] * wtp[k] - costs[n, s]
        for n in range(2):
            for s in range(2):
                for k in range(3):
                    expected = effects[n, s] * wtp[k] - costs[n, s]
                    assert abs(result[n, s, k] - expected) < 1e-9

    def test_calculate_net_benefit_with_1d_arrays(self):
        """Test calculate_net_benefit with 1D arrays."""
        costs = np.array([100.0, 150.0, 120.0])  # (3,)
        effects = np.array([0.5, 0.6, 0.55])    # (3,)
        wtp = 20000.0  # Scalar WTP

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        assert result.shape == costs.shape
        expected = (effects * wtp) - costs
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_net_benefit_with_1d_and_wtp_array(self):
        """Test calculate_net_benefit with 1D arrays and WTP array."""
        costs = np.array([100.0, 150.0, 120.0])  # (3,)
        effects = np.array([0.5, 0.6, 0.55])    # (3,)
        wtp = np.array([15000.0, 20000.0])      # (2,) - multiple thresholds

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        # Result should be (3, 2) - (n_samples, n_wtp_thresholds)
        assert result.shape == (3, 2)

        # Verify calculation
        for n in range(3):
            for k in range(2):
                expected = effects[n] * wtp[k] - costs[n]
                assert abs(result[n, k] - expected) < 1e-9

    def test_calculate_net_benefit_invalid_inputs(self):
        """Test calculate_net_benefit with invalid inputs."""
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])
        wtp = 20000.0

        # Test with mismatched shapes
        costs_mismatch = np.array([[100.0, 150.0, 120.0]])  # Different shape
        effects_mismatch = np.array([[0.5, 0.6]])  # Different shape

        with pytest.raises(DimensionMismatchError, match="must match"):
            calculate_net_benefit(costs_mismatch, effects_mismatch, wtp)

        # Test with invalid WTP type
        with pytest.raises(InputError, match="WTP must be"):
            calculate_net_benefit(costs, effects, "not a number")

        with pytest.raises(InputError, match="WTP must be"):
            calculate_net_benefit(costs, effects, [])

        with pytest.raises(InputError, match="WTP must be"):
            calculate_net_benefit(costs, effects, None)

    def test_calculate_net_benefit_wtp_with_too_many_dims(self):
        """Test calculate_net_benefit with WTP that has too many dimensions."""
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])

        # Create WTP with 3 dimensions (should fail)
        wtp_3d = np.array([[[15000.0]]])  # 3D array

        with pytest.raises(DimensionMismatchError, match="cannot have more than 2 dimensions"):
            calculate_net_benefit(costs, effects, wtp_3d)

    def test_calculate_net_benefit_with_negative_wtp(self):
        """Test calculate_net_benefit with negative WTP (should be allowed)."""
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])
        wtp = -5000.0  # Negative WTP

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        expected = (effects * wtp) - costs
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_net_benefit_with_2d_wtp(self):
        """Test calculate_net_benefit with 2D WTP array."""
        costs = np.array([[100.0, 150.0], [90.0, 140.0]])  # (2, 2)
        effects = np.array([[0.5, 0.6], [0.45, 0.55]])   # (2, 2)
        wtp = np.array([[15000.0, 20000.0], [16000.0, 18000.0]])  # (2, 2)

        result = calculate_net_benefit(costs, effects, wtp)

        assert isinstance(result, np.ndarray)
        assert result.shape == costs.shape  # Same shape as inputs due to broadcasting
        expected = (effects * wtp) - costs  # Element-wise multiplication
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_optimal_strategy_index_numpy_array(self):
        """Test get_optimal_strategy_index with numpy array."""
        # Create net benefit array (samples x strategies)
        nb_array = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0], [110.0, 100.0, 120.0]])

        result = get_optimal_strategy_index(nb_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result.shape == (3,)  # One index per sample

        # Verify results: max values are at index 1, 1, 2 respectively
        expected = np.array([1, 1, 2], dtype=np.int64)  # Strategy 1 (150), Strategy 1 (140), Strategy 2 (120)
        np.testing.assert_array_equal(result, expected)

    def test_get_optimal_strategy_index_value_array(self):
        """Test get_optimal_strategy_index with ValueArray."""
        # Create net benefit array
        nb_data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 100.0]])
        value_array = ValueArray.from_numpy(nb_data, ["Strategy A", "Strategy B"])

        result = get_optimal_strategy_index(value_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result.shape == (3,)  # One index per sample

        # Verify results
        expected = np.array([1, 1, 0], dtype=np.int64)  # Strategy 1 for first two, Strategy 0 for last
        np.testing.assert_array_equal(result, expected)

    def test_get_optimal_strategy_index_empty_array(self):
        """Test get_optimal_strategy_index with empty array."""
        empty_array = np.array([], dtype=np.float64).reshape(0, 2)  # 0 samples, 2 strategies

        result = get_optimal_strategy_index(empty_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result.shape == (0,)  # Empty result array
        assert len(result) == 0

    def test_get_optimal_strategy_index_single_strategy(self):
        """Test get_optimal_strategy_index with single strategy."""
        # Array with only one strategy per sample
        nb_array = np.array([[100.0], [150.0], [120.0]])  # (3, 1)

        result = get_optimal_strategy_index(nb_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result.shape == (3,)  # One index per sample

        # With single strategy, index should always be 0
        expected = np.array([0, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_get_optimal_strategy_index_tie_values(self):
        """Test get_optimal_strategy_index when there are ties (equal values)."""
        # Create array where first and second strategies have equal max values
        nb_array = np.array([[150.0, 150.0, 120.0], [90.0, 140.0, 140.0]])  # First has tie, second has tie

        result = get_optimal_strategy_index(nb_array)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result.shape == (2,)  # Two samples

        # When there's a tie, argmax returns the first occurrence
        expected = np.array([0, 1], dtype=np.int64)  # First for first sample, second for second
        np.testing.assert_array_equal(result, expected)

    def test_get_optimal_strategy_index_invalid_input_type(self):
        """Test get_optimal_strategy_index with invalid input type."""
        with pytest.raises(InputError, match="must be a NumPy array or NetBenefitArray"):
            get_optimal_strategy_index("not an array")

        with pytest.raises(InputError, match="must be a NumPy array or NetBenefitArray"):
            get_optimal_strategy_index([1, 2, 3])

        with pytest.raises(InputError, match="must be a NumPy array or NetBenefitArray"):
            get_optimal_strategy_index(123)

        with pytest.raises(InputError, match="must be a NumPy array or NetBenefitArray"):
            get_optimal_strategy_index(None)

    def test_get_optimal_strategy_index_1d_input(self):
        """Test get_optimal_strategy_index with 1D input (should raise error)."""
        array_1d = np.array([100.0, 150.0, 120.0])

        # This should fail because 1D arrays are not valid for net benefit
        with pytest.raises(DimensionMismatchError):
            get_optimal_strategy_index(array_1d)

    def test_utils_edge_cases(self):
        """Test utility functions with edge cases."""
        # Test check_input_array with high dimensional arrays
        high_dim_arr = np.random.rand(2, 3, 4, 5)
        check_input_array(high_dim_arr, expected_ndim=4)

        with pytest.raises(DimensionMismatchError):
            check_input_array(high_dim_arr, expected_ndim=3)

        # Test calculate_net_benefit with extreme values
        huge_costs = np.array([[1e10, 1e10], [1e10, 1e10]])
        tiny_effects = np.array([[1e-10, 1e-10], [1e-10, 1e-10]])
        normal_wtp = 10000.0

        result = calculate_net_benefit(huge_costs, tiny_effects, normal_wtp)
        assert isinstance(result, np.ndarray)

        # Test with zero WTP
        zero_wtp = 0.0
        result_zero = calculate_net_benefit(huge_costs, tiny_effects, zero_wtp)
        expected_zero = -huge_costs  # Since NMB = effects*0 - costs = -costs
        np.testing.assert_array_almost_equal(result_zero, expected_zero)

    def test_check_input_array_special_cases(self):
        """Test check_input_array with special cases."""
        # Test with zero-size but proper dimensions
        zero_size_2d = np.array([], dtype=np.float64).reshape(0, 3)  # 0x3 array
        # This should work if allow_empty=True
        check_input_array(zero_size_2d, expected_ndim=2, allow_empty=True)

        # But should fail if allow_empty=False
        with pytest.raises(InputError, match="cannot be empty"):
            check_input_array(zero_size_2d, expected_ndim=2, allow_empty=False)

        # Test with various dtypes
        int_arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        float_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Should work with matching dtypes
        check_input_array(int_arr, expected_ndim=2, expected_dtype=np.int32)
        check_input_array(float_arr, expected_ndim=2, expected_dtype=np.float32)
