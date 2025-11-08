"""Comprehensive tests for schema module to improve coverage."""

import numpy as np
import pytest
import xarray as xr
from voiage.schema import ValueArray, ParameterSet


class TestValueArrayComprehensive:
    """Tests for ValueArray to improve schema coverage."""

    def test_value_array_creation_from_numpy(self):
        """Test ValueArray creation from numpy array."""
        # Test with strategy names
        data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        strategies = ["Strategy A", "Strategy B"]
        
        value_array = ValueArray.from_numpy(data, strategy_names=strategies)
        
        assert isinstance(value_array, ValueArray)
        assert value_array.values.shape == (3, 2)
        assert value_array.strategy_names == strategies
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        
        # Verify the stored data
        np.testing.assert_array_equal(value_array.values.values, data)
        print("✅ ValueArray.from_numpy with strategy names works")

    def test_value_array_creation_without_strategy_names(self):
        """Test ValueArray creation from numpy without strategy names."""
        data = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        
        value_array = ValueArray.from_numpy(data)  # Without strategy names
        
        assert isinstance(value_array, ValueArray)
        assert value_array.values.shape == (2, 2)
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 2
        # Default strategy names should be generated
        assert len(value_array.strategy_names) == 2
        print("✅ ValueArray.from_numpy without strategy names works")

    def test_value_array_creation_from_xarray_dataset(self):
        """Test ValueArray creation from xarray Dataset."""
        # Create a dataset with the expected structure
        data = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), data)},
            coords={
                "n_samples": np.arange(2),
                "n_strategies": np.arange(3),
                "strategy": ("n_strategies", ["A", "B", "C"]),
            },
        )
        
        value_array = ValueArray(dataset=dataset)
        
        assert isinstance(value_array, ValueArray)
        assert value_array.values.shape == (2, 3)
        assert value_array.strategy_names == ["A", "B", "C"]
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 3
        
        # Check data integrity
        np.testing.assert_array_equal(value_array.values.values, data)
        print("✅ ValueArray creation from xarray Dataset works")

    def test_value_array_property_accessors(self):
        """Test all ValueArray property accessors."""
        data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["S1", "S2"])
        
        # Test property access
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        assert value_array.strategy_names == ["S1", "S2"]
        assert isinstance(value_array.values, xr.DataArray)
        assert value_array.values.shape == (3, 2)
        
        print("✅ ValueArray property accessors work")

    def test_value_array_str_repr(self):
        """Test ValueArray string representation."""
        data = np.array([[100.0, 150.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["Strat A", "Strat B"])
        
        # Test string representations
        str_repr = str(value_array)
        repr_repr = repr(value_array)
        
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert "ValueArray" in str_repr
        assert "ValueArray" in repr_repr
        print("✅ ValueArray string representations work")

    def test_value_array_equality(self):
        """Test ValueArray equality comparisons."""
        data1 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        data2 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        data3 = np.array([[100.0, 160.0], [90.0, 140.0]], dtype=np.float64)
        
        value_array1 = ValueArray.from_numpy(data1, strategy_names=["S1", "S2"])
        value_array2 = ValueArray.from_numpy(data2, strategy_names=["S1", "S2"])
        value_array3 = ValueArray.from_numpy(data3, strategy_names=["S1", "S2"])
        
        # Test equality
        assert value_array1 == value_array2  # Same data
        assert value_array1 != value_array3  # Different data
        
        # Test with different strategy names
        value_array4 = ValueArray.from_numpy(data1, strategy_names=["A", "B"])
        assert value_array1 != value_array4  # Same data, different strategy names
        
        print("✅ ValueArray equality comparisons work")

    def test_value_array_copy_method(self):
        """Test ValueArray copy functionality."""
        data = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        original_array = ValueArray.from_numpy(data, strategy_names=["S1", "S2"])
        
        # Test copying
        copied_array = original_array.copy()
        
        assert isinstance(copied_array, ValueArray)
        assert copied_array == original_array  # Should be equal
        assert copied_array is not original_array  # But different objects
        
        # Check that modifying one doesn't affect the other
        copied_array.dataset["net_benefit"].values[0, 0] = 999.0
        assert original_array.values.values[0, 0] != 999.0  # Original unchanged
        
        print("✅ ValueArray copy functionality works")

    def test_value_array_get_strategy_index(self):
        """Test getting strategy index by name."""
        data = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["Treatment A", "Treatment B"])
        
        # Test getting strategy index
        idx_a = value_array.get_strategy_index("Treatment A")
        idx_b = value_array.get_strategy_index("Treatment B")
        
        assert idx_a == 0
        assert idx_b == 1
        
        # Test with invalid name
        with pytest.raises(ValueError, match="not found"):
            value_array.get_strategy_index("Invalid Strategy")
        
        print("✅ ValueArray get_strategy_index works")

    def test_value_array_slice_by_strategy(self):
        """Test slicing ValueArray by strategy names."""
        data = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["S1", "S2", "S3"])
        
        # Slice to get only first 2 strategies
        sliced = value_array.slice_by_strategies(["S1", "S2"])
        
        assert isinstance(sliced, ValueArray)
        assert sliced.n_samples == 2
        assert sliced.n_strategies == 2
        assert sliced.strategy_names == ["S1", "S2"]
        # Verify values match the first 2 columns
        expected_values = np.array([[100.0, 150.0], [90.0, 140.0]])
        np.testing.assert_array_equal(sliced.values.values, expected_values)
        
        print("✅ ValueArray slice_by_strategies works")

    def test_value_array_edge_cases(self):
        """Test ValueArray with edge cases."""
        # Test with single sample and single strategy
        single_data = np.array([[100.0]], dtype=np.float64)
        single_value_array = ValueArray.from_numpy(single_data, strategy_names=["Single"])
        
        assert single_value_array.n_samples == 1
        assert single_value_array.n_strategies == 1
        assert single_value_array.strategy_names == ["Single"]
        
        # Test with many samples but few strategies
        many_samples_data = np.random.rand(100, 2).astype(np.float64)
        many_samples_array = ValueArray.from_numpy(many_samples_data, strategy_names=["A", "B"])
        
        assert many_samples_array.n_samples == 100
        assert many_samples_array.n_strategies == 2
        assert many_samples_array.values.shape == (100, 2)
        
        # Test with many strategies but few samples
        many_strategies_data = np.random.rand(2, 50).astype(np.float64)
        many_strategies_array = ValueArray.from_numpy(many_strategies_data, strategy_names=[f"S{i}" for i in range(50)])
        
        assert many_strategies_array.n_samples == 2
        assert many_strategies_array.n_strategies == 50
        assert many_strategies_array.values.shape == (2, 50)
        
        print("✅ ValueArray edge cases handled properly")


class TestParameterSetComprehensive:
    """Tests for ParameterSet to improve schema coverage."""

    def test_parameter_set_creation_from_numpy_dict(self):
        """Test ParameterSet creation from numpy dict."""
        # Test with numpy arrays
        param_dict = {
            "param1": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            "param2": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        }
        
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 4
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"param1", "param2"}
        assert "param1" in param_set.parameters
        assert "param2" in param_set.parameters
        
        # Check values
        np.testing.assert_array_equal(param_set.parameters["param1"], param_dict["param1"])
        np.testing.assert_array_equal(param_set.parameters["param2"], param_dict["param2"])
        
        print("✅ ParameterSet.from_numpy_or_dict with numpy arrays works")

    def test_parameter_set_creation_with_float_dict(self):
        """Test ParameterSet creation with Python float dict."""
        # Test with Python lists/floats that will be converted to numpy
        param_dict = {
            "param1": [0.1, 0.2, 0.3],
            "param2": [10.0, 20.0, 30.0],
        }
        
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"param1", "param2"}
        
        print("✅ ParameterSet.from_numpy_or_dict with float lists works")

    def test_parameter_set_creation_from_xarray_dataset(self):
        """Test ParameterSet creation from xarray Dataset."""
        # Create dataset with parameter data
        n_samples = 5
        param1_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        param2_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        dataset = xr.Dataset(
            {
                "param1": (("n_samples",), param1_values),
                "param2": (("n_samples",), param2_values)
            },
            coords={"n_samples": np.arange(n_samples)}
        )
        
        param_set = ParameterSet(dataset=dataset)
        
        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 5
        assert set(param_set.parameter_names) == {"param1", "param2"}
        assert np.array_equal(param_set.parameters["param1"], param1_values)
        assert np.array_equal(param_set.parameters["param2"], param2_values)
        
        print("✅ ParameterSet creation from xarray Dataset works")

    def test_parameter_set_property_accessors(self):
        """Test all ParameterSet property accessors."""
        param_dict = {
            "cost_param": np.array([100.0, 120.0, 110.0]),
            "eff_param": np.array([0.8, 0.9, 0.85])
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        # Test property access
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"cost_param", "eff_param"}
        assert isinstance(param_set.parameters, dict)
        assert len(param_set.parameters) == 2
        assert isinstance(param_set.parameters["cost_param"], np.ndarray)
        assert isinstance(param_set.parameters["eff_param"], np.ndarray)
        
        print("✅ ParameterSet property accessors work")

    def test_parameter_set_str_repr(self):
        """Test ParameterSet string representation."""
        param_dict = {"param1": np.array([0.1, 0.2])}
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        # Test string representations
        str_repr = str(param_set)
        repr_repr = repr(param_set)
        
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert "ParameterSet" in str_repr
        assert "ParameterSet" in repr_repr
        
        print("✅ ParameterSet string representations work")

    def test_parameter_set_equality(self):
        """Test ParameterSet equality comparisons."""
        param_dict1 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict2 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict3 = {"param1": np.array([0.1, 0.3]), "param2": np.array([10.0, 20.0])}  # Different values
        
        param_set1 = ParameterSet.from_numpy_or_dict(param_dict1)
        param_set2 = ParameterSet.from_numpy_or_dict(param_dict2)
        param_set3 = ParameterSet.from_numpy_or_dict(param_dict3)
        
        # Test equality
        assert param_set1 == param_set2  # Same data
        assert param_set1 != param_set3  # Different data
        
        # Test with different parameter names
        param_dict4 = {"A": np.array([0.1, 0.2]), "B": np.array([10.0, 20.0])}
        param_set4 = ParameterSet.from_numpy_or_dict(param_dict4)
        assert param_set1 != param_set4  # Same data, different parameter names
        
        print("✅ ParameterSet equality comparisons work")

    def test_parameter_set_subset_by_parameters(self):
        """Test subsetting ParameterSet by parameter names."""
        param_dict = {
            "param1": np.array([0.1, 0.2, 0.3]),
            "param2": np.array([10.0, 20.0, 30.0]),
            "param3": np.array([100.0, 200.0, 300.0])
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        # Subset to get only first 2 parameters
        subset = param_set.subset_by_parameters(["param1", "param2"])
        
        assert isinstance(subset, ParameterSet)
        assert subset.n_samples == 3  # Same number of samples
        assert len(subset.parameter_names) == 2  # Only 2 parameters
        assert set(subset.parameter_names) == {"param1", "param2"}
        
        # Check values are preserved
        np.testing.assert_array_equal(subset.parameters["param1"], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(subset.parameters["param2"], np.array([10.0, 20.0, 30.0]))
        
        print("✅ ParameterSet subset_by_parameters works")

    def test_parameter_set_copy_method(self):
        """Test ParameterSet copy functionality."""
        param_dict = {
            "param1": np.array([0.1, 0.2, 0.3]),
            "param2": np.array([10.0, 20.0, 30.0])
        }
        original = ParameterSet.from_numpy_or_dict(param_dict)
        
        # Test copying
        copied = original.copy()
        
        assert isinstance(copied, ParameterSet)
        assert copied == original  # Should be equal
        assert copied is not original  # But different objects
        
        # Check that modifying one doesn't affect the other
        copied.dataset["param1"].values[0] = 999.0
        assert original.parameters["param1"][0] != 999.0  # Original unchanged
        
        print("✅ ParameterSet copy functionality works")

    def test_parameter_set_edge_cases(self):
        """Test ParameterSet with edge cases."""
        # Test with single sample
        single_sample_dict = {"param1": np.array([0.1])}
        single_param_set = ParameterSet.from_numpy_or_dict(single_sample_dict)
        
        assert single_param_set.n_samples == 1
        assert len(single_param_set.parameter_names) == 1
        assert "param1" in single_param_set.parameters
        
        # Test with single parameter, many samples
        many_samples_dict = {"param1": np.array([i*0.1 for i in range(100)])}  # 100 samples
        many_samples_param_set = ParameterSet.from_numpy_or_dict(many_samples_dict)
        
        assert many_samples_param_set.n_samples == 100
        assert len(many_samples_param_set.parameter_names) == 1
        assert many_samples_param_set.parameters["param1"].shape[0] == 100
        
        # Test with many parameters, few samples
        many_params_dict = {f"param_{i}": np.array([i*0.1, (i+1)*0.1]) for i in range(50)}
        many_params_param_set = ParameterSet.from_numpy_or_dict(many_params_dict)
        
        assert many_params_param_set.n_samples == 2
        assert len(many_params_param_set.parameter_names) == 50
        
        print("✅ ParameterSet edge cases handled properly")

    def test_parameter_set_get_parameter_statistics(self):
        """Test getting parameter statistics."""
        param_dict = {
            "param1": np.array([0.1, 0.2, 0.3, 0.4]),
            "param2": np.array([10.0, 20.0, 30.0, 40.0])
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)
        
        # Test mean calculation
        param1_mean = np.mean(param_dict["param1"])
        param2_mean = np.mean(param_dict["param2"])
        
        assert abs(np.mean(param_set.parameters["param1"]) - param1_mean) < 1e-10
        assert abs(np.mean(param_set.parameters["param2"]) - param2_mean) < 1e-10
        
        # Test std calculation
        param1_std = np.std(param_dict["param1"])
        param2_std = np.std(param_dict["param2"])
        
        assert abs(np.std(param_set.parameters["param1"]) - param1_std) < 1e-10
        assert abs(np.std(param_set.parameters["param2"]) - param2_std) < 1e-10
        
        print("✅ ParameterSet parameter statistics work correctly")