"""Comprehensive tests for the schema module to improve coverage to >95%."""

import numpy as np
import pytest
import xarray as xr
from voiage.schema import ValueArray, ParameterSet
from voiage.exceptions import InputError


class TestValueArrayComprehensive:
    """Comprehensive tests for ValueArray to improve coverage."""

    def test_value_array_from_numpy_with_strategy_names(self):
        """Test ValueArray.from_numpy with explicit strategy names."""
        # Create test data
        values = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        strategy_names = ["Strategy A", "Strategy B"]
        
        # Test creation with strategy names
        value_array = ValueArray.from_numpy(values, strategy_names=strategy_names)
        
        assert isinstance(value_array, ValueArray)
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        assert value_array.strategy_names == strategy_names
        np.testing.assert_array_equal(value_array.values.values, values)

    def test_value_array_from_numpy_without_strategy_names(self):
        """Test ValueArray.from_numpy without explicit strategy names."""
        # Create test data
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        
        # Test creation without strategy names (should use defaults)
        value_array = ValueArray.from_numpy(values)
        
        assert isinstance(value_array, ValueArray)
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 2
        assert len(value_array.strategy_names) == 2
        # Verify default strategy names
        assert all(name.startswith("Strategy") for name in value_array.strategy_names)
        assert value_array.strategy_names[0] == "Strategy 1"
        assert value_array.strategy_names[1] == "Strategy 2"
        np.testing.assert_array_equal(value_array.values.values, values)

    def test_value_array_properties_access(self):
        """Test all property accessors."""
        values = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        strategy_names = ["A", "B", "C"]
        
        value_array = ValueArray.from_numpy(values, strategy_names=strategy_names)
        
        # Test properties
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 3
        assert value_array.strategy_names == ["A", "B", "C"]
        assert isinstance(value_array.values, xr.DataArray)
        np.testing.assert_array_equal(value_array.values.values, values)

    def test_value_array_from_numpy_invalid_inputs(self):
        """Test ValueArray.from_numpy with invalid inputs."""
        # Test with non-2D array
        invalid_1d = np.array([100.0, 150.0])
        with pytest.raises(InputError, match="values must be a 2D array"):
            ValueArray.from_numpy(invalid_1d)

        # Test with 3D array
        invalid_3d = np.array([[[100.0, 150.0]], [[90.0, 140.0]]])
        with pytest.raises(InputError, match="values must be a 2D array"):
            ValueArray.from_numpy(invalid_3d)

        # Test with mismatched strategy names length
        valid_data = np.array([[100.0, 150.0], [90.0, 140.0]])
        invalid_names = ["A", "B", "C"]  # 3 names for 2 strategies
        with pytest.raises(InputError, match="Number of strategy names"):
            ValueArray.from_numpy(valid_data, strategy_names=invalid_names)

        # Test with non-numpy array
        with pytest.raises(InputError, match="values must be a NumPy array"):
            ValueArray.from_numpy([[100.0, 150.0], [90.0, 140.0]])  # Plain list

        # Test with non-float array when float expected
        int_data = np.array([[100, 150], [90, 140]], dtype=np.int32)
        value_array_int = ValueArray.from_numpy(int_data)
        assert isinstance(value_array_int, ValueArray)
        assert value_array_int.values.dtype == np.float64  # Should convert to float

    def test_value_array_values_property(self):
        """Test the values property specifically."""
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, strategy_names=["A", "B"])
        
        # Test values property returns xarray DataArray
        vals = value_array.values
        assert isinstance(vals, xr.DataArray)
        assert vals.shape == (2, 2)  # 2 samples, 2 strategies
        assert np.array_equal(vals.values, values)

    def test_value_array_dataset_property(self):
        """Test the dataset property access."""
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        strategy_names = ["Strategy A", "Strategy B"]
        
        value_array = ValueArray.from_numpy(values, strategy_names=strategy_names)
        
        # Test dataset property
        ds = value_array.dataset
        assert isinstance(ds, xr.Dataset)
        assert "net_benefit" in ds.data_vars
        assert ds.net_benefit.shape == (2, 2)
        assert "strategy" in ds.coords
        assert list(ds.strategy.values) == strategy_names

    def test_value_array_equality_comparison(self):
        """Test ValueArray equality comparison."""
        values1 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        values2 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        values3 = np.array([[100.0, 160.0], [90.0, 140.0]], dtype=np.float64)  # Different values
        
        # Create ValueArrays
        va1 = ValueArray.from_numpy(values1, ["A", "B"])
        va2 = ValueArray.from_numpy(values2, ["A", "B"])  # Same values and names
        va3 = ValueArray.from_numpy(values1, ["X", "Y"])  # Different strategy names
        va4 = ValueArray.from_numpy(values3, ["A", "B"])  # Different values
        
        # Test equality
        assert va1 == va2  # Same content
        assert va1 != va3  # Different strategy names
        assert va1 != va4  # Different values
        assert va1 != "not a ValueArray"  # Different types

    def test_value_array_copy_method(self):
        """Test ValueArray copy functionality."""
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        original_va = ValueArray.from_numpy(values, ["A", "B"])
        
        # Test copy functionality
        copied_va = original_va.copy()
        
        # Should be equal but different objects
        assert copied_va == original_va
        assert copied_va is not original_va  # Different object
        assert isinstance(copied_va, ValueArray)

    def test_value_array_edge_cases(self):
        """Test ValueArray with edge cases."""
        # Single sample, single strategy
        single_data = np.array([[100.0]], dtype=np.float64)
        single_va = ValueArray.from_numpy(single_data, ["Single"])
        assert single_va.n_samples == 1
        assert single_va.n_strategies == 1
        assert single_va.strategy_names == ["Single"]
        
        # Many samples, single strategy
        many_samples_data = np.array([[100.0], [110.0], [90.0], [120.0]], dtype=np.float64)
        many_samples_va = ValueArray.from_numpy(many_samples_data, ["Strategy A"])
        assert many_samples_va.n_samples == 4
        assert many_samples_va.n_strategies == 1
        
        # Single sample, many strategies
        many_strat_data = np.array([[100.0, 150.0, 120.0, 140.0]], dtype=np.float64)
        many_strat_va = ValueArray.from_numpy(many_strat_data, ["S1", "S2", "S3", "S4"])
        assert many_strat_va.n_samples == 1
        assert many_strat_va.n_strategies == 4
        
        # Large arrays
        large_data = np.random.rand(1000, 10).astype(np.float64)
        large_va = ValueArray.from_numpy(large_data, [f"Strategy_{i}" for i in range(10)])
        assert large_va.n_samples == 1000
        assert large_va.n_strategies == 10
        assert large_va.values.shape == (1000, 10)


class TestParameterSetComprehensive:
    """Comprehensive tests for ParameterSet to improve coverage."""

    def test_parameter_set_from_numpy_or_dict_with_dict(self):
        """Test ParameterSet.from_numpy_or_dict with dictionary."""
        params = {
            "param1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "param2": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"param1", "param2"}
        assert "param1" in param_set.parameters
        assert "param2" in param_set.parameters
        np.testing.assert_array_equal(param_set.parameters["param1"], params["param1"])
        np.testing.assert_array_equal(param_set.parameters["param2"], params["param2"])

    def test_parameter_set_from_numpy_or_dict_with_numpy_array(self):
        """Test ParameterSet.from_numpy_or_dict with numpy array."""
        # Create 2D array: 3 samples, 2 parameters
        param_array = np.array([
            [0.1, 10.0],  # Sample 1: param1=0.1, param2=10.0
            [0.2, 20.0],  # Sample 2: param1=0.2, param2=20.0
            [0.3, 30.0]   # Sample 3: param1=0.3, param2=30.0
        ], dtype=np.float64)
        
        param_set = ParameterSet.from_numpy_or_dict(param_array)
        
        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert all(name.startswith("param_") for name in param_set.parameter_names)
        assert "param_0" in param_set.parameter_names
        assert "param_1" in param_set.parameter_names
        # Values should be accessible as columns
        assert param_set.parameters["param_0"][0] == 0.1
        assert param_set.parameters["param_1"][0] == 10.0
        assert param_set.parameters["param_0"][2] == 0.3
        assert param_set.parameters["param_1"][2] == 30.0

    def test_parameter_set_from_numpy_or_dict_invalid_inputs(self):
        """Test ParameterSet.from_numpy_or_dict with invalid inputs."""
        # Test with invalid type
        with pytest.raises(InputError, match="must be a NumPy array or Dict"):
            ParameterSet.from_numpy_or_dict("not valid")

        # Test with 1D array (not 2D as required for multiple parameters)
        with pytest.raises(InputError, match="must be a 2-dimensional NumPy array"):
            ParameterSet.from_numpy_or_dict(np.array([0.1, 0.2, 0.3]))

        # Test with empty dict
        with pytest.raises(InputError, match="parameters cannot be empty"):
            ParameterSet.from_numpy_or_dict({})

        # Test with non-numeric dict values
        with pytest.raises(InputError, match="must contain NumPy arrays"):
            ParameterSet.from_numpy_or_dict({"param1": ["a", "b", "c"]})

        # Test with mismatched sample counts in dict
        mismatched_params = {
            "param1": np.array([0.1, 0.2]),  # 2 samples
            "param2": np.array([10.0, 20.0, 30.0])  # 3 samples
        }
        with pytest.raises(InputError, match="All parameter arrays must have the same number of samples"):
            ParameterSet.from_numpy_or_dict(mismatched_params)

    def test_parameter_set_properties(self):
        """Test ParameterSet property accessors."""
        params = {
            "cost_param": np.array([100.0, 120.0, 110.0]),
            "effect_param": np.array([0.8, 0.9, 0.85])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test properties
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"cost_param", "effect_param"}
        assert "cost_param" in param_set.parameters
        assert "effect_param" in param_set.parameters
        assert isinstance(param_set.parameters, dict)
        assert isinstance(param_set.dataset, xr.Dataset)

    def test_parameter_set_get_parameter(self):
        """Test ParameterSet get_parameter method."""
        params = {
            "param1": np.array([0.1, 0.2, 0.3]),
            "param2": np.array([10.0, 20.0, 30.0])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test getting specific parameters
        param1_vals = param_set.get_parameter("param1")
        param2_vals = param_set.get_parameter("param2")
        
        assert isinstance(param1_vals, np.ndarray)
        assert isinstance(param2_vals, np.ndarray)
        np.testing.assert_array_equal(param1_vals, params["param1"])
        np.testing.assert_array_equal(param2_vals, params["param2"])
        
        # Test getting non-existent parameter
        with pytest.raises(KeyError, match="not found"):
            param_set.get_parameter("nonexistent_param")

    def test_parameter_set_subset_by_parameters(self):
        """Test ParameterSet subset_by_parameters method."""
        params = {
            "param1": np.array([0.1, 0.2, 0.3]),
            "param2": np.array([10.0, 20.0, 30.0]),
            "param3": np.array([100.0, 200.0, 300.0])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test subsetting to specific parameters
        subset = param_set.subset_by_parameters(["param1", "param3"])
        
        assert isinstance(subset, ParameterSet)
        assert subset.n_samples == 3  # Same number of samples
        assert len(subset.parameter_names) == 2  # Only 2 parameters
        assert set(subset.parameter_names) == {"param1", "param3"}
        np.testing.assert_array_equal(subset.parameters["param1"], params["param1"])
        np.testing.assert_array_equal(subset.parameters["param3"], params["param3"])
        
        # Verify param2 is not included
        assert "param2" not in subset.parameters
        
        # Test with empty parameter list
        empty_subset = param_set.subset_by_parameters([])
        assert len(empty_subset.parameter_names) == 0

    def test_parameter_set_parameter_statistics(self):
        """Test parameter statistics methods."""
        params = {
            "param1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "param2": np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test getting parameter statistics
        param1_stats = param_set.get_parameter_statistics("param1")
        param2_stats = param_set.get_parameter_statistics("param2")
        
        # Should return a dict with common statistics
        assert isinstance(param1_stats, dict)
        assert "mean" in param1_stats
        assert "std" in param1_stats
        assert "min" in param1_stats
        assert "max" in param1_stats
        assert "n_samples" in param1_stats
        
        # Verify calculations
        assert abs(param1_stats["mean"] - 0.3) < 1e-10  # (0.1+0.2+0.3+0.4+0.5)/5 = 0.3
        assert abs(param1_stats["min"] - 0.1) < 1e-10
        assert abs(param1_stats["max"] - 0.5) < 1e-10
        
        # Test with non-existent parameter
        with pytest.raises(KeyError, match="not found"):
            param_set.get_parameter_statistics("nonexistent")

    def test_parameter_set_all_parameters_statistics(self):
        """Test getting statistics for all parameters."""
        params = {
            "param1": np.array([0.1, 0.2, 0.3]),
            "param2": np.array([10.0, 20.0, 30.0])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test getting all parameter statistics
        all_stats = param_set.get_all_parameter_statistics()
        
        assert isinstance(all_stats, dict)
        assert len(all_stats) == 2  # Two parameters
        assert "param1" in all_stats
        assert "param2" in all_stats
        assert isinstance(all_stats["param1"], dict)
        assert isinstance(all_stats["param2"], dict)

    def test_parameter_set_to_dict(self):
        """Test ParameterSet to_dict method."""
        params = {
            "param_a": np.array([0.1, 0.2, 0.3]),
            "param_b": np.array([10.0, 20.0, 30.0])
        }
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Test converting to dictionary
        result_dict = param_set.to_dict()
        
        assert isinstance(result_dict, dict)
        assert len(result_dict) == 2  # Two parameters
        assert "param_a" in result_dict
        assert "param_b" in result_dict
        assert isinstance(result_dict["param_a"], np.ndarray)
        assert isinstance(result_dict["param_b"], np.ndarray)
        
        # Verify values are preserved
        np.testing.assert_array_equal(result_dict["param_a"], params["param_a"])
        np.testing.assert_array_equal(result_dict["param_b"], params["param_b"])

    def test_parameter_set_dataset_coord_consistency(self):
        """Test that ParameterSet maintains coordinate consistency."""
        params = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Check coordination consistency between dataset and properties
        assert param_set.dataset.sizes["n_samples"] == param_set.n_samples
        assert len(param_set.dataset.data_vars) == len(param_set.parameter_names)

    def test_parameter_set_equality(self):
        """Test ParameterSet equality comparisons."""
        params1 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        params2 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        params3 = {"param1": np.array([0.1, 0.3]), "param2": np.array([10.0, 20.0])}  # Different param1
        
        ps1 = ParameterSet.from_numpy_or_dict(params1)
        ps2 = ParameterSet.from_numpy_or_dict(params2)
        ps3 = ParameterSet.from_numpy_or_dict(params3)
        
        # Test equality
        assert ps1 == ps2  # Same content
        assert ps1 != ps3  # Different content
        assert ps1 != "not a ParameterSet"  # Different types

    def test_parameter_set_edge_cases(self):
        """Test ParameterSet with edge cases."""
        # Single parameter, single sample
        single_param_single_sample = {"param1": np.array([42.0])}
        ps1 = ParameterSet.from_numpy_or_dict(single_param_single_sample)
        assert ps1.n_samples == 1
        assert len(ps1.parameter_names) == 1
        assert ps1.get_parameter("param1")[0] == 42.0
        
        # Single parameter, many samples
        single_param_many_samples = {"param1": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
        ps2 = ParameterSet.from_numpy_or_dict(single_param_many_samples)
        assert ps2.n_samples == 5
        assert len(ps2.parameter_names) == 1
        
        # Many parameters, single sample
        many_params_single_sample = {f"param_{i}": np.array([i*10.0]) for i in range(10)}
        ps3 = ParameterSet.from_numpy_or_dict(many_params_single_sample)
        assert ps3.n_samples == 1
        assert len(ps3.parameter_names) == 10
        
        # Large arrays
        large_params = {f"large_param_{i}": np.random.rand(1000) for i in range(5)}
        ps4 = ParameterSet.from_numpy_or_dict(large_params)
        assert ps4.n_samples == 1000
        assert len(ps4.parameter_names) == 5