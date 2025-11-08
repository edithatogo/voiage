"""Comprehensive tests for schema module to increase coverage above 95%."""

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray


class TestValueArrayHighCoverage:
    """Comprehensive tests for ValueArray to improve coverage to >95%."""

    def test_value_array_creation_from_numpy_with_strategy_names(self):
        """Test ValueArray creation from numpy array with explicit strategy names."""
        # Create test data
        values = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        strategy_names = ["Strategy A", "Strategy B"]

        # Test creation with strategy names
        value_array = ValueArray.from_numpy(values, strategy_names=strategy_names)

        assert isinstance(value_array, ValueArray)
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        assert value_array.strategy_names == strategy_names
        np.testing.assert_array_equal(value_array.values, values)

    def test_value_array_creation_from_numpy_without_strategy_names(self):
        """Test ValueArray creation from numpy array without strategy names."""
        # Create test data
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)

        # Test creation without strategy names (should use defaults)
        value_array = ValueArray.from_numpy(values)

        assert isinstance(value_array, ValueArray)
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 2
        assert len(value_array.strategy_names) == 2
        # Verify default strategy names
        assert all(name.startswith("Strategy_") for name in value_array.strategy_names)
        assert value_array.strategy_names[0] == "Strategy_0"
        assert value_array.strategy_names[1] == "Strategy_1"
        np.testing.assert_array_equal(value_array.values, values)

    def test_value_array_properties(self):
        """Test all ValueArray property accessors."""
        values = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        strategy_names = ["A", "B", "C"]

        value_array = ValueArray.from_numpy(values, strategy_names=strategy_names)

        # Test properties
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 3
        assert value_array.strategy_names == ["A", "B", "C"]
        assert isinstance(value_array.values, np.ndarray)
        np.testing.assert_array_equal(value_array.values, values)

    def test_value_array_from_numpy_invalid_inputs(self):
        """Test ValueArray.from_numpy with invalid inputs."""
        # Test with non-2D array
        invalid_1d = np.array([100.0, 150.0])
        with pytest.raises(InputError, match="must be 2-dimensional"):
            ValueArray.from_numpy(invalid_1d)

        # Test with 3D array
        invalid_3d = np.array([[[100.0, 150.0]], [[90.0, 140.0]]])
        with pytest.raises(InputError, match="must be 2-dimensional"):
            ValueArray.from_numpy(invalid_3d)

        # Test with mismatched strategy names length
        valid_data = np.array([[100.0, 150.0], [90.0, 140.0]])
        invalid_names = ["A", "B", "C"]  # 3 names for 2 strategies
        with pytest.raises(InputError, match="strategy_names must have"):
            ValueArray.from_numpy(valid_data, strategy_names=invalid_names)

        # Test with non-numeric data
        non_numeric_data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(InputError, match="must be numeric"):
            ValueArray.from_numpy(non_numeric_data)

        # Test with complex dtypes that are not float
        complex_data = np.array([[1+2j, 3+4j]], dtype=complex)
        with pytest.raises(InputError, match="must be numeric"):
            ValueArray.from_numpy(complex_data)

    def test_value_array_from_xarray_dataset(self):
        """Test ValueArray creation from xarray Dataset."""
        # Create test dataset with expected structure
        values = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        xarray_dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(3),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["Treatment A", "Treatment B"])
            }
        )

        # Create ValueArray from dataset
        value_array = ValueArray(dataset=xarray_dataset)

        assert isinstance(value_array, ValueArray)
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        assert value_array.strategy_names == ["Treatment A", "Treatment B"]
        np.testing.assert_array_equal(value_array.values, values)

    def test_value_array_equality(self):
        """Test ValueArray equality comparisons."""
        values1 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        values2 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        values3 = np.array([[100.0, 160.0], [90.0, 140.0]], dtype=np.float64)  # Different values

        # Create ValueArrays
        va1 = ValueArray.from_numpy(values1, strategy_names=["A", "B"])
        va2 = ValueArray.from_numpy(values2, strategy_names=["A", "B"])  # Same values and names
        va3 = ValueArray.from_numpy(values1, strategy_names=["X", "Y"])  # Different strategy names
        va4 = ValueArray.from_numpy(values3, strategy_names=["A", "B"])  # Different values

        # Test equality
        assert va1 == va2  # Same content
        assert va1 != va3  # Different strategy names
        assert va1 != va4  # Different values
        assert va1 != "not a ValueArray"  # Different type

    def test_value_array_edge_cases(self):
        """Test ValueArray with edge cases."""
        # Single sample, single strategy
        single_data = np.array([[100.0]], dtype=np.float64)
        single_va = ValueArray.from_numpy(single_data, strategy_names=["Single"])
        assert single_va.n_samples == 1
        assert single_va.n_strategies == 1
        assert single_va.strategy_names == ["Single"]
        assert single_va.values.shape == (1, 1)

        # Many samples, single strategy
        many_samples_data = np.random.rand(1000, 1).astype(np.float64)
        many_samples_va = ValueArray.from_numpy(many_samples_data, strategy_names=["Mono"])
        assert many_samples_va.n_samples == 1000
        assert many_samples_va.n_strategies == 1

        # Single sample, many strategies
        many_strats_data = np.random.rand(1, 100).astype(np.float64)
        strategy_names = [f"Strategy_{i}" for i in range(100)]
        many_strats_va = ValueArray.from_numpy(many_strats_data, strategy_names=strategy_names)
        assert many_strats_va.n_samples == 1
        assert many_strats_va.n_strategies == 100
        assert len(many_strats_va.strategy_names) == 100

    def test_value_array_copy_method(self):
        """Test ValueArray copy functionality."""
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        original_va = ValueArray.from_numpy(values, strategy_names=["A", "B"])

        # Create copy
        copied_va = original_va.copy()

        # Objects should be different
        assert copied_va is not original_va
        # But content should be the same
        assert copied_va == original_va
        assert copied_va.n_samples == original_va.n_samples
        assert copied_va.n_strategies == original_va.n_strategies
        assert copied_va.strategy_names == original_va.strategy_names
        np.testing.assert_array_equal(copied_va.values, original_va.values)


class TestParameterSetHighCoverage:
    """Comprehensive tests for ParameterSet to improve coverage to >95%."""

    def test_parameter_set_creation_from_numpy_dict(self):
        """Test ParameterSet creation from numpy dict."""
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

    def test_parameter_set_creation_from_numpy_array(self):
        """Test ParameterSet creation from 2D numpy array."""
        # Array with shape (n_samples, n_parameters) = (3, 2)
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
        np.testing.assert_array_equal(param_set.parameters["param_0"], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(param_set.parameters["param_1"], np.array([10.0, 20.0, 30.0]))

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

    def test_parameter_set_from_numpy_or_dict_invalid_inputs(self):
        """Test ParameterSet.from_numpy_or_dict with invalid inputs."""
        # Test with invalid type
        with pytest.raises(InputError, match="must be a numpy array or Dict"):
            ParameterSet.from_numpy_or_dict("not valid")

        # Test with non-2D array
        invalid_1d = np.array([0.1, 0.2])
        with pytest.raises(InputError, match="must be 2-dimensional"):
            ParameterSet.from_numpy_or_dict(invalid_1d)

        # Test with 3D array
        invalid_3d = np.array([[[0.1, 0.2]]])
        with pytest.raises(InputError, match="must be 2-dimensional"):
            ParameterSet.from_numpy_or_dict(invalid_3d)

        # Test with empty dict
        empty_dict = {}
        with pytest.raises(InputError, match="parameters dictionary cannot be empty"):
            ParameterSet.from_numpy_or_dict(empty_dict)

        # Test with mismatched sample counts in dict
        mismatched_params = {
            "param1": np.array([0.1, 0.2]),  # 2 samples
            "param2": np.array([10.0, 20.0, 30.0])  # 3 samples - mismatch!
        }
        with pytest.raises(InputError, match="All parameter arrays must have the same length"):
            ParameterSet.from_numpy_or_dict(mismatched_params)

        # Test with non-numeric values
        non_numeric_params = {
            "param1": np.array(["a", "b"]),  # Non-numeric
            "param2": np.array([10.0, 20.0])
        }
        with pytest.raises(InputError, match="All parameter arrays must be numeric"):
            ParameterSet.from_numpy_or_dict(non_numeric_params)

    def test_parameter_set_equality(self):
        """Test ParameterSet equality comparisons."""
        param_dict1 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict2 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict3 = {"param1": np.array([0.1, 0.3]), "param2": np.array([10.0, 20.0])}  # Different param1

        ps1 = ParameterSet.from_numpy_or_dict(param_dict1)
        ps2 = ParameterSet.from_numpy_or_dict(param_dict2)
        ps3 = ParameterSet.from_numpy_or_dict(param_dict3)

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
        assert ps1.parameters["param1"][0] == 42.0

        # Single parameter, many samples
        single_param_many_samples = {"param1": np.array([i*0.1 for i in range(50)])}
        ps2 = ParameterSet.from_numpy_or_dict(single_param_many_samples)
        assert ps2.n_samples == 50
        assert len(ps2.parameter_names) == 1

        # Many parameters, single sample
        many_params_single_sample = {f"param_{i}": np.array([i*10.0]) for i in range(50)}
        ps3 = ParameterSet.from_numpy_or_dict(many_params_single_sample)
        assert ps3.n_samples == 1
        assert len(ps3.parameter_names) == 50

        # Large arrays
        large_params = {f"large_param_{i}": np.random.rand(1000) for i in range(5)}
        ps4 = ParameterSet.from_numpy_or_dict(large_params)
        assert ps4.n_samples == 1000
        assert len(ps4.parameter_names) == 5

    def test_parameter_set_copy_method(self):
        """Test ParameterSet copy functionality."""
        params = {
            "param_a": np.array([0.1, 0.2, 0.3]),
            "param_b": np.array([10.0, 20.0, 30.0])
        }

        original_ps = ParameterSet.from_numpy_or_dict(params)

        # Test copy method
        copied_ps = original_ps.copy()

        # Objects should be different
        assert copied_ps is not original_ps
        # But content should be the same
        assert copied_ps == original_ps
        assert copied_ps.n_samples == original_ps.n_samples
        assert set(copied_ps.parameter_names) == set(original_ps.parameter_names)
        for param_name in original_ps.parameter_names:
            np.testing.assert_array_equal(
                copied_ps.parameters[param_name],
                original_ps.parameters[param_name]
            )

    def test_parameter_set_dataset_creation_from_xarray(self):
        """Test ParameterSet creation from xarray dataset."""
        # Create dataset with proper structure
        param_values = np.array([
            [0.1, 10.0, 100.0],  # Sample 1
            [0.2, 20.0, 200.0],  # Sample 2
        ], dtype=np.float64)
        xarray_dataset = xr.Dataset(
            {"param1": (("n_samples",), param_values[:, 0]),
             "param2": (("n_samples",), param_values[:, 1]),
             "param3": (("n_samples",), param_values[:, 2])},
            coords={"n_samples": np.arange(2)}
        )

        param_set = ParameterSet(dataset=xarray_dataset)

        assert isinstance(param_set, ParameterSet)
        assert param_set.n_samples == 2
        assert len(param_set.parameter_names) == 3
        assert set(param_set.parameter_names) == {"param1", "param2", "param3"}
        assert param_set.parameters["param1"].shape == (2,)
        assert param_set.parameters["param2"].shape == (2,)
        assert param_set.parameters["param3"].shape == (2,)

    def test_value_array_str_rep(self):
        """Test ValueArray string representations."""
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, strategy_names=["Strat A", "Strat B"])

        # Test string representations
        str_repr = str(value_array)
        repr_repr = repr(value_array)

        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert "ValueArray" in str_repr
        assert "ValueArray" in repr_repr

    def test_parameter_set_str_rep(self):
        """Test ParameterSet string representations."""
        params = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_set = ParameterSet.from_numpy_or_dict(params)

        # Test string representations
        str_repr = str(param_set)
        repr_repr = repr(param_set)

        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert "ParameterSet" in str_repr
        assert "ParameterSet" in repr_repr
