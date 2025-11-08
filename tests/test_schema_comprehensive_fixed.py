"""Comprehensive tests for schema module to improve coverage."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray


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

    def test_value_array_creation_without_strategy_names(self):
        """Test ValueArray creation from numpy without strategy names."""
        # Test without strategy names (default names should be used)
        data = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)

        value_array = ValueArray.from_numpy(data)  # Without strategy names

        assert isinstance(value_array, ValueArray)
        assert value_array.values.shape == (2, 3)
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 3
        # Default strategy names should be generated
        assert all(name.startswith("Strategy ") for name in value_array.strategy_names)

    def test_value_array_creation_from_xarray_dataset(self):
        """Test ValueArray creation from xarray Dataset."""
        # Create dataset with the expected structure
        data = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        dataset = {
            "net_benefit": (("n_samples", "n_strategies"), data),
            "strategy": ("n_strategies", ["A", "B", "C"])
        }

        # Create from dataset
        value_array = ValueArray.from_numpy(data, strategy_names=["A", "B", "C"])

        assert isinstance(value_array, ValueArray)
        assert value_array.values.shape == (2, 3)
        assert value_array.strategy_names == ["A", "B", "C"]
        assert value_array.n_samples == 2
        assert value_array.n_strategies == 3

    def test_value_array_property_accessors(self):
        """Test all ValueArray property accessors."""
        data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["S1", "S2"])

        # Test property access
        assert value_array.n_samples == 3
        assert value_array.n_strategies == 2
        assert value_array.strategy_names == ["S1", "S2"]
        assert isinstance(value_array.values, np.ndarray)
        assert value_array.values.shape == (3, 2)

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

    def test_value_array_equality(self):
        """Test ValueArray equality comparisons."""
        data1 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        data2 = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        data3 = np.array([[100.0, 160.0], [90.0, 140.0]], dtype=np.float64)

        value_array1 = ValueArray.from_numpy(data1, strategy_names=["S1", "S2"])
        value_array2 = ValueArray.from_numpy(data2, strategy_names=["S1", "S2"])
        value_array3 = ValueArray.from_numpy(data3, strategy_names=["S1", "S2"])

        # Test equality - they should be compared by the dataset equality
        # Since data1 and data2 are identical, and strategy names are the same
        assert value_array1.dataset.equals(value_array2.dataset)
        assert not value_array1.dataset.equals(value_array3.dataset)

    def test_value_array_edge_cases(self):
        """Test ValueArray with edge cases."""
        # Single sample, single strategy
        single_data = np.array([[100.0]], dtype=np.float64)
        single_value_array = ValueArray.from_numpy(single_data, strategy_names=["Single"])

        assert single_value_array.n_samples == 1
        assert single_value_array.n_strategies == 1
        assert single_value_array.strategy_names == ["Single"]

        # Many samples, few strategies
        many_samples_data = np.random.rand(100, 2).astype(np.float64)
        many_samples_array = ValueArray.from_numpy(many_samples_data, strategy_names=["A", "B"])

        assert many_samples_array.n_samples == 100
        assert many_samples_array.n_strategies == 2
        assert many_samples_array.values.shape == (100, 2)

        # Few samples, many strategies
        many_strategy_data = np.random.rand(2, 50).astype(np.float64)
        many_strategy_array = ValueArray.from_numpy(many_strategy_data, strategy_names=[f"S{i}" for i in range(50)])

        assert many_strategy_array.n_samples == 2
        assert many_strategy_array.n_strategies == 50
        assert many_strategy_array.values.shape == (2, 50)

    def test_value_array_invalid_data(self):
        """Test ValueArray creation with invalid data."""
        # Test with 1D array (should fail)
        data_1d = np.array([100.0, 150.0], dtype=np.float64)

        with pytest.raises(InputError, match="values must be a 2D array"):
            ValueArray.from_numpy(data_1d, strategy_names=["S1", "S2"])

    def test_value_array_snapshot_basic(self, snapshot):
        """Test ValueArray with syrupy snapshot."""
        data = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0], [110.0, 130.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(data, strategy_names=["Strategy A", "Strategy B", "Strategy C"])

        snapshot_data = {
            "values": value_array.values.tolist(),
            "strategy_names": value_array.strategy_names,
            "n_samples": value_array.n_samples,
            "n_strategies": value_array.n_strategies
        }

        assert snapshot == snapshot_data

    def test_value_array_snapshot_edge_cases(self, snapshot):
        """Test ValueArray edge cases with syrupy snapshot."""
        # Single strategy case
        single_data = np.array([[100.0], [110.0], [90.0]], dtype=np.float64)
        single_array = ValueArray.from_numpy(single_data, strategy_names=["Single Strategy"])

        # Identical strategies case
        identical_data = np.array([[100.0, 100.0], [110.0, 110.0], [90.0, 90.0]], dtype=np.float64)
        identical_array = ValueArray.from_numpy(identical_data, strategy_names=["Strategy A", "Strategy B"])

        snapshot_data = {
            "single_strategy": {
                "values": single_array.values.tolist(),
                "n_samples": single_array.n_samples,
                "n_strategies": single_array.n_strategies
            },
            "identical_strategies": {
                "values": identical_array.values.tolist(),
                "n_samples": identical_array.n_samples,
                "n_strategies": identical_array.n_strategies
            }
        }

        assert snapshot == snapshot_data


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

    def test_parameter_set_creation_with_python_lists(self):
        """Test ParameterSet creation with Python lists that get converted to numpy."""
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

        # Check values are numpy arrays
        assert isinstance(param_set.parameters["param1"], np.ndarray)
        assert isinstance(param_set.parameters["param2"], np.ndarray)

    def test_parameter_set_property_accessors(self):
        """Test all ParameterSet property accessors."""
        param_dict = {
            "cost_param": np.array([100.0, 120.0, 110.0]),
            "effect_param": np.array([0.8, 0.9, 0.85])
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)

        # Test property access
        assert param_set.n_samples == 3
        assert len(param_set.parameter_names) == 2
        assert set(param_set.parameter_names) == {"cost_param", "effect_param"}
        assert isinstance(param_set.parameters, dict)
        assert len(param_set.parameters) == 2
        assert isinstance(param_set.parameters["cost_param"], np.ndarray)
        assert isinstance(param_set.parameters["effect_param"], np.ndarray)

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

    def test_parameter_set_equality(self):
        """Test ParameterSet equality comparisons."""
        param_dict1 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict2 = {"param1": np.array([0.1, 0.2]), "param2": np.array([10.0, 20.0])}
        param_dict3 = {"param1": np.array([0.1, 0.3]), "param2": np.array([10.0, 20.0])}  # Different values

        param_set1 = ParameterSet.from_numpy_or_dict(param_dict1)
        param_set2 = ParameterSet.from_numpy_or_dict(param_dict2)
        param_set3 = ParameterSet.from_numpy_or_dict(param_dict3)

        # Test equality - they should be compared by the dataset equality
        assert param_set1.dataset.equals(param_set2.dataset)  # Same data
        assert not param_set1.dataset.equals(param_set3.dataset)  # Different data

    def test_parameter_set_edge_cases(self):
        """Test ParameterSet with edge cases."""
        # Single parameter, single sample
        single_param_dict = {"param1": np.array([0.1])}
        single_param_set = ParameterSet.from_numpy_or_dict(single_param_dict)

        assert single_param_set.n_samples == 1
        assert len(single_param_set.parameter_names) == 1
        assert "param1" in single_param_set.parameters

        # Single parameter, many samples
        many_samples_dict = {"param1": np.array([i*0.1 for i in range(100)])}  # 100 samples
        many_samples_param_set = ParameterSet.from_numpy_or_dict(many_samples_dict)

        assert many_samples_param_set.n_samples == 100
        assert len(many_samples_param_set.parameter_names) == 1
        assert many_samples_param_set.parameters["param1"].shape[0] == 100

        # Many parameters, few samples
        many_params_dict = {f"param_{i}": np.array([i*0.1, (i+1)*0.1]) for i in range(50)}
        many_params_param_set = ParameterSet.from_numpy_or_dict(many_params_dict)

        assert many_params_param_set.n_samples == 2
        assert len(many_params_param_set.parameter_names) == 50

    def test_parameter_set_invalid_data(self):
        """Test ParameterSet creation with invalid data."""
        # Test with empty dict
        empty_dict = {}
        with pytest.raises(InputError, match="parameters dictionary cannot be empty"):
            ParameterSet.from_numpy_or_dict(empty_dict)

    def test_parameter_set_snapshot_basic(self, snapshot):
        """Test ParameterSet with syrupy snapshot."""
        param_dict = {
            "param1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "param2": np.array([10.0, 20.0, 30.0], dtype=np.float64)
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)

        snapshot_data = {
            "param_names": param_set.parameter_names,
            "n_samples": param_set.n_samples,
            "param_values": {k: v.tolist() for k, v in param_set.parameters.items()}
        }

        assert snapshot == snapshot_data

    def test_parameter_set_snapshot_edge_cases(self, snapshot):
        """Test ParameterSet edge cases with syrupy snapshot."""
        # Single parameter single sample
        single_dict = {"single_param": np.array([42.0], dtype=np.float64)}
        single_set = ParameterSet.from_numpy_or_dict(single_dict)

        # Many parameters few samples
        many_param_dict = {f"param{i}": np.array([i*1.0, (i+1)*1.0]) for i in range(5)}
        many_set = ParameterSet.from_numpy_or_dict(many_param_dict)

        snapshot_data = {
            "single": {
                "param_names": single_set.parameter_names,
                "n_samples": single_set.n_samples,
                "param_values": {k: v.tolist() for k, v in single_set.parameters.items()}
            },
            "many_params": {
                "param_names": many_set.parameter_names,
                "n_samples": many_set.n_samples,
                "n_param_sets": len(many_set.parameter_names)
            }
        }

        assert snapshot == snapshot_data
