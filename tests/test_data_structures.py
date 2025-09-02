# tests/test_data_structures.py

"""Unit tests for the core data structures in voiage.schema."""

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.schema import DecisionOption, ParameterSet, ValueArray


class TestValueArray:
    """Tests for the ValueArray data structure."""

    def test_init_and_properties(self):
        """Test initialization and properties of ValueArray."""
        data = np.random.rand(10, 3)
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), data)},
            coords={
                "n_samples": np.arange(10),
                "n_strategies": np.arange(3),
                "strategy": ("n_strategies", ["A", "B", "C"]),
            },
        )
        va = ValueArray(dataset)
        assert va.n_samples == 10
        assert va.n_strategies == 3
        assert va.strategy_names == ["A", "B", "C"]
        np.testing.assert_array_equal(va.values, data)

    def test_post_init_validations(self):
        """Test post-init validations of ValueArray."""
        with pytest.raises(InputError, match="'dataset' must be a xarray.Dataset"):
            ValueArray("not a dataset")

        with pytest.raises(InputError, match="must have a 'n_samples' dimension"):
            ValueArray(xr.Dataset({"value": (("s", "o"), np.random.rand(2, 2))}))


class TestParameterSet:
    """Tests for the ParameterSet data structure."""

    def test_init_and_properties(self):
        """Test initialization and properties of ParameterSet."""
        params = {"p1": np.random.rand(10), "p2": np.random.rand(10)}
        dataset = xr.Dataset(
            {k: (("n_samples",), v) for k, v in params.items()},
            coords={"n_samples": np.arange(10)},
        )
        ps = ParameterSet(dataset)
        assert ps.n_samples == 10
        assert set(ps.parameter_names) == {"p1", "p2"}
        np.testing.assert_array_equal(ps.parameters["p1"], params["p1"])

    def test_post_init_validations(self):
        """Test post-init validations of ParameterSet."""
        with pytest.raises(InputError, match="'dataset' must be a xarray.Dataset"):
            ParameterSet("not a dataset")


class TestDecisionOption:
    """Tests for the DecisionOption data structure."""

    def test_init(self):
        """Test initialization of DecisionOption."""
        do = DecisionOption("Test", 100)
        assert do.name == "Test"
        assert do.sample_size == 100

    def test_post_init_validations(self):
        """Test post-init validations of DecisionOption."""
        with pytest.raises(InputError, match="must be a non-empty string"):
            DecisionOption("", 100)
        with pytest.raises(InputError, match="must be a positive integer"):
            DecisionOption("Test", 0)
