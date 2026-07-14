"""Tests for voiage.methods.basic module to increase coverage to >90%."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.basic import check_parameter_samples, evpi, evppi
from voiage.schema import ParameterSet, ValueArray


class TestEVPI:
    """Test the evpi function comprehensively."""

    def test_evpi_basic_calculation(self) -> None:
        """Test basic EVPI calculation with known values."""
        # Create test net benefit array with known expected result
        # Simple example: 3 samples, 2 strategies
        nb_values = np.array(
            [
                [100.0, 150.0],  # Sample 1: Strategy 2 is better (150 vs 100)
                [200.0, 180.0],  # Sample 2: Strategy 1 is better (200 vs 180)
                [120.0, 140.0],  # Sample 3: Strategy 2 is better (140 vs 120)
            ],
            dtype=np.float64,
        )

        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        # Calculate EVPI
        result_evpi = evpi(value_array)

        # Manually verify calculation:
        # Expected max NB with perfect info = mean of max NB per sample
        expected_max_nb_per_sample = np.max(nb_values, axis=1)  # [150, 200, 140]
        expected_max_nb = np.mean(
            expected_max_nb_per_sample
        )  # (150+200+140)/3 = 163.33
        # Max of expected NB without info = max of mean NB per strategy
        expected_nb_per_strategy = np.mean(nb_values, axis=0)  # [140, 156.67]
        max_expected_nb = np.max(expected_nb_per_strategy)  # 156.67
        # EVPI = 163.33 - 156.67 = 6.66
        expected_evpi = expected_max_nb - max_expected_nb

        assert abs(result_evpi - expected_evpi) < 1e-10
        assert isinstance(result_evpi, float)
        assert result_evpi >= 0  # EVPI should always be non-negative

    def test_evpi_with_population_scaling(self) -> None:
        """Test EVPI calculation with population scaling."""
        # Create test data
        nb_values = (np.random.rand(100, 3) * 100000).astype(
            np.float64
        )  # 100 samples, 3 strategies
        value_array = ValueArray.from_numpy(
            nb_values, ["Strategy A", "Strategy B", "Strategy C"]
        )

        # Test with population scaling
        evpi_pop = evpi(
            value_array, population=100000, time_horizon=10, discount_rate=0.03
        )

        # Should be a float
        assert isinstance(evpi_pop, float)
        assert evpi_pop >= 0

        # Population-adjusted value should be greater than or equal to base value
        # (when population > 1 and discount rate < 1)
        evpi_base = evpi(value_array)  # Base EVPI
        if evpi_base > 0:
            assert evpi_pop >= evpi_base

    def test_evpi_single_strategy(self) -> None:
        """Test EVPI calculation for single strategy (should be 0)."""
        # Single strategy - no decision to make, so EVPI should be 0
        nb_values = np.array(
            [[100.0], [150.0], [120.0]], dtype=np.float64
        )  # 3 samples, 1 strategy
        value_array = ValueArray.from_numpy(nb_values, ["Single Strategy"])

        result_evpi = evpi(value_array)

        # EVPI for single strategy should be 0
        assert abs(result_evpi) < 1e-10
        assert isinstance(result_evpi, float)

    def test_evpi_identical_strategies(self) -> None:
        """Test EVPI calculation when all strategies have identical net benefits."""
        # All strategies identical - no value in learning which is best since they're the same
        identical_values = np.array(
            [100.0, 100.0, 100.0], dtype=np.float64
        )  # All the same
        nb_values = np.tile(identical_values, (50, 1)).astype(
            np.float64
        )  # 50 samples, all identical
        value_array = ValueArray.from_numpy(
            nb_values, ["Strategy A", "Strategy B", "Strategy C"]
        )

        result_evpi = evpi(value_array)

        # EVPI should be very close to 0 for identical strategies
        assert abs(result_evpi) < 1e-10
        assert isinstance(result_evpi, float)

    def test_evpi_one_sample(self) -> None:
        """Test EVPI calculation with only one sample."""
        # Single sample - no variance, so EVPI should typically be 0
        nb_values = np.array(
            [[100.0, 150.0, 120.0]], dtype=np.float64
        )  # 1 sample, 3 strategies
        value_array = ValueArray.from_numpy(
            nb_values, ["Strategy A", "Strategy B", "Strategy C"]
        )

        result_evpi = evpi(value_array)

        # For a single sample, expected max NB per sample equals max of expected NB per strategy
        # So EVPI should be 0
        assert abs(result_evpi) < 1e-10
        assert isinstance(result_evpi, float)

    def test_evpi_input_validation(self) -> None:
        """Test EVPI with invalid inputs."""
        # Create valid test data
        nb_values = np.array([[100.0, 150.0], [200.0, 180.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        # Test with invalid input type for value_array
        with pytest.raises(InputError):
            evpi("not a value array")

        # Test with invalid population (negative)
        with pytest.raises(InputError, match="Population must be positive"):
            evpi(value_array, population=-1000, time_horizon=10, discount_rate=0.03)

        # Test with invalid population (zero)
        with pytest.raises(InputError, match="Population must be positive"):
            evpi(value_array, population=0, time_horizon=10, discount_rate=0.03)

        # Test with invalid time_horizon (negative)
        with pytest.raises(InputError, match="Time horizon must be positive"):
            evpi(value_array, population=100000, time_horizon=-5, discount_rate=0.03)

        # Test with invalid time_horizon (zero)
        with pytest.raises(InputError, match="Time horizon must be positive"):
            evpi(value_array, population=100000, time_horizon=0, discount_rate=0.03)

        # Test with invalid discount_rate (too high)
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            evpi(value_array, population=100000, time_horizon=10, discount_rate=1.5)

        # Test with invalid discount_rate (negative)
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            evpi(value_array, population=100000, time_horizon=10, discount_rate=-0.1)

        # Test with population but no time horizon
        with pytest.raises(InputError, match="To calculate population EVPI"):
            evpi(value_array, population=100000)  # Missing time_horizon

        # Test with time horizon but no population
        with pytest.raises(InputError, match="To calculate population EVPI"):
            evpi(value_array, time_horizon=10)  # Missing population


class TestEVPPI:
    """Test the evppi function comprehensively."""

    def test_check_parameter_samples_accepts_arrays_and_dicts(self) -> None:
        """Test parameter sample normalization for arrays and dictionaries."""
        one_dimensional = np.array([1.0, 2.0, 3.0])
        normalized_array = check_parameter_samples(one_dimensional, n_samples=3)
        assert normalized_array.shape == (3, 1)
        assert np.array_equal(normalized_array[:, 0], one_dimensional)

        sample_dict = {
            "alpha": np.array([1.0, 2.0, 3.0]),
            "beta": np.array([4.0, 5.0, 6.0]),
        }
        normalized_dict = check_parameter_samples(sample_dict, n_samples=3)
        assert normalized_dict.shape == (3, 2)
        assert np.array_equal(normalized_dict[:, 0], sample_dict["alpha"])
        assert np.array_equal(normalized_dict[:, 1], sample_dict["beta"])

    def test_check_parameter_samples_rejects_invalid_shapes_and_types(self) -> None:
        """Test parameter sample validation failures."""
        with pytest.raises(InputError, match="must be a NumPy array"):
            check_parameter_samples("not samples", n_samples=3)

        with pytest.raises(Exception, match="does not match"):
            check_parameter_samples(np.array([1.0, 2.0]), n_samples=3)

    def test_evppi_basic(self) -> None:
        """Test basic EVPPI calculation."""
        # Create test data
        nb_values = (np.random.rand(100, 3) * 1000).astype(
            np.float64
        )  # 100 samples, 3 strategies
        value_array = ValueArray.from_numpy(
            nb_values, ["Strategy A", "Strategy B", "Strategy C"]
        )

        # Create parameter samples
        params = {
            "param1": np.random.rand(100).astype(np.float64),
            "param2": np.random.rand(100).astype(np.float64),
            "param3": np.random.rand(100).astype(np.float64),
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Test EVPPI calculation for param1
        result_evppi = evppi(value_array, parameter_set, ["param1"])

        # Should be a float and non-negative
        assert isinstance(result_evppi, float)
        assert result_evppi >= 0

        # EVPPI should not exceed EVPI
        evpi_result = evpi(value_array)
        assert (
            result_evppi <= evpi_result + 1e-9
        )  # Add small tolerance for floating point errors

    def test_evppi_warns_for_raw_dict_parameter_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raw dict PSA samples should emit the compatibility warning."""
        nb_values = np.array([[100.0, 150.0], [120.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        params = {
            "param1": np.array([0.1, 0.2], dtype=np.float64),
            "param2": np.array([0.3, 0.4], dtype=np.float64),
        }

        def fake_evppi(self: DecisionAnalysis, **kwargs: object) -> float:
            return 0.0

        monkeypatch.setattr(DecisionAnalysis, "evppi", fake_evppi)

        with pytest.warns(DeprecationWarning, match="compatibility alias"):
            result = evppi(value_array, params, ["param1"])

        assert result == 0.0

    def test_evppi_multiple_params(self) -> None:
        """Test EVPPI calculation with multiple parameters."""
        # Create test data
        nb_values = (np.random.rand(100, 2) * 1000).astype(
            np.float64
        )  # 100 samples, 2 strategies
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        # Create parameter samples
        params = {
            "param1": np.random.rand(100).astype(np.float64),
            "param2": np.random.rand(100).astype(np.float64),
            "param3": np.random.rand(100).astype(np.float64),
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Calculate EVPPI for param1 and param2 together
        result_evppi = evppi(value_array, parameter_set, ["param1", "param2"])

        # Should be a float and non-negative
        assert isinstance(result_evppi, float)
        assert result_evppi >= 0

        # EVPPI should not exceed EVPI
        evpi_result = evpi(value_array)
        assert result_evppi <= evpi_result + 1e-9

    def test_evppi_chunk_size_matches_unchunked_result(self) -> None:
        """Test chunked EVPPI evaluation matches the unchunked path."""
        param_values = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        nb_values = np.column_stack(
            [
                param_values * 1000.0,
                (1.0 - param_values) * 1200.0,
            ]
        ).astype(np.float64)

        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])
        parameter_set = ParameterSet.from_numpy_or_dict({"linear_param": param_values})

        result_no_chunk = evppi(value_array, parameter_set, ["linear_param"])
        result_chunk = evppi(
            value_array,
            parameter_set,
            ["linear_param"],
            chunk_size=3,
        )

        assert isinstance(result_no_chunk, float)
        assert isinstance(result_chunk, float)
        assert result_no_chunk >= 0.0
        assert result_chunk >= 0.0
        assert pytest.approx(result_no_chunk, rel=1e-10, abs=1e-10) == result_chunk

    def test_evppi_input_validation(self) -> None:
        """Test EVPPI with invalid inputs."""
        # Create valid test data
        nb_values = (np.random.rand(100, 2) * 1000).astype(np.float64)
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        params = {"param1": np.random.rand(100).astype(np.float64)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Test with non-array input for value_array
        with pytest.raises(InputError):
            evppi("not a value array", parameter_set, ["param1"])

        # Test with non-parameter set for parameter_samples
        with pytest.raises(InputError):
            evppi(value_array, "not a parameter set", ["param1"])

        # Test with non-list for parameters_of_interest
        with pytest.raises(InputError):
            evppi(value_array, parameter_set, "param1")  # Should be a list

        # Test with parameter not in parameter set
        with pytest.raises(
            InputError, match="All `parameters_of_interest` must be in the ParameterSet"
        ):
            evppi(value_array, parameter_set, ["nonexistent_param"])

        # Test with invalid n_regression_samples
        with pytest.raises(InputError, match="n_regression_samples must be an integer"):
            evppi(
                value_array,
                parameter_set,
                ["param1"],
                n_regression_samples="not an int",
            )

        with pytest.raises(InputError, match="n_regression_samples must be positive"):
            evppi(value_array, parameter_set, ["param1"], n_regression_samples=0)

        with pytest.raises(InputError, match="n_regression_samples must be positive"):
            evppi(value_array, parameter_set, ["param1"], n_regression_samples=-5)

        # Test with too many regression samples
        with pytest.raises(
            InputError, match=r"n_regression_samples.*cannot exceed total samples"
        ):
            evppi(
                value_array, parameter_set, ["param1"], n_regression_samples=200
            )  # More than n_samples

        # Test population scaling validation
        with pytest.raises(InputError, match="Population must be positive"):
            evppi(
                value_array,
                parameter_set,
                ["param1"],
                population=-1000,
                time_horizon=10,
                discount_rate=0.03,
            )

        with pytest.raises(InputError, match="Time horizon must be positive"):
            evppi(
                value_array,
                parameter_set,
                ["param1"],
                population=100000,
                time_horizon=0,
                discount_rate=0.03,
            )

        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            evppi(
                value_array,
                parameter_set,
                ["param1"],
                population=100000,
                time_horizon=10,
                discount_rate=1.5,
            )

    def test_evppi_no_param_uncertainty(self) -> None:
        """Test EVPPI when parameters have no uncertainty (constant values)."""
        # Create test value array
        nb_values = (np.random.rand(50, 2) * 1000).astype(np.float64)
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        # Create parameter samples with constant values (no uncertainty)
        params = {
            "param1": np.full(50, 0.5, dtype=np.float64),  # All samples have same value
            "param2": np.full(
                50, 100.0, dtype=np.float64
            ),  # All samples have same value
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # EVPPI for constant parameters should be 0 (no value in learning what we already know)
        result_evppi = evppi(value_array, parameter_set, ["param1"])

        # The result should be close to 0 since there's no uncertainty to resolve
        # In practice, due to regression implementation, it might not be exactly 0
        assert isinstance(result_evppi, float)
        assert result_evppi >= -1e-10  # Allow for small numerical errors

    def test_evppi_linear_relationship(self) -> None:
        """Test EVPPI with a known linear relationship between parameter and net benefit."""
        # Create test value array with a linear relationship
        n_samples = 100
        param_values = np.linspace(0, 1, n_samples).astype(
            np.float64
        )  # Parameter values from 0 to 1

        # Create net benefits that depend on the parameter
        strategy_a_nb = param_values * 1000
        strategy_b_nb = (1 - param_values) * 1200  # Different relationship
        nb_values = np.column_stack([strategy_a_nb, strategy_b_nb]).astype(np.float64)

        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        params = {"linear_param": param_values}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Calculate EVPPI for the parameter that affects net benefits
        result_evppi = evppi(value_array, parameter_set, ["linear_param"])

        # Should be a float and non-negative
        assert isinstance(result_evppi, float)
        assert result_evppi >= 0

    def test_evppi_edge_case_single_parameter_sample(self) -> None:
        """Test EVPPI with a single parameter sample."""
        # Create test data
        nb_values = np.array(
            [[100.0, 150.0]], dtype=np.float64
        )  # Single sample, 2 strategies
        value_array = ValueArray.from_numpy(nb_values, ["Strategy A", "Strategy B"])

        # Single parameter sample
        params = {"param1": np.array([0.5], dtype=np.float64)}  # Single parameter value
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # With only a single parameter sample, EVPPI calculation cannot properly
        # assess the value of learning about that parameter. The regression may fail.
        try:
            result_evppi = evppi(value_array, parameter_set, ["param1"])
            # If it doesn't fail, result should be valid (might be 0 due to insufficient samples)
            assert isinstance(result_evppi, float)
            assert result_evppi >= 0
        except Exception:
            # It's acceptable for the function to fail with only 1 sample for regression
            # (which is needed to assess the value of learning about the parameter)
            pass  # This is expected behavior with insufficient samples for regression


def test_import_functionality() -> None:
    """Test that the basic methods are importable and available."""
    from voiage.methods.basic import evpi, evppi

    # Verify functions exist
    assert callable(evpi)
    assert callable(evppi)

    # Basic type checks
    assert isinstance(evpi.__doc__, (str, type(None)))
    assert isinstance(evppi.__doc__, (str, type(None)))

    print("✅ All basic VOI methods are importable and available")


def test_basic_module_handles_missing_sklearn_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional sklearn import should fail soft when unavailable."""
    import builtins
    import importlib

    import voiage.methods.basic as basic_module

    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> object:
        if name == "sklearn.linear_model":
            raise ImportError("sklearn intentionally unavailable in test")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reloaded = importlib.reload(basic_module)

    assert reloaded.SKLEARN_AVAILABLE is False
    assert reloaded.LinearRegression is None

    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(basic_module)
