"""Comprehensive tests using syrupy for snapshot testing of VOI calculations."""

import numpy as np
import pytest

from voiage.methods.basic import evpi, evppi
from voiage.schema import ParameterSet, ValueArray


class TestVOISnapshotTests:
    """Test VOI calculations with syrupy snapshots."""

    def test_basic_evpi_snapshot(self, snapshot):
        """Test EVPI calculation and snapshot results."""
        # Create deterministic test data
        nb_data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(nb_data, ['Strategy A', 'Strategy B'])

        # Calculate EVPI
        result = evpi(value_array)

        # Return data structure for snapshot comparison
        snapshot_data = {
            "input_data": nb_data.tolist(),
            "strategy_names": value_array.strategy_names,
            "evpi_result": result,
            "expected_max_nb": float(np.mean(np.max(nb_data, axis=1))),
            "max_expected_nb": float(np.max(np.mean(nb_data, axis=0)))
        }

        assert snapshot == snapshot_data

    def test_evpi_with_scaling_snapshot(self, snapshot):
        """Test EVPI calculation with scaling and snapshot results."""
        # Create test data
        nb_data = np.array([[100000.0, 150000.0, 120000.0],
                            [90000.0, 140000.0, 130000.0],
                            [110000.0, 130000.0, 140000.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(nb_data, ['Treatment X', 'Treatment Y', 'Standard Care'])

        # Calculate EVPI with population scaling
        result = evpi(
            value_array,
            population=100000,
            time_horizon=10,
            discount_rate=0.035
        )

        snapshot_data = {
            "nb_data": nb_data.tolist(),
            "strategy_names": value_array.strategy_names,
            "population": 100000,
            "time_horizon": 10,
            "discount_rate": 0.035,
            "evpi_result": result
        }

        assert snapshot == snapshot_data

    @pytest.mark.skip(reason="Requires sklearn which may not be available")
    def test_evppi_calculation_snapshot(self, snapshot):
        """Test EVPPI calculation and snapshot results."""
        # Create test datasets
        nb_data = np.array([[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(nb_data, ['Strategy A', 'Strategy B'])

        param_data = {
            'param1': np.array([0.1, 0.2, 0.3], dtype=np.float64),
            'param2': np.array([10.0, 20.0, 30.0], dtype=np.float64)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(param_data)

        # Calculate EVPPI
        result = evppi(
            value_array,
            parameter_set,
            parameters_of_interest=['param1']
        )

        snapshot_data = {
            "nb_data": nb_data.tolist(),
            "param_data": {k: v.tolist() for k, v in param_data.items()},
            "strategy_names": value_array.strategy_names,
            "parameter_names": parameter_set.parameter_names,
            "evppi_result": result,
            "parameters_of_interest": ['param1']
        }

        assert snapshot == snapshot_data

    def test_edge_cases_snapshot(self, snapshot):
        """Test EVPI with edge cases and snapshot results."""
        # Single strategy case (should yield EVPI = 0)
        single_data = np.array([[100.0], [110.0], [90.0]], dtype=np.float64)
        single_value_array = ValueArray.from_numpy(single_data, ['Single Strategy'])

        single_result = evpi(single_value_array)

        # Identical strategies case (should yield EVPI ~0)
        identical_data = np.array([[100.0, 100.0], [110.0, 110.0], [90.0, 90.0]], dtype=np.float64)
        identical_value_array = ValueArray.from_numpy(identical_data, ['Strategy A', 'Strategy B'])

        identical_result = evpi(identical_value_array)

        snapshot_data = {
            "single_strategy_data": single_data.tolist(),
            "single_strategy_evpi": single_result,
            "identical_strategies_data": identical_data.tolist(),
            "identical_strategies_evpi": identical_result
        }

        assert snapshot == snapshot_data

    def test_complex_scenario_snapshot(self, snapshot):
        """Test a complex scenario and snapshot comprehensive results."""
        np.random.seed(42)  # For reproducibility

        # Create complex datasets
        large_nb_data = np.random.rand(20, 3).astype(np.float64) * 100000
        large_value_array = ValueArray.from_numpy(large_nb_data,
                                                  ['Intervention A', 'Intervention B', 'Control'])

        # Calculate EVPI
        evpi_result = evpi(large_value_array)

        # Create statistical summary
        stats_summary = {
            "mean_per_strategy": [float(np.mean(large_nb_data[:, i])) for i in range(3)],
            "std_per_strategy": [float(np.std(large_nb_data[:, i])) for i in range(3)],
            "min_per_strategy": [float(np.min(large_nb_data[:, i])) for i in range(3)],
            "max_per_strategy": [float(np.max(large_nb_data[:, i])) for i in range(3)],
            "overall_mean": float(np.mean(large_nb_data)),
            "overall_std": float(np.std(large_nb_data))
        }

        snapshot_data = {
            "scenario": "complex_20_samples_3_strategies",
            "dimensions": large_nb_data.shape,
            "evpi_result": evpi_result,
            "statistical_summary": stats_summary
        }

        assert snapshot == snapshot_data
