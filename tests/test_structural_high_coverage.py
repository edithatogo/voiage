"""Comprehensive tests for structural uncertainty methods to achieve >95% coverage."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.schema import ParameterSet, ValueArray


class TestStructuralMethodsHighCoverage:
    """Comprehensive tests for structural uncertainty methods."""

    def test_structural_evpi_basic(self):
        """Test basic structural EVPI functionality."""
        # Define evaluator functions that simulate different model structures
        def model_structure_1(param_set: ParameterSet) -> ValueArray:
            """Simple model structure 1 evaluator."""
            # Generate net benefit values based on the parameters
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            param2 = param_set.parameters.get("param2", np.ones(param_set.n_samples))

            # Simulate some relationship between parameters and net benefits
            net_benefits = np.column_stack([
                param1 * 100 + 20,
                param2 * 50 + 10
            ])

            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def model_structure_2(param_set: ParameterSet) -> ValueArray:
            """Simple model structure 2 evaluator."""
            # Generate net benefit values based on the parameters
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            param2 = param_set.parameters.get("param2", np.ones(param_set.n_samples))

            # Different relationship for a different model structure
            net_benefits = np.column_stack([
                param1 * 90 + 30,
                param2 * 60 + 5
            ])

            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Create parameter samples for each structure
        param_samples_1 = {
            "param1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "param2": np.array([10.0, 20.0, 30.0], dtype=np.float64)
        }
        param_samples_2 = {
            "param1": np.array([0.15, 0.25, 0.35], dtype=np.float64),
            "param2": np.array([12.0, 22.0, 32.0], dtype=np.float64)
        }

        param_set_1 = ParameterSet.from_numpy_or_dict(param_samples_1)
        param_set_2 = ParameterSet.from_numpy_or_dict(param_samples_2)

        # Define structure probabilities (equal probability for this example)
        structure_probs = [0.5, 0.5]

        # Test with single structure evaluator (using only one model)
        single_result = structural_evpi(
            model_structure_evaluators=[model_structure_1],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set_1]
        )
        assert isinstance(single_result, float)
        assert single_result >= 0.0

        # Test with multiple structure evaluators
        multi_result = structural_evpi(
            model_structure_evaluators=[model_structure_1, model_structure_2],
            structure_probabilities=structure_probs,
            psa_samples_per_structure=[param_set_1, param_set_2]
        )
        assert isinstance(multi_result, float)
        assert multi_result >= 0.0

        print("✅ Structural EVPI basic functionality works")

    def test_structural_evpi_with_scaling_params(self):
        """Test structural EVPI with scaling parameters."""
        def simple_model_evaluator(param_set: ParameterSet) -> ValueArray:
            """Simple model structure evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([
                param1 * 100 + 20,
                param1 * 80 + 30
            ])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Create parameter samples
        param_samples = {
            "param1": np.array([0.1, 0.2], dtype=np.float64)
        }
        param_set = ParameterSet.from_numpy_or_dict(param_samples)

        # Test with population scaling parameters
        result_scaled = structural_evpi(
            model_structure_evaluators=[simple_model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set],
            population=100000,
            time_horizon=10,
            discount_rate=0.035
        )

        assert isinstance(result_scaled, float)
        assert result_scaled >= 0.0

        print("✅ Structural EVPI with scaling parameters works")

    def test_structural_evpi_validation_errors(self):
        """Test structural EVPI with validation errors."""
        def simple_evaluator(param_set: ParameterSet) -> ValueArray:
            """Simple evaluator."""
            net_benefits = np.column_stack([
                np.ones(param_set.n_samples) * 100,
                np.ones(param_set.n_samples) * 120
            ])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        param_samples = {"param1": np.array([0.1, 0.2])}
        param_set = ParameterSet.from_numpy_or_dict(param_samples)

        # Test with empty model_structure_evaluators list
        with pytest.raises(InputError, match="model_structure_evaluators cannot be empty"):
            structural_evpi(
                model_structure_evaluators=[],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set]
            )

        # Test with mismatched lengths in inputs
        with pytest.raises(InputError, match="Number of structure probabilities must match number of model structure evaluators"):
            structural_evpi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[0.5, 0.5],  # Two probabilities, one model
                psa_samples_per_structure=[param_set]
            )

        # Test with negative population
        with pytest.raises(InputError, match="Population must be positive"):
            structural_evpi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                population=-1000
            )

        # Test with invalid discount rate
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            structural_evpi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                discount_rate=1.5
            )

        # Test with invalid time horizon
        with pytest.raises(InputError, match="Time horizon must be positive"):
            structural_evpi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                time_horizon=0
            )

        print("✅ Structural EVPI validation checks work")

    def test_structural_evpi_edge_cases(self):
        """Test structural EVPI with edge cases."""
        def simple_model_evaluator(param_set: ParameterSet) -> ValueArray:
            """Simple model evaluator."""
            # Create net benefits with just one strategy (should return 0 EVPI)
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = param1.reshape(-1, 1) * 100  # Shape (n_samples, 1) - only one strategy
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def constant_model_evaluator(param_set: ParameterSet) -> ValueArray:
            """Evaluator that gives constant values (EVPI should be 0)."""
            n_samples = param_set.n_samples
            # Return same values for all strategies - EVPI should be approximately 0
            net_benefits = np.tile(np.array([100.0, 100.0]), (n_samples, 1))  # Identical strategies
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Test with single strategy (should return 0 EVPI)
        param_samples_single = {"param1": np.array([0.1, 0.2, 0.3], dtype=np.float64)}
        param_set_single = ParameterSet.from_numpy_or_dict(param_samples_single)

        result_single = structural_evpi(
            model_structure_evaluators=[simple_model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set_single]
        )

        # With single strategy, EVPI should be 0.0
        assert result_single == 0.0

        # Test with identical strategies (should return ~0 EVPI)
        param_samples_const = {"param1": np.array([0.1, 0.2], dtype=np.float64)}
        param_set_const = ParameterSet.from_numpy_or_dict(param_samples_const)

        result_constant = structural_evpi(
            model_structure_evaluators=[constant_model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set_const]
        )

        # With identical strategies, EVPI should be approximately 0
        assert abs(result_constant) < 1e-10

        # Test with single sample
        single_sample_params = {"param1": np.array([0.5], dtype=np.float64)}
        single_sample_param_set = ParameterSet.from_numpy_or_dict(single_sample_params)

        result_single_sample = structural_evpi(
            model_structure_evaluators=[constant_model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[single_sample_param_set]
        )

        assert isinstance(result_single_sample, float)
        assert result_single_sample >= 0.0

        print("✅ Structural EVPI edge cases handled correctly")

    def test_structural_evppi_basic(self):
        """Test basic structural EVPPI functionality."""
        def model_structure_1(param_set: ParameterSet) -> ValueArray:
            """Model structure 1 evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            param2 = param_set.parameters.get("param2", np.ones(param_set.n_samples))

            net_benefits = np.column_stack([
                param1 * 100 + param2 * 10,
                param1 * 80 + param2 * 15
            ])

            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def model_structure_2(param_set: ParameterSet) -> ValueArray:
            """Model structure 2 evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            param2 = param_set.parameters.get("param2", np.ones(param_set.n_samples))

            net_benefits = np.column_stack([
                param1 * 90 + param2 * 12,
                param1 * 70 + param2 * 18
            ])

            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Create parameter samples
        param_samples_1 = {
            "param1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "param2": np.array([10.0, 20.0, 30.0], dtype=np.float64)
        }
        param_samples_2 = {
            "param1": np.array([0.15, 0.25, 0.35], dtype=np.float64),
            "param2": np.array([12.0, 22.0, 32.0], dtype=np.float64)
        }

        param_set_1 = ParameterSet.from_numpy_or_dict(param_samples_1)
        param_set_2 = ParameterSet.from_numpy_or_dict(param_samples_2)

        # Define structure probabilities
        structure_probs = [0.4, 0.6]

        # Test structural EVPPI with structure of interest (first structure)
        result = structural_evppi(
            model_structure_evaluators=[model_structure_1, model_structure_2],
            structure_probabilities=structure_probs,
            psa_samples_per_structure=[param_set_1, param_set_2],
            structures_of_interest=[0]  # Interest in the first model structure
        )

        assert isinstance(result, float)
        assert result >= 0.0

        print("✅ Structural EVPPI basic functionality works")

    def test_structural_evppi_multiple_structures_of_interest(self):
        """Test structural EVPPI with multiple structures of interest."""
        def model_1(param_set: ParameterSet) -> ValueArray:
            """Model 1 evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([param1 * 100, param1 * 120])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def model_2(param_set: ParameterSet) -> ValueArray:
            """Model 2 evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([param1 * 110, param1 * 110])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def model_3(param_set: ParameterSet) -> ValueArray:
            """Model 3 evaluator."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([param1 * 95, param1 * 125])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Create parameter samples for all structures
        param_samples = {
            "param1": np.array([0.1, 0.2], dtype=np.float64),
        }
        param_set = ParameterSet.from_numpy_or_dict(param_samples)

        # Use same parameters for all models (for simplicity)
        all_param_sets = [param_set] * 3
        evals = [model_1, model_2, model_3]

        # Define probabilities
        probs = [0.3, 0.4, 0.3]

        # Test with multiple structures of interest
        result = structural_evppi(
            model_structure_evaluators=evals,
            structure_probabilities=probs,
            psa_samples_per_structure=all_param_sets,
            structures_of_interest=[0, 1]  # Interest in structures 0 and 1
        )

        assert isinstance(result, float)
        assert result >= 0.0

        print("✅ Structural EVPPI with multiple structures of interest works")

    def test_structural_evppi_validation_errors(self):
        """Test structural EVPPI with validation errors."""
        def simple_evaluator(param_set: ParameterSet) -> ValueArray:
            """Simple model evaluator."""
            net_benefits = np.column_stack([
                np.ones(param_set.n_samples) * 100,
                np.ones(param_set.n_samples) * 120
            ])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        param_samples = {"param1": np.array([0.1, 0.2])}
        param_set = ParameterSet.from_numpy_or_dict(param_samples)

        # Test with empty structures_of_interest
        with pytest.raises((InputError, ValueError), match="(structures_of_interest|structures|interest).*empty"):
            structural_evppi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                structures_of_interest=[]
            )

        # Test with out-of-bounds structure index
        with pytest.raises(InputError, match="Structure index 1 is out of bounds with 1 total structures"):
            structural_evppi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                structures_of_interest=[1]  # Only 1 structure exists, index should be 0
            )

        # Test with negative structure index
        with pytest.raises(InputError, match="Structure index -1 is out of bounds"):
            structural_evppi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                structures_of_interest=[-1]  # Negative index
            )

        # Test with duplicated structure indices
        # (This is probably acceptable - just testing we don't fail with duplicates)
        result = structural_evppi(
            model_structure_evaluators=[simple_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set],
            structures_of_interest=[0, 0]  # Duplicates
        )
        assert isinstance(result, float)
        assert result >= 0.0

        # Test with invalid population
        with pytest.raises(InputError, match="Population must be positive"):
            structural_evppi(
                model_structure_evaluators=[simple_evaluator],
                structure_probabilities=[1.0],
                psa_samples_per_structure=[param_set],
                structures_of_interest=[0],
                population=-500
            )

        print("✅ Structural EVPPI validation checks work")

    def test_structural_evppi_edge_cases(self):
        """Test structural EVPPI with edge cases."""
        def single_structure_evaluator(param_set: ParameterSet) -> ValueArray:
            """Evaluator with single strategy (should return 0 EVPPI)."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            # Return single strategy - EVPPI should be 0
            net_benefits = param1.reshape(-1, 1) * 100  # Shape (n_samples, 1)
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        def two_strategy_evaluator(param_set: ParameterSet) -> ValueArray:
            """Evaluator with two strategies."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([
                param1 * 100,
                param1 * 105
            ])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Test with single strategy (should return 0 EVPPI)
        param_samples_single = {"param1": np.array([0.1, 0.2, 0.3], dtype=np.float64)}
        param_set_single = ParameterSet.from_numpy_or_dict(param_samples_single)

        result_single = structural_evppi(
            model_structure_evaluators=[single_structure_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set_single],
            structures_of_interest=[0]
        )

        # With single strategy, EVPPI should be 0.0
        assert result_single == 0.0

        # Test with single sample per structure
        single_sample_params = {"param1": np.array([0.5], dtype=np.float64)}
        single_sample_param_set = ParameterSet.from_numpy_or_dict(single_sample_params)

        result_single_sample = structural_evppi(
            model_structure_evaluators=[two_strategy_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[single_sample_param_set],
            structures_of_interest=[0]
        )

        assert isinstance(result_single_sample, float)
        assert result_single_sample >= 0.0

        print("✅ Structural EVPPI edge cases handled correctly")

    def test_structural_methods_consistency_check(self):
        """Test consistency between structural EVPI and EVPPI."""
        def model_evaluator(param_set: ParameterSet) -> ValueArray:
            """Simple model evaluator for consistency test."""
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            net_benefits = np.column_stack([
                param1 * 100,
                param1 * 110
            ])
            value_array = ValueArray.from_numpy(net_benefits.astype(np.float64)); return value_array

        # Create parameter samples
        param_samples = {"param1": np.array([0.1, 0.2])}
        param_set = ParameterSet.from_numpy_or_dict(param_samples)

        # Calculate structural EVPI and EVPPI
        evpi_result = structural_evpi(
            model_structure_evaluators=[model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set]
        )
        evppi_result = structural_evppi(
            model_structure_evaluators=[model_evaluator],
            structure_probabilities=[1.0],
            psa_samples_per_structure=[param_set],
            structures_of_interest=[0]
        )

        # EVPPI should never exceed EVPI (for the same model structure evaluation)
        assert evppi_result <= evpi_result or abs(evppi_result - evpi_result) < 1e-10

        print("✅ Structural methods consistency verified")

    def test_structural_multistructure_consistency(self):
        """Test consistency with multiple model structures."""
        def eval1(param_set: ParameterSet) -> ValueArray:
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            return ValueArray.from_numpy(
                np.column_stack([param1 * 100, param1 * 110]),
                ["S1", "S2"]
            )

        def eval2(param_set: ParameterSet) -> ValueArray:
            param1 = param_set.parameters.get("param1", np.ones(param_set.n_samples))
            return ValueArray.from_numpy(
                np.column_stack([param1 * 90, param1 * 120]),
                ["S1", "S2"]
            )

        # Parameters for each structure
        params1 = ParameterSet.from_numpy_or_dict({"param1": np.array([0.1, 0.2])})
        params2 = ParameterSet.from_numpy_or_dict({"param1": np.array([0.15, 0.25])})

        # Equal probabilities
        probs = [0.5, 0.5]

        # Evaluate EVPI for full model uncertainty
        evpi_result = structural_evpi(
            model_structure_evaluators=[eval1, eval2],
            structure_probabilities=probs,
            psa_samples_per_structure=[params1, params2]
        )

        # Evaluate EVPPI for learning just one structure's parameters
        evppi_result = structural_evppi(
            model_structure_evaluators=[eval1, eval2],
            structure_probabilities=probs,
            psa_samples_per_structure=[params1, params2],
            structures_of_interest=[0]
        )

        # Check that values are finite and non-negative
        assert np.isfinite(evpi_result)
        assert np.isfinite(evppi_result)
        assert evpi_result >= 0
        assert evppi_result >= 0

        print("✅ Multi-structure model consistency verified")
