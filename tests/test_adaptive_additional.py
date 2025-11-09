"""Additional tests for voiage.methods.adaptive module to improve coverage further."""


import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods.adaptive import (
    adaptive_evsi,
    bayesian_adaptive_trial_simulator,
    sophisticated_adaptive_trial_simulator,
)
from voiage.schema import DecisionOption, ParameterSet, TrialDesign


class TestAdaptiveEVSIAdditional:
    """Additional tests for adaptive_evsi function to cover remaining lines."""

    def test_adaptive_evsi_with_population_scaling_full_params(self):
        """Test adaptive_evsi with all population scaling parameters."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Create adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 50)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules with interim analysis points
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.9, "futility": 0.1},
            "sample_size_reestimation": True
        }

        # Test with all scaling parameters
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            population=100000,
            time_horizon=10,
            discount_rate=0.03,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a float
        assert isinstance(result, float)
        assert result >= 0

    def test_adaptive_evsi_interim_analysis_scenarios(self):
        """Test adaptive_evsi with different interim analysis scenarios."""
        # Create an adaptive trial simulator that simulates early stopping
        def simulator_with_early_stopping(psa_samples, trial_design=None, trial_data=None):
            """Simulate can trigger early stopping scenarios."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 50)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Test with interim analysis that triggers efficacy stopping
        adaptive_rules_efficacy = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.99, "futility": 0.01},  # Very high efficacy threshold to trigger stopping
            "sample_size_reestimation": True
        }

        result_eff = adaptive_evsi(
            adaptive_trial_simulator=simulator_with_early_stopping,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules_efficacy,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a float
        assert isinstance(result_eff, float)
        assert result_eff >= 0

        # Test with interim analysis that triggers futility stopping
        adaptive_rules_futility = {
            "interim_analysis_points": [0.3],  # Early stopping point
            "early_stopping_rules": {"efficacy": 0.01, "futility": 0.99},  # Very high futility threshold to trigger stopping
            "sample_size_reestimation": False
        }

        result_fut = adaptive_evsi(
            adaptive_trial_simulator=simulator_with_early_stopping,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules_futility,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a float
        assert isinstance(result_fut, float)
        assert result_fut >= 0

    def test_adaptive_evsi_exception_handling_in_simulator(self):
        """Test adaptive_evsi with simulator that properly handles exceptions."""
        # Create a simulator that simulates a more realistic scenario where the
        # exception occurs in the inner loop, not in the initial call
        def simple_modeler(psa_samples, trial_design=None, trial_data=None):
            """Create modeler that returns valid results."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        def simulator_that_fails_after_first_call():
            """Create to create a simulator that fails after first call."""
            call_count = 0
            def simulator(psa_samples, trial_design=None, trial_data=None):
                nonlocal call_count
                call_count += 1
                if call_count > 1:  # First call succeeds, subsequent calls fail
                    raise ValueError("Simulated error in trial simulator on subsequent calls")
                n_samples = psa_samples.n_samples
                # Create net benefits for 2 strategies
                nb_values = np.random.rand(n_samples, 2) * 1000
                # Make strategy 1 slightly better on average
                nb_values[:, 1] += 100

                import xarray as xr
                dataset = xr.Dataset(
                    {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                    coords={
                        "n_samples": np.arange(n_samples),
                        "n_strategies": np.arange(2),
                        "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                    }
                )
                from voiage.schema import ValueArray
                return ValueArray(dataset=dataset)
            return simulator

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 50)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.05},
        }

        # Test with simulator that fails on subsequent calls (within the loop)
        failing_simulator = simulator_that_fails_after_first_call()

        # The function should still work even if some evaluations fail within the outer loops
        result = adaptive_evsi(
            adaptive_trial_simulator=failing_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a float and non-negative (even with some simulation errors handled)
        assert isinstance(result, float)
        assert result >= 0  # Should handle failures gracefully and return non-negative value

    def test_adaptive_evsi_zero_discount_rate(self):
        """Test adaptive_evsi with zero discount rate for population scaling."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Create adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 50)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.05},
        }

        # Test with zero discount rate (special case in annuity calculation)
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            population=100000,
            time_horizon=10,
            discount_rate=0.0,  # Zero discount rate - special case
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a float
        assert isinstance(result, float)
        assert result >= 0

    def test_adaptive_evsi_negative_value_scenario(self):
        """Test adaptive_evsi in a scenario where the adaptive trial may have little value."""
        # Create a simulator that returns values with almost no difference between strategies
        # This simulates a case where the adaptive trial provides little value
        def low_value_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simulate returns values with minimal differences between strategies."""
            n_samples = psa_samples.n_samples
            # Create very similar net benefits for 2 strategies to simulate low value scenario
            base_value = 1000
            strategy1_values = np.full(n_samples, base_value)
            strategy2_values = np.full(n_samples, base_value + np.random.normal(0, 0.1, n_samples))  # Very small differences

            nb_values = np.column_stack([strategy1_values, strategy2_values])

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Strategy A", "Strategy B"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.001, 50)}  # Very low uncertainty
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.05},
        }

        # Calculate adaptive EVSI - even if the adaptive trial might not be valuable,
        # the function should return 0 (since it clips negative values to 0)
        result = adaptive_evsi(
            adaptive_trial_simulator=low_value_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be non-negative (the function should clip negative values to 0)
        assert isinstance(result, float)
        assert result >= 0  # Should be non-negative due to max(0.0, value) in the function

    def test_adaptive_evsi_small_loop_numbers(self):
        """Test adaptive_evsi with the smallest possible loop numbers."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Create adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 10)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=50),
            DecisionOption(name="Treatment B", sample_size=50)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.9, "futility": 0.1},
        }

        # Test with minimal loop numbers (1 each) to test edge case
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=1,
            n_inner_loops=1
        )

        # Result should be a float
        assert isinstance(result, float)
        assert result >= 0

    def test_adaptive_evsi_input_validation_specific_cases(self):
        """Test adaptive_evsi with specific input validation cases."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Create adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make strategy 1 slightly better on average
            nb_values[:, 1] += 100

            import xarray as xr
            dataset = xr.Dataset(
                {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
                coords={
                    "n_samples": np.arange(n_samples),
                    "n_strategies": np.arange(2),
                    "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
                }
            )
            from voiage.schema import ValueArray
            return ValueArray(dataset=dataset)

        # Create test parameter set
        params = {"effectiveness": np.random.normal(0.1, 0.05, 50)}
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        adaptive_rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.05},
        }

        # Test with negative population (should raise error)
        with pytest.raises(InputError, match="Population must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules=adaptive_rules,
                population=-1000,
                time_horizon=10,
                n_outer_loops=3,
                n_inner_loops=5
            )

        # Test with negative time horizon (should raise error)
        with pytest.raises(InputError, match="Time horizon must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules=adaptive_rules,
                population=10000,
                time_horizon=-10,
                n_outer_loops=3,
                n_inner_loops=5
            )

        # Test with invalid discount rate (too high)
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules=adaptive_rules,
                population=10000,
                time_horizon=10,
                discount_rate=1.5,
                n_outer_loops=3,
                n_inner_loops=5
            )

        # Test with invalid discount rate (negative)
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules=adaptive_rules,
                population=10000,
                time_horizon=10,
                discount_rate=-0.1,
                n_outer_loops=3,
                n_inner_loops=5
            )


class TestSophisticatedAdaptiveTrialSimulator:
    """Test the sophisticated_adaptive_trial_simulator function."""

    def test_sophisticated_adaptive_trial_simulator_basic(self):
        """Test basic functionality of sophisticated_adaptive_trial_simulator."""
        # Create test parameter set
        params = {
            "treatment_effect": np.random.normal(0.1, 0.05, 100),
            "control_rate": np.random.normal(0.3, 0.05, 100),
            "cost_per_patient": np.random.normal(5000, 500, 100)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Control", sample_size=100),
            DecisionOption(name="Treatment", sample_size=100)
        ]
        design = TrialDesign(arms=trial_arms)

        # Create adaptive rules
        rules = {
            "interim_analysis_points": [0.5],
            "early_stopping_rules": {"efficacy": 0.9, "futility": 0.1},
            "sample_size_reestimation": True
        }

        # Test sophisticated adaptive trial simulator
        result = sophisticated_adaptive_trial_simulator(
            psa_samples=parameter_set,
            base_design=design,
            adaptive_rules=rules
        )

        # Check result type and properties
        from voiage.schema import ValueArray
        assert isinstance(result, ValueArray)
        assert result.values.shape[0] == parameter_set.n_samples
        assert result.values.shape[1] >= 1  # At least one strategy
        assert np.all(np.isfinite(result.values))

    def test_sophisticated_adaptive_trial_simulator_with_true_parameters(self):
        """Test sophisticated_adaptive_trial_simulator with true parameters."""
        # Create test parameter set
        params = {
            "treatment_effect": np.random.normal(0.1, 0.05, 50),
            "control_rate": np.random.normal(0.3, 0.05, 50),
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=50),
            DecisionOption(name="Treatment B", sample_size=50)
        ]
        design = TrialDesign(arms=trial_arms)

        # Create adaptive rules
        rules = {
            "interim_analysis_points": [0.3],
        }

        # Define test adaptive rules
        rules = {
            "interim_analysis_points": [0.3],
        }

        # Test sophisticated adaptive trial simulator
        result = sophisticated_adaptive_trial_simulator(
            psa_samples=parameter_set,
            base_design=design,
            adaptive_rules=rules
        )

        # Check result type and properties
        from voiage.schema import ValueArray
        assert isinstance(result, ValueArray)
        assert result.values.shape[0] == parameter_set.n_samples
        assert np.all(np.isfinite(result.values))


class TestBayesianAdaptiveTrialSimulator:
    """Test the bayesian_adaptive_trial_simulator function."""

    def test_bayesian_adaptive_trial_simulator_basic(self):
        """Test basic functionality of bayesian_adaptive_trial_simulator."""
        # Create test parameter set
        params = {
            "treatment_effect": np.random.normal(0.1, 0.05, 50),
            "control_rate": np.random.normal(0.3, 0.05, 50),
            "cost_per_patient": np.random.normal(5000, 500, 50)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Control", sample_size=100),
            DecisionOption(name="Treatment", sample_size=100)
        ]
        design = TrialDesign(arms=trial_arms)

        # Create adaptive rules
        rules = {
            "interim_analysis_points": [0.3, 0.6],  # Two interim analyses
            "early_stopping_rules": {"efficacy": 0.9, "futility": 0.2},
            "sample_size_reestimation": False
        }

        # Test bayesian adaptive trial simulator
        result = bayesian_adaptive_trial_simulator(
            psa_samples=parameter_set,
            base_design=design,
            adaptive_rules=rules
        )

        # Check result type and properties
        from voiage.schema import ValueArray
        assert isinstance(result, ValueArray)
        assert result.values.shape[0] == parameter_set.n_samples
        assert result.values.shape[1] >= 1  # At least one strategy
        assert np.all(np.isfinite(result.values))
