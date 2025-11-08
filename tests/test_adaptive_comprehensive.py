"""Tests for voiage.methods.adaptive module to increase coverage to >90%."""

import numpy as np
import pytest

from voiage.methods.adaptive import adaptive_evsi, sophisticated_adaptive_trial_simulator, bayesian_adaptive_trial_simulator
from voiage.schema import ParameterSet, ValueArray, TrialDesign, DecisionOption
from voiage.exceptions import InputError


class TestAdaptiveEVSI:
    """Test the adaptive_evsi function comprehensively."""
    
    def test_adaptive_evsi_basic(self):
        """Test basic functionality of adaptive_evsi."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
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
        params = {
            "effectiveness": np.random.normal(0.1, 0.05, 100),
            "cost": np.random.normal(5000, 500, 100)
        }
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
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1},
            "sample_size_reestimation": True
        }

        # Test adaptive_evsi calculation
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Result should be a non-negative float
        assert isinstance(result, float)
        assert result >= 0

    def test_adaptive_evsi_with_population_scaling(self):
        """Test adaptive_evsi with population scaling."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
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
        params = {
            "effectiveness": np.random.normal(0.1, 0.05, 50),
            "cost": np.random.normal(5000, 500, 50)
        }
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
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1}
        }

        # Test with population scaling
        result_scaled = adaptive_evsi(
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

        # Test without population scaling
        result_unscaled = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Both should be floats and non-negative
        assert isinstance(result_scaled, float)
        assert isinstance(result_unscaled, float)
        assert result_scaled >= 0
        assert result_unscaled >= 0
        
        # The scaled result should be >= the unscaled result (when population > 1)
        if result_unscaled > 0:
            assert result_scaled >= result_unscaled

    def test_adaptive_evsi_no_adaptation(self):
        """Test adaptive_evsi with no adaptive rules (fixed design)."""
        # Create a simple adaptive trial simulator
        def simple_fixed_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple fixed trial simulator for testing."""
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
        params = {
            "effectiveness": np.random.normal(0.1, 0.05, 50)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Test with only base design and no adaptive rules
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_fixed_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules={},
            n_outer_loops=3,
            n_inner_loops=5
        )

        # Should return a non-negative float
        assert isinstance(result, float)
        assert result >= 0

    def test_adaptive_evsi_input_validation(self):
        """Test adaptive_evsi with invalid inputs."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
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
        params = {
            "effectiveness": np.random.normal(0.1, 0.05, 50)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Test invalid adaptive_trial_simulator
        with pytest.raises(InputError, match="`adaptive_trial_simulator` must be a callable function"):
            adaptive_evsi(
                adaptive_trial_simulator="not a function",
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={}
            )

        # Test invalid psa_prior
        with pytest.raises(InputError, match="`psa_prior` must be a PSASample"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior="not a psa",
                base_trial_design=trial_design,
                adaptive_rules={}
            )

        # Test invalid base_trial_design
        with pytest.raises(InputError, match="`base_trial_design` must be a TrialDesign"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design="not a design",
                adaptive_rules={}
            )

        # Test invalid adaptive_rules
        with pytest.raises(InputError, match="`adaptive_rules` must be a dictionary"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules="not a dict"
            )

        # Test invalid loop parameters
        with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                n_outer_loops=0,
                n_inner_loops=5
            )

        with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                n_outer_loops=5,
                n_inner_loops=0
            )

        with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                n_outer_loops=-5,
                n_inner_loops=5
            )

    def test_adaptive_evsi_population_validation(self):
        """Test adaptive_evsi with population validation."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
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
        params = {
            "effectiveness": np.random.normal(0.1, 0.05, 50)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=100),
            DecisionOption(name="Treatment B", sample_size=100)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Test invalid population (negative)
        with pytest.raises(InputError, match="Population must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                population=-1000,
                time_horizon=10,
                n_outer_loops=3,
                n_inner_loops=5
            )

        # Test invalid time_horizon (negative)
        with pytest.raises(InputError, match="Time horizon must be positive"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                population=100000,
                time_horizon=-10,
                n_outer_loops=3,
                n_inner_loops=5
            )

        # Test invalid discount_rate (out of bounds)
        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                population=100000,
                time_horizon=10,
                discount_rate=1.5,
                n_outer_loops=3,
                n_inner_loops=5
            )

        with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
            adaptive_evsi(
                adaptive_trial_simulator=simple_adaptive_simulator,
                psa_prior=parameter_set,
                base_trial_design=trial_design,
                adaptive_rules={},
                population=100000,
                time_horizon=10,
                discount_rate=-0.1,
                n_outer_loops=3,
                n_inner_loops=5
            )

    def test_adaptive_evsi_always_non_negative(self):
        """Test that adaptive_evsi always returns a non-negative value."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies with very similar values
            # to simulate a case where the adaptive trial might not provide value
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make sure strategies have very similar values
            nb_values[:, 1] = nb_values[:, 0] + np.random.normal(0, 1, n_samples)  # Very small difference

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

        # Create test parameter set with very little uncertainty
        params = {
            "effectiveness": np.full(100, 0.5),  # Constant parameter to minimize value
            "cost": np.full(100, 5000)
        }
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
        }

        # Calculate adaptive EVSI - even if the adaptive trial might not be valuable, 
        # the function should return 0 or a positive value
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=3,
            n_inner_loops=5
        )

        assert result >= 0
        assert isinstance(result, float)

    def test_adaptive_evsi_edge_case_single_sample(self):
        """Test adaptive_evsi with single sample."""
        # Create a simple adaptive trial simulator
        def simple_adaptive_simulator(psa_samples, trial_design=None, trial_data=None):
            """Simple adaptive trial simulator for testing."""
            n_samples = psa_samples.n_samples
            # Create net benefits for 2 strategies - just one sample
            nb_values = np.random.rand(n_samples, 2) * 1000
            # Make sure strategies have similar values
            nb_values[:, 1] = nb_values[:, 0] + np.random.normal(0, 10, n_samples)

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

        # Create test parameter set with just one sample
        params = {
            "effectiveness": np.array([0.5]),  # Single value
            "cost": np.array([5000])
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Create trial design
        trial_arms = [
            DecisionOption(name="Treatment A", sample_size=10),
            DecisionOption(name="Treatment B", sample_size=10)
        ]
        trial_design = TrialDesign(arms=trial_arms)

        # Define simple adaptive rules
        adaptive_rules = {}

        # Calculate adaptive EVSI with single sample
        result = adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_simulator,
            psa_prior=parameter_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            n_outer_loops=2,
            n_inner_loops=3
        )

        # Result should be a non-negative float
        assert isinstance(result, float)
        assert result >= 0


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
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1},
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

    def test_bayesian_adaptive_trial_simulator_with_true_parameters(self):
        """Test bayesian_adaptive_trial_simulator with true parameters."""
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
            "interim_analysis_points": [0.5],
        }

        # Define true parameters
        true_params = {
            "treatment_effect": 0.15,
            "control_rate": 0.28,
        }

        # Test with true parameters
        result = bayesian_adaptive_trial_simulator(
            psa_samples=parameter_set,
            base_design=design,
            adaptive_rules=rules,
            true_parameters=true_params
        )

        # Check result type and properties
        assert isinstance(result, ValueArray)
        assert result.values.shape[0] == parameter_set.n_samples
        assert np.all(np.isfinite(result.values))


def test_import_functionality():
    """Test that the adaptive methods are importable and available."""
    from voiage.methods.adaptive import adaptive_evsi, sophisticated_adaptive_trial_simulator, bayesian_adaptive_trial_simulator
    
    # Verify functions exist
    assert callable(adaptive_evsi)
    assert callable(sophisticated_adaptive_trial_simulator)
    assert callable(bayesian_adaptive_trial_simulator)
    
    print("✅ All adaptive VOI methods are importable and available")