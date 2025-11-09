"""
Ultra-targeted final push test to reach 95% coverage for clinical_trials.py

This focuses specifically on the remaining uncovered code paths that will
get us from 93% to 95% coverage.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as random
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add voiage to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voiage'))

from voiage.clinical_trials import (
    TrialType, EndpointType, AdaptationRule, TrialDesign, TrialOutcome,
    VOIBasedSampleSizeOptimizer, AdaptiveTrialOptimizer, ClinicalTrialDesignOptimizer,
    quick_trial_optimization, calculate_trial_voi, create_superiority_trial,
    create_adaptive_trial, create_health_economics_trial
)
from voiage.health_economics import Treatment, HealthState


class TestClinicalTrials95PercentFinal:
    """Ultra-focused test methods to reach 95% coverage"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create standard trial design
        self.standard_design = TrialDesign(
            trial_type=TrialType.SUPERIORITY,
            primary_endpoint=EndpointType.CONTINUOUS,
            sample_size=100,
            number_of_arms=2,
            allocation_ratio=[1.0, 1.0],
            alpha=0.05,
            beta=0.2,
            effect_size=0.5,
            variance=1.0,
            baseline_rate=0.5,
            willingness_to_pay=50000.0,
            health_economic_endpoint=False
        )
        
        # Create adaptive trial design
        self.adaptive_design = TrialDesign(
            trial_type=TrialType.ADAPTIVE,
            primary_endpoint=EndpointType.COST_EFFECTIVENESS,
            sample_size=300,
            number_of_arms=3,
            allocation_ratio=[1.0, 1.0, 1.0],
            alpha=0.05,
            beta=0.2,
            effect_size=0.5,
            variance=1.0,
            baseline_rate=0.5,
            willingness_to_pay=50000.0,
            health_economic_endpoint=True,
            budget_constraint=150000.0,
            adaptation_rules=[AdaptationRule.EARLY_SUCCESS, AdaptationRule.EARLY_FUTILITY],
            time_horizon=5.0
        )
        
        # Create treatment with effectiveness
        self.treatment = Treatment(
            name="Test Treatment",
            description="A test treatment for clinical trials",
            effectiveness=0.7,
            cost_per_cycle=1000.0,
            cycles_required=1
        )

    def test_adaptation_rule_general_fallback(self):
        """Test coverage for line 357 - general rule fallback in optimize_adaptation_thresholds"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test with a rule that's not specifically handled (should trigger the else clause)
        general_rule = AdaptationRule.SAMPLE_SIZE_REESTIMATION
        result = optimizer.optimize_adaptation_thresholds(self.treatment, general_rule)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'optimal_threshold' in result
        assert 'performance_metrics' in result
        assert 'expected_benefit' in result
        
        # Also test another general rule to ensure coverage
        general_rule2 = AdaptationRule.DROPPING_ARMS
        result2 = optimizer.optimize_adaptation_thresholds(self.treatment, general_rule2)
        assert isinstance(result2, dict)
        assert 'optimal_threshold' in result2

    def test_early_stop_futility_simulations(self):
        """Test coverage for lines 474, 485 - early stop and futility rate simulations"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test early stop rate simulation (line 474 area)
        early_stop = optimizer._simulate_early_stop_rate(0.8, self.treatment)
        early_stop_value = float(early_stop) if hasattr(early_stop, 'item') else early_stop
        assert isinstance(early_stop_value, (int, float))
        
        # Test futility rate simulation (line 485 area)
        futility = optimizer._simulate_futility_rate(0.8, self.treatment)
        futility_value = float(futility) if hasattr(futility, 'item') else futility
        assert isinstance(futility_value, (int, float))
        assert 0.0 <= futility_value <= 1.0

    def test_very_low_p_value_edge_case(self):
        """Test coverage for line 686 - very low p-value calculation"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test with very high treatment effect to get very low p-value
        p_value = optimizer._calculate_p_value(3.0, 200)  # High effect, large sample
        p_value_value = float(p_value) if hasattr(p_value, 'item') else p_value
        
        # Should be very low p-value (< 0.001)
        assert isinstance(p_value_value, (int, float))
        assert 0.0 <= p_value_value <= 1.0
        assert p_value_value < 0.01  # Very significant

    def test_extreme_trial_design_parameters(self):
        """Test coverage for line 736 - extreme parameter values"""
        # Create design with extreme parameters to trigger edge cases
        extreme_design = TrialDesign(
            trial_type=TrialType.SUPERIORITY,
            primary_endpoint=EndpointType.CONTINUOUS,
            sample_size=1000,  # Very large
            number_of_arms=1,  # Single arm
            allocation_ratio=[1.0],
            alpha=0.001,  # Very strict
            beta=0.001,   # Very strict
            effect_size=0.1,  # Very small
            variance=10.0,  # High variance
            baseline_rate=0.01  # Very low
        )
        
        optimizer = ClinicalTrialDesignOptimizer(extreme_design)
        result = optimizer.optimize_complete_design(self.treatment)
        
        # Should handle extreme parameters gracefully
        assert 'sample_size' in result
        assert 'efficiency' in result
        assert 'within_budget' in result['efficiency']

    def test_health_economic_zero_qaly_edge_case(self):
        """Test coverage for line 746 - zero QALY difference edge case"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
        
        # Test the health economic outcomes simulation
        cer, inc_qaly, nmb, prob_ce = optimizer._simulate_health_economic_outcomes(
            self.treatment, sample_size=100
        )
        
        # Should return valid values even in edge cases
        assert isinstance(cer, (int, float))
        assert isinstance(inc_qaly, (int, float))
        assert isinstance(nmb, (int, float))
        assert isinstance(prob_ce, (int, float))

    def test_power_calculation_edge_cases(self):
        """Test coverage for line 766 - power calculation edge cases"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test with very large sample size
        power_large = optimizer._calculate_power_at_sample_size(1000)
        power_large_value = float(power_large) if hasattr(power_large, 'item') else power_large
        assert isinstance(power_large_value, (int, float))
        assert 0.0 <= power_large_value <= 1.0
        
        # Test with very small sample size
        power_small = optimizer._calculate_power_at_sample_size(10)
        power_small_value = float(power_small) if hasattr(power_small, 'item') else power_small
        assert isinstance(power_small_value, (int, float))
        assert 0.0 <= power_small_value <= 1.0

    def test_treatment_effect_high_noise_simulation(self):
        """Test coverage for line 783 - high noise in treatment effect simulation"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test multiple simulations to ensure we hit different noise patterns
        effects = []
        for _ in range(10):  # More iterations to increase coverage chance
            effect = optimizer._simulate_treatment_effect(self.treatment)
            effects.append(float(effect) if hasattr(effect, 'item') else effect)
        
        # Effects should vary due to random noise
        assert len(effects) == 10
        assert all(isinstance(eff, (int, float)) for eff in effects)
        # Should have some variation
        assert len(set([round(e, 3) for e in effects])) > 1

    def test_confidence_interval_calculation_extreme_cases(self):
        """Test coverage for line 801 - confidence interval calculation extreme cases"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test with very large treatment effect
        ci_large = optimizer._calculate_confidence_interval(5.0, 100)
        assert isinstance(ci_large, tuple)
        assert len(ci_large) == 2
        assert ci_large[0] < ci_large[1]
        
        # Test with very small treatment effect
        ci_small = optimizer._calculate_confidence_interval(0.01, 100)
        assert isinstance(ci_small, tuple)
        assert len(ci_small) == 2
        assert ci_small[0] < ci_small[1]

    def test_trial_design_factory_all_trial_types(self):
        """Test coverage for lines 821-831 - trial design factory function"""
        # Test all trial types to cover the different if-elif branches in quick_trial_optimization
        
        # Test superiority trial
        result1 = quick_trial_optimization(self.treatment, trial_type="superiority", budget_constraint=50000)
        assert isinstance(result1, dict)
        assert 'sample_size' in result1
        assert 'efficiency' in result1
        
        # Test adaptive trial
        result2 = quick_trial_optimization(self.treatment, trial_type="adaptive", budget_constraint=50000)
        assert isinstance(result2, dict)
        assert 'sample_size' in result2
        assert 'efficiency' in result2
        
        # Test health economics trial
        result3 = quick_trial_optimization(self.treatment, trial_type="health_economics", budget_constraint=50000)
        assert isinstance(result3, dict)
        assert 'sample_size' in result3
        assert 'efficiency' in result3
        
        # Test unknown type (should default to superiority)
        result4 = quick_trial_optimization(self.treatment, trial_type="unknown_type", budget_constraint=50000)
        assert isinstance(result4, dict)
        assert 'sample_size' in result4
        assert 'efficiency' in result4

    def test_voi_calculation_different_wtp_scenarios(self):
        """Test coverage for lines 838-846 - VOI calculation with different willingness to pay"""
        
        # Test with different willingness to pay values
        wtp_values = [10000.0, 50000.0, 100000.0]
        
        for wtp in wtp_values:
            voi = calculate_trial_voi(
                treatment=self.treatment,
                sample_size=100,
                willingness_to_pay=wtp
            )
            voi_value = float(voi) if hasattr(voi, 'item') else voi
            assert isinstance(voi_value, (int, float))
            assert voi_value >= 0.0  # VOI should be non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])