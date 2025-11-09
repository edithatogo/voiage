"""
Comprehensive test suite for clinical_trials.py

This test file provides comprehensive coverage for the clinical trial design optimization module,
targeting specific line ranges identified in coverage analysis to achieve >95% coverage.
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
    VOIBasedSampleSizeOptimizer, AdaptiveTrialOptimizer, ClinicalTrialDesignOptimizer
)
from voiage.health_economics import Treatment, HealthState


class TestClinicalTrialsComprehensive:
    """Comprehensive test suite for clinical trials module"""
    
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

    def test_adaptive_optimizer_unknown_adaptation_rule(self):
        """Test coverage for unknown adaptation rule fallback (lines 356-359)"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
    
        # Test with unknown adaptation rule
        unknown_rule = AdaptationRule.DROPPING_ARMS
        result = optimizer.adaptive_optimizer.optimize_adaptation_thresholds(
            self.treatment, unknown_rule
        )
    
        # Should call _optimize_general_threshold
        assert isinstance(result, dict)
        assert 'optimal_threshold' in result
        assert 'performance_metrics' in result

    def test_adaptive_optimizer_optimize_adaptation_schedule_comprehensive(self):
        """Test comprehensive adaptation schedule optimization (lines 474, 485, 553-563)"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
    
        # Test with max_interim_analyses = 1 (avoid empty iterable)
        result_min = optimizer.adaptive_optimizer.optimize_adaptation_schedule(
            self.treatment, max_interim_analyses=1
        )
        assert 'optimal_schedule' in result_min
        assert 'num_adaptations' in result_min['optimal_schedule']

        # Test with standard optimization
        result = optimizer.adaptive_optimizer.optimize_adaptation_schedule(
            self.treatment, max_interim_analyses=5
        )
        assert 'optimal_schedule' in result
        assert 'expected_voi' in result['optimal_schedule']

    def test_adaptive_optimizer_early_futility_threshold_optimization(self):
        """Test early futility threshold optimization with edge cases"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
    
        result = optimizer.adaptive_optimizer._optimize_early_futility_threshold(self.treatment)
        assert isinstance(result, dict)
        assert 'optimal_threshold' in result
        assert 'performance_metrics' in result
        assert 'expected_benefit' in result

    def test_clinical_trial_design_comprehensive_optimization(self):
        """Test comprehensive optimization with different trial types and edge cases"""
    
        # Test with adaptive design
        adaptive_optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
        result_adaptive = adaptive_optimizer.optimize_complete_design(
            self.treatment
        )
        
        # Should have complete optimization results
        assert 'sample_size' in result_adaptive
        assert 'adaptation' in result_adaptive
        assert 'thresholds' in result_adaptive
        assert 'efficiency' in result_adaptive
        assert 'recommendations' in result_adaptive

        # Test with budget constraint
        result_with_budget = adaptive_optimizer.optimize_complete_design(
            self.treatment, budget_constraint=200000.0
        )
        assert 'efficiency' in result_with_budget
        assert 'within_budget' in result_with_budget['efficiency']

    def test_efficiency_metrics_comprehensive(self):
        """Test efficiency metrics calculation (lines 661-663, 686, 690-691)"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
    
        # Create mock optimization results
        mock_results = {
            'sample_size': {'optimal_sample_size': 200, 'max_net_benefit': 50000.0, 
                          'total_voi_at_optimal': 40000.0, 'total_cost_at_optimal': 200000.0},
            'adaptation': {
                'optimal_schedule': {
                    'expected_voi': 25000.0,
                    'num_adaptations': 2
                }
            }
        }
    
        efficiency = optimizer._calculate_trial_efficiency(mock_results, 100000.0)
        assert isinstance(efficiency, dict)
        assert 'voi_per_dollar' in efficiency
        assert 'cost_per_voi' in efficiency
        assert 'total_cost' in efficiency
        assert 'total_voi' in efficiency
        assert 'within_budget' in efficiency
        assert 'total_efficiency' in efficiency

    def test_treatment_effect_simulation_edge_cases(self):
        """Test treatment effect simulation (lines 766, 783, 801)"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
    
        # Test with treatment that has effectiveness attribute
        treatment_with_effectiveness = Treatment(
            name="Treatment with Effectiveness",
            description="A treatment with effectiveness attribute",
            effectiveness=0.8,
            cost_per_cycle=1500.0,
            cycles_required=1
        )
    
        effect1 = optimizer._simulate_treatment_effect(treatment_with_effectiveness)
        # JAX arrays should be converted to float for assertions
        effect1_value = float(effect1) if hasattr(effect1, 'item') else effect1
        assert isinstance(effect1_value, (int, float))
        
        # Test with treatment with very low effectiveness (uses default behavior)
        treatment_no_effectiveness = Treatment(
            name="Treatment with Low Effectiveness",
            description="A treatment with low effectiveness",
            effectiveness=0.1,  # Low effectiveness to test edge case
            cost_per_cycle=1500.0,
            cycles_required=1
        )
        
        effect2 = optimizer._simulate_treatment_effect(treatment_no_effectiveness)
        effect2_value = float(effect2) if hasattr(effect2, 'item') else effect2
        assert isinstance(effect2_value, (int, float))
        # Should be around 0.1 (effectiveness) with some noise (can be slightly negative)
        assert -0.5 <= effect2_value <= 0.5

    def test_health_economic_outcomes_simulation(self):
        """Test health economic outcomes simulation"""
        optimizer = ClinicalTrialDesignOptimizer(self.adaptive_design)
        
        # Test with health economic endpoint
        cer, inc_qaly, nmb, prob_ce = optimizer._simulate_health_economic_outcomes(
            self.treatment, sample_size=100
        )
        
        assert isinstance(cer, (int, float))
        assert isinstance(inc_qaly, (int, float))
        assert isinstance(nmb, (int, float))
        assert isinstance(prob_ce, (int, float))

    def test_adaptive_optimizer_power_calculations(self):
        """Test power calculation methods (lines 501-507)"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test power calculation
        power = optimizer._calculate_power_at_sample_size(100)
        power_value = float(power) if hasattr(power, 'item') else power
        assert isinstance(power_value, (int, float))
        assert 0.0 <= power_value <= 1.0
        
        # Test power with zero sample size
        power_zero = optimizer._calculate_power_at_sample_size(0)
        power_zero_value = float(power_zero) if hasattr(power_zero, 'item') else power_zero
        assert power_zero_value == 0.0

    def test_adaptive_optimizer_early_scenarios(self):
        """Test early stopping scenario simulations (line 542)"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test early stop rate simulation
        early_stop = optimizer._simulate_early_stop_rate(0.8, self.treatment)
        early_stop_value = float(early_stop) if hasattr(early_stop, 'item') else early_stop
        assert isinstance(early_stop_value, (int, float))
        assert 0.0 <= early_stop_value <= 1.0
        
        # Test futility rate simulation
        futility = optimizer._simulate_futility_rate(0.2, self.treatment)
        futility_value = float(futility) if hasattr(futility, 'item') else futility
        assert isinstance(futility_value, (int, float))
        assert 0.0 <= futility_value <= 1.0
        
        # Test power loss simulation
        power_loss = optimizer._simulate_power_loss(0.3, self.treatment)
        power_loss_value = float(power_loss) if hasattr(power_loss, 'item') else power_loss
        assert isinstance(power_loss_value, (int, float))
        assert 0.0 <= power_loss_value <= 1.0

    def test_sample_size_optimizer_power_increase(self):
        """Test power increase calculation (lines 363-385)"""
        optimizer = VOIBasedSampleSizeOptimizer(self.standard_design)
        
        # Test power increase calculation
        power_increase = optimizer._calculate_power_increase(50, 100)
        power_increase_value = float(power_increase) if hasattr(power_increase, 'item') else power_increase
        assert isinstance(power_increase_value, (int, float))
        assert power_increase_value >= 0.0
        
        # Test with invalid sample sizes
        power_increase_zero = optimizer._calculate_power_increase(0, 0)
        power_increase_zero_value = float(power_increase_zero) if hasattr(power_increase_zero, 'item') else power_increase_zero
        assert power_increase_zero_value >= 0.0

    def test_adaptive_optimizer_sample_size_reestimation(self):
        """Test sample size re-estimation optimization (lines 424-433)"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        result = optimizer._optimize_sample_size_reest_threshold(self.treatment)
        assert isinstance(result, dict)
        assert 'optimal_threshold' in result
        assert 'performance_metrics' in result
        assert 'expected_benefit' in result
        assert 'note' in result

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge case coverage"""
        
        # Test with invalid trial design
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test simulation with treatment
        outcome = optimizer.simulate_trial_outcomes(self.treatment, {
            'sample_size': {'optimal_sample_size': 100}
        })
        
        assert isinstance(outcome, TrialOutcome)
        assert hasattr(outcome, 'treatment_effect')
        assert hasattr(outcome, 'p_value')
        assert hasattr(outcome, 'confidence_interval')
        assert hasattr(outcome, 'power_achieved')

    def test_trial_outcome_comprehensive(self):
        """Test trial outcome data structure"""
        outcome = TrialOutcome(
            treatment_effect=0.5,
            p_value=0.03,
            confidence_interval=(0.2, 0.8),
            power_achieved=0.8,
            sample_size_used=100
        )
        
        assert outcome.treatment_effect == 0.5
        assert outcome.p_value == 0.03
        assert outcome.confidence_interval == (0.2, 0.8)
        assert outcome.power_achieved == 0.8
        assert outcome.sample_size_used == 100

    def test_integration_scenarios(self):
        """Test integration scenarios that combine multiple components"""
        
        # Test full integration: sample size optimization + adaptive design + health economics
        he_adaptive_design = TrialDesign(
            trial_type=TrialType.ADAPTIVE,
            primary_endpoint=EndpointType.COST_EFFECTIVENESS,
            sample_size=300,
            number_of_arms=2,
            allocation_ratio=[1.0, 1.0],
            alpha=0.05,
            beta=0.2,
            effect_size=0.5,
            variance=1.0,
            baseline_rate=0.5,
            willingness_to_pay=50000.0,
            health_economic_endpoint=True,
            budget_constraint=150000.0,
            time_horizon=5.0,
            adaptation_rules=[AdaptationRule.EARLY_SUCCESS, AdaptationRule.EARLY_FUTILITY]
        )
        
        he_optimizer = ClinicalTrialDesignOptimizer(he_adaptive_design)
        he_result = he_optimizer.optimize_complete_design(self.treatment)
        
        # Should have comprehensive results
        assert 'sample_size' in he_result
        assert 'adaptation' in he_result
        assert 'efficiency' in he_result
        assert 'recommendations' in he_result

    def test_design_recommendations_generation(self):
        """Test design recommendations generation"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        mock_results = {
            'sample_size': {
                'optimal_sample_size': 200,
                'max_net_benefit': 50000.0
            },
            'efficiency': {
                'voi_per_dollar': 0.2,
                'within_budget': True
            }
        }
        
        recommendations = optimizer._generate_design_recommendations(mock_results)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_various_trial_types(self):
        """Test different trial types and configurations"""
        trial_types = [
            TrialType.SUPERIORITY,
            TrialType.NON_INFERIORITY,
            TrialType.SUPERIORITY_WITH_HEALTH_ECONOMICS,
            TrialType.ADAPTIVE
        ]
        
        for trial_type in trial_types:
            design = TrialDesign(
                trial_type=trial_type,
                primary_endpoint=EndpointType.CONTINUOUS,
                sample_size=100,
                adaptation_rules=[AdaptationRule.EARLY_SUCCESS] if trial_type == TrialType.ADAPTIVE else []
            )
            
            optimizer = ClinicalTrialDesignOptimizer(design)
            result = optimizer.optimize_complete_design(self.treatment)
            
            assert 'sample_size' in result
            assert isinstance(result, dict)

    def test_power_and_confidence_calculations(self):
        """Test power and confidence interval calculations"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test p-value calculation
        p_value = optimizer._calculate_p_value(0.5, 100)
        p_value_value = float(p_value) if hasattr(p_value, 'item') else p_value
        assert isinstance(p_value_value, (int, float))
        assert 0.0 <= p_value_value <= 1.0
        
        # Test confidence interval calculation
        ci = optimizer._calculate_confidence_interval(0.5, 100)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        
        # Test with small sample size
        ci_small = optimizer._calculate_confidence_interval(0.5, 10)
        assert isinstance(ci_small, tuple)
        assert len(ci_small) == 2

    def test_comprehensive_edge_case_testing(self):
        """Test comprehensive edge cases and boundary conditions"""
        optimizer = ClinicalTrialDesignOptimizer(self.standard_design)
        
        # Test with very high effect size
        high_effect_treatment = Treatment(
            name="High Effect Treatment",
            description="Treatment with high effect size",
            effectiveness=0.99,
            cost_per_cycle=1000.0,
            cycles_required=1
        )
        
        effect = optimizer._simulate_treatment_effect(high_effect_treatment)
        effect_value = float(effect) if hasattr(effect, 'item') else effect
        assert isinstance(effect_value, (int, float))
        
        # Test with very low effect size
        low_effect_treatment = Treatment(
            name="Low Effect Treatment",
            description="Treatment with low effect size",
            effectiveness=0.01,
            cost_per_cycle=1000.0,
            cycles_required=1
        )
        
        effect_low = optimizer._simulate_treatment_effect(low_effect_treatment)
        effect_low_value = float(effect_low) if hasattr(effect_low, 'item') else effect_low
        assert isinstance(effect_low_value, (int, float))

    def test_adaptive_schedule_voi_calculations(self):
        """Test adaptation schedule VOI calculations (lines 474-478, 482-489)"""
        optimizer = AdaptiveTrialOptimizer(self.adaptive_design)
        
        # Test early decision VOI calculation
        early_voi = optimizer._calculate_early_decision_voi(self.treatment, 50, 200)
        early_voi_value = float(early_voi) if hasattr(early_voi, 'item') else early_voi
        assert isinstance(early_voi_value, (int, float))
        assert early_voi_value >= 0.0
        
        # Test adaptation schedule VOI calculation
        adaptation_times = [50, 100, 150]
        schedule_voi = optimizer._calculate_adaptation_schedule_voi(
            self.treatment, adaptation_times, 200
        )
        schedule_voi_value = float(schedule_voi) if hasattr(schedule_voi, 'item') else schedule_voi
        assert isinstance(schedule_voi_value, (int, float))
        assert schedule_voi_value >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])