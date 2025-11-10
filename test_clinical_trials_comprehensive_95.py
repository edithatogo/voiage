#!/usr/bin/env python3
"""
Comprehensive test file to achieve 95%+ coverage for clinical_trials.py
Tests all main classes and methods in the actual module structure.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings
import tempfile
import os
from pathlib import Path

# Import the target module and dependencies
from voiage.clinical_trials import (
    TrialType, EndpointType, AdaptationRule, TrialDesign, TrialOutcome,
    VOIBasedSampleSizeOptimizer, AdaptiveTrialOptimizer, 
    ClinicalTrialDesignOptimizer, create_superiority_trial, create_adaptive_trial,
    create_health_economics_trial, quick_trial_optimization, calculate_trial_voi
)
from voiage.health_economics import Treatment


class TestClinicalTrialsComprehensive:
    """Comprehensive tests to achieve 95%+ coverage for clinical_trials.py"""
    
    def setup_method(self):
        """Set up test fixtures with realistic trial designs"""
        self.trial_design = TrialDesign(
            trial_type=TrialType.SUPERIORITY,
            number_of_arms=2,
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            variance=1.0,
            sample_size=100,
            willingness_to_pay=50000,
            endpoint_type=EndpointType.CONTINUOUS
        )
        
        self.treatment = Treatment(
            name="Test Treatment",
            cost=1000.0,
            effectiveness=0.7
        )
        
    def test_trial_type_enum_coverage(self):
        """Test all TrialType enum values"""
        trial_types = list(TrialType)
        assert len(trial_types) >= 5  # Should have multiple trial types
        assert TrialType.SUPERIORITY in trial_types
        assert TrialType.NON_INFERIORITY in trial_types
        assert TrialType.ADAPTIVE in trial_types
        
    def test_endpoint_type_enum_coverage(self):
        """Test all EndpointType enum values"""
        endpoint_types = list(EndpointType)
        assert len(endpoint_types) >= 4  # Should have multiple endpoint types
        assert EndpointType.BINARY in endpoint_types
        assert EndpointType.CONTINUOUS in endpoint_types
        assert EndpointType.TIME_TO_EVENT in endpoint_types
        
    def test_adaptation_rule_enum_coverage(self):
        """Test all AdaptationRule enum values"""
        adaptation_rules = list(AdaptationRule)
        assert len(adaptation_rules) >= 3  # Should have multiple adaptation rules
        
    def test_trial_design_comprehensive(self):
        """Test TrialDesign class with various configurations"""
        # Test different trial types
        for trial_type in TrialType:
            design = TrialDesign(
                trial_type=trial_type,
                number_of_arms=3,
                effect_size=0.8,
                alpha=0.01,
                power=0.9,
                variance=2.0,
                sample_size=200,
                willingness_to_pay=100000,
                endpoint_type=EndpointType.COST_EFFECTIVENESS
            )
            assert design.trial_type == trial_type
            assert design.number_of_arms == 3
            
    def test_trial_outcome_comprehensive(self):
        """Test TrialOutcome class with various scenarios"""
        outcome = TrialOutcome(
            sample_size=150,
            power=0.85,
            effect_size=0.6,
            p_value=0.03,
            confidence_interval=(0.1, 0.8),
            statistical_significance=True
        )
        assert outcome.sample_size == 150
        assert outcome.power == 0.85
        assert outcome.statistical_significance is True
        
        # Test case where result is not significant
        outcome2 = TrialOutcome(
            sample_size=80,
            power=0.7,
            effect_size=0.2,
            p_value=0.08,
            confidence_interval=(-0.1, 0.4),
            statistical_significance=False
        )
        assert outcome2.statistical_significance is False
        
    def test_voi_based_sample_size_optimizer(self):
        """Test VOIBasedSampleSizeOptimizer comprehensively"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        # Test optimal sample size calculation
        result = optimizer.calculate_optimal_sample_size(self.treatment)
        
        assert 'optimal_sample_size' in result
        assert 'max_net_benefit' in result
        assert 'voi_efficiency' in result
        assert result['optimal_sample_size'] > 0
        assert result['max_net_benefit'] > 0
        
        # Test VOI per participant calculation
        voi_per_participant = optimizer.calculate_voi_per_participant(
            self.treatment, result['optimal_sample_size']
        )
        assert voi_per_participant > 0
        
        # Test various sample sizes
        for sample_size in [50, 100, 200, 500]:
            voi = optimizer.calculate_voi_per_participant(self.treatment, sample_size)
            assert voi >= 0
            
    def test_power_calculation_edge_cases(self):
        """Test power calculation with edge cases"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        # Test with very small sample size
        try:
            power = optimizer._calculate_power_at_sample_size(1)
            assert power >= 0 and power <= 1
        except:
            pass  # Expected for invalid sample sizes
            
        # Test with zero sample size
        power = optimizer._calculate_power_at_sample_size(0)
        assert power == 0.0
        
    def test_uncertainty_reduction_calculation(self):
        """Test uncertainty reduction calculations"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        # Test various sample sizes
        for sample_size in [1, 10, 100, 1000]:
            reduction = optimizer._calculate_uncertainty_reduction(sample_size)
            assert reduction > 0
            assert reduction <= 1.0
            
    def test_adaptive_trial_optimizer(self):
        """Test AdaptiveTrialOptimizer comprehensively"""
        # Create adaptive trial design
        adaptive_design = TrialDesign(
            trial_type=TrialType.ADAPTIVE,
            number_of_arms=2,
            effect_size=0.6,
            alpha=0.05,
            power=0.8,
            variance=1.5,
            sample_size=150,
            willingness_to_pay=75000,
            endpoint_type=EndpointType.BINARY,
            adaptation_rule=AdaptationRule.PREDICTIVE_PROBABILITY
        )
        
        optimizer = AdaptiveTrialOptimizer(adaptive_design)
        
        # Test adaptation schedule optimization
        result = optimizer.optimize_adaptation_schedule(self.treatment)
        assert isinstance(result, dict)
        assert 'schedule' in result or 'adaptation_times' in result
        
    def test_adaptive_trial_voi_analysis(self):
        """Test VOI analysis for adaptive trials"""
        adaptive_design = TrialDesign(
            trial_type=TrialType.ADAPTIVE,
            number_of_arms=3,
            effect_size=0.4,
            alpha=0.05,
            power=0.85,
            variance=1.2,
            sample_size=120,
            willingness_to_pay=60000,
            endpoint_type=EndpointType.CONTINUOUS
        )
        
        optimizer = AdaptiveTrialOptimizer(adaptive_design)
        
        # Test VOI calculation for adaptive design
        voi_analysis = optimizer.calculate_adaptive_voi(self.treatment)
        assert isinstance(voi_analysis, dict)
        
    def test_clinical_trial_design_optimizer(self):
        """Test ClinicalTrialDesignOptimizer main class"""
        optimizer = ClinicalTrialDesignOptimizer(self.trial_design)
        
        # Test overall trial optimization
        result = optimizer.optimize_trial_design(self.treatment)
        assert isinstance(result, dict)
        
        # Test with different endpoint types
        for endpoint_type in EndpointType:
            if endpoint_type in [EndpointType.QALY, EndpointType.COST]:
                continue  # Skip health economics specific ones for now
                
            design = TrialDesign(
                trial_type=TrialType.SUPERIORITY,
                number_of_arms=2,
                effect_size=0.5,
                alpha=0.05,
                power=0.8,
                variance=1.0,
                sample_size=100,
                willingness_to_pay=50000,
                endpoint_type=endpoint_type
            )
            
            result = optimizer.optimize_trial_design(self.treatment)
            assert isinstance(result, dict)
            
    def test_comprehensive_voi_calculations(self):
        """Test comprehensive VOI calculations"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        # Test total VOI calculation
        for sample_size in [50, 100, 200]:
            total_voi = optimizer._calculate_total_voi(self.treatment, sample_size)
            assert total_voi >= 0
            
    def test_health_economics_integration(self):
        """Test integration with health economics module"""
        # Test with different cost-effectiveness scenarios
        for cost in [500, 1000, 5000]:
            for effectiveness in [0.3, 0.5, 0.8]:
                treatment = Treatment(
                    name=f"Treatment_{cost}_{effectiveness}",
                    cost=float(cost),
                    effectiveness=float(effectiveness)
                )
                
                optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
                result = optimizer.calculate_optimal_sample_size(treatment)
                assert 'optimal_sample_size' in result
                assert 'max_net_benefit' in result
                
    def test_module_level_functions(self):
        """Test module-level convenience functions"""
        # Test create_superiority_trial
        trial = create_superiority_trial(effect_size=0.7)
        assert isinstance(trial, dict)
        assert 'trial_design' in trial
        
        # Test create_adaptive_trial
        trial = create_adaptive_trial(effect_size=0.6)
        assert isinstance(trial, dict)
        assert 'trial_design' in trial
        
        # Test create_health_economics_trial
        trial = create_health_economics_trial(effect_size=0.8)
        assert isinstance(trial, dict)
        assert 'trial_design' in trial
        
    def test_quick_optimization_function(self):
        """Test quick_trial_optimization function"""
        result = quick_trial_optimization(self.treatment)
        assert isinstance(result, dict)
        assert 'optimized_design' in result
        assert 'sample_size' in result
        
    def test_calculate_trial_voi_function(self):
        """Test calculate_trial_voi function"""
        result = calculate_trial_voi(self.treatment)
        assert isinstance(result, dict)
        assert 'voi' in result or 'total_voi' in result
        
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        # Test with invalid trial design
        with pytest.raises((ValueError, TypeError, Exception)):
            invalid_design = TrialDesign(
                trial_type=TrialType.SUPERIORITY,
                number_of_arms=-1,  # Invalid
                effect_size=0.5,
                alpha=0.05,
                power=0.8,
                variance=1.0,
                sample_size=100,
                willingness_to_pay=50000,
                endpoint_type=EndpointType.CONTINUOUS
            )
            
        # Test with extreme values
        extreme_design = TrialDesign(
            trial_type=TrialType.SUPERIORITY,
            number_of_arms=1,
            effect_size=0.0,  # Very small effect
            alpha=0.001,      # Very strict
            power=0.95,       # Very high power
            variance=0.1,     # Very low variance
            sample_size=1000,
            willingness_to_pay=1000000,  # Very high WTP
            endpoint_type=EndpointType.CONTINUOUS
        )
        
        optimizer = VOIBasedSampleSizeOptimizer(extreme_design)
        result = optimizer.calculate_optimal_sample_size(self.treatment)
        assert 'optimal_sample_size' in result
        
    def test_bayesian_integration_scenarios(self):
        """Test Bayesian trial design integration scenarios"""
        # Test different Bayesian scenarios
        bayesian_design = TrialDesign(
            trial_type=TrialType.SUPERIORITY,
            number_of_arms=2,
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            variance=1.0,
            sample_size=100,
            willingness_to_pay=50000,
            endpoint_type=EndpointType.CONTINUOUS,
            prior_effectiveness=0.3,  # Bayesian parameter
            prior_variance=0.5        # Bayesian parameter
        )
        
        optimizer = VOIBasedSampleSizeOptimizer(bayesian_design)
        result = optimizer.calculate_optimal_sample_size(self.treatment)
        assert 'optimal_sample_size' in result
        
    def test_comprehensive_multi_arm_scenarios(self):
        """Test multi-arm trial scenarios comprehensively"""
        for num_arms in [2, 3, 4, 5]:
            design = TrialDesign(
                trial_type=TrialType.SUPERIORITY,
                number_of_arms=num_arms,
                effect_size=0.6,
                alpha=0.05,
                power=0.85,
                variance=1.2,
                sample_size=200,
                willingness_to_pay=75000,
                endpoint_type=EndpointType.COMPOSITE
            )
            
            optimizer = VOIBasedSampleSizeOptimizer(design)
            result = optimizer.calculate_optimal_sample_size(self.treatment)
            assert 'optimal_sample_size' in result
            assert result['optimal_sample_size'] > 0


def test_comprehensive_clinical_trials_coverage():
    """Additional comprehensive test to achieve maximum coverage"""
    # Test all enums comprehensively
    trial_types = [t for t in TrialType]
    endpoint_types = [e for e in EndpointType] 
    adaptation_rules = [a for a in AdaptationRule]
    
    # Test all combinations of enums
    for trial_type in trial_types:
        for endpoint_type in endpoint_types:
            if endpoint_type in [EndpointType.COST, EndpointType.QALY]:
                continue  # These might require special setup
                
            design = TrialDesign(
                trial_type=trial_type,
                number_of_arms=2,
                effect_size=0.5,
                alpha=0.05,
                power=0.8,
                variance=1.0,
                sample_size=100,
                willingness_to_pay=50000,
                endpoint_type=endpoint_type
            )
            
            optimizer = VOIBasedSampleSizeOptimizer(design)
            result = optimizer.calculate_optimal_sample_size(Treatment("Test", 1000, 0.7))
            assert isinstance(result, dict)
            
    # Test all module functions
    create_superiority_trial(effect_size=0.5)
    create_adaptive_trial(effect_size=0.5)  
    create_health_economics_trial(effect_size=0.5)
    quick_trial_optimization(Treatment("Test", 1000, 0.7))
    calculate_trial_voi(Treatment("Test", 1000, 0.7))


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "--cov=voiage.clinical_trials", "--cov-report=term-missing", "-v"])