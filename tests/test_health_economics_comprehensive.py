"""
Comprehensive test to achieve >95% coverage for health_economics.py
Targeting specific missing lines: 96, 419-423, 427-445, 466-472, 489, 506-518
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
sys.path.append('.')

from voiage.health_economics import (
    HealthState, HealthEconomicsAnalysis, DecisionAnalysis, Treatment,
    calculate_icer_simple, calculate_net_monetary_benefit_simple, 
    qaly_calculator
)


class TestHealthEconomicsLine96:
    """Test line 96 specifically"""
    
    def test_line_96_coverage(self):
        """Test to cover line 96 - specific edge case in calculate_qaly method"""
        health_state = HealthState(
            state_id="test",
            description="Test state", 
            utility=0.8,
            cost=1000.0,
            duration=5.0
        )
        
        # Test case where time_horizon is None (should use health_state.duration)
        analysis = HealthEconomicsAnalysis(willingness_to_pay=50000)
        result = analysis.calculate_qaly(health_state, time_horizon=None)
        
        # This should hit line 96 where time_horizon is None
        assert result is not None
        assert isinstance(result, (float, jnp.ndarray))


class TestHealthEconomicsLines419423:
    """Test lines 419-423 - custom decision function logic"""
    
    def test_custom_decision_function_true_branch(self):
        """Test when decision_function is callable"""
        analysis = HealthEconomicsAnalysis(willingness_to_pay=50000)
        
        # Create mock treatments
        treatments = ["treatment1", "treatment2"]
        
        # Create a callable decision function
        def custom_decision(treatment, **kwargs):
            return {
                'qaly': 2.0,
                'cost': 15000,
                'nmb': (2.0 * 50000) - 15000,
                'icer': 7500
            }
        
        # This should hit the callable branch (line where hasattr returns True)
        outcomes = analysis._health_decision_outcomes(treatments, custom_decision)
        assert len(outcomes) == len(treatments)
        assert all('qaly' in outcome for outcome in outcomes)


class TestHealthEconomicsLines427445:
    """Test lines 427-445 - default outcome calculation branch"""
    
    def test_default_outcome_calculation(self):
        """Test the else branch for non-callable decision function"""
        analysis = HealthEconomicsAnalysis(willingness_to_pay=50000)
        
        # Create proper Treatment objects
        treatments = [
            Treatment("treatment1", "First treatment", 0.8, 1000, 5),
            Treatment("treatment2", "Second treatment", 0.6, 1500, 3)
        ]
        
        # Test with non-callable (e.g., string or None)
        outcomes = analysis._health_decision_outcomes(treatments, "not_callable")
        
        assert len(outcomes) == len(treatments)
        for outcome in outcomes:
            # Should have all expected keys from default calculation
            assert 'qaly' in outcome
            assert 'cost' in outcome
            assert 'nmb' in outcome
            assert 'icer' in outcome
    
    def test_icer_calculation_in_outcomes(self):
        """Test that ICER calculation is properly called in default outcomes"""
        analysis = HealthEconomicsAnalysis(willingness_to_pay=50000)
        
        treatment = Treatment("treatment1", "First treatment", 0.8, 1000, 5)
        
        # Mock the calculate_icer method to verify it's called
        with patch.object(analysis, 'calculate_icer') as mock_icer:
            mock_icer.return_value = 10000.0
            
            outcomes = analysis._health_decision_outcomes([treatment], "not_callable")
            
            # Verify calculate_icer was called
            mock_icer.assert_called_once_with(treatment)
            
            # Check that the ICER is included in the outcome
            assert outcomes[0]['icer'] == 10000.0


class TestHealthEconomicsLines466472:
    """Test lines 466-472 - calculate_icer_simple function"""
    
    def test_calculate_icer_simple_incremental_effect_zero(self):
        """Test when incremental_effect <= 0 (should return infinity)"""
        # (cost_intervention, effect_intervention, cost_comparator, effect_comparator)
        # effect_intervention(2.0) - effect_comparator(1.0) = 1.0 > 0, so should not be inf
        result = calculate_icer_simple(10000, 2.0, 5000, 1.0)
        assert result == 5000.0  # (10000-5000)/(2.0-1.0) = 5000/1.0 = 5000
        
        # Now test when incremental effect is <= 0
        result = calculate_icer_simple(10000, 1.0, 5000, 2.0)  # 1.0-2.0 = -1.0 <= 0
        assert result == float('inf')
    
    def test_calculate_icer_simple_positive_incremental_effect(self):
        """Test normal case with positive incremental effect"""
        # (cost_intervention, effect_intervention, cost_comparator, effect_comparator)
        result = calculate_icer_simple(15000, 3.0, 5000, 1.0)  # inc_cost=10000, inc_effect=2.0
        assert result == 5000.0  # 10000 / 2.0
    
    def test_calculate_icer_simple_edge_cases(self):
        """Test edge cases for calculate_icer_simple"""
        # Test equal costs and effects
        result = calculate_icer_simple(10000, 2.0, 10000, 2.0)  # 0/0 = should be inf due to <= 0 check
        assert result == float('inf')
        
        # Test zero incremental effect
        result = calculate_icer_simple(15000, 2.0, 5000, 2.0)  # 2.0-2.0 = 0 <= 0
        assert result == float('inf')


class TestHealthEconomicsLine489:
    """Test line 489 - specific line in calculate_net_monetary_benefit_simple"""
    
    def test_line_489_coverage(self):
        """Test specific calculation path in calculate_net_monetary_benefit_simple"""
        # Test the basic calculation
        result = calculate_net_monetary_benefit_simple(2.5, 10000, 50000)
        expected = (2.5 * 50000) - 10000
        assert result == expected
        
        # Test edge cases
        result = calculate_net_monetary_benefit_simple(0, 10000, 50000)
        assert result == -10000
        
        result = calculate_net_monetary_benefit_simple(2.5, 0, 50000)
        assert result == 2.5 * 50000


class TestHealthEconomicsLines506518:
    """Test lines 506-518 - calculate_qaly_simple function"""
    
    def test_calculate_qaly_simple_edge_cases(self):
        """Test edge cases in qaly_calculator"""
        # Test zero or negative inputs
        result = qaly_calculator(0, 0.8, 0.03)
        assert result == 0.0
        
        result = qaly_calculator(-1, 0.8, 0.03)
        assert result == 0.0
        
        result = qaly_calculator(5, -0.5, 0.03)  # negative utility
        assert result == 0.0
    
    def test_calculate_qaly_simple_single_year(self):
        """Test calculation for single year (no discounting applied)"""
        result = qaly_calculator(1, 0.8, 0.03)
        expected = 1.0 * 0.8  # life_years * utility_weight
        assert result == expected
    
    def test_calculate_qaly_simple_multi_year_with_discounting(self):
        """Test multi-year calculation with discounting"""
        life_years = 5
        utility = 0.8
        discount_rate = 0.03
        
        result = qaly_calculator(life_years, utility, discount_rate)
        
        # Manual calculation:
        # undiscounted_qaly = 5 * 0.8 = 4.0
        # discounted_qaly = 4.0 * (1 - (1.03)^(-5)) / (0.03 * 5)
        expected_undiscounted = life_years * utility
        assert result <= expected_undiscounted  # Should be less due to discounting
        assert result > 0  # Should be positive
    
    def test_calculate_qaly_simple_capping(self):
        """Test that result is capped at life_years"""
        result = qaly_calculator(10, 1.0, 0.01)  # Very high utility
        assert result <= 10.0  # Should not exceed life_years


class TestHealthEconomicsIntegration:
    """Integration tests to cover complex scenarios"""
    
    def test_full_analysis_with_custom_functions(self):
        """Test complete analysis flow with custom decision functions"""
        analysis = HealthEconomicsAnalysis(willingness_to_pay=50000)
        
        # Create proper Treatment objects
        treatment1 = Treatment("treatment1", "Treatment 1", 0.9, 2000, 5)
        treatment2 = Treatment("treatment2", "Treatment 2", 0.5, 10000, 3)
        
        treatments = [treatment1, treatment2]
        
        # Test with custom function
        def custom_function(treatment, **kwargs):
            return {'qaly': 2.5, 'cost': 15000}
        
        # This should cover the complete flow including the callable branch
        outcomes = analysis._health_decision_outcomes([treatment1], custom_function)
        assert len(outcomes) == 1
        assert outcomes[0]['qaly'] == 2.5
        
        # Test default flow
        outcomes = analysis._health_decision_outcomes([treatment1], "not_callable")
        assert len(outcomes) == 1
        assert 'qaly' in outcomes[0]
    
    def test_boundary_conditions_comprehensive(self):
        """Test boundary conditions across all functions"""
        # Test qaly_calculator with various boundary conditions
        assert qaly_calculator(0, 0.5, 0.03) == 0.0
        assert qaly_calculator(5, 0, 0.03) == 0.0
        
        # Test calculate_icer with boundary conditions (corrected parameters)
        # (cost_intervention, effect_intervention, cost_comparator, effect_comparator)
        assert calculate_icer_simple(10000, 1.0, 10000, 1.0) == float('inf')  # Equal costs and effects (incremental effect = 0)
        assert calculate_icer_simple(1000, 0, 1, 0) == float('inf')  # No effect gain (0 - 0 = 0 <= 0)
        assert calculate_icer_simple(1000, 1.0, 0, 0) == 1000.0  # Normal case
        
        # Test calculate_net_monetary_benefit with boundary conditions
        assert calculate_net_monetary_benefit_simple(0, 0, 50000) == 0.0
        assert calculate_net_monetary_benefit_simple(1, 100000, 50000) == -50000


if __name__ == "__main__":
    # Run specific test for line 96
    test_line_96 = TestHealthEconomicsLine96()
    test_line_96.test_line_96_coverage()
    print("✓ Line 96 coverage achieved")
    
    # Run other tests
    test_integration = TestHealthEconomicsIntegration()
    test_integration.test_full_analysis_with_custom_functions()
    test_integration.test_boundary_conditions_comprehensive()
    print("✓ Integration tests passed")
    
    print("✓ All comprehensive health_economics tests completed successfully")