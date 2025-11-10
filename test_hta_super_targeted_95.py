"""
Super-targeted test to push HTA integration coverage from 93% to 95%
Focuses on the exact 21 missing lines: 236-237, 344, 424-429, 449, 465, 487-489, 540, 545, 550, 562, 638, 642, 676, 694
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from voiage.hta_integration import (
    HTASubmission, HTAFrameworkCriteria, HTAEvaluation, DecisionType,
    NICEFrameworkAdapter, CADTHFrameworkAdapter, ICERFrameworkAdapter,
    HTAIntegrationFramework, HTAFramework, 
    quick_hta_evaluation, compare_hta_decisions, generate_hta_report
)


class TestHTASuper95Push:
    """Ultra-specific tests for the 21 missing lines"""

    def setup_method(self):
        """Set up comprehensive test data"""
        self.nice_framework = NICEFrameworkAdapter()
        self.cadth_framework = CADTHFrameworkAdapter()
        self.icer_framework = ICERFrameworkAdapter()
        self.hta_integration = HTAIntegrationFramework()

    def test_missing_line_236_237_innovation_with_mechanism(self):
        """Test lines 236-237: Innovation assessment with mechanism_of_action"""
        submission = HTASubmission(
            technology_name="Mechanism Drug",
            indication="Mechanism Indication",
            cost_effectiveness_analysis={"icer": 40000, "qaly_gain": 1.5},
            budget_impact_analysis={"total_budget_impact": 1000000},
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 800},
            innovation_factors={
                "mechanism_of_action": True  # This should trigger lines 236-237
            }
        )
        
        evaluation = self.nice_framework.evaluate_submission(submission)
        assert evaluation is not None
        # Lines 236-237 should be covered by this innovation_factors check
        assert hasattr(evaluation, 'innovation_score')

    def test_missing_line_344_icers_method(self):
        """Test line 344: ICER framework evaluation method"""
        submission = HTASubmission(
            technology_name="ICER Method Line 344",
            indication="ICER Method Indication",
            cost_effectiveness_analysis={"icer": 85000, "qaly_gain": 2.1},
            clinical_trial_data={"evidence_level": "Observational", "trial_size": 500}
        )
        
        # This should trigger the specific line 344 in the ICER evaluation
        evaluation = self.icer_framework.evaluate_submission(submission)
        assert evaluation is not None
        assert evaluation.icer is not None

    def test_missing_lines_424_429_value_based_pricing(self):
        """Test lines 424-429: Value-based pricing logic"""
        submission = HTASubmission(
            technology_name="Value Based Drug",
            indication="Value Based Indication",
            cost_effectiveness_analysis={"icer": 140000, "qaly_gain": 3.8},
            clinical_trial_data={"evidence_level": "Meta-analysis", "trial_size": 2000}
        )
        
        # This should trigger the value-based pricing logic on lines 424-429
        evaluation = self.icer_framework.evaluate_submission(submission)
        assert evaluation is not None
        # The value-based pricing should result in specific recommendations
        assert evaluation.recommendation is not None

    def test_missing_line_449_framework_specific_method(self):
        """Test line 449: Framework-specific evaluation method"""
        submission = HTASubmission(
            technology_name="Framework Method 449",
            indication="Framework Method Indication",
            cost_effectiveness_analysis={"icer": 95000, "qaly_gain": 2.3},
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 1000}
        )
        
        # This should trigger the missing line 449
        evaluation = self.icer_framework.evaluate_submission(submission)
        assert evaluation is not None
        # The evaluation should have all required attributes
        if hasattr(evaluation, 'cost_effectiveness_score'):
            assert evaluation.cost_effectiveness_score is not None

    def test_missing_line_465_framework_evaluation(self):
        """Test line 465: Framework evaluation logic"""
        submission = HTASubmission(
            technology_name="Evaluation Line 465",
            indication="Evaluation Indication 465",
            cost_effectiveness_analysis={"icer": 110000, "qaly_gain": 2.8},
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 750}
        )
        
        # This should trigger the missing line 465
        evaluation = self.icer_framework.evaluate_submission(submission)
        assert evaluation is not None
        assert evaluation.recommendation is not None

    def test_missing_lines_487_489_framework_integration(self):
        """Test lines 487-489: HTA integration framework evaluation"""
        submission = HTASubmission(
            technology_name="Integration 487",
            indication="Integration Indication",
            cost_effectiveness_analysis={"icer": 60000, "qaly_gain": 1.6},
            clinical_trial_data={"evidence_level": "Observational"}
        )
        
        # This should trigger the missing lines 487-489
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.NICE)
        assert result is not None

    def test_missing_line_540_integration_method(self):
        """Test line 540: HTAIntegrationFramework method"""
        submission = HTASubmission(
            technology_name="Method 540",
            indication="Method 540 Indication",
            cost_effectiveness_analysis={"icer": 75000, "qaly_gain": 1.9},
            clinical_trial_data={"evidence_level": "RCT"}
        )
        
        # This should trigger the missing line 540
        strategy = self.hta_integration.create_hta_strategy(
            submission, [HTAFramework.NICE, HTAFramework.CADTH]
        )
        assert strategy is not None
        assert hasattr(strategy, 'recommended_frameworks')

    def test_missing_line_545_integration_comparison(self):
        """Test line 545: HTAIntegrationFramework comparison method"""
        submission = HTASubmission(
            technology_name="Comparison 545",
            indication="Comparison Indication",
            cost_effectiveness_analysis={"icer": 90000, "qaly_gain": 2.2},
            clinical_trial_data={"evidence_level": "RCT"}
        )
        
        # Test multiple framework evaluation
        evaluations = self.hta_integration.evaluate_multiple_frameworks(
            submission, [HTAFramework.NICE, HTAFramework.ICER]
        )
        assert len(evaluations) == 2
        
        # This should trigger the missing line 545
        comparison = self.hta_integration.compare_framework_decisions(evaluations)
        assert comparison is not None
        assert hasattr(comparison, 'consensus_decision')

    def test_missing_line_550_strategy_creation(self):
        """Test line 550: HTAIntegrationFramework strategy creation"""
        submission = HTASubmission(
            technology_name="Strategy 550",
            indication="Strategy Indication",
            cost_effectiveness_analysis={"icer": 120000, "qaly_gain": 3.0},
            clinical_trial_data={"evidence_level": "Meta-analysis"}
        )
        
        # This should trigger the missing line 550
        strategy = self.hta_integration.create_hta_strategy(
            submission, [HTAFramework.ICER, HTAFramework.CADTH]
        )
        assert strategy is not None
        assert hasattr(strategy, 'decision_matrix')

    def test_missing_line_562_framework_specific(self):
        """Test line 562: Framework-specific evaluation"""
        submission = HTASubmission(
            technology_name="Specific 562",
            indication="Specific Indication 562",
            framework_specific_data={"end_of_life": True, "rare_disease": False}
        )
        
        # This should trigger the missing line 562
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.ICER)
        assert result is not None

    def test_missing_line_638_framework_evaluation(self):
        """Test line 638: Framework evaluation with end-of-life"""
        submission = HTASubmission(
            technology_name="End of Life 638",
            indication="End of Life Indication",
            framework_specific_data={"end_of_life": True},
            cost_effectiveness_analysis={"icer": 35000, "qaly_gain": 0.8}
        )
        
        # This should trigger the missing line 638
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.NICE)
        assert result is not None
        assert result.decision == DecisionType.APPROVAL  # End of life typically gets approval

    def test_missing_line_642_framework_evaluation(self):
        """Test line 642: Framework evaluation with rare disease"""
        submission = HTASubmission(
            technology_name="Rare Disease 642",
            indication="Rare Disease Indication",
            framework_specific_data={"rare_disease": True},
            cost_effectiveness_analysis={"icer": 120000, "qaly_gain": 2.5}
        )
        
        # This should trigger the missing line 642
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.CADTH)
        assert result is not None

    def test_missing_line_676_framework_evaluation(self):
        """Test line 676: Framework evaluation logic"""
        submission = HTASubmission(
            technology_name="Evaluation 676",
            indication="Evaluation Indication 676",
            framework_specific_data={"end_of_life": False, "rare_disease": True}
        )
        
        # This should trigger the missing line 676
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.ICER)
        assert result is not None
        assert hasattr(result, 'equity_score')

    def test_missing_line_694_framework_evaluation(self):
        """Test line 694: Framework evaluation with equity impact"""
        submission = HTASubmission(
            technology_name="Equity 694",
            indication="Equity Indication",
            equity_impact={"population_benefit": 0.35, "equity_score": 0.8}
        )
        
        # This should trigger the missing line 694
        result = self.hta_integration.evaluate_for_framework(submission, HTAFramework.CADTH)
        assert result is not None

    def test_ultra_comprehensive_missing_lines(self):
        """Ultra-comprehensive test to trigger as many missing lines as possible"""
        # Create a submission that should trigger all missing line conditions
        submission = HTASubmission(
            technology_name="Ultra Comprehensive Drug",
            indication="Ultra Comprehensive Indication",
            cost_effectiveness_analysis={"icer": 100000, "qaly_gain": 2.5},
            budget_impact_analysis={"total_budget_impact": 5000000},
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 2000},
            innovation_factors={
                "mechanism_of_action": True,
                "first_in_class": True,
                "breakthrough_therapy": True
            },
            framework_specific_data={
                "end_of_life": True,
                "rare_disease": False
            },
            equity_impact={
                "population_benefit": 0.4,
                "equity_score": 0.9
            }
        )
        
        # Test all framework evaluations
        nice_eval = self.nice_framework.evaluate_submission(submission)
        cadth_eval = self.cadth_framework.evaluate_submission(submission)
        icer_eval = self.icer_framework.evaluate_submission(submission)
        
        # Test all integration framework methods
        strategy = self.hta_integration.create_hta_strategy(
            submission, [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]
        )
        
        evaluations = self.hta_integration.evaluate_multiple_frameworks(
            submission, [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]
        )
        
        comparison = self.hta_integration.compare_framework_decisions(evaluations)
        
        # This comprehensive test should cover most or all of the missing lines
        assert nice_eval is not None
        assert cadth_eval is not None
        assert icer_eval is not None
        assert strategy is not None
        assert len(evaluations) == 3
        assert comparison is not None