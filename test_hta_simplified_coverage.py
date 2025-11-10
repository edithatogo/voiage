"""
Simplified targeted test suite to improve HTA integration coverage
Focuses on the actual available methods and realistic usage patterns
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


class TestHTACoverageImprovement:
    """Targeted tests to improve HTA integration coverage"""

    def setup_method(self):
        """Set up test data"""
        self.submission = HTASubmission(
            technology_name="Test Drug",
            indication="Test Indication",
            cost_effectiveness_analysis={"icer": 45000, "qaly_gain": 1.2},
            budget_impact_analysis={"total_budget_impact": 2000000},
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 1000}
        )
        
        self.nice_framework = NICEFrameworkAdapter()
        self.cadth_framework = CADTHFrameworkAdapter()
        self.icer_framework = ICERFrameworkAdapter()
        self.hta_integration = HTAIntegrationFramework()

    def test_nice_framework_basic_evaluation(self):
        """Test basic NICE framework evaluation"""
        evaluation = self.nice_framework.evaluate_submission(self.submission)
        assert evaluation is not None
        assert isinstance(evaluation, HTAEvaluation)
        assert evaluation.decision is not None

    def test_cadth_framework_basic_evaluation(self):
        """Test basic CADTH framework evaluation"""
        evaluation = self.cadth_framework.evaluate_submission(self.submission)
        assert evaluation is not None
        assert isinstance(evaluation, HTAEvaluation)

    def test_icer_framework_basic_evaluation(self):
        """Test basic ICER framework evaluation"""
        evaluation = self.icer_framework.evaluate_submission(self.submission)
        assert evaluation is not None
        assert isinstance(evaluation, HTAEvaluation)

    def test_hta_integration_framework_evaluation(self):
        """Test HTA integration framework"""
        evaluation = self.hta_integration.evaluate_for_framework(
            self.submission, HTAFramework.NICE
        )
        assert evaluation is not None

    def test_hta_integration_multiple_frameworks(self):
        """Test evaluation across multiple frameworks"""
        evaluations = self.hta_integration.evaluate_multiple_frameworks(
            self.submission, [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]
        )
        assert len(evaluations) == 3
        assert HTAFramework.NICE in evaluations
        assert HTAFramework.CADTH in evaluations
        assert HTAFramework.ICER in evaluations

    def test_hta_integration_framework_comparison(self):
        """Test framework decision comparison"""
        evaluations = self.hta_integration.evaluate_multiple_frameworks(
            self.submission, [HTAFramework.NICE, HTAFramework.CADTH]
        )
        
        comparison = self.hta_integration.compare_framework_decisions(evaluations)
        assert comparison is not None
        assert 'framework_evaluations' in comparison

    def test_hta_integration_strategy_creation(self):
        """Test HTA strategy creation"""
        strategy = self.hta_integration.create_hta_strategy(
            self.submission, [HTAFramework.NICE, HTAFramework.CADTH]
        )
        assert strategy is not None
        assert 'strategy_summary' in strategy

    def test_quick_hta_evaluation(self):
        """Test quick HTA evaluation function"""
        evaluation = quick_hta_evaluation(self.submission, HTAFramework.NICE)
        assert evaluation is not None
        assert isinstance(evaluation, HTAEvaluation)

    def test_compare_hta_decisions(self):
        """Test HTA decision comparison function"""
        result = compare_hta_decisions(self.submission)
        assert result is not None

    def test_generate_hta_report(self):
        """Test HTA report generation"""
        report = generate_hta_report(self.submission, HTAFramework.NICE)
        assert report is not None
        assert "HTA Evaluation Report" in report
        assert "Test Drug" in report
        assert "Test Indication" in report

    def test_edge_case_submission_no_data(self):
        """Test with minimal submission data"""
        empty_submission = HTASubmission(
            technology_name="Minimal Drug",
            indication="Minimal Indication"
        )
        
        evaluation = quick_hta_evaluation(empty_submission, HTAFramework.NICE)
        assert evaluation is not None

    def test_edge_case_submission_partial_data(self):
        """Test with partial submission data"""
        partial_submission = HTASubmission(
            technology_name="Partial Drug",
            indication="Partial Indication",
            cost_effectiveness_analysis={"icer": None, "qaly_gain": None},
            clinical_trial_data={}
        )
        
        evaluation = quick_hta_evaluation(partial_submission, HTAFramework.CADTH)
        assert evaluation is not None

    def test_framework_specific_data_handling(self):
        """Test handling of framework-specific data"""
        special_submission = HTASubmission(
            technology_name="Special Drug",
            indication="Special Indication",
            framework_specific_data={
                "end_of_life": True,
                "rare_disease": False,
                "breakthrough_therapy": True
            },
            equity_impact={"population_benefit": 0.3}
        )
        
        evaluation = quick_hta_evaluation(special_submission, HTAFramework.NICE)
        assert evaluation is not None

    def test_budget_impact_analysis(self):
        """Test budget impact analysis scenarios"""
        budget_submission = HTASubmission(
            technology_name="Budget Drug",
            indication="Budget Indication",
            budget_impact_analysis={"total_budget_impact": 5000000}
        )
        
        evaluation = self.nice_framework.evaluate_submission(budget_submission)
        assert evaluation is not None
        assert evaluation.budget_impact is not None

    def test_cost_effectiveness_scenarios(self):
        """Test different cost-effectiveness scenarios"""
        # High ICER scenario
        high_icer_submission = HTASubmission(
            technology_name="High ICER Drug",
            indication="High ICER Indication",
            cost_effectiveness_analysis={"icer": 200000, "qaly_gain": 1.0}
        )
        
        evaluation = self.icer_framework.evaluate_submission(high_icer_submission)
        assert evaluation is not None
        assert evaluation.icer == 200000

    def test_evidence_level_variations(self):
        """Test different evidence levels"""
        rct_submission = HTASubmission(
            technology_name="RCT Drug",
            indication="RCT Indication",
            clinical_trial_data={"evidence_level": "RCT", "trial_size": 2000}
        )
        
        observational_submission = HTASubmission(
            technology_name="Observational Drug",
            indication="Observational Indication",
            clinical_trial_data={"evidence_level": "Observational"}
        )
        
        rct_eval = self.cadth_framework.evaluate_submission(rct_submission)
        obs_eval = self.cadth_framework.evaluate_submission(observational_submission)
        
        assert rct_eval is not None
        assert obs_eval is not None

    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios"""
        # This should trigger error handling paths
        problematic_submission = HTASubmission(
            technology_name="",
            indication="",
            cost_effectiveness_analysis={},
            clinical_trial_data={}
        )
        
        evaluation = quick_hta_evaluation(problematic_submission, HTAFramework.ICER)
        assert evaluation is not None

    def test_evaluation_with_innovation_factors(self):
        """Test evaluation with innovation factors"""
        innovation_submission = HTASubmission(
            technology_name="Innovation Drug",
            indication="Innovation Indication",
            framework_specific_data={
                "innovation_factors": {
                    "mechanism_of_action": True,
                    "first_in_class": True,
                    "breakthrough_therapy": True
                }
            }
        )
        
        evaluation = self.nice_framework.evaluate_submission(innovation_submission)
        assert evaluation is not None

    def test_equity_impact_evaluation(self):
        """Test equity impact evaluation"""
        equity_submission = HTASubmission(
            technology_name="Equity Drug",
            indication="Equity Indication",
            equity_impact={
                "population_benefit": 0.4,
                "health_inequalities_reduction": 0.3
            }
        )
        
        evaluation = quick_hta_evaluation(equity_submission, HTAFramework.NICE)
        assert evaluation is not None
        assert evaluation.equity_score is not None