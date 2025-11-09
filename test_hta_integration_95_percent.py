"""
Targeted test file to achieve >95% coverage for hta_integration.py

This test file targets specific missing lines identified in the coverage report
to achieve >95% coverage for the hta_integration.py module.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voiage'))

from voiage.hta_integration import (
    HTAFramework, DecisionType, EvidenceRequirement,
    HTAFrameworkCriteria, HTASubmission, HTAEvaluation,
    NICEFrameworkAdapter, CADTHFrameworkAdapter
)
import jax.numpy as jnp
from datetime import datetime


class TestHTAIntegration95Percent(unittest.TestCase):
    """Test class to achieve >95% coverage for hta_integration.py"""

    def setUp(self):
        """Set up test fixtures"""
        self.nice_adapter = NICEFrameworkAdapter()
        
        # Create a comprehensive submission for testing
        self.submission_complete = HTASubmission(
            technology_name="TestDrug",
            manufacturer="TestPharma",
            indication="Test Condition",
            population="Adult patients",
            comparators=["Standard of Care"],
            clinical_trial_data={
                'evidence_level': 'RCT',
                'primary_endpoint': 'efficacy',
                'study_duration': 24,
                'sample_size': 500
            },
            real_world_evidence={'effectiveness': 0.75},
            cost_effectiveness_analysis={
                'icer': 25000.0,
                'qaly_gain': 2.5,
                'net_monetary_benefit': 50000.0
            },
            budget_impact_analysis={
                'total_impact': 2000000.0,
                'annual_impact': 500000.0
            },
            innovation_factors={
                'mechanism_of_action': True,
                'first_in_class': True,
                'breakthrough_therapy': True
            },
            framework_specific_data={
                'end_of_life': True,
                'rare_disease': True
            },
            equity_impact={
                'population_benefit': 0.3
            },
            economic_model={
                'structural_uncertainty': 0.4,
                'parameter_uncertainty': 0.2
            }
        )

    def test_budget_impact_high_threshold(self):
        """Test budget impact evaluation with high impact - covers lines 231-234"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 20000, 'qaly_gain': 1.0},
            budget_impact_analysis={'total_impact': 1000000.0}  # High impact
        )
        
        # Set threshold to trigger the high impact condition
        self.nice_adapter.criteria.budget_impact_threshold = 500000.0
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify high budget impact evaluation
        self.assertEqual(evaluation.budget_impact_score, 0.4)
        self.assertIn("Significant budget impact identified", evaluation.weaknesses)
        self.assertIn("Detailed budget impact modeling", evaluation.additional_evidence_needed)

    def test_budget_impact_acceptable(self):
        """Test budget impact evaluation with acceptable impact - covers lines 236-237"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 20000, 'qaly_gain': 1.0},
            budget_impact_analysis={'total_impact': 100000.0}  # Low impact
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify acceptable budget impact evaluation
        self.assertEqual(evaluation.budget_impact_score, 0.8)
        self.assertIn("Acceptable budget impact", evaluation.strengths)

    def test_innovation_factors_comprehensive(self):
        """Test comprehensive innovation factors evaluation - covers lines 242-247"""
        submission = HTASubmission(
            technology_name="TestDrug",
            innovation_factors={
                'mechanism_of_action': True,
                'first_in_class': True,
                'breakthrough_therapy': True,
                'novel_target': True
            }
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify innovation score calculation
        # Base score 0.5 + 0.2 + 0.2 + 0.1 = 0.9
        self.assertAlmostEqual(evaluation.innovation_score, 0.9, places=2)

    def test_end_of_life_approved(self):
        """Test end of life approval criteria - covers lines 254-255"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 45000, 'qaly_gain': 1.0},
            framework_specific_data={'end_of_life': True}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify end of life approval
        self.assertEqual(evaluation.recommendation, "Approved for end of life treatment")
        self.assertIn("End of life treatment consideration applied", evaluation.strengths)

    def test_equity_impact_high_benefit(self):
        """Test high equity benefit evaluation - covers lines 268-271"""
        submission = HTASubmission(
            technology_name="TestDrug",
            equity_impact={'population_benefit': 0.3}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify equity evaluation with high benefit
        # 0.5 + 0.3 = 0.8
        self.assertEqual(evaluation.equity_score, 0.8)
        self.assertIn("Significant equity benefits identified", evaluation.strengths)

    def test_price_negotiation_decision(self):
        """Test price negotiation decision logic - covers lines 278-279"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 35000, 'qaly_gain': 0.5},  # Low score
            budget_impact_analysis={'total_impact': 100000.0}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify price negotiation decision
        self.assertEqual(evaluation.decision, DecisionType.PRICE_NEGOTIATION)
        self.assertEqual(evaluation.recommendation, "Price negotiation required")

    def test_structural_uncertainty_identification(self):
        """Test structural uncertainty identification - covers line 285"""
        submission = HTASubmission(
            technology_name="TestDrug",
            real_world_evidence={'effectiveness': 0.5},  # Present
            economic_model={'structural_uncertainty': 0.4}  # High uncertainty
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify uncertainty identification
        self.assertIn("Significant structural uncertainty in economic model", evaluation.uncertainties)

    def test_additional_evidence_required(self):
        """Test additional evidence required decision - covers lines 340-342"""
        submission = HTASubmission(
            technology_name="TestDrug",
            clinical_trial_data={'evidence_level': 'Observational'}  # Low quality
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify additional evidence requirement
        self.assertEqual(evaluation.decision, DecisionType.ADDITIONAL_EVIDENCE_REQUIRED)
        self.assertIn("Additional comparative effectiveness evidence required", evaluation.recommendation)
        self.assertIn("Head-to-head comparative trials", evaluation.additional_evidence_needed)

    def test_budget_impact_extraction(self):
        """Test budget impact data extraction - covers line 201"""
        submission = HTASubmission(
            technology_name="TestDrug",
            budget_impact_analysis={'total_impact': 500000.0}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify budget impact extraction
        self.assertEqual(evaluation.budget_impact, 500000.0)

    def test_clinical_effectiveness_observational(self):
        """Test clinical effectiveness evaluation for observational data - covers lines 214-215"""
        submission = HTASubmission(
            technology_name="TestDrug",
            clinical_trial_data={'evidence_level': 'Observational'}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify observational evidence evaluation
        self.assertEqual(evaluation.clinical_effectiveness_score, 0.5)
        self.assertIn("Limited clinical evidence quality", evaluation.weaknesses)

    def test_cost_effectiveness_higher_threshold(self):
        """Test cost-effectiveness evaluation at higher threshold - covers lines 223-224"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 25000, 'qaly_gain': 1.0}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify higher threshold evaluation
        self.assertEqual(evaluation.cost_effectiveness_score, 0.7)
        self.assertIn("Cost-effective within higher threshold", evaluation.strengths)

    def test_rare_disease_approval(self):
        """Test rare disease approval criteria"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 80000, 'qaly_gain': 1.0},
            framework_specific_data={'rare_disease': True}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify rare disease approval
        self.assertEqual(evaluation.recommendation, "Approved for rare disease")
        self.assertIn("Rare disease modifier applied", evaluation.strengths)

    def test_missing_real_world_evidence(self):
        """Test uncertainty identification for missing real-world evidence"""
        submission = HTASubmission(
            technology_name="TestDrug",
            real_world_evidence=None  # Missing
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify uncertainty identification
        self.assertIn("Limited real-world effectiveness data", evaluation.uncertainties)

    def test_rejection_decision(self):
        """Test rejection decision logic"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 50000, 'qaly_gain': 0.2}  # Very low score
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify rejection decision
        self.assertEqual(evaluation.decision, DecisionType.REJECTION)
        self.assertEqual(evaluation.recommendation, "Not recommended for reimbursement")

    def test_end_of_life_rejection(self):
        """Test end of life treatment rejection when ICER too high"""
        submission = HTASubmission(
            technology_name="TestDrug",
            cost_effectiveness_analysis={'icer': 60000, 'qaly_gain': 1.0},  # Above EOL threshold
            framework_specific_data={'end_of_life': True}
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify end of life rejection
        self.assertEqual(evaluation.decision, DecisionType.REJECTION)
        self.assertEqual(evaluation.recommendation, "Not recommended")

    def test_mixed_innovation_factors(self):
        """Test mixed innovation factors evaluation"""
        submission = HTASubmission(
            technology_name="TestDrug",
            innovation_factors={
                'mechanism_of_action': False,  # Not novel
                'first_in_class': True,
                'breakthrough_therapy': False
            }
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify partial innovation score
        # Base score 0.5 + 0.2 = 0.7
        self.assertEqual(evaluation.innovation_score, 0.7)

    def test_equity_impact_low_benefit(self):
        """Test equity impact evaluation with low benefit"""
        submission = HTASubmission(
            technology_name="TestDrug",
            equity_impact={'population_benefit': 0.1}  # Low benefit
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify equity evaluation with low benefit
        # 0.5 + 0.1 = 0.6
        self.assertEqual(evaluation.equity_score, 0.6)
        # Should not have high benefit message
        self.assertNotIn("Significant equity benefits identified", evaluation.strengths)

    def test_empty_innovation_factors(self):
        """Test evaluation with empty innovation factors"""
        submission = HTASubmission(
            technology_name="TestDrug",
            innovation_factors={}  # Empty
        )
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify base innovation score
        self.assertEqual(evaluation.innovation_score, 0.5)

    def test_comprehensive_submission_full_coverage(self):
        """Test comprehensive submission to cover maximum lines"""
        submission = self.submission_complete
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Verify comprehensive evaluation
        self.assertIsNotNone(evaluation.icer)
        self.assertIsNotNone(evaluation.budget_impact)
        self.assertIsNotNone(evaluation.clinical_effectiveness_score)
        self.assertIsNotNone(evaluation.cost_effectiveness_score)
        self.assertIsNotNone(evaluation.innovation_score)
        self.assertGreater(len(evaluation.strengths), 0)
        self.assertGreater(len(evaluation.weaknesses), 0)
        self.assertGreater(len(evaluation.uncertainties), 0)

    def test_cadth_adapter_initialization(self):
        """Test CADTH adapter creation for additional coverage"""
        cadth_adapter = CADTHFrameworkAdapter()
        
        # Verify CADTH-specific criteria
        self.assertEqual(cadth_adapter.criteria.framework, HTAFramework.CADTH)
        self.assertEqual(cadth_adapter.criteria.max_icer_threshold, 50000.0)
        self.assertEqual(cadth_adapter.criteria.budget_impact_threshold, 0.03)
        self.assertTrue(cadth_adapter.criteria.submission_requirements['must_include_cea'])
        self.assertTrue(cadth_adapter.criteria.submission_requirements['comparative_effectiveness'])

    def test_edge_case_no_evidence(self):
        """Test evaluation with minimal evidence"""
        submission = HTASubmission(technology_name="Minimal")
        
        evaluation = self.nice_adapter.evaluate_submission(submission)
        
        # Should handle missing data gracefully
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation.decision, DecisionType.APPROVAL)  # Default

    def test_extreme_icer_values(self):
        """Test extreme ICER values for boundary coverage"""
        test_cases = [
            (0.01, 0.9),    # Very cost-effective
            (19999, 0.9),   # Just under standard threshold
            (20001, 0.7),   # Just over standard threshold
            (29999, 0.7),   # Just under higher threshold
            (30001, 0.3),   # Just over higher threshold
            (100000, 0.3)   # Very high cost
        ]
        
        for icer, expected_score in test_cases:
            with self.subTest(icer=icer):
                submission = HTASubmission(
                    technology_name="TestDrug",
                    cost_effectiveness_analysis={'icer': icer, 'qaly_gain': 1.0}
                )
                
                evaluation = self.nice_adapter.evaluate_submission(submission)
                self.assertEqual(evaluation.cost_effectiveness_score, expected_score)


if __name__ == '__main__':
    # Run tests with coverage
    import subprocess
    import sys
    
    print("Running targeted HTA integration tests for 95% coverage...")
    print("=" * 60)
    
    # First run our specific tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("Coverage analysis for targeted lines...")
    
    # Then run coverage on just hta_integration.py
    try:
        result = subprocess.run([
            sys.executable, '-m', 'coverage', 'run', 
            '-m', 'pytest', __file__, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd='/Users/doughnut/GitHub/voiage')
        
        if result.returncode == 0:
            print("Tests completed successfully!")
            
            # Get coverage for hta_integration specifically
            cov_result = subprocess.run([
                sys.executable, '-m', 'coverage', 'report',
                '--include=voiage/hta_integration.py'
            ], capture_output=True, text=True, cwd='/Users/doughnut/GitHub/voiage')
            
            print("\nCoverage report for hta_integration.py:")
            print(cov_result.stdout)
        else:
            print("Test execution failed:")
            print(result.stderr)
    except Exception as e:
        print(f"Coverage analysis failed: {e}")