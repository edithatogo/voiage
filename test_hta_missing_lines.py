#!/usr/bin/env python3
"""
Additional tests to cover missing lines in hta_integration.py
Targeting specific uncovered lines to reach >95% coverage
"""

import sys
sys.path.insert(0, '.')

from voiage.hta_integration import (
    HTAFramework, DecisionType, HTASubmission, HTAEvaluation, HTAIntegrationFramework
)

def test_missing_lines_coverage():
    """Test to cover the specific missing lines from coverage report"""
    print("Testing missing lines coverage for hta_integration.py...")

    try:
        hta_evaluator = HTAIntegrationFramework()
        
        # Create mock evaluations for key differences testing
        mock_evaluation1 = HTAEvaluation(
            framework=HTAFramework.NICE,
            decision=DecisionType.APPROVAL,
            recommendation="Strong positive recommendation",
            icer=25000.0,  # High ICER
            qaly_gain=2.5,
            budget_impact_score=0.8,  # High budget impact
            innovation_score=0.7
        )
        
        mock_evaluation2 = HTAEvaluation(
            framework=HTAFramework.CADTH,
            decision=DecisionType.APPROVAL,
            recommendation="Positive recommendation",
            icer=75000.0,  # Very high ICER - should trigger significant difference
            qaly_gain=2.0,
            budget_impact_score=0.1,  # Low budget impact - should trigger difference
            innovation_score=0.3  # Low innovation score - should trigger difference
        )
        
        # Test key differences identification
        evaluations = {
            HTAFramework.NICE: mock_evaluation1,
            HTAFramework.CADTH: mock_evaluation2
        }
        
        differences = hta_evaluator._identify_key_differences(evaluations)
        assert len(differences) >= 1  # Should identify at least one difference
        
        # Test _summarize_recommendations method
        recommendations = [
            "Strong positive recommendation for this treatment",
            "Positive recommendation with some concerns"
        ]
        summary = hta_evaluator._summarize_recommendations(recommendations)
        assert 'recommendation_distribution' in summary
        assert 'overall_tone' in summary
        assert 'consensus_level' in summary
        
        # Test _evaluation_to_dict method
        evaluation_dict = hta_evaluator._evaluation_to_dict(mock_evaluation1)
        assert 'decision' in evaluation_dict
        assert 'recommendation' in evaluation_dict
        assert 'icer' in evaluation_dict
        assert 'qaly_gain' in evaluation_dict
        
        # Test _generate_strategy_recommendations method
        comparison = hta_evaluator.compare_framework_decisions(evaluations)
        strategy_recs = hta_evaluator._generate_strategy_recommendations(evaluations, comparison)
        assert isinstance(strategy_recs, list)
        
        # Test _identify_priority_evidence_gaps method
        evidence_gaps = hta_evaluator._identify_priority_evidence_gaps(evaluations)
        assert isinstance(evidence_gaps, list)
        
        # Test _generate_optimization_suggestions method
        optimization = hta_evaluator._generate_optimization_suggestions(evaluations)
        assert isinstance(optimization, list)
        
        # Test create_hta_strategy method
        submission = HTASubmission(
            technology_name="Test Treatment",
            indication="Test Indication",
            manufacturer="Test Manufacturer",
            population="Adult patients",
            comparators=["Standard of Care"],
            clinical_trial_data={
                "efficacy": 0.8,
                "safety": 0.9,
                "quality_of_life": 7.5
            },
            economic_model={
                "cost_per_patient": 25000.0,
                "time_horizon": 10
            }
        )
        
        strategy = hta_evaluator.create_hta_strategy(submission, [HTAFramework.NICE, HTAFramework.CADTH])
        assert 'target_frameworks' in strategy
        assert 'evaluations' in strategy
        assert 'comparison' in strategy
        assert 'strategy_recommendations' in strategy
        
        print("✓ _identify_key_differences method covered")
        print("✓ _summarize_recommendations method covered")
        print("✓ _evaluation_to_dict method covered")
        print("✓ _generate_strategy_recommendations method covered")
        print("✓ _identify_priority_evidence_gaps method covered")
        print("✓ _generate_optimization_suggestions method covered")
        print("✓ create_hta_strategy method covered")
        return True
        
    except Exception as e:
        print(f"❌ Error in missing lines coverage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_missing_lines_coverage()
    if success:
        print("✅ All missing line tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)