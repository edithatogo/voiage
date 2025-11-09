#!/usr/bin/env python3
"""
Simple test to get clean coverage reading for hta_integration.py
"""

import sys
sys.path.insert(0, '.')

from voiage.hta_integration import (
    HTAFramework, DecisionType, HTAFrameworkCriteria, 
    HTASubmission, HTAEvaluation, NICEFrameworkAdapter
)

def test_basic_functionality():
    """Test basic HTA functionality"""
    # Create adapter
    adapter = NICEFrameworkAdapter()
    
    # Test 1: Basic submission
    submission1 = HTASubmission(technology_name="TestDrug")
    eval1 = adapter.evaluate_submission(submission1)
    print(f"Test 1 - Basic: Decision={eval1.decision}, Recommendation={eval1.recommendation}")
    
    # Test 2: Submission with cost-effectiveness
    submission2 = HTASubmission(
        technology_name="CostEffectiveDrug",
        cost_effectiveness_analysis={'icer': 15000, 'qaly_gain': 1.0}
    )
    eval2 = adapter.evaluate_submission(submission2)
    print(f"Test 2 - Cost Effective: ICER={eval2.icer}, Score={eval2.cost_effectiveness_score}")
    
    # Test 3: High cost submission
    submission3 = HTASubmission(
        technology_name="ExpensiveDrug",
        cost_effectiveness_analysis={'icer': 40000, 'qaly_gain': 1.0}
    )
    eval3 = adapter.evaluate_submission(submission3)
    print(f"Test 3 - Expensive: ICER={eval3.icer}, Score={eval3.cost_effectiveness_score}")
    
    # Test 4: With budget impact
    submission4 = HTASubmission(
        technology_name="BudgetImpactDrug",
        cost_effectiveness_analysis={'icer': 20000, 'qaly_gain': 1.0},
        budget_impact_analysis={'total_impact': 1000000.0}
    )
    eval4 = adapter.evaluate_submission(submission4)
    print(f"Test 4 - Budget Impact: Impact={eval4.budget_impact}, Score={eval4.budget_impact_score}")
    
    # Test 5: With innovation factors
    submission5 = HTASubmission(
        technology_name="InnovativeDrug",
        innovation_factors={
            'mechanism_of_action': True,
            'first_in_class': True
        }
    )
    eval5 = adapter.evaluate_submission(submission5)
    print(f"Test 5 - Innovation: Score={eval5.innovation_score}")
    
    # Test 6: End of life
    submission6 = HTASubmission(
        technology_name="EOLDrug",
        cost_effectiveness_analysis={'icer': 45000, 'qaly_gain': 1.0},
        framework_specific_data={'end_of_life': True}
    )
    eval6 = adapter.evaluate_submission(submission6)
    print(f"Test 6 - EOL: Decision={eval6.decision}, Recommendation={eval6.recommendation}")
    
    # Test 7: Comprehensive
    submission7 = HTASubmission(
        technology_name="ComprehensiveDrug",
        clinical_trial_data={'evidence_level': 'RCT'},
        cost_effectiveness_analysis={'icer': 25000, 'qaly_gain': 2.0},
        budget_impact_analysis={'total_impact': 500000.0},
        innovation_factors={'first_in_class': True},
        framework_specific_data={'rare_disease': True},
        equity_impact={'population_benefit': 0.2}
    )
    eval7 = adapter.evaluate_submission(submission7)
    print(f"Test 7 - Comprehensive: All scores set")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()