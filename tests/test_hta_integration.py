"""Focused regression tests for NICE HTA threshold logic."""

import pytest

from voiage.hta_integration import (
    DecisionType,
    HTAFramework,
    HTASubmission,
    NICEFrameworkAdapter,
)


def test_nice_framework_adapter_approves_high_quality_submission() -> None:
    """High-quality evidence and favorable economics should retain approval."""
    adapter = NICEFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "RCT"},
        cost_effectiveness_analysis={"icer": 15_000},
        budget_impact_analysis={"total_impact": 0.01},
        innovation_factors={
            "mechanism_of_action": True,
            "first_in_class": True,
            "breakthrough_therapy": True,
        },
        real_world_evidence={"available": True},
        framework_specific_data={},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.framework == HTAFramework.NICE
    assert evaluation.decision == DecisionType.APPROVAL
    assert evaluation.recommendation == "Provisional approval"
    assert evaluation.clinical_effectiveness_score == 0.8
    assert evaluation.cost_effectiveness_score == 0.9
    assert evaluation.budget_impact_score == 0.8
    assert evaluation.innovation_score == pytest.approx(1.0)
    assert evaluation.strengths == [
        "High quality RCT evidence available",
        "Cost-effective within standard threshold",
        "Acceptable budget impact",
    ]
    assert evaluation.weaknesses == []
    assert evaluation.additional_evidence_needed == []


def test_nice_framework_adapter_rejects_weak_submission() -> None:
    """Poor evidence and unfavorable thresholds should drive rejection."""
    adapter = NICEFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "observational"},
        cost_effectiveness_analysis={"icer": 60_000},
        budget_impact_analysis={"total_impact": 0.05},
        innovation_factors={},
        real_world_evidence={"available": True},
        framework_specific_data={},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.framework == HTAFramework.NICE
    assert evaluation.decision == DecisionType.REJECTION
    assert evaluation.recommendation == "Not recommended for reimbursement"
    assert evaluation.clinical_effectiveness_score == 0.5
    assert evaluation.cost_effectiveness_score == 0.3
    assert evaluation.budget_impact_score == 0.4
    assert evaluation.weaknesses == [
        "Limited clinical evidence quality",
        "Cost-effectiveness exceeds standard thresholds",
        "Significant budget impact identified",
    ]
    assert evaluation.additional_evidence_needed == [
        "Detailed budget impact modeling"
    ]
