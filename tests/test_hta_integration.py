"""Focused regression tests for NICE HTA threshold logic."""

import pytest

from voiage.hta_integration import (
    CADTHFrameworkAdapter,
    DecisionType,
    HTAFramework,
    HTAIntegrationFramework,
    HTASubmission,
    ICERFrameworkAdapter,
    NICEFrameworkAdapter,
    compare_hta_decisions,
    create_hta_submission,
    generate_hta_report,
    quick_hta_evaluation,
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
    assert evaluation.additional_evidence_needed == ["Detailed budget impact modeling"]


def test_nice_framework_adapter_applies_special_modifiers_and_uncertainties() -> None:
    """Special-case NICE modifiers should alter recommendations and diagnostics."""
    adapter = NICEFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "RCT"},
        cost_effectiveness_analysis={"icer": 25_000, "qaly_gain": 0.4},
        budget_impact_analysis={"total_impact": 0.04},
        framework_specific_data={"end_of_life": True, "rare_disease": True},
        equity_impact={"population_benefit": 0.35},
        economic_model={"structural_uncertainty": 0.5},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.decision == DecisionType.PRICE_NEGOTIATION
    assert evaluation.recommendation == "Price negotiation required"
    assert evaluation.cost_effectiveness_score == 0.7
    assert evaluation.equity_score == pytest.approx(0.85)
    assert "End of life treatment consideration applied" in evaluation.strengths
    assert "Rare disease modifier applied" in evaluation.strengths
    assert "Significant equity benefits identified" in evaluation.strengths
    assert evaluation.uncertainties == [
        "Limited real-world effectiveness data",
        "Significant structural uncertainty in economic model",
    ]


def test_nice_framework_adapter_rejects_failed_end_of_life_case() -> None:
    """End-of-life submissions above the modifier threshold should be rejected."""
    adapter = NICEFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "RCT"},
        cost_effectiveness_analysis={"icer": 80_000},
        framework_specific_data={"end_of_life": True},
        real_world_evidence={"available": True},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.decision == DecisionType.REJECTION
    assert evaluation.recommendation == "Not recommended for reimbursement"


def test_cadth_adapter_records_evidence_and_economic_findings() -> None:
    """CADTH should expose comparative evidence and threshold findings."""
    adapter = CADTHFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "RCT"},
        cost_effectiveness_analysis={"icer": 45_000},
        framework_specific_data={"comparative_effectiveness": True},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.decision == DecisionType.APPROVAL
    assert evaluation.icer == 45_000
    assert evaluation.clinical_effectiveness_score == 0.8
    assert evaluation.cost_effectiveness_score == 0.8
    assert evaluation.strengths == [
        "Comparative effectiveness data provided",
        "High quality RCT evidence",
        "Cost-effective within CADTH threshold",
    ]


def test_cadth_adapter_requests_additional_evidence_for_missing_comparator() -> None:
    """CADTH should request head-to-head evidence when comparator data is missing."""
    adapter = CADTHFrameworkAdapter()
    submission = HTASubmission(
        clinical_trial_data={"evidence_level": "observational"},
        cost_effectiveness_analysis={"icer": 75_000},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.decision == DecisionType.ADDITIONAL_EVIDENCE_REQUIRED
    assert evaluation.recommendation == (
        "Additional comparative effectiveness evidence required"
    )
    assert evaluation.clinical_effectiveness_score == 0.6
    assert evaluation.cost_effectiveness_score == 0.4
    assert evaluation.additional_evidence_needed == ["Head-to-head comparative trials"]
    assert evaluation.weaknesses == [
        "Limited clinical trial evidence",
        "Cost-effectiveness exceeds CADTH threshold",
    ]


def test_icer_adapter_uses_submitted_icer_for_value_based_pricing() -> None:
    """ICER value-based pricing should be driven by submitted ICER results."""
    adapter = ICERFrameworkAdapter()
    submission = HTASubmission(
        cost_effectiveness_analysis={"icer": 95_000},
        budget_impact_analysis={"monthly_per_member_increase": 0.0005},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.icer == 95_000
    assert evaluation.recommendation == "High value: <$100,000 per QALY"
    assert evaluation.budget_impact_score == 0.9
    assert evaluation.strengths == ["Acceptable budget impact"]


def test_icer_adapter_flags_excessive_budget_impact() -> None:
    """ICER should request access strategies for excessive budget impact."""
    adapter = ICERFrameworkAdapter()
    submission = HTASubmission(
        cost_effectiveness_analysis={"icer": 125_000},
        budget_impact_analysis={"monthly_per_member_increase": 0.002},
    )

    evaluation = adapter.evaluate_submission(submission)

    assert evaluation.recommendation == "Intermediate value: $100,000-$150,000 per QALY"
    assert evaluation.budget_impact_score == 0.3
    assert evaluation.weaknesses == ["Excessive budget impact identified"]
    assert evaluation.additional_evidence_needed == [
        "Patient access programs and outcomes-based agreements"
    ]


def test_framework_evaluation_warns_and_skips_unavailable_framework() -> None:
    """Multi-framework evaluation should warn and continue on missing adapters."""
    framework = HTAIntegrationFramework()
    submission = HTASubmission()

    with pytest.warns(UserWarning, match="Framework adapter not available"):
        evaluations = framework.evaluate_multiple_frameworks(
            submission, [HTAFramework.NICE, HTAFramework.TLV]
        )

    assert list(evaluations) == [HTAFramework.NICE]


def test_framework_strategy_summarizes_differences_and_recommendations() -> None:
    """Strategy creation should expose comparisons, gaps, and optimization advice."""
    framework = HTAIntegrationFramework()
    evaluations = {
        HTAFramework.NICE: framework.evaluate_for_framework(
            HTASubmission(
                clinical_trial_data={"evidence_level": "RCT"},
                cost_effectiveness_analysis={"icer": 20_000},
                budget_impact_analysis={"total_impact": 0.01},
                innovation_factors={"breakthrough_therapy": True},
                real_world_evidence={"available": True},
            ),
            HTAFramework.NICE,
        ),
        HTAFramework.CADTH: framework.evaluate_for_framework(
            HTASubmission(
                clinical_trial_data={"evidence_level": "observational"},
                cost_effectiveness_analysis={"icer": 90_000},
            ),
            HTAFramework.CADTH,
        ),
        HTAFramework.ICER: framework.evaluate_for_framework(
            HTASubmission(
                cost_effectiveness_analysis={"icer": 140_000},
                budget_impact_analysis={"monthly_per_member_increase": 0.002},
            ),
            HTAFramework.ICER,
        ),
    }

    comparison = framework.compare_framework_decisions(evaluations)
    recommendations = framework._generate_strategy_recommendations(
        evaluations, comparison
    )
    optimization = framework._generate_optimization_suggestions(evaluations)

    assert comparison["decision_agreement"] is False
    assert comparison["icer_range"]["min"] == 20_000
    assert comparison["icer_range"]["max"] == 140_000
    assert (
        "Significant variation in cost-effectiveness thresholds"
        in comparison["key_differences"]
    )
    assert (
        "Different budget impact evaluation approaches" in comparison["key_differences"]
    )
    assert "Address evidence gaps for cadth: Head-to-head comparative trials" in (
        recommendations
    )
    assert (
        "Mixed framework responses - consider targeted strategies for each market"
        in (recommendations)
    )
    assert "Consider cost reduction strategies to improve ICER across frameworks" in (
        optimization
    )
    assert "Implement patient access programs to mitigate budget impact" in optimization


def test_framework_strategy_identifies_repeated_evidence_gaps() -> None:
    """Repeated evidence gaps should be promoted as priority gaps."""
    framework = HTAIntegrationFramework()
    evaluations = {
        HTAFramework.NICE: framework.evaluate_for_framework(
            HTASubmission(
                clinical_trial_data={"evidence_level": "observational"},
                cost_effectiveness_analysis={"icer": 45_000},
                budget_impact_analysis={"total_impact": 0.05},
                real_world_evidence={"available": True},
            ),
            HTAFramework.NICE,
        ),
        HTAFramework.ICER: framework.evaluate_for_framework(
            HTASubmission(
                cost_effectiveness_analysis={"icer": 125_000},
                budget_impact_analysis={"monthly_per_member_increase": 0.002},
            ),
            HTAFramework.ICER,
        ),
    }
    shared_gap = "Patient access programs and outcomes-based agreements"
    evaluations[HTAFramework.NICE].additional_evidence_needed.append(shared_gap)

    priority_gaps = framework._identify_priority_evidence_gaps(evaluations)

    assert priority_gaps == [shared_gap]


def test_hta_utility_functions_create_evaluate_compare_and_report() -> None:
    """Convenience helpers should compose the framework APIs."""
    submission = create_hta_submission(
        technology_name="ExampleTx",
        manufacturer="ExampleCo",
        indication="Rare disease",
        economic_results={"icer": 18_000, "qaly_gain": 0.3},
        clinical_results={"evidence_level": "RCT"},
    )

    evaluation = quick_hta_evaluation(submission)
    strategy = compare_hta_decisions(submission, frameworks=[HTAFramework.NICE])
    default_strategy = compare_hta_decisions(submission)
    report = generate_hta_report(submission)

    assert submission.population == "Adult patients"
    assert evaluation.framework == HTAFramework.NICE
    assert strategy["target_frameworks"] == ["nice"]
    assert default_strategy["target_frameworks"] == ["nice", "cadth", "icer"]
    assert "Technology: ExampleTx" in report
    assert "DECISION: approval" in report


def test_framework_evaluation_handles_exceptions() -> None:
    """Multi-framework evaluation should catch exceptions, warn, and continue."""
    from unittest.mock import patch

    framework = HTAIntegrationFramework()
    submission = HTASubmission()

    with patch.object(
        HTAIntegrationFramework,
        "evaluate_for_framework",
        side_effect=Exception("Mocked evaluation failure"),
    ):
        with pytest.warns(
            UserWarning,
            match="Error evaluating HTAFramework.NICE: Mocked evaluation failure",
        ):
            evaluations = framework.evaluate_multiple_frameworks(
                submission, [HTAFramework.NICE]
            )

    assert len(evaluations) == 0
