"""
Health Technology Assessment (HTA) Integration Module.

This module provides integration with health technology assessment frameworks:
- NICE (UK National Institute for Health and Care Excellence) guidelines
- CADTH (Canadian Agency for Drugs and Technologies in Health) methods
- IQWiG (Institute for Quality and Efficiency in Health Care) criteria
- AMCP (Academy of Managed Care Pharmacy) formulary evaluations
- ICER (Institute for Clinical and Economic Review) assessments
- PBAC (Pharmaceutical Benefits Advisory Committee) evaluations

Author: voiage Development Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import cast
import warnings

import numpy as np

from voiage.exceptions import raise_value_error


class HTAFramework(Enum):
    """Supported HTA frameworks."""

    NICE = "nice"  # UK
    CADTH = "cadth"  # Canada
    IQWIG = "iqwig"  # Germany
    AMCP = "amcp"  # US Managed Care
    ICER = "icer"  # US
    PBAC = "pbac"  # Australia
    TLV = "tlv"  # Sweden
    HAS = "has"  # France


class DecisionType(Enum):
    """Types of HTA decisions."""

    APPROVAL = "approval"
    REJECTION = "rejection"
    RESTRICTED_APPROVAL = "restricted_approval"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence"
    PRICE_NEGOTIATION = "price_negotiation"
    MANAGED_ENTRY = "managed_entry"


class EvidenceRequirement(Enum):
    """Types of evidence requirements."""

    CLINICAL_EFFECTIVENESS = "clinical_effectiveness"
    COST_EFFECTIVENESS = "cost_effectiveness"
    BUDGET_IMPACT = "budget_impact"
    EQUITY = "equity"
    INNOVATION = "innovation"
    UNMET_NEED = "unmet_need"


@dataclass
class HTAFrameworkCriteria:
    """HTA framework evaluation criteria."""

    framework: HTAFramework
    max_icer_threshold: float
    min_qaly_threshold: float
    budget_impact_threshold: float  # as % of total health budget
    evidence_hierarchy: list[str]  # evidence types in order of importance
    special_considerations: list[str]
    decision_factors: dict[str, float]  # weights for different factors
    submission_requirements: dict[str, object]

    # Framework-specific parameters
    discount_rate: float = 0.035
    time_horizon: float = 10.0
    population_threshold: int = 1000  # minimum population for analysis
    severity_modifier: float = 1.0  # for severe conditions
    innovation_score_weight: float = 0.1


@dataclass
class HTASubmission:
    """HTA submission data structure."""

    technology_name: str = ""
    manufacturer: str = ""
    indication: str = ""
    population: str = ""
    comparators: list[str] = field(default_factory=list)

    # Clinical evidence
    clinical_trial_data: dict[str, object] = field(default_factory=dict)
    real_world_evidence: dict[str, object] | None = None
    systematic_review: dict[str, object] | None = None

    # Economic evidence
    economic_model: dict[str, object] = field(default_factory=dict)
    cost_effectiveness_analysis: dict[str, object] = field(default_factory=dict)
    budget_impact_analysis: dict[str, object] = field(default_factory=dict)

    # Additional considerations
    equity_impact: dict[str, object] | None = None
    innovation_factors: dict[str, object] | None = None
    unmet_need_assessment: dict[str, object] | None = None

    # Framework specific
    framework_specific_data: dict[str, object] = field(default_factory=dict)
    submission_metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class HTAEvaluation:
    """HTA evaluation result."""

    framework: HTAFramework
    decision: DecisionType
    recommendation: str

    # Quantitative metrics
    icer: float | None = None
    qaly_gain: float | None = None
    net_monetary_benefit: float | None = None
    budget_impact: float | None = None

    # Decision scores
    clinical_effectiveness_score: float | None = None
    cost_effectiveness_score: float | None = None
    budget_impact_score: float | None = None
    innovation_score: float | None = None
    equity_score: float | None = None

    # Detailed reasoning
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    additional_evidence_needed: list[str] = field(default_factory=list)

    # Recommendations
    price_recommendation: float | None = None
    population_restrictions: list[str] | None = None
    evidence_requirements: list[str] = field(default_factory=list)
    post_launch_studies: list[str] = field(default_factory=list)


class NICEFrameworkAdapter:
    """NICE (UK) HTA framework adapter."""

    def __init__(self) -> None:
        self.criteria = HTAFrameworkCriteria(
            framework=HTAFramework.NICE,
            max_icer_threshold=30000.0,  # £30,000 per QALY
            min_qaly_threshold=0.0,
            budget_impact_threshold=0.02,  # 2% of total health budget
            evidence_hierarchy=[
                "randomized_controlled_trials",
                "systematic_reviews",
                "real_world_evidence",
                "observational_studies",
            ],
            special_considerations=[
                "end_of_life",
                "rare_diseases",
                "severity_modifier",
            ],
            decision_factors={
                "clinical_effectiveness": 0.4,
                "cost_effectiveness": 0.3,
                "innovation": 0.1,
                "equity": 0.1,
                "uncertainties": 0.1,
            },
            submission_requirements={
                "must_include_cea": True,
                "must_include_bia": True,
                "min_trial_duration": 12,  # months
                "required_evidence_levels": ["RCT", "Systematic Review"],
            },
        )

    def evaluate_submission(self, submission: HTASubmission) -> HTAEvaluation:
        """Evaluate submission according to NICE criteria."""
        evaluation = HTAEvaluation(
            framework=HTAFramework.NICE,
            decision=DecisionType.APPROVAL,  # Default
            recommendation="Provisional approval",
        )

        # Extract key metrics from submission
        cea_results = submission.cost_effectiveness_analysis
        bia_results = submission.budget_impact_analysis

        if cea_results:
            evaluation.icer = cea_results.get("icer")
            evaluation.qaly_gain = cea_results.get("qaly_gain")
            evaluation.net_monetary_benefit = cea_results.get("net_monetary_benefit")

        if bia_results:
            evaluation.budget_impact = bia_results.get("total_impact")

        # NICE-specific evaluation logic
        strengths = []
        weaknesses = []
        uncertainties = []

        # Clinical effectiveness evaluation
        trial_data = submission.clinical_trial_data
        if trial_data.get("evidence_level") == "RCT":
            evaluation.clinical_effectiveness_score = 0.8
            strengths.append("High quality RCT evidence available")
        else:
            evaluation.clinical_effectiveness_score = 0.5
            weaknesses.append("Limited clinical evidence quality")

        # Cost-effectiveness evaluation
        if evaluation.icer is not None:
            if evaluation.icer <= 20000:  # £20,000 threshold
                evaluation.cost_effectiveness_score = 0.9
                strengths.append("Cost-effective within standard threshold")
            elif evaluation.icer <= 30000:  # £30,000 threshold
                evaluation.cost_effectiveness_score = 0.7
                strengths.append("Cost-effective within higher threshold")
            else:
                evaluation.cost_effectiveness_score = 0.3
                weaknesses.append("Cost-effectiveness exceeds standard thresholds")

        # Budget impact evaluation
        if evaluation.budget_impact:
            if evaluation.budget_impact > self.criteria.budget_impact_threshold:
                evaluation.budget_impact_score = 0.4
                weaknesses.append("Significant budget impact identified")
                evaluation.additional_evidence_needed.append(
                    "Detailed budget impact modeling"
                )
            else:
                evaluation.budget_impact_score = 0.8
                strengths.append("Acceptable budget impact")

        # Innovation assessment
        innovation_score = 0.5
        if submission.innovation_factors:
            if submission.innovation_factors.get("mechanism_of_action", False):
                innovation_score += 0.2
            if submission.innovation_factors.get("first_in_class", False):
                innovation_score += 0.2
            if submission.innovation_factors.get("breakthrough_therapy", False):
                innovation_score += 0.1

        evaluation.innovation_score = min(1.0, innovation_score)

        # End of life considerations
        if submission.framework_specific_data.get("end_of_life", False):
            if (
                evaluation.icer is not None and evaluation.icer <= 50000
            ):  # £50,000 for end of life
                evaluation.recommendation = "Approved for end of life treatment"
                strengths.append("End of life treatment consideration applied")
            else:
                evaluation.decision = DecisionType.REJECTION
                evaluation.recommendation = "Not recommended"

        # Rare disease considerations
        if (
            submission.framework_specific_data.get("rare_disease", False)
            and evaluation.icer is not None
            and evaluation.icer <= 100000  # Higher threshold for rare diseases
        ):
            evaluation.recommendation = "Approved for rare disease"
            strengths.append("Rare disease modifier applied")

        # Equity considerations
        if submission.equity_impact:
            equity_benefit = submission.equity_impact.get("population_benefit", 0.0)
            evaluation.equity_score = min(1.0, 0.5 + equity_benefit)
            if equity_benefit > 0.2:
                strengths.append("Significant equity benefits identified")

        # Overall decision logic
        if (
            evaluation.cost_effectiveness_score
            and evaluation.cost_effectiveness_score < 0.5
        ):
            evaluation.decision = DecisionType.REJECTION
            evaluation.recommendation = "Not recommended for reimbursement"
        elif evaluation.budget_impact_score and evaluation.budget_impact_score < 0.5:
            evaluation.decision = DecisionType.PRICE_NEGOTIATION
            evaluation.recommendation = "Price negotiation required"

        # Uncertainties identification
        if not submission.real_world_evidence:
            uncertainties.append("Limited real-world effectiveness data")
        if submission.economic_model.get("structural_uncertainty", 0) > 0.3:
            uncertainties.append("Significant structural uncertainty in economic model")

        evaluation.uncertainties = uncertainties
        evaluation.strengths = strengths
        evaluation.weaknesses = weaknesses

        return evaluation


class CADTHFrameworkAdapter:
    """CADTH (Canada) HTA framework adapter."""

    def __init__(self) -> None:
        self.criteria = HTAFrameworkCriteria(
            framework=HTAFramework.CADTH,
            max_icer_threshold=50000.0,  # CAD$50,000 per QALY
            min_qaly_threshold=0.0,
            budget_impact_threshold=0.03,  # 3% of drug budget
            evidence_hierarchy=[
                "systematic_reviews",
                "randomized_controlled_trials",
                "controlled_clinical_trials",
                "observational_studies",
            ],
            special_considerations=[
                "rare_diseases",
                "pediatric_indications",
                "comparative_effectiveness",
            ],
            decision_factors={
                "clinical_effectiveness": 0.5,
                "cost_effectiveness": 0.3,
                "feasibility": 0.2,
            },
            submission_requirements={
                "must_include_cea": True,
                "must_include_bia": True,
                "comparative_effectiveness": True,
            },
        )

    def evaluate_submission(self, submission: HTASubmission) -> HTAEvaluation:
        """Evaluate submission according to CADTH criteria."""
        evaluation = HTAEvaluation(
            framework=HTAFramework.CADTH,
            decision=DecisionType.APPROVAL,
            recommendation="Recommended for listing",
        )

        # CADTH-specific evaluation logic
        strengths = []
        weaknesses = []

        # Comparative effectiveness requirement
        if not submission.framework_specific_data.get(
            "comparative_effectiveness", False
        ):
            evaluation.decision = DecisionType.ADDITIONAL_EVIDENCE_REQUIRED
            evaluation.recommendation = (
                "Additional comparative effectiveness evidence required"
            )
            evaluation.additional_evidence_needed.append(
                "Head-to-head comparative trials"
            )
        else:
            strengths.append("Comparative effectiveness data provided")

        # Clinical evidence quality
        if submission.clinical_trial_data.get("evidence_level") == "RCT":
            evaluation.clinical_effectiveness_score = 0.8
            strengths.append("High quality RCT evidence")
        else:
            evaluation.clinical_effectiveness_score = 0.6
            weaknesses.append("Limited clinical trial evidence")

        # Economic evaluation
        if submission.cost_effectiveness_analysis:
            evaluation.icer = submission.cost_effectiveness_analysis.get("icer")
            if evaluation.icer and evaluation.icer <= self.criteria.max_icer_threshold:
                evaluation.cost_effectiveness_score = 0.8
                strengths.append("Cost-effective within CADTH threshold")
            else:
                evaluation.cost_effectiveness_score = 0.4
                weaknesses.append("Cost-effectiveness exceeds CADTH threshold")

        evaluation.strengths = strengths
        evaluation.weaknesses = weaknesses
        return evaluation


class ICERFrameworkAdapter:
    """ICER (US) HTA framework adapter."""

    def __init__(self) -> None:
        self.criteria = HTAFrameworkCriteria(
            framework=HTAFramework.ICER,
            max_icer_threshold=150000.0,  # $150,000 per QALY (ICER threshold)
            min_qaly_threshold=0.0,
            budget_impact_threshold=0.001,  # $0.001 per member per month increase
            evidence_hierarchy=[
                "randomized_controlled_trials",
                "real_world_evidence",
                "systematic_reviews",
            ],
            special_considerations=["innovation", "uncertainty", "budget_impact"],
            decision_factors={
                "clinical_effectiveness": 0.4,
                "cost_effectiveness": 0.3,
                "budget_impact": 0.2,
                "innovation": 0.1,
            },
            submission_requirements={
                "must_include_cea": True,
                "must_include_bia": True,
                "uncertainty_analysis": True,
            },
        )

    def evaluate_submission(self, submission: HTASubmission) -> HTAEvaluation:
        """Evaluate submission according to ICER criteria."""
        evaluation = HTAEvaluation(
            framework=HTAFramework.ICER,
            decision=DecisionType.APPROVAL,
            recommendation="Value-based price benchmark",
        )

        # ICER-specific evaluation
        strengths = []
        weaknesses = []

        # Budget impact analysis (critical for ICER)
        if submission.budget_impact_analysis:
            bia_monthly_per_member = submission.budget_impact_analysis.get(
                "monthly_per_member_increase", 0
            )
            if bia_monthly_per_member <= self.criteria.budget_impact_threshold:
                evaluation.budget_impact_score = 0.9
                strengths.append("Acceptable budget impact")
            else:
                evaluation.budget_impact_score = 0.3
                weaknesses.append("Excessive budget impact identified")
                evaluation.additional_evidence_needed.append(
                    "Patient access programs and outcomes-based agreements"
                )

        # Value-based pricing
        if submission.cost_effectiveness_analysis:
            evaluation.icer = submission.cost_effectiveness_analysis.get("icer")

        if evaluation.icer and evaluation.icer <= self.criteria.max_icer_threshold:
            if evaluation.icer <= 100000:
                evaluation.recommendation = "High value: <$100,000 per QALY"
            elif evaluation.icer <= 150000:
                evaluation.recommendation = (
                    "Intermediate value: $100,000-$150,000 per QALY"
                )
            else:
                evaluation.recommendation = "Low value: >$150,000 per QALY"

        evaluation.strengths = strengths
        evaluation.weaknesses = weaknesses
        return evaluation


class HTAIntegrationFramework:
    """Main HTA integration framework.

    Provides a unified interface for different HTA agencies and
    decision-making bodies.
    """

    def __init__(self) -> None:
        self.framework_adapters = {
            HTAFramework.NICE: NICEFrameworkAdapter(),
            HTAFramework.CADTH: CADTHFrameworkAdapter(),
            HTAFramework.ICER: ICERFrameworkAdapter(),
        }

    def add_framework_adapter(self, framework: HTAFramework, adapter: object) -> None:
        """Add custom framework adapter."""
        self.framework_adapters[framework] = adapter

    def evaluate_for_framework(
        self, submission: HTASubmission, framework: HTAFramework
    ) -> HTAEvaluation:
        """
        Evaluate submission for specific HTA framework.

        Args:
            submission: HTA submission data
            framework: Target HTA framework

        Returns
        -------
            Evaluation result for the framework
        """
        if framework not in self.framework_adapters:
            raise_value_error(f"Framework adapter not available for {framework}")

        adapter: object = self.framework_adapters[framework]
        return cast("HTAEvaluation", adapter.evaluate_submission(submission))

    def evaluate_multiple_frameworks(
        self, submission: HTASubmission, frameworks: list[HTAFramework]
    ) -> dict[HTAFramework, HTAEvaluation]:
        """
        Evaluate submission across multiple frameworks.

        Args:
            submission: HTA submission data
            frameworks: List of HTA frameworks to evaluate

        Returns
        -------
            Dictionary of evaluations by framework
        """
        evaluations: dict[HTAFramework, HTAEvaluation] = {}

        def _evaluate_framework(framework: HTAFramework) -> HTAEvaluation | None:
            try:
                return self.evaluate_for_framework(submission, framework)
            except Exception as e:
                warnings.warn(f"Error evaluating {framework}: {e}", stacklevel=2)
                return None

        for framework in frameworks:
            evaluation = _evaluate_framework(framework)
            if evaluation is not None:
                evaluations[framework] = evaluation

        return evaluations

    def compare_framework_decisions(
        self, submissions_evaluations: dict[HTAFramework, HTAEvaluation]
    ) -> dict[str, object]:
        """
        Compare decisions across different HTA frameworks.

        Args:
            submissions_evaluations: Evaluation results by framework

        Returns
        -------
            Comparison analysis
        """
        frameworks = list(submissions_evaluations.keys())
        decisions = [
            eval_result.decision.value
            for eval_result in submissions_evaluations.values()
        ]
        recommendations = [
            eval_result.recommendation
            for eval_result in submissions_evaluations.values()
        ]

        # Agreement analysis
        unique_decisions = list(set(decisions))
        decision_agreement = len(unique_decisions) == 1

        # ICER comparisons
        icers = [
            eval_result.icer
            for eval_result in submissions_evaluations.values()
            if eval_result.icer is not None
        ]

        return {
            "frameworks_evaluated": [f.value for f in frameworks],
            "decisions": decisions,
            "recommendations": recommendations,
            "decision_agreement": decision_agreement,
            "unique_decisions": unique_decisions,
            "icer_range": {
                "min": min(icers) if icers else None,
                "max": max(icers) if icers else None,
                "mean": np.mean(icers) if icers else None,
            }
            if icers
            else None,
            "key_differences": self._identify_key_differences(submissions_evaluations),
            "recommendation_summary": self._summarize_recommendations(recommendations),
        }

    def _identify_key_differences(
        self, evaluations: dict[HTAFramework, HTAEvaluation]
    ) -> list[str]:
        """Identify key differences between framework evaluations."""
        differences = []

        # Compare cost-effectiveness thresholds
        icers = [e.icer for e in evaluations.values() if e.icer is not None]
        if len(icers) > 1 and max(icers) - min(icers) > 50000:
            differences.append("Significant variation in cost-effectiveness thresholds")

        # Compare budget impact considerations
        budget_scores = [
            e.budget_impact_score
            for e in evaluations.values()
            if e.budget_impact_score is not None
        ]
        if len(budget_scores) > 1 and max(budget_scores) - min(budget_scores) > 0.3:
            differences.append("Different budget impact evaluation approaches")

        # Compare innovation considerations
        innovation_scores = [
            e.innovation_score
            for e in evaluations.values()
            if e.innovation_score is not None
        ]
        if (
            len(innovation_scores) > 1
            and max(innovation_scores) - min(innovation_scores) > 0.3
        ):
            differences.append("Different innovation assessment criteria")

        return differences

    def _summarize_recommendations(
        self, recommendations: list[str]
    ) -> dict[str, object]:
        """Summarize recommendations across frameworks."""
        recommendation_counts: dict[str, int] = {}
        for rec in recommendations:
            # Categorize recommendations
            if "approved" in rec.lower() or "recommended" in rec.lower():
                category = "Positive"
            elif "rejected" in rec.lower() or "not recommended" in rec.lower():
                category = "Negative"
            elif "additional" in rec.lower() or "evidence" in rec.lower():
                category = "Conditional"
            else:
                category = "Other"

            recommendation_counts[category] = recommendation_counts.get(category, 0) + 1

        return {
            "recommendation_distribution": recommendation_counts,
            "overall_tone": max(
                recommendation_counts,
                key=lambda key: recommendation_counts[key],
            ),
            "consensus_level": max(recommendation_counts.values())
            / len(recommendations)
            if recommendations
            else 0,
        }

    def create_hta_strategy(
        self, submission: HTASubmission, target_frameworks: list[HTAFramework]
    ) -> dict[str, object]:
        """
        Create HTA strategy for multiple frameworks.

        Args:
            submission: HTA submission data
            target_frameworks: Target HTA frameworks

        Returns
        -------
            Comprehensive HTA strategy
        """
        evaluations = self.evaluate_multiple_frameworks(submission, target_frameworks)
        comparison = self.compare_framework_decisions(evaluations)

        # Generate strategy recommendations
        return {
            "target_frameworks": [f.value for f in target_frameworks],
            "evaluations": {
                f.value: self._evaluation_to_dict(eval_result)
                for f, eval_result in evaluations.items()
            },
            "comparison": comparison,
            "strategy_recommendations": self._generate_strategy_recommendations(
                evaluations, comparison
            ),
            "priority_evidence_gaps": self._identify_priority_evidence_gaps(
                evaluations
            ),
            "optimization_suggestions": self._generate_optimization_suggestions(
                evaluations
            ),
        }

    def _evaluation_to_dict(self, evaluation: HTAEvaluation) -> dict[str, object]:
        """Convert evaluation to dictionary."""
        return {
            "decision": evaluation.decision.value,
            "recommendation": evaluation.recommendation,
            "icer": evaluation.icer,
            "qaly_gain": evaluation.qaly_gain,
            "budget_impact": evaluation.budget_impact,
            "scores": {
                "clinical_effectiveness": evaluation.clinical_effectiveness_score,
                "cost_effectiveness": evaluation.cost_effectiveness_score,
                "budget_impact": evaluation.budget_impact_score,
                "innovation": evaluation.innovation_score,
                "equity": evaluation.equity_score,
            },
            "strengths": evaluation.strengths,
            "weaknesses": evaluation.weaknesses,
            "uncertainties": evaluation.uncertainties,
            "additional_evidence_needed": evaluation.additional_evidence_needed,
        }

    def _generate_strategy_recommendations(
        self,
        evaluations: dict[HTAFramework, HTAEvaluation],
        comparison: dict[str, object],
    ) -> list[str]:
        """Generate strategic recommendations for HTA submission."""
        recommendations = []

        # Framework-specific recommendations
        for framework, evaluation in evaluations.items():
            if evaluation.decision == DecisionType.REJECTION:
                recommendations.append(
                    f"Consider reformulation or additional evidence for {framework.value}"
                )
            elif evaluation.decision == DecisionType.ADDITIONAL_EVIDENCE_REQUIRED:
                recommendations.append(
                    f"Address evidence gaps for {framework.value}: {', '.join(evaluation.additional_evidence_needed)}"
                )
            elif evaluation.decision == DecisionType.PRICE_NEGOTIATION:
                recommendations.append(
                    f"Prepare for price negotiation in {framework.value}"
                )

        # Overall strategy
        if comparison["decision_agreement"]:
            recommendations.append(
                "Strong consensus across frameworks - leverage for market access"
            )
        else:
            recommendations.append(
                "Mixed framework responses - consider targeted strategies for each market"
            )

        return recommendations

    def _identify_priority_evidence_gaps(
        self, evaluations: dict[HTAFramework, HTAEvaluation]
    ) -> list[str]:
        """Identify priority evidence gaps across all frameworks."""
        all_gaps = []
        for evaluation in evaluations.values():
            all_gaps.extend(evaluation.additional_evidence_needed)

        # Count frequency of evidence gaps
        gap_counts: dict[str, int] = {}
        for gap in all_gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1

        # Prioritize by frequency and importance
        priority_gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            gap for gap, count in priority_gaps if count >= 2
        ]  # Gaps identified by multiple frameworks

    def _generate_optimization_suggestions(
        self, evaluations: dict[HTAFramework, HTAEvaluation]
    ) -> list[str]:
        """Generate optimization suggestions based on evaluation results."""
        suggestions = []

        # ICER optimization
        low_icer_count = sum(
            1 for e in evaluations.values() if e.icer and e.icer < 50000
        )
        if low_icer_count < len(evaluations) / 2:
            suggestions.append(
                "Consider cost reduction strategies to improve ICER across frameworks"
            )

        # Budget impact optimization
        high_budget_impact_count = sum(
            1
            for e in evaluations.values()
            if e.budget_impact_score and e.budget_impact_score < 0.5
        )
        if high_budget_impact_count > 0:
            suggestions.append(
                "Implement patient access programs to mitigate budget impact"
            )

        # Innovation positioning
        low_innovation_count = sum(
            1
            for e in evaluations.values()
            if e.innovation_score and e.innovation_score < 0.6
        )
        if low_innovation_count > 0:
            suggestions.append("Strengthen innovation narrative and value proposition")

        return suggestions


# Utility functions for HTA integration


def create_hta_submission(
    technology_name: str,
    manufacturer: str,
    indication: str,
    economic_results: dict[str, object],
    clinical_results: dict[str, object],
) -> HTASubmission:
    """Create basic HTA submission."""
    return HTASubmission(
        technology_name=technology_name,
        manufacturer=manufacturer,
        indication=indication,
        population="Adult patients",
        comparators=["Standard of care"],
        clinical_trial_data=clinical_results,
        economic_model={"type": "Markov model", "time_horizon": 10},
        cost_effectiveness_analysis=economic_results,
    )


def quick_hta_evaluation(
    submission: HTASubmission, framework: HTAFramework = HTAFramework.NICE
) -> HTAEvaluation:
    """Quick HTA evaluation for single framework."""
    hta_framework = HTAIntegrationFramework()
    return hta_framework.evaluate_for_framework(submission, framework)


def compare_hta_decisions(
    submission: HTASubmission, frameworks: list[HTAFramework] | None = None
) -> dict[str, object]:
    """Compare HTA decisions across multiple frameworks."""
    if frameworks is None:
        frameworks = [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]

    hta_framework = HTAIntegrationFramework()
    return hta_framework.create_hta_strategy(submission, frameworks)


def generate_hta_report(
    submission: HTASubmission, framework: HTAFramework = HTAFramework.NICE
) -> str:
    """Generate HTA evaluation report."""
    evaluation = quick_hta_evaluation(submission, framework)

    return f"""
HTA Evaluation Report
====================

Technology: {submission.technology_name}
Indication: {submission.indication}
Framework: {framework.value}

DECISION: {evaluation.decision.value}
RECOMMENDATION: {evaluation.recommendation}

Key Metrics:
- ICER: {evaluation.icer or "Not calculated"}
- QALY Gain: {evaluation.qaly_gain or "Not specified"}
- Budget Impact: {evaluation.budget_impact or "Not specified"}

Strengths:
{chr(10).join(f"- {strength}" for strength in evaluation.strengths)}

Weaknesses:
{chr(10).join(f"- {weakness}" for weakness in evaluation.weaknesses)}

Additional Evidence Required:
{chr(10).join(f"- {evidence}" for evidence in evaluation.additional_evidence_needed)}
"""
