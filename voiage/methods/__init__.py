"""Lazy public exports for Value of Information methods.

The stable kernel imports method facades directly.  Optional and experimental
methods remain discoverable through this compatibility namespace without being
imported during ``import voiage``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adaptive import adaptive_evsi
    from .adaptive_learning_bandit import (
        AdaptiveLearningBanditResult,
        value_of_adaptive_learning_bandit,
    )
    from .ai_assisted_evidence_triage import (
        AIAssistedEvidenceTriageResult,
        value_of_ai_assisted_evidence_triage,
    )
    from .ambiguity_distribution_shift import (
        AmbiguityDistributionShiftResult,
        value_of_ambiguity_distribution_shift,
    )
    from .basic import evpi, evppi
    from .belief_state_information import (
        BeliefStateInformationResult,
        belief_state_information_value,
    )
    from .calibration import voi_calibration
    from .capacity_budget_constrained import (
        CapacityBudgetConstrainedResult,
        value_of_capacity_budget_constrained,
    )
    from .causal_transportability import (
        CausalTransportabilityResult,
        value_of_causal_transportability,
    )
    from .ceaf import CEAFResult, calculate_ceaf
    from .computational import ComputationalResult, value_of_computational_refinement
    from .data_quality import DataQualityResult, value_of_data_quality
    from .deterministic_sensitivity import (
        DeterministicSensitivityResult,
        DsaParameterSummary,
        DsaPoint,
        DsaSwitchInterval,
        deterministic_sensitivity,
        deterministic_sensitivity_from_specification,
    )
    from .distributional import (
        DistributionalEquityResult,
        value_of_distributional_equity,
    )
    from .distributional_information import (
        DistributionalInformationResult,
        ResolvedDistributionModel,
        distributional_information_from_specification,
        value_of_distributional_information,
    )
    from .dominance import (
        DominanceResult,
        calculate_dominance,
        calculate_extended_dominance,
        calculate_icers,
        calculate_strong_dominance,
        cost_effectiveness_frontier,
    )
    from .dynamic_real_options import (
        DynamicRealOptionsResult,
        ValueOfFlexibilityResult,
        value_of_dynamic_real_options,
        value_of_flexibility,
    )
    from .equity_information import EquityInformationResult, value_of_equity_information
    from .event_localized_information import (
        EventLocalizedInformationResult,
        event_localized_information_value,
    )
    from .evidence_obsolescence_refresh import (
        EvidenceObsolescenceRefreshResult,
        value_of_evidence_obsolescence_refresh,
    )
    from .expert_synthesis import ExpertSynthesisResult, value_of_expert_synthesis
    from .explainability_transparency import (
        ExplainabilityTransparencyResult,
        value_of_explainability_transparency,
    )
    from .federated_privacy_preserving import (
        FederatedPrivacyPreservingResult,
        value_of_federated_privacy_preserving,
    )
    from .forecast_signal_information import (
        ForecastSignalInformationResult,
        forecast_signal_information_value,
    )
    from .heterogeneity import (
        HeterogeneityResult,
        identify_optimal_subgroups,
        value_of_heterogeneity,
    )
    from .heterogeneity_value import (
        HeterogeneityValueDecompositionResult,
        heterogeneity_value_decomposition,
    )
    from .implementation import ImplementationAdjustedResult, value_of_implementation
    from .implementation_information import (
        ImplementationInformationResult,
        implementation_information_value,
    )
    from .implementation_strategy import (
        ImplementationStrategyComparisonResult,
        value_of_implementation_strategy_comparison,
    )
    from .information_source_portfolio import (
        InformationSourcePortfolioResult,
        information_source_portfolio_value,
    )
    from .interoperability_standardization import (
        InteroperabilityStandardizationResult,
        value_of_interoperability_standardization,
    )
    from .mcda_information import McdaInformationResult, mcda_information_value
    from .monitoring_surveillance import (
        MonitoringSurveillanceResult,
        value_of_monitoring_surveillance,
    )
    from .network_nma import evsi_nma
    from .observational import voi_observational
    from .outcome_conditional_sample_information import (
        OutcomeConditionalSampleInformationResult,
        outcome_conditional_sample_information_value,
    )
    from .perspective import (
        Perspective,
        PerspectiveSet,
        ValueOfPerspectiveResult,
        perspective_arrow_schema_fingerprint,
        perspective_optimal_strategies,
        perspective_result_to_arrow,
        value_of_perspective,
        write_perspective_result_ipc,
        write_perspective_result_parquet,
    )
    from .portfolio import portfolio_voi
    from .preference import (
        PreferenceHeterogeneityResult,
        PreferenceProfile,
        PreferenceProfileSet,
        preference_optimal_strategies,
        value_of_preference,
        value_of_preference_heterogeneity,
        value_of_preference_information,
    )
    from .qualitative_information import (
        QualitativeInformationResult,
        QualitativeQuestionResult,
        qualitative_information_from_specification,
        render_qualitative_information_text,
    )
    from .regulatory_market_access import (
        RegulatoryMarketAccessResult,
        value_of_regulatory_market_access,
    )
    from .replication_reproducibility import (
        ReplicationReproducibilityResult,
        value_of_replication_reproducibility,
    )
    from .risk_sensitive_voi import (
        RiskSensitiveVoiResult,
        risk_sensitive_constrained_voi,
    )
    from .sample_information import enbs, evsi
    from .sequential import sequential_voi
    from .signed_social_information import (
        SignedSocialInformationResult,
        signed_social_information_value,
    )
    from .strategic_behavior import StrategicBehaviorResult, value_of_strategic_behavior
    from .structural import structural_evpi, structural_evppi
    from .threshold import (
        ThresholdProfile,
        ThresholdProfileSet,
        ThresholdResult,
        value_of_threshold,
        value_of_threshold_information,
    )
    from .uncertainty_modelling_value import (
        UncertaintyModellingValueResult,
        value_of_uncertainty_modelling,
    )
    from .utility_information import (
        expected_utility_information_value,
        value_of_clairvoyance,
    )
    from .validation import (
        ModelValidationResult,
        ValidationProfile,
        ValidationProfileSet,
        value_of_model_validation,
        value_of_validation,
    )

_MODULES = (
    "adaptive",
    "adaptive_learning_bandit",
    "ai_assisted_evidence_triage",
    "ambiguity_distribution_shift",
    "basic",
    "belief_state_information",
    "calibration",
    "capacity_budget_constrained",
    "causal_transportability",
    "ceaf",
    "computational",
    "data_quality",
    "distributional",
    "distributional_information",
    "dominance",
    "dynamic_real_options",
    "deterministic_sensitivity",
    "equity_information",
    "event_localized_information",
    "estimation",
    "evidence_obsolescence_refresh",
    "expert_synthesis",
    "explainability_transparency",
    "federated_privacy_preserving",
    "forecast_signal_information",
    "heterogeneity",
    "heterogeneity_value",
    "implementation",
    "implementation_information",
    "implementation_strategy",
    "interoperability_standardization",
    "information_source_portfolio",
    "monitoring_surveillance",
    "mcda_information",
    "network_nma",
    "observational",
    "outcome_conditional_sample_information",
    "perspective",
    "portfolio",
    "preference",
    "qualitative_information",
    "regulatory_market_access",
    "replication_reproducibility",
    "risk_sensitive_voi",
    "signed_social_information",
    "sample_information",
    "sequential",
    "strategic_behavior",
    "structural",
    "threshold",
    "utility_information",
    "uncertainty_modelling_value",
    "validation",
)


def __getattr__(name: str) -> object:
    """Resolve a method export only when requested."""
    for module_name in _MODULES:
        module = import_module(f".{module_name}", __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: RUF022 - public export order is a compatibility contract
    "AIAssistedEvidenceTriageResult",
    "AdaptiveLearningBanditResult",
    "AmbiguityDistributionShiftResult",
    "BeliefStateInformationResult",
    "CEAFResult",
    "CapacityBudgetConstrainedResult",
    "CausalTransportabilityResult",
    "ComputationalResult",
    "DataQualityResult",
    "DistributionalEquityResult",
    "DistributionalInformationResult",
    "DominanceResult",
    "DynamicRealOptionsResult",
    "DeterministicSensitivityResult",
    "DsaParameterSummary",
    "DsaPoint",
    "DsaSwitchInterval",
    "ValueOfFlexibilityResult",
    "EquityInformationResult",
    "EventLocalizedInformationResult",
    "EvidenceObsolescenceRefreshResult",
    "ExpertSynthesisResult",
    "ExplainabilityTransparencyResult",
    "FederatedPrivacyPreservingResult",
    "ForecastSignalInformationResult",
    "HeterogeneityResult",
    "HeterogeneityValueDecompositionResult",
    "ImplementationAdjustedResult",
    "ImplementationInformationResult",
    "ImplementationStrategyComparisonResult",
    "InformationSourcePortfolioResult",
    "InteroperabilityStandardizationResult",
    "ModelValidationResult",
    "MonitoringSurveillanceResult",
    "OutcomeConditionalSampleInformationResult",
    "McdaInformationResult",
    "Perspective",
    "PerspectiveSet",
    "PreferenceHeterogeneityResult",
    "PreferenceProfile",
    "PreferenceProfileSet",
    "QualitativeInformationResult",
    "QualitativeQuestionResult",
    "RegulatoryMarketAccessResult",
    "ReplicationReproducibilityResult",
    "RiskSensitiveVoiResult",
    "SignedSocialInformationResult",
    "ResolvedDistributionModel",
    "StrategicBehaviorResult",
    "ThresholdProfile",
    "ThresholdProfileSet",
    "ThresholdResult",
    "UncertaintyModellingValueResult",
    "ValidationProfile",
    "ValidationProfileSet",
    "ValueOfPerspectiveResult",
    "adaptive_evsi",
    "belief_state_information_value",
    "calculate_ceaf",
    "calculate_dominance",
    "calculate_extended_dominance",
    "calculate_icers",
    "calculate_strong_dominance",
    "cost_effectiveness_frontier",
    "deterministic_sensitivity",
    "deterministic_sensitivity_from_specification",
    "distributional_information_from_specification",
    "enbs",
    "evpi",
    "evppi",
    "evsi",
    "evsi_nma",
    "event_localized_information_value",
    "expected_utility_information_value",
    "forecast_signal_information_value",
    "identify_optimal_subgroups",
    "implementation_information_value",
    "information_source_portfolio_value",
    "mcda_information_value",
    "outcome_conditional_sample_information_value",
    "perspective_optimal_strategies",
    "perspective_arrow_schema_fingerprint",
    "perspective_result_to_arrow",
    "portfolio_voi",
    "preference_optimal_strategies",
    "qualitative_information_from_specification",
    "render_qualitative_information_text",
    "risk_sensitive_constrained_voi",
    "signed_social_information_value",
    "sequential_voi",
    "structural_evpi",
    "structural_evppi",
    "value_of_uncertainty_modelling",
    "value_of_adaptive_learning_bandit",
    "value_of_clairvoyance",
    "value_of_ai_assisted_evidence_triage",
    "value_of_ambiguity_distribution_shift",
    "value_of_capacity_budget_constrained",
    "value_of_causal_transportability",
    "value_of_computational_refinement",
    "value_of_data_quality",
    "value_of_distributional_equity",
    "value_of_distributional_information",
    "value_of_dynamic_real_options",
    "value_of_flexibility",
    "value_of_equity_information",
    "value_of_evidence_obsolescence_refresh",
    "value_of_expert_synthesis",
    "value_of_explainability_transparency",
    "value_of_federated_privacy_preserving",
    "value_of_heterogeneity",
    "heterogeneity_value_decomposition",
    "value_of_implementation",
    "value_of_implementation_strategy_comparison",
    "value_of_interoperability_standardization",
    "value_of_model_validation",
    "value_of_monitoring_surveillance",
    "value_of_perspective",
    "write_perspective_result_ipc",
    "write_perspective_result_parquet",
    "value_of_preference",
    "value_of_preference_heterogeneity",
    "value_of_preference_information",
    "value_of_regulatory_market_access",
    "value_of_replication_reproducibility",
    "value_of_strategic_behavior",
    "value_of_threshold",
    "value_of_threshold_information",
    "value_of_validation",
    "voi_calibration",
    "voi_observational",
]
